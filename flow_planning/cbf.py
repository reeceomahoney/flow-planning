import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import POINTS_PER_LINK, box_sdf

EE_AXES = (0.06, 0.12, 0.11)
EE_OFFSET = (0.0, 0.0, -0.056)
FAR = 1e3


def mvee(points, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    p = torch.as_tensor(points, dtype=torch.float64).cpu()
    n, d = p.shape
    q = torch.cat([p, p.new_ones(n, 1)], 1)
    u = p.new_full((n,), 1.0 / n)
    err = 1.0
    while err > tol:
        x = q.T @ (u[:, None] * q)
        m = (q @ torch.linalg.solve(x, q.T)).diagonal()
        j = int(m.argmax())
        step = (m[j] - d - 1) / ((d + 1) * (m[j] - 1))
        new = (1 - step) * u
        new[j] += step
        err = float((new - u).norm())
        u = new
    c = u @ p
    a = torch.linalg.inv(p.T @ (u[:, None] * p) - c[:, None] * c[None]) / d
    return c.float(), a.float()


def size_matrix(a: Tensor) -> Tensor:
    w, v = torch.linalg.eigh(a)
    return v @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ v.T


def fibonacci_sphere(n: int) -> Tensor:
    i = torch.arange(n, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n
    r = (1 - z * z).sqrt()
    ang = math.pi * (1 + 5**0.5) * i
    return torch.stack([r * ang.cos(), r * ang.sin(), z], 1)


class EllipsoidCBF:
    def __init__(
        self,
        device,
        base_pos,
        alpha: float = 10.0,
        dt: float = 0.05,
        n_dirs: int = 1024,
        axes: Sequence[float] = EE_AXES,
        offset: Sequence[float] = EE_OFFSET,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.chain, self.njoints, _ = build_franka_chain(device)
        self.base_pos = torch.as_tensor(base_pos, **f32)
        self.axes = torch.as_tensor(axes, **f32)
        self.offset = torch.as_tensor(offset, **f32)
        self.dirs = fibonacci_sphere(n_dirs).to(device)
        self.alpha, self.dt = alpha, dt
        self.margin = 0.0
        self.arm_links: tuple[int, ...] = ()
        self.fc = None
        self.boxes = None
        self.ob_c = self.ob_q = None

    def set_obstacles(self, point_sets):
        cs, qs = [], []
        for pts in point_sets:
            if pts is None or len(pts) < 4:
                cs.append(torch.full((3,), FAR))
                qs.append(torch.eye(3) * 1e-3)
                continue
            c, a = mvee(pts)
            cs.append(c)
            qs.append(size_matrix(a))
        self.ob_c = torch.stack(cs).to(self.device)[:, None]
        self.ob_q = torch.stack(qs).to(self.device)[:, None]

    def set_boxes(self, boxes_per_world):
        g = max(len(b) if b is not None else 1 for b in boxes_per_world)
        w = len(boxes_per_world)
        c = torch.full((w, g, 3), FAR)
        q = torch.eye(3).repeat(w, g, 1, 1) * 1e-3
        boxes = torch.tensor([FAR, FAR, FAR, 0.01, 0.01, 0.01]).repeat(w, g, 1)
        for i, b in enumerate(boxes_per_world):
            if b is None:
                continue
            b = torch.as_tensor(np.asarray(b), dtype=torch.float32)
            c[i, : len(b)] = b[:, :3]
            q[i, : len(b)] = torch.diag_embed(b[:, 3:] * 3**0.5)
            boxes[i, : len(b)] = b
        self.ob_c, self.ob_q = c.to(self.device), q.to(self.device)
        self.boxes = boxes.to(self.device)

    def h_arm(self, q: Tensor) -> Tensor:
        assert self.fc is not None and self.boxes is not None
        pts = self.fc.arm_points(q) + self.base_pos
        idx = torch.cat(
            [
                torch.arange(POINTS_PER_LINK * k, POINTS_PER_LINK * (k + 1))
                for k in self.arm_links
            ]
        ).to(q.device)
        pts = pts[:, idx]
        boxes = self.boxes
        if len(pts) != len(boxes):
            boxes = boxes.repeat_interleave(len(pts) // len(boxes), 0)
        d = torch.stack(
            [
                box_sdf(pts, boxes[:, k, None, :3], boxes[:, k, None, 3:])
                for k in range(boxes.shape[1])
            ]
        ).amin(0)
        return (d - self.fc.radius).amin(1)

    def ee(self, q: Tensor) -> tuple[Tensor, Tensor]:
        pad = q.new_zeros(q.shape[0], self.njoints - q.shape[1])
        m = self.chain.forward_kinematics(torch.cat([q, pad], 1))[EE_FRAME].get_matrix()
        r = m[:, :3, :3]
        p = m[:, :3, 3] + self.base_pos + (r @ self.offset)
        return p, r

    def attach_offset(self, q: Tensor, world_pt: Tensor) -> Tensor:
        p, r = self.ee(q)
        p_tcp = p - (r @ self.offset)
        return (r.transpose(1, 2) @ (world_pt - p_tcp)[..., None])[..., 0]

    def h_ellipsoid(self, p: Tensor, qe_inv: Tensor) -> Tensor:
        assert self.ob_c is not None and self.ob_q is not None
        ob_c, ob_q = self.ob_c, self.ob_q
        if len(p) != len(ob_c):
            rep = len(p) // len(ob_c)
            ob_c = ob_c.repeat_interleave(rep, 0)
            ob_q = ob_q.repeat_interleave(rep, 0)
        n = self.dirs[None] @ qe_inv
        num = (
            -(n[:, None] @ ob_q).norm(dim=-1)
            + ((ob_c[:, :, None] - p[:, None, None]) * n[:, None]).sum(-1)
            - 1.0
        )
        return (num / n.norm(dim=-1)[:, None]).amax(2).amin(1)

    def h(self, p_ep: Tensor, r: Tensor) -> Tensor:
        qe_inv = r @ torch.diag(1.0 / self.axes) @ r.transpose(1, 2)
        return self.h_ellipsoid(p_ep, qe_inv)

    def value(self, q: Tensor, attach=None) -> Tensor:
        p, r = self.ee(q)
        h = self.h(p, r)
        if self.arm_links:
            h = torch.minimum(h, self.h_arm(q))
        if attach is not None:
            offset, radius = attach
            p_a = p - (r @ self.offset) + (r @ offset[..., None])[..., 0]
            eye = torch.eye(3, device=q.device).expand(len(q), 3, 3) / radius
            h = torch.minimum(h, self.h_ellipsoid(p_a, eye))
        return h

    def project_chunk(self, q0: Tensor, targets: Tensor, attach=None) -> Tensor:
        q, out = q0, []
        for i in range(targets.shape[1]):
            dq, _ = self.filter(q, targets[:, i] - q, attach)
            q = q + dq
            out.append(q)
        return torch.stack(out, 1)

    def constrain(self, dq_nom: Tensor, g: Tensor, h: Tensor) -> Tensor:
        slack = -self.alpha * self.dt * (h - self.margin) - (g * dq_nom).sum(1)
        lam = (slack / (g * g).sum(1).clamp_min(1e-9)).clamp_min(0.0)
        return dq_nom + lam[:, None] * g

    def filter(self, q: Tensor, dq_nom: Tensor, attach=None) -> tuple[Tensor, Tensor]:
        q = q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            h = self.value(q, attach)
            g = torch.autograd.grad(h.sum(), q)[0]
        return self.constrain(dq_nom, g, h.detach()), h.detach()
