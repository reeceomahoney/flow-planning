"""Inference-time cost guidance for the flow-matching policy."""

import os

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_planning.kinematics import build_franka_chain


def box_sdf(points: Tensor, center: Tensor, half_extents: Tensor) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


class ObstacleGuidance:
    """Push the trajectory position at `pos_start` off the obstacle box.

    `around`/`over_top` route past a thin wall's edge (a plain push-off stalls
    against its broad face). `pos_mean`/`pos_std` + `pos_start` pick the dims:
    action EE, or state EE/cube to compose in task space.
    """

    def __init__(
        self,
        center,
        half_extents,
        pos_mean,
        pos_std,
        scale: float,
        margin: float,
        device: str,
        pos_start: int = 0,
        around: bool = False,
        over_top: bool = False,
        interp: int = 4,
    ):
        kw = {"dtype": torch.float32, "device": device}
        self.center = torch.as_tensor(center, **kw)
        self.half_extents = torch.as_tensor(half_extents, **kw)
        if self.center.ndim == 2:  # per-world boxes: broadcast over path points
            self.center = self.center.unsqueeze(1)
            self.half_extents = self.half_extents.unsqueeze(1)
        d = self.center.shape[-1]
        self.pos_start = pos_start
        self.mean = torch.as_tensor(pos_mean, **kw)[:d]
        self.std = torch.as_tensor(pos_std, **kw)[:d]
        self.scale = scale
        self.margin = margin
        self.around = around
        self.over_top = over_top
        self.interp = interp
        self.side = None  # detour side, committed once per rollout

    def reset(self):
        self.side = None

    def path_points(self, pos: Tensor) -> Tensor:
        """Interpolate `interp` points per segment along the path, plus the end."""
        if self.interp <= 1:
            return pos
        a, b = pos[..., :-1, :], pos[..., 1:, :]
        s = torch.linspace(0.0, 1.0, self.interp + 1, device=pos.device)[:-1]
        pts = a.unsqueeze(-2) + s.view(-1, 1) * (b - a).unsqueeze(-2)
        return torch.cat([pts.flatten(-3, -2), pos[..., -1:, :]], dim=-2)

    def __call__(self, x1_hat: Tensor, obs: Tensor | None = None) -> Tensor:
        d = self.center.shape[-1]
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            pos = x[..., self.pos_start : self.pos_start + d] * self.std + self.mean
            pts = self.path_points(pos)
            sdf = box_sdf(pts, self.center, self.half_extents)
            cost = torch.relu(self.margin - sdf).square().sum()

            q = (pts - self.center).abs()
            if self.around:
                band = q[..., 1] < self.half_extents[..., 1] + self.margin
                # commit the detour side once per rollout: recomputing it each ODE
                # step lets the still-noisy early path flip sides and wobble
                if self.side is None:
                    xrel = (pts[..., 0] - self.center[..., 0]).detach()
                    side = torch.sign((xrel * band).sum(-1, keepdim=True))
                    side[side == 0] = 1.0
                    self.side = side
                side = self.side
                out = torch.relu(
                    self.half_extents[..., 0]
                    + self.margin
                    - side * (pts[..., 0] - self.center[..., 0])
                )
                cost = cost + (out * band).square().sum()

            if self.over_top:
                band = (q[..., 0] < self.half_extents[..., 0] + self.margin) & (
                    q[..., 1] < self.half_extents[..., 1] + self.margin
                )
                z_top = self.center[..., 2] + self.half_extents[..., 2]
                over = torch.relu(z_top + self.margin - pts[..., 2]) * band
                cost = cost + over.square().sum()

            cost = self.scale * cost
            (grad,) = torch.autograd.grad(cost, x)
        return grad


class ArmObstacleGuidance:
    """Push FK arm/gripper collision points off the obstacle box.

    Points carry each frame's full transform, so a wrist roll moves the
    fingertips and the gradient reaches the wrist — clearance a single-point EE
    cost can't express. Routing is left to the EE-position term.
    """

    # body-frame collision points (m); fingertips offset along the hand y so a
    # wrist roll moves them relative to the wall
    SAMPLES = {
        "link5": [[0.0, 0.0, 0.0]],
        "link6": [[0.0, 0.0, 0.0]],
        "link7": [[0.0, 0.0, 0.0]],
        "hand": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10],
            [0.0, 0.04, 0.10],
            [0.0, -0.04, 0.10],
            [0.0, 0.04, 0.06],
            [0.0, -0.04, 0.06],
        ],
    }

    def __init__(
        self,
        chain,
        njoints,
        center,
        half_extents,
        base_pos,
        joint_mean,
        joint_std,
        scale: float,
        margin: float,
        device: str,
        joint_start: int,
        n_arm: int = 7,
        stride: int = 2,
    ):
        kw = {"dtype": torch.float32, "device": device}
        self.chain = chain
        self.njoints = njoints
        self.n_arm = n_arm
        self.scale = scale
        self.margin = margin
        self.joint_start = joint_start
        self.stride = stride
        self.mean = torch.as_tensor(joint_mean, **kw)
        self.std = torch.as_tensor(joint_std, **kw)
        # obstacle box in the robot base frame (FK frames are base-relative)
        self.center = torch.as_tensor(center, **kw) - torch.as_tensor(base_pos, **kw)
        self.half_extents = torch.as_tensor(half_extents, **kw)
        # resolve each SAMPLES key to a real frame and stash its body-frame points
        names = chain.get_frame_names(exclude_fixed=False)
        self.frames, self.offsets = [], []
        for key, pts in self.SAMPLES.items():
            match = next((n for n in names if key in n and "tcp" not in n), None)
            assert match is not None, f"no frame matches {key}"
            self.frames.append(match)
            self.offsets.append(torch.as_tensor(pts, **kw))
        self.frame_indices = chain.get_frame_indices(*self.frames)

    def reset(self):
        pass

    def collision_points(self, q: Tensor) -> Tensor:
        """FK the (m, n_arm) configs and place the body-fixed points: (m, K, 3)."""
        m = q.shape[0]
        th = torch.cat([q, q.new_zeros(m, self.njoints - self.n_arm)], dim=1)
        tf = self.chain.forward_kinematics(th, self.frame_indices)
        pts = []
        for name, off in zip(self.frames, self.offsets):
            mat = tf[name].get_matrix()  # (m, 4, 4)
            r, t = mat[:, :3, :3], mat[:, :3, 3]
            pts.append(torch.einsum("mij,kj->mki", r, off) + t[:, None, :])
        return torch.cat(pts, dim=1)

    def __call__(self, x1_hat: Tensor, obs: Tensor | None = None) -> Tensor:
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            s = self.joint_start
            q = x[..., s : s + self.n_arm] * self.std + self.mean  # (B, H, n_arm)
            q = q[:, :: self.stride]  # subsample horizon for speed
            pts = self.collision_points(q.reshape(-1, self.n_arm))  # (m, K, 3)
            sdf = box_sdf(pts, self.center, self.half_extents)
            cost = self.scale * torch.relu(self.margin - sdf).square().sum()
            (grad,) = torch.autograd.grad(cost, x)
        if os.environ.get("GUIDANCE_DEBUG"):
            pen = (sdf < self.margin).float().mean().item()
            print(
                f"[arm] grad={grad.norm().item():.4f} "
                f"sdf_min={sdf.min().item():.3f} pen_frac={pen:.3f}",
                flush=True,
            )
        return grad


class EllipseGuidance:
    """Push the trajectory out of a smooth ellipse that envelops the obstacle and
    spans start->goal. Because a long arc of a blocked path lies inside a convex
    region, the outward gradient shifts the whole arc coherently and it wraps around
    the end (a detour), instead of the local tear a box-collision cost produces."""

    def __init__(
        self,
        box,
        obs_stats,
        device,
        scale: float,
        margin: float = 0.15,
        ay_scale: float = 0.85,
        bias: float = 0.0,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        m = torch.as_tensor(obs_stats["mean"], **f32)
        s = torch.as_tensor(obs_stats["std"], **f32)
        self.pos_m, self.pos_s = m[:2], s[:2]  # position obs block
        self.goal_m, self.goal_s = m[-2:], s[-2:]  # goal obs block
        box = torch.as_tensor(box, **f32)  # [cx, cy, hx, hy] physical
        self.center = (box[:2] - self.pos_m) / self.pos_s  # normalized position space
        self.half = box[2:] / self.pos_s
        self.scale, self.margin, self.ay_scale = scale, margin, ay_scale
        self.bias = bias  # symmetry break: push inside points around an end
        self.exit_side = 1.0 if float(box[0]) <= 0 else -1.0  # arena centered at 0

    def reset(self):
        pass

    def __call__(self, x: Tensor, v: Tensor, t: Tensor, obs: Tensor | None = None):
        assert obs is not None
        goal = (obs[:, -2:] * self.goal_s + self.goal_m - self.pos_m) / self.pos_s
        start = obs[:, :2]  # normalized position
        cx, cy = self.center
        ax = self.half[0] + self.margin  # cover the bar width -> exit round the end
        span = 0.5 * (start[:, 1] - goal[:, 1]).abs()  # (B,) reach toward start/goal
        ay = (self.ay_scale * span).clamp_min(self.half[1] + self.margin)
        # exit side: round the end nearer start/goal (roomier end if centered)
        mid = 0.5 * (start[:, 0] + goal[:, 0]) - cx
        side = torch.where(
            mid.abs() < 0.15, mid.new_full((), self.exit_side), mid.sign()
        )
        with torch.enable_grad():
            x1_hat = (x + (1 - t.view(-1, 1, 1)) * v).detach().requires_grad_(True)
            p = x1_hat[..., :2]
            ex = (p[..., 0] - cx) / ax
            ey = (p[..., 1] - cy) / ay[:, None]
            e = ex * ex + ey * ey  # <1 inside the ellipse
            inside = F.softplus(1.0 - e)
            cost = self.scale * inside.sum()  # penalize inside
            if self.bias > 0:  # break symmetry: drive inside points to the exit side
                cost = (
                    cost - self.scale * self.bias * (inside * side[:, None] * ex).sum()
                )
            (grad,) = torch.autograd.grad(cost, x1_hat)
        return grad


class Guidance:
    """Sum of the enabled guidance terms, with a per-rollout reset for stateful
    terms. `action_stats` is the action `{"mean", "std"}` (the obstacle term acts
    on the action's leading position dims, which start at `state_dim` in the
    trajectory); `env` supplies the obstacle geometry. With no term enabled the
    sum is a harmless zero."""

    def __init__(self, env, state_dim, action_stats, device, obs_stats=None):
        self.terms = []
        geom = env.obstacle_geometry if env.cfg.obstacle else None
        arm = hasattr(env.cfg, "ee_index")
        # franka: FK arm/gripper clearance (wrist DOF) + EE-position routing
        if geom and arm and env.cfg.arm_scale > 0:
            chain, njoints, _ = build_franka_chain(device)
            n_arm = 7
            self.terms.append(
                ArmObstacleGuidance(
                    chain,
                    njoints,
                    center=geom["center"],
                    half_extents=geom["half_extents"],
                    base_pos=list(env.robot_base_pos),
                    joint_mean=action_stats["mean"][:n_arm],
                    joint_std=action_stats["std"][:n_arm],
                    scale=env.cfg.arm_scale,
                    margin=env.cfg.arm_margin,
                    device=device,
                    joint_start=state_dim,
                    n_arm=n_arm,
                    stride=env.cfg.arm_stride,
                )
            )
        # franka: over-top routing on the EE position (in the state dims)
        if geom and arm and obs_stats is not None and env.cfg.ee_scale > 0:
            ee = env.ee_state_index
            self.terms.append(
                ObstacleGuidance(
                    center=geom["center"],
                    half_extents=geom["half_extents"],
                    pos_mean=obs_stats["mean"][ee:],
                    pos_std=obs_stats["std"][ee:],
                    scale=env.cfg.ee_scale,
                    margin=env.cfg.ee_margin,
                    device=device,
                    pos_start=ee,
                    over_top=geom.get("over_top", False),
                )
            )
        # particle: EE-position term pushes the planned end-effector path off it
        elif geom and not arm and env.cfg.obstacle_scale > 0:
            self.terms.append(
                ObstacleGuidance(
                    **geom,
                    pos_mean=action_stats["mean"],
                    pos_std=action_stats["std"],
                    scale=env.cfg.obstacle_scale,
                    margin=env.cfg.obstacle_margin,
                    device=device,
                    pos_start=state_dim,
                )
            )

    def __call__(self, x, v, t, obs):
        x1_hat = x + (1 - t.view(-1, 1, 1)) * v
        return sum(g(x1_hat, obs) for g in self.terms)

    def reset(self):
        for g in self.terms:
            getattr(g, "reset", lambda: None)()
