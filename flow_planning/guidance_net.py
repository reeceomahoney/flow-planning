"""Learned per-timestep collision guidance: a net scores each trajectory step as
colliding-or-not given an obstacle box, and its gradient replaces the hand-crafted
SDF cost. Scoring per step (not the whole path) means the gradient only nudges the
steps that actually hit, so a collision-free grasp right next to the box is left
alone — unlike a proximity-margin SDF, which fights it.

The supervision is free and exact: take demo trajectories, sample random boxes,
and label every step via FK + `box_sdf` (the same geometry the sim penalizes).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flow_planning.guidance import box_sdf
from flow_planning.kinematics import build_franka_chain

# arm frames FK'd for collision; hand carries a few extra offset points so a
# wrist roll moves the fingertips relative to the wall
ARM_FRAMES = ["link2", "link3", "link4", "link5", "link6", "link7", "hand"]
HAND_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.10],
    [0.0, 0.04, 0.10],
    [0.0, -0.04, 0.10],
]


class FrankaCollision:
    """FK-based collision labeler for (trajectory, box) pairs. Offline only —
    samples the arm densely; correctness matters more than speed here."""

    def __init__(self, device: str, base_pos, cube_size: float, interp: int = 3):
        chain, njoints, _ = build_franka_chain(device)
        self.chain = chain
        self.njoints = njoints
        self.interp = interp
        self.base_pos = torch.as_tensor(base_pos, dtype=torch.float32, device=device)
        self.cube_half = 0.5 * cube_size
        names = chain.get_frame_names(exclude_fixed=False)
        self.frames = []
        for key in ARM_FRAMES:
            m = next((n for n in names if key in n and "tcp" not in n), None)
            assert m is not None, f"no frame matches {key}"
            self.frames.append(m)
        self.frame_indices = chain.get_frame_indices(*self.frames)
        self.hand_off = torch.as_tensor(
            HAND_OFFSETS, dtype=torch.float32, device=device
        )

    def arm_points(self, q: Tensor) -> Tensor:
        """q (m, 7) arm joints -> (m, K, 3) collision points in the base frame."""
        m = q.shape[0]
        th = torch.cat([q, q.new_zeros(m, self.njoints - 7)], dim=1)
        tf = self.chain.forward_kinematics(th, self.frame_indices)
        origins = [tf[n].get_matrix()[:, :3, 3] for n in self.frames]
        spine = torch.stack(origins, dim=1)  # (m, F, 3)
        a, b = spine[:, :-1, None], spine[:, 1:, None]
        s = torch.linspace(0.0, 1.0, self.interp + 1, device=q.device).view(1, 1, -1, 1)
        pts = (a + s * (b - a)).reshape(m, -1, 3)
        hmat = tf[self.frames[-1]].get_matrix()  # hand
        r, t = hmat[:, :3, :3], hmat[:, :3, 3]
        hoff = torch.einsum("mij,kj->mki", r, self.hand_off) + t[:, None, :]
        return torch.cat([pts, hoff], dim=1)

    @torch.no_grad()
    def collisions(
        self,
        joints: Tensor,
        cube_pos: Tensor | None,
        box: Tensor,
        margin: float = 0.02,
        chunk: int = 4096,
    ) -> Tensor:
        """joints (P, T, 7), cube_pos (P, T, 3) world or None, box (P, 6)
        [center, half]. Returns bool (P, T): True at timesteps where the arm (or
        cube, if given) hits the box (inflated by `margin`, a thin safety buffer —
        sharp, not a proximity band)."""
        out = []
        for i in range(0, joints.shape[0], chunk):
            j, bx = joints[i : i + chunk], box[i : i + chunk]
            p, t = j.shape[0], j.shape[1]
            center, half = bx[:, :3], bx[:, 3:]
            # arm: FK in the base frame, so shift the box into it
            pts = self.arm_points(j.reshape(p * t, 7)).reshape(p, t, -1, 3)
            cb = (center - self.base_pos)[:, None, None, :]
            hit = (box_sdf(pts, cb, half[:, None, None, :]) < margin).any(dim=2)
            if cube_pos is not None:
                # cube: axis-aligned box overlap in the world frame
                d = (cube_pos[i : i + chunk] - center[:, None, :]).abs()
                hit |= (d < half[:, None, :] + self.cube_half + margin).all(-1)
            out.append(hit)
        return torch.cat(out)


def sample_boxes(rng, n: int, center0, half0, device: str) -> Tensor:
    """Random world boxes around the workspace; ~half resemble the eval wall, the
    rest are scattered so labels stay balanced. Returns (n, 6) [center, half]."""
    c0 = torch.as_tensor(center0, dtype=torch.float32)
    h0 = torch.as_tensor(half0, dtype=torch.float32)

    def u(lo, hi):
        return torch.from_numpy(rng.uniform(lo, hi, (n, 3)).astype("float32"))

    wall = torch.rand(n, 1) < 0.5
    center = torch.where(
        wall, c0 + u(-0.1, 0.1), u([-0.3, -0.8, 0.1], [0.3, -0.2, 0.5])
    )
    half = torch.where(wall, (h0 + u(-0.03, 0.06)).clamp(0.01), u(0.02, 0.16))
    return torch.cat([center, half], dim=1).to(device)


def sample_boxes_2d(rng, n: int, device: str) -> Tensor:
    """Random 2D bars spanning the particle arena (matches ParticleEnv's
    obstacle_random ranges) so the classifier generalizes over placements.
    Returns (n, 4) [cx, cy, hx, hy]."""

    def u(lo, hi):
        return torch.from_numpy(rng.uniform(lo, hi, n).astype("float32"))

    return torch.stack(
        [u(-0.2, 0.2), u(-0.3, 0.3), u(0.3, 0.6), u(0.03, 0.1)], dim=1
    ).to(device)


def normalize_box(box: Tensor, pos_mean: Tensor, pos_std: Tensor) -> Tensor:
    """Put the box on the same scale as the trajectory's position dims. `d` (2 or
    3) is inferred from the stats; box is [center(d), half(d)]."""
    d = pos_mean.shape[-1]
    center = (box[..., :d] - pos_mean) / pos_std
    half = box[..., d:] / pos_std
    return torch.cat([center, half], dim=-1)


class GuidanceNet(nn.Module):
    """Per-timestep collision predictor: scores each [state, action] step as
    collision-free given a box. Collision is memoryless in the config, so this is
    a per-step MLP (no temporal mixing) — its gradient touches only the steps that
    actually hit, leaving a collision-free grasp next to the box untouched. Applied
    to the denoised estimate x1_hat at inference (reconstruction guidance)."""

    def __init__(
        self, traj_dim: int, box_dim: int = 6, d: int = 256, n_layers: int = 3
    ):
        super().__init__()
        self.box_dim = box_dim
        layers = [nn.Linear(traj_dim + box_dim, d), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, traj: Tensor, box: Tensor) -> Tensor:
        b = box[:, None, :].expand(-1, traj.shape[1], -1)  # (B, H, 6)
        logit = self.mlp(torch.cat([traj, b], dim=-1))  # (B, H, 1)
        return logit.squeeze(-1)  # (B, H) collision logit


class LearnedGuidance:
    """Inference guidance: per-step gradient of -log P(collision-free) w.r.t. the
    trajectory, so the sampler nudges only the colliding steps off the box."""

    def __init__(
        self,
        ckpt: str,
        box,
        obs_stats,
        pos_index: int,
        device,
        scale: float,
        pos_dims: int = 3,
    ):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.box_dim = blob.get("box_dim", 6)
        self.net = GuidanceNet(blob["traj_dim"], self.box_dim).to(device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        f32 = {"dtype": torch.float32, "device": device}
        pos_mean = torch.as_tensor(obs_stats["mean"][pos_index:], **f32)[:pos_dims]
        pos_std = torch.as_tensor(obs_stats["std"][pos_index:], **f32)[:pos_dims]
        box = torch.as_tensor(box, dtype=torch.float32, device=device)
        self.box = normalize_box(box, pos_mean, pos_std).view(1, self.box_dim)
        self.scale = scale

    def reset(self):
        pass

    def __call__(self, x: Tensor, v: Tensor, t: Tensor, obs: Tensor | None = None):
        # reconstruction guidance: score the denoised estimate x1_hat = x + (1-t)v
        with torch.enable_grad():
            x1_hat = (x + (1 - t.view(-1, 1, 1)) * v).detach().requires_grad_(True)
            box = self.box.expand(x.shape[0], self.box_dim)
            logit = self.net(x1_hat, box)  # (B, H) collision
            cost = self.scale * F.softplus(logit).sum()  # penalize per-step collision
            (grad,) = torch.autograd.grad(cost, x1_hat)
        return grad


class FKLearnedGuidance:
    """LearnedGuidance for franka: the per-step net scores FK arm collision points
    of the planned JOINT actions (box-relative, world frame), so the gradient
    reaches the executed dims through the FK Jacobian — raw-trajectory input only
    touches them via model coupling, which proved weak."""

    def __init__(
        self,
        ckpt: str,
        box,  # physical world [center(3), half_extents(3)]
        base_pos,
        joint_stats,  # action {"mean", "std"}, leading 7 dims are the arm joints
        joint_start: int,
        device,
        scale: float,
        n_arm: int = 7,
        stride: int = 2,
    ):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.net = GuidanceNet(blob["traj_dim"], blob["box_dim"]).to(device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        f32 = {"dtype": torch.float32, "device": device}
        self.fc = FrankaCollision(device, base_pos, cube_size=0.0)
        box = torch.as_tensor(box, **f32)
        self.center, self.half = box[:3], box[3:].view(1, 3)
        self.jm = torch.as_tensor(joint_stats["mean"], **f32)[:n_arm]
        self.js = torch.as_tensor(joint_stats["std"], **f32)[:n_arm]
        self.joint_start, self.n_arm = joint_start, n_arm
        self.stride, self.scale = stride, scale

    def reset(self):
        pass

    def __call__(self, x: Tensor, v: Tensor, t: Tensor, obs: Tensor | None = None):
        with torch.enable_grad():
            x1_hat = (x + (1 - t.view(-1, 1, 1)) * v).detach().requires_grad_(True)
            s = self.joint_start
            q = x1_hat[..., s : s + self.n_arm] * self.js + self.jm  # (B, H, 7)
            q = q[:, :: self.stride]
            b, h = q.shape[:2]
            pts = self.fc.arm_points(q.reshape(-1, self.n_arm)).reshape(b, h, -1, 3)
            feat = (pts + self.fc.base_pos - self.center).flatten(-2)
            logit = self.net(feat, self.half.expand(b, 3))  # (B, H') collision
            cost = self.scale * F.softplus(logit).sum()
            (grad,) = torch.autograd.grad(cost, x1_hat)
        return grad
