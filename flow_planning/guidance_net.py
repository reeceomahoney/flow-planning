"""Learned classifier guidance: a net scores a planned trajectory as collision-
free given an obstacle box, and its gradient replaces the hand-crafted SDF cost.

The supervision is free and exact: take demo trajectories, sample random boxes,
and label each pair via FK + `box_sdf` (the same geometry the sim penalizes).
The classifier sees the whole trajectory, so its gradient is non-myopic — it
selects the collision-free mode the prior can already stitch, rather than
shoving the nearest point off the box like the per-point cost does.
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
    def collision_free(
        self, joints: Tensor, cube_pos: Tensor, box: Tensor, chunk: int = 4096
    ) -> Tensor:
        """joints (P, T, 7), cube_pos (P, T, 3) world, box (P, 6) world
        [center, half]. Returns bool (P,): True if the whole path clears the box."""
        out = []
        for i in range(0, joints.shape[0], chunk):
            j, cu, bx = (
                joints[i : i + chunk],
                cube_pos[i : i + chunk],
                box[i : i + chunk],
            )
            p, t = j.shape[0], j.shape[1]
            center, half = bx[:, :3], bx[:, 3:]
            # arm: FK in the base frame, so shift the box into it
            pts = self.arm_points(j.reshape(p * t, 7)).reshape(p, t, -1, 3)
            cb = (center - self.base_pos)[:, None, None, :]
            arm_hit = (box_sdf(pts, cb, half[:, None, None, :]) < 0).any(dim=(1, 2))
            # cube: axis-aligned box overlap in the world frame
            d = (cu - center[:, None, :]).abs()
            cube_hit = (d < half[:, None, :] + self.cube_half).all(-1).any(1)
            out.append(~(arm_hit | cube_hit))
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


def normalize_box(box: Tensor, cube_mean: Tensor, cube_std: Tensor) -> Tensor:
    """Put the box on the same scale as the trajectory's position dims."""
    center = (box[..., :3] - cube_mean) / cube_std
    half = box[..., 3:] / cube_std
    return torch.cat([center, half], dim=-1)


class GuidanceNet(nn.Module):
    """Score a clean [state, action] trajectory as collision-free given a box.
    The box token is the readout: it attends over the whole path. Trained on
    clean trajectories and applied to the denoised estimate x1_hat at inference
    (reconstruction guidance — smoother than raw classifier guidance here)."""

    def __init__(self, traj_dim: int, max_len: int, d: int = 128, n_layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(traj_dim, d)
        self.box_proj = nn.Linear(6, d)
        self.pos = nn.Parameter(0.02 * torch.randn(max_len + 1, d))
        layer = nn.TransformerEncoderLayer(
            d, 4, 4 * d, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def forward(self, traj: Tensor, box: Tensor) -> Tensor:
        h = self.in_proj(traj)  # (B, H, d)
        b = self.box_proj(box)[:, None]  # (B, 1, d) readout token
        x = torch.cat([b, h], dim=1) + self.pos[: h.shape[1] + 1]
        return self.head(self.tr(x)[:, 0]).squeeze(-1)  # logit


class LearnedGuidance:
    """Inference guidance: gradient of -log P(collision-free) w.r.t. the
    trajectory, so the sampler ascends toward the collision-free mode."""

    def __init__(self, ckpt: str, box, obs_stats, ee_index: int, device, scale: float):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.net = GuidanceNet(blob["traj_dim"], blob["max_len"]).to(device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        f32 = {"dtype": torch.float32, "device": device}
        cube_mean = torch.as_tensor(obs_stats["mean"][ee_index:], **f32)[:3]
        cube_std = torch.as_tensor(obs_stats["std"][ee_index:], **f32)[:3]
        box = torch.as_tensor(box, dtype=torch.float32, device=device)
        self.box = normalize_box(box, cube_mean, cube_std).view(1, 6)
        self.scale = scale

    def reset(self):
        pass

    def __call__(self, x: Tensor, v: Tensor, t: Tensor, obs: Tensor | None = None):
        # reconstruction guidance: score the denoised estimate x1_hat = x + (1-t)v
        with torch.enable_grad():
            x1_hat = (x + (1 - t.view(-1, 1, 1)) * v).detach().requires_grad_(True)
            logit = self.net(x1_hat, self.box.expand(x.shape[0], 6))
            cost = self.scale * F.softplus(-logit).sum()  # -log p_free
            (grad,) = torch.autograd.grad(cost, x1_hat)
        return grad
