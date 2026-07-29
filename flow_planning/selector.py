"""Best-of-N plan selection: exact FK + box geometry scores each sampled plan
and the lowest-cost one is executed. Selection needs no gradients, so exact
geometry beats a learned proxy."""

import torch
from torch import Tensor

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


def box_sdf(points: Tensor, center: Tensor, half_extents: Tensor) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


class FrankaCollision:
    """FK-based collision geometry for (trajectory, box) pairs."""

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


class AnalyticSelector:
    """Best-of-N scorer: worst-case obstacle clearance of the arm and the plan's
    predicted cube position, capped at 'enough', minus a small progress term."""

    def __init__(
        self,
        box,  # physical world [center(3), half_extents(3)]
        base_pos,
        joint_stats,  # action stats; leading 7 dims are the arm joints
        obs_stats,  # obs stats; cube pos block normalizes the plan's cube dims
        joint_start: int,
        device,
        cube_size: float,
        cube_index: int = 18,
        n_arm: int = 7,
        cap: float = 0.08,  # clearance beyond this buys nothing
        prog: float = 0.03,  # weight of the mean cube-to-goal progress term
        speed: float = 0.0,  # weight of the overspeed hinge (0 disables)
        vlim=None,  # (n_arm,) per-step joint speed the demos never exceed
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.fc = FrankaCollision(device, base_pos, cube_size)
        self.box = torch.as_tensor(box, **f32).view(1, 6)
        self.jm = torch.as_tensor(joint_stats["mean"], **f32)[:n_arm]
        self.js = torch.as_tensor(joint_stats["std"], **f32)[:n_arm]
        # the plan's predicted joint STATES share the action normalization only
        # if recorded identically; use obs stats for the state block instead
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:n_arm]
        self.ss = torch.as_tensor(obs_stats["std"], **f32)[:n_arm]
        ci = cube_index
        self.cm = torch.as_tensor(obs_stats["mean"], **f32)[ci : ci + 3]
        self.cs = torch.as_tensor(obs_stats["std"], **f32)[ci : ci + 3]
        self.joint_start, self.n_arm, self.cube_index = joint_start, n_arm, ci
        self.cap, self.prog, self.speed = cap, prog, speed
        self.vlim = None if vlim is None else torch.as_tensor(vlim, **f32)

    @torch.no_grad()
    def score(self, traj: Tensor) -> Tensor:
        """Uncapped margin-max games the goal clamp: the safest plan is one
        that never approaches the cube/goal at all — hence cap + progress."""
        out = [self.clearance(traj[i : i + 1024]) for i in range(0, len(traj), 1024)]
        clear = torch.cat(out)
        cube = traj[..., self.cube_index : self.cube_index + 3] * self.cs + self.cm
        prog = (cube - cube[:, -1:]).norm(dim=-1).mean(dim=1)  # mean dist to goal
        s = -clear.clamp(max=self.cap) + self.prog * prog
        if self.speed and self.vlim is not None:
            s = s + self.speed * self.overspeed(traj)
        return s

    def overspeed(self, traj: Tensor) -> Tensor:
        """Peak fraction by which a plan's demanded joint speed exceeds anything
        the demos asked for. Horizon is fixed, so a longer detour is a faster one
        — past what the arm can track, execution lags and the plan is fiction."""
        assert self.vlim is not None
        j = self.joint_start
        q = traj[..., j : j + self.n_arm] * self.js + self.jm
        dq = (q[:, 1:] - q[:, :-1]).abs()
        return (dq / self.vlim - 1.0).clamp(min=0.0).amax(dim=2).amax(dim=1)

    def clearance(self, traj: Tensor) -> Tensor:
        """Worst clearance per plan (min over time and points)."""
        return self.clearance_t(traj).amin(dim=1)

    def clearance_t(self, traj: Tensor) -> Tensor:
        """Per-timestep worst clearance (b, t): min over arm points, the cube, and
        both joint sources. Differentiable — the guidance penalty rides on this."""
        s, n = self.joint_start, self.n_arm
        center, half = self.box[:, :3], self.box[:, None, None, 3:]
        cube = traj[..., self.cube_index : self.cube_index + 3] * self.cs + self.cm
        worst = []
        for q in (  # joint targets AND predicted joint states must both clear
            traj[..., s : s + n] * self.js + self.jm,
            traj[..., :n] * self.ss + self.sm,
        ):
            b, t = q.shape[:2]
            pts = self.fc.arm_points(q.reshape(-1, n).float()).reshape(b, t, -1, 3)
            d = box_sdf(pts + self.fc.base_pos, center[:, None, None], half)
            worst.append(d.amin(dim=2))  # (b, t)
        dc = box_sdf(cube[:, :, None].float(), center[:, None, None], half)
        worst.append(dc.amin(dim=2) - self.fc.cube_half)
        return torch.stack(worst).amin(dim=0)

    def penalty(self, traj: Tensor, margin: float) -> Tensor:
        """Differentiable per-plan collision cost for guidance: hinge below the
        keep-out margin, summed over time. Its gradient pushes near-wall frames out."""
        return (margin - self.clearance_t(traj)).clamp(min=0.0).sum(dim=1)
