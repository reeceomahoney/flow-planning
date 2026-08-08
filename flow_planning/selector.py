"""Best-of-N plan selection: exact FK + box geometry scores each sampled plan
and the lowest-cost one is executed. Selection needs no gradients, so exact
geometry beats a learned proxy."""

import torch
from torch import Tensor

from flow_planning.kinematics import build_franka_chain

# arm frames FK'd for collision; hand carries a few extra offset points so a
# wrist roll moves the fingertips relative to the wall
ARM_FRAMES = ["link2", "link3", "link4", "link5", "link6", "link7", "hand"]
# per-segment collision radius: the spine is a CENTRELINE, so without these a
# "4cm clearance" plan already has the mesh touching. Non-uniform matters --
# a uniform radius is just a margin shift (same feasible set, same ranking),
# but the thick forearm and the thin fingers need opposite standoffs: the arm
# must clear the wall while the hand reaches a cube sitting right next to it.
SEGMENT_RADII = [0.06, 0.06, 0.05, 0.05, 0.04, 0.04]  # link2-3 .. link7-hand
HAND_RADIUS = 0.05
HAND_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.06],
    [0.0, 0.07, 0.03],
    [0.0, -0.07, 0.03],
    [0.0, 0.07, 0.09],
    [0.0, -0.07, 0.09],
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
        seg = torch.as_tensor(SEGMENT_RADII, dtype=torch.float32, device=device)
        self.radii = torch.cat(
            [
                seg.repeat_interleave(interp + 1),
                torch.full((len(HAND_OFFSETS),), HAND_RADIUS, device=device),
            ]
        )  # (K,), aligned with arm_points

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
        margin: float = 0.0,  # keep-out inflation; 0 = bare collision check
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
        self.margin = margin
        self.rad = self.fc.radii

    @torch.no_grad()
    def score(self, traj: Tensor) -> Tensor:
        """Total penetration below the keep-out margin. Ranking on clearance
        instead games the goal clamp -- the safest plan never approaches the
        cube at all -- so clearance past the margin buys nothing and every
        collision-free plan ties."""
        out = [
            (self.margin - self.clearance_t(traj[i : i + 1024])).clamp(min=0.0)
            for i in range(0, len(traj), 1024)
        ]
        return torch.cat(out).sum(dim=1)

    def clearance_t(self, traj: Tensor) -> Tensor:
        """Per-timestep worst clearance (b, t): min over arm points, the cube, and
        both joint sources."""
        s, n = self.joint_start, self.n_arm
        center, half = self.box[:, :3], self.box[:, None, None, 3:]
        cube = traj[..., self.cube_index : self.cube_index + 3] * self.cs + self.cm
        # require both the joint targets and the plan's own predicted states to
        # clear: the two disagree by the controller's tracking lag
        sources = (
            traj[..., s : s + n] * self.js + self.jm,
            traj[..., :n] * self.ss + self.sm,
        )
        worst = []
        for q in sources:
            b, t = q.shape[:2]
            pts = self.fc.arm_points(q.reshape(-1, n).float()).reshape(b, t, -1, 3)
            d = box_sdf(pts + self.fc.base_pos, center[:, None, None], half)
            worst.append((d - self.rad).amin(dim=2))  # (b, t), mesh not centreline
        dc = box_sdf(cube[:, :, None].float(), center[:, None, None], half)
        worst.append(dc.amin(dim=2) - self.fc.cube_half)
        return torch.stack(worst).amin(dim=0)
