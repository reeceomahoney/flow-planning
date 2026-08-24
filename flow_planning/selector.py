"""Best-of-N plan selection: exact FK + box geometry scores each sampled plan
and the lowest-cost one is executed. Selection needs no gradients, so exact
geometry beats a learned proxy."""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch_volumetric import sdf as pv_sdf
from torch import Tensor

from flow_planning.kinematics import build_franka_chain

LINKS = [f"fr3_link{i}" for i in range(8)] + ["fr3_hand"]
FINGERS = ["fr3_leftfinger", "fr3_rightfinger"]
FINGER_POINTS = [[0.0, 0.01, 0.01], [0.0, 0.01, 0.03], [0.0, 0.008, 0.045]]
FINGER_Q = 0.02
POINTS_PER_LINK = 128
POINT_RADIUS = 0.015
SELF_SUBSET = 64
SELF_STRIDE = 4
SELF_GAP = 0.01


def box_sdf(points: Tensor, center: Tensor, half_extents) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


def sample_mesh(mesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    if len(pts) < n:
        extra, _ = trimesh.sample.sample_surface(mesh, n - len(pts), seed=0)
        pts = np.concatenate([pts, extra])
    return np.asarray(pts[:n], np.float32)


class LinkSDF:
    """Signed distance of a link's collision mesh, baked to a voxel grid in the
    link frame and read back with trilinear interpolation."""

    def __init__(self, path: str, bounds, device: str, res=0.005, pad=0.03):
        lo, hi = np.asarray(bounds[0]) - pad, np.asarray(bounds[1]) + pad
        axes = [torch.arange(lo[k], hi[k], res) for k in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        f = pv_sdf.MeshSDF(pv_sdf.MeshObjectFactory(mesh_name=path))
        d, _ = f(grid.reshape(-1, 3), compute_grad=False)
        self.grid = d.reshape(1, 1, *grid.shape[:3]).float().to(device)
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.tensor([a[-1] for a in axes], device=device)

    def __call__(self, p: Tensor) -> Tensor:
        g = (2.0 * (p - self.lo) / (self.hi - self.lo) - 1.0).flip(-1)
        out = F.grid_sample(
            self.grid,
            g.reshape(1, 1, 1, -1, 3),
            padding_mode="border",
            align_corners=True,
        )
        return out.reshape(p.shape[:-1])


class FrankaCollision:
    """Collision points sampled once from each link's collision mesh and FK'd
    per frame (exact wall and cube tests), plus per-link SDFs so one link's
    points can be tested inside another's volume (self-collision)."""

    def __init__(self, device: str, base_pos, cube_size: float, n=POINTS_PER_LINK):
        chain, njoints, root = build_franka_chain(device)
        self.chain = chain
        self.njoints = njoints
        self.device = device
        self.base_pos = torch.as_tensor(base_pos, dtype=torch.float32, device=device)
        self.cube_half = 0.5 * cube_size
        self.frames = LINKS + FINGERS
        self.frame_indices = chain.get_frame_indices(*self.frames)
        local, owner, self.sdfs = [], [], []
        for i, name in enumerate(LINKS):
            path = root + chain.find_frame(name).link.visuals[0].geom_param[0]
            mesh = trimesh.load(path, force="mesh")
            local.append(sample_mesh(mesh, n))
            owner += [i] * n
            self.sdfs.append(LinkSDF(path, mesh.bounds, device))
        for i, _ in enumerate(FINGERS):
            local.append(np.asarray(FINGER_POINTS, np.float32))
            owner += [len(LINKS) + i] * len(FINGER_POINTS)
        self.local = torch.as_tensor(np.concatenate(local), device=device)
        self.owner = torch.as_tensor(owner, device=device)
        self.arm_mask = self.owner < len(LINKS) - 1
        self.radius = POINT_RADIUS
        self.sub = SELF_SUBSET
        self.pairs = [
            (i, j) for i in range(len(LINKS)) for j in range(i + 2, len(LINKS))
        ]
        self.self_gap = torch.full((len(self.pairs),), SELF_GAP, device=device)

    def forward(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """q (m, 7) -> frame matrices (m, F, 4, 4) and points (m, K, 3), base frame."""
        m = q.shape[0]
        th = torch.cat([q, q.new_full((m, self.njoints - 7), FINGER_Q)], dim=1)
        tf = self.chain.forward_kinematics(th, self.frame_indices)
        mats = torch.stack([tf[n].get_matrix() for n in self.frames], dim=1)
        rot, t = mats[:, self.owner, :3, :3], mats[:, self.owner, :3, 3]
        return mats, torch.einsum("mkij,kj->mki", rot, self.local) + t

    def arm_points(self, q: Tensor) -> Tensor:
        return self.forward(q)[1]

    def into(self, pts: Tensor, mats: Tensor, j: int) -> Tensor:
        """World points (m, S, 3) expressed in link j's frame."""
        r, t = mats[:, j, :3, :3], mats[:, j, :3, 3]
        return torch.einsum("mji,msj->msi", r, pts - t[:, None])

    def self_dists(self, mats: Tensor, pts: Tensor) -> Tensor:
        """(m, P) signed distance per non-adjacent link pair: each link's sampled
        surface against the other's SDF, negative = interpenetration."""
        n, s = POINTS_PER_LINK, self.sub
        out = []
        for i, j in self.pairs:
            pi, pj = pts[:, i * n : i * n + s], pts[:, j * n : j * n + s]
            dij = self.sdfs[j](self.into(pi, mats, j)).amin(dim=1)
            dji = self.sdfs[i](self.into(pj, mats, i)).amin(dim=1)
            out.append(torch.minimum(dij, dji))
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def calibrate_self(self, q: Tensor, chunk: int = 4096):
        """Pairs that sit close in valid postures get a smaller (or no) gap."""
        lo = torch.cat(
            [
                self.self_dists(*self.forward(q[i : i + chunk])).amin(dim=0)[None]
                for i in range(0, len(q), chunk)
            ]
        ).amin(dim=0)
        self.self_gap = (lo - 0.005).clamp(max=SELF_GAP)

    def cube_penetration(self, pts: Tensor, cube: Tensor) -> Tensor:
        """Arm links (not hand/fingers) inside the held cube: (m,) worst."""
        d = box_sdf(pts[:, self.arm_mask], cube[:, None], self.cube_half)
        return (self.radius - d).clamp(min=0.0).amax(dim=1)

    def body_penetration(self, q: Tensor, cube: Tensor | None) -> tuple[Tensor, Tensor]:
        """(m,) worst self + held-cube penetration per frame, and the points."""
        mats, pts = self.forward(q)
        pen = (self.self_gap - self.self_dists(mats, pts)).clamp(min=0.0).amax(dim=1)
        if cube is not None:
            pen = pen + self.cube_penetration(pts, cube)
        return pen, pts


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
        cube_index: int | None = 18,
        n_arm: int = 7,
        pointcloud=None,
        ee_index: int | None = None,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.fc = FrankaCollision(device, base_pos, cube_size)
        self.box = torch.as_tensor(box, **f32).view(-1, 6)
        self.cloud = None if pointcloud is None else torch.as_tensor(pointcloud, **f32)
        self.jm = torch.as_tensor(joint_stats["mean"], **f32)[:n_arm]
        self.js = torch.as_tensor(joint_stats["std"], **f32)[:n_arm]
        # the plan's predicted joint STATES share the action normalization only
        # if recorded identically; use obs stats for the state block instead
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:n_arm]
        self.ss = torch.as_tensor(obs_stats["std"], **f32)[:n_arm]
        ci = cube_index
        self.cm = self.cs = None
        if ci is not None:
            self.cm = torch.as_tensor(obs_stats["mean"], **f32)[ci : ci + 3]
            self.cs = torch.as_tensor(obs_stats["std"], **f32)[ci : ci + 3]
        self.joint_start, self.n_arm, self.cube_index = joint_start, n_arm, ci
        self.rad = self.fc.radius
        self.om = torch.as_tensor(obs_stats["mean"], **f32)
        self.ov = torch.as_tensor(obs_stats["std"], **f32)
        self.ee_index = ee_index
        self.progress_w = 0.0

    def set_boxes(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float32)
        self.box = torch.as_tensor(boxes, device=self.device).view(-1, 6)

    @torch.no_grad()
    def score(self, traj: Tensor) -> Tensor:
        """Total penetration. Ranking on clearance instead games the goal clamp
        -- the safest plan never approaches the cube at all -- so clearance buys
        nothing and every collision-free plan ties."""
        boxes = self.box
        if len(boxes) > 1:
            boxes = boxes.repeat_interleave(len(traj) // len(boxes), dim=0)
        out = []
        for i in range(0, len(traj), 32):
            c = traj[i : i + 32]
            pen = (-self.clearance_t(c, boxes[i : i + 32])).clamp(min=0.0).sum(dim=1)
            out.append(pen + self.body_penetration(c))
        score = torch.cat(out)
        if self.progress_w and self.ee_index is not None:
            obj = self.cube(traj)
            if obj is not None:
                j = self.ee_index
                ee = traj[..., j : j + 3] * self.ov[j : j + 3] + self.om[j : j + 3]
                reach = (ee - obj[:, :1]).norm(dim=-1).amin(dim=1)
                score = score + self.progress_w * reach
        return score

    def dist(self, points: Tensor, box: Tensor) -> Tensor:
        if self.cloud is None:
            shape = (len(box),) + (1,) * (points.dim() - 2) + (3,)
            return box_sdf(points, box[:, :3].view(shape), box[:, 3:].view(shape))
        p = points.reshape(-1, 3)
        step = 1 << 19
        out = [
            torch.cdist(p[i : i + step], self.cloud).amin(dim=1)
            for i in range(0, len(p), step)
        ]
        return torch.cat(out).reshape(points.shape[:-1])

    def joints(self, traj: Tensor) -> tuple[Tensor, Tensor]:
        s, n = self.joint_start, self.n_arm
        return traj[..., s : s + n] * self.js + self.jm, traj[
            ..., :n
        ] * self.ss + self.sm

    def cube(self, traj: Tensor) -> Tensor | None:
        if self.cube_index is None or self.cs is None or self.cm is None:
            return None
        return traj[..., self.cube_index : self.cube_index + 3] * self.cs + self.cm

    def clearance_t(self, traj: Tensor, box: Tensor | None = None) -> Tensor:
        """Per-timestep worst clearance (b, t): min over arm points, the cube, and
        both joint sources."""
        if box is None:
            box = self.box
        n = self.n_arm
        worst = []
        # require both the joint targets and the plan's own predicted states to
        # clear: the two disagree by the controller's tracking lag
        for q in self.joints(traj):
            b, t = q.shape[:2]
            pts = self.fc.arm_points(q.reshape(-1, n).float()).reshape(b, t, -1, 3)
            d = self.dist(pts + self.fc.base_pos, box)
            worst.append((d - self.rad).amin(dim=2))  # (b, t), mesh not centreline
        cube = self.cube(traj)
        if cube is not None:
            worst.append(self.dist(cube.float(), box) - self.fc.cube_half)
        return torch.stack(worst).amin(dim=0)

    def body_penetration(self, traj: Tensor) -> Tensor:
        """(b,) self- and arm-vs-held-cube penetration summed over strided frames."""
        q = self.joints(traj)[0][:, ::SELF_STRIDE]
        b, t = q.shape[:2]
        cube = self.cube(traj)
        if cube is not None:
            cube = (cube[:, ::SELF_STRIDE].float() - self.fc.base_pos).reshape(-1, 3)
        pen, _ = self.fc.body_penetration(q.reshape(-1, self.n_arm).float(), cube)
        return SELF_STRIDE * pen.reshape(b, t).sum(dim=1)
