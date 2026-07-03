"""Inference-time cost guidance for the flow-matching policy."""

import os

import torch
import torch.nn.functional as F
from torch import Tensor

from flow_planning.kinematics import build_franka_chain, ee_positions


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
    the end (a detour), instead of the local tear a box-collision cost produces.

    The ellipse lives in the plane of `travel` (start->goal axis) and `exit_axis`
    (detour axis): particle guides the action xy round the bar end in x, franka
    guides the EE state xyz over the wall top in z. `guide_stats` normalize the
    guided block; `start_stats`/`goal_stats` read the anchors out of the obs.
    `anchored` picks the travel-extent rule: static visit points that must stay
    graspable (cube/goal) shrink the keep-out to the nearer one; a moving current
    position (particle ball) spans the cross-box gap instead."""

    def __init__(
        self,
        box,  # physical [center(d), half_extents(d)]
        guide_stats,  # (mean, std) of the guided trajectory block
        start_stats,  # (mean, std) of the current-position obs block
        goal_stats,  # (mean, std) of the goal obs block (post goal-aliasing)
        pos_start: int,  # guided block offset in the trajectory
        start_index: int,  # current-position offset in the obs
        travel: int,
        exit_axis: int,
        device,
        scale: float,
        margin: float = 0.15,
        ay_scale: float = 0.85,
        bias: float = 0.0,
        exit_side: float = 0.0,  # fixed side along exit_axis; 0 = pick per batch
        anchored: bool = False,  # True: anchors are static visit points (cube/goal)
    ):
        def t(v):
            return torch.as_tensor(v, dtype=torch.float32, device=device)

        box = t(box)
        d = box.shape[0] // 2
        self.d, self.pos_start, self.start_index = d, pos_start, start_index
        self.travel, self.exit_axis = travel, exit_axis
        self.gm, self.gs = t(guide_stats[0])[:d], t(guide_stats[1])[:d]
        self.sm, self.ss = t(start_stats[0])[:d], t(start_stats[1])[:d]
        self.qm, self.qs = t(goal_stats[0])[:d], t(goal_stats[1])[:d]
        self.center = (box[:d] - self.gm) / self.gs  # guided-normalized space
        self.half = box[d:] / self.gs
        self.scale, self.margin, self.ay_scale = scale, margin, ay_scale
        self.bias = bias  # symmetry break: push inside points toward the exit
        self.anchored = anchored
        self.fixed_side = exit_side
        # per-batch fallback for a start/goal tie: exit round the roomier end
        self.tie_side = 1.0 if float(box[exit_axis]) <= 0 else -1.0

    def reset(self):
        pass

    def __call__(self, x1_hat: Tensor, obs: Tensor | None = None) -> Tensor:
        assert obs is not None
        d, tv, ex = self.d, self.travel, self.exit_axis
        si = self.start_index
        # start/goal mapped into the normalized space the guided dims live in
        start = (obs[:, si : si + d] * self.ss + self.sm - self.gm) / self.gs
        goal = (obs[:, -d:] * self.qs + self.qm - self.gm) / self.gs
        c = self.center
        a_exit = self.half[ex] + self.margin  # cover the box -> exit past its end
        if self.anchored:
            # static visit points (grasp/place) must stay outside the keep-out:
            # reach to the nearer one, floor covers only the box itself
            span = torch.minimum(
                (start[:, tv] - c[tv]).abs(), (goal[:, tv] - c[tv]).abs()
            )
            a_travel = (self.ay_scale * span).clamp_min(self.half[tv])
        else:
            # start is the CURRENT position: it legitimately enters the keep-out
            # mid-detour, so span the cross-box gap instead of shrinking to it
            span = 0.5 * (start[:, tv] - goal[:, tv]).abs()
            a_travel = (self.ay_scale * span).clamp_min(self.half[tv] + self.margin)
        if self.fixed_side != 0.0:
            side = start.new_full((start.shape[0],), self.fixed_side)
        else:  # exit the side nearer start/goal (tie_side if centered)
            mid = 0.5 * (start[:, ex] + goal[:, ex]) - c[ex]
            side = torch.where(
                mid.abs() < 0.15, mid.new_full((), self.tie_side), mid.sign()
            )
        with torch.enable_grad():
            x1_hat = x1_hat.detach().requires_grad_(True)
            p = x1_hat[..., self.pos_start : self.pos_start + d]
            eu = (p[..., ex] - c[ex]) / a_exit
            ev = (p[..., tv] - c[tv]) / a_travel[:, None]
            e = eu * eu + ev * ev  # <1 inside the ellipse
            inside = F.softplus(1.0 - e)
            cost = self.scale * inside.sum()  # penalize inside
            if self.bias > 0:  # break symmetry: drive inside points to the exit side
                cost = (
                    cost - self.scale * self.bias * (inside * side[:, None] * eu).sum()
                )
            (grad,) = torch.autograd.grad(cost, x1_hat)
        return grad


class FKEllipseGuidance:
    """EllipseGuidance's anchored keep-out, evaluated at the FK EE position of the
    planned JOINT actions so the gradient reaches the dims that are executed
    (pushing the EE *state* dims relies on model coupling, which proved weak).
    Works in physical world space: `margin` is metres, anchors are the cube/goal
    obs blocks, and the exit is fixed up (+z)."""

    def __init__(
        self,
        chain,
        box,  # physical world [center(3), half_extents(3)]
        base_pos,  # robot base -> world, added to FK outputs
        joint_stats,  # (mean, std) of the 7 arm joint action dims
        anchor_stats,  # (mean, std) of the cube obs block (goal aliases to it)
        anchor_index: int,  # cube block offset in the obs
        joint_start: int,  # joint action dims offset in the trajectory
        device,
        scale: float,
        margin: float,
        ay_scale: float,
        bias: float,
        travel: int = 1,
        exit_axis: int = 2,
        n_arm: int = 7,
        stride: int = 2,
    ):
        def t(v):
            return torch.as_tensor(v, dtype=torch.float32, device=device)

        box = t(box)
        self.chain = chain
        self.center, self.half = box[:3], box[3:]
        self.base_pos = t(base_pos)
        self.jm, self.js = t(joint_stats[0])[:n_arm], t(joint_stats[1])[:n_arm]
        self.am, self.as_ = t(anchor_stats[0])[:3], t(anchor_stats[1])[:3]
        self.anchor_index, self.joint_start = anchor_index, joint_start
        self.scale, self.margin, self.ay_scale, self.bias = (
            scale,
            margin,
            ay_scale,
            bias,
        )
        self.travel, self.exit_axis = travel, exit_axis
        self.n_arm, self.stride = n_arm, stride

    def reset(self):
        pass

    def __call__(self, x1_hat: Tensor, obs: Tensor | None = None) -> Tensor:
        assert obs is not None
        tv, ex, ai = self.travel, self.exit_axis, self.anchor_index
        cube = obs[:, ai : ai + 3] * self.as_ + self.am  # physical anchors
        goal = obs[:, -3:] * self.as_ + self.am
        c = self.center
        a_exit = self.half[ex] + self.margin
        span = torch.minimum((cube[:, tv] - c[tv]).abs(), (goal[:, tv] - c[tv]).abs())
        a_travel = (self.ay_scale * span).clamp_min(self.half[tv])
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            s = self.joint_start
            q = x[..., s : s + self.n_arm] * self.js + self.jm  # (B, H, n_arm)
            q = q[:, :: self.stride]
            b, h = q.shape[:2]
            ee = ee_positions(self.chain, q.reshape(-1, self.n_arm)).reshape(b, h, 3)
            ee = ee + self.base_pos  # world frame
            eu = (ee[..., ex] - c[ex]) / a_exit
            ev = (ee[..., tv] - c[tv]) / a_travel[:, None]
            inside = F.softplus(1.0 - (eu * eu + ev * ev))
            # penalize inside; bias drives inside points up toward the fixed exit
            cost = self.scale * (inside - self.bias * inside * eu).sum()
            (grad,) = torch.autograd.grad(cost, x)
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
        # ellipse keep-out. particle: guide the executed action xy round the bar's
        # end. franka: actions are joints, so FK the planned joints and push the
        # resulting EE path over the wall top (state-dim coupling proved weak).
        if geom and obs_stats is not None and env.cfg.ellipse_scale > 0:
            om, os_ = obs_stats["mean"], obs_stats["std"]
            if arm:
                gs = env.cfg.goal_state_start
                chain, _, _ = build_franka_chain(device)
                self.terms.append(
                    FKEllipseGuidance(
                        chain,
                        list(geom["center"]) + list(geom["half_extents"]),
                        list(env.robot_base_pos),
                        (action_stats["mean"], action_stats["std"]),
                        (om[gs : gs + 3], os_[gs : gs + 3]),
                        anchor_index=gs,
                        joint_start=state_dim,
                        device=device,
                        scale=env.cfg.ellipse_scale,
                        margin=env.cfg.ellipse_margin,
                        ay_scale=env.cfg.ellipse_ay,
                        bias=env.cfg.ellipse_bias,
                        stride=env.cfg.arm_stride,
                    )
                )
            else:
                c = torch.as_tensor(geom["center"], dtype=torch.float32)[0]
                h = torch.as_tensor(geom["half_extents"], dtype=torch.float32)[0]
                am, as_ = action_stats["mean"], action_stats["std"]
                self.terms.append(
                    EllipseGuidance(
                        torch.cat([c, h]),
                        (am[:2], as_[:2]),
                        (om[:2], os_[:2]),
                        (om[:2], os_[:2]),
                        pos_start=state_dim,
                        start_index=0,
                        travel=1,
                        exit_axis=0,  # round the bar end, side picked per batch
                        device=device,
                        scale=env.cfg.ellipse_scale,
                        margin=env.cfg.ellipse_margin,
                        ay_scale=env.cfg.ellipse_ay,
                        bias=env.cfg.ellipse_bias,
                    )
                )

    def __call__(self, x, v, t, obs):
        x1_hat = x + (1 - t.view(-1, 1, 1)) * v
        return sum(g(x1_hat, obs) for g in self.terms)

    def reset(self):
        for g in self.terms:
            getattr(g, "reset", lambda: None)()
