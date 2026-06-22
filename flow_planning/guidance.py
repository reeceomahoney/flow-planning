"""Inference-time cost guidance for the flow-matching policy."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GuidanceConfig:
    # obstacle scale/margin are env-specific, so they live on the env config
    smooth_scale: float = 0.5  # >0 adds a plan acceleration penalty (smoothness)


def box_sdf(points: Tensor, center: Tensor, half_extents: Tensor) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


class SmoothGuidance:
    """Acceleration penalty over the planned trajectory, for smoother paths and
    actions. Composes with the goal/obstacle terms via the same gradient sum."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x1_hat: Tensor, obs: Tensor) -> Tensor:
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            accel = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
            cost = self.scale * accel.square().sum()
            (grad,) = torch.autograd.grad(cost, x)
        return grad


class ObstacleGuidance:
    """Collision-cost gradient on a normalized action chunk.

    The leading `len(center)` action dims are unnormalized into positions and
    pushed off the obstacle by a squared hinge on its box SDF. The cost is
    evaluated on points interpolated along the path so segments crossing a thin
    obstacle between waypoints are also penalized. A thin obstacle's SDF
    gradient points off its broad face, which stalls a crossing path against it
    instead of routing it past an edge, so points inside the blocked band can
    additionally be pushed past the nearest free edge: `around` routes them
    beyond the obstacle's x extent, `over_top` lifts them above it. The
    returned gradient lives in the policy's normalized action space.
    `center`/`half_extents` are one box (d,) or per-batch boxes (B, d).
    """

    def __init__(
        self,
        center,
        half_extents,
        action_mean,
        action_std,
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
        # the avoided position is the leading d action dims (EE/target position);
        # actions sit after the state in the trajectory, hence pos_start
        self.pos_start = pos_start
        self.mean = torch.as_tensor(action_mean, **kw)[:d]
        self.std = torch.as_tensor(action_std, **kw)[:d]
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


class Guidance:
    """Sum of the enabled guidance terms, with a per-rollout reset for stateful
    terms. `action_stats` is the action `{"mean", "std"}` (the obstacle term acts
    on the action's leading position dims, which start at `state_dim` in the
    trajectory); `env` supplies the obstacle geometry. With no term enabled the
    sum is a harmless zero."""

    def __init__(self, cfg, env, state_dim, action_stats, device):
        self.terms = []
        if env.cfg.obstacle and env.cfg.obstacle_scale > 0:
            self.terms.append(
                ObstacleGuidance(
                    **env.obstacle_geometry,
                    action_mean=action_stats["mean"],
                    action_std=action_stats["std"],
                    scale=env.cfg.obstacle_scale,
                    margin=env.cfg.obstacle_margin,
                    device=device,
                    pos_start=state_dim,
                )
            )
        if cfg.smooth_scale > 0:
            self.terms.append(SmoothGuidance(scale=cfg.smooth_scale))

    def __call__(self, x1, obs):
        return sum(g(x1, obs) for g in self.terms)

    def reset(self):
        for g in self.terms:
            getattr(g, "reset", lambda: None)()
