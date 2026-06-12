"""Inference-time cost guidance for the flow-matching policy."""

import torch
from torch import Tensor


def box_sdf(points: Tensor, center: Tensor, half_extents: Tensor) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


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
        self.mean = torch.as_tensor(action_mean, **kw)[:d]
        self.std = torch.as_tensor(action_std, **kw)[:d]
        self.scale = scale
        self.margin = margin
        self.around = around
        self.over_top = over_top
        self.interp = interp

    def path_points(self, pos: Tensor) -> Tensor:
        """Interpolate `interp` points per segment along the path, plus the end."""
        if self.interp <= 1:
            return pos
        a, b = pos[..., :-1, :], pos[..., 1:, :]
        s = torch.linspace(0.0, 1.0, self.interp + 1, device=pos.device)[:-1]
        pts = a.unsqueeze(-2) + s.view(-1, 1) * (b - a).unsqueeze(-2)
        return torch.cat([pts.flatten(-3, -2), pos[..., -1:, :]], dim=-2)

    def __call__(self, x1_hat: Tensor) -> Tensor:
        d = self.center.shape[-1]
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            pos = x[..., :d] * self.std + self.mean
            pts = self.path_points(pos)
            sdf = box_sdf(pts, self.center, self.half_extents)
            cost = torch.relu(self.margin - sdf).square().sum()

            q = (pts - self.center).abs()
            if self.around:
                band = q[..., 1] < self.half_extents[..., 1] + self.margin
                # one detour side per path, else points dither around x=0
                xrel = (pts[..., 0] - self.center[..., 0]).detach()
                side = torch.sign((xrel * band).sum(-1, keepdim=True))
                side = torch.where(side == 0, torch.ones_like(side), side)
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
            (grad,) = torch.autograd.grad(cost, x)
        return self.scale * grad
