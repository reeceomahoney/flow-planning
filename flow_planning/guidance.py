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
    pushed off the obstacle by a squared hinge on its box SDF. With `over_top`,
    waypoints inside the obstacle's xy footprint are also lifted above it (a
    thin obstacle's SDF gradient is sideways, which would split a crossing path
    around the obstacle instead of routing it over). The returned gradient
    lives in the policy's normalized action space.
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
        over_top: bool = False,
    ):
        kw = {"dtype": torch.float32, "device": device}
        self.center = torch.as_tensor(center, **kw)
        self.half_extents = torch.as_tensor(half_extents, **kw)
        d = len(self.center)
        self.mean = torch.as_tensor(action_mean, **kw)[:d]
        self.std = torch.as_tensor(action_std, **kw)[:d]
        self.scale = scale
        self.margin = margin
        self.over_top = over_top

    def __call__(self, x1_hat: Tensor) -> Tensor:
        d = len(self.center)
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            pos = x[..., :d] * self.std + self.mean
            sdf = box_sdf(pos, self.center, self.half_extents)
            cost = torch.relu(self.margin - sdf).square().sum()

            if self.over_top:
                q = (pos - self.center).abs()
                band = (q[..., 0] < self.half_extents[0] + self.margin) & (
                    q[..., 1] < self.half_extents[1] + self.margin
                )
                z_top = self.center[2] + self.half_extents[2]
                over = torch.relu(z_top + self.margin - pos[..., 2]) * band
                cost = cost + over.square().sum()
            (grad,) = torch.autograd.grad(cost, x)
        return self.scale * grad
