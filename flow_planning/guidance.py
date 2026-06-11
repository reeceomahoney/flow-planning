"""Inference-time cost guidance for the flow-matching policy."""

import torch
from torch import Tensor


def box_sdf(points: Tensor, center: Tensor, half_extents: Tensor) -> Tensor:
    """Signed distance from points (..., 3) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


class WallGuidance:
    """Collision-cost gradient on a normalized action chunk.

    Two terms, evaluated on the unnormalized EE positions of the predicted
    clean chunk: a squared hinge on the wall's box SDF (clearance around the
    whole wall), and an over-the-top term that lifts waypoints inside the
    wall's xy footprint above it (a thin wall's SDF gradient is sideways, which
    would split a crossing path around the wall instead of routing it over).
    The returned gradient lives in the policy's normalized action space.
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
    ):
        kw = {"dtype": torch.float32, "device": device}
        self.center = torch.as_tensor(center, **kw)
        self.half_extents = torch.as_tensor(half_extents, **kw)
        self.mean = torch.as_tensor(action_mean, **kw)[:3]
        self.std = torch.as_tensor(action_std, **kw)[:3]
        self.scale = scale
        self.margin = margin

    def __call__(self, x1_hat: Tensor) -> Tensor:
        with torch.enable_grad():
            x = x1_hat.detach().requires_grad_(True)
            pos = x[..., :3] * self.std + self.mean
            sdf = box_sdf(pos, self.center, self.half_extents)
            cost = torch.relu(self.margin - sdf).square().sum()

            d = (pos - self.center).abs()
            band = (d[..., 0] < self.half_extents[0] + self.margin) & (
                d[..., 1] < self.half_extents[1] + self.margin
            )
            z_top = self.center[2] + self.half_extents[2]
            over = torch.relu(z_top + self.margin - pos[..., 2]) * band
            cost = cost + over.square().sum()
            (grad,) = torch.autograd.grad(cost, x)
        return self.scale * grad
