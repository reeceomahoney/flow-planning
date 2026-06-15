"""Phase (arc-length) parameterization and path tracking for Design B.

Plans are a fixed number of support points; their physical length is decoupled
and handled by the tracker at execution time. This lets endpoint inpainting pin
the goal at a known index while the realized horizon stays flexible.
"""

import torch
from torch import Tensor


def resample_arc_length(
    p: Tensor, pad: Tensor | None, n: int, pos_dim: int = 0
) -> Tensor:
    """Resample each path to `n` points evenly spaced by arc length.

    Arc length uses the leading `pos_dim` dims (0 = all dims); every dim is then
    interpolated at those phase locations. `pad` (True = padded/invalid, a suffix
    per row as produced by lerobot) marks points past the trajectory end.
    """
    B, H, D = p.shape
    d = pos_dim or D
    valid = torch.ones(B, H, dtype=torch.bool, device=p.device) if pad is None else ~pad

    # forward-fill the padded tail with the last valid point so it can't skew arc length
    lv = (valid.sum(1) - 1).clamp(min=0)
    last = p.gather(1, lv.view(B, 1, 1).expand(B, 1, D))
    p = torch.where(valid.unsqueeze(-1), p, last)

    seg = (p[:, 1:, :d] - p[:, :-1, :d]).norm(dim=-1)
    zero = torch.zeros(B, 1, device=p.device, dtype=seg.dtype)
    arclen = torch.cat([zero, seg.cumsum(-1)], dim=1)  # plateaus past the end
    total = arclen[:, -1:]
    s = torch.where(
        total > 1e-8, arclen / total.clamp(min=1e-8), torch.zeros_like(arclen)
    )

    u = torch.linspace(0.0, 1.0, n, device=p.device).expand(B, n)
    idx = torch.searchsorted(s, u, right=True).clamp(1, H - 1)
    s0, s1 = s.gather(1, idx - 1), s.gather(1, idx)
    w = ((u - s0) / (s1 - s0).clamp(min=1e-8)).clamp(0.0, 1.0).unsqueeze(-1)
    p0 = p.gather(1, (idx - 1).unsqueeze(-1).expand(B, n, D))
    p1 = p.gather(1, idx.unsqueeze(-1).expand(B, n, D))
    return p0 + w * (p1 - p0)


class PathTracker:
    """Vectorized pure-pursuit tracker over a batch of geometric plans.

    Holds one plan per world; `target` returns the full action vector a fixed
    arc-length `lookahead` ahead of the closest point to the current position.
    """

    def __init__(self, pos_dim: int, lookahead: float):
        self.pos_dim = pos_dim
        self.lookahead = lookahead
        self.plan: Tensor | None = None
        self.arclen: Tensor | None = None

    def set_plan(self, plan: Tensor) -> None:
        """plan: (B, N, D) physical action waypoints."""
        self.plan = plan
        d = self.pos_dim or plan.shape[-1]
        seg = (plan[:, 1:, :d] - plan[:, :-1, :d]).norm(dim=-1)
        zero = torch.zeros(plan.shape[0], 1, device=plan.device, dtype=seg.dtype)
        self.arclen = torch.cat([zero, seg.cumsum(-1)], dim=1)

    def target(self, pos: Tensor) -> Tensor:
        """pos: (B, pos_dim) current position. Returns (B, D) action target."""
        assert self.plan is not None and self.arclen is not None
        plan, arclen = self.plan, self.arclen
        B, N, D = plan.shape
        d = self.pos_dim or D
        dist = (plan[..., :d] - pos[:, None, :]).norm(dim=-1)
        closest = dist.argmin(1, keepdim=True)
        s_goal = arclen.gather(1, closest) + self.lookahead
        idx = torch.searchsorted(arclen, s_goal, right=True).clamp(max=N - 1)
        return plan.gather(1, idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
