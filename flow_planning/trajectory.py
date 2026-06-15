"""Phase (arc-length) parameterization and path tracking for Design B.

Plans are a fixed number of support points; their physical length is decoupled
and handled by the tracker at execution time. This lets endpoint inpainting pin
the goal at a known index while the realized horizon stays flexible.
"""

import numpy as np
import torch
from lerobot.utils.constants import OBS_STATE
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

    # forward-fill padded points with the last valid point (the goal) so their
    # values never leak into the interpolation
    lv = (valid.sum(1) - 1).clamp(min=0)
    last = p.gather(1, lv.view(B, 1, 1).expand(B, 1, D))
    p = torch.where(valid.unsqueeze(-1), p, last)

    seg = (p[:, 1:, :d] - p[:, :-1, :d]).norm(dim=-1)  # (B, H-1)
    zero = torch.zeros(B, 1, device=p.device, dtype=seg.dtype)
    arclen = torch.cat([zero, seg.cumsum(-1)], dim=1)  # (B, H), plateaus past the end
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
        dist = (plan[..., :d] - pos[:, None, :]).norm(dim=-1)  # (B, N)
        closest = dist.argmin(1, keepdim=True)  # (B, 1)
        s_goal = arclen.gather(1, closest) + self.lookahead
        idx = torch.searchsorted(arclen, s_goal, right=True).clamp(max=N - 1)
        return plan.gather(1, idx.view(B, 1, 1).expand(B, 1, D)).squeeze(1)


def normalize(x, mean: Tensor, std: Tensor, pos_dim: int) -> Tensor:
    """Physical (..., pos_dim) -> normalized action space using leading stats."""
    x = torch.as_tensor(x, dtype=torch.float32, device=mean.device)
    return (x - mean[:pos_dim]) / std[:pos_dim]


def unnormalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Normalized action chunk -> physical."""
    return x * std + mean


class Planner:
    """Plan a geometric path with endpoint inpainting, then pure-pursuit track it.

    `act(obs)` replans every `replan_every` steps and returns the per-frame
    action; the realized horizon is whatever the tracker takes to reach the goal.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        env,
        action_mean,
        action_std,
        lookahead: float,
        replan_every: int,
        guidance_fn=None,
        inpaint: bool = True,
        device: str = "cuda",
    ):
        self.policy = policy
        self.pre = preprocessor
        self.env = env
        self.device = device
        self.mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
        self.pos_dim = policy.config.position_dim or self.mean.shape[0]
        self.tracker = PathTracker(self.pos_dim, lookahead)
        self.replan_every = replan_every
        self.guidance_fn = guidance_fn
        self.inpaint = inpaint
        self.step = 0

    def reset(self) -> None:
        self.step = 0
        self.tracker.plan = None

    @torch.no_grad()
    def replan(self, obs: np.ndarray) -> None:
        start, goal = self.env.endpoints(obs)
        kw = {"guidance_fn": self.guidance_fn}
        if self.inpaint:
            kw["inpaint"] = {
                "start": normalize(start, self.mean, self.std, self.pos_dim),
                "goal": normalize(goal, self.mean, self.std, self.pos_dim),
            }
        batch = self.pre({OBS_STATE: torch.from_numpy(obs)})
        plan = self.policy.predict_action_chunk(batch, **kw)
        self.tracker.set_plan(unnormalize(plan, self.mean, self.std))

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.tracker.plan is None or self.step % self.replan_every == 0:
            self.replan(obs)
        self.step += 1
        pos = torch.as_tensor(
            self.env.endpoints(obs)[0], dtype=torch.float32, device=self.device
        )
        return self.tracker.target(pos).cpu().numpy().astype(np.float32)
