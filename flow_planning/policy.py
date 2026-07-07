"""A small flow-matching transformer policy in the lerobot policy style.

Outer normalization is handled by the processor pipeline (see
`make_flow_matching_pre_post_processors`), so `forward` and `predict_action_chunk`
operate on already-normalized tensors. `select_action` follows the standard
lerobot API: it plans a [state, action] chunk and executes the action dims one
step per call, replanning every `replan_every` steps (receding horizon).
"""

import copy
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from torch import Tensor


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Config for the flow-matching transformer policy."""

    # dimensions
    horizon: int = 50
    n_action_steps: int = 75
    goal_dim: int = field(kw_only=True)  # trailing obs dims; set from env.goal_dim
    goal_state_start: int = 0  # state block the goal clamps onto; set from the env

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # flow matching
    num_inference_steps: int = 10
    action_lpf: float = 0.3  # EMA alpha on executed actions; <1 smooths seam jumps

    # bend-parameter conditioning (0 disables); dropout trains the null token
    cond_dim: int = 0
    cond_dropout: float = 0.15
    adaln: bool = True  # ignored (AdaLN-Zero is the only path); kept for ckpt compat

    # training
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-2

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError(
                "Flow matching policy requires an observation.state input."
            )
        if self.action_feature is None:
            raise ValueError("Flow matching policy requires an action output.")

    # unused (train.py gathers windows itself) but abstract in PreTrainedConfig
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer over global SDPA, with AdaLN-Zero
    modulation by a conditioning vector (DiT-style)."""

    def __init__(self, d: int, n_heads: int, dim_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.GELU(), nn.Linear(dim_ff, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def attn(self, h: Tensor) -> Tensor:
        b, n, d = h.shape
        qkv = self.qkv(h).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
        o = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.proj(o.transpose(1, 2).reshape(b, n, d))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        sc1, sh1, g1, sc2, sh2, g2 = self.mod(c)[:, None].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + sh1)
        return x + g2 * self.ff(self.norm2(x) * (1 + sc2) + sh2)


class FlowTransformer(nn.Module):
    """Unconditional velocity field v(x_t, t) over a chunk of [state, action] steps.

    No boundary tokens: the start and goal are imposed at inference by clamping the
    trajectory's first/last frames (inpainting); global attention spreads them across
    the horizon.
    """

    def __init__(self, traj_dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        d = cfg.dim_model
        self.horizon = cfg.horizon
        self.traj_in = nn.Linear(traj_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos_emb = nn.Parameter(torch.randn(self.horizon, d) * 0.02)
        if cfg.cond_dim:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cfg.cond_dim, d), nn.GELU(), nn.Linear(d, d)
            )
            self.null_cond = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(
            EncoderLayer(d, cfg.n_heads, 4 * d) for _ in range(cfg.n_layers)
        )
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, traj_dim)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor | None = None,
        cond_drop: Tensor | None = None,
    ) -> Tensor:
        # x_t: (B, H, traj_dim), t: (B,), cond: (B, cond_dim) or None (-> null).
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        c = temb
        if hasattr(self, "cond_mlp"):
            ce = self.null_cond.expand(x_t.shape[0], -1)
            if cond is not None:
                ce = self.cond_mlp(cond)
                if cond_drop is not None:
                    ce = torch.where(cond_drop[:, None], self.null_cond, ce)
            c = c + ce
        h = self.traj_in(x_t) + self.pos_emb
        for layer in self.layers:
            h = layer(h, c)
        return self.head(self.norm_out(h))


class FlowMatchingPolicy(PreTrainedPolicy):
    config_class = FlowMatchingConfig
    name = "flow_matching"
    config: FlowMatchingConfig

    def __init__(self, config: FlowMatchingConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        assert config.robot_state_feature is not None
        assert config.action_feature is not None
        obs_dim = config.robot_state_feature.shape[0]
        act_dim = config.action_feature.shape[0]
        self.act_dim = act_dim
        # trajectory = [state, action] per step; state is the non-goal obs (pos+vel)
        self.state_dim = obs_dim - config.goal_dim
        self.traj_dim = self.state_dim + act_dim

        self.model = FlowTransformer(self.traj_dim, config)
        self.action_clip = None  # optional (low, high) normalized action bounds
        self.selector_fn = None  # optional plan scorer for best-of-N selection
        self.n_samples = 1  # plans sampled per world when selector_fn is set
        self.cond = None  # commanded bend params (1, cond_dim), None = null token
        self.cond_candidates = None  # (K, cond_dim): search + latch via selector_fn
        self.cond_opt = None  # gradient latent search: dict(cost, inits, bounds, ...)
        self.warm_start_t = 0.0  # >0: renoise the shifted previous plan to this t
        # and integrate t..1, keeping timing/mode continuity across replans
        self.reset()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Policy parameters: {n_params / 1e6:.2f}M")

    def get_optim_params(self) -> Iterator[nn.Parameter]:  # ty: ignore[invalid-method-override]
        return self.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.last_chunk = None
        self.last_traj = None
        self.lpf_state = None  # EMA state for the executed-action low-pass
        self.latched_cond = None  # per-world bend picked at the first replan

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # full-trajectory window: [state, action] per step, both normalized. The
        # tail past the episode end clamps to the goal frame (pos=goal, vel~0,
        # action=goal target). Unconditional flow: x_t already carries the real
        # start/goal in its first/last frames, so the net learns to use them; at
        # inference those frames are clamped to the commanded values (inpainting).
        # ponytail: assumes lerobot pads delta frames by clamping to the episode end.
        obs = batch[OBS_STATE]  # (B, H, obs_dim); trailing dims = goal
        sd, gd = self.state_dim, self.config.goal_dim
        gs = self.config.goal_state_start
        x_1 = torch.cat([obs[..., :sd], batch[ACTION]], dim=-1)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        # boundary frames are clean conditioning context (start state, goal pos),
        # not noised; exclude them from the loss since their velocity is undefined.
        x_t[:, 0, :sd] = x_1[:, 0, :sd]
        x_t[:, -1, gs : gs + gd] = x_1[:, -1, gs : gs + gd]
        cond = batch.get("bend") if self.config.cond_dim else None
        drop = None
        if cond is not None:
            drop = (
                torch.rand(cond.shape[0], device=cond.device) < self.config.cond_dropout
            )
        v = self.model(x_t, t, cond, drop)
        err = (v - (x_1 - x_0)) ** 2
        mask = torch.ones_like(err)
        mask[:, 0, :sd] = 0.0
        mask[:, -1, gs : gs + gd] = 0.0
        loss = (err * mask).sum() / mask.sum()
        return loss, {"loss": loss.item()}

    def flow_integrate(
        self,
        x: Tensor,
        state: Tensor,
        goal: Tensor,
        cond: Tensor | None,
        t0: float = 0.0,
        ckpt: bool = False,
    ) -> Tensor:
        """ODE-integrate x from t0 to 1, re-clamping the boundary frames each
        step. `ckpt` checkpoints each step so gradients (w.r.t. cond) fit."""
        sd, gd = self.state_dim, self.config.goal_dim
        gs = self.config.goal_state_start
        dt = (1.0 - t0) / self.config.num_inference_steps

        def step(x, t, cond):
            x = x.clone()
            x[:, 0, :sd] = state
            x[:, -1, gs : gs + gd] = goal
            return x + dt * self.model(x, t, cond)

        for i in range(self.config.num_inference_steps):
            t = torch.full((x.shape[0],), t0 + i * dt, device=x.device)
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    step, x, t, cond, use_reentrant=False
                )
            else:
                x = step(x, t, cond)
        return x

    @torch.enable_grad()
    def optimize_cond(self, obs: Tensor):
        """Gradient-descend the bend latent through the ODE from a few restarts,
        latch the best per world (by the hard selector score). Fixed noise per
        restart makes the objective deterministic."""
        o = self.cond_opt
        assert o is not None and self.selector_fn is not None
        n, r = obs.shape[0], o["inits"].shape[0]
        sd, gd = self.state_dim, self.config.goal_dim
        state = obs[:, :sd].repeat_interleave(r, dim=0)
        goal = obs[:, -gd:].repeat_interleave(r, dim=0)
        cond = o["inits"].repeat(n, 1).clone().requires_grad_(True)
        x0 = torch.randn(n * r, self.config.horizon, self.traj_dim, device=obs.device)
        lo, hi = o["bounds"]
        opt = torch.optim.Adam([cond], lr=o["lr"])
        mb = 128  # grad accumulation: ~90MB per 2 plans through the ODE
        for _ in range(o["iters"]):
            opt.zero_grad()
            for i in range(0, n * r, mb):
                x = self.flow_integrate(
                    x0[i : i + mb],
                    state[i : i + mb],
                    goal[i : i + mb],
                    cond[i : i + mb],
                    ckpt=True,
                )
                o["cost"](x).sum().backward()
            opt.step()
            with torch.no_grad():
                cond.clamp_(lo, hi)
        with torch.no_grad():
            x = self.flow_integrate(x0, state, goal, cond.detach())
            pick = self.selector_fn(x).view(n, r).argmin(dim=1)
            rows = torch.arange(n, device=obs.device)
            self.latched_cond = cond.detach().view(n, r, -1)[rows, pick]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Flow-ODE integrate a [state, action] trajectory, conditioning by
        inpainting: each step clamp frame 0's state to the current state and frame
        -1's goal dims to the goal (already in position space, see alias_goal_stats).
        Returns the (B, horizon, act_dim) action chunk."""
        self.eval()
        obs = batch[OBS_STATE]
        sel = self.selector_fn
        n_worlds = obs.shape[0]
        ns = self.n_samples if sel is not None else 1
        # bend conditioning: fixed command, latched pick, or a search (gradient
        # descent on the latent, or a candidate grid); ns samples per mode are
        # drawn and the selector keeps the best plan
        cond, k, searching = None, ns, False
        if self.config.cond_dim and self.cond_opt is not None:
            if self.latched_cond is None:
                self.optimize_cond(obs)
            assert self.latched_cond is not None
            cond = self.latched_cond.repeat_interleave(ns, dim=0)
        elif self.config.cond_dim and self.cond_candidates is not None:
            if self.latched_cond is not None:
                cond = self.latched_cond.repeat_interleave(ns, dim=0)
            else:
                searching = True
                k = self.cond_candidates.shape[0] * ns
                cond = self.cond_candidates.repeat_interleave(ns, dim=0).repeat(
                    n_worlds, 1
                )
        elif self.config.cond_dim and self.cond is not None:
            cond = self.cond.expand(n_worlds, -1).repeat_interleave(ns, dim=0)
        if k > 1:
            obs = obs.repeat_interleave(k, dim=0)
        sd, gd = self.state_dim, self.config.goal_dim
        state, goal = obs[:, :sd], obs[:, -gd:]
        m = obs.shape[0]
        x = torch.randn(m, self.config.horizon, self.traj_dim, device=obs.device)
        last = self.last_traj
        t0 = self.warm_start_t if last is not None else 0.0
        if t0 > 0 and last is not None:
            # re-anchor: shift by the plan frame closest to the current state —
            # rolling by executed steps drifts into the absorbing tail (and
            # stalls there) whenever execution lags the plan's schedule
            na, h = self.config.n_action_steps, last.shape[1]
            cur = state[::k, :sd]  # one row per world
            w = min(2 * na, h - 1)
            d = (last[:, :w, :sd] - cur[:, None]).norm(dim=-1)
            shift = d.argmin(dim=1)  # (n_worlds,)
            idx = (torch.arange(h, device=x.device)[None] + shift[:, None]).clamp(
                max=h - 1
            )
            prev = last.gather(1, idx[..., None].expand(-1, -1, last.shape[-1]))
            x = (1 - t0) * x + t0 * prev.repeat_interleave(k, dim=0)

        mb = 4096  # micro-batch cap sized for a 24GB card at horizon 600
        x = torch.cat(
            [
                self.flow_integrate(
                    x[i : i + mb],
                    state[i : i + mb],
                    goal[i : i + mb],
                    None if cond is None else cond[i : i + mb],
                    t0=t0,
                )
                for i in range(0, m, mb)
            ]
        )
        if sel is not None and k > 1:  # best-of-N: lowest-scoring plan per world
            pick = sel(x).view(-1, k).argmin(dim=1)
            rows = torch.arange(len(pick), device=x.device)
            x = x.view(-1, k, *x.shape[1:])[rows, pick]
            if searching:  # commit to the picked bend mode for later replans
                assert cond is not None
                self.latched_cond = cond.view(-1, k, cond.shape[-1])[rows, pick]
        self.last_traj = x  # full [state, action] plan, for viz/diagnostics
        acts = x[..., sd:]  # normalized action dims
        if self.action_clip is not None:
            acts = acts.clamp(self.action_clip[0], self.action_clip[1])
        return acts

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action per call from the planned chunk, replanning when the
        queue empties (receding horizon)."""
        if len(self._action_queue) == 0:
            self.last_chunk = self.predict_action_chunk(batch)  # full horizon
            actions = self.last_chunk[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        action = self._action_queue.popleft()
        # low-pass the executed action stream to smooth replan-seam jumps; EMA in
        # normalized space == filtering physical actions (unnormalize is affine)
        a = self.config.action_lpf
        if a < 1.0:
            self.lpf_state = (
                action
                if self.lpf_state is None
                else a * action + (1 - a) * self.lpf_state
            )
            action = self.lpf_state
        return action


def alias_goal_stats(
    stats: dict[str, dict[str, Any]], goal_dim: int, goal_state_start: int = 0
) -> dict[str, dict[str, Any]]:
    """The goal (obs tail) and the goal-equivalent state block (ball pos, cube pos)
    are the same physical quantity, so normalize the goal with that block's stats.
    Then the commanded goal lands in that block's space and inpainting can clamp it
    onto the trajectory directly, no per-call renormalization."""
    stats = copy.deepcopy(stats)
    o = stats[OBS_STATE]
    gs = goal_state_start
    for k in ("mean", "std"):
        o[k][-goal_dim:] = o[k][gs : gs + goal_dim]
    return stats


def make_flow_matching_pre_post_processors(
    config: FlowMatchingConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the (preprocessor, postprocessor) pipelines.

    The preprocessor batches, moves to device, and normalizes inputs; the
    postprocessor unnormalizes the predicted action and returns it on CPU.
    """
    device = config.device
    assert device is not None
    features = {**(config.input_features or {}), **(config.output_features or {})}
    norm_stats = (
        alias_goal_stats(dataset_stats, config.goal_dim, config.goal_state_start)
        if dataset_stats is not None
        else None
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=device),
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=norm_stats,
            device=device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features or {},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
