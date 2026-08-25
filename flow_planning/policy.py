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

    # bend-parameter conditioning (0 disables); dropout trains the null token
    cond_dim: int = 0
    cond_dropout: float = 0.15
    zero_cond: bool = False

    # training
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-2

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
    )

    pretrained_revision: str | None = None

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
        self.selector_fn = None  # optional plan scorer for candidate selection
        self.cond = None  # commanded bend params (1, cond_dim), None = null token
        self.cond_candidates = None  # (K, cond_dim): search + latch via selector_fn
        self.latched_idx = None
        self.n_samples = 1  # >1 with no candidates: best-of-N over noise alone
        self.guidance_scale = 1.0  # CFG on the bend/posture latent; 1 = off
        self.reset()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Policy parameters: {n_params / 1e6:.2f}M")

    def get_optim_params(self) -> Iterator[nn.Parameter]:  # ty: ignore[invalid-method-override]
        return self.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.last_chunk = None
        self.latched_cond = None  # per-world bend picked at the first replan
        self.latched_idx = None

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
    ) -> Tensor:
        """ODE-integrate x from 0 to 1, re-clamping the boundary frames each step."""
        sd, gd = self.state_dim, self.config.goal_dim
        gs = self.config.goal_state_start
        dt = 1.0 / self.config.num_inference_steps
        for i in range(self.config.num_inference_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x.clone()
            x[:, 0, :sd] = state
            x[:, -1, gs : gs + gd] = goal
            if cond is not None and self.guidance_scale != 1.0:
                vc = self.model(x, t, cond)
                vu = self.model(x, t, None)
                v = vu + self.guidance_scale * (vc - vu)
            else:
                v = self.model(x, t, cond)
            x = x + dt * v
        return x

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
        # bend conditioning: fixed command, latched pick, or a candidate search
        # where the selector keeps the plan with the lowest collision score
        cond, k, searching = None, 1, False
        if self.config.cond_dim and self.cond_candidates is not None:
            searching = True
            k = self.cond_candidates.shape[0]
            cond = self.cond_candidates.repeat(n_worlds, 1)
        else:
            k = max(self.n_samples, 1)
            if self.config.cond_dim and self.cond is not None:
                cond = self.cond.expand(n_worlds * k, -1)
            elif self.config.cond_dim and self.config.zero_cond:
                cond = obs.new_zeros(n_worlds * k, self.config.cond_dim)
        if k > 1:
            obs = obs.repeat_interleave(k, dim=0)
        sd = self.state_dim
        state, goal = obs[:, :sd], obs[:, sd:]
        m = obs.shape[0]
        x = torch.randn(m, self.config.horizon, self.traj_dim, device=obs.device)
        # micro-batch cap for a 24GB card: memory scales with horizon, so hold
        # plans x horizon constant instead of hard-coding a count (retiming took
        # the horizon 599 -> 832 and OOM'd a fixed 4096)
        mb = max(256, int(4096 * 600 / self.config.horizon))
        x = torch.cat(
            [
                self.flow_integrate(
                    x[i : i + mb],
                    state[i : i + mb],
                    goal[i : i + mb],
                    None if cond is None else cond[i : i + mb],
                )
                for i in range(0, m, mb)
            ]
        )
        if sel is not None and k > 1:  # lowest-scoring candidate per world
            score = sel(x).view(n_worlds, k)
            if searching and self.latched_idx is not None:  # keep the held pick
                keep = F.one_hot(self.latched_idx, k).bool()
                score = score.masked_fill(~keep, torch.inf)
            pick = score.argmin(dim=1)
            rows = torch.arange(len(pick), device=x.device)
            x = x.view(-1, k, *x.shape[1:])[rows, pick]
            if searching:
                assert cond is not None
                self.latched_idx = pick
                self.latched_cond = cond.view(-1, k, cond.shape[-1])[rows, pick]
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
        return self._action_queue.popleft()


def alias_goal_stats(
    stats: dict[str, dict[str, Any]], goal_dim: int, goal_state_start: int = 0
) -> dict[str, dict[str, Any]]:
    """The goal (obs tail) and the goal-equivalent state block (ball pos, cube pos)
    are the same physical quantity, so normalize the goal with that block's stats.
    Then the commanded goal lands in that block's space and inpainting can clamp it
    onto the trajectory directly, no per-call renormalization."""
    if goal_dim == 0:
        return stats
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
