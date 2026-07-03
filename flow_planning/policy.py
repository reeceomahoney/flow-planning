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

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # flow matching
    num_inference_steps: int = 10
    action_lpf: float = 0.3  # EMA alpha on executed actions; <1 smooths seam jumps

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

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

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
    """Pre-norm transformer encoder layer over global SDPA."""

    def __init__(self, d: int, n_heads: int, dim_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.GELU(), nn.Linear(dim_ff, d))

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
        o = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        x = x + self.proj(o.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


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
        self.layers = nn.ModuleList(
            EncoderLayer(d, cfg.n_heads, 4 * d) for _ in range(cfg.n_layers)
        )
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, traj_dim)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # x_t: (B, H, traj_dim), t: (B,).
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        h = self.traj_in(x_t) + temb[:, None] + self.pos_emb
        for layer in self.layers:
            h = layer(h)
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
        self.guidance_fn = None  # optional obstacle cost, set before rollout
        self.action_clip = None  # optional (low, high) normalized action bounds
        self.reset()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Policy parameters: {n_params / 1e6:.2f}M")

    def get_optim_params(self) -> Iterator[nn.Parameter]:  # ty: ignore[invalid-method-override]
        return self.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.last_chunk = None
        self.lpf_state = None  # EMA state for the executed-action low-pass
        if self.guidance_fn is not None:
            self.guidance_fn.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # full-trajectory window: [state, action] per step, both normalized. The
        # tail past the episode end clamps to the goal frame (pos=goal, vel~0,
        # action=goal target). Unconditional flow: x_t already carries the real
        # start/goal in its first/last frames, so the net learns to use them; at
        # inference those frames are clamped to the commanded values (inpainting).
        # ponytail: assumes lerobot pads delta frames by clamping to the episode end.
        obs = batch[OBS_STATE]  # (B, H, obs_dim); trailing dims = goal
        sd, gd = self.state_dim, self.config.goal_dim
        x_1 = torch.cat([obs[..., :sd], batch[ACTION]], dim=-1)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        # boundary frames are clean conditioning context (start state, goal pos),
        # not noised; exclude them from the loss since their velocity is undefined.
        x_t[:, 0, :sd] = x_1[:, 0, :sd]
        x_t[:, -1, :gd] = x_1[:, -1, :gd]
        v = self.model(x_t, t)
        err = (v - (x_1 - x_0)) ** 2
        mask = torch.ones_like(err)
        mask[:, 0, :sd] = 0.0
        mask[:, -1, :gd] = 0.0
        loss = (err * mask).sum() / mask.sum()
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Flow-ODE integrate a [state, action] trajectory, conditioning by
        inpainting: each step clamp frame 0's state to the current state and frame
        -1's goal dims to the goal (already in position space, see alias_goal_stats).
        `guidance_fn` (if set) steers around the obstacle. Returns the
        (B, horizon, act_dim) action chunk."""
        self.eval()
        obs = batch[OBS_STATE]
        sd, gd = self.state_dim, self.config.goal_dim
        state, goal = obs[:, :sd], obs[:, -gd:]
        m = obs.shape[0]
        dt = 1.0 / self.config.num_inference_steps
        x = torch.randn(m, self.config.horizon, self.traj_dim, device=obs.device)
        for i in range(self.config.num_inference_steps):
            x[:, 0, :sd] = state
            x[:, -1, :gd] = goal
            t = torch.full((m,), i * dt, device=x.device)
            v = self.model(x, t)
            if self.guidance_fn is not None:
                v = v - self.guidance_fn(x, v, t, obs)
            x = x + dt * v
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
    stats: dict[str, dict[str, Any]], goal_dim: int
) -> dict[str, dict[str, Any]]:
    """The goal (obs tail) and the position (obs head) are the same physical
    quantity, so normalize the goal with the position's stats. Then the commanded
    goal lands in position space and inpainting can clamp it onto the trajectory
    directly, no per-call renormalization."""
    stats = copy.deepcopy(stats)
    o = stats[OBS_STATE]
    for k in ("mean", "std"):
        o[k][-goal_dim:] = o[k][:goal_dim]
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
        alias_goal_stats(dataset_stats, config.goal_dim)
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
