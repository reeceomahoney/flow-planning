"""A small flow-matching transformer policy in the lerobot policy style.

Outer normalization is handled by the processor pipeline (see
`make_flow_matching_pre_post_processors`), so `forward` and `predict_action_chunk`
operate on already-normalized tensors. `select_action` follows the standard
lerobot API: it plans a [state, action] chunk and executes the action dims one
step per call, replanning every `replan_every` steps (receding horizon).
"""

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


def even_spacing(acts: Tensor, uniform: float) -> Tensor:
    """Reparametrize the planned action path toward equal arc-length spacing."""
    if uniform <= 0:
        return acts
    b, h, c = acts.shape
    seg = (acts[:, 1:] - acts[:, :-1]).norm(dim=-1)  # (B, H-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], dim=1)  # (B, H)
    total = cum[:, -1:]
    u = torch.linspace(0, 1, h, device=acts.device, dtype=acts.dtype)
    tgt = (1 - uniform) * cum + uniform * u[None, :] * total  # (B, H) arc lengths
    hi = torch.searchsorted(cum, tgt).clamp(1, h - 1)
    lo = hi - 1
    c_lo, c_hi = torch.gather(cum, 1, lo), torch.gather(cum, 1, hi)
    w = ((tgt - c_lo) / (c_hi - c_lo).clamp_min(1e-9)).unsqueeze(-1)
    g_lo = lo.unsqueeze(-1).expand(-1, -1, c)
    g_hi = hi.unsqueeze(-1).expand(-1, -1, c)
    p_lo, p_hi = torch.gather(acts, 1, g_lo), torch.gather(acts, 1, g_hi)
    return p_lo + w * (p_hi - p_lo)


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Config for the flow-matching transformer policy."""

    # dimensions
    horizon: int = 50
    n_action_steps: int = 40  # replan interval; ~40 best with uniform spacing
    goal_dim: int = 2  # trailing obs dims holding the goal

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    attn_window: int = 5  # local attention half-width over step tokens; 0 = global

    # flow matching
    num_inference_steps: int = 10
    action_lpf: float = 0.3  # EMA alpha on executed actions; <1 smooths seam jumps
    spacing_uniform: float = 1.0  # arc-length resample of the plan; 0=raw, 1=uniform

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


class FlowTransformer(nn.Module):
    """Velocity field v(x_t, t | state, goal) over a chunk of [state, action] steps.

    Each step token carries the full [pos, vel, action] vector and reads out its
    per-step velocity. The current state is prepended as a token and the goal
    appended as a token, so they act as the trajectory's boundary conditions.
    `attn_window` restricts each token to a local neighbourhood (so the prior
    stays closed under stitching); the state token then anchors the early steps
    and the goal token the late steps.
    """

    attn_mask: Tensor | None

    def __init__(self, traj_dim: int, state_dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        d = cfg.dim_model
        self.horizon = cfg.horizon
        self.traj_in = nn.Linear(traj_dim, d)
        self.state_in = nn.Linear(state_dim, d)  # leading state token
        self.goal_in = nn.Linear(cfg.goal_dim, d)  # trailing goal token
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        n_tokens = self.horizon + 2  # state token + steps + goal token
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 4 * d, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, cfg.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d, traj_dim)

        # local attention: step tokens see only +/-attn_window neighbours, but
        # state (col 0) and goal (col -1) are global registers
        if cfg.attn_window > 0:
            i = torch.arange(n_tokens)
            band = (i[:, None] - i[None, :]).abs() > cfg.attn_window
            band[:, 0] = False
            band[:, -1] = False
            self.register_buffer("attn_mask", band, persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x_t: Tensor, t: Tensor, state: Tensor, goal: Tensor) -> Tensor:
        # x_t: (B, H, traj_dim), t: (B,), state/goal: (B, state_dim)/(B, goal_dim).
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        a = self.traj_in(x_t) + temb[:, None]
        s = (self.state_in(state) + temb)[:, None]
        g = (self.goal_in(goal) + temb)[:, None]
        tokens = torch.cat([s, a, g], dim=1) + self.pos_emb  # state | steps | goal
        h = self.transformer(tokens, mask=self.attn_mask)
        return self.head(h[:, 1:-1])


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

        self.model = FlowTransformer(self.traj_dim, self.state_dim, config)
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
        # action=goal target), so training on it teaches the absorbing goal.
        # ponytail: assumes lerobot pads delta frames by clamping to the episode end.
        obs = batch[OBS_STATE]  # (B, H, obs_dim); trailing dims = goal
        state = obs[..., : self.state_dim]
        x_1 = torch.cat([state, batch[ACTION]], dim=-1)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        goal = obs[:, 0, -self.config.goal_dim :]  # commanded goal, already in obs
        v = self.model(x_t, t, state[:, 0], goal)  # condition on state + goal
        loss = F.mse_loss(v, x_1 - x_0)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Flow-ODE integrate a full [state, action] trajectory conditioned on the
        current state (leading token) and the goal (trailing token), returning the
        (B, horizon, act_dim) action chunk. The goal is read from the trailing obs
        dims. Any `self.guidance_fn` cost gradient is subtracted each step.
        """
        self.eval()
        obs = batch[OBS_STATE]  # (B, obs_dim) current obs = [state, goal]
        state = obs[:, : self.state_dim]
        n = obs.shape[0]
        goal = obs[:, -self.config.goal_dim :]

        dt = 1.0 / self.config.num_inference_steps
        x = torch.randn(n, self.config.horizon, self.traj_dim, device=obs.device)
        for i in range(self.config.num_inference_steps):
            t = torch.full((n,), i * dt, device=x.device)
            v = self.model(x, t, state, goal)
            if self.guidance_fn is not None:
                v = v - self.guidance_fn(x + (1 - i * dt) * v, obs)
            x = x + dt * v
        acts = x[..., self.state_dim :]  # normalized action dims
        acts = even_spacing(acts, self.config.spacing_uniform)
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

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=device),
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
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
