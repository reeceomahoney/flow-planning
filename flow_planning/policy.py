"""A small flow-matching transformer policy in the lerobot policy style.

Outer normalization is handled by the processor pipeline (see
`make_flow_matching_pre_post_processors`), so `forward` and `predict_action_chunk`
operate on already-normalized tensors. `select_action` follows the standard
lerobot API: it plans a chunk and pure-pursuit-tracks it one action per call,
using stat buffers only to bridge the obs and action normalization frames.
"""

import math
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

from flow_planning.trajectory import PathTracker, resample_arc_length


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Config for the flow-matching transformer policy."""

    n_obs_steps: int = 1

    horizon: int = 32  # support points per plan (plan resolution N)
    traj_steps: int = 0  # data window in frames; train.py auto-sets to max ep length
    position_dim: int = 0  # leading action dims for arc length/inpainting; 0 = all
    obs_pos_start: int = 0  # index of the current position within the observation

    # rollout (select_action): replan, inpaint endpoints, pure-pursuit track
    lookahead: float = 0.15  # pursuit lookahead in physical units
    replan_every: int = 10  # frames between replans
    inpaint: bool = True  # pin plan endpoints to current state and goal

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # flow matching
    num_inference_steps: int = 10

    # classifier-free goal conditioning
    goal_dim: int = 2  # trailing obs dims holding the goal, 0 disables
    goal_dropout: float = 0.0  # P(mask the goal token) during training
    goal_guidance_weight: float = 1.0  # w=1: conditional, w=0: unconditional

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
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        # load the full window to the goal, later resampled to `horizon` points
        return list(range(self.traj_steps))

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
    """Velocity field v(x_t, t | obs) over a chunk of normalized actions.

    The `horizon` noisy-action tokens, the state token, and the goal token,
    each fused with a time embedding, attend to each other; the action tokens
    read out the per-step velocity. `drop_goal` masks the goal token out of
    attention entirely, so the unconditional model never sees a goal value.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        d = cfg.dim_model
        self.horizon = cfg.horizon
        self.goal_dim = cfg.goal_dim
        self.act_in = nn.Linear(act_dim, d)
        self.obs_in = nn.Linear(obs_dim - self.goal_dim, d)
        n_tokens = self.horizon + 1
        if self.goal_dim > 0:
            self.goal_in = nn.Linear(self.goal_dim, d)
            n_tokens += 1
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 4 * d, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, cfg.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d, act_dim)

    def forward(
        self, x_t: Tensor, t: Tensor, obs: Tensor, drop_goal: Tensor | None = None
    ) -> Tensor:
        # x_t: (B, H, act_dim), t: (B,), obs: (B, obs_dim), drop_goal: (B,) bool
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        a = self.act_in(x_t) + temb[:, None]
        if self.goal_dim > 0:
            state, goal = obs[:, : -self.goal_dim], obs[:, -self.goal_dim :]
            o = self.obs_in(state)[:, None] + temb[:, None]
            g = self.goal_in(goal)[:, None] + temb[:, None]
            tokens = torch.cat([a, o, g], dim=1) + self.pos_emb
        else:
            o = self.obs_in(obs)[:, None] + temb[:, None]
            tokens = torch.cat([a, o], dim=1) + self.pos_emb
        mask = None
        if drop_goal is not None:
            mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=obs.device)
            mask[:, -1] = drop_goal
        h = self.transformer(tokens, src_key_padding_mask=mask)
        return self.head(h[:, : self.horizon])


class FlowMatchingPolicy(PreTrainedPolicy):
    config_class = FlowMatchingConfig
    name = "flow_matching"
    config: FlowMatchingConfig
    obs_mean: Tensor
    obs_std: Tensor
    action_mean: Tensor
    action_std: Tensor

    def __init__(self, config: FlowMatchingConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        assert config.robot_state_feature is not None
        assert config.action_feature is not None
        obs_dim = config.robot_state_feature.shape[0]
        act_dim = config.action_feature.shape[0]
        self.act_dim = act_dim
        self.pos_dim = config.position_dim or act_dim

        self.model = FlowTransformer(obs_dim, act_dim, config)

        # stats bridge the obs and action normalization frames inside
        # select_action; not persisted (reloaded from the dataset at eval)
        self.register_buffer("obs_mean", torch.zeros(obs_dim), persistent=False)
        self.register_buffer("obs_std", torch.ones(obs_dim), persistent=False)
        self.register_buffer("action_mean", torch.zeros(act_dim), persistent=False)
        self.register_buffer("action_std", torch.ones(act_dim), persistent=False)
        if dataset_stats is not None:
            self.load_stats(dataset_stats)

        self.guidance_fn = None  # optional obstacle cost, set before rollout
        self.tracker = PathTracker(self.pos_dim, config.lookahead)
        self.reset()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Policy parameters: {n_params / 1e6:.2f}M")

    def get_optim_params(self) -> Iterator[nn.Parameter]:  # ty: ignore[invalid-method-override]
        return self.parameters()

    def load_stats(self, stats: dict[str, Any]) -> None:
        def to_buf(v):
            return torch.as_tensor(v, dtype=torch.float32, device=self.obs_mean.device)

        self.obs_mean.copy_(to_buf(stats[OBS_STATE]["mean"]))
        self.obs_std.copy_(to_buf(stats[OBS_STATE]["std"]))
        self.action_mean.copy_(to_buf(stats[ACTION]["mean"]))
        self.action_std.copy_(to_buf(stats[ACTION]["std"]))

    def reset(self):
        self.step = 0
        self.tracker.plan = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        obs = batch[OBS_STATE]
        x_1 = resample_arc_length(
            batch[ACTION],
            batch.get(f"{ACTION}_is_pad"),
            self.config.horizon,
            self.config.position_dim,
        )
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        drop = None
        if self.config.goal_dim > 0 and self.config.goal_dropout > 0:
            drop = (
                torch.rand(obs.shape[0], device=obs.device) < self.config.goal_dropout
            )
        v = self.model(x_t, t, obs, drop)
        loss = F.mse_loss(v, x_1 - x_0)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Flow-ODE integrate a full (B, horizon, act_dim) action chunk.

        `guidance_fn`, if given, maps the predicted clean chunk to a cost
        gradient that is subtracted from the velocity at each step. With
        `goal_guidance_weight` w != 1 the velocity is the classifier-free mix
        v_uncond + w * (v_cond - v_uncond).
        """
        guidance_fn = kwargs.get("guidance_fn")
        inpaint = kwargs.get("inpaint")  # {"start": (n,p), "goal": (n,p)} normalized
        self.eval()
        obs = batch[OBS_STATE]
        n = obs.shape[0]
        w = self.config.goal_guidance_weight
        cfg_mix = w != 1.0 and self.config.goal_dim > 0
        if cfg_mix:
            obs = obs.repeat(2, 1)
            drop = torch.zeros(2 * n, dtype=torch.bool, device=obs.device)
            drop[n:] = True
        x = torch.randn(n, self.config.horizon, self.act_dim, device=obs.device)

        # pin the ends along the noise->data interpolant: exact start/goal at t=1
        pin = None
        if inpaint is not None:
            p = inpaint["start"].shape[-1]
            z0, z1 = x[:, 0, :p].clone(), x[:, -1, :p].clone()

            def pin(x: Tensor, t: float) -> Tensor:
                x[:, 0, :p] = (1 - t) * z0 + t * inpaint["start"]
                x[:, -1, :p] = (1 - t) * z1 + t * inpaint["goal"]
                return x

            x = pin(x, 0.0)

        dt = 1.0 / self.config.num_inference_steps
        for i in range(self.config.num_inference_steps):
            t = torch.full((obs.shape[0],), i * dt, device=obs.device)
            if cfg_mix:
                v_c, v_u = self.model(x.repeat(2, 1, 1), t, obs, drop).chunk(2)
                v = v_u + w * (v_c - v_u)
            else:
                v = self.model(x, t, obs)
            if guidance_fn is not None:
                v = v - guidance_fn(x + (1 - i * dt) * v)
            x = x + dt * v
            if pin is not None:
                x = pin(x, (i + 1) * dt)
        return x

    def physical_slice(self, obs: Tensor, lo: int, n: int) -> Tensor:
        """Unnormalize obs[:, lo:lo+n] back to physical units."""
        sl = slice(lo, lo + n)
        return obs[:, sl] * self.obs_std[sl] + self.obs_mean[sl]

    @torch.no_grad()
    def replan(self, obs: Tensor) -> None:
        inpaint = None
        if self.config.inpaint:
            p, g = self.pos_dim, self.config.goal_dim
            start = self.physical_slice(obs, self.config.obs_pos_start, p)
            goal = self.physical_slice(obs, obs.shape[-1] - g, g)
            inpaint = {
                "start": (start - self.action_mean[:p]) / self.action_std[:p],
                "goal": (goal - self.action_mean[:g]) / self.action_std[:g],
            }
        plan = self.predict_action_chunk(
            {OBS_STATE: obs}, inpaint=inpaint, guidance_fn=self.guidance_fn
        )
        self.tracker.set_plan(plan * self.action_std + self.action_mean)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Plan + pure-pursuit one action per call, replanning periodically."""
        obs = batch[OBS_STATE]
        if self.tracker.plan is None or self.step % self.config.replan_every == 0:
            self.replan(obs)
        self.step += 1
        self.tracker.lookahead = self.config.lookahead
        pos = self.physical_slice(obs, self.config.obs_pos_start, self.pos_dim)
        target = self.tracker.target(pos)  # physical action
        return (target - self.action_mean) / self.action_std


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
