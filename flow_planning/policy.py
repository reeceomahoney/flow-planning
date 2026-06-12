"""A small flow-matching transformer policy in the lerobot policy style.

Normalization is handled by the processor pipeline (see
`make_flow_matching_pre_post_processors`), not inside the policy, so `forward`
and `select_action` operate on already-normalized tensors.
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


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Config for the flow-matching transformer policy."""

    n_obs_steps: int = 1

    # action chunking
    horizon: int = 50
    n_action_steps: int = 20

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # flow matching
    num_inference_steps: int = 10

    # classifier-free goal conditioning
    goal_dim: int = 2  # trailing obs dims holding the goal, 0 disables
    goal_dropout: float = 0.2  # P(replace goal with null token) during training
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
    """Velocity field v(x_t, t | obs) over a chunk of normalized actions.

    The `horizon` noisy-action tokens and the observation token, each fused with
    a time embedding, attend to each other; the action tokens read out the
    per-step velocity.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        d = cfg.dim_model
        self.horizon = cfg.horizon
        self.act_in = nn.Linear(act_dim, d)
        self.obs_in = nn.Linear(obs_dim, d)
        if cfg.goal_dim > 0:
            self.null_goal = nn.Parameter(torch.zeros(cfg.goal_dim))
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos_emb = nn.Parameter(torch.randn(self.horizon + 1, d) * 0.02)

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
        if drop_goal is not None:
            gd = self.null_goal.shape[0]
            null = self.null_goal.expand(obs.shape[0], -1)
            goal = torch.where(drop_goal[:, None], null, obs[:, -gd:])
            obs = torch.cat([obs[:, :-gd], goal], dim=-1)
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        a = self.act_in(x_t) + temb[:, None]
        o = self.obs_in(obs)[:, None] + temb[:, None]
        tokens = torch.cat([a, o], dim=1) + self.pos_emb
        h = self.transformer(tokens)
        return self.head(h[:, : self.horizon])


class FlowMatchingPolicy(PreTrainedPolicy):
    config_class = FlowMatchingConfig
    name = "flow_matching"
    config: FlowMatchingConfig

    def __init__(self, config: FlowMatchingConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        assert config.robot_state_feature is not None
        assert config.action_feature is not None
        obs_dim = config.robot_state_feature.shape[0]
        act_dim = config.action_feature.shape[0]
        self.act_dim = act_dim

        self.model = FlowTransformer(obs_dim, act_dim, config)
        self.reset()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Policy parameters: {n_params / 1e6:.2f}M")

    def get_optim_params(self) -> Iterator[nn.Parameter]:  # ty: ignore[invalid-method-override]
        return self.parameters()

    def reset(self):
        self.chunk: Tensor | None = None
        self.chunk_step = 0

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        obs = batch[OBS_STATE]
        x_1 = batch[ACTION]  # (B, H, act_dim)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        drop = None
        if self.config.goal_dim > 0 and self.config.goal_dropout > 0:
            drop = (
                torch.rand(obs.shape[0], device=obs.device) < self.config.goal_dropout
            )
        v = self.model(x_t, t, obs, drop)
        loss = F.mse_loss(v, x_1 - x_0, reduction="none").mean(-1)  # (B, H)

        pad = batch.get(f"{ACTION}_is_pad")
        if pad is not None:
            mask = (~pad).to(loss.dtype)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
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
        return x

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action per call, replanning every n_action_steps frames."""
        if self.chunk is None or self.chunk_step >= self.config.n_action_steps:
            self.chunk = self.predict_action_chunk(batch, **kwargs)
            self.chunk_step = 0
        action = self.chunk[:, self.chunk_step]
        self.chunk_step += 1
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
