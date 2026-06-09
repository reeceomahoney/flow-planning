"""A small flow-matching transformer policy in the lerobot policy style.

Normalization is handled by the processor pipeline (see
`make_flow_matching_pre_post_processors`), not inside the policy, so `forward`
and `select_action` operate on already-normalized tensors.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
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

    # architecture
    dim_model: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # flow matching
    num_inference_steps: int = 10

    # training
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-2

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
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


class FlowTransformer(nn.Module):
    """Velocity field v(x_t, t | obs) over normalized actions.

    The noisy-action token and observation token, each fused with a time
    embedding, attend to each other; the action token reads out the velocity.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        d = cfg.dim_model
        self.act_in = nn.Linear(act_dim, d)
        self.obs_in = nn.Linear(obs_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos_emb = nn.Parameter(torch.randn(2, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, 4 * d, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, cfg.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d, act_dim)

    def forward(self, x_t: Tensor, t: Tensor, obs: Tensor) -> Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.pos_emb.shape[-1]))
        a = self.act_in(x_t) + temb
        o = self.obs_in(obs) + temb
        tokens = torch.stack([a, o], dim=1) + self.pos_emb
        h = self.transformer(tokens)
        return self.head(h[:, 0])


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

    def get_optim_params(self) -> dict:
        return self.parameters()  # ty: ignore[invalid-return-type]

    def reset(self):
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        obs = batch[OBS_STATE]
        x_1 = batch[ACTION]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
        v = self.model(x_t, t, obs)
        loss = F.mse_loss(v, x_1 - x_0)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.select_action(batch).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        obs = batch[OBS_STATE]
        x = torch.randn(obs.shape[0], self.act_dim, device=obs.device)
        dt = 1.0 / self.config.num_inference_steps
        for i in range(self.config.num_inference_steps):
            t = torch.full((obs.shape[0],), i * dt, device=obs.device)
            x = x + dt * self.model(x, t, obs)
        return x


def make_flow_matching_pre_post_processors(
    config: FlowMatchingConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
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
            norm_map=config.normalization_mapping,  # ty: ignore[invalid-argument-type]
            stats=dataset_stats,
            device=device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features or {},
            norm_map=config.normalization_mapping,  # ty: ignore[invalid-argument-type]
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
