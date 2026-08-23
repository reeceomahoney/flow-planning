import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from flow_planning.policy import FlowMatchingConfig, FlowMatchingPolicy


def make_policy(k=4):
    cfg = FlowMatchingConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (36,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (8,))},
        device="cpu",
        goal_dim=0,
        n_action_steps=5,
        cond_dim=7,
        dim_model=32,
        n_layers=1,
        num_inference_steps=2,
    )
    cfg.horizon = 12
    policy = FlowMatchingPolicy(cfg)
    policy.cond_candidates = torch.randn(k, 7)
    policy.selector_fn = lambda x: (torch.arange(x.shape[0]) % k).float()
    policy.reset()
    return policy


def test_stall_escape_blocks_the_held_candidate():
    policy = make_policy()
    obs = torch.randn(2, 36)
    policy.predict_action_chunk({OBS_STATE: obs})
    assert policy.latched_idx.tolist() == [0, 0]
    policy.predict_action_chunk({OBS_STATE: obs})
    assert policy.escapes == 2
    assert policy.latched_idx.tolist() == [1, 1]
    assert policy.blocked[:, 0].all()
    policy.predict_action_chunk({OBS_STATE: obs + 1.0})
    assert policy.escapes == 2
    assert policy.latched_idx.tolist() == [1, 1]
