import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flow_planning.envs.franka import box_pointcloud, subsample_cloud
from flow_planning.policy import FlowMatchingConfig, FlowTransformer


def test_cloud_pathway_and_null_fallback():
    cfg = FlowMatchingConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        goal_dim=2,
        horizon=8,
        dim_model=32,
        n_layers=1,
        cloud_points=16,
    )
    net = FlowTransformer(5, cfg)
    x = torch.randn(3, 8, 5)
    t = torch.rand(3)
    cloud = torch.randn(3, 16, 3)
    cloud[1] = 0.0
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.endswith("mod.weight") or name == "cloud_out.weight":
                param.normal_(std=0.1)
    v_null = net(x, t)
    v_cloud = net(x, t, None, None, cloud)
    assert torch.allclose(v_cloud[1], v_null[1], atol=1e-5)
    assert not torch.allclose(v_cloud[0], v_null[0])
    drop = torch.tensor([True, False, True])
    v_drop = net(x, t, None, drop, cloud)
    assert torch.allclose(v_drop[0], v_null[0], atol=1e-5)
    assert torch.isfinite(v_cloud).all()


def test_box_pointcloud_and_subsample():
    pts = box_pointcloud(
        [[0.0, -0.5, 0.225], [0.0, -0.5, 0.05]],
        [[0.15, 0.01, 0.125], [0.4, 0.4, 0.05]],
        0.1,
    )
    assert len(pts) > 100 and (pts[:, 2] > 0.1).all()
    sub = subsample_cloud(pts, 256, np.random.default_rng(0))
    assert sub.shape == (256, 3)
    assert (subsample_cloud(np.zeros((0, 3)), 4, np.random.default_rng(0)) == 0).all()
