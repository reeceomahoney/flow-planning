"""Train the flow-matching transformer policy on the dataset from record.py."""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE
from torch.utils.data import DataLoader

from flow_planning.franka import FrankaConfig, FrankaEnv
from flow_planning.policy import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config(FrankaConfig):
    repo_id: str = "reece-omahoney/franka_cube"
    batch_size: int = 256
    num_iters: int = 20_000
    device: str = "cuda"
    out_dir: str = "outputs"
    eval_every: int = 5_000  # 0 disables sim eval
    eval_episodes: int = 64


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(policy, env, preprocessor, postprocessor, n_episodes):
    """Roll the policy out in sim; return the cube-placement success rate."""
    policy.eval()
    n = env.cfg.world_count
    successes = 0
    total = 0
    for _ in range(math.ceil(n_episodes / n)):
        env.reset()
        policy.reset()
        for _ in range(env.episode_frames):
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            joint_targets = postprocessor(action).numpy().astype(np.float32)
            env.apply_action(joint_targets)
        successes += int(env.success().sum())
        total += n
    policy.train()
    return successes / total


@draccus.wrap()
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(cfg.repo_id)
    features = dataset_to_policy_features(dataset.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    input_features = {k: v for k, v in features.items() if k not in output_features}

    policy_cfg = FlowMatchingConfig(
        input_features=input_features, output_features=output_features, device=device
    )

    delta_timestamps = {
        "action": [i / dataset.fps for i in policy_cfg.action_delta_indices]
    }
    dataset = LeRobotDataset(cfg.repo_id, delta_timestamps=delta_timestamps)

    policy = FlowMatchingPolicy(policy_cfg).to(device)
    policy.model = torch.compile(policy.model, mode="reduce-overhead")  # ty: ignore[invalid-assignment]

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset.meta.stats,  # ty: ignore[invalid-argument-type]
    )

    env = None
    if cfg.eval_every > 0:
        env = FrankaEnv(cfg, newton.viewer.ViewerNull())

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy_cfg.optimizer_lr,
        weight_decay=policy_cfg.optimizer_weight_decay,
        fused=True,
    )

    policy.train()
    data_iter = cycle(loader)
    for it in range(cfg.num_iters):
        batch = preprocessor(next(data_iter))
        with amp:
            loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"step {it}/{cfg.num_iters}  loss {loss.item():.4f}")

        if env is not None and it > 0 and it % cfg.eval_every == 0:
            sr = evaluate(policy, env, preprocessor, postprocessor, cfg.eval_episodes)
            print(f"step {it}  eval success rate {sr:.1%}")

    if env is not None:
        sr = evaluate(policy, env, preprocessor, postprocessor, cfg.eval_episodes)
        print(f"final eval success rate {sr:.1%}")

    out_dir = Path(cfg.out_dir)
    policy.model = policy.model._orig_mod  # ty: ignore[invalid-assignment]
    policy.save_pretrained(out_dir)
    print(f"Saved policy to {out_dir}")


if __name__ == "__main__":
    main()
