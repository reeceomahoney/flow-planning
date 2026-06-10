"""Train the flow-matching transformer policy on the dataset from record.py."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_planning.policy import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config:
    repo_id: str = "reece-omahoney/franka_cube"
    batch_size: int = 256
    num_iters: int = 20_000
    device: str = "cuda"
    out_dir: str = "outputs"


def cycle(loader):
    while True:
        yield from loader


@draccus.wrap()
def main(cfg: Config):
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
    policy = FlowMatchingPolicy(policy_cfg).to(device)
    policy.model = torch.compile(policy.model, mode="reduce-overhead")  # ty: ignore[invalid-assignment]

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    preprocessor, _ = make_flow_matching_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset.meta.stats,  # ty: ignore[invalid-argument-type]
    )

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
    pbar = tqdm(range(cfg.num_iters), desc="Training")
    for it in pbar:
        batch = preprocessor(next(data_iter))
        with amp:
            loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    out_dir = Path(cfg.out_dir)
    policy.model = policy.model._orig_mod  # ty: ignore[invalid-assignment]
    policy.save_pretrained(out_dir)
    print(f"Saved policy to {out_dir}")


if __name__ == "__main__":
    main()
