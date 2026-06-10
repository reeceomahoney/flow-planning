"""Train the flow-matching transformer policy on the dataset from record.py."""

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import draccus
import newton.viewer
import numpy as np
import torch
import wandb
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE
from torch.utils.data import DataLoader

from flow_planning.franka import FrankaConfig, FrankaEnv
from flow_planning.policy import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    FlowTransformer,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config:
    env: FrankaConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka_cube"
    batch_size: int = 256
    num_iters: int = 50_000
    warmup_iters: int = 500
    ema_decay: float = 0.999
    device: str = "cuda"
    out_dir: str = "outputs"
    eval_every: int = 10_000
    eval_episodes: int = 256
    log_every: int = 100
    wandb_project: str = "flow-planning"
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


def cycle(loader):
    while True:
        yield from loader


class EMA:
    """Exponential moving average of model parameters, used at eval time."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters()]
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)

    def store(self, model):
        self.backup = [p.detach().clone() for p in model.parameters()]
        for s, p in zip(self.shadow, model.parameters()):
            p.data.copy_(s)

    def restore(self, model):
        assert self.backup is not None
        for b, p in zip(self.backup, model.parameters()):
            p.data.copy_(b)
        self.backup = None


@torch.no_grad()
def evaluate(policy, env, preprocessor, postprocessor, n_episodes, step):
    """Roll the policy out in sim"""
    policy.eval()
    t0 = time.perf_counter()
    env.rng = np.random.default_rng(0)
    n = env.cfg.world_count
    successes = 0
    total = 0

    for _ in range(math.ceil(n_episodes / n)):
        env.reset()
        policy.reset()
        for _ in range(env.episode_frames):
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            ee_action = postprocessor(action).numpy().astype(np.float32)
            env.apply_ee_action(ee_action)
        successes += int(env.success().sum())
        total += n
    policy.train()

    sr = successes / total
    wandb.log(
        {"eval/success_rate": sr, "eval/time": time.perf_counter() - t0}, step=step
    )
    print(f"step {step}  eval success rate {sr:.1%}", flush=True)


@draccus.wrap()
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=cfg.wandb_project, mode=cfg.wandb_mode, config=draccus.encode(cfg)
    )

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
    policy.model = cast(
        FlowTransformer, torch.compile(policy.model, mode="reduce-overhead")
    )

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset.meta.stats,
    )

    env = None
    if cfg.eval_every > 0:
        env = FrankaEnv(cfg.env, newton.viewer.ViewerNull())

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

    def lr_lambda(it):
        if it < cfg.warmup_iters:
            return (it + 1) / cfg.warmup_iters
        progress = (it - cfg.warmup_iters) / max(1, cfg.num_iters - cfg.warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(policy.model, cfg.ema_decay)

    policy.train()
    data_iter = cycle(loader)
    last_log_time = time.perf_counter()
    for it in range(cfg.num_iters):
        batch = preprocessor(next(data_iter))
        with amp:
            loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(policy.model)

        if it % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            metrics = {"train/loss": loss.item(), "train/lr": lr}
            if it > 0:
                now = time.perf_counter()
                metrics["perf/steps_per_sec"] = cfg.log_every / (now - last_log_time)
                last_log_time = now
            wandb.log(metrics, step=it)
        if it % 1000 == 0:
            print(f"step {it}/{cfg.num_iters}  loss {loss.item():.4f}", flush=True)

        if env is not None and it > 0 and it % cfg.eval_every == 0:
            ema.store(policy.model)
            evaluate(policy, env, preprocessor, postprocessor, cfg.eval_episodes, it)
            ema.restore(policy.model)
            last_log_time = time.perf_counter()

    ema.store(policy.model)
    if env is not None:
        evaluate(
            policy, env, preprocessor, postprocessor, cfg.eval_episodes, cfg.num_iters
        )

    out_dir = Path(cfg.out_dir)
    policy.model = cast(FlowTransformer, policy.model._orig_mod)
    policy.save_pretrained(out_dir)
    print(f"Saved policy to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
