"""Train the flow-matching transformer policy on the dataset from record.py."""

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal, cast

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import DataLoader

import wandb
from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.guidance import Guidance
from flow_planning.policy import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    FlowTransformer,
    make_flow_matching_pre_post_processors,
)
from flow_planning.utils import make_run_dir


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/particle"
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
    """Roll the policy out in sim via the standard select_action loop."""
    policy.eval()
    t0 = time.perf_counter()
    env.rng = np.random.default_rng(0)
    n = env.cfg.world_count
    successes = 0
    stage_sum = 0  # avg furthest pick-place stage reached, as a dense reward
    total = 0

    has_stage = hasattr(env, "stage")  # franka pick-place ladder; particle has none
    for _ in range(math.ceil(n_episodes / n)):
        env.reset()
        policy.reset()
        max_stage = np.zeros(n, int)
        succ = np.zeros(n, dtype=bool)
        for _ in range(int(1.5 * env.episode_frames)):
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            env.apply_action(postprocessor(action).numpy().astype(np.float32))
            succ |= env.success()  # accrue: reaching the goal sticks past drift
            if has_stage:
                max_stage = np.maximum(max_stage, env.stage())
        successes += int(succ.sum())
        stage_sum += int(max_stage.sum())
        total += n
    policy.train()

    sr = successes / total
    reward = stage_sum / total
    wandb.log(
        {
            "eval/success_rate": sr,
            "eval/reward": reward,
            "eval/time": time.perf_counter() - t0,
        },
        step=step,
    )
    print(f"step {step}  eval success rate {sr:.1%}  reward {reward:.2f}", flush=True)


def save_policy(policy, out_dir):
    compiled = policy.model
    policy.model = cast(FlowTransformer, compiled._orig_mod)
    policy.save_pretrained(out_dir)
    policy.model = compiled
    print(f"Saved policy to {out_dir}", flush=True)


@draccus.wrap()
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = make_run_dir(cfg.out_dir, cfg.env.type)
    print(f"Output directory: {out_dir}")

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
        input_features=input_features,
        output_features=output_features,
        device=device,
        goal_dim=cfg.env.goal_dim,
    )
    # plan the full trajectory: horizon spans the longest episode; shorter
    # episodes pad their tail with the goal frame (absorbing).
    policy_cfg.horizon = max(dataset.meta.episodes["length"])

    delta_timestamps = {
        "action": [i / dataset.fps for i in policy_cfg.action_delta_indices],
        "observation.state": [
            i / dataset.fps for i in policy_cfg.observation_delta_indices
        ],
    }
    dataset = LeRobotDataset(cfg.repo_id, delta_timestamps=delta_timestamps)
    stats = dataset.meta.stats
    assert stats is not None

    policy = FlowMatchingPolicy(policy_cfg, dataset_stats=stats).to(device)
    policy.model = cast(FlowTransformer, torch.compile(policy.model))

    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy_cfg,
        dataset_stats=dataset.meta.stats,
    )

    env = None
    if cfg.eval_every > 0:
        env = make_env(cfg.env, newton.viewer.ViewerNull())
        # inference-time guidance for eval rollouts (unused by training forward)
        policy.guidance_fn = Guidance(
            env,
            policy.state_dim,
            stats[ACTION],
            device,
            obs_stats=stats[OBS_STATE],
        )

    # dataset is tiny, just materialize it in VRAM for fast training
    mat_loader = DataLoader(dataset, batch_size=512, num_workers=8, shuffle=False)
    obs_all, act_all = [], []
    for b in mat_loader:
        obs_all.append(b[OBS_STATE])
        act_all.append(b[ACTION])
    obs_all = torch.cat(obs_all).to(device)
    act_all = torch.cat(act_all).to(device)
    n_samples = obs_all.shape[0]
    mb = (obs_all.nbytes + act_all.nbytes) / 1e6
    print(f"Materialized {n_samples} windows ({mb:.0f} MB) in VRAM", flush=True)

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
    last_log_time = time.perf_counter()
    for it in range(cfg.num_iters):
        idx = torch.randint(0, n_samples, (cfg.batch_size,), device=device)
        batch = preprocessor({OBS_STATE: obs_all[idx], ACTION: act_all[idx]})
        with amp:
            loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
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
            grads = {p: g for p in policy.parameters() if (g := p.grad) is not None}
            pnorm = torch.norm(torch.stack([p.norm() for p in grads])).item()
            gnorm = torch.norm(torch.stack([g.norm() for g in grads.values()])).item()
            print(
                f"step {it}/{cfg.num_iters}  loss {loss.item():.4f}"
                f"  pnorm {pnorm:.2f}  gnorm {gnorm:.3f}",
                flush=True,
            )

        if env is not None and it > 0 and it % cfg.eval_every == 0:
            ema.store(policy.model)
            evaluate(policy, env, preprocessor, postprocessor, cfg.eval_episodes, it)
            save_policy(policy, out_dir)
            ema.restore(policy.model)
            last_log_time = time.perf_counter()

    ema.store(policy.model)
    if env is not None:
        evaluate(
            policy, env, preprocessor, postprocessor, cfg.eval_episodes, cfg.num_iters
        )
    save_policy(policy, out_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
