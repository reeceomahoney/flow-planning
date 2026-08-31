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
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features
from safetensors.torch import load_file

import wandb
from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.envs.franka import box_pointcloud, subsample_cloud
from flow_planning.policy import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    FlowTransformer,
    make_flow_matching_pre_post_processors,
)
from flow_planning.utils import hf_column, make_run_dir


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka"
    batch_size: int = 256
    num_iters: int = 75_000
    dim_model: int = 256
    n_layers: int = 8
    cond_dim: int = -1  # -1: take it from the dataset's bend feature
    cloud_points: int = 0  # >0: condition on a scene point cloud of this many points
    init_from: str = ""  # checkpoint dir to fine-tune from (new cond pathway zero-init)
    freeze_backbone: bool = False  # train only the conditioning pathway + AdaLN
    episodes: int = 0  # >0: random subset of this many episodes
    aug_episodes: int = 0  # >0: subset of this many augmented (nonzero bend) episodes
    balance_tasks: bool = False  # sample windows uniformly per task_index
    orig_episodes: int = 0  # with aug_episodes: how many unaugmented episodes to add
    seed: int = 0
    warmup_iters: int = 500
    ema_decay: float = 0.999
    device: str = "cuda"
    out_dir: str = "outputs"
    run_dir: str = ""  # non-empty: exact output directory instead of a timestamp
    eval_every: int = 10_000
    horizon: int = (
        0  # >0: action-chunk length; 0 = longest episode (whole-trajectory plans)
    )
    eval_episodes: int = 256
    eval_search: int = (
        0  # >0 with env.obstacle: selector search over this many bend candidates
    )
    log_every: int = 100
    wandb_project: str = "flow-planning"
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"


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
        for _ in range(getattr(env, "max_frames", int(1.5 * env.episode_frames))):
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


def save_policy(policy, preprocessor, postprocessor, out_dir):
    compiled = policy.model
    policy.model = cast(FlowTransformer, compiled._orig_mod)
    policy.save_pretrained(out_dir)
    policy.model = compiled
    preprocessor.save_pretrained(out_dir)
    postprocessor.save_pretrained(out_dir)
    print(f"Saved policy to {out_dir}", flush=True)


@draccus.wrap()
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = (
        Path(cfg.run_dir) if cfg.run_dir else make_run_dir(cfg.out_dir, cfg.env.type)
    )
    print(f"Output directory: {out_dir}")

    mode = cast(Literal["online", "offline", "disabled"], cfg.wandb_mode)
    wandb.init(project=cfg.wandb_project, mode=mode, config=draccus.encode(cfg))

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    subset = None
    if cfg.episodes > 0:
        n_total = LeRobotDataset(cfg.repo_id).meta.total_episodes
        subset = sorted(rng.choice(n_total, cfg.episodes, replace=False).tolist())
    elif cfg.aug_episodes > 0:
        full = LeRobotDataset(cfg.repo_id).hf_dataset
        ep_all = hf_column(full, "episode_index")
        first = np.unique(ep_all, return_index=True)[1]
        aug = np.abs(hf_column(full, "bend")[first]).sum(1) > 0
        ids = ep_all[first]
        subset = sorted(
            rng.choice(ids[aug], cfg.aug_episodes, replace=False).tolist()
            + rng.choice(ids[~aug], cfg.orig_episodes, replace=False).tolist()
        )
        print(f"subset: {cfg.aug_episodes} augmented + {cfg.orig_episodes} original")
    dataset = LeRobotDataset(cfg.repo_id, episodes=subset)
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
        goal_state_start=cfg.env.goal_state_start,
        n_action_steps=cfg.env.n_action_steps,
        cond_dim=cfg.cond_dim
        if cfg.cond_dim >= 0
        else dataset.features["bend"]["shape"][0]
        if "bend" in dataset.features
        else 0,
        dim_model=cfg.dim_model,
        n_layers=cfg.n_layers,
        cloud_points=cfg.cloud_points,
    )
    # plan the full trajectory: horizon spans the longest episode; shorter
    # episodes pad their tail with the goal frame (absorbing).
    assert dataset.meta.episodes is not None
    policy_cfg.horizon = cfg.horizon or max(dataset.meta.episodes["length"])
    stats = dataset.meta.stats
    assert stats is not None

    policy = FlowMatchingPolicy(policy_cfg, dataset_stats=stats).to(device)
    if cfg.init_from:
        base = load_file(str(Path(cfg.init_from) / "model.safetensors"), device=device)
        missing, unexpected = policy.load_state_dict(base, strict=False)
        assert not unexpected, unexpected
        print(f"init from {cfg.init_from}: new params {sorted(missing)}")
        if hasattr(policy.model, "cond_mlp"):
            for param in policy.model.cond_mlp[-1].parameters():
                param.data.zero_()
    if cfg.freeze_backbone:
        keep = ("cond_mlp", "cloud_enc", "cloud_out", "null_cond", ".mod.")
        for name, param in policy.named_parameters():
            param.requires_grad = any(k in name for k in keep)
        n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"backbone frozen: {n_train / 1e6:.2f}M trainable")
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
    if env is not None and cfg.eval_search and cfg.env.obstacle:
        from eval import sample_cond

        from flow_planning.selector import AnalyticSelector

        geom = env.obstacle_geometry
        sel = AnalyticSelector(
            geom["center"] + geom["half_extents"],
            list(env.robot_base_pos),
            stats[ACTION],
            stats[OBS_STATE],
            joint_start=policy.state_dim,
            device=device,
            cube_size=env.cube_size,
            cube_index=getattr(env, "cube_index", 18),
        )
        joints = hf_column(dataset.hf_dataset, "observation.state")[::50, :7]
        sel.fc.calibrate_self(torch.as_tensor(joints, device=device))
        policy.selector_fn = sel.score
        policy.cond_candidates = sample_cond(
            hf_column(dataset.hf_dataset, "bend"), cfg.eval_search, rng, device
        )
        policy.hold_tol = 0.0
        print(f"eval: wall search over {cfg.eval_search} candidates", flush=True)

    # frames live once in host RAM; full-horizon windows are gathered per batch
    # with a clamped index, which IS the absorbing goal-frame tail pad
    hf = dataset.hf_dataset
    obs_all = torch.from_numpy(hf_column(hf, "observation.state"))
    act_all = torch.from_numpy(hf_column(hf, "action"))
    bend_all = torch.from_numpy(hf_column(hf, "bend")) if policy_cfg.cond_dim else None
    ep = torch.as_tensor(hf_column(hf, "episode_index"), dtype=torch.long)
    clouds = None
    if policy_cfg.cloud_points:
        walls = hf_column(hf, "wall")
        first = np.unique(ep.numpy(), return_index=True)[1]
        clouds = torch.zeros(int(ep.max()) + 1, policy_cfg.cloud_points, 3)
        th = getattr(cfg.env, "table_height", 0.1)
        table = [0.0, -0.5, 0.5 * th], [0.4, 0.4, 0.5 * th]
        n_walls = 0
        for f in first:
            w = walls[f]
            if np.abs(w).sum() == 0:
                continue
            pts = box_pointcloud([w[:3], table[0]], [w[3:], table[1]], th)
            clouds[ep[f]] = torch.from_numpy(
                subsample_cloud(pts, policy_cfg.cloud_points, rng)
            )
            n_walls += 1
        clouds = clouds.to(device)
        print(f"clouds: {n_walls}/{len(first)} episodes have a wall", flush=True)
    last = torch.zeros(int(ep.max()) + 1, dtype=torch.long)
    last[ep] = torch.arange(len(ep))  # sequential writes leave each episode's end
    ep_end = last[ep]
    n_samples = obs_all.shape[0]
    steps = torch.arange(policy_cfg.horizon)
    weights = None
    if cfg.balance_tasks:
        ti = torch.as_tensor(hf_column(hf, "task_index"), dtype=torch.long)
        counts = torch.bincount(ti).float()
        weights = 1.0 / counts[ti]
        print(f"task-balanced sampling over {len(counts)} tasks", flush=True)
    mb = (obs_all.nbytes + act_all.nbytes) / 1e6
    print(f"Loaded {n_samples} frames ({mb:.0f} MB) in RAM", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy.get_optim_params() if p.requires_grad],
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
        if weights is not None:
            idx = torch.multinomial(weights, cfg.batch_size, replacement=True)
        else:
            idx = torch.randint(0, n_samples, (cfg.batch_size,))
        rows = torch.minimum(idx[:, None] + steps, ep_end[idx][:, None])
        batch = preprocessor({OBS_STATE: obs_all[rows], ACTION: act_all[rows]})
        if bend_all is not None:
            batch["bend"] = bend_all[idx].to(device, non_blocking=True)
        if clouds is not None:
            batch["cloud"] = clouds[ep[idx].to(device)]
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
            save_policy(policy, preprocessor, postprocessor, out_dir)
            ema.restore(policy.model)
            last_log_time = time.perf_counter()

    ema.store(policy.model)
    if env is not None:
        evaluate(
            policy, env, preprocessor, postprocessor, cfg.eval_episodes, cfg.num_iters
        )
    save_policy(policy, preprocessor, postprocessor, out_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
