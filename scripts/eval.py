"""Roll out the trained policy: headless batched eval or interactive viewer."""

import itertools
import time
from dataclasses import dataclass, field

import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from tqdm import tqdm

from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.guidance import Guidance, GuidanceConfig
from flow_planning.kinematics import build_franka_chain, ee_positions
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)
from flow_planning.utils import latest_run_dir, make_viewer


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    episodes: int = 2  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 20.0
    viewer: str = "none"  # "none", "rerun", or "opengl"
    rrd: str = ""  # record a rerun .rrd to this path (view locally: `rerun <rrd>`)
    seed: int = 0


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    checkpoint = cfg.checkpoint or latest_run_dir("outputs", cfg.env.type)
    print(f"Loading checkpoint: {checkpoint}")
    policy = FlowMatchingPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    policy.to(device)

    dataset = LeRobotDataset(cfg.repo_id)
    stats = dataset.meta.stats
    assert stats is not None
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config, dataset_stats=stats
    )

    # action stats for unnormalizing the planned path when visualizing
    act_mean = np.asarray(stats[ACTION]["mean"])
    act_std = np.asarray(stats[ACTION]["std"])

    live = cfg.viewer != "none" or bool(cfg.rrd)
    viewer = make_viewer(cfg.viewer, cfg.rrd)

    env = make_env(cfg.env, viewer)
    arena = getattr(cfg.env, "arena_size", None)
    if arena is not None:  # particle: clamp actions to the arena
        low = torch.as_tensor(
            (-arena - act_mean) / act_std, dtype=torch.float32, device=device
        )
        high = torch.as_tensor(
            (arena - act_mean) / act_std, dtype=torch.float32, device=device
        )
        policy.action_clip = (low, high)

    policy.guidance_fn = Guidance(
        cfg.guidance,
        env,
        policy.state_dim,
        stats[ACTION],
        device,
        obs_stats=stats[OBS_STATE],
    )
    chain = build_franka_chain(device)[0]
    base_pos = np.asarray(env.robot_base_pos, np.float32)  # base frame -> world

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    plan_t, n_plan = 0.0, 0  # planning latency (only the frames that replan)
    accels = []  # per-step action accel magnitudes (smoothness proxy)
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        acts = []
        frame = 0
        pbar = tqdm(total=frames, desc=f"batch {ep}", leave=False)
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = torch.from_numpy(env.get_obs())
                replanning = len(policy._action_queue) == 0
                if replanning:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                action = policy.select_action(preprocessor({OBS_STATE: obs}))
                if replanning:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    plan_t += time.perf_counter() - t0
                    n_plan += 1
                phys = postprocessor(action).numpy().astype(np.float32)
                acts.append(phys)
                env.apply_action(phys)
                if live and policy.last_chunk is not None:
                    chunk = policy.last_chunk[0].cpu().numpy() * act_std + act_mean
                    q = torch.from_numpy(chunk[:, :7]).float().to(device)
                    path = ee_positions(chain, q).cpu().numpy() + base_pos
                    env.set_predicted_path(path)
                frame += 1
                pbar.update(1)
                succ |= env.success()
                if (succ | env.failure()).all():
                    break
            if live:
                env.render()
        pbar.close()
        if not viewer.is_running():
            break
        fail = env.failure()
        stuck = ~succ & ~fail
        a = np.stack(acts)  # (T, n, act_dim)
        accel = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)
        accels.append(accel.ravel())
        accel_p95 = np.percentile(accel, 95) if accel.size else float("nan")
        total_succ += int(succ.sum())
        total_fail += int(fail.sum())
        total += n
        print(
            f"batch {ep}: success {succ.mean():.3f} "
            f"failure {fail.mean():.3f} timeout {stuck.mean():.3f} "
            f"accel_p95 {accel_p95:.4f} "
        )

    if total:
        ac = np.concatenate(accels)
        print(
            f"overall: success {total_succ / total:.3f} "
            f"failure {total_fail / total:.3f} "
            f"timeout {(total - total_succ - total_fail) / total:.3f} "
            f"accel_p95 {np.percentile(ac, 95):.4f} "
            f"p99 {np.percentile(ac, 99):.4f} max {ac.max():.3f} "
            f"({total} episodes)"
        )
    if n_plan:
        print(f"plan latency: {1e3 * plan_t / n_plan:.1f} ms/replan ({n_plan} replans)")
    viewer.close()


if __name__ == "__main__":
    main()
