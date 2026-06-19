"""Roll out the trained policy: headless batched eval or interactive viewer."""

import itertools
from dataclasses import dataclass, field
from urllib.parse import quote

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from flow_planning.envs import EnvConfig, ParticleConfig, make_env
from flow_planning.guidance import Guidance, GuidanceConfig
from flow_planning.paths import latest_run_dir
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config:
    env: EnvConfig = field(
        default_factory=lambda: ParticleConfig(
            world_count=256, obstacle=True, obstacle_random=True
        )
    )
    repo_id: str = "reece-omahoney/particle"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    guidance: GuidanceConfig = field(
        default_factory=lambda: GuidanceConfig(
            goal_scale=0.0, obstacle_scale=20.0, smooth_scale=0.5
        )
    )
    episodes: int = 2  # batches of env.world_count episodes, 0 = unlimited
    n_action_steps: int = 75  # >0 overrides the checkpoint's replan interval
    episode_seconds: float = 20.0
    viewer: str = "none"  # "none", "rerun", or "opengl"
    seed: int = 0


def make_viewer(name: str):
    if name == "none":
        return newton.viewer.ViewerNull()
    if name == "opengl":
        return newton.viewer.ViewerGL()
    if name == "rerun":
        web_port = 9090
        viewer = newton.viewer.ViewerRerun(web_port=web_port)
        if viewer._grpc_server_uri is not None:
            grpc_uri = quote(viewer._grpc_server_uri, safe="")
            print(f"Rerun viewer: http://localhost:{web_port}/?url={grpc_uri}")
        return viewer
    raise ValueError(f"Unknown viewer: {name}")


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    checkpoint = cfg.checkpoint or latest_run_dir("outputs", cfg.env.type)
    print(f"Loading checkpoint: {checkpoint}")
    policy = FlowMatchingPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    if cfg.n_action_steps > 0:
        policy.config.n_action_steps = cfg.n_action_steps
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

    live = cfg.viewer != "none"
    viewer = make_viewer(cfg.viewer)

    if cfg.guidance.obstacle_scale > 0:
        cfg.env.obstacle = True
    env = make_env(cfg.env, viewer)
    arena = getattr(cfg.env, "arena_size", np.inf)
    low = torch.as_tensor(
        (-arena - act_mean) / act_std, dtype=torch.float32, device=device
    )
    high = torch.as_tensor(
        (arena - act_mean) / act_std, dtype=torch.float32, device=device
    )
    policy.action_clip = (low, high)

    policy.guidance_fn = Guidance(
        cfg.guidance, env, policy.config.goal_dim, stats[OBS_STATE], device
    )

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    path_ratios = []  # per-world path-length / straight-line ratio (>1 = wandering)
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        acts = []
        positions = []
        frame = 0
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = torch.from_numpy(env.get_obs())
                positions.append(obs[:, :2].numpy().copy())
                action = policy.select_action(preprocessor({OBS_STATE: obs}))
                phys = postprocessor(action).numpy().astype(np.float32)
                acts.append(phys)
                env.apply_action(phys)
                if live and policy.last_chunk is not None:
                    path = policy.last_chunk[0].cpu().numpy() * act_std + act_mean
                    env.set_predicted_path(path)
                frame += 1
                succ |= env.success()
                if (succ | env.failure()).all():
                    break
            if live:
                env.render()
        if not viewer.is_running():
            break
        fail = env.failure()
        stuck = ~succ & ~fail
        if stuck.any():
            obs = env.get_obs()
            pos, goal = obs[stuck, :2], obs[stuck, 4:6]
            goal_dist = np.linalg.norm(pos - goal, axis=1)
            print(
                f"  timeouts: |y| median {np.median(np.abs(pos[:, 1])):.2f}, "
                f"goal dist median {np.median(goal_dist):.2f}, "
                f"wrong side {(pos[:, 1] * goal[:, 1] < 0).mean():.2f}"
            )
        a = np.stack(acts)  # (T, n, act_dim)
        accel = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)
        accel_p95 = np.percentile(accel, 95) if accel.size else float("nan")
        # path length vs straight-line start->goal: >1 means the ball wandered
        p = np.stack(positions)  # (T, n, 2)
        goal_xy = env.get_obs()[:, 4:6]
        path_len = np.linalg.norm(np.diff(p, axis=0), axis=-1).sum(0)  # (n,)
        straight = np.linalg.norm(p[0] - goal_xy, axis=-1) + 1e-6
        ratio = path_len / straight
        path_ratios.append(ratio)
        total_succ += int(succ.sum())
        total_fail += int(fail.sum())
        total += n
        print(
            f"batch {ep}: success {succ.mean():.3f} "
            f"failure {fail.mean():.3f} timeout {stuck.mean():.3f} "
            f"accel_p95 {accel_p95:.4f} path_ratio_med {np.median(ratio):.2f} "
        )

    if total:
        pr = np.concatenate(path_ratios)
        print(
            f"overall: success {total_succ / total:.3f} "
            f"failure {total_fail / total:.3f} "
            f"timeout {(total - total_succ - total_fail) / total:.3f} "
            f"path_ratio med {np.median(pr):.2f} p95 {np.percentile(pr, 95):.2f} "
            f"({total} episodes)"
        )
    viewer.close()


if __name__ == "__main__":
    main()
