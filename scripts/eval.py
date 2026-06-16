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
from flow_planning.guidance import GoalGuidance
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
    goal_scale: float = 4.0  # >0 enables goal-distance attractor guidance
    goal_discount: float = 0.9  # weight decay toward earlier plan steps
    guidance_scale: float = 150.0  # >0 enables obstacle guidance
    guidance_margin: float = 0.08
    guidance_smooth: float = 3.0  # smoothness weight (accel penalty); see sweep_smooth
    episodes: int = 2  # batches of env.world_count episodes, 0 = unlimited
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

    if cfg.guidance_scale > 0:
        cfg.env.obstacle = True
    env = make_env(cfg.env, viewer)

    # compose inference-time guidance terms: goal attractor (+ obstacle cost)
    terms = []
    if cfg.goal_scale > 0:
        terms.append(
            GoalGuidance(
                obs_mean=stats[OBS_STATE]["mean"],
                obs_std=stats[OBS_STATE]["std"],
                goal_dim=policy.config.goal_dim,
                scale=cfg.goal_scale,
                device=device,
                discount=cfg.goal_discount,
            )
        )
    if cfg.guidance_scale > 0:
        terms.append(
            env.make_guidance(
                action_mean=stats[OBS_STATE]["mean"],
                action_std=stats[OBS_STATE]["std"],
                scale=cfg.guidance_scale,
                margin=cfg.guidance_margin,
                device=device,
                smooth=cfg.guidance_smooth,
            )
        )
    if terms:
        policy.guidance_fn = lambda x1, obs: sum(g(x1, obs) for g in terms)

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        frame = 0
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = torch.from_numpy(env.get_obs())
                action = policy.select_action(preprocessor({OBS_STATE: obs}))
                env.apply_action(postprocessor(action).numpy().astype(np.float32))
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
        total_succ += int(succ.sum())
        total_fail += int(fail.sum())
        total += n
        print(
            f"batch {ep}: success {succ.mean():.3f} "
            f"failure {fail.mean():.3f} timeout {(~succ & ~fail).mean():.3f}"
        )

    if total:
        print(
            f"overall: success {total_succ / total:.3f} "
            f"failure {total_fail / total:.3f} "
            f"timeout {(total - total_succ - total_fail) / total:.3f} "
            f"({total} episodes)"
        )
    viewer.close()


if __name__ == "__main__":
    main()
