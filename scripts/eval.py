"""Roll out the trained policy: headless batched eval or interactive viewer."""

import itertools
from dataclasses import dataclass, field
from urllib.parse import quote

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from flow_planning.envs import EnvConfig, ParticleConfig, make_env
from flow_planning.paths import latest_run_dir
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)
from flow_planning.trajectory import Planner


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
    guidance_scale: float = 50.0  # >0 enables obstacle guidance
    guidance_margin: float = 0.08
    goal_weight: float = 1.0  # classifier-free goal conditioning weight
    episodes: int = 2  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 20.0
    replan_every: int = 10  # frames between replans (receding horizon)
    lookahead: float = 0.15  # pure-pursuit lookahead distance (physical units)
    inpaint: bool = True  # pin the plan endpoints to current state and goal
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
    policy.config.goal_guidance_weight = cfg.goal_weight

    stats = LeRobotDataset(cfg.repo_id).meta.stats
    assert stats is not None
    preprocessor, _ = make_flow_matching_pre_post_processors(
        policy.config, dataset_stats=stats
    )

    live = cfg.viewer != "none"
    viewer = make_viewer(cfg.viewer)

    if cfg.guidance_scale > 0:
        cfg.env.obstacle = True
    env = make_env(cfg.env, viewer)

    guidance_fn = None
    if cfg.guidance_scale > 0:
        guidance_fn = env.make_guidance(
            action_mean=stats["action"]["mean"],
            action_std=stats["action"]["std"],
            scale=cfg.guidance_scale,
            margin=cfg.guidance_margin,
            device=device,
        )

    planner = Planner(
        policy,
        preprocessor,
        env,
        action_mean=stats["action"]["mean"],
        action_std=stats["action"]["std"],
        lookahead=cfg.lookahead,
        replan_every=cfg.replan_every,
        guidance_fn=guidance_fn,
        inpaint=cfg.inpaint,
        device=device,
    )

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        planner.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        frame = 0
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = env.get_obs()
                env.apply_action(planner.act(obs))
                if live:
                    # visualize the current plan (world 0)
                    assert planner.tracker.plan is not None
                    env.set_predicted_path(planner.tracker.plan[0].cpu().numpy())
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
