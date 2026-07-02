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
from flow_planning.guidance import Guidance
from flow_planning.guidance_net import LearnedGuidance
from flow_planning.kinematics import build_franka_chain, ee_positions
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)
from flow_planning.utils import latest_run_dir, make_viewer


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/particle"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    episodes: int = 1  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 20.0
    viewer: str = "none"  # "none", "rerun", or "opengl"
    rrd: str = ""  # record a rerun .rrd to this path (view locally: `rerun <rrd>`)
    seed: int = 0
    learned_guidance_ckpt: str = "outputs/guidance_net_particle.pt"
    learned_guidance_scale: float = 3.0
    n_action_steps: int = 0  # >0 overrides the checkpoint replan interval
    num_inference_steps: int = 0  # >0 overrides the checkpoint ODE step count
    guidance_mode: str = "analytic"  # "analytic" (Guidance terms) | "learned"


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
    if cfg.num_inference_steps > 0:
        policy.config.num_inference_steps = cfg.num_inference_steps
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

    if cfg.learned_guidance_ckpt == "none":
        policy.guidance_fn = None
    elif cfg.guidance_mode == "learned":
        geom = env.obstacle_geometry
        if hasattr(env, "ee_state_index"):  # franka: 3D box, EE-relative
            box, pos_index, pos_dims = (
                geom["center"] + geom["half_extents"],
                env.ee_state_index,
                3,
            )
        else:  # particle: representative 2D bar, ball pos at obs[0:2]
            c, h = np.asarray(geom["center"]), np.asarray(geom["half_extents"])
            box, pos_index, pos_dims = np.concatenate([c[0], h[0]]), 0, 2
        policy.guidance_fn = LearnedGuidance(
            cfg.learned_guidance_ckpt,
            box,
            stats[OBS_STATE],
            pos_index,
            device,
            cfg.learned_guidance_scale,
            pos_dims=pos_dims,
        )
    else:
        policy.guidance_fn = Guidance(
            env,
            policy.state_dim,
            stats[ACTION],
            device,
            obs_stats=stats[OBS_STATE],
        )
    arm = hasattr(env, "ee_state_index")  # franka: planned path needs EE FK
    if arm:
        chain = build_franka_chain(device)[0]
        base_pos = np.asarray(env.robot_base_pos, np.float32)  # base frame -> world

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    plan_t, n_plan = 0.0, 0  # planning latency (only the frames that replan)
    accels = []  # per-step action accel magnitudes (smoothness proxy)
    stages = getattr(env, "STAGES", None)  # franka pick-place stage ladder
    stage_hist = np.zeros(len(stages) if stages else 0, int)
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        max_stage = np.zeros(n, int)
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
                    if arm:
                        q = torch.from_numpy(chunk[:, :7]).float().to(device)
                        path = ee_positions(chain, q).cpu().numpy() + base_pos
                    else:
                        path = chunk[:, :2]  # particle: action is the xy target
                    env.set_predicted_path(path)
                frame += 1
                pbar.update(1)
                succ |= env.success()
                if stages is not None:
                    max_stage = np.maximum(max_stage, env.stage())
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
        if stages is not None:
            counts = np.bincount(max_stage, minlength=len(stages))
            stage_hist += counts
            print(
                "  stage: "
                + "  ".join(f"{name} {c / n:.2f}" for name, c in zip(stages, counts))
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
    if stages is not None and total:
        reward = sum(i * c for i, c in enumerate(stage_hist)) / total
        print(f"overall reward (avg furthest stage reached): {reward:.2f}")
        print("overall stage (furthest reached, fraction of episodes):")
        print(
            "  "
            + "  ".join(
                f"{name} {c / total:.3f}" for name, c in zip(stages, stage_hist)
            )
        )
    if n_plan:
        print(f"plan latency: {1e3 * plan_t / n_plan:.1f} ms/replan ({n_plan} replans)")
    viewer.close()


if __name__ == "__main__":
    main()
