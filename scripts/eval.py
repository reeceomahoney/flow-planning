"""Roll out the trained policy: headless batched eval or interactive viewer."""

import enum
import itertools
from collections import Counter
from dataclasses import dataclass, field

import cv2
import draccus
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from tqdm import tqdm

from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.kinematics import build_franka_chain, ee_positions
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)
from flow_planning.selector import AnalyticSelector
from flow_planning.utils import latest_run_dir, make_viewer


class Cond(enum.StrEnum):
    NULL = "null"  # the learned CFG null token
    ZERO = "zero"  # the label the unbent originals carry
    RANDOM = "random"  # one uniform latent per world, redrawn each episode
    SEARCH = "search"  # n_cond uniform candidates, selector-scored and latched


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    repo_id: str = "reece-omahoney/franka"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    episodes: int = 1  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 0.0  # 0 = train.py's 1.5 * env.episode_frames
    viewer: str = "none"  # "none", "rerun", or "opengl"
    video: str = ""  # non-empty: render headless to this mp4
    seed: int = 0
    n_action_steps: int = 0  # >0 overrides the checkpoint replan interval
    num_inference_steps: int = 0  # >0 overrides the checkpoint ODE step count
    cond: Cond = Cond.NULL
    n_cond: int = 8  # search: uniform candidates drawn at startup
    selector_margin: float = 0.0  # keep-out inflation on the collision check
    selector_radii: bool = False  # clearance to the link MESH, not the centreline


def sample_cond(bend: np.ndarray, k: int, rng, device) -> torch.Tensor:
    r = np.linalg.norm(bend, axis=1)
    nz = r > 0
    assert nz.any(), "dataset has no bent episodes; every bend label is zero"
    ang = np.arctan2(bend[nz, 1], bend[nz, 0])
    phi = rng.uniform(r[nz].min(), r.max(), k)
    theta = rng.uniform(ang.min(), ang.max(), k)
    xy = np.stack([phi * np.cos(theta), phi * np.sin(theta)], axis=1)
    return torch.tensor(xy, dtype=torch.float32, device=device)


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

    live = cfg.viewer != "none" or bool(cfg.video)
    viewer = make_viewer(
        "opengl" if cfg.video else cfg.viewer, headless=bool(cfg.video)
    )

    env = make_env(cfg.env, viewer)
    env.rng = np.random.default_rng(cfg.seed)  # same instances as train.py's evaluate
    arena = getattr(cfg.env, "arena_size", None)
    if arena is not None:  # particle: clamp actions to the arena
        low = torch.as_tensor(
            (-arena - act_mean) / act_std, dtype=torch.float32, device=device
        )
        high = torch.as_tensor(
            (arena - act_mean) / act_std, dtype=torch.float32, device=device
        )
        policy.action_clip = (low, high)

    searching = cfg.cond is Cond.SEARCH
    if searching and cfg.env.obstacle and hasattr(env, "ee_state_index"):
        geom = env.obstacle_geometry
        sel = AnalyticSelector(
            geom["center"] + geom["half_extents"],
            list(env.robot_base_pos),
            stats[ACTION],
            stats[OBS_STATE],
            joint_start=policy.state_dim,
            device=device,
            cube_size=env.cube_size,
            margin=cfg.selector_margin,
            radii=cfg.selector_radii,
        )
        policy.selector_fn = sel.score
    rng = np.random.default_rng(cfg.seed)
    bend = None
    if getattr(policy.config, "cond_dim", 0):
        bend = np.asarray(dataset.hf_dataset.with_format("numpy")["bend"])
        if cfg.cond is Cond.ZERO:
            policy.cond = torch.zeros(1, policy.config.cond_dim, device=device)
        elif cfg.cond is Cond.SEARCH:  # candidates scored + latched via selector
            assert policy.selector_fn is not None, "search needs --env.obstacle"
            cand = sample_cond(bend, cfg.n_cond, rng, device)
            policy.cond_candidates = cand
            print(f"search candidates ({len(cand)}):\n{cand.cpu().numpy().round(2)}")

    arm = hasattr(env, "ee_state_index")  # franka: planned path needs EE FK
    if arm:
        chain = build_franka_chain(device)[0]
        base_pos = np.asarray(env.robot_base_pos, np.float32)  # base frame -> world

    n = cfg.env.world_count
    frames = (
        round(cfg.episode_seconds * cfg.env.fps)
        if cfg.episode_seconds > 0
        else int(1.5 * env.episode_frames)
    )
    writer = None
    total_succ = total_fail = total = 0
    stages = getattr(env, "STAGES", None)  # franka pick-place stage ladder
    nz = len(stages) if stages else 0
    stage_hist = np.zeros(nz, int)
    stuck_hist = {k: np.zeros(nz, int) for k in ("timeout", "failure")}
    contacts = Counter()  # what hit the wall, pooled over batches
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        if cfg.cond is Cond.RANDOM and bend is not None:
            policy.cond = sample_cond(bend, n, rng, device)
        succ = np.zeros(n, dtype=bool)
        max_stage = np.zeros(n, int)
        frame = 0
        pbar = tqdm(total=frames, desc=f"batch {ep}", leave=False)
        while frame < frames and viewer.is_running():
            if viewer.should_step():
                obs = torch.from_numpy(env.get_obs())
                action = policy.select_action(preprocessor({OBS_STATE: obs}))
                phys = postprocessor(action).numpy().astype(np.float32)
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
                if not cfg.video and (succ | env.failure()).all():
                    break
            if live:
                env.render()
            if cfg.video:
                f = viewer.get_frame().numpy()
                if writer is None:
                    writer = cv2.VideoWriter(
                        cfg.video,
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        cfg.env.fps,
                        (f.shape[1], f.shape[0]),
                    )
                writer.write(f[..., ::-1])
        pbar.close()
        if not viewer.is_running():
            break
        fail = env.failure()
        stuck = ~succ & ~fail
        total_succ += int(succ.sum())
        total_fail += int(fail.sum())
        total += n
        print(
            f"batch {ep}: success {succ.mean():.3f} "
            f"failure {fail.mean():.3f} timeout {stuck.mean():.3f}"
        )
        if hasattr(env, "contact_labels") and fail.any():
            contacts.update(env.contact_labels()[fail].tolist())
        if stages is not None:
            stage_hist += np.bincount(max_stage, minlength=nz)
            for name, m in (("timeout", stuck), ("failure", fail)):
                if m.any():
                    stuck_hist[name] += np.bincount(max_stage[m], minlength=nz)

    if total:
        print(
            f"overall: success {total_succ / total:.3f} "
            f"failure {total_fail / total:.3f} "
            f"timeout {(total - total_succ - total_fail) / total:.3f} "
            f"({total} episodes, cond {cfg.cond})"
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
        for k, h in stuck_hist.items():
            if h.sum():
                print(
                    f"overall stage|{k} (n={h.sum()}): "
                    + "  ".join(f"{s} {v / h.sum():.3f}" for s, v in zip(stages, h))
                )
    if contacts:
        hits = "  ".join(f"{k}:{v}" for k, v in contacts.most_common())
        print(f"wall contacts (n={sum(contacts.values())}): {hits}")
    lc = policy.latched_cond
    if searching and lc is not None:
        lc = lc.cpu().numpy()
        print(f"latched cond mean {lc.mean(0).round(3)} std {lc.std(0).round(3)}")
    if writer is not None:
        writer.release()
        print(f"Saved video to {cfg.video}")
    viewer.close()


if __name__ == "__main__":
    main()
