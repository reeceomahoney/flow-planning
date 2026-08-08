"""Roll out the trained policy: headless batched eval or interactive viewer."""

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
    best_of: int = 1  # >1: sample K plans/world, execute the lowest collision score
    cond: str = "null"  # "null" | "zero" (unbent original) | "search" | "a,b" fixed
    cond_grid: str = ""  # override search candidates: "a,b,c;d,e,f;..."
    n_cond: int = 8  # search: # candidates drawn from the data when no cond_grid
    cond_whiten: bool = True  # scale-normalize bend dims before FPS candidate spread
    cond_pick: str = "fps"  # "fps" (spread, hits degenerate extremes) | "dense"
    mode_reduce: str = "min"  # rank modes by "min"|"median"|"max" of their samples
    selector_speed: float = 0.0  # weight of the demanded-joint-speed hinge
    selector_margin: float = 0.0  # >0: shortest path among plans clearing this much
    selector_obj: str = "path"  # objective under selector_margin: "path"|"speed"
    selector_radii: bool = False  # clearance to the link MESH, not the centreline
    selector_cap: float = 0.08  # clearance sufficiency cap
    selector_prog: float = 0.03  # progress-term weight
    guidance_scale: float = 0.0  # >0: collision-cost guidance instead of best-of-N
    guidance_margin: float = 0.1  # keep-out margin for the guidance hinge
    guidance_len: float = 0.0  # path-length weight in the guidance cost
    guidance_target: str = "x1"  # guide the predicted clean traj, or raw "xt"
    warm_start_t: float = 0.0  # >0: renoise-and-refine the previous plan (SDEdit)


def fps_idx(pts: np.ndarray, k: int, seed: int) -> list[int]:
    """Farthest-point sampling: indices of k points with maximal mutual spread."""
    idx = [seed]
    d = np.linalg.norm(pts - pts[seed], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return idx


def dense_idx(pts: np.ndarray, k: int, keep: float = 0.7, m: int = 5) -> list[int]:
    """FPS restricted to the well-supported region. Plain FPS seeks the extremes
    of this space, and those are degenerate modes (cube barely lifts), so drop
    isolated labels by m-NN distance first. k-means is no good here: a lone
    outlier forms a singleton cluster whose centroid is the outlier itself."""
    d = np.sort(np.linalg.norm(pts[:, None] - pts[None], axis=-1), axis=1)[:, m]
    keep_i = np.argsort(d)[: max(k, int(keep * len(pts)))]
    sub = pts[keep_i]
    seed = int(np.linalg.norm(sub - sub.mean(0), axis=1).argmin())
    return [int(keep_i[i]) for i in fps_idx(sub, k, seed)]


def demo_speed_limit(dataset, n_arm: int = 7, q: float = 99.0) -> np.ndarray:
    """Per-joint step speed the demos stay under — the plan's feasibility bar.
    A statistic of the free-space training set, like the action stats; no extra
    rollouts and nothing obstacle-specific."""
    hf = dataset.hf_dataset.with_format("numpy")
    act = np.asarray(hf[ACTION])[:, :n_arm]
    ep = np.asarray(hf["episode_index"])
    dq = np.abs(np.diff(act, axis=0))[ep[1:] == ep[:-1]]  # drop episode seams
    return np.percentile(dq, q, axis=0)


def data_cond_candidates(
    dataset,
    k: int,
    device,
    whiten: bool = True,
    pick: str = "fps",
) -> torch.Tensor:
    """K conditioning modes drawn from the dataset's own bend labels via FPS. The
    augmentation only kept labels whose sim-replay succeeded, so the set is
    on-manifold and execution-validated — no hand-tuned grid needed."""
    bend = np.asarray(dataset.hf_dataset.with_format("numpy")["bend"])
    modes = np.unique(bend, axis=0)  # distinct latents (bend is const per episode)
    if len(modes) <= k:
        return torch.tensor(modes, dtype=torch.float32, device=device)
    # bend is polar (phi*cos(theta), phi*sin(theta)); whitening equalizes the axes
    w = modes / (modes.std(0) + 1e-6) if whiten else modes
    if pick == "dense":
        idx = dense_idx(w, k)
    else:
        seed = int(np.linalg.norm(w - w.mean(0), axis=1).argmin())
        idx = fps_idx(w, k, seed)
    return torch.tensor(modes[idx], dtype=torch.float32, device=device)


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
    policy.warm_start_t = cfg.warm_start_t
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

    searching = cfg.cond == "search"
    want_sel = cfg.best_of > 1 or searching or cfg.guidance_scale > 0
    if want_sel and hasattr(env, "ee_state_index"):
        geom = env.obstacle_geometry
        sel = AnalyticSelector(
            geom["center"] + geom["half_extents"],
            list(env.robot_base_pos),
            stats[ACTION],
            stats[OBS_STATE],
            joint_start=policy.state_dim,
            device=device,
            cube_size=env.cube_size,
            cap=cfg.selector_cap,
            prog=cfg.selector_prog,
            speed=cfg.selector_speed,
            vlim=demo_speed_limit(dataset),
            margin=cfg.selector_margin,
            obj=cfg.selector_obj,
            radii=cfg.selector_radii,
        )
        if cfg.guidance_scale > 0:  # ablation: guidance instead of best-of-N selection
            policy.guidance_fn = lambda t: sel.guidance_cost(
                t, cfg.guidance_margin, cfg.guidance_len
            )
            policy.guidance_scale = cfg.guidance_scale
            policy.guidance_target = cfg.guidance_target
        if cfg.best_of > 1 or searching:
            policy.selector_fn = sel.score
            policy.n_samples = cfg.best_of
            policy.mode_reduce = cfg.mode_reduce
    if getattr(policy.config, "cond_dim", 0):
        if cfg.cond == "zero":
            policy.cond = torch.zeros(1, policy.config.cond_dim, device=device)
        elif "," in cfg.cond:  # fixed bend command, e.g. "0.5,0.5"
            vals = [float(v) for v in cfg.cond.split(",")]
            policy.cond = torch.tensor([vals], device=device)
        elif cfg.cond == "search":  # candidates scored + latched via selector
            assert policy.selector_fn is not None, "search needs a selector"
            if cfg.cond_grid:  # explicit hand grid
                grid = [
                    [float(v) for v in g.split(",")] for g in cfg.cond_grid.split(";")
                ]
                cand = torch.tensor(grid, device=device)
            else:  # scalable: sample execution-validated modes from the data itself
                cand = data_cond_candidates(
                    dataset, cfg.n_cond, device, cfg.cond_whiten, cfg.cond_pick
                )
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
