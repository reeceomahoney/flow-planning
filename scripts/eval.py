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
    repo_id: str = "reece-omahoney/particle"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    episodes: int = 1  # batches of env.world_count episodes, 0 = unlimited
    episode_seconds: float = 20.0
    viewer: str = "none"  # "none", "rerun", or "opengl"
    rrd: str = ""  # record a rerun .rrd to this path (view locally: `rerun <rrd>`)
    seed: int = 0
    n_action_steps: int = 0  # >0 overrides the checkpoint replan interval
    num_inference_steps: int = 0  # >0 overrides the checkpoint ODE step count
    best_of: int = 1  # >1: sample K plans/world, execute the lowest collision score
    cond: str = "null"  # "null" | "zero" | "search" | "a,b,..." fixed
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
    # whiten: raw dims span 2.0 (lat) vs 0.8 (vert), so FPS ignores the short dim
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
        sel.n_worlds = cfg.env.world_count
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
        gs = cfg.env.goal_state_start  # plan-cube dims, for hover diagnostics
        cube_m = np.asarray(stats[OBS_STATE]["mean"])[gs : gs + 3]
        cube_s = np.asarray(stats[OBS_STATE]["std"])[gs : gs + 3]

    n = cfg.env.world_count
    frames = round(cfg.episode_seconds * cfg.env.fps)
    total_succ = total_fail = total = 0
    plan_t, n_plan = 0.0, 0  # planning latency (only the frames that replan)
    accels = []  # per-step action accel magnitudes (smoothness proxy)
    stages = getattr(env, "STAGES", None)  # franka pick-place stage ladder
    stage_hist = np.zeros(len(stages) if stages else 0, int)
    nz = len(stages) if stages else 0
    stuck_hist = {k: np.zeros(nz, int) for k in ("timeout", "failure")}
    episodes = range(cfg.episodes) if cfg.episodes > 0 else itertools.count()
    for ep in episodes:
        if not viewer.is_running():
            break
        policy.reset()
        env.reset()
        succ = np.zeros(n, dtype=bool)
        max_stage = np.zeros(n, int)
        acts = []
        plan_low = np.full(n, np.inf)  # closest interior plan-cube approach to goal
        term_jumps, max_jumps = [], []
        shifts = []  # warm-start re-anchor offset per replan
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
                    if policy.last_shift is not None:
                        shifts.append(policy.last_shift.cpu().numpy())
                    if arm and policy.last_traj is not None:
                        # hover diagnostics: does the plan descend to the goal
                        # before the clamped final frame, or teleport into it?
                        pc = policy.last_traj[..., gs : gs + 3].cpu().numpy()
                        pc = pc * cube_s + cube_m
                        gw = obs[:, -3:].numpy()
                        d = np.linalg.norm(pc[:, :-1] - gw[:, None], axis=-1)
                        plan_low = np.minimum(plan_low, d.min(axis=1))
                        term_jumps.append(
                            np.linalg.norm(pc[:, -1] - pc[:, -2], axis=-1)
                        )
                        steps = np.linalg.norm(np.diff(pc, axis=1), axis=-1)
                        max_jumps.append(steps.max(axis=1))
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
        if shifts:  # re-anchor advance vs the n_action_steps it should be
            s = np.stack(shifts)  # (replans, n_worlds)
            print(
                f"  re-anchor shift: median {np.median(s):.0f} "
                f"frac_zero {(s == 0).mean():.2f} of {len(s)} replans"
            )
            if n <= 4:
                print(f"    per-world seq: {s.T.tolist()}")
        if n <= 16:  # which world is which, for picking one out in the viewer
            tag = np.where(succ, "succ", np.where(fail, "FAIL", "stuck"))
            print("  per-world: " + "  ".join(f"{i}:{t}" for i, t in enumerate(tag)))
        if hasattr(env, "contact_labels") and fail.any():  # what hit the wall
            labs = env.contact_labels()[fail]
            uniq, cnt = np.unique(labs, return_counts=True)
            order = np.argsort(-cnt)
            hits = "  ".join(f"{uniq[i]}:{cnt[i]}" for i in order)
            print(f"  wall contacts (n={fail.sum()}): {hits}")
        if want_sel and policy.last_traj is not None:  # is the picked plan trackable?
            ov = sel.overspeed(policy.last_traj).cpu().numpy()
            print(f"  overspeed mean {ov.mean():.3f} p90 {np.percentile(ov, 90):.3f}")
        if want_sel and sel.feas_calls:  # supply or execution?
            print(
                f"  plans clearing margin {sel.feas_ok / sel.feas_n:.3f}; "
                f"mean fraction of WORLDS with no feasible candidate "
                f"{sel.feas_none / sel.feas_calls:.3f} ({int(sel.feas_calls)} replans)"
            )
            sel.feas_n = sel.feas_ok = sel.feas_none = sel.feas_calls = 0.0
        lc = policy.latched_cond
        if searching and lc is not None:  # latent distribution, for diagnosis
            lc = lc.cpu().numpy()
            print(f"  latched cond mean {lc.mean(0).round(3)} std {lc.std(0).round(3)}")
        if term_jumps:
            tj, mj = np.concatenate(term_jumps), np.concatenate(max_jumps)
            print(
                f"  plan term_jump p50 {np.median(tj):.3f} "
                f"p90 {np.percentile(tj, 90):.3f} "
                f"max_jump p50 {np.median(mj):.3f} p90 {np.percentile(mj, 90):.3f}"
            )
            for name, m in (("succ", succ), ("fail", fail), ("stuck", stuck)):
                if m.any():
                    print(
                        f"  plan_low[{name}] p50 {np.median(plan_low[m]):.3f} "
                        f"p90 {np.percentile(plan_low[m], 90):.3f} "
                        f"hover_frac {(plan_low[m] > 0.05).mean():.2f}"
                    )
        if stages is not None:
            counts = np.bincount(max_stage, minlength=len(stages))
            stage_hist += counts
            print(
                "  stage: "
                + "  ".join(f"{name} {c / n:.2f}" for name, c in zip(stages, counts))
            )
            for name, m in (("timeout", stuck), ("failure", fail)):
                if m.any():
                    c = np.bincount(max_stage[m], minlength=len(stages))
                    stuck_hist[name] += c
                    print(
                        f"  stage|{name} (n={m.sum()}): "
                        + "  ".join(f"{s} {v / m.sum():.2f}" for s, v in zip(stages, c))
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
        for k, h in stuck_hist.items():
            if h.sum():
                print(
                    f"overall stage|{k} (n={h.sum()}): "
                    + "  ".join(f"{s} {v / h.sum():.3f}" for s, v in zip(stages, h))
                )
    if n_plan:
        print(f"plan latency: {1e3 * plan_t / n_plan:.1f} ms/replan ({n_plan} replans)")
    viewer.close()


if __name__ == "__main__":
    main()
