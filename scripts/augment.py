from dataclasses import dataclass, field, replace
from typing import Any

import draccus
import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.signal import lfilter, lfilter_zi
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from flow_planning.bend import (
    ARM,
    CUBE,
    EE,
    LABEL_DIM,
    ROT,
    SEG_ZERO,
    bend_delta,
    build,
    encode_label,
    fk,
    grasp_segments,
    held_object,
    seg_options,
    solve_seq,
    tpad,
    transit_segments,
)
from flow_planning.envs import EnvConfig, FrankaConfig
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision, box_sdf
from flow_planning.utils import hf_column, quat_to_rot6d


@dataclass
class Config:
    src_repo: str = "reece-omahoney/franka-src"
    dst_repo: str = "reece-omahoney/franka"
    copies: int = 2  # bent copies per episode; the original is always kept
    bend_min: float = 0.15  # half-angle, units of pi
    bend_max: float = 1.0
    bend_margin: float = 1.5
    limit: int = 0
    ik_iters: int = 8  # per frame, warm-started
    ik_damping: float = 0.05
    seed: int = 0
    env: EnvConfig = field(default_factory=FrankaConfig)  # table geometry only

    phis: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    n_theta: int = 4
    alphas: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    arcs: str = "additive,replace"
    ik_tol: float = 0.01
    vel_frac: float = 0.8
    hold: int = 20

    demogen: bool = False
    wall_h_min: float = 0.10
    wall_h_max: float = 0.30
    wall_w_min: float = 0.10
    wall_w_max: float = 0.20
    wall_margin: float = 0.12
    outside_margin: float = 0.08
    arm_margin: float = 0.01


DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.linspace(0.0, 0.98, 25)


def held_block(obs, close, opened, n_obj) -> slice:
    ee = obs[close:opened, EE] - obs[close, EE]
    errs = []
    for k in range(n_obj):
        blk = slice(CUBE.start + 9 * k, CUBE.start + 9 * k + 3)
        errs.append(np.abs(obs[close:opened, blk] - obs[close, blk] - ee).mean())
    k = int(np.argmin(errs))
    return slice(CUBE.start + 9 * k, CUBE.start + 9 * k + 3)


def write(dst, obs, act, bend, task="pick_place"):
    for f in range(len(obs)):
        dst.add_frame(
            {
                "observation.state": obs[f],
                "action": act[f],
                "bend": bend.astype(np.float32),
                "task": task,
            }
        )
    dst.save_episode()


def replay_libero(env, state0, act, hold):
    env.reset()
    env.obs[0] = env.envs[0].set_init_state(state0)
    env.succ[:] = False
    obs = []
    for a in act:
        obs.append(env.get_obs()[0])
        env.apply_action(a[None])
    for _ in range(hold):
        env.apply_action(act[-1][None])
    return bool(env.succ[0]), np.stack(obs)


def libero_demos(cfg, env, chain, base, limit):
    import h5py
    from huggingface_hub import hf_hub_download

    from flow_planning.envs.libero import (
        HF_DEMOS,
        SOURCE_SUITE,
        body_aabb,
        demo_to_episode,
    )

    names = env.objects
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[cfg.env.suite]}/{env.name}_demo.hdf5",
        repo_type="dataset",
    )
    f = h5py.File(path, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    demos: list[dict[str, Any]] = []
    for k in tqdm(keys[: limit or len(keys)], desc="demos"):
        obs, act = demo_to_episode(env, f["data"][k])
        segs = grasp_segments(act[:, 7])
        held = [held_object(obs, c, o, len(names)) for c, o in segs]
        ee, quat = fk(chain, act[:, ARM], base)
        env.set_state(0, f["data"][k]["states"][0])
        half = [
            body_aabb(env.envs[0], names[j])[3:5] if j is not None else np.ones(2)
            for j in held
        ]
        demos.append(
            dict(
                key=k,
                state0=f["data"][k]["states"][0],
                obs=obs,
                act=act,
                segs=segs,
                held=held,
                ee=ee,
                quat=quat,
                half=half,
                dyaw=[0.0] * len(segs),
                dbs=[np.zeros(3)] * len(segs),
                dps=[np.zeros(3)] * len(segs),
                ends=[np.zeros(3)] * len(segs),
            )
        )
    return demos


def libero_candidates(cfg, demos, rng, per_demo):
    m = round(cfg.bend_margin * cfg.env.fps)
    modes = cfg.arcs.split(",")
    options = seg_options(cfg.alphas, cfg.phis, cfg.n_theta, modes)
    cands: list[dict[str, Any]] = []
    for d, x in enumerate(demos):
        tries, mine = 0, 0
        while mine < per_demo and tries < 10 * per_demo:
            tries += 1
            combo = tuple(
                options[rng.integers(len(options))] if j is not None else SEG_ZERO
                for j in x["held"]
            )
            if all(o == SEG_ZERO for o in combo):
                continue
            c = build(x, combo, m, modes)
            if c is not None:
                c["d"] = d
                cands.append(c)
                mine += 1
    return cands


def libero_solve(cfg, cands, demos, chain, base, fcol):
    """IK + feasibility gate; sets c["act"] / c["ok"] / c["why"] in place."""
    tm = max(len(x["act"]) for x in demos)
    q_all = solve_seq(
        chain,
        np.stack([tpad(c["ee_t"], tm) for c in cands], axis=1),
        np.stack([tpad(c["quat"], tm) for c in cands], axis=1),
        np.stack([demos[c["d"]]["act"][0, ARM] for c in cands]),
        cfg.ik_iters,
        cfg.ik_damping,
        base,
    )
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2
    for i, c in enumerate(cands):
        x = demos[c["d"]]
        nf = len(x["act"])
        q = q_all[:nf, i]
        pos, quat = fk(chain, q, base)
        err = np.linalg.norm(pos - c["ee_t"], axis=1).max()
        rot_err = R.from_quat(quat) * R.from_quat(c["quat"]).inv()
        assert isinstance(rot_err, R)
        act = x["act"].copy()
        act[:, ARM] = q
        c["act"] = act.astype(np.float32)
        c["ok"], c["why"] = True, ""
        if err > cfg.ik_tol or rot_err.magnitude().max() > np.radians(15):
            c["ok"], c["why"] = False, "ik"
        elif np.abs(np.diff(q, n=2, axis=0)).max() > jerk_limit:
            c["ok"], c["why"] = False, "jerk"
        elif np.abs(np.diff(q, axis=0)).max() > cfg.vel_frac * cfg.env.delta_max:
            c["ok"], c["why"] = False, "vel"
        else:
            body, _ = fcol.body_penetration(
                torch.as_tensor(q, dtype=torch.float32, device=DEV), None
            )
            if float(body.max()) > 0.0:
                c["ok"], c["why"] = False, "self"


def augment_libero(cfg):
    from flow_planning.envs.libero import LiberoConfig, LiberoEnv

    assert isinstance(cfg.env, LiberoConfig)
    rng = np.random.default_rng(cfg.seed)
    env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
    base = np.asarray(env.robot_base_pos, np.float32)
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    demos = libero_demos(cfg, env, chain, base, cfg.limit)
    n_seg = max(len(x["segs"]) for x in demos)
    print(f"{len(demos)} demos, {n_seg} grasp segments, label dim {LABEL_DIM * n_seg}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": demos[0]["obs"].shape[1:],
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (8,), "names": None},
        "bend": {"dtype": "float32", "shape": (LABEL_DIM * n_seg,), "names": None},
    }
    dst = LeRobotDataset.create(
        repo_id=cfg.dst_repo, fps=cfg.env.fps, features=features, use_videos=False
    )
    for x in demos:
        write(dst, x["obs"], x["act"], np.zeros(LABEL_DIM * n_seg), env.language)

    cands = libero_candidates(cfg, demos, rng, 4 * cfg.copies)
    print(f"{len(cands)} candidates")
    libero_solve(cfg, cands, demos, chain, base, fcol)
    written = rej_replay = 0
    why: dict[str, int] = {}
    labels = []
    per_demo = np.zeros(len(demos), int)
    for c in tqdm(cands, desc="replay"):
        if not c["ok"]:
            why[c["why"]] = why.get(c["why"], 0) + 1
            continue
        if per_demo[c["d"]] >= cfg.copies:
            continue
        x = demos[c["d"]]
        succ, obs = replay_libero(env, x["state0"], c["act"], cfg.hold)
        if not succ:
            rej_replay += 1
            continue
        label = encode_label(c["combo"], n_seg)
        write(dst, obs.astype(np.float32), c["act"], label, env.language)
        labels.append(label)
        written += 1
        per_demo[c["d"]] += 1
    print(f"wrote {written}/{len(cands)} copies; rejected {why}, {rej_replay} replay")
    if labels:
        lab = np.stack(labels)
        print("label nonzero fraction per dim:", (np.abs(lab) > 0).mean(0).round(2))
    dst.finalize()


def sdf_np(p, center, half):
    q = np.abs(p - center) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)


def plan_wall_bend(x, rng, cfg, margin):
    s0, s1 = x["close"] + margin, x["opened"] - margin
    obs_ee, act_ee = x["obs"][:, EE], x["ee"]
    outside = np.concatenate([act_ee[: x["close"]], act_ee[x["opened"] :]])
    phis = np.pi * np.concatenate([[0.0], np.linspace(cfg.bend_min, cfg.bend_max, 18)])
    for _ in range(20):
        h = rng.uniform(cfg.wall_h_min, cfg.wall_h_max)
        w = rng.uniform(cfg.wall_w_min, cfg.wall_w_max)
        center = np.array([0.0, -0.5, cfg.env.table_height + 0.5 * h])
        half = np.array([w, cfg.env.obstacle_thickness, 0.5 * h])
        if sdf_np(outside, center, half).min() < cfg.outside_margin:
            continue
        theta = rng.uniform(0.0, np.pi)
        for j, phi in enumerate(phis):
            d = bend_delta(obs_ee, s0, s1, phi, theta)
            p = (act_ee + d)[x["close"] : x["opened"]]
            if sdf_np(p, center, half).min() > cfg.wall_margin:
                k = min(j + int(rng.integers(0, 3)), len(phis) - 1)
                dk = bend_delta(obs_ee, s0, s1, phis[k], theta)
                pk = (act_ee + dk)[x["close"] : x["opened"]]
                if sdf_np(pk, center, half).min() > cfg.wall_margin:
                    phi = phis[k]
                return (center, half), np.array([h, w], np.float32), phi, theta
    return None


def smooth(a, alpha):
    b, p = [1 - alpha], [1, -alpha]
    return lfilter(b, p, a, axis=0, zi=lfilter_zi(b, p)[:, None] * a[:1])[0]


def track_error(obs, act):
    o, a = obs[:, ARM], act[:, ARM]
    return np.array([np.abs(smooth(a, al) - o).mean() for al in ALPHAS])


def lag_states(x, chain, base, alpha):
    # add the filtered action delta onto the recorded arm joints
    obs = x["obs"]
    dq = smooth(x["na"][:, ARM] - x["act"][:, ARM], alpha)
    new = obs.copy()
    new[:, ARM] += dq
    ee, quat = fk(chain, new[:, ARM], base)
    new[:, EE], new[:, ROT] = ee, quat_to_rot6d(quat)

    # shift the cube by the EE displacement over the frames it is held
    held = slice(x["close"], x["opened"])
    new[held, x["blk"]] += ee[held] - obs[held, EE]
    return new.astype(np.float32)


@draccus.wrap()
def main(cfg: Config):
    if not isinstance(cfg.env, FrankaConfig):
        augment_libero(cfg)
        return
    rng = np.random.default_rng(cfg.seed)

    # build kinematic chain
    n_obj = None
    if isinstance(cfg.env, FrankaConfig):
        base = np.array([-0.5, -0.5, cfg.env.table_height], np.float32)
        cube_size = cfg.env.cube_size
    else:
        assert not cfg.demogen, "demogen bends need the franka wall geometry"
        from flow_planning.envs.libero import LiberoConfig, LiberoEnv

        assert isinstance(cfg.env, LiberoConfig)
        lenv = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
        base = np.asarray(lenv.robot_base_pos, np.float32)
        cube_size = 0.0
        n_obj = len(lenv.objects)
        del lenv
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)

    # load datasets
    src = LeRobotDataset(cfg.src_repo)
    hf = src.hf_dataset
    epi = hf_column(hf, "episode_index")
    obs_all = hf_column(hf, "observation.state")
    act_all = hf_column(hf, "action")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": obs_all.shape[1:],
            "names": None,
        },
        "action": {"dtype": "float32", "shape": act_all.shape[1:], "names": None},
        "bend": {"dtype": "float32", "shape": (2,), "names": None},
    }

    dst = LeRobotDataset.create(
        repo_id=cfg.dst_repo, fps=src.fps, features=features, use_videos=False
    )

    # write originals, FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    grid = np.zeros(len(ALPHAS))
    margin = round(cfg.bend_margin * src.fps)
    n_src = min(src.num_episodes, cfg.limit or src.num_episodes)
    for e in tqdm(range(n_src), desc="originals"):
        # write original episode
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        write(dst, obs, act, np.zeros(2))
        grid += track_error(obs, act) / n_src

        # fk bendable episodes
        if opened - close < 2 * margin:
            continue
        act_ee, act_quat = fk(chain, act[:, :7], base)
        blk = CUBE if n_obj is None else held_block(obs, close, opened, n_obj)
        todo.append(
            dict(
                obs=obs,
                act=act,
                close=close,
                opened=opened,
                ee=act_ee,
                quat=act_quat,
                blk=blk,
            )
        )

    # take the alpha with the lowest prediction error
    j = int(grid.argmin())
    alpha = float(ALPHAS[j])
    print(f"tracking fit: alpha {alpha:.2f}")
    print(f"  joint MAE {grid[j]:.5f} rad vs {grid[0]:.5f} unfiltered")
    if j in (0, len(ALPHAS) - 1):
        print("  WARNING: optimum on a grid edge")

    # report the same fit in EE space, against doing no filtering at all
    for name, a_ in (("unfiltered", 0.0), ("fitted", alpha)):
        d = np.concatenate(
            [
                fk(chain, smooth(x["act"][:, ARM], a_), base)[0] - x["obs"][:, EE]
                for x in todo
            ]
        )
        print(f"  {name:>10} EE MAE {1e3 * np.linalg.norm(d, axis=1).mean():.1f} mm")

    # queue up `copies` bends per episode and retry the rejects up to 4 times
    ee_err: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    written = rej_ik = rej_plan = rej_wall = rej_body = 0
    fcol = FrankaCollision(DEV, base, cube_size)
    jerk_limit = 0.6 * (50 / src.fps) ** 2
    base_t = torch.as_tensor(base, dtype=torch.float32, device=DEV)
    fcol.calibrate_self(
        torch.as_tensor(obs_all[::50, :7], dtype=torch.float32, device=DEV)
    )
    pending = [dict(x) for x in todo for _ in range(cfg.copies)]
    bar = tqdm(total=len(pending), desc="bends")
    for attempt in range(4):
        rejected = []
        bar.write(f"attempt {attempt}: {len(pending)} pending")

        # sample arcs
        if cfg.demogen:
            active = []
            for x in pending:
                plan = plan_wall_bend(x, rng, cfg, margin)
                if plan is None:
                    rej_plan += 1
                    continue
                x["wall"], x["label"], phi, theta = plan
                d = bend_delta(
                    x["obs"][:, EE],
                    x["close"] + margin,
                    x["opened"] - margin,
                    phi,
                    theta,
                )
                x["ee_t"] = x["ee"] + d
                x["bent"] = np.linalg.norm(d, axis=1) > 1e-6
                active.append(x)
            if not active:
                pending = rejected
                if not pending:
                    break
                continue
        else:
            active = pending
            phis = np.pi * rng.uniform(cfg.bend_min, cfg.bend_max, len(active))
            thetas = rng.uniform(0.0, np.pi, len(active))
            for x, phi, theta in zip(active, phis, thetas):
                d = bend_delta(
                    x["obs"][:, EE],
                    x["close"] + margin,
                    x["opened"] - margin,
                    phi,
                    theta,
                )
                x["label"] = phi * np.array([np.cos(theta), np.sin(theta)])
                x["ee_t"] = x["ee"] + d
                x["bent"] = np.linalg.norm(d, axis=1) > 1e-6

        # solve IK to get joints
        tm = max(len(x["act"]) for x in active)
        na_j = solve_seq(
            chain,
            np.stack([tpad(x["ee_t"], tm) for x in active], axis=1),
            np.stack([tpad(x["quat"], tm) for x in active], axis=1),
            np.stack([x["act"][0, :7] for x in active]),
            cfg.ik_iters,
            cfg.ik_damping,
            base,
        )

        # measure what the IK actually achieved and drop copies that miss or jerk
        achieved = fk(chain, na_j.reshape(-1, 7), base)[0].reshape(tm, len(active), 3)
        for i, x in enumerate(active):
            nf = len(x["act"])
            err = np.linalg.norm(achieved[:nf, i] - x["ee_t"], axis=1)
            jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
            if err.max() > 0.05 or jerk > jerk_limit:
                rejected.append(x)
                rej_ik += 1
                continue
            # splice the new arm joints in over the original gripper channel
            na = x["act"].copy()
            na[:, :7] = na_j[:nf, i]
            x["na"] = na.astype(np.float32)
            new_obs = lag_states(x, chain, base, alpha)

            # reject copies whose arm folds into itself or the held cube
            q = torch.as_tensor(na_j[:nf, i], dtype=torch.float32, device=DEV)
            cube = None
            if cube_size > 0:
                cube = torch.as_tensor(new_obs[:, CUBE], device=DEV) - base_t
            body, pts = fcol.body_penetration(q, cube)
            if float(body.max()) > 0.0:
                rejected.append(x)
                rej_body += 1
                continue
            if cfg.demogen:
                wc, wh = (
                    torch.as_tensor(v, dtype=torch.float32, device=DEV)
                    for v in x["wall"]
                )
                pen = fcol.radius + cfg.arm_margin - box_sdf(pts + base_t, wc, wh)
                if float(pen.max()) > 0.0:
                    rejected.append(x)
                    rej_wall += 1
                    continue
            write(dst, new_obs, na, x["label"])
            labels.append(x["label"])
            written += 1
            bar.update(1)
            if x["bent"].any():
                ee_err.append(err[x["bent"]])
        pending = rejected
        if not pending:
            break
    bar.close()

    total = written + len(pending) + rej_plan
    print(f"wrote {written}/{total} copies ({len(pending)} dropped after retries)")
    print(
        f"rejected: {rej_ik} on IK/jerk, {rej_body} on self/cube collision, "
        f"{rej_wall} on arm-wall clearance"
    )
    print(f"dropped: {rej_plan} with no plannable wall")
    if cfg.demogen and labels:
        lab = np.stack(labels)
        for name, col in zip(("wall_h", "wall_w"), lab.T):
            q = np.percentile(col, [0, 25, 50, 75, 100])
            print(f"  {name}: " + " ".join(f"{v:.3f}" for v in q))
    if ee_err:
        err = np.concatenate(ee_err)
        print(
            f"bent-frame IK error: mean {err.mean():.4f} "
            f"p95 {np.percentile(err, 95):.4f}"
        )
    dst.finalize()


if __name__ == "__main__":
    main()
