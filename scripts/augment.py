from dataclasses import dataclass, field, replace
from typing import Any

import draccus
import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.signal import lfilter, lfilter_zi
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
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
    carry_segment,
    encode_label,
    fk,
    held_segments,
    obj_slice,
    solve_seq,
    tpad,
    track_delta,
    transit_segments,
)
from flow_planning.envs import EnvConfig, FrankaConfig
from flow_planning.envs.env import PiperConfig
from flow_planning.kinematics import EE_FRAME, build_franka_chain, build_piper_chain
from flow_planning.selector import FrankaCollision, box_sdf
from flow_planning.utils import hf_column, quat_to_rot6d, rot6d_to_quat


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

    phi_range: tuple[float, float] = (0.15, 0.85)
    alphas: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    alpha_prob: float = 0.0
    alpha_tasks: list[int] = field(default_factory=list)
    n_theta: int = 4
    focus: bool = False
    task_ids: list[int] = field(default_factory=list)
    n_seg: int = 0
    orig_copies: int = 1
    retarget: float = 0.0
    retarget_place: float = 0.0
    retarget_only: float = 0.3
    ik_tol: float = 0.01
    vel_frac: float = 0.8
    ik_tol_aug: float = 0.05
    vel_frac_aug: float = 1.5
    hold: int = 20

    demogen: bool = False
    wall_h_min: float = 0.10
    wall_h_max: float = 0.30
    wall_w_min: float = 0.10
    wall_w_max: float = 0.20
    wall_margin: float = 0.12
    outside_margin: float = 0.08
    arm_margin: float = 0.01

    recover: int = 0
    recover_approach: int = 0
    recover_noise: tuple[float, float] = (0.02, 0.15)
    filter: bool = False
    postures: int = 0
    posture_range: tuple[float, float] = (0.3, 1.0)
    posture_gain: float = 0.2
    posture_min: float = 0.15
    track: bool = False
    track_tau: float = 0.25
    track_desc: float = 0.6
    track_h: tuple[float, float] = (0.05, 0.35)
    track_switches: int = 2
    track_zero: float = 0.3


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


def write(dst, obs, act, bend, task="pick_place", wall=None):
    bend = np.broadcast_to(np.asarray(bend, np.float32), (len(obs), np.shape(bend)[-1]))
    for f in range(len(obs)):
        frame = {
            "observation.state": obs[f],
            "action": act[f],
            "bend": bend[f].copy(),
            "task": task,
        }
        if wall is not None:
            frame["wall"] = np.asarray(wall, np.float32)
        dst.add_frame(frame)
    dst.save_episode()


def replay_libero(env, state0, act, hold, shift=None):
    env.reset()
    state0 = np.asarray(state0, np.float64).copy()
    m = env.envs[0].sim.model._model
    for name, delta in shift or []:
        jnt = m.body(f"{name}_main").jntadr[0]
        if jnt < 0:
            continue
        adr = m.jnt_qposadr[jnt]
        state0[1 + adr : 1 + adr + 2] += delta[:2]
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
        segs, held = held_segments(obs, act[:, 7], len(names))
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
                dbs=[np.zeros(3)] * len(segs),
                dps=[np.zeros(3)] * len(segs),
                ends=[np.zeros(3)] * len(segs),
            )
        )
    return demos


def sample_option(cfg, rng):
    thetas = np.linspace(0.0, np.pi, cfg.n_theta, endpoint=False)
    phi = lambda: float(rng.uniform(*cfg.phi_range))  # noqa: E731
    a = 0.0
    if len(cfg.alphas) > 1 and rng.random() < cfg.alpha_prob:
        a = float(rng.choice(cfg.alphas[1:]))
    if cfg.focus:
        half_pi = np.pi / 2
        app = (phi(), [0.0, 3 * np.pi / 4, half_pi][rng.integers(3)])
        tr = (phi(), half_pi) if rng.random() < 0.5 else (0.0, 0.0)
        return (a, *app, *tr)
    app = (phi(), rng.choice(thetas)) if rng.random() < 0.5 else (0.0, 0.0)
    tr = (phi(), rng.choice(thetas)) if rng.random() < 0.5 else (0.0, 0.0)
    return (a, *app, *tr)


def libero_candidates(cfg, demos, rng, per_demo):
    m = round(cfg.bend_margin * cfg.env.fps)
    cands: list[dict[str, Any]] = []
    for d, x in enumerate(demos):
        held_idx = [k for k, j in enumerate(x["held"]) if j is not None]
        if not held_idx:
            continue
        tries, mine = 0, 0
        while mine < per_demo and tries < 10 * per_demo:
            tries += 1
            shift = None
            xd = x
            if cfg.focus:
                pick = held_idx[rng.integers(len(held_idx))]
                combo = tuple(
                    sample_option(cfg, rng) if k == pick else SEG_ZERO
                    for k in range(len(x["held"]))
                )
                if cfg.retarget > 0:
                    delta = np.append(rng.uniform(-cfg.retarget, cfg.retarget, 2), 0.0)
                    r = cfg.retarget_place
                    dplace = np.append(rng.uniform(-r, r, 2), 0.0) if r else None
                    dbs = [np.zeros(3)] * len(x["segs"])
                    dps = [np.zeros(3)] * len(x["segs"])
                    dbs[pick] = delta
                    if dplace is not None:
                        dps[pick] = dplace
                    xd = dict(x) | {"dbs": dbs, "dps": dps}
                    shift = (pick, delta, dplace)
                    if rng.random() < cfg.retarget_only:
                        combo = tuple(SEG_ZERO for _ in x["held"])
            else:
                combo = tuple(
                    sample_option(cfg, rng) if j is not None else SEG_ZERO
                    for j in x["held"]
                )
            if shift is None and all(o == SEG_ZERO for o in combo):
                continue
            c = build(xd, combo, m)
            if c is not None:
                c["d"] = d
                c["shift"] = shift
                cands.append(c)
                mine += 1
    return cands


def held_combo(combo, held):
    return [o for o, j in zip(combo, held) if j is not None]


def posture_copies(cfg, demos, chain, base, fcol, rng, movable):
    m = round(cfg.bend_margin * cfg.env.fps)
    idx, psis, ws, tgts, seeds, ees, quats = [], [], [], [], [], [], []
    tm = max(len(x["act"]) for x in demos)
    for d, x in enumerate(demos):
        if not x["segs"]:
            continue
        T = len(x["act"])
        ramp_end = max(x["segs"][0][0] - m, 2)
        w = np.clip(np.arange(T) / ramp_end, 0.0, 1.0) * cfg.posture_gain
        for _ in range(3 * cfg.postures):
            psi = float(rng.choice([-1.0, 1.0])) * float(
                rng.uniform(*cfg.posture_range)
            )
            tgt = x["act"][:, ARM].copy()
            tgt[:, 2] += psi
            idx.append(d)
            psis.append(psi)
            ws.append(np.pad(w, (0, tm - T), mode="edge"))
            tgts.append(tpad(tgt, tm))
            seeds.append(tpad(x["act"][:, ARM], tm))
            ees.append(tpad(x["ee"], tm))
            quats.append(tpad(x["quat"], tm))
    if not idx:
        return {}, {"noseg": len(demos)}
    q_all = solve_seq(
        chain,
        np.stack(ees, axis=1),
        np.stack(quats, axis=1),
        np.stack(seeds, axis=1),
        cfg.ik_iters,
        cfg.ik_damping,
        base,
        ns=(np.stack(tgts, axis=1), np.stack(ws, axis=1)),
    )
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2
    out: dict[int, list] = {}
    why: dict[str, int] = {}
    for i, (d, psi) in enumerate(zip(idx, psis)):
        if len(out.get(d, [])) >= cfg.postures:
            continue
        x = demos[d]
        T = len(x["act"])
        act = x["act"]
        q = q_all[:T, i]
        pos, quat = fk(chain, q, base)
        err = np.linalg.norm(pos - x["ee"], axis=1).max()
        rot_err = R.from_quat(quat) * R.from_quat(x["quat"]).inv()
        assert isinstance(rot_err, R)
        if err > cfg.ik_tol or rot_err.magnitude().max() > np.radians(15):
            why["ik"] = why.get("ik", 0) + 1
            continue
        if np.abs(q - act[:, ARM]).max() < cfg.posture_min:
            why["dup"] = why.get("dup", 0) + 1
            continue
        if np.abs(np.diff(q, n=2, axis=0)).max() > jerk_limit:
            why["jerk"] = why.get("jerk", 0) + 1
            continue
        if np.abs(np.diff(q, axis=0)).max() > cfg.vel_frac * cfg.env.delta_max:
            why["vel"] = why.get("vel", 0) + 1
            continue
        body, _ = fcol.body_penetration(
            torch.as_tensor(q, dtype=torch.float32, device=DEV), None
        )
        if float(body.max()) > 0.0:
            why["self"] = why.get("self", 0) + 1
            continue
        a = act.copy()
        a[:, ARM] = q
        c = {
            "act": a.astype(np.float32),
            "objs": [
                x["obs"][:, obj_slice(j)].copy() if j is not None else None
                for j in x["held"]
            ],
            "shift": None,
        }
        o = kinematic_obs(x, c, chain, base, movable)
        out.setdefault(d, []).append(
            (o, c["act"], float(psi), float(np.abs(q - act[:, ARM]).max()))
        )
    return out, why


def kinematic_obs(x, c, chain, base, movable):
    obs, act = x["obs"], c["act"]
    q_obs = np.concatenate([obs[:1, ARM], act[:-1, ARM]])
    pos0, quat0 = fk(chain, obs[:, ARM], base)
    pos, quat = fk(chain, q_obs, base)
    r0, r = R.from_quat(quat0), R.from_quat(quat)
    out = obs.copy()
    out[:, ARM] = q_obs
    out[:, EE] = pos + r.apply(r0.inv().apply(obs[:, EE] - pos0))
    rel = r0.inv() * R.from_quat(rot6d_to_quat(obs[:, ROT]))
    assert isinstance(rel, R)
    rot = r * rel
    assert isinstance(rot, R)
    out[:, ROT] = quat_to_rot6d(rot.as_quat())
    shift = c.get("shift")
    if shift is not None and shift[2] is not None:
        for k, mv in enumerate(movable):
            if mv and k not in x["held"]:
                out[:, obj_slice(k)] += shift[2]
    for k, j in enumerate(x["held"]):
        if j is not None:
            out[:, obj_slice(j)] = c["objs"][k]
    return out.astype(np.float32)


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
        loose = any(o[0] for o in c["combo"])
        tol = cfg.ik_tol_aug if loose else cfg.ik_tol
        vf = cfg.vel_frac_aug if loose else cfg.vel_frac
        if err > tol or rot_err.magnitude().max() > np.radians(15):
            c["ok"], c["why"] = False, "ik"
        elif np.abs(np.diff(q, n=2, axis=0)).max() > jerk_limit:
            c["ok"], c["why"] = False, "jerk"
        elif np.abs(np.diff(q, axis=0)).max() > vf * cfg.env.delta_max:
            c["ok"], c["why"] = False, "vel"
        else:
            body, _ = fcol.body_penetration(
                torch.as_tensor(q, dtype=torch.float32, device=DEV), None
            )
            if float(body.max()) > 0.0:
                c["ok"], c["why"] = False, "self"


def recover_phase(cfg, x, chain, base, fcol, rng, lo, hi, e, span, j, count):
    obs, act = x["obs"], x["act"]
    fps = cfg.env.fps
    q_obs = obs[:, ARM]
    pos, quat = fk(chain, q_obs, base)
    speed = np.linalg.norm(np.diff(pos[span[0] : span[1]], axis=0), axis=1).mean() * fps
    z_min = pos[span[0] : span[1], 2].min() - 0.02
    jerk_limit = 0.6 * (50 / fps) ** 2
    out, why = [], {}
    tries = 0
    while len(out) < count and tries < 10 * count:
        tries += 1
        f = int(rng.integers(lo, hi))
        d = rng.normal(size=3)
        d *= rng.uniform(*cfg.recover_noise) / np.linalg.norm(d)
        p0 = pos[f] + d
        if p0[2] < z_min:
            why["low"] = why.get("low", 0) + 1
            continue
        n = max(3, int(round(np.linalg.norm(pos[e] - p0) / speed * fps)))
        u = np.arange(n) / n
        p_t = p0 + (pos[e] - p0) * u[:, None]
        r_f = R.from_quat(quat[f])
        key = R.from_quat(np.stack([quat[f], quat[e]]))
        slerp = Slerp([0.0, 1.0], key)
        q_t = slerp(u).as_quat()
        ref = q_obs[f] + (q_obs[e] - q_obs[f]) * u[:, None]
        pad = 10
        q_lin = solve_seq(
            chain,
            np.concatenate([np.repeat(p_t[:1], pad, 0), p_t])[:, None],
            np.concatenate([np.repeat(q_t[:1], pad, 0), q_t])[:, None],
            np.concatenate([np.repeat(ref[:1], pad, 0), ref])[:, None],
            cfg.ik_iters,
            cfg.ik_damping,
            base,
        )[pad:, 0]
        fpos, fquat = fk(chain, q_lin, base)
        rot_err = R.from_quat(fquat) * R.from_quat(q_t).inv()
        assert isinstance(rot_err, R)
        q_all = np.concatenate([q_lin, q_obs[e : e + 3]])
        if np.linalg.norm(fpos - p_t, axis=1).max() > cfg.ik_tol:
            why["ik"] = why.get("ik", 0) + 1
            continue
        if rot_err.magnitude().max() > np.radians(15):
            why["rot"] = why.get("rot", 0) + 1
            continue
        if np.abs(np.diff(q_all, n=2, axis=0)).max() > jerk_limit:
            why["jerk"] = why.get("jerk", 0) + 1
            continue
        if np.abs(np.diff(q_all, axis=0)).max() > cfg.vel_frac * cfg.env.delta_max:
            why["vel"] = why.get("vel", 0) + 1
            continue
        body, _ = fcol.body_penetration(
            torch.as_tensor(q_lin, dtype=torch.float32, device=DEV), None
        )
        if float(body.max()) > 0.0:
            why["self"] = why.get("self", 0) + 1
            continue
        r_t = R.from_quat(fquat)
        assert isinstance(r_t, R)
        off_ee = r_f.inv().apply(obs[f, EE] - pos[f])
        rel_ee = r_f.inv() * R.from_quat(rot6d_to_quat(obs[f : f + 1, ROT]))
        assert isinstance(rel_ee, R)
        seg = np.repeat(obs[e : e + 1], n, 0)
        seg[:, ARM] = q_lin
        seg[:, 7:9] = obs[f, 7:9]
        seg[:, EE] = fpos + r_t.apply(np.repeat(off_ee[None], n, 0))
        rot_ee_t = r_t * rel_ee
        assert isinstance(rot_ee_t, R)
        seg[:, ROT] = quat_to_rot6d(rot_ee_t.as_quat())
        if j is not None:
            off_obj = r_f.inv().apply(obs[f, obj_slice(j)] - pos[f])
            rot_obj = slice(obj_slice(j).stop, obj_slice(j).stop + 6)
            rel_obj = r_f.inv() * R.from_quat(rot6d_to_quat(obs[f : f + 1, rot_obj]))
            assert isinstance(rel_obj, R)
            seg[:, obj_slice(j)] = fpos + r_t.apply(np.repeat(off_obj[None], n, 0))
            rot_obj_t = r_t * rel_obj
            assert isinstance(rot_obj_t, R)
            seg[:, rot_obj] = quat_to_rot6d(rot_obj_t.as_quat())
        a = np.repeat(act[f : f + 1], n, 0)
        a[:, ARM] = np.concatenate([q_lin[1:], q_obs[e : e + 1]])
        out.append(
            (
                np.concatenate([seg, obs[e:]]).astype(np.float32),
                np.concatenate([a, act[e:]]).astype(np.float32),
                n,
                float(np.linalg.norm(d)),
            )
        )
    return out, why


def recover_segments(cfg, x, chain, base, fcol, rng, n_obj):
    if not x["segs"]:
        return [], {"noseg": 1}
    (c, o), j = x["segs"][0], x["held"][0]
    m = int(round(cfg.bend_margin * cfg.env.fps))
    out, why = [], {}
    phases = []
    if cfg.recover and o - m - (c + m) >= 2 and j is not None:
        phases.append((c + m, o - m, o - m, (c, o), j, cfg.recover))
    if cfg.recover_approach and c - m >= 2:
        phases.append((0, c - m, c - m, (0, c), None, cfg.recover_approach))
    for lo, hi, e, span, held, count in phases:
        segs, w = recover_phase(
            cfg, x, chain, base, fcol, rng, lo, hi, e, span, held, count
        )
        out += segs
        for k, v in w.items():
            why[k] = why.get(k, 0) + v
    return out, why


def augment_libero(cfg):
    from flow_planning.envs.libero import LiberoConfig, LiberoEnv

    assert isinstance(cfg.env, LiberoConfig)
    rng = np.random.default_rng(cfg.seed)
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)
    dst = None
    for task_id in cfg.task_ids or [cfg.env.task_id]:
        env = LiberoEnv(
            replace(cfg.env, task_id=task_id, world_count=1, obstacle=False)
        )
        base = np.asarray(env.robot_base_pos, np.float32)
        fcol = FrankaCollision(DEV, base, 0.0)
        demos = libero_demos(cfg, env, chain, base, cfg.limit)
        mj = env.envs[0].sim.model._model
        movable = [mj.body(f"{n}_main").jntadr[0] >= 0 for n in env.objects]
        if cfg.n_seg:
            kept = [x for x in demos if len(x["segs"]) <= cfg.n_seg]
            print(
                f"task {task_id}: dropped {len(demos) - len(kept)} demos "
                f"with >{cfg.n_seg} grasps"
            )
            demos = kept
        n_seg = cfg.n_seg or max(sum(j is not None for j in x["held"]) for x in demos)
        ldim = LABEL_DIM * n_seg + (1 if cfg.postures else 0)
        print(
            f"task {task_id}: {len(demos)} demos, {n_seg} grasp segments, "
            f"label dim {LABEL_DIM * n_seg}"
        )
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": demos[0]["obs"].shape[1:],
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (8,), "names": None},
            "bend": {"dtype": "float32", "shape": (ldim,), "names": None},
        }
        if dst is None:
            dst = LeRobotDataset.create(
                repo_id=cfg.dst_repo,
                fps=cfg.env.fps,
                features=features,
                use_videos=False,
            )
            shapes = features
        assert features == shapes, (task_id, features, shapes)
        for x in demos:
            for _ in range(cfg.orig_copies):
                write(dst, x["obs"], x["act"], np.zeros(ldim), env.language)
        if cfg.recover or cfg.recover_approach:
            n_rec, lens, mags = 0, [], []
            why_all: dict[str, int] = {}
            for x in tqdm(demos, desc="recover"):
                segs, why = recover_segments(
                    cfg, x, chain, base, fcol, rng, len(env.objects)
                )
                for k, v in why.items():
                    why_all[k] = why_all.get(k, 0) + v
                for o, a, n, mag in segs:
                    write(dst, o, a, np.zeros(ldim), env.language)
                    n_rec += 1
                    lens.append(n)
                    mags.append(mag)
            print(
                f"task {task_id}: {n_rec} recovery segments "
                f"({n_rec / max(len(demos), 1):.1f}/demo), rejected {why_all}, "
                f"linear frames p50 {np.median(lens) if lens else 0:.0f} "
                f"max {max(lens) if lens else 0}, |noise| mean "
                f"{np.mean(mags) if mags else 0:.3f}"
            )
        if cfg.postures:
            n_pos = 0
            mags = []
            by_demo, why_pos = posture_copies(
                cfg, demos, chain, base, fcol, rng, movable
            )
            for segs_p in by_demo.values():
                for o, a, psi, mag in segs_p:
                    label = np.zeros(ldim, np.float32)
                    label[-1] = psi
                    write(dst, o, a, label, env.language)
                    n_pos += 1
                    mags.append(mag)
            print(
                f"task {task_id}: {n_pos} posture copies "
                f"({n_pos / max(len(demos), 1):.1f}/demo), rejected {why_pos}, "
                f"|dq| max mean {np.mean(mags) if mags else 0:.2f}"
            )
        if cfg.copies == 0:
            del env
            continue

        tcfg = cfg
        if cfg.alpha_tasks and task_id not in cfg.alpha_tasks:
            tcfg = replace(cfg, alpha_prob=0.0)
        cands = libero_candidates(tcfg, demos, rng, 4 * cfg.copies)
        print(f"{len(cands)} candidates")
        libero_solve(cfg, cands, demos, chain, base, fcol)
        written = 0
        why: dict[str, int] = {}
        labels = []
        per_demo = np.zeros(len(demos), int)
        for c in tqdm(cands, desc="copies"):
            if not c["ok"]:
                why[c["why"]] = why.get(c["why"], 0) + 1
                continue
            if per_demo[c["d"]] >= cfg.copies:
                continue
            x = demos[c["d"]]
            if cfg.filter:
                shift = c.get("shift")
                if shift is not None:
                    seg, delta, dplace = shift
                    held_name = env.objects[x["held"][seg]]
                    shift = [(held_name, delta)]
                    if dplace is not None:
                        shift += [(n, dplace) for n in env.objects if n != held_name]
                succ, _ = replay_libero(env, x["state0"], c["act"], cfg.hold, shift)
                if not succ:
                    why["replay"] = why.get("replay", 0) + 1
                    continue
            obs = kinematic_obs(x, c, chain, base, movable)
            label = np.zeros(ldim, np.float32)
            label[: LABEL_DIM * n_seg] = encode_label(
                held_combo(c["combo"], x["held"]), n_seg
            )
            write(dst, obs, c["act"], label, env.language)
            labels.append(label)
            written += 1
            per_demo[c["d"]] += 1
        print(f"wrote {written}/{len(cands)} copies; rejected {why}")
        if labels:
            lab = np.stack(labels)
            print("label nonzero fraction per dim:", (np.abs(lab) > 0).mean(0).round(2))
        del env
    assert dst is not None
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


def sample_track(x, rng, cfg, margin):
    s0, s1 = x["close"] + margin, x["opened"] - margin
    tau, desc = cfg.track_tau * cfg.env.fps, cfg.track_desc * cfg.env.fps
    lo, hi = int(s0 + 2 * tau), int(s1 - desc - 2 * tau)
    k = int(rng.integers(cfg.track_switches + 1)) if hi > lo else 0
    switches = sorted(rng.integers(lo, hi, k).tolist())
    targets = []
    for j in range(k + 1):
        if j and rng.random() < cfg.track_zero:
            targets.append(np.zeros(2))
        else:
            h, th = rng.uniform(*cfg.track_h), rng.uniform(0.0, np.pi)
            targets.append(h * np.array([np.cos(th), np.sin(th)]))
    return track_delta(x["obs"][:, EE], s0, s1, targets, switches, tau, desc)


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


def augment_piper(cfg):
    # ponytail: carry-phase bends only, IK error + jerk as the only filters;
    # phase-wide bends and a self-collision check are the upgrades
    assert isinstance(cfg.env, PiperConfig)
    rng = np.random.default_rng(cfg.seed)
    chain = build_piper_chain(DEV)
    base = np.zeros(3, np.float32)
    arm = slice(0, 6)

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
    lo, hi = chain.get_joint_limits()
    q_all = np.radians(act_all[:, arm])
    limits = (
        np.minimum(lo, q_all.min(0)).tolist(),
        np.maximum(hi, q_all.max(0)).tolist(),
    )

    todo: list[dict[str, Any]] = []
    grid = np.zeros(len(ALPHAS))
    margin = round(cfg.bend_margin * src.fps)
    n_src = min(src.num_episodes, cfg.limit or src.num_episodes)
    for e in tqdm(range(n_src), desc="originals"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        write(dst, obs, act, np.zeros(2))
        grid += track_error(obs[:, arm], act[:, arm]) / n_src
        seg = carry_segment(act[:, 6], cfg.env.gripper_closed)
        if seg is None or seg[1] - seg[0] < 2 * margin:
            continue
        ee, quat = fk(chain, np.radians(act[:, arm]), base)
        obs_ee = fk(chain, np.radians(obs[:, arm]), base)[0]
        todo.append(
            dict(
                obs=obs,
                act=act,
                close=seg[0],
                opened=seg[1],
                ee=ee,
                quat=quat,
                obs_ee=obs_ee,
            )
        )

    j = int(grid.argmin())
    alpha = float(ALPHAS[j])
    print(f"tracking fit: alpha {alpha:.2f}")
    print(f"  joint MAE {grid[j]:.3f} deg vs {grid[0]:.3f} unfiltered")
    print(f"{len(todo)}/{n_src} episodes bendable")

    ee_err: list[np.ndarray] = []
    written = rej_ik = 0
    jerk_limit = 0.6 * (50 / src.fps) ** 2
    pending = [dict(x) for x in todo for _ in range(cfg.copies)]
    bar = tqdm(total=len(pending), desc="bends")
    for attempt in range(4):
        if not pending:
            break
        bar.write(f"attempt {attempt}: {len(pending)} pending")
        phis = np.pi * rng.uniform(cfg.bend_min, cfg.bend_max, len(pending))
        thetas = rng.uniform(0.0, np.pi, len(pending))
        for x, phi, theta in zip(pending, phis, thetas):
            d = bend_delta(
                x["obs_ee"], x["close"] + margin, x["opened"] - margin, phi, theta
            )
            x["label"] = phi * np.array([np.cos(theta), np.sin(theta)])
            x["ee_t"] = x["ee"] + d
            x["bent"] = np.linalg.norm(d, axis=1) > 1e-6

        tm = max(len(x["act"]) for x in pending)
        na_j = solve_seq(
            chain,
            np.stack([tpad(x["ee_t"], tm) for x in pending], axis=1),
            np.stack([tpad(x["quat"], tm) for x in pending], axis=1),
            np.radians(np.stack([tpad(x["act"][:, arm], tm) for x in pending], axis=1)),
            cfg.ik_iters,
            cfg.ik_damping,
            base,
            limits,
        )
        achieved = fk(chain, na_j.reshape(-1, 6), base)[0].reshape(tm, len(pending), 3)
        rejected = []
        for i, x in enumerate(pending):
            nf = len(x["act"])
            err = np.linalg.norm(achieved[:nf, i] - x["ee_t"], axis=1)
            jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
            if err.max() > cfg.ik_tol or jerk > jerk_limit:
                rejected.append(x)
                rej_ik += 1
                continue
            na = x["act"].copy()
            na[:, arm] = np.degrees(na_j[:nf, i])
            new_obs = x["obs"].copy()
            new_obs[:, arm] += smooth(na[:, arm] - x["act"][:, arm], alpha)
            write(dst, new_obs.astype(np.float32), na.astype(np.float32), x["label"])
            written += 1
            bar.update(1)
            if x["bent"].any():
                ee_err.append(err[x["bent"]])
        pending = rejected
    bar.close()

    total = written + len(pending)
    print(f"wrote {written}/{total} copies ({len(pending)} dropped after retries)")
    print(f"rejected: {rej_ik} on IK/jerk")
    if ee_err:
        err = np.concatenate(ee_err)
        print(
            f"bent-frame IK error: mean {err.mean():.4f} "
            f"p95 {np.percentile(err, 95):.4f}"
        )
    dst.finalize()


@draccus.wrap()
def main(cfg: Config):
    if isinstance(cfg.env, PiperConfig):
        augment_piper(cfg)
        return
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
        "wall": {"dtype": "float32", "shape": (6,), "names": None},
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
        write(dst, obs, act, np.zeros(2), wall=np.zeros(6))
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
            for x in active:
                if cfg.track:
                    d, x["label"] = sample_track(x, rng, cfg, margin)
                else:
                    phi = np.pi * rng.uniform(cfg.bend_min, cfg.bend_max)
                    theta = rng.uniform(0.0, np.pi)
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
            wall = np.concatenate(x["wall"]) if cfg.demogen else np.zeros(6)
            write(dst, new_obs, na, x["label"], wall=wall)
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
