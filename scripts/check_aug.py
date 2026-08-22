from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

import draccus
import h5py
import numpy as np
import pytorch_kinematics as pk
import torch
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from flow_planning.bend import ARM, EE, bend_delta, fk, grasp_segments, solve_seq, tpad
from flow_planning.envs import EnvConfig
from flow_planning.envs.libero import (
    HF_DEMOS,
    SOURCE_SUITE,
    LiberoConfig,
    LiberoEnv,
    body_aabb,
    demo_to_episode,
    obstacle_geom_boxes,
)
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision, box_sdf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OBJ0 = 18
ZERO = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    n_inits: int = 10
    n_demos: int = 3
    margin: float = 0.3
    phis: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    n_theta: int = 4
    alphas: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    top_k: int = 4
    replay_max: int = 12
    ik_iters: int = 8
    ik_damping: float = 0.05
    hold: int = 20
    verbose: bool = True


def obj_slice(k):
    return slice(OBJ0 + 9 * k, OBJ0 + 9 * k + 3)


def held_object(obs, close, opened, n_obj):
    ee = obs[close:opened, EE] - obs[close, EE]
    best, best_err = None, 0.02
    for k in range(n_obj):
        p = obs[close:opened, obj_slice(k)] - obs[close, obj_slice(k)]
        err = np.abs(p - ee).mean()
        if err < best_err and np.linalg.norm(p[-1]) > 0.02:
            best, best_err = k, err
    return best


def target_object(goals, objects, name):
    for _, a, b in goals:
        if a != name:
            continue
        hits = [o for o in objects if b.startswith(o) and o != name]
        if hits:
            return objects.index(max(hits, key=len))
    return None


def shift_path(T, segs, m, dbs, dps):
    knots, vals = [0], [np.zeros(3)]
    for (c, o), db, dp in zip(segs, dbs, dps):
        for t, v in [(c - m, db), (c + m, db), (o - m, dp), (o + m, dp)]:
            knots.append(t)
            vals.append(v)
    knots.append(T - 1)
    vals.append(vals[-1])
    k = np.array(knots)
    for i in range(1, len(k)):
        k[i] = max(k[i], k[i - 1] + 1)
    keep = k < T
    k, v = k[keep], np.stack(vals)[keep]
    t = np.arange(T)
    return np.stack([np.interp(t, k, v[:, j]) for j in range(3)], axis=1)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def grasp_rotation(ee, quat, w0, close, opened, w1, m, centre, alpha):
    T = len(ee)
    a1 = max(close - m, w0 + 1)
    centre = np.array([centre[0], centre[1], 0.0])
    oh = ee[close] - centre
    oh[2] = 0.0
    t = np.arange(T)
    ramp = np.clip((t - w0) / (a1 - w0), 0.0, 1.0)
    if w1 is not None:
        decay = np.clip(1 - (t - opened) / max(w1 - opened, 1), 0.0, 1.0)
        ramp = np.where(t > opened, decay, ramp)
    new_ee = ee.copy()
    off = rotz(alpha) @ oh - oh
    for i in range(w0, T):
        if i < a1:
            new_ee[i] = centre + rotz(ramp[i] * alpha) @ (ee[i] - centre)
        else:
            new_ee[i] = ee[i] + ramp[i] * off
    yaw = (alpha + np.pi) % (2 * np.pi) - np.pi
    if np.isclose(abs(yaw), np.pi):
        yaw = -np.pi
    rot = R.from_euler("z", (ramp * yaw)[:, None]) * R.from_quat(quat)
    assert isinstance(rot, R)
    return new_ee, rot.as_quat().astype(np.float32)


def bulge_delta(ee, s0, s1, phi, theta):
    d = bend_delta(ee, s0, s1, phi, theta)
    if d.any():
        d[s0:s1] += ee[s0:s1] - np.linspace(ee[s0], ee[s1 - 1], s1 - s0)
    return d


def seg_options(cfg, held):
    if held is None:
        return [ZERO]
    thetas = np.linspace(0.0, np.pi, cfg.n_theta, endpoint=False)
    arcs = [(0.0, 0.0)] + [(p, th) for p in cfg.phis for th in thetas]
    return [(a, *ap, *tr) for a in cfg.alphas for ap in arcs for tr in arcs]


def build(cfg, x, base_shift, combo):
    segs, m = x["segs"], round(cfg.margin * cfg.env.fps)
    ee, quat = x["ee"].copy(), x["quat"].copy()
    objs = [
        x["obs"][:, obj_slice(j)].copy() if j is not None else None for j in x["held"]
    ]
    w0 = 0
    fam = set()
    for k, ((c, o), (a, pa, ta, pt, tt)) in enumerate(zip(segs, combo)):
        if a:
            centre = x["obs"][0, obj_slice(x["held"][k])]
            w1 = segs[k + 1][0] - m if k + 1 < len(segs) else None
            ee, quat = grasp_rotation(ee, quat, w0, c, o, w1, m, centre, np.radians(a))
            fam.add("rot")
        if pa:
            ee = ee + bulge_delta(ee, w0 + m if k else 1, c - m, pa * np.pi, ta)
            fam.add("app")
        if pt:
            d = bulge_delta(ee, c + m, o - m, pt * np.pi, tt)
            ee = ee + d
            objs[k] = objs[k] + d
            fam.add("tra")
        w0 = o
    return dict(
        fam="+".join(sorted(fam)) or "none",
        combo=combo,
        ee_t=ee + base_shift,
        quat=quat,
        objs=[o + base_shift if o is not None else None for o in objs],
    )


def clearance_batch(fcol, base_t, q_all, cands, demos, boxes, stride=4):
    tm, B = q_all.shape[:2]
    bx = torch.as_tensor(np.asarray(boxes, np.float32), device=DEV)
    c, h = bx[:, :3], bx[:, 3:]
    arm = np.zeros((B, tm), np.float32)
    worst = np.zeros((B, tm), np.int64)
    q_t = torch.as_tensor(q_all, dtype=torch.float32, device=DEV)
    n_pts = (len(fcol.owner) + stride - 1) // stride
    chunk = max(1, int(2e8 / (tm * n_pts * len(bx) * 12)))
    for s0 in range(0, B, chunk):
        q = q_t[:, s0 : s0 + chunk].reshape(-1, 7)
        pts = fcol.arm_points(q)[:, ::stride] + base_t
        d = box_sdf(pts[..., None, :], c, h).amin(-1) - fcol.radius
        per_pt, idx = d.min(dim=1)
        n = q.shape[0] // tm
        arm[s0 : s0 + n] = per_pt.reshape(tm, n).T.cpu().numpy()
        worst[s0 : s0 + n] = idx.reshape(tm, n).T.cpu().numpy()
    held = np.ones((B, tm), np.float32)
    for k, cd in enumerate(cands):
        x = demos[cd["d"]]
        for (c0, o0), path, r in zip(x["segs"], cd["objs"], cd["radii"]):
            if path is None:
                continue
            p = torch.as_tensor(path[c0:o0], dtype=torch.float32, device=DEV)
            dh = (box_sdf(p[:, None, :], c, h).amin(-1) - r).cpu().numpy()
            held[k, c0:o0] = np.minimum(held[k, c0:o0], dh)
    owner = fcol.owner.cpu().numpy()[::stride]
    return arm, held, owner[worst]


def seg_windows(segs, m, T):
    prev = 0
    out = []
    for c, o in segs:
        out.append((prev, min(o + m, T)))
        prev = o
    return out


def phase_report(prof, segs, m):
    arm, held, link = prof
    out = []
    prev = 0
    for c, o in segs:
        phases = {
            "approach": slice(prev, max(c - m, prev + 1)),
            "grasp": slice(max(c - m, 0), min(c + m, len(arm))),
            "transit": slice(min(c + m, len(arm)), max(o - m, c + m)),
            "place": slice(max(o - m, 0), min(o + m, len(arm))),
        }
        for name, sl in phases.items():
            if sl.stop <= sl.start:
                continue
            k = int(arm[sl].argmin()) + sl.start
            out.append(
                f"{name} {arm[sl].min():+.3f}@{k}({link[k]}) obj {held[sl].min():+.3f}"
            )
        prev = o
    return " | ".join(out)


def run(env, act, hold, names, targets=()):
    p0 = [env.obs[0][f"{n}_pos"].copy() for n in names]
    lift = [0.0] * len(names)
    for a in act:
        env.apply_action(a[None])
        for j, n in enumerate(names):
            lift[j] = max(lift[j], env.obs[0][f"{n}_pos"][2] - p0[j][2])
        if env.collided[0]:
            break
    for _ in range(hold):
        if env.collided[0]:
            break
        env.apply_action(act[-1][None])
    o = env.obs[0]
    moved = [np.linalg.norm(o[f"{n}_pos"][:2] - p[:2]) for n, p in zip(names, p0)]
    dist = [
        np.linalg.norm(o[f"{n}_pos"][:2] - o[f"{t}_pos"][:2]) if t else np.nan
        for n, t in zip(names, targets)
    ]
    info = (
        "lift "
        + "/".join(f"{v:.3f}" for v in lift)
        + " moved "
        + "/".join(f"{v:.3f}" for v in moved)
        + " to-target "
        + "/".join(f"{v:.3f}" for v in dist)
    )
    return bool(env.collided[0]), bool(env.succ[0]), env.contact[0], info


def replay(env, init_idx, act, hold, names, targets):
    env.episode_idx = init_idx
    env.reset()
    return run(env, act, hold, names, targets)


def source_replay(env, state0, act, hold, names, targets):
    env.reset()
    env.obs[0] = env.envs[0].set_init_state(state0)
    env.succ[:] = False
    return run(env, act, hold, names, targets)


def held_names(names, x):
    hn = [names[j] for j in x["held"] if j is not None]
    tn = [
        names[t] if t is not None else ""
        for j, t in zip(x["held"], x["targets"])
        if j is not None
    ]
    return hn, tn


def load_demos(cfg, free_env, goals):
    names = free_env.objects
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[cfg.env.suite]}/{free_env.name}_demo.hdf5",
        repo_type="dataset",
    )
    f = h5py.File(path, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)
    base = np.asarray(free_env.robot_base_pos, np.float32)
    demos: list[dict[str, Any]] = []
    for k in keys[: cfg.n_demos]:
        obs, act = demo_to_episode(free_env, f["data"][k])
        segs = grasp_segments(act[:, 7])
        held = [held_object(obs, c, o, len(names)) for c, o in segs]
        targets = [
            target_object(goals, names, names[j]) if j is not None else None
            for j in held
        ]
        ee, quat = fk(chain, act[:, ARM], base)
        demos.append(
            dict(
                obs=obs,
                act=act,
                segs=segs,
                held=held,
                targets=targets,
                ee=ee,
                quat=quat,
            )
        )
        _, suc, _, info = source_replay(
            free_env,
            f["data"][k]["states"][0],
            act,
            cfg.hold,
            *held_names(names, demos[-1]),
        )
        nm = [names[j] if j is not None else None for j in held + targets]
        print(
            f"demo {k}: T={len(act)} segs={segs} held/targets={nm} "
            f"source replay succ={int(suc)} {info}"
        )
    return demos, chain, base


def evaluate(cfg, cands, demos, chain, base, base_t, fcol, boxes, jerk_limit, stats):
    m = round(cfg.margin * cfg.env.fps)
    tm = max(len(demos[c["d"]]["act"]) for c in cands)
    q_all = solve_seq(
        chain,
        np.stack([tpad(c["ee_t"], tm) for c in cands], axis=1),
        np.stack([tpad(c["quat"], tm) for c in cands], axis=1),
        np.stack([demos[c["d"]]["act"][0, ARM] for c in cands]),
        cfg.ik_iters,
        cfg.ik_damping,
        base,
    )
    achieved = np.zeros((tm, len(cands), 3), np.float32)
    ach_quat = np.zeros((tm, len(cands), 4), np.float32)
    for s0 in range(0, len(cands), 64):
        pos, quat = fk(chain, q_all[:, s0 : s0 + 64].reshape(-1, 7), base)
        n = q_all[:, s0 : s0 + 64].shape[1]
        achieved[:, s0 : s0 + n] = pos.reshape(tm, n, 3)
        ach_quat[:, s0 : s0 + n] = quat.reshape(tm, n, 4)
    arm, held, link = clearance_batch(fcol, base_t, q_all, cands, demos, boxes)
    for k, c in enumerate(cands):
        x = demos[c["d"]]
        nf = len(x["act"])
        q = q_all[:nf, k]
        err = np.linalg.norm(achieved[:nf, k] - c["ee_t"], axis=1).max()
        rot_err = R.from_quat(ach_quat[:nf, k]) * R.from_quat(c["quat"]).inv()
        assert isinstance(rot_err, R)
        rot_err = rot_err.magnitude().max()
        jerk = np.abs(np.diff(q, n=2, axis=0)).max()
        c["ik_ok"] = err < 0.02 and rot_err < np.radians(15) and jerk < jerk_limit
        c["prof"] = (arm[k, :nf], held[k, :nf], [fcol.frames[j] for j in link[k, :nf]])
        frame = np.minimum(c["prof"][0], c["prof"][1])
        c["clear"] = float(frame.min())
        c["seg_clear"] = [
            float(frame[a:b].min()) for a, b in seg_windows(x["segs"], m, nf)
        ]
        act = x["act"].copy()
        act[:, ARM] = q
        c["act"] = act.astype(np.float32)
        s = stats[c["fam"]]
        s[0] += 1
        s[1] += c["ik_ok"]
        s[2] += c["ik_ok"] and c["clear"] > 0


def replay_round(cfg, env, i, cands, demos, names, stats, m, replay_none):
    ok = [c for c in cands if c["ik_ok"]]
    ok.sort(key=lambda c: -c["clear"])
    to_replay = [c for c in ok if c["fam"] == "none"] if replay_none else []
    aug = [c for c in ok if c["fam"] != "none"]
    seen = set()
    picked = []
    for c in aug:
        key = (c["d"],) + tuple((a, pt, tt) for a, _, _, pt, tt in c["combo"])
        if key not in seen and len(picked) < cfg.replay_max:
            seen.add(key)
            picked.append(c)
    picked += [c for c in aug if c not in picked][: cfg.replay_max - len(picked)]
    to_replay += picked
    best = None
    for c in to_replay:
        x = demos[c["d"]]
        col, suc, label, info = replay(
            env, i, c["act"], cfg.hold, *held_names(names, x)
        )
        if cfg.verbose and (c["fam"] == "none" or not col):
            hit = label.split("/")[1] if col else ""
            print(
                f"   {c['fam']:11s} {fmt(c['combo'])} col={int(col)} succ={int(suc)} "
                f"{hit} {info} clear={c['clear']:+.3f} :: "
                f"{phase_report(c['prof'], x['segs'], m)}"
            )
        s = stats[c["fam"]]
        s[3] += 1
        s[4] += not col
        s[5] += suc and not col
        if suc and not col and best is None:
            best = c
    return best, sum(c["clear"] > 0 for c in ok), len(ok)


def fmt(combo):
    return " ".join(
        f"[{a:.0f} {pa:.2f} {ta:.2f} {pt:.2f} {tt:.2f}]" for a, pa, ta, pt, tt in combo
    )


def scene_candidates(cfg, demos, names, pos, radii):
    cands = []
    for d, x in enumerate(demos):
        dbs, dps = [], []
        for (c, op), j, tg in zip(x["segs"], x["held"], x["targets"]):
            if j is None:
                dbs.append(np.zeros(3))
                dps.append(np.zeros(3))
                continue
            dbs.append(
                np.append(pos[names[j]][:2] - x["obs"][0, obj_slice(j)][:2], 0.0)
            )
            if tg is None:
                dps.append(np.zeros(3))
            else:
                end = x["obs"][min(op + 10, len(x["obs"]) - 1), obj_slice(j)][:2]
                dps.append(np.append(pos[names[tg]][:2] - end, 0.0))
        x["base_shift"] = shift_path(
            len(x["act"]), x["segs"], round(cfg.margin * cfg.env.fps), dbs, dps
        )
        x["radii"] = [radii[names[j]] if j is not None else 0.0 for j in x["held"]]
        n = len(x["segs"])
        combos = [tuple([ZERO] * n)]
        for k in range(n):
            for opt in seg_options(cfg, x["held"][k]):
                if opt != ZERO:
                    combos.append(tuple(opt if j == k else ZERO for j in range(n)))
        for combo in combos:
            c = build(cfg, x, x["base_shift"], combo)
            c["d"], c["radii"] = d, x["radii"]
            cands.append(c)
    return cands


def compose_candidates(cfg, cands, demos):
    out = []
    for d, x in enumerate(demos):
        n = len(x["segs"])
        if n < 2:
            continue
        tops = []
        for k in range(n):
            mine = [
                c for c in cands if c["d"] == d and c["ik_ok"] and c["combo"][k] != ZERO
            ]
            mine.sort(key=lambda c: -c["seg_clear"][k])
            tops.append([c["combo"][k] for c in mine[: cfg.top_k]] or [ZERO])
        for combo in product(*tops):
            if sum(o != ZERO for o in combo) < 2:
                continue
            c = build(cfg, x, x["base_shift"], combo)
            c["d"], c["radii"] = d, x["radii"]
            out.append(c)
    return out


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=True))
    free_env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
    assert env.objects == free_env.objects, (env.objects, free_env.objects)
    names = env.objects
    goals = env.goals()
    print(f"task {env.name} level {cfg.env.level}: objects {names} goals {goals}")
    demos, chain, base = load_demos(cfg, free_env, goals)
    base_t = torch.as_tensor(base, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2
    m = round(cfg.margin * cfg.env.fps)

    stats = defaultdict(lambda: np.zeros(6, int))
    scene_success = []
    for i in range(cfg.n_inits):
        env.episode_idx = i
        env.reset()
        boxes = obstacle_geom_boxes(env.envs[0], env.obstacle[0])
        o = env.obs[0]
        pos = {n: o[f"{n}_pos"] for n in names}
        radii = {n: float(body_aabb(env.envs[0], n)[3:5].max()) for n in names}
        cands = scene_candidates(cfg, demos, names, pos, radii)
        args = (demos, chain, base, base_t, fcol, boxes, jerk_limit, stats)
        evaluate(cfg, cands, *args)
        best, geo, n_ok = replay_round(cfg, env, i, cands, demos, names, stats, m, True)
        if best is None:
            more = compose_candidates(cfg, cands, demos)
            if more:
                evaluate(cfg, more, *args)
                best, g2, n2 = replay_round(
                    cfg, env, i, more, demos, names, stats, m, False
                )
                geo, n_ok = geo + g2, n_ok + n2
        scene_success.append(best is not None)
        desc = (
            "none"
            if best is None
            else f"{best['fam']} {fmt(best['combo'])} clear={best['clear']:.3f}"
        )
        print(
            f"scene {i} {env.obstacle[0]} boxes={len(boxes)}: geo-clean {geo}/{n_ok} | "
            f"first collision-free success: {desc}",
            flush=True,
        )

    print(
        f"{'family':11s} {'cands':>6s} {'ik_ok':>6s} {'geo':>6s} "
        f"{'replay':>6s} {'nocol':>6s} {'succ':>6s}"
    )
    for fam, s in sorted(stats.items()):
        print(f"{fam:11s} " + " ".join(f"{v:6d}" for v in s))
    aug = sum(s[5] for f, s in stats.items() if f != "none")
    print(
        f"RESULT {cfg.env.suite} task {cfg.env.task_id} level {cfg.env.level}: "
        f"scenes {sum(scene_success)}/{len(scene_success)} "
        f"orig {stats['none'][5]}/{stats['none'][3]} aug {aug}"
    )


if __name__ == "__main__":
    main()
