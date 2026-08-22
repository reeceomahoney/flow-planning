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
)
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision, box_sdf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OBJ0 = 18


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    n_inits: int = 10
    n_demos: int = 3
    margin: float = 0.3
    phis: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    n_theta: int = 6
    alphas: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    max_cands: int = 3000
    replay_max: int = 16
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


def grasp_rotation(ee, quat, w0, close, opened, m, centre, alpha):
    T = len(ee)
    a1 = max(close - m, w0 + 1)
    centre = np.array([centre[0], centre[1], 0.0])
    oh = ee[close] - centre
    oh[2] = 0.0
    t = np.arange(T)
    ramp = np.clip((t - w0) / (a1 - w0), 0.0, 1.0)
    ramp = np.where(t > opened, np.clip(1 - (t - opened) / (2 * m), 0.0, 1.0), ramp)
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


def candidates(cfg, x, dbs, dps):
    T, segs, m = len(x["act"]), x["segs"], round(cfg.margin * cfg.env.fps)
    base_shift = shift_path(T, segs, m, dbs, dps)
    thetas = np.linspace(0.0, np.pi, cfg.n_theta, endpoint=False)
    arcs = [(0.0, 0.0)] + [(p, th) for p in cfg.phis for th in thetas]
    options = []
    for k in range(len(segs)):
        if x["held"][k] is None:
            options.append([(0.0, 0.0, 0.0)])
        else:
            options.append([(a, p, th) for a in cfg.alphas for p, th in arcs])
    out = []
    for combo in product(*options):
        ee, quat = x["ee"].copy(), x["quat"].copy()
        objs = [
            x["obs"][:, obj_slice(j)].copy() if j is not None else None
            for j in x["held"]
        ]
        w0 = 0
        for k, ((c, o), (a, p, th)) in enumerate(zip(segs, combo)):
            if a:
                centre = x["obs"][0, obj_slice(x["held"][k])]
                ee, quat = grasp_rotation(ee, quat, w0, c, o, m, centre, np.radians(a))
            if p:
                d = bend_delta(ee, c + m, o - m, p * np.pi, th)
                ee = ee + d
                objs[k] = objs[k] + d
            w0 = o
        fam = ("rot" if any(a for a, _, _ in combo) else "") + (
            "+arc" if any(p for _, p, _ in combo) else ""
        )
        out.append(
            dict(
                fam=fam.strip("+") or "none",
                label=np.array(combo).ravel(),
                ee_t=ee + base_shift,
                quat=quat,
                objs=[o + base_shift if o is not None else None for o in objs],
            )
        )
    return out


def clearance_profile(fcol, base_t, q, c, x, box, radii):
    box_t = torch.as_tensor(np.asarray(box, np.float32), device=DEV)
    bc, bh = box_t[:3], box_t[3:]
    pts = fcol.arm_points(torch.as_tensor(q, dtype=torch.float32, device=DEV)) + base_t
    d = box_sdf(pts, bc, bh) - fcol.radius
    per_frame, worst_pt = d.min(dim=1)
    held = torch.ones(len(q), device=DEV)
    for (c0, o0), path, r in zip(x["segs"], c["objs"], radii):
        if path is None:
            continue
        p = torch.as_tensor(path[c0:o0], dtype=torch.float32, device=DEV)
        held[c0:o0] = torch.minimum(held[c0:o0], box_sdf(p, bc, bh) - r)
    link = [fcol.frames[int(fcol.owner[i])] for i in worst_pt.cpu().numpy()]
    return per_frame.cpu().numpy(), held.cpu().numpy(), link


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


def replay(env, init_idx, act, hold, names):
    env.episode_idx = init_idx
    env.reset()
    z0 = [env.obs[0][f"{n}_pos"][2] for n in names]
    lift = [0.0] * len(names)
    for a in act:
        env.apply_action(a[None])
        for j, n in enumerate(names):
            lift[j] = max(lift[j], env.obs[0][f"{n}_pos"][2] - z0[j])
        if env.collided[0]:
            break
    for _ in range(hold):
        if env.collided[0]:
            break
        env.apply_action(act[-1][None])
    info = "lift " + "/".join(f"{v:.3f}" for v in lift)
    return bool(env.collided[0]), bool(env.succ[0]), env.contact[0], info


def load_demos(cfg, free_env, n_obj, goals):
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
        held = [held_object(obs, c, o, n_obj) for c, o in segs]
        targets = [
            target_object(goals, free_env.objects, free_env.objects[j])
            if j is not None
            else None
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
        nm = [free_env.objects[j] if j is not None else None for j in held + targets]
        print(f"demo {k}: T={len(act)} segs={segs} held/targets={nm}")
    return demos, chain, base


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=True))
    free_env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
    assert env.objects == free_env.objects, (env.objects, free_env.objects)
    names = env.objects
    goals = env.goals()
    print(f"task {env.name} level {cfg.env.level}: objects {names} goals {goals}")
    demos, chain, base = load_demos(cfg, free_env, len(names), goals)
    base_t = torch.as_tensor(base, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2
    rng = np.random.default_rng(0)
    m = round(cfg.margin * cfg.env.fps)

    stats = defaultdict(lambda: np.zeros(6, int))
    scene_success = []
    for i in range(cfg.n_inits):
        env.episode_idx = i
        env.reset()
        box = env.obstacle_boxes()[0]
        o = env.obs[0]
        pos = {n: o[f"{n}_pos"] for n in names}
        radii = {n: float(body_aabb(env.envs[0], n)[3:5].max()) for n in names}
        cands: list[dict[str, Any]] = []
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
                    dps.append(dbs[-1] * 0.0)
                else:
                    end = x["obs"][min(op + 10, len(x["obs"]) - 1), obj_slice(j)][:2]
                    dps.append(np.append(pos[names[tg]][:2] - end, 0.0))
            for c in candidates(cfg, x, dbs, dps):
                c["d"] = d
                c["radii"] = [
                    radii[names[j]] if j is not None else 0.0 for j in x["held"]
                ]
                cands.append(c)
        if len(cands) > cfg.max_cands:
            keep = rng.choice(len(cands), cfg.max_cands, replace=False)
            keep = sorted(
                set(keep) | {k for k, c in enumerate(cands) if c["fam"] == "none"}
            )
            print(f"   subsampled {len(cands)} -> {len(keep)} candidates")
            cands = [cands[k] for k in keep]
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
        ach_pos, ach_quat = fk(chain, q_all.reshape(-1, 7), base)
        achieved = ach_pos.reshape(tm, len(cands), 3)
        ach_quat = ach_quat.reshape(tm, len(cands), 4)
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
            c["prof"] = clearance_profile(fcol, base_t, q, c, x, box, c["radii"])
            c["clear"] = float(min(c["prof"][0].min(), c["prof"][1].min()))
            act = x["act"].copy()
            act[:, ARM] = q
            c["act"] = act.astype(np.float32)
            s = stats[c["fam"]]
            s[0] += 1
            s[1] += c["ik_ok"]
            s[2] += c["ik_ok"] and c["clear"] > 0
        ok = [c for c in cands if c["ik_ok"]]
        ok.sort(key=lambda c: -c["clear"])
        to_replay = [c for c in ok if c["fam"] == "none"] + [
            c for c in ok if c["fam"] != "none"
        ][: cfg.replay_max]
        best = None
        for c in to_replay:
            x = demos[c["d"]]
            held = [names[j] for j in x["held"] if j is not None]
            col, suc, label, info = replay(env, i, c["act"], cfg.hold, held)
            if cfg.verbose and (c["fam"] == "none" or not col):
                hit = label.split("/")[1] if col else ""
                print(
                    f"   {c['fam']:8s} {c['label'].round(2)} col={int(col)} "
                    f"succ={int(suc)} {hit} {info} clear={c['clear']:+.3f} :: "
                    f"{phase_report(c['prof'], x['segs'], m)}"
                )
            s = stats[c["fam"]]
            s[3] += 1
            s[4] += not col
            s[5] += suc and not col
            if suc and not col and best is None:
                best = c
        scene_success.append(best is not None)
        desc = (
            "none"
            if best is None
            else f"{best['fam']} {best['label'].round(2)} clear={best['clear']:.3f}"
        )
        print(
            f"scene {i} {env.obstacle[0]}: geo-clean {sum(c['clear'] > 0 for c in ok)}"
            f"/{len(ok)} | first collision-free success: {desc}",
            flush=True,
        )

    print(
        f"{'family':10s} {'cands':>6s} {'ik_ok':>6s} {'geo':>6s} "
        f"{'replay':>6s} {'nocol':>6s} {'succ':>6s}"
    )
    for fam, s in stats.items():
        print(f"{fam:10s} " + " ".join(f"{v:6d}" for v in s))
    aug = sum(s[5] for f, s in stats.items() if f != "none")
    print(
        f"RESULT {cfg.env.suite} task {cfg.env.task_id} level {cfg.env.level}: "
        f"scenes {sum(scene_success)}/{len(scene_success)} "
        f"orig {stats['none'][5]}/{stats['none'][3]} aug {aug}"
    )


if __name__ == "__main__":
    main()
