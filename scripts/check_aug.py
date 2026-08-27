from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

import cv2
import draccus
import h5py
import numpy as np
import pytorch_kinematics as pk
import torch
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from flow_planning.bend import (
    ARM,
    SEG_ZERO,
    build,
    fk,
    held_segments,
    obj_slice,
    seg_options,
    solve_seq,
    tpad,
    yaw_of_quat,
)
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
ZERO = SEG_ZERO


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    n_inits: int = 10
    init_start: int = 0
    n_demos: int = 3
    margin: float = 0.3
    phi_range: tuple[float, float] = (0.15, 0.85)
    n_phi: int = 3
    n_theta: int = 4
    seed: int = 0
    top_k: int = 4
    replay_max: int = 12
    ik_iters: int = 12
    ik_damping: float = 0.05
    hold: int = 20
    verbose: bool = True
    vel_frac: float = 0.8
    ik_tol: float = 0.01
    video: str = ""
    size: int = 512


VIDEO: dict = {}


def target_name(goals, objects, sites, name):
    for _, a, b in goals:
        if a != name:
            continue
        hits = [o for o in objects if b.startswith(o) and o != name]
        if hits:
            return max(hits, key=len)
        if b in sites:
            return b
    return None


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


def frame(env, t, done=None):
    from view_aug import draw, project

    f = np.ascontiguousarray(env.obs[0]["agentview_image"][::-1, ::-1])
    plan = project(env.envs[0].sim, VIDEO["plan"], VIDEO["size"])
    draw(f, plan, (0, 220, 0), 1)
    draw(f, plan[t : t + 1], (0, 0, 255), 5)
    cv2.putText(f, VIDEO["label"], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    if done:
        cv2.putText(
            f, done, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
    VIDEO["writer"].write(f[..., ::-1])


def run(env, act, hold, names, expected):
    rec = bool(VIDEO) and env.cfg.render
    p0 = [env.obs[0][f"{n}_pos"].copy() for n in names]
    lift = [0.0] * len(names)
    trk = 0.0
    for t, a in enumerate(act):
        env.apply_action(a[None])
        if rec:
            frame(env, t)
        trk = max(trk, np.abs(env.obs[0]["robot0_joint_pos"] - a[:7]).max())
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
    dist = [np.linalg.norm(o[f"{n}_pos"][:2] - e[:2]) for n, e in zip(names, expected)]
    if rec:
        for _ in range(env.cfg.fps):
            frame(
                env,
                len(act) - 1,
                f"collided={int(env.collided[0])} success={int(env.succ[0])}",
            )
    info = (
        "lift "
        + "/".join(f"{v:.3f}" for v in lift)
        + " moved "
        + "/".join(f"{v:.3f}" for v in moved)
        + " to-target "
        + "/".join(f"{v:.3f}" for v in dist)
    )
    return bool(env.collided[0]), bool(env.succ[0]), env.contact[0], info


def replay(env, init_idx, act, hold, names, expected, plan=None, label=""):
    if VIDEO:
        VIDEO["plan"], VIDEO["label"] = plan, label
    env.episode_idx = init_idx
    env.reset()
    return run(env, act, hold, names, expected)


def source_replay(env, state0, act, hold, names, expected):
    env.reset()
    env.obs[0] = env.envs[0].set_init_state(state0)
    env.succ[:] = False
    return run(env, act, hold, names, expected)


def held_names(names, x):
    return [names[j] for j in x["held"] if j is not None]


def load_demos(cfg, free_env, goals):
    names = free_env.objects
    sites = set(free_env.envs[0].env.object_sites_dict)
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
        segs, held = held_segments(obs, act[:, 7], len(names))
        targets = [
            target_name(goals, names, sites, names[j]) if j is not None else None
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
        x = demos[-1]
        ends = [x["obs"][-1, obj_slice(j)] for j in held if j is not None]
        _, suc, _, info = source_replay(
            free_env,
            f["data"][k]["states"][0],
            act,
            cfg.hold,
            held_names(names, x),
            ends,
        )
        nm = [names[j] if j is not None else None for j in held] + targets
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
        vel = np.abs(np.diff(q, axis=0)).max()
        c["ik_err"] = float(err)
        c["ik_ok"] = (
            err < cfg.ik_tol
            and rot_err < np.radians(15)
            and jerk < jerk_limit
            and vel < cfg.vel_frac * cfg.env.delta_max
        )
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
    groups = defaultdict(list)
    for c in aug:
        groups[tuple(o[:2] for o in c["combo"])].append(c)
    picked = []
    while len(picked) < cfg.replay_max // 2 and any(groups.values()):
        for g in list(groups):
            if groups[g] and len(picked) < cfg.replay_max // 2:
                picked.append(groups[g].pop(0))
    ids = {id(c) for c in picked}
    picked += [c for c in aug if id(c) not in ids][: cfg.replay_max - len(picked)]
    to_replay += picked
    best = None
    for c in to_replay:
        x = demos[c["d"]]
        hn = held_names(names, x)
        col, suc, label, info = replay(
            env,
            i,
            c["act"],
            cfg.hold,
            hn,
            c["expected"],
            c["ee_t"],
            f"scene{i} {c['fam']} {fmt(c['combo'])}",
        )
        if cfg.verbose and (c["fam"] == "none" or not col):
            hit = label.split("/")[1] if col else ""
            print(
                f"   {c['fam']:11s} {fmt(c['combo'])} col={int(col)} succ={int(suc)} "
                f"{hit} {info} ik {c['ik_err']:.3f} clear={c['clear']:+.3f} :: "
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
        f"[{pa:.2f} {ta:.2f} {pt:.2f} {tt:.2f}]" for pa, ta, pt, tt in combo
    )


def scene_geometry(env, target_names):
    o = env.obs[0]
    sim = env.envs[0].sim
    names = env.objects
    pos = {n: o[f"{n}_pos"] for n in names}
    yaws = {n: yaw_of_quat(o[f"{n}_quat"]) for n in names}
    halves = {n: body_aabb(env.envs[0], n)[3:5] for n in names}
    radii = {n: float(halves[n].max()) for n in names}
    tpos = {}
    for t in target_names:
        tpos[t] = pos[t] if t in pos else sim.data.get_site_xpos(t).copy()
    return pos, yaws, radii, halves, tpos


def scene_candidates(cfg, demos, names, geo, tshift):
    rng = np.random.default_rng(cfg.seed)
    pos, yaws, radii, halves, tpos = geo
    m = round(cfg.margin * cfg.env.fps)
    cands = []
    for d, x in enumerate(demos):
        dbs, dps, diag, ends = [], [], [], []
        for (c, op), j, tg in zip(x["segs"], x["held"], x["targets"]):
            if j is None:
                dbs.append(np.zeros(3))
                dps.append(np.zeros(3))
                ends.append(np.zeros(3))
                continue
            sl = obj_slice(j)
            db = pos[names[j]] - x["obs"][0, sl]
            dbs.append(db)
            ends.append(x["obs"][-1, sl])
            diag.append(f"{names[j]} d={db.round(3)}")
            if tg is None:
                dps.append(np.zeros(3))
            elif tg in names:
                end = x["obs"][min(op + 10, len(x["obs"]) - 1), sl][:2]
                dps.append(np.append(tpos[tg][:2] - end, 0.0))
            else:
                dps.append(np.append(tshift[tg][:2], 0.0))
        x["dbs"], x["dps"], x["ends"] = dbs, dps, ends
        x["radii"] = [radii[names[j]] if j is not None else 0.0 for j in x["held"]]
        x["half"] = [
            halves[names[j]] if j is not None else np.ones(2) for j in x["held"]
        ]
        if d == 0:
            print("   retarget " + " | ".join(diag))
        n = len(x["segs"])
        combos = [tuple([ZERO] * n)]
        for k in range(n):
            for opt in (
                seg_options(rng, cfg.phi_range, cfg.n_phi, cfg.n_theta)
                if x["held"][k] is not None
                else [ZERO]
            ):
                if opt != ZERO:
                    combos.append(tuple(opt if j == k else ZERO for j in range(n)))
        for combo in combos:
            c = build(x, combo, m)
            if c is not None:
                c["d"], c["radii"] = d, x["radii"]
                cands.append(c)
    return cands


def compose_candidates(cfg, cands, demos):
    m = round(cfg.margin * cfg.env.fps)
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
            c = build(x, combo, m)
            if c is not None:
                c["d"], c["radii"] = d, x["radii"]
                out.append(c)
    return out


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    env = LiberoEnv(
        replace(
            cfg.env,
            world_count=1,
            obstacle=True,
            render=bool(cfg.video),
            render_size=cfg.size,
        )
    )
    if cfg.video:
        VIDEO.update(
            writer=cv2.VideoWriter(
                cfg.video,
                cv2.VideoWriter.fourcc(*"mp4v"),
                cfg.env.fps,
                (cfg.size, cfg.size),
            ),
            size=cfg.size,
        )
    free_env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
    assert env.objects == free_env.objects, (env.objects, free_env.objects)
    names = env.objects
    goals = env.goals()
    print(f"task {env.name} level {cfg.env.level}: objects {names} goals {goals}")
    demos, chain, base = load_demos(cfg, free_env, goals)
    tn = sorted({t for x in demos for t in x["targets"] if t is not None})
    base_t = torch.as_tensor(base, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2
    m = round(cfg.margin * cfg.env.fps)

    stats = defaultdict(lambda: np.zeros(6, int))
    scene_success = []
    for i in range(cfg.init_start, cfg.init_start + cfg.n_inits):
        free_env.episode_idx = i
        free_env.reset()
        geo_free = scene_geometry(free_env, tn)
        zero = {t: np.zeros(3) for t in tn}
        control = scene_candidates(cfg, demos, names, geo_free, zero)
        control = [c for c in control if c["fam"] == "none"]
        env.episode_idx = i
        env.reset()
        boxes = obstacle_geom_boxes(env.envs[0], env.obstacle[0])
        args = (demos, chain, base, base_t, fcol, boxes, jerk_limit, stats)
        evaluate(cfg, control, *args)
        for c in control:
            if not c["ik_ok"]:
                continue
            hn = held_names(names, demos[c["d"]])
            _, suc, _, info = replay(free_env, i, c["act"], cfg.hold, hn, c["expected"])
            stats["free"][3] += 1
            stats["free"][5] += suc
            if cfg.verbose:
                print(f"   free-scene  demo {c['d']} succ={int(suc)} {info}")
        geo = scene_geometry(env, tn)
        tshift = {t: geo[4][t] - geo_free[4][t] for t in tn}
        if tn:
            print(
                "   fixture shift "
                + " ".join(f"{t}={tshift[t][:2].round(3)}" for t in tn)
            )
        cands = scene_candidates(cfg, demos, names, geo, tshift)
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
    aug = sum(s[5] for f, s in stats.items() if f not in ("none", "free"))
    if cfg.video:
        VIDEO["writer"].release()
    print(
        f"RESULT {cfg.env.suite} task {cfg.env.task_id} level {cfg.env.level}: "
        f"scenes {sum(scene_success)}/{len(scene_success)} "
        f"orig {stats['none'][5]}/{stats['none'][3]} aug {aug} "
        f"free-scene {stats['free'][5]}/{stats['free'][3]}"
    )


if __name__ == "__main__":
    main()
