from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import draccus
import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R

from flow_planning.bend import (
    ARM,
    bend_delta,
    fk,
    solve_seq,
    tpad,
    transit_segments,
)
from flow_planning.envs import EnvConfig
from flow_planning.envs.libero import LiberoConfig, LiberoEnv
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.selector import FrankaCollision, box_sdf
from flow_planning.utils import hf_column

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOWL = slice(18, 21)
PLATE = slice(27, 30)
BOWL_RADIUS = 0.07
BOWL_DROP = 0.03


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    repo_id: str = "reece-omahoney/safelibero_goal_0_v2"
    n_inits: int = 10
    n_demos: int = 3
    margin: float = 0.3
    phis: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    n_theta: int = 6
    replay_max: int = 12
    ik_iters: int = 8
    ik_damping: float = 0.05
    hold: int = 20
    verbose: bool = True
    families: str = "arcs,grasprot"
    alphas: list[float] = field(default_factory=lambda: [90.0, 180.0, 270.0])


def shift_path(T, close, opened, m, db, dp):
    knots = [
        0,
        max(close - m, 1),
        min(close + m, T - 1),
        max(opened - m, close + m + 1),
        T - 1,
    ]
    vals = np.stack([np.zeros(3), db, db, dp, dp])
    t = np.arange(T)
    return np.stack([np.interp(t, knots, vals[:, k]) for k in range(3)], axis=1)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def grasp_rotation(x, a1, alpha):
    ee, quat = x["ee"], x["quat"]
    T = len(ee)
    bowl = x["obs"][0, BOWL].copy()
    bowl[2] = 0.0
    o = ee[x["close"]] - bowl
    o[2] = 0.0
    ramp = np.clip(np.arange(T) / max(a1, 1), 0.0, 1.0) * alpha
    new_ee = ee.copy()
    for t in range(T):
        if t < a1:
            new_ee[t] = bowl + rotz(ramp[t]) @ (ee[t] - bowl)
        else:
            new_ee[t] = ee[t] + rotz(alpha) @ o - o
    rot = R.from_euler("z", ramp) * R.from_quat(quat)
    assert isinstance(rot, R)
    new_quat = rot.as_quat()
    return new_ee, new_quat.astype(np.float32)


def candidates(cfg, x, db, dp):
    T, close, opened = len(x["act"]), x["close"], x["opened"]
    m = round(cfg.margin * cfg.env.fps)
    base_shift = shift_path(T, close, opened, m, db, dp)
    ee, quat = x["ee"], x["quat"]
    out = [("none", np.zeros(2), np.zeros(2), ee + base_shift, quat)]
    thetas = np.linspace(0.0, np.pi, cfg.n_theta, endpoint=False)
    a0, a1 = 1, close - m
    t0, t1 = close + m, opened - m
    fams = cfg.families.split(",")
    if "arcs" in fams:
        for phi in cfg.phis:
            for th in thetas:
                da = bend_delta(ee, a0, a1, phi * np.pi, th)
                dt = bend_delta(ee, t0, t1, phi * np.pi, th)
                la, lt = np.array([phi, th]), np.array([phi, th])
                out.append(("approach", la, np.zeros(2), ee + base_shift + da, quat))
                out.append(("transit", np.zeros(2), lt, ee + base_shift + dt, quat))
                out.append(("both", la, lt, ee + base_shift + da + dt, quat))
    if "grasprot" in fams:
        for deg in cfg.alphas:
            ee_r, quat_r = grasp_rotation(x, a1, np.radians(deg))
            la = np.array([deg, 0.0])
            out.append(("grasprot", la, np.zeros(2), ee_r + base_shift, quat_r))
            for phi in cfg.phis:
                for th in thetas:
                    dt = bend_delta(ee_r, t0, t1, phi * np.pi, th)
                    lt = np.array([phi, th])
                    out.append(
                        ("grasprot+transit", la, lt, ee_r + base_shift + dt, quat_r)
                    )
    return out


def clearance_profile(fcol, base_t, q, ee_t, close, opened, box):
    box_t = torch.as_tensor(np.asarray(box, np.float32), device=DEV)
    c, h = box_t[:3], box_t[3:]
    pts = fcol.arm_points(torch.as_tensor(q, dtype=torch.float32, device=DEV)) + base_t
    d = box_sdf(pts, c, h) - fcol.radius
    per_frame, worst_pt = d.min(dim=1)
    held = torch.as_tensor(ee_t, dtype=torch.float32, device=DEV)
    held = held - torch.tensor([0.0, 0.0, BOWL_DROP], device=DEV)
    bowl = box_sdf(held, c, h) - BOWL_RADIUS
    bowl[:close] = 1.0
    bowl[opened:] = 1.0
    link = [fcol.frames[int(fcol.owner[i])] for i in worst_pt.cpu().numpy()]
    return per_frame.cpu().numpy(), bowl.cpu().numpy(), link


def geometry_clearance(fcol, base_t, q, ee_t, close, opened, box):
    arm, bowl, _ = clearance_profile(fcol, base_t, q, ee_t, close, opened, box)
    return float(min(arm.min(), bowl.min()))


def phase_report(fcol, base_t, q, ee_t, close, opened, box, m):
    arm, bowl, link = clearance_profile(fcol, base_t, q, ee_t, close, opened, box)
    phases = {
        "approach": slice(0, max(close - m, 1)),
        "grasp": slice(max(close - m, 0), min(close + m, len(q))),
        "transit": slice(min(close + m, len(q)), max(opened - m, close + m)),
        "place": slice(max(opened - m, 0), len(q)),
    }
    out = []
    for name, sl in phases.items():
        if sl.stop <= sl.start:
            continue
        k = int(arm[sl].argmin()) + sl.start
        out.append(
            f"{name} arm {arm[sl].min():+.3f}@{k}({link[k]}) bowl {bowl[sl].min():+.3f}"
        )
    return " | ".join(out)


def replay(env, init_idx, act, hold):
    env.episode_idx = init_idx
    env.reset()
    for a in act:
        env.apply_action(a[None])
        if env.collided[0]:
            break
    for _ in range(hold):
        if env.collided[0]:
            break
        env.apply_action(act[-1][None])
    return bool(env.collided[0]), bool(env.succ[0]), env.contact[0]


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=True))
    base = np.asarray(env.robot_base_pos, np.float32)
    base_t = torch.as_tensor(base, device=DEV)
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)
    fcol = FrankaCollision(DEV, base, 0.0)
    jerk_limit = 0.6 * (50 / cfg.env.fps) ** 2

    src = LeRobotDataset(cfg.repo_id)
    hf = src.hf_dataset
    epi = hf_column(hf, "episode_index")
    obs_all = hf_column(hf, "observation.state")
    act_all = hf_column(hf, "action")
    demos: list[dict[str, Any]] = []
    for e in range(cfg.n_demos):
        sel = epi == e
        obs, act = obs_all[sel], act_all[sel]
        close, opened = transit_segments(act[:, 7])
        ee, quat = fk(chain, act[:, ARM], base)
        demos.append(
            dict(obs=obs, act=act, close=close, opened=opened, ee=ee, quat=quat)
        )

    stats = defaultdict(lambda: np.zeros(6, int))
    scene_success = []
    for i in range(cfg.n_inits):
        env.episode_idx = i
        env.reset()
        box = env.obstacle_boxes()[0]
        o = env.obs[0]
        bowl, plate = o["akita_black_bowl_1_pos"], o["plate_1_pos"]
        cands: list[dict[str, Any]] = []
        for d, x in enumerate(demos):
            db = np.append(bowl[:2] - x["obs"][0, BOWL][:2], 0.0)
            dp = np.append(plate[:2] - x["obs"][0, PLATE][:2], 0.0)
            for fam, la, lt, ee_t, quat_t in candidates(cfg, x, db, dp):
                cands.append(dict(d=d, fam=fam, la=la, lt=lt, ee_t=ee_t, quat=quat_t))
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
        achieved = fk(chain, q_all.reshape(-1, 7), base)[0].reshape(tm, len(cands), 3)
        for k, c in enumerate(cands):
            x = demos[c["d"]]
            nf = len(x["act"])
            q = q_all[:nf, k]
            err = np.linalg.norm(achieved[:nf, k] - c["ee_t"], axis=1).max()
            jerk = np.abs(np.diff(q, n=2, axis=0)).max()
            c["ik_ok"] = err < 0.05 and jerk < jerk_limit
            c["clear"] = geometry_clearance(
                fcol, base_t, q, c["ee_t"], x["close"], x["opened"], box
            )
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
        m = round(cfg.margin * cfg.env.fps)
        for c in to_replay:
            col, suc, label = replay(env, i, c["act"], cfg.hold)
            x = demos[c["d"]]
            if cfg.verbose and (c["fam"] == "none" or not col):
                prof = phase_report(
                    fcol,
                    base_t,
                    c["act"][:, ARM],
                    c["ee_t"],
                    x["close"],
                    x["opened"],
                    box,
                    m,
                )
                hit = label.split("/")[1] if col else ""
                print(
                    f"   {c['fam']:8s} a={c['la'].round(2)} t={c['lt'].round(2)} "
                    f"col={int(col)} succ={int(suc)} {hit} "
                    f"clear={c['clear']:+.3f} :: {prof}"
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
            else (
                f"{best['fam']} approach(phi,theta)={best['la'].round(2)} "
                f"transit={best['lt'].round(2)} clear={best['clear']:.3f}"
            )
        )
        print(
            f"scene {i} {env.obstacle[0]}: geo-clean {sum(c['clear'] > 0 for c in ok)}"
            f"/{len(ok)} | first collision-free success: {desc}",
            flush=True,
        )

    print(
        f"\nlevel {cfg.env.level}: scenes with a collision-free success "
        f"{sum(scene_success)}/{len(scene_success)}"
    )
    print(
        f"{'family':10s} {'cands':>6s} {'ik_ok':>6s} {'geo':>6s} "
        f"{'replay':>6s} {'nocol':>6s} {'succ':>6s}"
    )
    for fam, s in stats.items():
        print(f"{fam:10s} " + " ".join(f"{v:6d}" for v in s))


if __name__ == "__main__":
    main()
