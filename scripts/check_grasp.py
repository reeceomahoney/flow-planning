import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import draccus
import h5py
import numpy as np
import pytorch_kinematics as pk
import torch
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from flow_planning.bend import ARM, fk, held_segments, obj_slice, solve_seq, tpad
from flow_planning.envs import EnvConfig
from flow_planning.envs.libero import (
    GRIPPER_OPEN,
    HF_DEMOS,
    SOURCE_SUITE,
    LiberoConfig,
    LiberoEnv,
    demo_to_episode,
)
from flow_planning.kinematics import EE_FRAME, build_franka_chain

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    n_inits: int = 20
    n_demos: int = 50
    margin: float = 0.4
    lift: float = 0.02
    settle: int = 3
    threshold: float = 0.1
    ik_iters: int = 32
    ik_damping: float = 0.05
    ik_tol: float = 0.01
    out: str = "outputs/check_grasp"


def load_demos(cfg, free_env):
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
    m = round(cfg.margin * cfg.env.fps)
    demos = []
    for k in keys[: cfg.n_demos]:
        obs, act = demo_to_episode(free_env, f["data"][k])
        segs, held = held_segments(obs, act[:, 7], len(names))
        if not segs:
            print(f"demo {k}: no held segment, skipped")
            continue
        (c, o), j = segs[0], held[0]
        z = obs[:, obj_slice(j)][:, 2]
        up = np.flatnonzero(z[c:o] > z[c] + cfg.lift)
        lift = c + int(up[0]) if len(up) else c + m
        g0, g1 = max(c - m, 0), min(max(c + m, lift + 2), o)
        ee, quat = fk(chain, act[:, ARM], base)
        demos.append(
            dict(
                key=k,
                obs=obs,
                act=act,
                obj=j,
                close=c,
                g0=g0,
                g1=g1,
                ee=ee,
                quat=quat,
            )
        )
        print(
            f"demo {k}: T={len(act)} seg=({c},{o}) held={names[j]} "
            f"lift@{lift} window=[{g0},{g1})"
        )
    return demos, chain, base


def solve_windows(cfg, demos, chain, base, shifts):
    W = shifts.shape[1]
    tm = max(x["g1"] - x["g0"] for x in demos)
    ee_t, quat_t, seed = [], [], []
    for d, x in enumerate(demos):
        sl = slice(x["g0"], x["g1"])
        for w in range(W):
            ee_t.append(tpad(x["ee"][sl] + shifts[d, w], tm))
            quat_t.append(tpad(x["quat"][sl], tm))
            seed.append(tpad(x["act"][sl, ARM], tm))
    ee_t, quat_t = np.stack(ee_t, 1), np.stack(quat_t, 1)
    q = solve_seq(
        chain, ee_t, quat_t, np.stack(seed, 1), cfg.ik_iters, cfg.ik_damping, base
    )
    pos, quat = fk(chain, q.reshape(-1, 7), base)
    pos, quat = pos.reshape(q.shape[:2] + (3,)), quat.reshape(q.shape[:2] + (4,))
    err = np.linalg.norm(pos - ee_t, axis=-1)
    rel = R.from_quat(quat.reshape(-1, 4)) * R.from_quat(quat_t.reshape(-1, 4)).inv()
    assert isinstance(rel, R)
    rot = np.asarray(rel.magnitude()).reshape(q.shape[:2])
    ok = np.zeros((len(demos), W), bool)
    for d, x in enumerate(demos):
        n = x["g1"] - x["g0"]
        e = err[:n, d * W : (d + 1) * W].max(0)
        r = rot[:n, d * W : (d + 1) * W].max(0)
        ok[d] = (e < cfg.ik_tol) & (r < np.radians(15))
    return q.reshape(tm, len(demos), W, 7).transpose(1, 2, 0, 3), ok


def teleport(env, w, q):
    e = env.envs[w]
    idx = e.robots[0]._ref_joint_pos_indexes
    e.sim.data.qpos[idx] = q
    e.sim.data.qvel[idx] = 0.0
    e.sim.forward()


def replay(cfg, env, x, q, b, names):
    W = len(env.envs)
    n = x["g1"] - x["g0"]
    env.episode_idx = b
    env.reset()
    for w in range(W):
        teleport(env, w, q[w, 0])
    grip = x["act"][x["g0"] : x["g1"], 7]
    a0 = np.concatenate([q[:, 0], np.full((W, 1), GRIPPER_OPEN, np.float32)], 1)
    for _ in range(cfg.settle):
        env.apply_action(a0)
    name = names[x["obj"]]
    z0 = np.array([env.obs[w][f"{name}_pos"][2] for w in range(W)])
    col_t = np.where(env.collided, 0, -1)
    for t in range(1, n):
        a = np.concatenate([q[:, t], np.full((W, 1), grip[t - 1], np.float32)], 1)
        env.apply_action(a)
        col_t[(col_t < 0) & env.collided] = t
    z1 = np.array([env.obs[w][f"{name}_pos"][2] for w in range(W)])
    return col_t, z1 - z0, list(env.contact)


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    W = cfg.env.world_count
    env = LiberoEnv(replace(cfg.env, obstacle=True))
    free_env = LiberoEnv(replace(cfg.env, world_count=1, obstacle=False))
    names = env.objects
    assert names == free_env.objects
    tag = f"{cfg.env.suite}_{cfg.env.task_id}_{cfg.env.level}"
    print(f"task {env.name} level {cfg.env.level}: objects {names}")
    demos, chain, base = load_demos(cfg, free_env)
    D = len(demos)
    n_b = -(-cfg.n_inits // W)
    N = n_b * W
    ik_ok = np.zeros((D, N), bool)
    col = np.full((D, N), -2, int)
    lifted = np.zeros((D, N), bool)
    contact = np.empty((D, N), object)
    for bi in range(n_b):
        b = bi * W
        env.episode_idx = b
        env.reset()
        shifts = np.zeros((D, W, 3), np.float32)
        for d, x in enumerate(demos):
            for w in range(W):
                p = env.obs[w][f"{names[x['obj']]}_pos"]
                shifts[d, w] = p - x["obs"][0, obj_slice(x["obj"])]
        print(
            f"inits {b}-{b + W - 1}: obstacle {env.obstacle[0]} "
            f"shift {shifts[0].mean(0).round(3)} spread {shifts[0].std(0).round(3)}"
        )
        q, ok = solve_windows(cfg, demos, chain, base, shifts)
        ik_ok[:, b : b + W] = ok
        for d, x in enumerate(demos):
            ct, dz, lab = replay(cfg, env, x, q[d], b, names)
            col[d, b : b + W] = ct
            lifted[d, b : b + W] = dz > cfg.lift / 2
            contact[d, b : b + W] = lab
            clean = ok[d] & (ct < 0) & (dz > cfg.lift / 2)
            print(
                f"  demo {x['key']}: ik {ok[d].sum()}/{W} nocol {(ct < 0).sum()} "
                f"lifted {(dz > cfg.lift / 2).sum()} clean {clean.sum()} "
                f"col_t {ct.tolist()} {[c.split('/')[1] if c else '' for c in lab]}",
                flush=True,
            )
    ik_ok, col, lifted = (
        ik_ok[:, : cfg.n_inits],
        col[:, : cfg.n_inits],
        lifted[:, : cfg.n_inits],
    )
    clean = ik_ok & (col < 0) & lifted
    nocol = ik_ok & (col < 0)
    per_init = clean.sum(0)
    per_demo = clean.sum(1)
    hits = {}
    for d in range(D):
        for i in range(cfg.n_inits):
            if col[d, i] >= 0 and contact[d, i]:
                k = contact[d, i].split("/")[1]
                hits[k] = hits.get(k, 0) + 1
    print(
        f"RESULT {tag}: pairs {D * cfg.n_inits} ik_ok {ik_ok.sum()} "
        f"nocol {nocol.sum()} clean {clean.sum()} "
        f"| inits with a clean demo {(per_init > 0).sum()}/{cfg.n_inits} "
        f"| demos clean somewhere {(per_demo > 0).sum()}/{D} | contacts {hits} "
        f"| FEASIBLE {int(clean.sum() >= cfg.threshold * D * cfg.n_inits)}"
    )
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    Path(cfg.out, f"{tag}.json").write_text(
        json.dumps(
            dict(
                keys=[x["key"] for x in demos],
                ik_ok=ik_ok.tolist(),
                col_t=col.tolist(),
                lifted=lifted.tolist(),
                contact=contact[:, : cfg.n_inits].tolist(),
            )
        )
    )


if __name__ == "__main__":
    main()
