"""Offline detour augmentation of a recorded dataset (teleop-compatible).

Bends each episode's transit segments into EE-space arcs and re-IKs to joint
actions; only the gripper signal segments the episode, so this applies to any
pre-existing demos. States come from REPLAYING the bent actions in sim —
IK-synthesized states are kinematically right but dynamically sterile (no
tracking lag, no contact) and collapse training silently.
"""

from dataclasses import dataclass, field
from typing import Any

import draccus
import newton
import numpy as np
import torch
import warp as wp
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from newton.viewer import ViewerNull
from pytorch_kinematics.transforms import matrix_to_quaternion
from scipy.signal import filtfilt
from tqdm import tqdm

from flow_planning.envs import FrankaConfig, make_env
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.utils import quat_to_rot6d

ARM, EE, ROT, CUBE = slice(0, 7), slice(9, 12), slice(12, 18), slice(18, 21)


@dataclass
class Config:
    src_repo: str = "reece-omahoney/franka-src"
    dst_repo: str = "reece-omahoney/franka"
    lag_repo: str = ""
    copies: int = 2  # bent copies per episode; the original is always kept
    bend_min: float = 0.3
    bend_amp: float = 1.28
    bend_margin: float = 0.3
    limit: int = 0
    ik_iters: int = 24  # per frame, warm-started; matches the online recorder
    seed: int = 0
    env: FrankaConfig = field(default_factory=FrankaConfig)  # IK rig only


def transit_segments(gripper: np.ndarray) -> tuple[int, int]:
    """(close, open) frame indices from the gripper command alone."""
    closed = gripper < 0.03
    if not closed.any():
        return 0, 0
    close = int(closed.argmax())
    after = ~closed[close:]
    opened = close + int(after.argmax()) if after.any() else len(gripper)
    return close, opened


def bend_delta(ee, close, opened, phi, theta, margin):
    d = np.zeros_like(ee)
    if phi < 1e-3:
        return d
    up = np.array([0.0, 0.0, 1.0])
    segs = [(0, close - margin), (close + margin, opened - margin)]
    for s0, s1 in segs:
        if s1 - s0 < 4:
            continue
        p0, chord = ee[s0], ee[s1 - 1] - ee[s0]
        latd = np.cross(chord, up)
        n = np.linalg.norm(latd)
        latd = latd / n if n > 1e-4 else np.zeros(3)
        t = np.arange(s1 - s0) / (s1 - s0 - 1)
        psi = phi * (2.0 * t - 1.0)
        sag = np.linalg.norm(chord) / 2 * (np.cos(psi) - np.cos(phi)) / np.sin(phi)
        nhat = np.cos(theta) * latd + np.sin(theta) * up
        line = p0 + t[:, None] * chord
        clear = ((ee[s0:s1] - line) @ up).max()
        fill = max(0.0, clear - sag.max() * nhat[2])
        d[s0:s1] = line + sag[:, None] * nhat + fill * (sag / sag.max())[:, None] * up
        d[s0:s1] -= ee[s0:s1]
    return d


def fk(chain, q, base):  # (T, 7) joints -> world EE pos, quat xyzw
    njoints = len(chain.get_joint_parameter_names())
    th = torch.as_tensor(q, dtype=torch.float32, device=chain.device)
    th = torch.cat([th, th.new_zeros(len(th), njoints - 7)], dim=1)
    m = chain.forward_kinematics(th)[EE_FRAME].get_matrix()
    pos = m[:, :3, 3].cpu().numpy() + base
    quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
    return pos, quat


def tpad(a, tm):  # (T, ...) -> (tm, ...) repeating the last frame
    return np.concatenate([a, np.repeat(a[-1:], tm - len(a), axis=0)])


def write(dst, obs, act, bend):
    for f in range(len(obs)):
        dst.add_frame(
            {
                "observation.state": obs[f],
                "action": act[f],
                "bend": bend.astype(np.float32),
                "task": "pick_place",
            }
        )
    dst.save_episode()


ALPHAS = np.linspace(0.0, 0.95, 20)


def smooth(a, alpha):
    return filtfilt([1 - alpha], [1, -alpha], a, axis=0)


def lag_error(obs, act, kmax=6):
    o, a = obs[:, ARM], act[:, ARM]
    pos = np.array(
        [np.abs(o[k:] - a[: len(a) - k or None]).mean() for k in range(kmax)]
    )
    jerk = np.abs(np.diff(o, n=3, axis=0)).max()
    jfit = np.array(
        [abs(np.abs(np.diff(smooth(a, al), n=3, axis=0)).max() - jerk) for al in ALPHAS]
    )
    return pos, jfit


def lag_states(x, chain, base, lag, alpha):
    obs, dq = x["obs"], x["na"][:, ARM] - x["act"][:, ARM]
    if lag:
        dq = np.concatenate([np.zeros((lag, 7), np.float32), dq[:-lag]])
    dq = smooth(dq, alpha)
    new = obs.copy()
    new[:, ARM] += dq
    ee, quat = fk(chain, new[:, ARM], base)
    new[:, EE], new[:, ROT] = ee, quat_to_rot6d(quat)
    held = slice(x["close"], x["opened"])
    new[held, CUBE] += ee[held] - obs[held, EE]
    return new.astype(np.float32)


def replay(env, rig, grp):
    """Open-loop rollout from each copy's initial conditions -> states + success."""
    n = env.cfg.world_count
    tm = max(len(x["na"]) for x in grp)
    acts = rig.pad(np.stack([tpad(x["na"], tm) for x in grp]))
    jq = np.zeros((n, env.coords_per_world), np.float32)
    o0 = rig.pad(np.stack([x["obs"][0] for x in grp]))
    jq[:, :9] = o0[:, :9]
    jq[:, 9:12] = o0[:, 18:21]
    yaw = np.arctan2(o0[:, 21], o0[:, 22])
    jq[:, 14] = np.sin(yaw / 2)
    jq[:, 15] = np.cos(yaw / 2)
    env.reset()
    env.goal_pos = o0[:, 23:26]
    env.cube_pos_prev = o0[:, 18:21].copy()
    env.cube_start_z = o0[:, 20].copy()
    wp.copy(env.state_0.joint_q, wp.array(jq.flatten(), dtype=wp.float32))
    env.state_0.joint_qd.zero_()
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    obs_seq = np.empty((tm, n, o0.shape[1]), np.float32)
    for t in range(tm):
        env.apply_action(acts[:, t])
        obs_seq[t] = env.get_obs()
    return obs_seq, env.success()


class IKRig:
    """Env IK solver, batched over episodes and SEQUENTIAL over frames: each frame
    warm-starts from the last. Independent solves hop posture families and are
    unusably jerky."""

    def __init__(self, env, iters: int):
        self.env, self.iters = env, iters
        self.W = env.cfg.world_count
        self.dofs = env.joint_q_ik.shape[1]

    def pad(self, a):  # (B, ...) -> (W, ...) by repeating the last row
        p = np.repeat(a[-1:], self.W, axis=0)
        p[: len(a)] = a
        return np.ascontiguousarray(p, np.float32)

    def solve_seq(self, ee_t, quat_t, seed0):
        """(T, B, 3/4) target sequences + (B, 9) initial joints -> (T, B, 7)."""
        env = self.env
        T, B = ee_t.shape[:2]
        seed = np.zeros((self.W, self.dofs), np.float32)
        seed[:, :9] = self.pad(seed0)
        wp.copy(env.joint_q_ik, wp.array(seed, dtype=wp.float32))
        out = np.empty((T, B, 7), np.float32)
        for t in range(T):
            env.pos_obj.set_target_positions(wp.array(self.pad(ee_t[t]), dtype=wp.vec3))
            env.rot_obj.set_target_rotations(
                wp.array(self.pad(quat_t[t]), dtype=wp.vec4)
            )
            env.ik_solver.step(env.joint_q_ik, env.joint_q_ik, iterations=self.iters)
            out[t] = env.joint_q_ik.numpy()[:B, :7]
        return out


@draccus.wrap()
def main(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    env = make_env(cfg.env, ViewerNull())
    rig = IKRig(env, cfg.ik_iters)
    base = np.asarray(env.robot_base_pos, np.float32)
    chain, _, _ = build_franka_chain("cuda" if torch.cuda.is_available() else "cpu")

    src = LeRobotDataset(cfg.src_repo)
    hf = src.hf_dataset.with_format("numpy")
    epi = np.asarray(hf["episode_index"])
    obs_all = np.asarray(hf["observation.state"])
    act_all = np.asarray(hf["action"])

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": obs_all.shape[1:],
            "names": None,
        },
        "action": {"dtype": "float32", "shape": act_all.shape[1:], "names": None},
        "bend": {"dtype": "float32", "shape": (2,), "names": None},
    }

    def make(r):
        return LeRobotDataset.create(
            repo_id=r, fps=src.fps, features=features, use_videos=False
        )

    dsts = [make(cfg.dst_repo)] + ([make(cfg.lag_repo)] if cfg.lag_repo else [])

    # pass 1: write originals, FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    lag_err, jerk_err = np.zeros(6), np.zeros(len(ALPHAS))
    margin = round(cfg.bend_margin * src.fps)
    n_src = min(src.num_episodes, cfg.limit or src.num_episodes)
    for e in tqdm(range(n_src), desc="originals"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        for d in dsts:
            write(d, obs, act, np.zeros(2))
        p, j = lag_error(obs, act)
        lag_err += p
        jerk_err += j
        if opened - close < 2 * margin:
            continue
        act_ee, act_quat = fk(chain, act[:, :7], base)
        todo.append(
            dict(obs=obs, act=act, close=close, opened=opened, ee=act_ee, quat=act_quat)
        )

    # pass 2: bend + re-IK actions, then replay in sim for real states. Copies
    # that fail either gate — IK can't track the arc (unreachable / posture
    # flip) or the replayed rollout fails the task — get their latents
    # resampled, like the recorder's success filter.
    lag, alpha = int(lag_err.argmin()), float(ALPHAS[jerk_err.argmin()])
    print(f"tracking fit: lag {lag} frames, alpha {alpha:.2f}")

    ee_err: list[np.ndarray] = []
    written = rej_ik = rej_sim = 0
    pending = [dict(x) for x in todo for _ in range(cfg.copies)]
    bar = tqdm(total=len(pending), desc="bends")
    for attempt in range(4):
        rejected = []
        bar.set_postfix(attempt=attempt, retrying=len(pending))
        for g in range(0, len(pending), rig.W):
            grp = pending[g : g + rig.W]
            n = len(grp)
            phis = 2.0 * np.arctan(2.0 * rng.uniform(cfg.bend_min, cfg.bend_amp, n))
            thetas = rng.uniform(0.0, np.pi, n)  # upper half only: never dip
            for x, phi, theta in zip(grp, phis, thetas):
                d = bend_delta(
                    x["obs"][:, EE], x["close"], x["opened"], phi, theta, margin
                )
                # stored Cartesian: theta wraps and degenerates at phi=0, which
                # would break the FPS/whitening candidate search in eval.py
                x["label"] = phi * np.array([np.cos(theta), np.sin(theta)])
                x["ee_t"] = x["ee"] + d
                x["bent"] = np.linalg.norm(d, axis=1) > 1e-6
            tm = max(len(x["act"]) for x in grp)
            seed = np.stack([x["act"][0, [*range(7), 7, 7]] for x in grp])  # 7 + 2 tips
            na_j = rig.solve_seq(
                np.stack([tpad(x["ee_t"], tm) for x in grp], axis=1),
                np.stack([tpad(x["quat"], tm) for x in grp], axis=1),
                seed,
            )

            ready = []
            achieved = fk(chain, na_j.reshape(-1, 7), base)[0].reshape(tm, n, 3)
            for i, x in enumerate(grp):
                nf = len(x["act"])
                err = np.linalg.norm(achieved[:nf, i] - x["ee_t"], axis=1)
                jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
                if err.max() > 0.05 or jerk > 0.6:
                    rejected.append(x)
                    rej_ik += 1
                    continue
                na = x["act"].copy()
                na[:, :7] = na_j[:nf, i]
                x["na"], x["err"] = na.astype(np.float32), err[x["bent"]]
                ready.append(x)

            if not ready:
                continue
            obs_seq, succ = replay(env, rig, ready)
            for i, x in enumerate(ready):
                if not succ[i]:
                    rejected.append(x)
                    rej_sim += 1
                    continue
                nf = len(x["na"])
                write(dsts[0], obs_seq[:nf, i], x["na"], x["label"])
                if cfg.lag_repo:
                    write(
                        dsts[1],
                        lag_states(x, chain, base, lag, alpha),
                        x["na"],
                        x["label"],
                    )
                written += 1
                bar.update(1)
                ee_err.append(x["err"])
        pending = rejected
        if not pending:
            break
    bar.close()

    err = np.concatenate(ee_err)
    total = written + len(pending)
    print(f"wrote {written}/{total} copies ({len(pending)} dropped after retries)")
    print(f"rejected: {rej_ik} on IK/jerk, {rej_sim} on replay (re-sampled, not lost)")
    print(
        f"bent-frame IK error: mean {err.mean():.4f} p95 {np.percentile(err, 95):.4f}"
    )
    for d in dsts:
        d.finalize()


if __name__ == "__main__":
    main()
