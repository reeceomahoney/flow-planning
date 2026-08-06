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
from tqdm import tqdm

from flow_planning.envs import FrankaConfig, make_env
from flow_planning.kinematics import EE_FRAME, build_franka_chain

EE = slice(9, 12)  # obs block (see FrankaEnv.get_obs)


@dataclass
class Config:
    src_repo: str = "reece-omahoney/franka-src"
    dst_repo: str = "reece-omahoney/franka"
    copies: int = 2  # bent copies per episode; the original is always kept
    bend_amp: float = 1.05  # max arc half-angle in rad (~60 deg, 1.21x path)
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


def bend_delta(ee, close, opened, phi, theta, clear=0.1):
    """Per-frame EE displacement: arc of radius L/(2 sin phi) on each transit
    segment of chord L, bulging in a plane `theta` off the floor (0 = lateral,
    pi/2 = up). Peak bulge is L/2 * tan(phi/2), path grows ~phi/sin(phi).
    Segments stop `clear` above the grasp/place height, leaving the descents,
    grasp and lift untouched (the online recorder's phase gating, geometrically)."""
    d = np.zeros_like(ee)
    if phi < 1e-3:
        return d
    up = np.array([0.0, 0.0, 1.0])
    z = ee[:, 2]
    hi1 = np.nonzero(z[:close] > z[max(close - 1, 0)] + clear)[0]
    hi2 = np.nonzero(z[close:opened] > z[opened - 1] + clear)[0] + close
    lifted2 = np.nonzero(z[close:opened] > z[close] + clear)[0] + close
    segs = []
    if len(hi1):
        segs.append((0, int(hi1[-1])))  # home -> just above the grasp
    if len(hi2) and len(lifted2):  # lift done -> just above the place
        segs.append((int(lifted2[0]), int(hi2[-1])))
    for s0, s1 in segs:
        if s1 - s0 < 4:
            continue
        chord = ee[s1 - 1] - ee[s0]
        latd = np.cross(chord, up)
        n = np.linalg.norm(latd)
        latd = latd / n if n > 1e-4 else np.zeros(3)
        psi = phi * (2.0 * np.arange(s1 - s0) / (s1 - s0 - 1) - 1.0)  # [-phi, phi]
        sag = np.linalg.norm(chord) / 2 * (np.cos(psi) - np.cos(phi)) / np.sin(phi)
        d[s0:s1] = sag[:, None] * (np.cos(theta) * latd + np.sin(theta) * up)
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
    dst = LeRobotDataset.create(
        repo_id=cfg.dst_repo, fps=src.fps, features=features, use_videos=False
    )

    # pass 1: write originals, FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    for e in tqdm(range(src.num_episodes), desc="originals"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        write(dst, obs, act, np.zeros(2))
        if opened - close < 4:
            continue
        act_ee, act_quat = fk(chain, act[:, :7], base)
        todo.append(
            dict(obs=obs, act=act, close=close, opened=opened, ee=act_ee, quat=act_quat)
        )

    # pass 2: bend + re-IK actions, then replay in sim for real states. Copies
    # that fail either gate — IK can't track the arc (unreachable / posture
    # flip) or the replayed rollout fails the task — get their latents
    # resampled, like the recorder's success filter.
    ee_err: list[np.ndarray] = []
    written = 0
    pending = [dict(x) for x in todo for _ in range(cfg.copies)]
    for attempt in range(4):
        rejected = []
        for g in tqdm(range(0, len(pending), rig.W), desc=f"bend (try {attempt})"):
            grp = pending[g : g + rig.W]
            n = len(grp)
            phis = rng.uniform(0.0, cfg.bend_amp, n)
            thetas = rng.uniform(0.0, np.pi, n)  # upper half only: never dip
            for x, phi, theta in zip(grp, phis, thetas):
                d = bend_delta(x["obs"][:, EE], x["close"], x["opened"], phi, theta)
                # stored Cartesian: theta wraps and degenerates at phi=0, which
                # would break the FPS/whitening candidate search in eval.py
                x["label"] = phi * np.array([np.cos(theta), np.sin(theta)])
                x["ee_t"] = x["ee"] + d
            tm = max(len(x["act"]) for x in grp)
            seed = np.stack([x["act"][0, [*range(7), 7, 7]] for x in grp])  # 7 + 2 tips
            na_j = rig.solve_seq(
                np.stack([tpad(x["ee_t"], tm) for x in grp], axis=1),
                np.stack([tpad(x["quat"], tm) for x in grp], axis=1),
                seed,
            )

            ready = []
            for i, x in enumerate(grp):
                nf = len(x["act"])
                achieved, _ = fk(chain, na_j[:nf, i], base)
                err = np.linalg.norm(achieved - x["ee_t"], axis=1)
                jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
                if err.max() > 0.05 or jerk > 0.6:
                    rejected.append(x)
                    continue
                na = x["act"].copy()
                na[:, :7] = na_j[:nf, i]
                x["na"], x["err"] = na.astype(np.float32), err
                ready.append(x)

            if not ready:
                continue
            obs_seq, succ = replay(env, rig, ready)
            for i, x in enumerate(ready):
                if not succ[i]:
                    rejected.append(x)
                    continue
                nf = len(x["na"])
                write(dst, obs_seq[:nf, i], x["na"], x["label"])
                written += 1
                ee_err.append(x["err"])
        pending = rejected
        if not pending:
            break

    err = np.concatenate(ee_err)
    total = written + len(pending)
    print(f"wrote {written}/{total} copies ({len(pending)} dropped after retries)")
    print(f"re-IK EE error: mean {err.mean():.4f}m  p95 {np.percentile(err, 95):.4f}m")
    dst.finalize()


if __name__ == "__main__":
    main()
