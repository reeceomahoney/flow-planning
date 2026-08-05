"""Offline detour augmentation of a recorded dataset (teleop-compatible).

Bends the transit segments of each episode in EE space (continuous lateral/
vertical arcs) and re-IKs to joint actions; only the
gripper signal segments the episode, so the action-side augmentation applies
to any pre-existing demo dataset. States come from REPLAYING the bent actions
in sim: IK-synthesized states are kinematically right but dynamically sterile
(no tracking lag, no contact), and a policy trained on them collapses when
eval clamps a real state into the plan.
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

EE, CUBE = slice(9, 12), slice(18, 21)  # obs blocks (see FrankaEnv.get_obs)


@dataclass
class Config:
    src_repo: str = "reece-omahoney/franka-src"
    dst_repo: str = "reece-omahoney/franka"
    copies: int = 2  # bent copies per episode; the original is always kept
    bend_amp: float = 1.0  # scales the sampled arc, as in the online recorder
    retime: bool = False  # stretch bent segments so a detour costs TIME not speed
    retime_carry_only: bool = True  # leave the approach clock alone (grasp precision)
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


def bend_delta(ee, close, opened, lat, vert, clear=0.1):
    """Per-frame EE displacement: plateau-profile arc on each transit segment.
    Segments shrink to where the EE is `clear` above its grasp/place height, so
    the approach descent, grasp, lift and place descent stay untouched — the
    geometric analogue of the online recorder's phase gating."""
    d = np.zeros_like(ee)
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
    used = []
    for s0, s1 in segs:
        if s1 - s0 < 4:
            continue
        chord = ee[s1 - 1] - ee[s0]
        latd = np.cross(chord, up)
        n = np.linalg.norm(latd)
        latd = latd / n if n > 1e-4 else np.zeros(3)
        t = np.arange(s1 - s0) / (s1 - s0 - 1)
        prof = np.minimum(1.0, 2.0 * np.sin(np.pi * t))
        arc = lat * latd + vert * up
        d[s0:s1] = np.linalg.norm(chord) * prof[:, None] * arc
        used.append((s0, s1))
    return d, used


def retime_map(n, segs, stretch):
    """New->old fractional frame indices; total length is EXACTLY round(n*stretch),
    with the extra frames split across the bent segments in proportion to their
    length. Only those stretch: a global resampler eats the grasp/place dwell.

    Duration must be a function of the conditioning label ALONE. Deriving it from
    each episode's own path growth (the obvious choice) gives two episodes with
    the same bend different durations; that variance is unexplained by cond and
    the flow averages it into slow-motion plans that never finish -- free space
    0.910 -> 0.461, every point of it timeout. Even a fixed per-SEGMENT factor
    leaks, because the segments are 20-30% of frames and that share varies."""
    total = sum(s1 - s0 for s0, s1 in segs)
    extra = max(0, int(round(n * stretch)) - n)
    u, prev, left = [], 0, extra
    for i, (s0, s1) in enumerate(segs):
        u.extend(range(prev, s0))
        seg = s1 - s0
        add = left if i == len(segs) - 1 else int(round(extra * seg / total))
        left -= add
        m = seg + add
        u.extend(s0 + seg * np.arange(m) / m)
        prev = s1
    u.extend(range(prev, n))
    return np.asarray(u, np.float64)


def interp_rows(a, u):
    """Linear resample of (T, d) at fractional indices u."""
    lo = np.floor(u).astype(int).clip(0, len(a) - 1)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = (u - lo)[:, None]
    return a[lo] * (1 - w) + a[hi] * w


class IKRig:
    """Re-IK through the env's solver, batched over episodes and SEQUENTIAL over
    frames: each frame warm-starts from the previous frame's solution, exactly
    like the online recorder. Independent per-frame solves hop between IK
    posture families and produce unusably jerky joint trajectories."""

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    env = make_env(cfg.env, ViewerNull())
    rig = IKRig(env, cfg.ik_iters)
    base = np.asarray(env.robot_base_pos, np.float32)
    chain, njoints, _ = build_franka_chain(device)

    def fk(q):  # (T, 7) joints -> world EE pos, quat xyzw
        th = torch.as_tensor(q, dtype=torch.float32, device=device)
        th = torch.cat([th, th.new_zeros(len(th), njoints - 7)], dim=1)
        tf = chain.forward_kinematics(th)
        m = tf[EE_FRAME].get_matrix()
        pos = m[:, :3, 3].cpu().numpy() + base
        quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
        return pos, quat

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

    def write(obs, act, bend):
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

    # pass 1: write originals, FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    for e in tqdm(range(src.num_episodes), desc="originals"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        write(obs, act, np.zeros(2))
        if opened - close < 4:
            continue
        # segments depend only on the EE height profile, not on lat/vert, so
        # settle this once: retiming the carry needs a carry to retime, and
        # an episode silently left unstretched is duration the label does not
        # explain (4.3% of episodes; keep them as originals only)
        if cfg.retime and cfg.retime_carry_only:
            if len(bend_delta(obs[:, EE], close, opened, 1.0, 1.0)[1]) < 2:
                continue
        act_ee, act_quat = fk(act[:, :7])
        todo.append(
            dict(
                obs=obs,
                act=act,
                close=close,
                opened=opened,
                act_ee=act_ee,
                act_quat=act_quat,
            )
        )

    def tpad(a, tm):  # (T, ...) -> (tm, ...) repeating the last frame
        return np.concatenate([a, np.repeat(a[-1:], tm - len(a), axis=0)])

    def targets(grp, tm):
        """Bent EE/rot target stacks (tm, n, .) for the action IK pass.
        With retiming each copy has its own length: everything is resampled onto
        the copy's timebase here, and x["u"] carries it to the gripper so the
        whole episode stays on one clock."""
        out: dict[str, list] = {"e": [], "q": []}
        for x in grp:
            e = x["act_ee"] + x["d"]
            q = x["act_quat"]
            u = x["u"]
            if u is not None:
                e = interp_rows(e, u)
                q = interp_rows(q, u)
                q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
            for k, v in zip(out, (e, q)):
                out[k].append(tpad(v, tm))
        return tuple(np.stack(out[k], axis=1) for k in out)

    def replay(grp):
        """Execute each copy's actions open-loop from its episode's initial
        conditions; real dynamics produce the states (and a success filter)."""
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
        newton.eval_fk(
            env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0
        )
        obs_seq = np.empty((tm, n, o0.shape[1]), np.float32)
        for t in range(tm):
            env.apply_action(acts[:, t])
            obs_seq[t] = env.get_obs()
        return obs_seq, env.success()

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
            lats = cfg.bend_amp * rng.uniform(-1.0, 1.0, n)
            verts = cfg.bend_amp * rng.uniform(0.0, 0.8, n)
            for x, lat, vert in zip(grp, lats, verts):
                d, segs = bend_delta(
                    x["obs"][:, EE], x["close"], x["opened"], lat, vert
                )
                x["d"], x["label"] = d, np.array([lat, vert])
                src_ee = x["act_ee"]  # NOT `base`: fk() closes over that
                # Stretch only the CARRY segment (post-lift -> above the place).
                # Both segments still BEND; only the timing of the second moves.
                # Retiming the approach too cost 0.22 free space, all of it at
                # the grasp (21% never picked the cube up vs 6%): the same grasp
                # then appears at varying approach speeds, and multimodal grasp
                # targets interpolate into misses (cf grasp_symmetry, 0.76->0.96
                # when removed). The detour that has to dodge an obstacle happens
                # during the carry anyway.
                # The carry is ~18% of frames and absorbs the whole stretch,
                # so 0.18 total gives it ~(1 + |bend|) growth; 0.5 would make
                # it 3.8x. Episodes without a carry segment are excluded in
                # pass 1, so total duration stays exact in the label.
                rt = segs[1:] if cfg.retime_carry_only else segs
                k = 0.18 if cfg.retime_carry_only else 0.25
                stretch = 1.0 + k * float(np.hypot(lat, vert))
                x["u"] = (
                    retime_map(len(src_ee), rt, stretch) if cfg.retime and rt else None
                )
                x["nf"] = len(x["u"]) if x["u"] is not None else len(src_ee)
                # target EE on the copy's own clock, for the IK tracking gate
                x["ee_t"] = (
                    interp_rows(src_ee + d, x["u"])
                    if x["u"] is not None
                    else src_ee + d
                )
            tm = max(x["nf"] for x in grp)
            aseed = np.stack([x["act"][0, [0, 1, 2, 3, 4, 5, 6, 7, 7]] for x in grp])
            ae, aq = targets(grp, tm)
            na_j = rig.solve_seq(ae, aq, aseed)

            ready = []
            for i, x in enumerate(grp):
                nf = x["nf"]
                achieved, _ = fk(na_j[:nf, i])
                err = np.linalg.norm(achieved - x["ee_t"], axis=1)
                jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
                if err.max() > 0.05 or jerk > 0.6:
                    rejected.append(x)
                    continue
                na = (
                    interp_rows(x["act"], x["u"])
                    if x["u"] is not None
                    else x["act"].copy()
                )
                na[:, :7] = na_j[:nf, i]
                x["na"], x["err"] = na.astype(np.float32), err
                ready.append(x)

            if not ready:
                continue
            obs_seq, succ = replay(ready)
            for i, x in enumerate(ready):
                if not succ[i]:
                    rejected.append(x)
                    continue
                nf = len(x["na"])
                write(obs_seq[:nf, i], x["na"], x["label"])
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
