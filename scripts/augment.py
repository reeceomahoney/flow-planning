"""Offline detour augmentation of a recorded dataset (teleop-compatible).

Bends the transit segments of each episode in EE space (continuous lateral/
vertical arcs + wrist-carriage posture) and re-IKs to joint actions; only the
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
    wrist_lo: float = 0.12  # wrist-carriage height range above the EE
    wrist_hi: float = 0.32
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
    return d


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

    def solve_seq(self, ee_t, quat_t, wrist_t, seed0):
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
            wp.copy(env.wrist_target, wp.array(self.pad(wrist_t[t]), dtype=wp.vec3))
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
    link7 = next(n for n in chain.get_frame_names() if "link7" in n and "tcp" not in n)

    def fk(q):  # (T, 7) joints -> world EE pos, quat xyzw, link7 z
        th = torch.as_tensor(q, dtype=torch.float32, device=device)
        th = torch.cat([th, th.new_zeros(len(th), njoints - 7)], dim=1)
        tf = chain.forward_kinematics(th)
        m = tf[EE_FRAME].get_matrix()
        pos = m[:, :3, 3].cpu().numpy() + base
        quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
        wz = tf[link7].get_matrix()[:, 2, 3].cpu().numpy() + base[2]
        return pos, quat, wz

    src = LeRobotDataset(cfg.src_repo)
    hf = src.hf_dataset.with_format("numpy")
    epi = np.asarray(hf["episode_index"])
    obs_all = np.asarray(hf["observation.state"])
    act_all = np.asarray(hf["action"])
    src_bend = np.asarray(hf["bend"]) if "bend" in src.meta.features else None

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": obs_all.shape[1:],
            "names": None,
        },
        "action": {"dtype": "float32", "shape": act_all.shape[1:], "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "bend": {"dtype": "float32", "shape": (4,), "names": None},
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
                    "next.success": np.ones(1, bool),
                    "bend": bend.astype(np.float32),
                    "task": "pick_place",
                }
            )
        dst.save_episode()

    # pass 1: write originals (measured lift label), FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    for e in tqdm(range(src.num_episodes), desc="originals"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        held = slice(close, opened)
        lift = float(obs[held, EE][:, 2].max() - obs[close, EE.start + 2])
        wz_src = float(src_bend[sel][0, 3]) if src_bend is not None else 0.18
        write(obs, act, np.array([0.0, 0.0, lift, wz_src]))
        if opened - close < 4:
            continue
        act_ee, act_quat, al7 = fk(act[:, :7])
        todo.append(
            dict(
                obs=obs,
                act=act,
                close=close,
                opened=opened,
                act_ee=act_ee,
                act_quat=act_quat,
                awz0=al7 - act_ee[:, 2],
            )
        )

    def tpad(a, tm):  # (T, ...) -> (tm, ...) repeating the last frame
        return np.concatenate([a, np.repeat(a[-1:], tm - len(a), axis=0)])

    def targets(grp, wzs, tm):
        """Bent EE/rot/wrist target stacks (tm, n, .) for the action IK pass."""
        out: dict[str, list] = {"e": [], "q": [], "w": []}
        for x, d, wz in zip(grp, (x["d"] for x in grp), wzs):
            carr = x["awz0"].copy()
            z = x["obs"][:, EE.start + 2]
            up = np.nonzero(z[x["close"] :] > z[x["close"]] + 0.1)[0]
            c = x["close"] + (int(up[0]) if len(up) else 0)  # lift done
            r = 25  # original posture until the lift completes, then ramp in
            carr[c + r :] = wz
            carr[c : c + r] = (
                carr[c] + (wz - carr[c]) * np.arange(r)[: len(carr) - c] / r
            )
            e = x["act_ee"] + d
            w = e.copy()
            w[:, 2] += carr
            for k, v in zip(out, (e, x["act_quat"], w)):
                out[k].append(tpad(v, tm))
        e, q, w = (np.stack(out[k], axis=1) for k in out)
        return e, q, w

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
            n, tm = len(grp), max(len(x["obs"]) for x in grp)
            lats = cfg.bend_amp * rng.uniform(-1.0, 1.0, n)
            verts = cfg.bend_amp * rng.uniform(0.0, 0.8, n)
            wzs = rng.uniform(cfg.wrist_lo, cfg.wrist_hi, n)
            for x, lat, vert, wz in zip(grp, lats, verts, wzs):
                x["d"] = bend_delta(x["obs"][:, EE], x["close"], x["opened"], lat, vert)
                x["label"] = np.array([lat, vert, 0.0, wz])
            aseed = np.stack([x["act"][0, [0, 1, 2, 3, 4, 5, 6, 7, 7]] for x in grp])
            ae, aq, aw = targets(grp, wzs, tm)
            na_j = rig.solve_seq(ae, aq, aw, aseed)

            ready = []
            for i, x in enumerate(grp):
                nf = len(x["obs"])
                achieved, _, _ = fk(na_j[:nf, i])
                err = np.linalg.norm(achieved - (x["act_ee"] + x["d"]), axis=1)
                jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
                if err.max() > 0.05 or jerk > 0.6:
                    rejected.append(x)
                    continue
                na = x["act"].copy()
                na[:, :7] = na_j[:nf, i]
                x["na"], x["err"] = na, err
                ready.append(x)

            if not ready:
                continue
            obs_seq, succ = replay(ready)
            for i, x in enumerate(ready):
                if not succ[i]:
                    rejected.append(x)
                    continue
                nf = len(x["na"])
                no = obs_seq[:nf, i]
                held = slice(x["close"], x["opened"])
                x["label"][2] = no[held, EE][:, 2].max() - no[x["close"], EE.start + 2]
                write(no, x["na"], x["label"])
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
