"""Offline detour augmentation of a recorded dataset (teleop-compatible).

Bends the transit segments of each episode in EE space (continuous lateral/
vertical arcs + wrist-carriage posture), re-IKs to joint actions, and labels
every copy with its latents. Only the gripper signal segments the episode, so
this applies to any pre-existing demo dataset, not just scripted ones.
"""

from dataclasses import dataclass, field

import draccus
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
    ik_iters: int = 40  # seeds are the source joints, so this converges fast
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


def bend_delta(ee, close, opened, lat, vert):
    """Per-frame EE displacement: plateau-profile arc on each transit segment,
    zero at the endpoints so grasp/release are untouched."""
    d = np.zeros_like(ee)
    up = np.array([0.0, 0.0, 1.0])
    for s0, s1 in ((0, close), (close, opened)):
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
    """Batched re-IK through the env's solver: EE pose + wrist-carriage target
    -> arm joints, warm-started from the source joints."""

    def __init__(self, env, iters: int):
        self.env, self.iters = env, iters
        self.W = env.cfg.world_count
        self.dofs = env.joint_q_ik.shape[1]

    def solve(self, ee_t, quat_t, wrist_t, seed_q):
        env, W = self.env, self.W
        out = np.empty((len(ee_t), 7), np.float32)
        for i in range(0, len(ee_t), W):
            j = min(i + W, len(ee_t))

            def pad(a):
                p = np.repeat(a[j - 1 : j], W, axis=0)
                p[: j - i] = a[i:j]
                return np.ascontiguousarray(p, np.float32)

            env.pos_obj.set_target_positions(wp.array(pad(ee_t), dtype=wp.vec3))
            env.rot_obj.set_target_rotations(wp.array(pad(quat_t), dtype=wp.vec4))
            wp.copy(env.wrist_target, wp.array(pad(wrist_t), dtype=wp.vec3))
            seed = np.zeros((W, self.dofs), np.float32)
            seed[:, :9] = pad(seed_q)
            wp.copy(env.joint_q_ik, wp.array(seed, dtype=wp.float32))
            env.ik_solver.step(env.joint_q_ik, env.joint_q_ik, iterations=self.iters)
            out[i:j] = env.joint_q_ik.numpy()[: j - i, :7]
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

    ee_err = []
    for e in tqdm(range(src.num_episodes), desc="augmenting"):
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        held = slice(close, opened)
        lift = float(obs[held, EE][:, 2].max() - obs[close, EE.start + 2])
        wz_src = float(src_bend[sel][0, 3]) if src_bend is not None else 0.18
        write(obs, act, np.array([0.0, 0.0, lift, wz_src]))
        if opened - close < 4:
            continue

        act_ee, act_quat, _ = fk(act[:, :7])
        st_ee, st_quat, _ = fk(obs[:, :7])
        act_seed = np.concatenate([act[:, :7], act[:, 7:8], act[:, 7:8]], axis=1)
        for _ in range(cfg.copies):
            lat = cfg.bend_amp * rng.uniform(-1.0, 1.0)
            vert = cfg.bend_amp * rng.uniform(0.0, 0.8)
            wz = rng.uniform(cfg.wrist_lo, cfg.wrist_hi)
            d = bend_delta(obs[:, EE], close, opened, lat, vert)
            # wrist carriage is neutral until the cube is held (grasp precision)
            wzf = np.full(len(obs), 0.18, np.float32)
            wzf[close:] = wz
            wrist = np.zeros((len(obs), 3), np.float32)
            wrist[:, 2] = wzf

            na, no = act.copy(), obs.copy()
            na[:, :7] = rig.solve(act_ee + d, act_quat, act_ee + d + wrist, act_seed)
            no[:, :7] = rig.solve(st_ee + d, st_quat, st_ee + d + wrist, obs[:, :9])
            no[:, EE] = obs[:, EE] + d
            no[held, CUBE] = obs[held, CUBE] + d[held]  # cube rides the gripper
            nl = float(no[held, EE][:, 2].max() - no[close, EE.start + 2])
            write(no, na, np.array([lat, vert, nl, wz]))

            achieved, _, _ = fk(na[:, :7])
            ee_err.append(np.linalg.norm(achieved - (act_ee + d), axis=1))

    err = np.concatenate(ee_err)
    print(f"re-IK EE error: mean {err.mean():.4f}m  p95 {np.percentile(err, 95):.4f}m")
    dst.finalize()


if __name__ == "__main__":
    main()
