from dataclasses import dataclass, field
from typing import Any

import draccus
import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pytorch_kinematics.transforms import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from scipy.signal import lfilter, lfilter_zi
from tqdm import tqdm

from flow_planning.bend import ARM, CUBE, EE, ROT, bend_delta, transit_segments
from flow_planning.envs import FrankaConfig
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.utils import hf_column, quat_to_rot6d


@dataclass
class Config:
    src_repo: str = "reece-omahoney/franka-src"
    dst_repo: str = "reece-omahoney/franka"
    copies: int = 2  # bent copies per episode; the original is always kept
    bend_min: float = 0.15  # half-angle, units of pi
    bend_max: float = 0.75
    bend_margin: float = 1.5
    limit: int = 0
    ik_iters: int = 8  # per frame, warm-started
    ik_damping: float = 0.05
    seed: int = 0
    env: FrankaConfig = field(default_factory=FrankaConfig)  # table geometry only


DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.linspace(0.0, 0.98, 25)


def fk(chain, q, base):  # (T, 7) joints -> world EE pos, quat xyzw
    th = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=DEV)
    m = chain.forward_kinematics(th).get_matrix()
    pos = m[:, :3, 3].cpu().numpy() + base
    quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
    return pos, quat


def solve_seq(chain, ee_t, quat_t, seed0, iters, damping, base):
    """(T, B, 3/4) world targets + (B, 7) initial joints -> (T, B, 7).

    Damped least squares, batched over copies and SEQUENTIAL over frames: each
    frame warm-starts from the last. Independent solves hop posture families
    and are unusably jerky."""

    lo, hi = (torch.tensor(x, device=DEV) for x in chain.get_joint_limits())
    eye = damping * torch.eye(6, device=DEV)
    p = torch.as_tensor(ee_t - base, dtype=torch.float32, device=DEV)
    q = torch.as_tensor(quat_t[..., [3, 0, 1, 2]], dtype=torch.float32, device=DEV)
    rot = quaternion_to_matrix(q)
    th = torch.as_tensor(seed0, dtype=torch.float32, device=DEV)
    out = torch.empty(p.shape[:2] + (7,), device=DEV)
    for t in range(len(p)):
        # IK loop
        for _ in range(iters):
            jac, m = chain.jacobian(th, ret_eef_pose=True)
            err = torch.cat(
                [
                    p[t] - m[:, :3, 3],
                    matrix_to_axis_angle(rot[t] @ m[:, :3, :3].transpose(1, 2)),
                ],
                dim=-1,
            )
            jt = jac.transpose(1, 2)
            th = th + (jt @ torch.linalg.solve(jac @ jt + eye, err[..., None]))[..., 0]
            th = th.clamp(lo, hi)
        out[t] = th
    return out.cpu().numpy()


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


def smooth(a, alpha):
    b, p = [1 - alpha], [1, -alpha]
    return lfilter(b, p, a, axis=0, zi=lfilter_zi(b, p)[:, None] * a[:1])[0]


def track_error(obs, act):
    o, a = obs[:, ARM], act[:, ARM]
    return np.array([np.abs(smooth(a, al) - o).mean() for al in ALPHAS])


def lag_states(x, chain, base, alpha):
    # add the filtered action delta onto the recorded arm joints
    obs = x["obs"]
    dq = smooth(x["na"][:, ARM] - x["act"][:, ARM], alpha)
    new = obs.copy()
    new[:, ARM] += dq
    ee, quat = fk(chain, new[:, ARM], base)
    new[:, EE], new[:, ROT] = ee, quat_to_rot6d(quat)

    # shift the cube by the EE displacement over the frames it is held
    held = slice(x["close"], x["opened"])
    new[held, CUBE] += ee[held] - obs[held, EE]
    return new.astype(np.float32)


@draccus.wrap()
def main(cfg: Config):
    rng = np.random.default_rng(cfg.seed)

    # build kinematic chain
    base = np.array([-0.5, -0.5, cfg.env.table_height], np.float32)
    chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(chain, EE_FRAME).to(dtype=torch.float32, device=DEV)

    # load datasets
    src = LeRobotDataset(cfg.src_repo)
    hf = src.hf_dataset
    epi = hf_column(hf, "episode_index")
    obs_all = hf_column(hf, "observation.state")
    act_all = hf_column(hf, "action")

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

    # write originals, FK-cache bendable episodes
    todo: list[dict[str, Any]] = []
    grid = np.zeros(len(ALPHAS))
    margin = round(cfg.bend_margin * src.fps)
    n_src = min(src.num_episodes, cfg.limit or src.num_episodes)
    for e in tqdm(range(n_src), desc="originals"):
        # write original episode
        sel = epi == e
        obs, act = obs_all[sel].copy(), act_all[sel].copy()
        close, opened = transit_segments(act[:, 7])
        write(dst, obs, act, np.zeros(2))
        grid += track_error(obs, act) / n_src

        # fk bendable episodes
        if opened - close < 2 * margin:
            continue
        act_ee, act_quat = fk(chain, act[:, :7], base)
        todo.append(
            dict(obs=obs, act=act, close=close, opened=opened, ee=act_ee, quat=act_quat)
        )

    # take the alpha with the lowest prediction error
    j = int(grid.argmin())
    alpha = float(ALPHAS[j])
    print(f"tracking fit: alpha {alpha:.2f}")
    print(f"  joint MAE {grid[j]:.5f} rad vs {grid[0]:.5f} unfiltered")
    if j in (0, len(ALPHAS) - 1):
        print("  WARNING: optimum on a grid edge")

    # report the same fit in EE space, against doing no filtering at all
    for name, a_ in (("unfiltered", 0.0), ("fitted", alpha)):
        d = np.concatenate(
            [
                fk(chain, smooth(x["act"][:, ARM], a_), base)[0] - x["obs"][:, EE]
                for x in todo
            ]
        )
        print(f"  {name:>10} EE MAE {1e3 * np.linalg.norm(d, axis=1).mean():.1f} mm")

    # queue up `copies` bends per episode and retry the rejects up to 4 times
    ee_err: list[np.ndarray] = []
    written = rej_ik = 0
    pending = [dict(x) for x in todo for _ in range(cfg.copies)]
    bar = tqdm(total=len(pending), desc="bends")
    for attempt in range(4):
        rejected = []
        bar.write(f"attempt {attempt}: {len(pending)} pending")

        # sample arcs
        n = len(pending)
        phis = np.pi * rng.uniform(cfg.bend_min, cfg.bend_max, n)
        thetas = rng.uniform(0.0, np.pi, n)
        for x, phi, theta in zip(pending, phis, thetas):
            d = bend_delta(
                x["obs"][:, EE], x["close"] + margin, x["opened"] - margin, phi, theta
            )
            x["label"] = phi * np.array([np.cos(theta), np.sin(theta)])
            x["ee_t"] = x["ee"] + d
            x["bent"] = np.linalg.norm(d, axis=1) > 1e-6

        # solve IK to get joints
        tm = max(len(x["act"]) for x in pending)
        na_j = solve_seq(
            chain,
            np.stack([tpad(x["ee_t"], tm) for x in pending], axis=1),
            np.stack([tpad(x["quat"], tm) for x in pending], axis=1),
            np.stack([x["act"][0, :7] for x in pending]),
            cfg.ik_iters,
            cfg.ik_damping,
            base,
        )

        # measure what the IK actually achieved and drop copies that miss or jerk
        achieved = fk(chain, na_j.reshape(-1, 7), base)[0].reshape(tm, n, 3)
        for i, x in enumerate(pending):
            nf = len(x["act"])
            err = np.linalg.norm(achieved[:nf, i] - x["ee_t"], axis=1)
            jerk = np.abs(np.diff(na_j[:nf, i], n=2, axis=0)).max()
            if err.max() > 0.05 or jerk > 0.6:
                rejected.append(x)
                rej_ik += 1
                continue

            # splice the new arm joints in over the original gripper channel
            na = x["act"].copy()
            na[:, :7] = na_j[:nf, i]
            x["na"] = na.astype(np.float32)
            write(dst, lag_states(x, chain, base, alpha), na, x["label"])
            written += 1
            bar.update(1)
            ee_err.append(err[x["bent"]])
        pending = rejected
        if not pending:
            break
    bar.close()

    err = np.concatenate(ee_err)
    total = written + len(pending)
    print(f"wrote {written}/{total} copies ({len(pending)} dropped after retries)")
    print(f"rejected: {rej_ik} on IK/jerk (re-sampled, not lost)")
    print(
        f"bent-frame IK error: mean {err.mean():.4f} p95 {np.percentile(err, 95):.4f}"
    )
    dst.finalize()


if __name__ == "__main__":
    main()
