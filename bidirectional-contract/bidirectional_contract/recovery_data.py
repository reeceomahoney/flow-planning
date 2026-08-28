"""Construct recovery supervision for the controller's intervention support.

The controller owns the outward ramp-and-hold.  Training windows therefore
start at a controller-reachable offset and contract back to the demonstration
while demonstration phase continues to advance.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import newton.viewer
import numpy as np
import pytorch_kinematics as pk
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from flow_planning.bend import ARM, EE, ROT, fk, grasp_segments, held_object, obj_slice
from flow_planning.envs.libero import LiberoConfig, LiberoEnv
from flow_planning.kinematics import EE_FRAME, build_franka_chain
from flow_planning.utils import hf_column, quat_to_rot6d, rot6d_to_quat

from .interventions import (
    local_normal_frame,
    normal_direction,
    recovery_profile,
    solve_recovery_ik,
)


@dataclass(frozen=True)
class RecoveryBankConfig:
    repo_id: str = "local/safelibero-spatial-task2-base"
    dataset_root: Path = Path("slop/goal/data/spatial_task2_base")
    output: Path = Path("bidirectional-contract/artifacts/recovery_windows.npz")
    suite: str = "safelibero_spatial"
    task_id: int = 2
    horizon: int = 50
    directions: int = 5
    amplitudes: tuple[float, ...] = (0.03, 0.06, 0.09, 0.12)
    approach_anchors: int = 1
    carry_anchors: int = 4
    approach_recovery: int = 20
    carry_recovery: int = 30
    contact_margin: int = 4
    ik_iters: int = 15
    ik_damping: float = 0.05
    posture_gain: float = 0.1
    ik_tolerance: float = 0.008
    rotation_tolerance_deg: float = 12.0
    max_joint_step: float = 0.18
    limit: int = 0
    device: str = "cuda"


def indices_between(start: int, stop: int, count: int) -> list[int]:
    if count <= 0 or stop < start:
        return []
    return np.unique(np.rint(np.linspace(start, stop, count)).astype(int)).tolist()


def build_recovery_bank(config: RecoveryBankConfig) -> dict:
    device = config.device if torch.cuda.is_available() else "cpu"
    if config.directions < 2:
        raise ValueError("directions must be at least two")
    if not config.amplitudes or min(config.amplitudes) <= 0.0:
        raise ValueError("amplitudes must contain positive distances in metres")

    dataset = LeRobotDataset(
        config.repo_id, root=Path(config.dataset_root).resolve()
    )
    hf = dataset.hf_dataset
    obs_all = np.asarray(hf_column(hf, "observation.state"), np.float32)
    act_all = np.asarray(hf_column(hf, "action"), np.float32)
    episode_all = np.asarray(hf_column(hf, "episode_index"), np.int64)
    episode_ids = np.unique(episode_all)
    if config.limit:
        episode_ids = episode_ids[: config.limit]

    env = LiberoEnv(
        LiberoConfig(
            suite=config.suite,
            task_id=config.task_id,
            world_count=1,
            obstacle=False,
        ),
        newton.viewer.ViewerNull(),
    )
    base = np.asarray(env.robot_base_pos, np.float32)
    # SerialChain construction combines fixed transforms.  Construct all of
    # them on one device, then move the finished subchain to the accelerator.
    root_chain, _, _ = build_franka_chain("cpu")
    chain = pk.SerialChain(root_chain, EE_FRAME).to(
        dtype=torch.float32, device=device
    )

    windows_obs: list[np.ndarray] = []
    windows_act: list[np.ndarray] = []
    # demo, anchor, direction, amplitude, recovery_frames, phase,
    # maximum Cartesian error, terminal joint error
    metadata: list[tuple[int, int, int, float, int, int, float, float]] = []
    rejected = {"ik": 0, "rotation": 0, "velocity": 0}

    for demo_id in episode_ids:
        selected = episode_all == demo_id
        obs_ref = obs_all[selected]
        act_ref = act_all[selected]
        total = len(obs_ref)
        segments = grasp_segments(act_ref[:, 7])
        if not segments:
            continue
        close, opened = max(segments, key=lambda segment: segment[1] - segment[0])
        held = held_object(obs_ref, close, opened, len(env.objects))
        if held is None:
            continue

        q_ref = obs_ref[:, ARM]
        ee_ref, quat_ref = fk(chain, q_ref, base)
        side, vertical = local_normal_frame(ee_ref)
        pre_stop = close - config.contact_margin - config.approach_recovery
        anchors: list[tuple[int, int, int]] = [
            (anchor, config.approach_recovery, 0)
            for anchor in indices_between(1, pre_stop, config.approach_anchors)
        ]
        carry_start = close + config.contact_margin
        carry_stop = opened - config.contact_margin - config.carry_recovery
        anchors += [
            (anchor, config.carry_recovery, 1)
            for anchor in indices_between(
                carry_start, carry_stop, config.carry_anchors
            )
        ]

        candidates = []
        for anchor, recovery_frames, phase in anchors:
            ref_index = np.minimum(
                anchor + np.arange(config.horizon + 1), total - 1
            )
            rho = recovery_profile(config.horizon + 1, recovery_frames)
            for direction_index in range(config.directions):
                direction = normal_direction(
                    side[ref_index],
                    vertical[ref_index],
                    direction_index,
                    config.directions,
                )
                for amplitude in config.amplitudes:
                    candidates.append(
                        {
                            "anchor": anchor,
                            "recovery_frames": recovery_frames,
                            "phase": phase,
                            "direction": direction_index,
                            "amplitude": amplitude,
                            "ref_index": ref_index,
                            "target_pos": (
                                ee_ref[ref_index]
                                + amplitude * rho[:, None] * direction
                            ),
                            "target_quat": quat_ref[ref_index],
                        }
                    )
        if not candidates:
            continue

        target_pos = np.stack([row["target_pos"] for row in candidates], axis=1)
        target_quat = np.stack(
            [row["target_quat"] for row in candidates], axis=1
        )
        seeds = np.stack([q_ref[row["anchor"]] for row in candidates])
        reference_q = np.stack(
            [q_ref[row["ref_index"]] for row in candidates], axis=1
        )
        q_solved = solve_recovery_ik(
            chain,
            target_pos,
            target_quat,
            reference_q,
            seeds,
            config.ik_iters,
            config.ik_damping,
            config.posture_gain,
            base,
        )

        for candidate_index, candidate in enumerate(candidates):
            q = q_solved[:, candidate_index]
            actual_pos, actual_quat = fk(chain, q, base)
            position_error = float(
                np.linalg.norm(actual_pos - candidate["target_pos"], axis=1).max()
            )
            rotation_error = Rotation.from_quat(actual_quat) * Rotation.from_quat(
                candidate["target_quat"]
            ).inv()
            max_rotation = float(np.degrees(rotation_error.magnitude().max()))
            max_joint_step = float(np.abs(np.diff(q, axis=0)).max())
            if position_error > config.ik_tolerance:
                rejected["ik"] += 1
                continue
            if max_rotation > config.rotation_tolerance_deg:
                rejected["rotation"] += 1
                continue
            if max_joint_step > config.max_joint_step:
                rejected["velocity"] += 1
                continue

            ref_index = candidate["ref_index"]
            states = obs_ref[ref_index[: config.horizon]].copy()
            states[:, ARM] = q[: config.horizon]
            reference_fk_pos = ee_ref[ref_index[: config.horizon]]
            executed_delta = actual_pos[: config.horizon] - reference_fk_pos
            states[:, EE] += executed_delta.astype(np.float32)
            reference_fk_rot = Rotation.from_quat(
                quat_ref[ref_index[: config.horizon]]
            )
            executed_rot = Rotation.from_quat(actual_quat[: config.horizon])
            libero_rot = Rotation.from_quat(rot6d_to_quat(states[:, ROT]))
            states[:, ROT] = quat_to_rot6d(
                (executed_rot * reference_fk_rot.inv() * libero_rot).as_quat()
            )
            if candidate["phase"] == 1:
                states[:, obj_slice(held)] += executed_delta.astype(np.float32)

            actions = act_ref[ref_index[: config.horizon]].copy()
            actions[:, ARM] = q[1 : config.horizon + 1]
            terminal_joint_error = float(
                np.linalg.norm(q[-1] - q_ref[ref_index[-1]])
            )
            windows_obs.append(states.astype(np.float32))
            windows_act.append(actions.astype(np.float32))
            metadata.append(
                (
                    int(demo_id),
                    candidate["anchor"],
                    candidate["direction"],
                    candidate["amplitude"],
                    candidate["recovery_frames"],
                    candidate["phase"],
                    position_error,
                    terminal_joint_error,
                )
            )

    if not windows_obs:
        raise RuntimeError(f"No recovery windows survived: {rejected}")
    observations = np.stack(windows_obs)
    actions = np.stack(windows_act)
    meta = np.asarray(metadata, np.float32)
    consistency = float(
        np.abs(actions[:, :-1, ARM] - observations[:, 1:, ARM]).max()
    )
    output = Path(config.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, observation=observations, action=actions, metadata=meta
    )
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "windows": len(observations),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "accepted_by_direction": {
            str(direction): int((meta[:, 2] == direction).sum())
            for direction in range(config.directions)
        },
        "accepted_by_amplitude": {
            f"{amplitude:g}": int(np.isclose(meta[:, 3], amplitude).sum())
            for amplitude in config.amplitudes
        },
        "rejected": rejected,
        "maximum_ik_error_m": float(meta[:, 6].max()),
        "maximum_terminal_joint_error": float(meta[:, 7].max()),
        "state_action_consistency_error": consistency,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> RecoveryBankConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=RecoveryBankConfig.repo_id)
    parser.add_argument(
        "--dataset-root", type=Path, default=RecoveryBankConfig.dataset_root
    )
    parser.add_argument("--output", type=Path, default=RecoveryBankConfig.output)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amplitudes", default="0.03,0.06,0.09,0.12")
    values = parser.parse_args()
    return RecoveryBankConfig(
        repo_id=values.repo_id,
        dataset_root=values.dataset_root,
        output=values.output,
        limit=values.limit,
        device=values.device,
        amplitudes=tuple(float(value) for value in values.amplitudes.split(",")),
    )


def main():
    print(json.dumps(build_recovery_bank(parse_args()), indent=2))


if __name__ == "__main__":
    main()
