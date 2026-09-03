from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from camera_web import LiveState, Trajectory3D

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "reece-omahoney/pick-and-place"
DATASET_FILE = "data/chunk-000/file-000.parquet"


def add_pixi_site_packages() -> None:
    site_packages = sorted(
        (REPO_ROOT / ".pixi" / "envs" / "default" / "lib").glob("python*/site-packages")
    )
    if not site_packages:
        raise RuntimeError("The repository Pixi environment is unavailable")
    path = str(site_packages[-1])
    if path not in sys.path:
        sys.path.append(path)


def recorded_joint_trajectories(
    count: int,
) -> tuple[list[np.ndarray], tuple[int, ...], tuple[int, ...]]:
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(DATASET_ID, DATASET_FILE, repo_type="dataset"))
    table = parquet.read_table(
        source,
        columns=["observation.state", "episode_index"],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_values = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    episode_indices = tuple(range(count))
    trajectories = [states[episode_values == index] for index in episode_indices]
    if any(not len(trajectory) for trajectory in trajectories):
        raise RuntimeError("Requested training episodes are unavailable")
    pickup_steps = []
    for trajectory in trajectories:
        gripper = trajectory[:, 6]
        open_step = int(np.argmax(gripper))
        closing = np.flatnonzero(
            (np.arange(len(gripper)) > open_step) & (gripper < 2.0)
        )
        pickup_steps.append(int(closing[0]) if len(closing) else open_step)
    return trajectories, episode_indices, tuple(pickup_steps)


def end_effector_trajectories(
    actions: list[np.ndarray],
) -> tuple[Trajectory3D, ...]:
    from piper_sdk import C_PiperForwardKinematics

    kinematics = C_PiperForwardKinematics()
    trajectories: list[Trajectory3D] = []
    for candidate in actions:
        points = []
        for joints in candidate:
            point = (
                np.asarray(
                    kinematics.CalFK(np.radians(joints[:6]))[-1][:3],
                    dtype=np.float32,
                )
                / 1000.0
            )
            points.append(point)
        trajectories.append(
            tuple(
                (float(point[0]), float(point[1]), float(point[2])) for point in points
            )
        )
    return tuple(trajectories)


class DemoTrajectoryStream:
    def __init__(
        self,
        live_state: LiveState,
        samples: int,
    ):
        self.live_state = live_state
        self.samples = samples
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def publish_status(self, state: str, **values: Any) -> None:
        self.live_state.publish_policy_status({"state": state, **values})

    def run(self) -> None:
        try:
            add_pixi_site_packages()
            self.publish_status("loading recorded training demos")
            started = time.perf_counter()
            actions, episode_indices, pickup_steps = recorded_joint_trajectories(
                self.samples
            )
            trajectories = end_effector_trajectories(actions)
            conversion_ms = (time.perf_counter() - started) * 1000.0
            self.live_state.publish_policy_trajectories(trajectories, pickup_steps)
            self.publish_status(
                "ready",
                source="recorded demos",
                samples=len(trajectories),
                conversion_ms=conversion_ms,
                episodes=episode_indices,
                lengths=tuple(len(trajectory) for trajectory in trajectories),
                pickup_steps=pickup_steps,
            )
        except Exception as error:
            self.publish_status("error", message=str(error))
