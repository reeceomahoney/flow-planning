import gc
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from flow_planning.bend import grasp_segments
from flow_planning.envs.libero import (
    GRIPPER_OPEN,
    HF_DEMOS,
    SOURCE_SUITE,
    TASKS,
    LiberoConfig,
    LiberoEnv,
    obstacle_geom_boxes,
)
from flow_planning.selector import FrankaCollision, box_sdf

TITLES = (
    "between plate and ramekin",
    "on ramekin",
    "on stove",
    "on wooden cabinet",
)
SUITE = "safelibero_spatial"
BEND_MARGIN_SECONDS = 0.3
SIZE = 384
HEADER = 68
DIVIDER = 8


def put_centered(
    image: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    scale: float,
    thickness: int,
    color: tuple[int, int, int] = (15, 15, 15),
) -> None:
    width = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )[0][0]
    cv2.putText(
        image,
        text,
        (center_x - width // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def mark_point(
    frame: np.ndarray,
    env: LiberoEnv,
    point: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    pixel = np.rint(env.project(point[None])[0]).astype(int)
    pixel = np.clip(pixel, 0, SIZE - 1)
    cv2.circle(frame, tuple(pixel), 17, color, 3, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        tuple(pixel),
        color,
        cv2.MARKER_CROSS,
        21,
        2,
        cv2.LINE_AA,
    )


def load_frame(env: LiberoEnv, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = env.envs[0].set_init_state(state)
    env.obs[0] = obs
    return env.get_frame().copy(), obs["robot0_eef_pos"].copy()


def copy_common_joints(source, target) -> int:
    source_model = source.model._model
    target_model = target.model._model
    source_joints = {
        source_model.joint(index).name: index for index in range(source_model.njnt)
    }
    copied = 0
    for target_index in range(target_model.njnt):
        name = target_model.joint(target_index).name
        if name not in source_joints:
            continue
        source_index = source_joints[name]
        source_q0 = source_model.jnt_qposadr[source_index]
        source_q1 = (
            source_model.nq
            if source_index + 1 == source_model.njnt
            else source_model.jnt_qposadr[source_index + 1]
        )
        target_q0 = target_model.jnt_qposadr[target_index]
        target_q1 = (
            target_model.nq
            if target_index + 1 == target_model.njnt
            else target_model.jnt_qposadr[target_index + 1]
        )
        source_v0 = source_model.jnt_dofadr[source_index]
        source_v1 = (
            source_model.nv
            if source_index + 1 == source_model.njnt
            else source_model.jnt_dofadr[source_index + 1]
        )
        target_v0 = target_model.jnt_dofadr[target_index]
        target_v1 = (
            target_model.nv
            if target_index + 1 == target_model.njnt
            else target_model.jnt_dofadr[target_index + 1]
        )
        if source_q1 - source_q0 != target_q1 - target_q0:
            continue
        if source_v1 - source_v0 != target_v1 - target_v0:
            continue
        target.data.qpos[target_q0:target_q1] = source.data.qpos[source_q0:source_q1]
        target.data.qvel[target_v0:target_v1] = source.data.qvel[source_v0:source_v1]
        copied += 1
    target.forward()
    return copied


def metric_clearance(
    collision: FrankaCollision,
    joints: np.ndarray,
    eef: np.ndarray,
    boxes: np.ndarray,
) -> tuple[float, float, str]:
    box = torch.as_tensor(boxes, dtype=torch.float32)
    centers, half = box[:, :3], box[:, 3:]
    q = torch.as_tensor(joints[None], dtype=torch.float32)
    points = collision.arm_points(q)[0] + collision.base_pos
    distances = box_sdf(points[:, None], centers, half).amin(dim=1)
    closest = int(distances.argmin())
    robot_clearance = float(distances[closest] - collision.radius)
    point = torch.as_tensor(eef, dtype=torch.float32)
    eef_clearance = float(box_sdf(point[None], centers, half).min() - collision.radius)
    link = collision.frames[int(collision.owner[closest])]
    return robot_clearance, eef_clearance, link


def hybrid_frame(
    task_id: int,
    source_env: LiberoEnv,
    collision: FrankaCollision,
) -> tuple[np.ndarray, str, float, float, str]:
    cfg = LiberoConfig(
        suite=SUITE,
        task_id=task_id,
        level="I",
        world_count=1,
        obstacle=True,
        render=True,
        render_size=SIZE,
        settle_steps=10,
    )
    env = LiberoEnv(cfg)
    env.episode_idx = 0
    env.reset()
    copied = copy_common_joints(source_env.envs[0].sim, env.envs[0].sim)
    obs = env.envs[0].regenerate_obs_from_state(env.envs[0].get_sim_state())
    env.obs[0] = obs
    name = env.obstacle[0]
    if name is None:
        raise RuntimeError(f"No Level-I obstacle for task {task_id}")
    boxes = obstacle_geom_boxes(env.envs[0], name)
    robot_clearance, eef_clearance, link = metric_clearance(
        collision,
        obs["robot0_joint_pos"],
        obs["robot0_eef_pos"],
        boxes,
    )
    frame = env.get_frame().copy()
    mark_point(frame, env, obs["robot0_eef_pos"], (255, 55, 30))
    label = name.replace("_obstacle_1", "").replace("_", " ")
    robot_status = "OVERLAP" if robot_clearance < 0.0 else "CLEAR"
    eef_status = "OVERLAP" if eef_clearance < 0.0 else "CLEAR"
    color = (255, 70, 45) if robot_clearance < 0.0 else (45, 225, 70)
    cv2.rectangle(frame, (0, 0), (SIZE, 48), (20, 20, 20), -1)
    put_centered(
        frame,
        f"B + Level-I ep 0: {label}",
        SIZE // 2,
        19,
        0.45,
        1,
        (255, 255, 255),
    )
    put_centered(
        frame,
        f"robot {robot_status} {1e3 * robot_clearance:+.0f}mm | "
        f"EEF {eef_status} {1e3 * eef_clearance:+.0f}mm",
        SIZE // 2,
        39,
        0.39,
        1,
        color,
    )
    print(
        task_id,
        "hybrid",
        label,
        "copied joints",
        copied,
        "robot clearance",
        round(robot_clearance, 4),
        "eef clearance",
        round(eef_clearance, 4),
        "closest link",
        link,
    )
    del env
    gc.collect()
    return frame, label, robot_clearance, eef_clearance, link


def task_frames(
    task_id: int,
    collision: FrankaCollision,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    cfg = LiberoConfig(
        suite=SUITE,
        task_id=task_id,
        world_count=1,
        obstacle=False,
        render=True,
        render_size=SIZE,
        settle_steps=0,
    )
    env = LiberoEnv(cfg)
    name = TASKS[SUITE][task_id]
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[SUITE]}/{name}_demo.hdf5",
        repo_type="dataset",
    )
    with h5py.File(path, "r") as source:
        key = sorted(
            source["data"],
            key=lambda value: int(value.split("_")[1]),
        )[0]
        demo = source["data"][key]
        states = demo["states"][:]
        gripper = GRIPPER_OPEN * (demo["actions"][:, 6] < 0)
    close, opened = grasp_segments(gripper)[0]
    margin = round(BEND_MARGIN_SECONDS * cfg.fps)
    boundary = close - margin
    frame_a, point_a = load_frame(env, states[0])
    mark_point(frame_a, env, point_a, (40, 220, 60))
    frame_b, point_b = load_frame(env, states[boundary])
    mark_point(frame_b, env, point_b, (255, 55, 30))
    frame_hybrid, _, _, _, _ = hybrid_frame(task_id, env, collision)
    for frame, text in (
        (frame_a, "A - recorded state 0"),
        (frame_b, f"B - state {boundary} | close {close}"),
    ):
        cv2.rectangle(frame, (0, 0), (SIZE, 30), (20, 20, 20), -1)
        put_centered(frame, text, SIZE // 2, 21, 0.52, 1, (255, 255, 255))
    print(
        task_id,
        "frames",
        len(states),
        "B",
        boundary,
        "close",
        close,
        "open",
        opened,
        "A xyz",
        np.round(point_a, 4).tolist(),
        "B xyz",
        np.round(point_b, 4).tolist(),
    )
    del env
    gc.collect()
    return frame_a, frame_b, frame_hybrid, boundary, close, opened


def main() -> None:
    top = []
    middle = []
    bottom = []
    probe = LiberoEnv(
        LiberoConfig(
            suite=SUITE,
            task_id=0,
            world_count=1,
            obstacle=False,
            render=False,
        )
    )
    collision = FrankaCollision("cpu", probe.robot_base_pos, 0.0)
    del probe
    gc.collect()
    for task_id in range(len(TASKS[SUITE])):
        frame_a, frame_b, frame_hybrid, _, _, _ = task_frames(task_id, collision)
        top.append(frame_a)
        middle.append(frame_b)
        bottom.append(frame_hybrid)
    width = SIZE * len(top)
    header = np.full((HEADER, width, 3), 226, np.uint8)
    for task_id, title in enumerate(TITLES):
        center = task_id * SIZE + SIZE // 2
        put_centered(header, f"TASK {task_id}", center, 26, 0.64, 2)
        put_centered(header, title, center, 51, 0.47, 1)
    divider = np.full((DIVIDER, width, 3), 235, np.uint8)
    image = np.concatenate(
        [
            header,
            np.concatenate(top, axis=1),
            divider,
            np.concatenate(middle, axis=1),
            divider,
            np.concatenate(bottom, axis=1),
        ]
    )
    output = Path(__file__).parent / "images" / "spatial_3x4_demo_A_B_level_I.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(output)


if __name__ == "__main__":
    main()
