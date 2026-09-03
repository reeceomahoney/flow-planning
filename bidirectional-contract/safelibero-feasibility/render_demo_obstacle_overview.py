import argparse
import gc
from pathlib import Path

import cv2
import h5py
import numpy as np
from huggingface_hub import hf_hub_download

from flow_planning.envs.libero import (
    DUMMY_ACTION,
    HF_DEMOS,
    SOURCE_SUITE,
    TASKS,
    LiberoConfig,
    LiberoEnv,
)

TITLES = [
    "bowl between plate and ramekin to plate",
    "bowl on ramekin to plate",
    "bowl on stove to plate",
    "bowl on wooden cabinet to plate",
]
SIZE = 512
HEADER = 62
ROW_HEADER = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("I", "II"), default="I")
    return parser.parse_args()


def demo_frame(task_id: int) -> np.ndarray:
    suite = "safelibero_spatial"
    name = TASKS[suite][task_id]
    cfg = LiberoConfig(
        suite=suite,
        task_id=task_id,
        world_count=1,
        obstacle=False,
        render=True,
        render_size=SIZE,
        settle_steps=10,
    )
    env = LiberoEnv(cfg)
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[suite]}/{name}_demo.hdf5",
        repo_type="dataset",
    )
    with h5py.File(path, "r") as source:
        key = sorted(
            source["data"],
            key=lambda value: int(value.split("_")[1]),
        )[0]
        obs = env.envs[0].set_init_state(source["data"][key]["states"][0])
        for _ in range(cfg.settle_steps):
            obs = env.envs[0].step(DUMMY_ACTION)[0]
        env.obs[0] = obs
        frame = env.get_frame().copy()
    del env
    gc.collect()
    return frame


def obstacle_frame(task_id: int, level: str) -> tuple[np.ndarray, str, np.ndarray]:
    cfg = LiberoConfig(
        suite="safelibero_spatial",
        task_id=task_id,
        level=level,
        world_count=1,
        obstacle=True,
        render=True,
        render_size=SIZE,
        settle_steps=10,
    )
    env = LiberoEnv(cfg)
    env.episode_idx = 0
    env.reset()
    frame = env.get_frame().copy()
    name = env.obstacle[0]
    if name is None:
        raise RuntimeError(f"No active obstacle for task {task_id}")
    position = env.obstacle_pos0[0].copy()
    pixel = np.round(env.project(position[None])[0]).astype(int)
    pixel = np.clip(pixel, 0, SIZE - 1)
    cv2.circle(frame, tuple(pixel), 23, (255, 50, 30), 4, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        tuple(pixel),
        (255, 50, 30),
        cv2.MARKER_CROSS,
        28,
        3,
        cv2.LINE_AA,
    )
    del env
    gc.collect()
    return frame, name.replace("_obstacle_1", "").replace("_", " "), position


def main() -> None:
    args = parse_args()
    rows = []
    for task_id, title in enumerate(TITLES):
        left = demo_frame(task_id)
        right, obstacle, position = obstacle_frame(task_id, args.level)
        banner = np.full((ROW_HEADER, 2 * SIZE, 3), 242, np.uint8)
        text = (
            f"Task {task_id}: {title}    |    active obstacle: {obstacle} at "
            f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m"
        )
        cv2.putText(
            banner,
            text,
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([banner, np.concatenate([left, right], axis=1)]))
        print(task_id, title, obstacle, position.round(4).tolist())
    header = np.full((HEADER, 2 * SIZE, 3), 225, np.uint8)
    cv2.putText(
        header,
        "DEMO 0 - SETTLED START",
        (125, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"SAFE-LIBERO LEVEL {args.level} - EPISODE 0",
        (585, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    overview = np.concatenate([header, *rows])
    output = (
        Path(__file__).parent
        / "images"
        / f"spatial_4x2_demo_vs_level_{args.level}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    print(output)


if __name__ == "__main__":
    main()
