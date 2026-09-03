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
LEVEL_EPISODES = {
    "I": (
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (0, 7, 14, 16),
    ),
    "II": ((0, 1, 2, 3),) * 4,
}
CELL = 512
SUBCELL = CELL // 2
TOP_HEADER = 70
ROW_HEADER = 42
WIDTH = CELL * 3


def put_centered(
    image: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    scale: float,
    thickness: int,
    color: tuple[int, int, int] = (15, 15, 15),
) -> None:
    size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )[0]
    cv2.putText(
        image,
        text,
        (center_x - size[0] // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def settled_demo_frame(task_id: int) -> np.ndarray:
    suite = "safelibero_spatial"
    name = TASKS[suite][task_id]
    cfg = LiberoConfig(
        suite=suite,
        task_id=task_id,
        world_count=1,
        obstacle=False,
        render=True,
        render_size=CELL,
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
    cv2.rectangle(frame, (0, 0), (CELL, 30), (20, 20, 20), -1)
    put_centered(frame, "demo 0 - settled", CELL // 2, 21, 0.55, 1, (255, 255, 255))
    del env
    gc.collect()
    return frame


def obstacle_grid(task_id: int, level: str) -> np.ndarray:
    cfg = LiberoConfig(
        suite="safelibero_spatial",
        task_id=task_id,
        level=level,
        world_count=1,
        obstacle=True,
        render=True,
        render_size=SUBCELL,
        settle_steps=10,
    )
    env = LiberoEnv(cfg)
    tiles = []
    for episode in LEVEL_EPISODES[level][task_id]:
        env.episode_idx = episode
        env.reset()
        frame = env.get_frame().copy()
        name = env.obstacle[0]
        if name is None:
            raise RuntimeError(f"No obstacle for task {task_id}, level {level}")
        position = env.obstacle_pos0[0]
        pixel = np.rint(env.project(position[None])[0]).astype(int)
        pixel = np.clip(pixel, 0, SUBCELL - 1)
        cv2.circle(frame, tuple(pixel), 12, (255, 45, 25), 3, cv2.LINE_AA)
        cv2.drawMarker(
            frame,
            tuple(pixel),
            (255, 45, 25),
            cv2.MARKER_CROSS,
            15,
            2,
            cv2.LINE_AA,
        )
        label = name.replace("_obstacle_1", "").replace("_", " ")
        cv2.rectangle(frame, (0, 0), (SUBCELL, 26), (20, 20, 20), -1)
        put_centered(
            frame,
            f"ep {episode}: {label}",
            SUBCELL // 2,
            19,
            0.43,
            1,
            (255, 255, 255),
        )
        tiles.append(frame)
        print(
            task_id,
            level,
            episode,
            label,
            np.round(position, 4).tolist(),
        )
    grid = np.concatenate(
        [
            np.concatenate(tiles[:2], axis=1),
            np.concatenate(tiles[2:], axis=1),
        ],
        axis=0,
    )
    del env
    gc.collect()
    return grid


def main() -> None:
    rows = []
    for task_id, title in enumerate(TITLES):
        banner = np.full((ROW_HEADER, WIDTH, 3), 241, np.uint8)
        cv2.putText(
            banner,
            f"Task {task_id}: {title}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
        images = [
            settled_demo_frame(task_id),
            obstacle_grid(task_id, "I"),
            obstacle_grid(task_id, "II"),
        ]
        rows.append(np.concatenate([banner, np.concatenate(images, axis=1)]))
    header = np.full((TOP_HEADER, WIDTH, 3), 221, np.uint8)
    titles = (
        "NO OBSTACLE",
        "LEVEL I - NEAR TARGET",
        "LEVEL II - IN TRANSPORT PATH",
    )
    for column, title in enumerate(titles):
        put_centered(header, title, column * CELL + CELL // 2, 43, 0.72, 2)
    output = np.concatenate([header, *rows], axis=0)
    path = (
        Path(__file__).parent
        / "images"
        / "spatial_4x3_no_obstacle_level_I_level_II.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(path)


if __name__ == "__main__":
    main()
