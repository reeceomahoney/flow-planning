import csv
import gc
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUITE = "safelibero_spatial"
BEND_MARGIN_SECONDS = 0.3
LEVELS = ("I", "II")
TITLES = (
    "between plate and ramekin",
    "on ramekin",
    "on stove",
    "on wooden cabinet",
)


def demo_boundaries(task_id: int, env: LiberoEnv) -> tuple[np.ndarray, np.ndarray]:
    name = TASKS[SUITE][task_id]
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[SUITE]}/{name}_demo.hdf5",
        repo_type="dataset",
    )
    margin = round(BEND_MARGIN_SECONDS * env.cfg.fps)
    joints = []
    boundaries = []
    with h5py.File(path, "r") as source:
        keys = sorted(
            source["data"],
            key=lambda value: int(value.split("_")[1]),
        )
        for key in keys:
            demo = source["data"][key]
            gripper = GRIPPER_OPEN * (demo["actions"][:, 6] < 0)
            close, _ = grasp_segments(gripper)[0]
            boundary = close - margin
            obs = env.envs[0].set_init_state(demo["states"][boundary])
            joints.append(obs["robot0_joint_pos"])
            boundaries.append(boundary)
    return np.asarray(joints, np.float32), np.asarray(boundaries, int)


def clearances_for_level(
    task_id: int,
    level: str,
    points: torch.Tensor,
    collision: FrankaCollision,
) -> tuple[np.ndarray, list[str]]:
    env = LiberoEnv(
        LiberoConfig(
            suite=SUITE,
            task_id=task_id,
            level=level,
            world_count=1,
            obstacle=True,
            render=False,
            settle_steps=10,
        )
    )
    clearances = []
    obstacles = []
    for init_index in range(len(env.init_states)):
        env.episode_idx = init_index
        env.reset()
        name = env.obstacle[0]
        if name is None:
            raise RuntimeError(
                f"No obstacle for task {task_id}, level {level}, init {init_index}"
            )
        boxes = torch.as_tensor(
            obstacle_geom_boxes(env.envs[0], name),
            dtype=torch.float32,
        )
        distance = box_sdf(
            points[:, :, None],
            boxes[None, None, :, :3],
            boxes[None, None, :, 3:],
        ).amin(dim=(1, 2))
        clearances.append((distance - collision.radius).cpu().numpy())
        obstacles.append(name.replace("_obstacle_1", "").replace("_", " "))
    del env
    gc.collect()
    return np.stack(clearances), obstacles


def write_rows(
    rows: list[dict[str, object]],
    task_id: int,
    level: str,
    boundaries: np.ndarray,
    clearances: np.ndarray,
    obstacles: list[str],
) -> None:
    overlap = clearances < 0.0
    for init_index, obstacle in enumerate(obstacles):
        clear = ~overlap[init_index]
        best_demo = int(clearances[init_index].argmax())
        rows.append(
            {
                "task": task_id,
                "level": level,
                "initialization": init_index,
                "obstacle": obstacle,
                "demo_count": len(boundaries),
                "overlap_count": int(overlap[init_index].sum()),
                "overlap_rate": float(overlap[init_index].mean()),
                "clear_count": int(clear.sum()),
                "any_clear_B": bool(clear.any()),
                "best_demo": best_demo,
                "best_demo_B_state": int(boundaries[best_demo]),
                "best_clearance_m": float(clearances[init_index, best_demo]),
            }
        )


def summarize(
    task_id: int,
    level: str,
    clearances: np.ndarray,
) -> dict[str, object]:
    clear = clearances >= 0.0
    clear_per_init = clear.sum(axis=1)
    clear_per_demo = clear.sum(axis=0)
    return {
        "task": task_id,
        "level": level,
        "pairs": int(clear.size),
        "overlap_rate": float((~clear).mean()),
        "inits_with_any_clear_B": int((clear_per_init > 0).sum()),
        "inits_with_no_clear_B": int((clear_per_init == 0).sum()),
        "min_clear_Bs_per_init": int(clear_per_init.min()),
        "median_clear_Bs_per_init": float(np.median(clear_per_init)),
        "max_clear_Bs_per_init": int(clear_per_init.max()),
        "best_single_demo": int(clear_per_demo.argmax()),
        "best_single_demo_clear_inits": int(clear_per_demo.max()),
        "demos_clear_for_every_init": int((clear_per_demo == len(clearances)).sum()),
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_matrices(path: Path, matrices: dict[tuple[int, str], np.ndarray]) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13, 20), constrained_layout=True)
    cmap = ListedColormap(["#32b85a", "#df4638"])
    for task_id in range(4):
        for column, level in enumerate(LEVELS):
            overlap = matrices[(task_id, level)] < 0.0
            clear_per_init = (~overlap).sum(axis=1)
            axis = axes[task_id, column]
            axis.imshow(overlap.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            axis.set_title(
                f"Task {task_id}: {TITLES[task_id]} | Level {level}\n"
                f"overlap {100 * overlap.mean():.1f}% | "
                f"no clear B: {(clear_per_init == 0).sum()}/50"
            )
            axis.set_xlabel("Safety initialization")
            axis.set_ylabel("Source demo B")
            axis.set_xticks(np.arange(0, 50, 5))
            axis.set_yticks(np.arange(0, 50, 5))
    figure.legend(
        handles=[
            Patch(facecolor="#32b85a", label="clear"),
            Patch(facecolor="#df4638", label="overlap"),
        ],
        loc="outside upper center",
        ncols=2,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
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
    rows = []
    summaries = []
    matrices = {}
    for task_id in range(len(TASKS[SUITE])):
        free_env = LiberoEnv(
            LiberoConfig(
                suite=SUITE,
                task_id=task_id,
                world_count=1,
                obstacle=False,
                render=False,
                settle_steps=0,
            )
        )
        joints, boundaries = demo_boundaries(task_id, free_env)
        del free_env
        gc.collect()
        q = torch.as_tensor(joints, dtype=torch.float32)
        points = collision.arm_points(q) + collision.base_pos
        for level in LEVELS:
            clearances, obstacles = clearances_for_level(
                task_id,
                level,
                points,
                collision,
            )
            matrices[(task_id, level)] = clearances
            write_rows(
                rows,
                task_id,
                level,
                boundaries,
                clearances,
                obstacles,
            )
            summary = summarize(task_id, level, clearances)
            summaries.append(summary)
            print(summary, flush=True)
    output = Path(__file__).parent / "images"
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "B_collision_by_initialization.csv", rows)
    save_csv(output / "B_collision_summary.csv", summaries)
    plot_matrices(output / "B_collision_matrix.png", matrices)


if __name__ == "__main__":
    main()
