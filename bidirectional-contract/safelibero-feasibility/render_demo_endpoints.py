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


def render_task(task_id: int, output: Path) -> tuple[str, int, bool]:
    suite = "safelibero_spatial"
    name = TASKS[suite][task_id]
    cfg = LiberoConfig(
        suite=suite,
        task_id=task_id,
        world_count=1,
        obstacle=False,
        render=True,
        render_size=512,
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
        states = source["data"][key]["states"]
        frames = []
        for label, state in (("start", states[0]), ("after", states[-1])):
            if label == "start":
                obs = env.envs[0].set_init_state(state)
                for _ in range(cfg.settle_steps):
                    obs = env.envs[0].step(DUMMY_ACTION)[0]
                env.obs[0] = obs
            else:
                env.set_state(0, state)
            frame = env.get_frame().copy()
            frames.append(frame)
            cv2.imwrite(
                str(output / f"task{task_id}_{label}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        success = bool(env.envs[0].check_success())
        frame_count = len(states)
    panel = np.concatenate(frames, axis=1)
    banner = np.full((74, panel.shape[1], 3), 245, np.uint8)
    cv2.putText(
        banner,
        f"Task {task_id}: {TITLES[task_id]}",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "START",
        (220, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "AFTER",
        (730, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    paired = np.concatenate([banner, panel], axis=0)
    cv2.imwrite(
        str(output / f"task{task_id}_start_after.png"),
        cv2.cvtColor(paired, cv2.COLOR_RGB2BGR),
    )
    del env
    gc.collect()
    return name, frame_count, success


def main() -> None:
    output = Path(__file__).parent / "images" / "demo_endpoints"
    output.mkdir(parents=True, exist_ok=True)
    for task_id in range(len(TASKS["safelibero_spatial"])):
        name, frame_count, success = render_task(task_id, output)
        print(task_id, name, frame_count, success)


if __name__ == "__main__":
    main()
