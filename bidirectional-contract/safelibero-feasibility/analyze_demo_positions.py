import csv
import gc
from pathlib import Path

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


def rows_for_task(task_id: int) -> list[dict[str, object]]:
    suite = "safelibero_spatial"
    name = TASKS[suite][task_id]
    cfg = LiberoConfig(
        suite=suite,
        task_id=task_id,
        world_count=1,
        obstacle=False,
        render=False,
        settle_steps=0,
    )
    env = LiberoEnv(cfg)
    simulator = env.envs[0]
    objects = sorted(simulator.env.objects_dict)
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
        states = source["data"][key]["states"][:]
    rows = []
    for index, state in enumerate(states):
        obs = simulator.regenerate_obs_from_state(state)
        for name in objects:
            position = np.asarray(obs[f"{name}_pos"])
            rows.append(
                {
                    "task": task_id,
                    "source": "recorded_state",
                    "index": index,
                    "object": name,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                }
            )
    obs = simulator.set_init_state(states[0])
    for index in range(11):
        for name in objects:
            position = np.asarray(obs[f"{name}_pos"])
            rows.append(
                {
                    "task": task_id,
                    "source": "settling_step",
                    "index": index,
                    "object": name,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                }
            )
        if index < 10:
            obs = simulator.step(DUMMY_ACTION)[0]
    del env
    gc.collect()
    return rows


def main() -> None:
    rows = []
    for task_id in range(len(TASKS["safelibero_spatial"])):
        rows.extend(rows_for_task(task_id))
    output = Path(__file__).parent / "images" / "demo_object_positions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(output, len(rows))


if __name__ == "__main__":
    main()
