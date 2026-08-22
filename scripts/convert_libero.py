from dataclasses import dataclass, field

import draccus
import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from flow_planning.envs import EnvConfig
from flow_planning.envs.libero import (
    HF_DEMOS,
    SOURCE_SUITE,
    LiberoConfig,
    LiberoEnv,
    demo_to_episode,
)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    repo_id: str = ""
    push_to_hub: bool = False
    replay: int = 10


def replay(env: LiberoEnv, states, act) -> tuple[bool, float]:
    e = env.envs[0]
    e.reset()
    env.obs[0] = e.set_init_state(states[0])
    env.succ[:] = False
    err = []
    for a in act:
        env.apply_action(a[None])
        err.append(np.abs(env.get_obs()[0, :7] - a[:7]).max())
    return bool(env.succ[0]), float(np.mean(err))


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    cfg.env.obstacle = False
    cfg.env.world_count = 1
    env = LiberoEnv(cfg.env)
    repo_id = cfg.repo_id or f"reece-omahoney/{cfg.env.suite}_{cfg.env.task_id}"
    path = hf_hub_download(
        HF_DEMOS,
        f"{SOURCE_SUITE[cfg.env.suite]}/{env.name}_demo.hdf5",
        repo_type="dataset",
    )
    f = h5py.File(path, "r")
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    obs0 = env.get_obs()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs0.shape[1],),
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (8,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=cfg.env.fps, features=features, use_videos=False
    )
    converted = []
    for k in tqdm(demos, desc="Converting demos"):
        obs, act = demo_to_episode(env, f["data"][k])
        converted.append((f["data"][k]["states"][:], act))
        for o, a in zip(obs, act):
            dataset.add_frame(
                {"observation.state": o, "action": a, "task": env.language}
            )
        dataset.save_episode()
    dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")
    if cfg.push_to_hub:
        dataset.push_to_hub()

    if cfg.replay > 0:
        results = [replay(env, s, a) for s, a in converted[: cfg.replay]]
        succ = np.mean([r[0] for r in results])
        err = np.mean([r[1] for r in results])
        print(f"Replay success {succ:.2f}, mean max joint tracking error {err:.4f} rad")


if __name__ == "__main__":
    main()
