from dataclasses import dataclass

import draccus
import newton.viewer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from flow_planning.franka import FrankaConfig, FrankaEnv


@dataclass
class Config(FrankaConfig):
    record: bool = True
    episodes: int = 64
    repo_id: str = "reece-omahoney/franka_ik"
    task: str = "Reach the target end-effector position."


def collect(cfg: Config, env: FrankaEnv):
    ik_dofs = env.model_single.joint_coord_count
    features = {
        "observation.state": {"dtype": "float32", "shape": (ik_dofs,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id, fps=cfg.fps, features=features, use_videos=False
    )

    # one reach per world = one episode; flush all worlds when a new goal is sampled
    buffers = [[] for _ in range(cfg.world_count)]
    collected = 0
    pbar = tqdm(total=cfg.episodes, desc="Collecting episodes")
    while collected < cfg.episodes:
        resampled = env.step()
        if resampled and buffers[0]:
            remaining = cfg.episodes - collected
            for buf in buffers[:remaining]:
                for state, action in buf:
                    dataset.add_frame(
                        {"observation.state": state, "action": action, "task": cfg.task}
                    )
                dataset.save_episode()
            n = min(remaining, cfg.world_count)
            collected += n
            pbar.update(n)
            buffers = [[] for _ in range(cfg.world_count)]
            continue

        joint_q, target = env.record_frame()
        for w in range(cfg.world_count):
            buffers[w].append((joint_q[w], target[w]))

    pbar.close()
    dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")


@draccus.wrap()
def main(cfg: Config):
    if cfg.record:
        env = FrankaEnv(cfg, newton.viewer.ViewerNull())
        collect(cfg, env)
        return

    viewer = newton.viewer.ViewerRerun()
    env = FrankaEnv(cfg, viewer)
    while viewer.is_running():
        if viewer.should_step():
            env.step()
        env.render()
    viewer.close()


if __name__ == "__main__":
    main()
