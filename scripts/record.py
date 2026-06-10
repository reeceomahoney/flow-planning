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
    repo_id: str = "reece-omahoney/franka_cube"
    task: str = "Pick up the cube and place it at the goal position."


def collect(cfg: Config, env: FrankaEnv):
    ik_dofs = env.model_single.joint_coord_count  # 9 joint targets = the action
    obs_dim = ik_dofs + 6  # joint coords (9) + cube pos (3) + goal pos (3)
    features = {
        "observation.state": {"dtype": "float32", "shape": (obs_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (ik_dofs,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id, fps=cfg.fps, features=features, use_videos=False
    )

    # one pick-and-place per world = one episode; all worlds reset together
    buffers = [[] for _ in range(cfg.world_count)]
    collected = 0
    pbar = tqdm(total=cfg.episodes, desc="Collecting episodes")
    while collected < cfg.episodes:
        new_episode = env.step()
        if new_episode and buffers[0]:
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

        obs, action = env.record_frame()
        for w in range(cfg.world_count):
            buffers[w].append((obs[w], action[w]))

    pbar.close()
    dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")


@draccus.wrap()
def main(cfg: Config):
    if cfg.record:
        env = FrankaEnv(cfg, newton.viewer.ViewerNull())
        collect(cfg, env)
        return

    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(cfg, viewer)
    while viewer.is_running():
        if viewer.should_step():
            env.step()
        env.render()
    viewer.close()


if __name__ == "__main__":
    main()
