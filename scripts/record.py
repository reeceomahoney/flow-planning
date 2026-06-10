from dataclasses import dataclass, field

import draccus
import newton.viewer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from flow_planning.franka import FrankaConfig, FrankaEnv


@dataclass
class Config:
    env: FrankaConfig = field(default_factory=FrankaConfig)
    record: bool = True
    episodes: int = 256
    repo_id: str = "reece-omahoney/franka_cube"
    task: str = "Pick up the cube and place it at the goal position."


def collect(cfg: Config, env: FrankaEnv):
    ik_dofs = env.model_single.joint_coord_count
    # joints (9), EE pos (3), EE rot6d (6), cube pos (3), cube yaw sc (2), goal (3)
    obs_dim = ik_dofs + 17
    action_dim = 10  # ee pos (3), 6D rotation (6), gripper (1)
    features = {
        "observation.state": {"dtype": "float32", "shape": (obs_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id, fps=cfg.env.fps, features=features, use_videos=False
    )

    # one pick-and-place per world = one episode; all worlds reset together
    buffers = [[] for _ in range(cfg.env.world_count)]
    collected = 0
    pbar = tqdm(total=cfg.episodes, desc="Collecting episodes")
    while collected < cfg.episodes:
        new_episode, success = env.step()
        if new_episode and buffers[0]:
            remaining = cfg.episodes - collected
            for w, buf in enumerate(buffers[:remaining]):
                for state, action in buf:
                    dataset.add_frame(
                        {
                            "observation.state": state,
                            "action": action,
                            "next.success": success[w : w + 1],
                            "task": cfg.task,
                        }
                    )
                dataset.save_episode()
            n = min(remaining, cfg.env.world_count)
            collected += n
            pbar.update(n)
            buffers = [[] for _ in range(cfg.env.world_count)]
            continue

        obs, action = env.record_frame()
        for w in range(cfg.env.world_count):
            buffers[w].append((obs[w], action[w]))

    pbar.close()
    dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")


@draccus.wrap()
def main(cfg: Config):
    if cfg.record:
        env = FrankaEnv(cfg.env, newton.viewer.ViewerNull())
        collect(cfg, env)
        return

    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(cfg.env, viewer)
    while viewer.is_running():
        if viewer.should_step():
            env.step()
        env.render()
    viewer.close()


if __name__ == "__main__":
    main()
