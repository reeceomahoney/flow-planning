from dataclasses import dataclass, field

import draccus
import newton.viewer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from flow_planning.envs import EnvConfig, ParticleConfig, make_env


@dataclass
class Config:
    env: EnvConfig = field(default_factory=ParticleConfig)
    record: bool = True
    episodes: int = 256
    repo_id: str = "reece-omahoney/particle"
    task: str = "Move the particle to the target position"


def collect(cfg: Config, env):
    obs, action = env.record_frame()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs.shape[1],),
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (action.shape[1],), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id, fps=cfg.env.fps, features=features, use_videos=False
    )

    # one pick-and-place per world = one episode; all worlds reset together
    buffers = [[] for _ in range(cfg.env.world_count)]
    collected = 0
    successes = 0
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
            successes += int(success[:n].sum())
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
    print(f"Demo success rate: {successes / collected:.1%}")


@draccus.wrap()
def main(cfg: Config):
    if cfg.record:
        env = make_env(cfg.env, newton.viewer.ViewerNull())
        collect(cfg, env)
        return

    viewer = newton.viewer.ViewerGL()
    env = make_env(cfg.env, viewer)
    while viewer.is_running():
        if viewer.should_step():
            env.step()
        env.render()
    viewer.close()


if __name__ == "__main__":
    main()
