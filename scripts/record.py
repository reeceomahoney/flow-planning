from dataclasses import dataclass, field

import draccus
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from flow_planning.envs import EnvConfig, FrankaConfig, make_env
from flow_planning.utils import make_viewer


@dataclass
class Config:
    env: EnvConfig = field(default_factory=FrankaConfig)
    record: bool = True
    episodes: int = 256
    push_to_hub: bool = False
    repo_id: str = "reece-omahoney/franka"
    task: str = "Pick up the cube and place it at the target"
    viewer: str = "opengl"  # "none", "rerun", or "opengl"; playback only
    rrd: str = ""  # record a rerun .rrd to this path (view locally: `rerun <rrd>`)


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

    buffers = [[] for _ in range(cfg.env.world_count)]
    collected = 0
    attempted = 0
    successful = 0
    pbar = tqdm(total=cfg.episodes, desc="Collecting episodes")
    while collected < cfg.episodes:
        new_episode, success = env.step()
        if new_episode and buffers[0]:
            attempted += cfg.env.world_count
            successful += int(success.sum())  # count whole batch, not just saved
            for w, buf in enumerate(buffers):
                if collected >= cfg.episodes:
                    break
                if not success[w]:
                    continue
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
                collected += 1
                pbar.update(1)
            buffers = [[] for _ in range(cfg.env.world_count)]
            continue

        obs, action = env.record_frame()
        for w in range(cfg.env.world_count):
            buffers[w].append((obs[w], action[w]))

    pbar.close()
    dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")
    print(f"Demo success rate: {successful / attempted:.1%}")
    if cfg.push_to_hub:
        dataset.push_to_hub()
        print(f"Pushed {cfg.repo_id} to the hub")


@draccus.wrap()
def main(cfg: Config):
    if cfg.record:
        env = make_env(cfg.env, make_viewer("none"))
        collect(cfg, env)
        return

    viewer = make_viewer(cfg.viewer, cfg.rrd)
    env = make_env(cfg.env, viewer)
    done = 0
    while viewer.is_running() and done < cfg.episodes:
        if viewer.should_step():
            new_episode, _ = env.step()
            done += int(new_episode)
        env.render()
    viewer.close()


if __name__ == "__main__":
    main()
