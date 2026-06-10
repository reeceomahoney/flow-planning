"""Roll out the trained flow-matching policy in the Franka pick-and-place env."""

from dataclasses import dataclass

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from flow_planning.franka import FrankaConfig, FrankaEnv
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config(FrankaConfig):
    repo_id: str = "reece-omahoney/franka_cube"
    checkpoint: str = "outputs"
    device: str = "cuda"
    world_count: int = 1


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"

    policy = FlowMatchingPolicy.from_pretrained(cfg.checkpoint)
    policy.config.device = device
    policy.to(device)

    stats = LeRobotDataset(cfg.repo_id).meta.stats
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config,
        dataset_stats=stats,  # ty: ignore[invalid-argument-type]
    )

    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(cfg, viewer)

    frame = 0
    while viewer.is_running():
        if viewer.should_step():
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            joint_targets = postprocessor(action).numpy().astype(np.float32)
            env.apply_action(joint_targets)
            frame += 1
            if frame % env.episode_frames == 0:
                env.reset()
        env.render()

    viewer.close()


if __name__ == "__main__":
    main()
