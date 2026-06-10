"""Roll out the trained flow-matching policy in the Franka pick-and-place env."""

from dataclasses import dataclass, field

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
class Config:
    env: FrankaConfig = field(default_factory=lambda: FrankaConfig(world_count=1))
    repo_id: str = "reece-omahoney/franka_cube"
    checkpoint: str = "outputs"
    device: str = "cuda"


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"

    policy = FlowMatchingPolicy.from_pretrained(cfg.checkpoint)
    policy.config.device = device
    policy.to(device)

    stats = LeRobotDataset(cfg.repo_id).meta.stats
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config,
        dataset_stats=stats,
    )

    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(cfg.env, viewer)

    timeout_frames = 20 * cfg.env.fps
    frame = 0
    while viewer.is_running():
        if viewer.should_step():
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            ee_action = postprocessor(action).numpy().astype(np.float32)
            env.apply_ee_action(ee_action)

            # visualize the policy's planned EE path (world 0)
            assert policy.chunk is not None
            chunk = postprocessor(policy.chunk[0]).numpy().astype(np.float32)
            env.set_predicted_ee(chunk)
            frame += 1
            if env.success().all() or frame >= timeout_frames:
                env.reset()
                frame = 0
        env.render()

    viewer.close()


if __name__ == "__main__":
    main()
