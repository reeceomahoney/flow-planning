"""Roll out the trained flow-matching policy in the Franka IK environment."""

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
    repo_id: str = "reece-omahoney/franka_ik"
    checkpoint: str = "output"
    device: str = "cuda"


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"

    policy = FlowMatchingPolicy.from_pretrained(cfg.checkpoint)
    policy.config.device = device
    policy.to(device)

    # processors aren't saved with the policy; rebuild them from dataset stats
    stats = LeRobotDataset(cfg.repo_id).meta.stats
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config,
        dataset_stats=stats,  # ty: ignore[invalid-argument-type]
    )

    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(cfg, viewer)

    while viewer.is_running():
        if viewer.should_step():
            obs = torch.from_numpy(env.joint_q_ik.numpy())
            action = policy.select_action(preprocessor({OBS_STATE: obs}))
            target = postprocessor(action).numpy().astype(np.float32)
            env.solve_to_target(target)
        env.render()

    viewer.close()


if __name__ == "__main__":
    main()
