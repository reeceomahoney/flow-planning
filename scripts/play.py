"""Roll out the trained flow-matching policy in sim."""

from dataclasses import dataclass, field
from urllib.parse import quote

import draccus
import newton.viewer
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from flow_planning.envs import EnvConfig, ParticleConfig, make_env
from flow_planning.paths import latest_run_dir
from flow_planning.policy import (
    FlowMatchingPolicy,
    make_flow_matching_pre_post_processors,
)


@dataclass
class Config:
    env: EnvConfig = field(
        default_factory=lambda: ParticleConfig(world_count=1, obstacle=True)
    )
    repo_id: str = "reece-omahoney/particle"
    checkpoint: str = ""  # empty: latest run under outputs/<env>
    device: str = "cuda"
    guidance_scale: float = 50.0  # >0 enables obstacle guidance
    guidance_margin: float = 0.08
    n_action_steps: int = 10  # >0 overrides the policy's replan interval
    viewer: str = "rerun"  # "rerun" or "opengl"


@draccus.wrap()
def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"

    checkpoint = cfg.checkpoint or latest_run_dir("outputs", cfg.env.type)
    print(f"Loading checkpoint: {checkpoint}")
    policy = FlowMatchingPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    policy.to(device)
    if cfg.n_action_steps > 0:
        policy.config.n_action_steps = cfg.n_action_steps

    stats = LeRobotDataset(cfg.repo_id).meta.stats
    assert stats is not None
    preprocessor, postprocessor = make_flow_matching_pre_post_processors(
        policy.config,
        dataset_stats=stats,
    )

    if cfg.viewer == "opengl":
        viewer = newton.viewer.ViewerGL()
    elif cfg.viewer == "rerun":
        web_port = 9090
        viewer = newton.viewer.ViewerRerun(web_port=web_port)
        if viewer._grpc_server_uri is not None:
            grpc_uri = quote(viewer._grpc_server_uri, safe="")
            print(f"Rerun viewer: http://localhost:{web_port}/?url={grpc_uri}")
    else:
        raise ValueError(f"Unknown viewer: {cfg.viewer}")

    use_guidance = cfg.guidance_scale > 0
    if use_guidance:
        cfg.env.obstacle = True
    env = make_env(cfg.env, viewer)

    guidance_fn = None
    if use_guidance:
        guidance_fn = env.make_guidance(
            action_mean=stats["action"]["mean"],
            action_std=stats["action"]["std"],
            scale=cfg.guidance_scale,
            margin=cfg.guidance_margin,
            device=device,
        )

    timeout_frames = 20 * cfg.env.fps
    frame = 0
    while viewer.is_running():
        if viewer.should_step():
            obs = torch.from_numpy(env.get_obs())
            action = policy.select_action(
                preprocessor({OBS_STATE: obs}), guidance_fn=guidance_fn
            )
            env.apply_action(postprocessor(action).numpy().astype(np.float32))

            # visualize the policy's planned path (world 0)
            assert policy.chunk is not None
            chunk = postprocessor(policy.chunk[0]).numpy().astype(np.float32)
            env.set_predicted_path(chunk)
            frame += 1
            if (env.success() | env.failure()).all() or frame >= timeout_frames:
                env.reset()
                policy.reset()
                frame = 0
        env.render()

    viewer.close()


if __name__ == "__main__":
    main()
