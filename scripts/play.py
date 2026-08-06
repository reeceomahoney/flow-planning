"""Render recorded episodes from a dataset: state playback, no dynamics."""

from dataclasses import dataclass, field

import cv2
import draccus
import newton
import numpy as np
import warp as wp
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from flow_planning.envs import FrankaConfig, make_env
from flow_planning.utils import make_viewer


@dataclass
class Config:
    repo_id: str = "reece-omahoney/franka"
    episodes: str = "0,1,2"
    video: str = ""  # non-empty: render headless to this mp4 instead of a window
    env: FrankaConfig = field(default_factory=FrankaConfig)


@draccus.wrap()
def main(cfg: Config):
    viewer = make_viewer("opengl", headless=bool(cfg.video))
    env = make_env(cfg.env, viewer)
    hf = LeRobotDataset(cfg.repo_id).hf_dataset.with_format("numpy")
    epi = np.asarray(hf["episode_index"])
    obs_all = np.asarray(hf["observation.state"])
    bend_all = np.asarray(hf["bend"]) if "bend" in hf.column_names else None
    n_ep = int(epi.max()) + 1
    print(f"{n_ep} episodes in {cfg.repo_id}")

    n, writer = cfg.env.world_count, None
    for e in [int(v) % n_ep for v in cfg.episodes.split(",")]:
        sel = epi == e
        obs = obs_all[sel]
        if bend_all is not None:
            print(f"episode {e}: bend {bend_all[sel][0].round(3)}")
        env.set_predicted_path(obs[:, 9:12])
        for o in obs:
            jq = np.zeros((n, env.coords_per_world), np.float32)
            jq[:, :9] = o[:9]
            jq[:, 9:12] = o[18:21]
            yaw = np.arctan2(o[21], o[22])
            jq[:, 14], jq[:, 15] = np.sin(yaw / 2), np.cos(yaw / 2)
            wp.copy(env.state_0.joint_q, wp.array(jq.flatten(), dtype=wp.float32))
            newton.eval_fk(
                env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0
            )
            env.goal_pos = np.repeat(o[None, 23:26], n, axis=0)
            env.render()
            if cfg.video:
                frame = viewer.get_frame().numpy()
                if writer is None:
                    writer = cv2.VideoWriter(
                        cfg.video,
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        cfg.env.fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                writer.write(frame[..., ::-1])
    if writer is not None:
        writer.release()
        print(f"Saved video to {cfg.video}")
    viewer.close()


if __name__ == "__main__":
    main()
