from dataclasses import dataclass, field, replace
from pathlib import Path

import draccus
import numpy as np
from PIL import Image

from flow_planning.envs import EnvConfig
from flow_planning.envs.libero import LiberoConfig, LiberoEnv


@dataclass
class Config:
    env: EnvConfig = field(default_factory=LiberoConfig)
    n: int = 2
    out: str = "outputs/scenes"


@draccus.wrap()
def main(cfg: Config):
    assert isinstance(cfg.env, LiberoConfig)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for obstacle in (False, True):
        env = LiberoEnv(replace(cfg.env, world_count=1, render=True, obstacle=obstacle))
        tag = (
            f"{cfg.env.suite}_{cfg.env.task_id}_{cfg.env.level if obstacle else 'free'}"
        )
        for i in range(cfg.n):
            env.episode_idx = i
            env.reset()
            Image.fromarray(env.get_frame()).save(out / f"{tag}_{i}.png")
            o = env.obs[0]
            pos = {
                k[:-4]: np.round(v, 3).tolist()
                for k, v in o.items()
                if k.endswith("_pos") and "robot" not in k
            }
            print(tag, i, env.obstacle[0], pos)


if __name__ == "__main__":
    main()
