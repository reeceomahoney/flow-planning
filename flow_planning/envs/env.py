"""Env config registry: scripts pick an env with `--env.type <name>`."""

from dataclasses import dataclass

import draccus


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
    world_count: int = 128
    fps: int = 50
    obstacle: bool = False  # add a barrier between start and goal
    goal_dim: int = 2  # trailing obs dims holding the goal
    spacing_uniform: float = 1.0  # arc-length resample of the plan at inference
    obstacle_scale: float = 15.0  # obstacle-avoidance guidance strength
    obstacle_margin: float = 0.08  # obstacle-avoidance guidance clearance

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


def world_offset(viewer, first=False):
    o = getattr(viewer, "world_offsets", None)
    if o is None:
        return 0.0
    return o.numpy()[0] if first else o.numpy()


def make_env(cfg: EnvConfig, viewer):
    from flow_planning.envs.franka import FrankaConfig, FrankaEnv
    from flow_planning.envs.particle import ParticleConfig, ParticleEnv

    if isinstance(cfg, FrankaConfig):
        return FrankaEnv(cfg, viewer)
    if isinstance(cfg, ParticleConfig):
        return ParticleEnv(cfg, viewer)
    raise ValueError(f"Unknown env config: {type(cfg).__name__}")
