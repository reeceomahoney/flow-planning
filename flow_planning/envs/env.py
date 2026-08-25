"""Env config registry: scripts pick an env with `--env.type <name>`."""

from dataclasses import dataclass

import draccus


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
    world_count: int = 128
    fps: int = 50
    obstacle: bool = False  # add a barrier between start and goal
    goal_dim: int = 2  # trailing obs dims holding the goal
    goal_state_start: int = 0  # state block the goal clamps onto (same quantity)
    n_action_steps: int = 75  # replan interval, env-tuned

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@EnvConfig.register_subclass("piper")
@dataclass
class PiperConfig(EnvConfig):
    fps: int = 20
    goal_dim: int = 0
    n_action_steps: int = 30
    gripper_closed: float = 1.5


def world_offset(viewer, first=False):
    o = getattr(viewer, "world_offsets", None)
    if o is None:
        return 0.0
    return o.numpy()[0] if first else o.numpy()


def make_env(cfg: EnvConfig, viewer):
    from flow_planning.envs.franka import FrankaConfig, FrankaEnv
    from flow_planning.envs.libero import LiberoConfig, LiberoEnv
    from flow_planning.envs.particle import ParticleConfig, ParticleEnv

    if isinstance(cfg, FrankaConfig):
        return FrankaEnv(cfg, viewer)
    if isinstance(cfg, ParticleConfig):
        return ParticleEnv(cfg, viewer)
    if isinstance(cfg, LiberoConfig):
        return LiberoEnv(cfg, viewer)
    raise ValueError(f"Unknown env config: {type(cfg).__name__}")
