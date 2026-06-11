"""Env config registry: scripts pick an env with `--env.type <name>`."""

from dataclasses import dataclass

import draccus


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
    world_count: int = 256
    fps: int = 50

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


def make_env(cfg: EnvConfig, viewer):
    from flow_planning.envs.franka import FrankaConfig, FrankaEnv
    from flow_planning.envs.particle import ParticleConfig, ParticleEnv

    if isinstance(cfg, FrankaConfig):
        return FrankaEnv(cfg, viewer)
    if isinstance(cfg, ParticleConfig):
        return ParticleEnv(cfg, viewer)
    raise ValueError(f"Unknown env config: {type(cfg).__name__}")
