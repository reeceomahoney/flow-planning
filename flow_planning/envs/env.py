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
    obstacle_scale: float = 15.0  # obstacle-avoidance guidance strength
    obstacle_margin: float = 0.08  # obstacle-avoidance guidance clearance

    # ellipse keep-out guidance (0 disables)
    ellipse_scale: float = 0.0
    ellipse_margin: float = 0.15  # ellipse inflation past the box (normalized)
    ellipse_ay: float = 0.85  # travel extent as a fraction of half the start->goal gap
    ellipse_bias: float = 1.0  # symmetry-break push toward the exit side

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
