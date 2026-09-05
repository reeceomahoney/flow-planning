from dataclasses import dataclass, field, replace

import cv2
import draccus
import newton
import newton.viewer
import numpy as np
import pyglet
import warp as wp
from augment import sdf_np
from render_figure import FigureEnv, render, run_to_phase

import flow_planning.envs.franka as franka
from flow_planning.bend import bend_delta
from flow_planning.envs.franka import FrankaConfig, TaskType

BLUE, GREY = (0.2, 0.45, 0.9), (0.55, 0.15, 0.12)


@dataclass
class Config:
    env: FrankaConfig = field(default_factory=FrankaConfig)
    out: str = "outputs/figure_select.png"
    width: int = 1080
    height: int = 1080
    azimuth: float = 90.0
    elevation: float = 88.0
    distance: float = 1.2
    seed: int = 0
    phase: TaskType = TaskType.MOVE_TO_DROP_OFF
    frac: float = 0.5
    spread: float = 0.28
    base_back: float = 0.6
    lift_height: float = 0.35
    depth: float = 0.1
    book_fill: float = 0.4
    wood: str = "assets/oak_veneer.jpg"
    wood_gain: float = 0.6
    floor: str = "assets/concrete.jpg"
    floor_gain: float = 1.25
    floor_size: float = 0.4
    n_arcs: int = 6
    n_dots: int = 30
    dot_radius: float = 0.006
    phi_range: tuple[float, float] = (0.1, 0.85)
    wall_margin: float = 0.03
    shadows: bool = False
    sun: tuple[float, float, float] = (0.5, -0.6, 0.55)


class SelectEnv(FigureEnv):
    def build_franka_with_table(self):
        h = self.cfg.table_height
        self.robot_base_pos = wp.vec3(0.0, -0.5 - self.fig.base_back, h)
        return super().build_franka_with_table()


def sample_arcs(cfg, ee_a, ee_b, center, half, rng):
    line = np.linspace(ee_a, ee_b, cfg.n_dots)
    clear, hit = [], []
    while not clear or len(hit) < cfg.n_arcs - 1:
        phi = np.pi * rng.uniform(*cfg.phi_range)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        arc = line + bend_delta(line, 0, cfg.n_dots, phi, theta)
        ok = sdf_np(arc, center, half).min() > cfg.wall_margin
        (clear if ok else hit).append(arc)
    return [clear[0], *hit[: cfg.n_arcs - 1]], [BLUE, *[GREY] * (cfg.n_arcs - 1)]


@draccus.wrap()
def main(cfg: Config):
    pyglet.options["headless"] = True
    viewer = newton.viewer.ViewerGL(width=cfg.width, height=cfg.height, headless=True)
    env_cfg = replace(
        cfg.env,
        world_count=1,
        obstacle=True,
        obstacle_width=0.4,
        jitter=0.0,
        lift_height=cfg.lift_height,
        table_height=0.02,
    )
    env = SelectEnv(env_cfg, viewer, cfg)

    focus = np.array(env.obstacle_center)
    az, el = np.radians(cfg.azimuth), np.radians(cfg.elevation)
    pos = focus + cfg.distance * np.array(
        [-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)]
    )
    viewer.set_camera(pos=wp.vec3(*pos), pitch=-cfg.elevation, yaw=cfg.azimuth)
    viewer.camera.look_at(wp.vec3(*focus))
    sun = np.array(cfg.sun)
    viewer.renderer._sun_direction = sun / np.linalg.norm(sun)

    wall = np.array(env.obstacle_center)
    z = env.cube_center[2]
    near = np.array([wall[0], wall[1] - cfg.spread, z], np.float32)
    goal = np.array([wall[0], wall[1] + cfg.spread, z], np.float32)
    env.goal_center = goal
    q, ee = run_to_phase(env, cfg, near, franka.HOME_Q[0] + 0.5 * np.pi)

    half = np.array([0.4, 0.5 * cfg.depth, 0.5 * env_cfg.obstacle_height])
    rng = np.random.default_rng(cfg.seed)
    arcs, colors = sample_arcs(cfg, ee, goal + [0.0, 0.0, 0.05], wall, half, rng)
    lit, flat = render(env, q, arcs, cfg.dot_radius, colors)
    frame = lit if cfg.shadows else flat
    cv2.imwrite(cfg.out, np.ascontiguousarray(frame[..., ::-1]).astype(np.uint8))
    print(f"wrote {cfg.out}")
    viewer.close()


if __name__ == "__main__":
    main()
