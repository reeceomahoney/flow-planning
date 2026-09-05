from dataclasses import dataclass, field, replace

import cv2
import draccus
import newton
import newton.viewer
import numpy as np
import pyglet
import warp as wp
from augment import sdf_np
from scipy.ndimage import gaussian_filter

import flow_planning.envs.franka as franka
from flow_planning.bend import bend_delta
from flow_planning.envs.env import world_offset
from flow_planning.envs.franka import FrankaConfig, FrankaEnv, TaskType

COLORS = [(0.2, 0.45, 0.9)]
ORANGE = (1.0, 0.4, 0.0)
VISUAL = newton.ModelBuilder.ShapeConfig(
    density=0.0, has_shape_collision=False, has_particle_collision=False
)


def hide_last_shape(builder):
    builder.shape_flags[-1] = builder.shape_flags[-1] & ~int(newton.ShapeFlags.VISIBLE)


def smooth_noise(rng, n, sigma):
    return gaussian_filter(rng.standard_normal((n, n)), sigma)


def wood_texture(rng, n=256):
    v, u = np.mgrid[0:n, 0:n] / n
    rings = v * 9.0 + 0.5 * smooth_noise(rng, n, 12) + 0.1 * np.sin(u * 9.0)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * rings)
    g = np.clip(g + 0.12 * smooth_noise(rng, n, 1.2), 0.0, 1.0)[..., None]
    light, dark = np.array([0.7, 0.51, 0.31]), np.array([0.56, 0.38, 0.21])
    return (255 * (dark + (light - dark) * g)).astype(np.uint8)


def book_texture(rng, color, n=128):
    v = np.mgrid[0:n, 0:n][0] / n
    cover = np.tile(np.array(color), (n, n // 2, 1))
    cover += 0.04 * smooth_noise(rng, n, 1.0)[:, : n // 2, None]
    band = (v > 0.28) & (v < 0.42)
    cover[band[:, : n // 2]] = 0.35 * np.array(color) + 0.65 * np.array(
        [0.9, 0.85, 0.7]
    )
    for lo in (0.06, 0.9):
        cover[((v > lo) & (v < lo + 0.025))[:, : n // 2]] *= 0.55
    paper = np.tile(np.array([0.78, 0.72, 0.6]), (n, n - n // 2, 1))
    paper *= 1 - 0.08 * (0.5 + 0.5 * np.sin(v[:, : n - n // 2, None] * 400))
    return (255 * np.clip(np.concatenate([cover, paper], 1), 0, 1)).astype(np.uint8)


def load_image(path, gain):
    img = cv2.imread(path)
    assert img is not None, path
    return np.clip(img[::2, ::2, ::-1] * gain, 0, 255).astype(np.uint8)


def load_wood(path, rng, gain):
    return load_image(path, gain) if path else wood_texture(rng)


def floor_uv(axis, u, v, hu, hv):
    return (u * hu * 1.5, v * hv * 1.5)


def box_mesh(half, texture, uv_fn, roughness):
    axes = ((1, 2), (2, 0), (0, 1))
    verts, norms, uvs, idx = [], [], [], []
    for axis in range(3):
        ua, va = axes[axis]
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            base = len(verts)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = n * half
                p[ua], p[va] = su * half[ua], sv * half[va]
                verts.append(p)
                norms.append(n)
                uvs.append(
                    uv_fn(axis, 0.5 * (su + 1), 0.5 * (sv + 1), half[ua], half[va])
                )
            tri = [0, 1, 2, 0, 2, 3] if sign > 0 else [0, 2, 1, 0, 3, 2]
            idx += [base + i for i in tri]
    return newton.Mesh(
        np.array(verts, np.float32),
        np.array(idx, np.int32),
        normals=np.array(norms, np.float32),
        uvs=np.array(uvs, np.float32),
        texture=texture,
        roughness=roughness,
        compute_inertia=False,
    )


def wood_uv(axis, u, v, hu, hv):
    return (u * hu * 4.0, v * hv * 4.0)


def book_uv(axis, u, v, hu, hv):
    if axis == 2:
        return (0.5 + 0.5 * u, v)
    return (0.5 * v, u) if axis == 1 else (0.5 * u, v)


def add_box(b, pos, half, mesh):
    b.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(*pos), wp.quat(0.0, 0.0, 0.0, 1.0)),
        mesh=mesh,
        cfg=VISUAL,
        color=[1.0, 1.0, 1.0],
    )


def add_shelf(b, center, hx, depth, height, rng, fill, side, wood):
    t = 0.008
    plank = lambda half: box_mesh(np.array(half), wood, wood_uv, 0.75)  # noqa: E731
    cx, cy, z0 = center[0], center[1], center[2] - 0.5 * height
    hy = 0.5 * depth
    zc, hz = z0 + 0.5 * height, 0.5 * height
    for sx in (-1.0, 1.0):
        add_box(b, (cx + sx * (hx - t), cy, zc), (t, hy, hz), plank((t, hy, hz)))
    add_box(b, (cx, cy - side * (hy - t), zc), (hx, t, hz), plank((hx, t, hz)))
    levels = [z0 + t, z0 + 0.5 * height, z0 + height - t]
    for z in levels:
        add_box(b, (cx, cy, z), (hx - 2 * t, hy, t), plank((hx - 2 * t, hy, t)))
    palette = [
        [0.4, 0.1, 0.1],
        [0.1, 0.16, 0.35],
        [0.12, 0.28, 0.16],
        [0.5, 0.36, 0.12],
        [0.2, 0.12, 0.25],
        [0.45, 0.4, 0.32],
    ]
    for z in levels[:-1]:
        x = cx - hx + 3 * t
        while x < cx + hx - 5 * t:
            w, h = rng.uniform(0.008, 0.016), rng.uniform(0.17, 0.21) * height
            color = palette[int(rng.integers(len(palette)))]
            if rng.random() < fill:
                half = (w, 0.7 * hy, h)
                mesh = box_mesh(np.array(half), book_texture(rng, color), book_uv, 0.6)
                add_box(b, (x + w, cy + side * 0.2 * hy, z + t + h), half, mesh)
            x += 2 * w + 0.002


class FigureEnv(FrankaEnv):
    def __init__(self, cfg, viewer, fig):
        self.fig = fig
        super().__init__(cfg, viewer)

    def build_franka_with_table(self):
        b = super().build_franka_with_table()
        hide_last_shape(b)
        scales = np.array(b.shape_scale)[:, :2]
        table = int(np.argmin(np.abs(scales - 0.4).sum(1)))
        b.shape_flags[table] = b.shape_flags[table] & ~int(newton.ShapeFlags.VISIBLE)
        c = self.cfg
        half = np.array(
            [self.fig.floor_size, self.fig.floor_size, 0.5 * c.table_height]
        )
        floor = load_image(self.fig.floor, self.fig.floor_gain)
        add_box(
            b,
            (0.0, -0.5, 0.5 * c.table_height),
            half,
            box_mesh(half, floor, floor_uv, 0.9),
        )
        c = self.cfg
        self.obstacle_half = np.array(
            [c.obstacle_width, 0.5 * self.fig.depth, 0.5 * c.obstacle_height]
        )
        add_shelf(
            b,
            np.array(self.obstacle_center),
            c.obstacle_width,
            self.fig.depth,
            c.obstacle_height,
            np.random.default_rng(self.fig.seed),
            self.fig.book_fill,
            -np.sign(np.sin(np.radians(self.fig.azimuth))),
            load_wood(
                self.fig.wood, np.random.default_rng(self.fig.seed), self.fig.wood_gain
            ),
        )
        return b


@dataclass
class Config:
    env: FrankaConfig = field(default_factory=FrankaConfig)
    out: str = "outputs/figure.png"
    width: int = 1080
    height: int = 1080
    azimuth: float = 135.0
    elevation: float = 25.0
    distance: float = 1.4
    shift: float = 0.0
    seed: int = 0
    phase: TaskType = TaskType.LIFT
    frac: float = 1.0
    ghost_alpha: float = 0.45
    spread: float = 0.22
    shadows: bool = False
    n_arcs: int = 6
    depth: float = 0.1
    book_fill: float = 0.4
    wood: str = "assets/oak_veneer.jpg"
    wood_gain: float = 0.6
    floor: str = "assets/concrete.jpg"
    floor_gain: float = 1.25
    floor_size: float = 0.4
    phi_range: tuple[float, float] = (0.1, 0.7)
    theta_range: tuple[float, float] = (0.0, 1.0)
    wall_margin: float = 0.03
    swoop: float = 0.12
    dot_radius: float = 0.006
    n_dots: int = 30


def run_to_phase(env, cfg, cube_center, home_yaw):
    franka.HOME_Q[0] = home_yaw
    env.cube_center = cube_center
    env.rng = np.random.default_rng(cfg.seed)
    env.reset()
    while env.task_idx < int(cfg.phase) or (
        env.task_idx == int(cfg.phase) and env.task_time < cfg.frac * cfg.env.task_time
    ):
        env.step()
    q = env.state_0.joint_q.numpy().copy()
    ee = env.state_0.body_q.numpy()[env.cfg.ee_index][:3].copy()
    return q, ee


def render(env, q, arcs, width, colors=None):
    wp.copy(env.state_0.joint_q, wp.array(q, dtype=wp.float32))
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    off = world_offset(env.viewer, first=True)
    frames = []
    for shadows in (True, False):
        env.viewer.renderer.draw_shadows = shadows
        env.viewer.begin_frame(env.sim_time)
        env.viewer.log_state(env.state_0)
        for i, arc in enumerate(arcs):
            pts = (arc + off).astype(np.float32)
            color = wp.vec3(*(colors[i] if colors else COLORS[0]))
            env.viewer.log_points(
                f"arc{i}",
                wp.array(pts, dtype=wp.vec3),
                radii=wp.full(len(pts), width, dtype=wp.float32),
                colors=wp.full(len(pts), color, dtype=wp.vec3),
            )
        env.viewer.end_frame()
        wp.synchronize()
        frames.append(env.viewer.get_frame().numpy().astype(np.float32))
    return frames


def sample_arcs(cfg, ee_a, ee_b, center, half, rng):
    n_pts = cfg.n_dots
    line = np.linspace(ee_a, ee_b, n_pts)
    arcs = []
    while len(arcs) < cfg.n_arcs:
        phi = np.pi * rng.uniform(*cfg.phi_range)
        theta = np.pi * rng.uniform(*cfg.theta_range)
        arc = line + bend_delta(line, 0, n_pts, phi, theta)
        if sdf_np(arc, center, half).min() > cfg.wall_margin:
            arcs.append(arc)
    return arcs


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
        table_height=0.02,
    )
    env = FigureEnv(env_cfg, viewer, cfg)

    focus = np.array(env.obstacle_center) + [0.0, 0.0, 0.1]
    az, el = np.radians(cfg.azimuth), np.radians(cfg.elevation)
    pos = focus + cfg.distance * np.array(
        [-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)]
    )
    right = np.cross(focus - pos, [0.0, 0.0, 1.0])
    right = cfg.shift * right / np.linalg.norm(right)
    pos, focus = pos + right, focus + right
    viewer.set_camera(pos=wp.vec3(*pos), pitch=-cfg.elevation, yaw=cfg.azimuth)
    viewer.camera.look_at(wp.vec3(*focus))
    sun = np.array([-0.6 * np.cos(az), -0.6 * np.sin(az), 0.8])
    viewer.renderer._sun_direction = sun / np.linalg.norm(sun)

    wall = np.array(env.obstacle_center)
    z = env.cube_center[2]
    cube_side = np.array([wall[0], wall[1] + cfg.spread, z], np.float32)
    goal_side = np.array([wall[0], wall[1] - cfg.spread, z], np.float32)
    home_yaw = franka.HOME_Q[0]
    q_main, ee_main = run_to_phase(env, cfg, cube_side, home_yaw)
    q_ghost, ee_ghost = run_to_phase(env, cfg, goal_side, -home_yaw)

    half = np.array(
        [
            env_cfg.obstacle_width,
            env_cfg.obstacle_thickness,
            0.5 * env_cfg.obstacle_height,
        ]
    )
    rng = np.random.default_rng(cfg.seed)
    arcs = sample_arcs(cfg, ee_main, ee_ghost, wall, half, rng)
    colors = [COLORS[0]] * len(arcs)
    for ee in (ee_main, ee_ghost):
        away = np.array([0.0, np.sign(ee[1] - wall[1]) * cfg.swoop, 0.0])
        p0 = np.array([ee[0], ee[1], z]) + away
        edge = cfg.floor_size - 0.06
        p0[1] = np.clip(p0[1], wall[1] - edge, wall[1] + edge)
        p1 = p0 + [0.0, 0.0, 0.9 * (ee[2] - z)]
        p2 = ee + 0.5 * away
        t = np.linspace(0.0, 1.0, cfg.n_dots // 2)[:, None]
        u = 1 - t
        arcs.append(u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * ee)
        colors.append(ORANGE)

    q_ref = q_main.copy()
    q_ref[:9] = franka.HOME_Q
    q_ref[0] = home_yaw - np.pi
    frames = [
        render(env, q, arcs, cfg.dot_radius, colors) for q in (q_main, q_ghost, q_ref)
    ]
    cam = np.array(viewer.camera.pos)
    far = np.argmax([np.linalg.norm(ee - cam) for ee in (ee_main, ee_ghost)])
    near_frame, near_flat = frames[1 - far]
    far_frame, far_flat = frames[far]
    bg = np.median(np.stack([near_flat, far_flat, frames[2][1]]), axis=0)
    differs = lambda f: np.abs(f - bg).max(-1, keepdims=True) > 8  # noqa: E731
    moved = differs(far_flat) & ~differs(near_flat)
    if not cfg.shadows:
        near_frame, far_frame = near_flat, far_flat
    a = cfg.ghost_alpha * moved
    out = near_frame * (1 - a) + far_frame * a
    cv2.imwrite(cfg.out, np.ascontiguousarray(out[..., ::-1]).astype(np.uint8))
    print(f"wrote {cfg.out}")
    viewer.close()


if __name__ == "__main__":
    main()
