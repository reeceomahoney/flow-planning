"""Particle environment built on newton: a ball confined to a square arena.

A dynamic ball rolls on the ground inside four static walls; the action is a
2D target position tracked by a PD force on the ball's center. An optional
obstacle (optionally randomized per world) bisects the arena, with start and
goal sampled on opposite sides. Mirrors the
FrankaEnv interface: vectorized worlds, get_obs/apply_action/success/render,
plus a scripted straight-to-goal controller (step/record_frame) for demos.
"""

from dataclasses import dataclass

import newton
import newton.solvers
import numpy as np
import warp as wp

from flow_planning.envs.contact import ObstacleContactSensor
from flow_planning.envs.env import EnvConfig
from flow_planning.guidance import ObstacleGuidance

wp.config.quiet = True


@EnvConfig.register_subclass("particle")
@dataclass
class ParticleConfig(EnvConfig):
    sim_substeps: int = 4
    task_time: float = 4.0  # seconds per scripted demo episode
    arena_size: float = 1.0  # half-extent of the square
    wall_height: float = 0.1
    wall_thickness: float = 0.05
    ball_radius: float = 0.05
    kp: float = 50.0  # PD gains mapping target position to force
    kd: float = 10.0
    sample_margin: float = 0.15  # start/goal distance from the walls
    success_dist: float = 0.05

    # barrier bisecting start and goal, running along x
    obstacle_width: float = 0.5  # half-extent along x
    obstacle_thickness: float = 0.05
    obstacle_random: bool = False  # randomize per-world obstacle size and position


@wp.kernel(enable_backward=False)
def pd_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target: wp.array(dtype=wp.vec3),
    kp: float,
    kd: float,
    bodies_per_world: int,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    bid = tid * bodies_per_world
    pos = wp.transform_get_translation(body_q[bid])
    vel = wp.spatial_top(body_qd[bid])  # linear velocity
    f = kp * (target[tid] - pos) - kd * vel
    body_f[bid] = wp.spatial_vector(f[0], f[1], 0.0, 0.0, 0.0, 0.0)


class ParticleEnv:
    def __init__(self, cfg: ParticleConfig, viewer):
        self.cfg = cfg
        self.frame_dt = 1.0 / cfg.fps
        self.sim_dt = self.frame_dt / cfg.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.rng = np.random.default_rng(0)

        scene = newton.ModelBuilder()
        world = self.build_world()
        self.obstacle_box = self.sample_obstacles()
        for i in range(cfg.world_count):
            scene.begin_world()
            scene.add_builder(world)
            if cfg.obstacle:
                self.add_obstacle(scene, self.obstacle_box[i])
            scene.end_world()
        scene.add_ground_plane()

        self.model = scene.finalize()
        n = cfg.world_count
        self.num_bodies_per_world = self.model.body_count // n
        self.coords_per_world = self.model.joint_coord_count // n

        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.obstacle_sensor = (
            ObstacleContactSensor(self.model) if cfg.obstacle else None
        )

        self.target = wp.zeros(n, dtype=wp.vec3)
        self.goal_pos = np.zeros((n, 2), np.float32)
        self.predicted_path = None
        self.episode_frames = round(cfg.task_time * cfg.fps)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets(wp.vec3(3.0 * cfg.arena_size, 0.0, 0.0))
        if hasattr(self.viewer, "picking_enabled"):
            self.viewer.picking_enabled = False
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -2.5 * cfg.arena_size, 2.5 * cfg.arena_size),
            pitch=-45.0,
            yaw=90.0,
        )

        self.reset()

    # ------------------------------------------------------------------ build
    def build_world(self):
        cfg = self.cfg
        builder = newton.ModelBuilder()

        ball_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, cfg.ball_radius), wp.quat_identity())
        )
        builder.add_shape_sphere(
            body=body, radius=cfg.ball_radius, cfg=ball_cfg, color=[0.8, 0.2, 0.2]
        )

        # four walls enclosing the square
        a, t, h = cfg.arena_size, cfg.wall_thickness, 0.5 * cfg.wall_height
        for i, (x, y) in enumerate([(a + t, 0.0), (-a - t, 0.0)]):
            builder.add_shape_box(
                body=-1,
                hx=t,
                hy=a + 2.0 * t,
                hz=h,
                xform=wp.transform(wp.vec3(x, y, h), wp.quat_identity()),
                label=f"wall_x{i}",
                color=[0.5, 0.5, 0.5],
            )
        for i, (x, y) in enumerate([(0.0, a + t), (0.0, -a - t)]):
            builder.add_shape_box(
                body=-1,
                hx=a + 2.0 * t,
                hy=t,
                hz=h,
                xform=wp.transform(wp.vec3(x, y, h), wp.quat_identity()),
                label=f"wall_y{i}",
                color=[0.5, 0.5, 0.5],
            )
        return builder

    def sample_obstacles(self):
        """Per-world obstacle box [cx, cy, hx, hy], fixed unless obstacle_random."""
        cfg = self.cfg
        n, a = cfg.world_count, cfg.arena_size
        box = np.tile(
            np.array(
                [0.0, 0.0, cfg.obstacle_width, cfg.obstacle_thickness], np.float32
            ),
            (n, 1),
        )
        if cfg.obstacle_random:
            # ranges keep a >=0.2*a gap to the walls so every world stays solvable
            box[:, 0] = self.rng.uniform(-0.2 * a, 0.2 * a, n)
            box[:, 1] = self.rng.uniform(-0.3 * a, 0.3 * a, n)
            box[:, 2] = self.rng.uniform(0.3 * a, 0.6 * a, n)
            box[:, 3] = self.rng.uniform(0.03 * a, 0.1 * a, n)
        return box

    def add_obstacle(self, builder, box):
        cx, cy, hx, hy = (float(v) for v in box)
        h = 0.5 * self.cfg.wall_height
        builder.add_shape_box(
            body=-1,
            hx=hx,
            hy=hy,
            hz=h,
            xform=wp.transform(wp.vec3(cx, cy, h), wp.quat_identity()),
            label="obstacle",
            color=[0.5, 0.5, 0.5],
        )

    # ----------------------------------------------------------------- reset
    def sample(self, side: float = 0.0):
        """Uniform xy in the arena; side=+/-1 keeps y on that side of the obstacle."""
        n = self.cfg.world_count
        lim = self.cfg.arena_size - self.cfg.sample_margin
        pos = self.rng.uniform(-lim, lim, (n, 2)).astype(np.float32)
        if side != 0.0:
            cy, hy = self.obstacle_box[:, 1], self.obstacle_box[:, 3]
            lo = side * cy + hy + self.cfg.sample_margin
            pos[:, 1] = (side * self.rng.uniform(lo, lim)).astype(np.float32)
        return pos

    def reset(self):
        n = self.cfg.world_count
        if self.cfg.obstacle:
            self.start_pos = self.sample(side=1.0)
            self.goal_pos = self.sample(side=-1.0)
        else:
            self.start_pos = self.sample()
            self.goal_pos = self.sample()
        self.task_time = 0.0
        if self.obstacle_sensor is not None:
            self.obstacle_sensor.reset()

        jq = np.zeros((n, self.coords_per_world), np.float32)
        jq[:, :2] = self.start_pos
        jq[:, 2] = self.cfg.ball_radius
        jq[:, 6] = 1.0  # identity quat (xyzw)
        wp.copy(self.state_0.joint_q, wp.array(jq.flatten(), dtype=wp.float32))
        self.state_0.joint_qd.zero_()
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
        )

        self.set_target(self.start_pos)

    # ------------------------------------------------------------------ step
    def set_target(self, xy):
        n = self.cfg.world_count
        a = np.asarray(xy, np.float32).reshape(n, 2)
        target = np.concatenate(
            [a, np.full((n, 1), self.cfg.ball_radius, np.float32)], axis=1
        )
        wp.copy(self.target, wp.array(np.ascontiguousarray(target), dtype=wp.vec3))

    def apply_action(self, action):
        """Drive the ball with target-position actions, (world_count, 2)."""
        self.set_target(action)
        self.simulate()
        self.sim_time += self.frame_dt

    def step(self):
        """Scripted demo: ramp the target from start to goal, then hold to settle."""
        self.task_time += self.frame_dt
        t = min(1.0, self.task_time / (0.75 * self.cfg.task_time))
        self.set_target(self.start_pos * (1.0 - t) + self.goal_pos * t)
        self.simulate()
        self.sim_time += self.frame_dt

        new_episode = False
        success = None
        if self.task_time >= self.cfg.task_time or self.failure().all():
            new_episode = True
            success = self.success()
            self.reset()
        return new_episode, success

    def record_frame(self):
        # action: target xy position (2)
        return self.get_obs(), self.target.numpy()[:, :2].astype(np.float32)

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        if self.obstacle_sensor is not None:
            self.obstacle_sensor.update(self.contacts, self.state_0.body_q)
        for _ in range(self.cfg.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                pd_force_kernel,
                dim=self.cfg.world_count,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.target,
                    self.cfg.kp,
                    self.cfg.kd,
                    self.num_bodies_per_world,
                ],
                outputs=[self.state_0.body_f],
            )
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    # ----------------------------------------------------------- observation
    def get_obs(self):
        # ball xy pos (2), xy vel (2), goal xy (2)
        n = self.cfg.world_count
        q = self.state_0.body_q.numpy().reshape(n, self.num_bodies_per_world, 7)
        qd = self.state_0.body_qd.numpy().reshape(n, self.num_bodies_per_world, 6)
        pos, vel = q[:, 0, :2], qd[:, 0, :2]  # linear velocity first
        return np.concatenate([pos, vel, self.goal_pos], axis=1).astype(np.float32)

    def success(self):
        """Per-world bool: ball within success_dist of the goal, obstacle untouched."""
        obs = self.get_obs()
        dist = np.linalg.norm(obs[:, :2] - self.goal_pos, axis=1)
        return (dist < self.cfg.success_dist) & ~self.failure()

    def failure(self):
        """Per-world bool: ball has touched the obstacle this episode."""
        if self.obstacle_sensor is None:
            return np.zeros(self.cfg.world_count, dtype=bool)
        return self.obstacle_sensor.read()

    # -------------------------------------------------------------- planning
    def endpoints(self, obs):
        """Physical (start, goal) xy for inpainting/tracking; obs=[pos,vel,goal]."""
        return obs[:, :2], obs[:, 4:6]

    # -------------------------------------------------------------- guidance
    def make_guidance(self, action_mean, action_std, scale, margin, device, smooth=0.0):
        """Cost gradient keeping planned xy targets clear of the obstacle."""
        return ObstacleGuidance(
            center=self.obstacle_box[:, :2],
            half_extents=self.obstacle_box[:, 2:],
            action_mean=action_mean,
            action_std=action_std,
            scale=scale,
            margin=margin,
            device=device,
            around=True,
            smooth=smooth,
        )

    # --------------------------------------------------------------- render
    def set_predicted_path(self, positions):
        """Store a planned xy path (steps, 2+) for visualization (world 0)."""
        pts = np.ascontiguousarray(positions, np.float32)[:, :2]
        z = np.full((len(pts), 1), self.cfg.ball_radius, np.float32)
        self.predicted_path = np.concatenate([pts, z], axis=1)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.log_goal()
        self.log_predicted_path()
        self.viewer.end_frame()
        wp.synchronize()

    def log_goal(self):
        offsets = getattr(self.viewer, "world_offsets", None)
        off = offsets.numpy() if offsets is not None else 0.0
        n = len(self.goal_pos)
        z = np.full((n, 1), self.cfg.ball_radius, np.float32)
        goal = np.concatenate([self.goal_pos, z], axis=1) + off
        radii = wp.full(n, 0.02, dtype=wp.float32)
        colors = wp.full(n, wp.vec3(0.2, 0.8, 0.2), dtype=wp.vec3)
        self.viewer.log_points(
            "goal", wp.array(goal.astype(np.float32), dtype=wp.vec3), radii, colors
        )

    def log_predicted_path(self):
        if self.predicted_path is None:
            return
        offsets = getattr(self.viewer, "world_offsets", None)
        off = offsets.numpy()[0] if offsets is not None else 0.0
        pts = (self.predicted_path + off).astype(np.float32)
        n = len(pts)
        radii = wp.full(n, 0.01, dtype=wp.float32)
        colors = wp.full(n, wp.vec3(0.2, 0.4, 0.9), dtype=wp.vec3)
        self.viewer.log_points(
            "predicted_path", wp.array(pts, dtype=wp.vec3), radii, colors
        )
