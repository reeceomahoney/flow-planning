"""Franka FR3 pick-and-place environment built on newton, with physics + IK.

Shared by `record.py` (collect demos) and `play.py` (roll out a policy). A cube
is grasped at `cube_start` and placed at `cube_goal`; a scripted task state
machine drives an analytic-IK controller through a MuJoCo simulation (adapted
from newton's `ik_cube_stacking` example). The observation is the per-world
joint coords plus the cube and goal positions; the action is the 9 joint targets
(7 arm + 2 fingers).
"""

import enum
from dataclasses import dataclass

import newton
import newton.ik as ik
import newton.solvers
import newton.utils
import numpy as np
import warp as wp


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP_OFF = 4
    REFINE_DROP_OFF = 5
    RELEASE = 6
    RETRACT = 7
    HOME = 8


# home arm + finger configuration (fr3_franka_hand)
HOME_Q = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.05,
    0.05,
]


@dataclass
class FrankaConfig:
    world_count: int = 64
    ee_index: int = 11  # fr3_hand_tcp
    fps: int = 60
    sim_substeps: int = 10
    task_time: float = 1.0  # seconds per task phase
    ik_iters: int = 24
    cube_size: float = 0.05
    table_height: float = 0.1
    jitter: float = 0.1  # +/- xy range for cube/goal sampling
    success_dist: float = 0.05  # cube-to-goal distance for success (m)


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task: int,
    t: float,
    drop_off_pos: wp.array(dtype=wp.vec3),  # ty: ignore[invalid-type-form]
    off_approach: wp.vec3,
    off_lift: wp.vec3,
    off_retract: wp.vec3,
    home_pos: wp.vec3,
    task_init_body_q: wp.array(dtype=wp.transform),  # ty: ignore[invalid-type-form]
    body_q: wp.array(dtype=wp.transform),  # ty: ignore[invalid-type-form]
    ee_index: int,
    cube_body_index: int,
    num_bodies_per_world: int,
    # outputs
    ee_pos_out: wp.array(dtype=wp.vec3),  # ty: ignore[invalid-type-form]
    ee_rot_out: wp.array(dtype=wp.vec4),  # ty: ignore[invalid-type-form]
    gripper_out: wp.array2d(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    tid = wp.tid()

    ee_body_id = tid * num_bodies_per_world + ee_index  # ty: ignore[unsupported-operator]
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    obj_body_id = tid * num_bodies_per_world + cube_body_index  # ty: ignore[unsupported-operator]
    obj_pos = wp.transform_get_translation(body_q[obj_body_id])
    obj_quat = wp.transform_get_rotation(body_q[obj_body_id])

    drop = drop_off_pos[tid]
    ee_pos_target = home_pos
    ee_quat_target = ee_quat_down
    t_gripper = float(0.0)

    if task == TaskType.APPROACH.value:
        ee_pos_target = obj_pos + off_approach
        ee_quat_target = ee_quat_down * wp.quat_inverse(obj_quat)  # ty: ignore[unsupported-operator]
    elif task == TaskType.REFINE_APPROACH.value:
        ee_pos_target = obj_pos
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        ee_pos_target = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        ee_pos_target = ee_pos_prev + off_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_DROP_OFF.value:
        ee_pos_target = drop + off_approach
        t_gripper = 1.0
    elif task == TaskType.REFINE_DROP_OFF.value:
        ee_pos_target = drop
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        ee_pos_target = drop
        t_gripper = 1.0 - t
    elif task == TaskType.RETRACT.value:
        ee_pos_target = drop + off_retract
    else:  # HOME
        ee_pos_target = home_pos

    ee_pos_out[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target * t  # ty: ignore[unsupported-operator]
    q = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)
    ee_rot_out[tid] = wp.vec4(q[0], q[1], q[2], q[3])  # ty: ignore[not-subscriptable]

    gripper_pos = 0.06 * (1.0 - t_gripper)
    gripper_out[tid, 0] = gripper_pos
    gripper_out[tid, 1] = gripper_pos


class FrankaEnv:
    TASKS = list(TaskType)

    def __init__(self, cfg: FrankaConfig, viewer):
        self.cfg = cfg
        self.frame_dt = 1.0 / cfg.fps
        self.sim_dt = self.frame_dt / cfg.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.rng = np.random.default_rng(0)

        # scene geometry (sim frame): a table with the robot at one edge
        self.cube_size = cfg.cube_size
        h = cfg.table_height
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * h)
        top = wp.vec3(0.0, -0.5, h)
        self.robot_base_pos = wp.vec3(top[0] - 0.5, top[1], top[2])
        self.off_approach = wp.vec3(0.0, 0.0, 1.0 * self.cube_size)
        self.off_lift = wp.vec3(0.0, 0.0, 4.0 * self.cube_size)
        self.off_retract = wp.vec3(0.0, 0.0, 2.0 * self.cube_size)
        cube_z = top[2] + 0.5 * self.cube_size
        self.cube_center = np.array([top[0], top[1] + 0.15, cube_z], np.float32)
        self.goal_center = np.array([top[0], top[1] - 0.15, cube_z], np.float32)

        franka = self.build_franka_with_table()
        self.robot_body_count = franka.body_count
        scene = self.build_scene(franka)

        self.model_single = franka.finalize()
        self.model = scene.finalize()
        n = cfg.world_count
        self.num_bodies_per_world = self.model.body_count // n
        self.coords_per_world = self.model.joint_coord_count // n

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=20,
            ls_iterations=100,
            nconmax=1000,
            njmax=2000,
            cone="elliptic",
            impratio=1000.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # state used only for FK queries (EE home pose, reset body poses)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.setup_ik()
        self.setup_tasks()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
        if hasattr(self.viewer, "picking_enabled"):
            self.viewer.picking_enabled = False
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.5, 2.0), pitch=-25.0, yaw=50.0)

        self.episode_frames = self.num_tasks * round(cfg.task_time * cfg.fps)
        self.reset()

    # ------------------------------------------------------------------ build
    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_urdf(
            str(
                newton.utils.download_asset("franka_emika_panda")
                / "urdf/fr3_franka_hand.urdf"
            ),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),  # ty: ignore[missing-argument]
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        builder.joint_q[:9] = list(HOME_Q)
        builder.joint_target_pos[:9] = list(HOME_Q)
        builder.joint_target_ke[:9] = [
            4500,
            4500,
            3500,
            3500,
            2000,
            2000,
            2000,
            100,
            100,
        ]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # gravity compensation so position control tracks tightly
        gravcomp_dof = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        gravcomp_dof.values = gravcomp_dof.values or {}
        for dof in range(7):
            gravcomp_dof.values[dof] = True
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp_body.values = gravcomp_body.values or {}
        for body in range(2, 14):
            gravcomp_body.values[body] = 1.0

        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        shape_cfg.ke, shape_cfg.kd, shape_cfg.kf, shape_cfg.mu = 5e4, 5e2, 1e3, 0.75
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.cfg.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),  # ty: ignore[missing-argument]
            cfg=shape_cfg,
        )
        return builder

    def build_scene(self, franka: newton.ModelBuilder):
        scene = newton.ModelBuilder()
        for world_id in range(self.cfg.world_count):
            scene.begin_world()
            scene.add_builder(franka)
            self.add_cube(scene, world_id)
            scene.end_world()
        scene.add_ground_plane()
        return scene

    def add_cube(self, scene: newton.ModelBuilder, world_id: int):
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, margin=0.0)
        half = 0.5 * self.cube_size
        body = scene.add_body(
            xform=wp.transform(wp.vec3(*self.cube_center), wp.quat_identity())  # ty: ignore[missing-argument]
        )
        scene.add_shape_box(
            body=body,
            hx=half,
            hy=half,
            hz=half,
            cfg=shape_cfg,
            label=f"world_{world_id}/cube",
            color=[0.8, 0.2, 0.2],
        )

    # -------------------------------------------------------------------- ik
    def setup_ik(self):
        n = self.cfg.world_count
        body_q = self.state.body_q.numpy()
        ee_tf = body_q[self.cfg.ee_index]
        self.home_ee_pos = wp.vec3(*ee_tf[:3])
        ee_quat = wp.vec4(*ee_tf[3:7])

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.cfg.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_ee_pos] * n, dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.cfg.ee_index,
            link_offset_rotation=wp.quat_identity(),  # ty: ignore[invalid-argument-type, missing-argument]
            target_rotations=wp.array([ee_quat] * n, dtype=wp.vec4),
        )

        ik_dofs = self.model_single.joint_coord_count
        lower = wp.clone(self.model.joint_limit_lower.reshape((n, -1))[:, :ik_dofs])
        upper = wp.clone(self.model.joint_limit_upper.reshape((n, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=lower.flatten(),
            joint_limit_upper=upper.flatten(),
        )

        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((n, -1))[:, :ik_dofs])
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=n,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        n = self.cfg.world_count
        self.num_tasks = len(self.TASKS)
        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.ee_pos_target = wp.zeros(n, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(n, dtype=wp.vec4)
        self.gripper_target = wp.zeros((n, 2), dtype=wp.float32)
        self.goal_wp = wp.zeros(n, dtype=wp.vec3)
        self.task_idx = 0
        self.task_time = 0.0

    # ----------------------------------------------------------------- reset
    def sample(self, center):
        n = self.cfg.world_count
        pos = np.tile(center, (n, 1)).astype(np.float32)
        pos[:, 0] += self.rng.uniform(-self.cfg.jitter, self.cfg.jitter, n)
        pos[:, 1] += self.rng.uniform(-self.cfg.jitter, self.cfg.jitter, n)
        return pos

    def reset(self):
        n = self.cfg.world_count
        cube_start = self.sample(self.cube_center)
        self.goal_pos = self.sample(self.goal_center)

        # random cube yaw (xyzw quaternion)
        theta = self.rng.uniform(-0.9 * np.pi, 0.9 * np.pi, n)
        quat = np.zeros((n, 4), np.float32)
        quat[:, 2] = np.sin(theta / 2)
        quat[:, 3] = np.cos(theta / 2)

        jq = np.zeros((n, self.coords_per_world), np.float32)
        jq[:, :9] = HOME_Q
        jq[:, 9:12] = cube_start
        jq[:, 12:16] = quat
        wp.copy(self.state_0.joint_q, wp.array(jq.flatten(), dtype=wp.float32))
        self.state_0.joint_qd.zero_()

        home = np.tile(np.array(HOME_Q, np.float32), (n, 1))
        wp.copy(self.joint_q_ik, wp.array(home, dtype=wp.float32))
        wp.copy(self.goal_wp, wp.array(self.goal_pos, dtype=wp.vec3))

        # body poses for the first task's interpolation start
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        wp.copy(self.task_init_body_q, self.state_0.body_q)
        self.task_idx = 0
        self.task_time = 0.0

    # ------------------------------------------------------------------ step
    def set_joint_targets(self):
        n = self.cfg.world_count
        wp.launch(
            set_target_pose_kernel,
            dim=n,
            inputs=[
                int(self.TASKS[self.task_idx]),
                min(1.0, self.task_time / self.cfg.task_time),
                self.goal_wp,
                self.off_approach,
                self.off_lift,
                self.off_retract,
                self.home_ee_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.cfg.ee_index,
                self.robot_body_count,
                self.num_bodies_per_world,
            ],
            outputs=[self.ee_pos_target, self.ee_rot_target, self.gripper_target],
        )
        self.pos_obj.set_target_positions(self.ee_pos_target)
        self.rot_obj.set_target_rotations(self.ee_rot_target)
        jq = self.joint_q_ik
        self.ik_solver.step(jq, jq, iterations=self.cfg.ik_iters)  # ty: ignore[invalid-argument-type]

        jt = self.control.joint_target_pos.reshape((n, -1))
        wp.copy(dest=jt[:, :7], src=self.joint_q_ik[:, :7])
        wp.copy(dest=jt[:, 7:9], src=self.gripper_target[:, :2])

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.cfg.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.task_time += self.frame_dt
        self.set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

        new_episode = False
        if self.task_time >= self.cfg.task_time:
            self.task_time = 0.0
            if self.task_idx < self.num_tasks - 1:
                self.task_idx += 1
                wp.copy(self.task_init_body_q, self.state_0.body_q)
            else:
                new_episode = True
                self.reset()
        return new_episode

    def apply_action(self, joint_targets):
        """Drive the sim with externally supplied joint targets, (world_count, 9)."""
        n = self.cfg.world_count
        src = wp.array(np.asarray(joint_targets, np.float32).reshape(n, 9))
        jt = self.control.joint_target_pos.reshape((n, -1))
        wp.copy(dest=jt[:, :9], src=src)
        self.simulate()
        self.sim_time += self.frame_dt

    # ----------------------------------------------------------- observation
    def get_obs(self):
        # joint coords (9), cube position (3), goal position (3)
        n = self.cfg.world_count
        jq = self.state_0.joint_q.numpy().reshape(n, -1)
        obs = np.concatenate([jq[:, :9], jq[:, 9:12], self.goal_pos], axis=1)
        return obs.astype(np.float32)

    def record_frame(self):
        n = self.cfg.world_count
        action = self.control.joint_target_pos.numpy().reshape(n, -1)[:, :9]
        return self.get_obs(), action.astype(np.float32)

    def success(self):
        """Per-world bool: cube placed within success_dist of the goal."""
        n = self.cfg.world_count
        cube_pos = self.state_0.joint_q.numpy().reshape(n, -1)[:, 9:12]
        dist = np.linalg.norm(cube_pos - self.goal_pos, axis=1)
        return dist < self.cfg.success_dist

    # --------------------------------------------------------------- render
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.log_goal()
        self.viewer.end_frame()
        wp.synchronize()

    def log_goal(self):
        offsets = getattr(self.viewer, "world_offsets", None)
        off = offsets.numpy() if offsets is not None else 0.0
        n = len(self.goal_pos)
        goal = wp.array((self.goal_pos + off).astype(np.float32), dtype=wp.vec3)
        radii = wp.full(n, 0.02, dtype=wp.float32)
        colors = wp.full(n, wp.vec3(0.2, 0.8, 0.2), dtype=wp.vec3)
        self.viewer.log_points("goal", goal, radii=radii, colors=colors)
