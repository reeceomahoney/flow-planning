"""Franka FR3 reaching environment built on newton, with analytic IK.

Shared by `record.py` (collect IK demos) and `play.py` (roll out a policy):
the per-world joint coords are the observation and the TCP target is the action.
"""

from dataclasses import dataclass

import newton
import newton.ik as ik
import newton.utils
import numpy as np
import warp as wp


@dataclass
class FrankaConfig:
    world_count: int = 16
    ee_index: int = 11  # fr3_hand_tcp
    resample_period: float = 2.0  # seconds between new random targets
    ik_iters: int = 8  # per-frame IK iterations (target moves slowly, warm-starts)
    fps: int = 50

    # workspace box (in a single world's base frame) to sample TCP targets from
    pos_x: tuple[float, float] = (0.3, 0.6)
    pos_y: tuple[float, float] = (-0.3, 0.3)
    pos_z: tuple[float, float] = (0.2, 0.6)


class FrankaEnv:
    def __init__(self, cfg: FrankaConfig, viewer):
        self.cfg = cfg

        self.frame_dt = 1.0 / cfg.fps
        self.sim_time = 0.0
        self.time_since_resample = 0.0

        self.viewer = viewer
        self.rng = np.random.default_rng(0)

        # build a single FR3, then replicate across worlds
        franka = newton.ModelBuilder()
        franka.add_urdf(
            str(
                newton.utils.download_asset("franka_emika_panda")
                / "urdf/fr3_franka_hand.urdf"
            ),
            floating=False,
        )

        scene = newton.ModelBuilder()
        for _ in range(cfg.world_count):
            scene.begin_world()
            scene.add_builder(franka)
            scene.end_world()
        scene.add_ground_plane()

        self.model_single = franka.finalize()
        self.model = scene.finalize()
        self.viewer.set_model(self.model)

        # grid offset so the arms don't overlap
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets(wp.vec3(2.0, 2.0, 0.0))

        self.viewer.set_camera(
            pos=wp.vec3(4.0, -6.0, 4.0),
            pitch=-20.0,
            yaw=90.0,
        )

        self.state = self.model.state()
        self._eval_fk()

        self._setup_ik()

    def _eval_fk(self):
        # ty: stubs type joint_q/joint_qd as float | None; both are set post-finalize.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)  # ty: ignore[invalid-argument-type]

    def _setup_ik(self):
        n = self.cfg.world_count
        ik_dofs = self.model_single.joint_coord_count

        joint_limit_lower = self.model.joint_limit_lower
        joint_limit_upper = self.model.joint_limit_upper
        joint_q = self.model.joint_q
        body_q = self.state.body_q
        assert joint_limit_lower is not None
        assert joint_limit_upper is not None
        assert joint_q is not None
        assert body_q is not None

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.cfg.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.zeros(n, dtype=wp.vec3),
        )

        limit_lower = wp.clone(joint_limit_lower.reshape((n, -1))[:, :ik_dofs])
        limit_upper = wp.clone(joint_limit_upper.reshape((n, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=limit_lower.flatten(),
            joint_limit_upper=limit_upper.flatten(),
            weight=10.0,
        )

        # coords the solver updates, one row per world
        self.joint_q_ik = wp.clone(joint_q.reshape((n, -1))[:, :ik_dofs])

        self.solver = ik.IKSolver(
            model=self.model_single,
            n_problems=n,
            objectives=[self.pos_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.start_pos = (
            body_q.numpy().reshape(n, -1, 7)[:, self.cfg.ee_index, :3].copy()
        )
        self.goal_pos = self._sample_targets()

    def _sample_targets(self):
        cfg = self.cfg
        pos = np.empty((cfg.world_count, 3), dtype=np.float32)
        pos[:, 0] = self.rng.uniform(*cfg.pos_x, cfg.world_count)
        pos[:, 1] = self.rng.uniform(*cfg.pos_y, cfg.world_count)
        pos[:, 2] = self.rng.uniform(*cfg.pos_z, cfg.world_count)
        return pos

    def solve_to_target(self, target):
        """Run IK toward an externally supplied TCP target, shape (world_count, 3)."""
        self.target = target
        self.pos_obj.set_target_positions(wp.array(target, dtype=wp.vec3))

        # ty: stubs want array2d here; the cloned reshape view already is one.
        self.solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.cfg.ik_iters)  # ty: ignore[invalid-argument-type]

        # write solved coords back into the full model and run FK
        joint_q = self.model.joint_q
        assert joint_q is not None
        joint_q_view = joint_q.reshape((self.cfg.world_count, -1))
        wp.copy(
            dest=joint_q_view[:, : self.model_single.joint_coord_count],
            src=self.joint_q_ik,
        )
        self._eval_fk()

        self.sim_time += self.frame_dt

    def step(self):
        self.time_since_resample += self.frame_dt
        resampled = self.time_since_resample >= self.cfg.resample_period
        if resampled:
            # reached goal: hold there and pick a fresh target
            self.start_pos = self.goal_pos
            self.goal_pos = self._sample_targets()
            self.time_since_resample = 0.0

        # smoothstep ease from start to goal
        t = min(1.0, self.time_since_resample / self.cfg.resample_period)
        s = t * t * (3.0 - 2.0 * t)
        target = self.start_pos + (self.goal_pos - self.start_pos) * s
        self.solve_to_target(target)
        return resampled

    def record_frame(self):
        # per-world joint coords (state) and TCP target (action)
        joint_q = self.joint_q_ik.numpy().astype(np.float32)
        return joint_q, self.target.astype(np.float32)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize()
