###########################################################################
# Test Newton env with 16 parallel Franka robots (standalone)
#
# Builds a Newton scene with 16 worlds, each containing a Franka FR3 arm
# (fixed base). A single multi-problem IK solver drives every arm's TCP
# smoothly towards a random target position, picking a new random target
# every 2 seconds. Kinematic only (no physics) - the IK solution is shown
# directly via forward kinematics.
#
# Command: python scripts/helpers/test_newton.py
###########################################################################

import newton
import newton.ik as ik
import newton.utils
import newton.viewer
import numpy as np
import warp as wp

WORLD_COUNT = 16
EE_INDEX = 11  # fr3_hand_tcp
RESAMPLE_PERIOD = 2.0  # seconds between new random targets
IK_ITERS = 8  # per-frame IK iterations (target moves slowly, solver warm-starts)

# Workspace box (in a single world's base frame) to sample TCP targets from.
POS_X = (0.3, 0.6)
POS_Y = (-0.3, 0.3)
POS_Z = (0.2, 0.6)


class FrankaEnv:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.time_since_resample = 0.0

        self.viewer = viewer
        self.rng = np.random.default_rng(0)

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base), then replicate across worlds
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        franka.add_urdf(
            str(
                newton.utils.download_asset("franka_emika_panda")
                / "urdf/fr3_franka_hand.urdf"
            ),
            floating=False,
        )

        scene = newton.ModelBuilder()
        for _ in range(WORLD_COUNT):
            scene.begin_world()
            scene.add_builder(franka)
            scene.end_world()
        scene.add_ground_plane()

        self.model_single = franka.finalize()
        self.model = scene.finalize()
        self.viewer.set_model(self.model)

        # Spread the worlds out in a grid so the arms don't overlap visually.
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets(wp.vec3(2.0, 2.0, 0.0))

        self.viewer.set_camera(
            pos=wp.vec3(4.0, -6.0, 4.0),
            pitch=-20.0,
            yaw=90.0,
        )

        # state, initialised from the model's default joint configuration
        self.state = self.model.state()
        self._eval_fk()

        self._setup_ik()

    def _eval_fk(self):
        # ty: stubs type joint_q/joint_qd as float | None; both are set post-finalize.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)  # ty: ignore[invalid-argument-type]

    # ----------------------------------------------------------------------
    # IK setup
    # ----------------------------------------------------------------------
    def _setup_ik(self):
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
            link_index=EE_INDEX,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.zeros(WORLD_COUNT, dtype=wp.vec3),
        )

        limit_lower = wp.clone(
            joint_limit_lower.reshape((WORLD_COUNT, -1))[:, :ik_dofs]
        )
        limit_upper = wp.clone(
            joint_limit_upper.reshape((WORLD_COUNT, -1))[:, :ik_dofs]
        )
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=limit_lower.flatten(),
            joint_limit_upper=limit_upper.flatten(),
            weight=10.0,
        )

        # Joint coords the solver updates (one row per world).
        self.joint_q_ik = wp.clone(joint_q.reshape((WORLD_COUNT, -1))[:, :ik_dofs])

        self.solver = ik.IKSolver(
            model=self.model_single,
            n_problems=WORLD_COUNT,
            objectives=[self.pos_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Start interpolating from the current TCP pose towards the first goal.
        self.start_pos = (
            body_q.numpy().reshape(WORLD_COUNT, -1, 7)[:, EE_INDEX, :3].copy()
        )
        self.goal_pos = self._sample_targets()

    def _sample_targets(self):
        pos = np.empty((WORLD_COUNT, 3), dtype=np.float32)
        pos[:, 0] = self.rng.uniform(*POS_X, WORLD_COUNT)
        pos[:, 1] = self.rng.uniform(*POS_Y, WORLD_COUNT)
        pos[:, 2] = self.rng.uniform(*POS_Z, WORLD_COUNT)
        return pos

    # ----------------------------------------------------------------------
    # Loop
    # ----------------------------------------------------------------------
    def step(self):
        self.time_since_resample += self.frame_dt
        if self.time_since_resample >= RESAMPLE_PERIOD:
            # Reached the goal; hold there and pick a fresh target to move to.
            self.start_pos = self.goal_pos
            self.goal_pos = self._sample_targets()
            self.time_since_resample = 0.0

        # Smoothstep interpolation between current and goal TCP positions.
        t = min(1.0, self.time_since_resample / RESAMPLE_PERIOD)
        s = t * t * (3.0 - 2.0 * t)
        target = self.start_pos + (self.goal_pos - self.start_pos) * s
        self.pos_obj.set_target_positions(wp.array(target, dtype=wp.vec3))

        # ty: stubs want array2d here; the cloned reshape view already is one.
        self.solver.step(self.joint_q_ik, self.joint_q_ik, iterations=IK_ITERS)  # ty: ignore[invalid-argument-type]

        # Write the solved arm coords back into the full model and run FK.
        joint_q = self.model.joint_q
        assert joint_q is not None
        joint_q_view = joint_q.reshape((WORLD_COUNT, -1))
        wp.copy(
            dest=joint_q_view[:, : self.model_single.joint_coord_count],
            src=self.joint_q_ik,
        )
        self._eval_fk()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        wp.synchronize()


def main():
    viewer = newton.viewer.ViewerGL()
    env = FrankaEnv(viewer)

    while viewer.is_running():
        if viewer.should_step():
            env.step()
        env.render()

    viewer.close()


if __name__ == "__main__":
    main()
