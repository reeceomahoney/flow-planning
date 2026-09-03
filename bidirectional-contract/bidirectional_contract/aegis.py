"""AEGIS baseline adapted from THU-RCSCT/vlsa-aegis.

The barrier equations and constants follow upstream commit
57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b. The released weighted one-constraint
QP is evaluated as an analytic projection onto a weighted half-space.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from robosuite.utils.control_utils import orientation_error
from scipy.spatial.transform import Rotation

from flow_planning.envs.libero import GRIPPER_OPEN, WORKSPACE, LiberoConfig, LiberoEnv


OSC_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], np.float32)


def _hat(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def _project_matrix(vector: np.ndarray) -> np.ndarray:
    vector = vector / (np.linalg.norm(vector) + 1e-12)
    return np.eye(3) - np.outer(vector, vector)


def compute_h(
    p_i: np.ndarray,
    q_i_diag: np.ndarray,
    r_i: np.ndarray,
    p_j: np.ndarray,
    q_j_diag: np.ndarray,
    r_j: np.ndarray,
    z_ij: np.ndarray,
) -> float:
    qbar_i = r_i @ np.diag(q_i_diag) @ r_i.T
    qbar_j = r_j @ np.diag(q_j_diag) @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    z = z_ij / np.linalg.norm(z_ij)
    term1 = np.linalg.norm(qbar_j @ qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ qbar_i_inv @ z
    denominator = np.linalg.norm(qbar_i_inv @ z)
    return float((-term1 + term2 - 1.0) / denominator)


def compute_h_coefficients(
    p_i: np.ndarray,
    q_i_diag: np.ndarray,
    r_i: np.ndarray,
    p_j: np.ndarray,
    q_j_diag: np.ndarray,
    r_j: np.ndarray,
    z: np.ndarray,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    qbar_i = r_i @ np.diag(q_i_diag) @ r_i.T
    qbar_j = r_j @ np.diag(q_j_diag) @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    qbar_i_inv2 = qbar_i_inv @ qbar_i_inv
    qbar_j2 = qbar_j @ qbar_j
    z = z / (np.linalg.norm(z) + epsilon)
    a_vec = qbar_i_inv @ z
    denominator = np.linalg.norm(a_vec) + epsilon
    b_vec = qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + epsilon
    sigma = term1 * denominator + epsilon
    rho = 1.0 - (p_j - p_i).T @ a_vec + term1

    eta_row = -(z.T @ qbar_i_inv) / denominator
    term_mu_1 = rho * (z.T @ qbar_i_inv2) / (denominator**3 + epsilon)
    term_mu_2 = ((p_j - p_i).T @ qbar_i_inv) / denominator
    term_mu_3 = (
        z.T @ qbar_i_inv @ qbar_j2 @ qbar_i_inv
    ) / sigma
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    temporary1 = z.T @ qbar_i_inv2 @ _hat(z)
    left = z.T @ qbar_i_inv @ qbar_j2
    temporary2 = left @ (_hat(a_vec) - qbar_i_inv @ _hat(z))
    temporary3 = (p_j - p_i).T @ qbar_i_inv @ _hat(z)
    temporary3 += z.T @ qbar_i_inv @ _hat(p_j - p_i)
    zeta_tilde = (
        rho * temporary1 / (denominator**3 + epsilon)
        + temporary2 / sigma
        + temporary3 / denominator
    )
    a_v = np.asarray(eta_row @ r_i).ravel()
    a_omega = np.asarray(zeta_tilde @ r_i).ravel()
    a_uz = np.asarray(mu_row @ _project_matrix(z)).ravel()
    h = compute_h(p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z)
    return a_v, a_omega, a_uz, h, np.asarray(mu_row).ravel()


def minimum_volume_enclosing_ellipsoid(
    points: np.ndarray, tolerance: float = 1e-7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    count, dimension = points.shape
    if count <= dimension:
        raise ValueError(f"Need more than {dimension} points, got {count}")
    q = np.vstack([points.T, np.ones(count)])
    u = np.full(count, 1.0 / count)
    for _ in range(100_000):
        x = (q * u) @ q.T
        values = np.einsum("ij,ji->i", q.T @ np.linalg.inv(x), q)
        index = int(values.argmax())
        maximum = float(values[index])
        step = (maximum - dimension - 1.0) / (
            (dimension + 1.0) * (maximum - 1.0)
        )
        updated = (1.0 - step) * u
        updated[index] += step
        if np.linalg.norm(updated - u) <= tolerance:
            u = updated
            break
        u = updated
    center = points.T @ u
    covariance = points.T @ (u[:, None] * points) - np.outer(center, center)
    shape = np.linalg.inv(covariance) / dimension
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    axes = 1.0 / np.sqrt(np.clip(eigenvalues, 1e-15, None))
    order = np.argsort(axes)[::-1]
    return center, eigenvectors[:, order], axes[order]


@dataclass
class AegisResult:
    action: np.ndarray
    h: float
    intervened: bool
    correction_norm: float


class AegisCBF:
    def __init__(
        self,
        obstacle_center: np.ndarray,
        obstacle_rotation: np.ndarray,
        obstacle_axes: np.ndarray,
        initial_obs: dict,
        safety_margin: float = 0.0,
    ):
        self.p2 = np.asarray(obstacle_center, dtype=np.float64)
        self.r2 = np.asarray(obstacle_rotation, dtype=np.float64)
        self.q2 = np.asarray(obstacle_axes, dtype=np.float64)
        if safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        self.safety_margin = float(safety_margin)
        p1, _, _ = self._eef_ellipsoid(initial_obs)
        self.z = self.p2 - p1
        self.z /= np.linalg.norm(self.z)
        self.dt = 0.05
        self.q1 = np.array([0.06, 0.12, 0.11], dtype=np.float64)

    @staticmethod
    def _eef_ellipsoid(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        rotation = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
        center = eef_pos + rotation @ np.array([0.0, 0.0, -0.08])
        return center, rotation, eef_pos

    def filter(self, nominal_action: np.ndarray, obs: dict) -> AegisResult:
        nominal_action = np.asarray(nominal_action, dtype=np.float64)
        p1, r1, _ = self._eef_ellipsoid(obs)
        a_v, a_omega, a_uz, h, mu = compute_h_coefficients(
            p1, self.q1, r1, self.p2, self.q2, self.r2, self.z
        )
        u_reference = np.concatenate(
            [
                5.0 * (r1.T @ nominal_action[:3]),
                5.0 * nominal_action[3:6],
                10.0 * mu,
            ]
        )
        a = np.concatenate([0.2 * a_v, 0.2 * a_omega, a_uz])
        weight_inverse = np.diag([25.0] * 6 + [1.0] * 3)
        constraint = float(
            a @ u_reference + 10.0 * (h - self.safety_margin)
        )
        u = u_reference.copy()
        if constraint < 0.0:
            direction = weight_inverse @ a
            denominator = float(a @ direction)
            if denominator > 1e-12:
                u += (-constraint / denominator) * direction
        self.z += _project_matrix(self.z) @ u[6:] * self.dt
        self.z /= np.linalg.norm(self.z)
        safe = nominal_action.copy()
        safe[:3] = 0.2 * (r1 @ u[:3])
        safe[3:6] = 0.2 * u[3:6]
        return AegisResult(
            action=safe,
            h=h,
            intervened=constraint < 0.0,
            correction_norm=float(
                np.linalg.norm(safe[:6] - nominal_action[:6])
            ),
        )


class AegisLiberoEnv(LiberoEnv):
    """SafeLIBERO with the OSC_POSE interface used by released AEGIS."""

    def make_env(self, config: LiberoConfig, bddl: Path, index: int):
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.errors import RandomizationError

        for attempt in range(10):
            try:
                return ControlEnv(
                    bddl_file_name=str(bddl),
                    controller="OSC_POSE",
                    use_camera_obs=config.render and index == 0,
                    has_offscreen_renderer=config.render and index == 0,
                    has_renderer=False,
                    camera_names=["agentview"] if config.render and index == 0 else [],
                    camera_heights=config.render_size,
                    camera_widths=config.render_size,
                )
            except RandomizationError:
                np.random.seed(attempt + 1)
        raise RuntimeError(f"could not build {bddl}")

    def reset(self):
        count = len(self.envs)
        self.obs = [None] * count
        self.obstacle = [None] * count
        self.obstacle_pos0 = np.zeros((count, 3), np.float32)
        self.collided = np.zeros(count, dtype=bool)
        self.succ = np.zeros(count, dtype=bool)
        self.contact = [""] * count
        self.step = 0
        self.path = None
        self.trace = []
        for index, env in enumerate(self.envs):
            env.reset()
            state = self.init_states[self.episode_idx % len(self.init_states)]
            self.episode_idx += 1
            self.obs[index] = env.set_init_state(state)
            for _ in range(self.cfg.settle_steps):
                self.obs[index] = env.step(OSC_DUMMY_ACTION)[0]
            if not self.cfg.obstacle:
                continue
            for name in env.env.objects_dict:
                position = self.obs[index].get(f"{name}_pos")
                if "obstacle" not in name or position is None:
                    continue
                if (
                    position[2] > 0
                    and abs(position[0]) < WORKSPACE
                    and abs(position[1]) < WORKSPACE
                ):
                    self.obstacle[index] = name
                    self.obstacle_pos0[index] = position
                    break

    def apply_osc_action(self, action: np.ndarray):
        env = self.envs[0]
        self.step += 1
        self.obs[0] = env.step(np.asarray(action, np.float32))[0]
        self.succ[0] |= bool(env.check_success())
        name = self.obstacle[0]
        if name is not None and not self.collided[0]:
            moved = np.abs(
                self.obs[0][f"{name}_pos"] - self.obstacle_pos0[0]
            ).sum()
            self.collided[0] = moved > self.cfg.collision_displacement
            if self.collided[0]:
                self.contact[0] = self.contact_body(0, name, action[6] > 0)


class JointTargetToOSC:
    """Convert an absolute joint target to a normalized OSC pose delta."""

    def __init__(self, env: AegisLiberoEnv):
        sim = env.envs[0].sim
        self.model = sim.model._model
        self.data = sim.data._data
        self.shadow = mujoco.MjData(self.model)
        self.site = self.model.site("gripper0_grip_site").id
        self.qpos_addresses = np.array(
            [self.model.joint(f"robot0_joint{i}").qposadr[0] for i in range(1, 8)]
        )
        controller = env.envs[0].env.robots[0].controller
        self.output_scale = np.asarray(controller.output_max, dtype=np.float64)

    def __call__(self, joint_action: np.ndarray) -> np.ndarray:
        joint_action = np.asarray(joint_action, dtype=np.float64)
        self.shadow.qpos[:] = self.data.qpos
        self.shadow.qvel[:] = 0.0
        self.shadow.qpos[self.qpos_addresses] = joint_action[:7]
        mujoco.mj_forward(self.model, self.shadow)
        current_position = self.data.site_xpos[self.site].copy()
        current_rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        target_position = self.shadow.site_xpos[self.site].copy()
        target_rotation = self.shadow.site_xmat[self.site].reshape(3, 3).copy()
        delta = np.concatenate(
            [
                target_position - current_position,
                orientation_error(target_rotation, current_rotation),
            ]
        )
        osc = np.empty(7, dtype=np.float64)
        osc[:6] = np.clip(delta / self.output_scale, -1.0, 1.0)
        osc[6] = 1.0 if joint_action[7] < 0.5 * GRIPPER_OPEN else -1.0
        return osc


def obstacle_points(env: AegisLiberoEnv) -> np.ndarray:
    """World-space corners of collision geoms in the active obstacle."""
    instance = env.envs[0]
    model, data = instance.sim.model._model, instance.sim.data._data
    body = model.body(f"{env.obstacle[0]}_main").id
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).reshape(3, -1).T
    points = []
    for geom in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[geom]] != body:
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        local = model.geom_aabb[geom, :3] + signs * model.geom_aabb[geom, 3:]
        world = (
            data.geom_xpos[geom]
            + local @ data.geom_xmat[geom].reshape(3, 3).T
        )
        points.append(world)
    return np.concatenate(points)
