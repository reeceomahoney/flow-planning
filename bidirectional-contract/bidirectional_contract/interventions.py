"""Geometry shared by recovery-data generation and inference-time intervention."""

import numpy as np
import torch
from pytorch_kinematics.transforms import matrix_to_axis_angle, quaternion_to_matrix


def local_normal_frame(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return side/up axes spanning the plane normal to path motion."""
    tangent = np.gradient(positions, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), tangent.shape).copy()
    vertical = up - (up * tangent).sum(1, keepdims=True) * tangent
    bad = np.linalg.norm(vertical, axis=1) < 1e-4
    if bad.any():
        fallback = np.broadcast_to(np.array([1.0, 0.0, 0.0]), tangent.shape).copy()
        fallback -= (fallback * tangent).sum(1, keepdims=True) * tangent
        vertical[bad] = fallback[bad]
    vertical /= np.maximum(np.linalg.norm(vertical, axis=1, keepdims=True), 1e-8)
    side = np.cross(tangent, vertical)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-8)
    return side, vertical


def recovery_profile(length: int, recovery_frames: int) -> np.ndarray:
    """Quintic 1->0 contraction with zero velocity at both endpoints."""
    u = np.clip(
        np.arange(length, dtype=np.float64) / max(recovery_frames, 1), 0.0, 1.0
    )
    smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
    return 1.0 - smooth


def ramp_profile(length: int, ramp_frames: int) -> np.ndarray:
    """Quintic 0->1 controller ramp followed by a hold."""
    u = np.clip(np.arange(1, length + 1) / max(ramp_frames, 1), 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def normal_direction(
    side: np.ndarray,
    vertical: np.ndarray,
    direction_index: int,
    direction_count: int = 5,
) -> np.ndarray:
    """One ray from the upper normal half-plane."""
    angle = np.linspace(0.0, np.pi, direction_count)[direction_index]
    return np.cos(angle) * side + np.sin(angle) * vertical


def solve_recovery_ik(
    chain,
    positions: np.ndarray,
    quaternions: np.ndarray,
    reference_q: np.ndarray,
    seed_q: np.ndarray,
    iterations: int,
    damping: float,
    posture_gain: float,
    base: np.ndarray,
) -> np.ndarray:
    """Sequential Cartesian IK with a demonstrated-posture null-space pull."""
    device = chain.device
    low, high = (
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in chain.get_joint_limits()
    )
    position = torch.as_tensor(
        positions - base, dtype=torch.float32, device=device
    )
    quaternion = torch.as_tensor(
        quaternions[..., [3, 0, 1, 2]], dtype=torch.float32, device=device
    )
    rotation = quaternion_to_matrix(quaternion)
    q_nominal = torch.as_tensor(reference_q, dtype=torch.float32, device=device)
    q = torch.as_tensor(seed_q, dtype=torch.float32, device=device)
    batch = q.shape[0]
    eye6 = damping * torch.eye(6, dtype=torch.float32, device=device)
    eye7 = torch.eye(7, dtype=torch.float32, device=device).expand(batch, -1, -1)
    output = torch.empty(
        position.shape[:2] + (7,), dtype=torch.float32, device=device
    )

    for step in range(len(position)):
        for _ in range(iterations):
            jacobian, matrix = chain.jacobian(q, ret_eef_pose=True)
            error = torch.cat(
                [
                    position[step] - matrix[:, :3, 3],
                    matrix_to_axis_angle(
                        rotation[step] @ matrix[:, :3, :3].transpose(1, 2)
                    ),
                ],
                dim=-1,
            )
            jacobian_t = jacobian.transpose(1, 2)
            gram = jacobian @ jacobian_t + eye6
            task_delta = (
                jacobian_t @ torch.linalg.solve(gram, error[..., None])
            )[..., 0]
            damped_pinv_j = jacobian_t @ torch.linalg.solve(gram, jacobian)
            null_delta = (eye7 - damped_pinv_j) @ (
                q_nominal[step] - q
            )[..., None]
            q = q + task_delta + posture_gain * null_delta[..., 0]
            q = q.clamp(low, high)
        output[step] = q
    return output.cpu().numpy()
