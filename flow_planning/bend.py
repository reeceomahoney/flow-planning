import numpy as np
import torch
from pytorch_kinematics.transforms import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

ARM, EE, ROT, CUBE = slice(0, 7), slice(9, 12), slice(12, 18), slice(18, 21)


def transit_segments(gripper: np.ndarray) -> tuple[int, int]:
    """(close, open) frame indices from the gripper command alone."""
    closed = gripper < 0.03
    if not closed.any():
        return 0, 0
    close = int(closed.argmax())
    after = ~closed[close:]
    opened = close + int(after.argmax()) if after.any() else len(gripper)
    return close, opened


def grasp_segments(gripper: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    closed = np.concatenate([[False], gripper < 0.03, [False]])
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    segs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]
    return [(a, b) for a, b in segs if b - a >= min_len]


def bend_delta(ee, s0, s1, phi, theta):
    d = np.zeros_like(ee)
    n = s1 - s0
    if phi < 1e-3 or n < 4:
        return d

    up = np.array([0.0, 0.0, 1.0])
    a, b = ee[s0], ee[s1 - 1]
    chord = b - a
    c = np.linalg.norm(chord)
    chord = chord / c
    vert = up - chord[2] * chord
    vert /= np.linalg.norm(vert)
    nhat = np.cos(theta) * np.cross(chord, vert) + np.sin(theta) * vert

    r = c / (2 * np.sin(phi))
    sag = r * (1 - np.cos(phi))
    psi = np.linspace(-1.0, 1.0, n)[:, None] * phi
    bulge = r * np.cos(psi) - r + sag
    arc = (a + b) / 2 + r * np.sin(psi) * chord + bulge * nhat

    d[s0:s1] = arc - ee[s0:s1]
    return d


def fk(chain, q, base):  # (T, 7) joints -> world EE pos, quat xyzw
    th = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=chain.device)
    m = chain.forward_kinematics(th).get_matrix()
    pos = m[:, :3, 3].cpu().numpy() + base
    quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
    return pos, quat


def solve_seq(chain, ee_t, quat_t, seed0, iters, damping, base):
    """(T, B, 3/4) world targets + (B, 7) initial joints -> (T, B, 7).

    Damped least squares, batched over copies and SEQUENTIAL over frames: each
    frame warm-starts from the last. Independent solves hop posture families
    and are unusably jerky."""

    lo, hi = (torch.tensor(x, device=chain.device) for x in chain.get_joint_limits())
    eye = damping * torch.eye(6, device=chain.device)
    p = torch.as_tensor(ee_t - base, dtype=torch.float32, device=chain.device)
    q = torch.as_tensor(
        quat_t[..., [3, 0, 1, 2]], dtype=torch.float32, device=chain.device
    )
    rot = quaternion_to_matrix(q)
    th = torch.as_tensor(seed0, dtype=torch.float32, device=chain.device)
    out = torch.empty(p.shape[:2] + (7,), device=chain.device)
    for t in range(len(p)):
        # IK loop
        for _ in range(iters):
            jac, m = chain.jacobian(th, ret_eef_pose=True)
            err = torch.cat(
                [
                    p[t] - m[:, :3, 3],
                    matrix_to_axis_angle(rot[t] @ m[:, :3, :3].transpose(1, 2)),
                ],
                dim=-1,
            )
            jt = jac.transpose(1, 2)
            th = th + (jt @ torch.linalg.solve(jac @ jt + eye, err[..., None]))[..., 0]
            th = th.clamp(lo, hi)
        out[t] = th
    return out.cpu().numpy()


def tpad(a, tm):  # (T, ...) -> (tm, ...) repeating the last frame
    return np.concatenate([a, np.repeat(a[-1:], tm - len(a), axis=0)])
