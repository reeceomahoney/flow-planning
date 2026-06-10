"""6D-rotation <-> quaternion conversions for EE-pose actions.

Orientation is carried as a 6D rotation (Zhou et al. 2019): the first two
columns of the rotation matrix. It is continuous, unlike quaternions, so the
flow model can interpolate it safely; Gram-Schmidt maps it back to SO(3).
Quaternion <-> matrix is delegated to scipy (xyzw convention).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_rot6d(q):
    """(n, 4) xyzw quaternions -> (n, 6) rotation representation."""
    R = Rotation.from_quat(q).as_matrix()
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)


def rot6d_to_quat(d):
    """(n, 6) rotation representation -> (n, 4) xyzw quaternions."""
    a1, a2 = d[:, 0:3], d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=2)
    return Rotation.from_matrix(R).as_quat().astype(np.float32)
