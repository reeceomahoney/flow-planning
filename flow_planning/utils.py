"""Small shared helpers: run-dir paths, 6D-rotation conversions."""

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import newton.viewer
import numpy as np
from scipy.spatial.transform import Rotation


def make_viewer(name: str, headless: bool = False):
    if name == "none":
        return newton.viewer.ViewerNull()
    if name == "opengl":
        if headless:
            import pyglet

            pyglet.options["headless"] = True
            return newton.viewer.ViewerGL(width=960, height=540, headless=True)
        return newton.viewer.ViewerGL()
    if name == "rerun":
        web_port = 9090
        viewer = newton.viewer.ViewerRerun(web_port=web_port)
        if viewer._grpc_server_uri is not None:
            grpc_uri = quote(viewer._grpc_server_uri, safe="")
            print(f"Rerun viewer: http://localhost:{web_port}/?url={grpc_uri}")
        return viewer
    raise ValueError(f"Unknown viewer: {name}")


# --------------------------------------------------------------- run-dir paths
# runs live under <base>/<env>/<date>/<time>


def make_run_dir(base: str, env_name: str) -> Path:
    return Path(base) / env_name / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def latest_run_dir(base: str, env_name: str) -> Path:
    # sort by mtime, not name: clock/tz skew makes timestamp names unreliable
    runs = [p for p in (Path(base) / env_name).glob("*/*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}/{env_name}")
    return max(runs, key=lambda p: p.stat().st_mtime)


# ------------------------------------------------------------- 6D rotations
# Orientation is carried as a 6D rotation (Zhou et al. 2019): the first two
# columns of the rotation matrix. It is continuous, unlike quaternions, so the
# flow model can interpolate it safely; Gram-Schmidt maps it back to SO(3).
# Quaternion <-> matrix is delegated to scipy (xyzw convention).


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
