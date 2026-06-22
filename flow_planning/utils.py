"""Small shared helpers: run-dir paths, 6D-rotation conversions, plan spacing."""

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import newton.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


def make_viewer(name: str, rrd: str = ""):
    if rrd:
        import newton._src.viewer.viewer_rerun as vr

        vr.is_jupyter_notebook = lambda: True  # ty: ignore[invalid-assignment]
        return newton.viewer.ViewerRerun(record_to_rrd=rrd, keep_historical_data=True)
    if name == "none":
        return newton.viewer.ViewerNull()
    if name == "opengl":
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
    runs = sorted(p for p in (Path(base) / env_name).glob("*/*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}/{env_name}")
    return runs[-1]


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


# --------------------------------------------------------------- plan spacing


def even_spacing(acts: Tensor, uniform: float) -> Tensor:
    """Reparametrize the planned action path toward equal arc-length spacing."""
    if uniform <= 0:
        return acts
    b, h, c = acts.shape
    seg = (acts[:, 1:] - acts[:, :-1]).norm(dim=-1)  # (B, H-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], dim=1)  # (B, H)
    total = cum[:, -1:]
    u = torch.linspace(0, 1, h, device=acts.device, dtype=acts.dtype)
    tgt = (1 - uniform) * cum + uniform * u[None, :] * total  # (B, H) arc lengths
    hi = torch.searchsorted(cum, tgt).clamp(1, h - 1)
    lo = hi - 1
    c_lo, c_hi = torch.gather(cum, 1, lo), torch.gather(cum, 1, hi)
    w = ((tgt - c_lo) / (c_hi - c_lo).clamp_min(1e-9)).unsqueeze(-1)
    g_lo = lo.unsqueeze(-1).expand(-1, -1, c)
    g_hi = hi.unsqueeze(-1).expand(-1, -1, c)
    p_lo, p_hi = torch.gather(acts, 1, g_lo), torch.gather(acts, 1, g_hi)
    return p_lo + w * (p_hi - p_lo)
