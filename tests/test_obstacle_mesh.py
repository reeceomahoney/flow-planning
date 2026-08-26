import numpy as np

from flow_planning.envs.franka import load_obstacle_mesh
from flow_planning.selector import LinkSDF


def test_bunny_loads_z_up_and_scaled(tmp_path):
    m = load_obstacle_mesh("bunny", 0.25)
    lo, hi = m.bounds
    assert abs(hi[2] - lo[2] - 0.25) < 1e-6 and abs(lo[2]) < 1e-6
    assert abs(lo[0] + hi[0]) < 1e-6 and abs(lo[1] + hi[1]) < 1e-6
    assert m.is_watertight

    eye = np.array([[0.7, 0.2, 0.9]])
    d = np.array([[0.0, 0.0, 0.125]]) - eye
    pts, _, _ = m.ray.intersects_location(
        eye, d / np.linalg.norm(d), multiple_hits=False
    )
    assert len(pts) == 1

    path = str(tmp_path / "bunny.obj")
    m.export(path)
    sdf = LinkSDF(path, m.bounds, "cpu", res=0.02, pad=0.1)
    import torch

    inside = torch.tensor([[0.0, 0.0, 0.1]])
    outside = torch.tensor([[0.0, 0.0, 0.5]])
    assert sdf(inside).item() < 0 < sdf(outside).item()
