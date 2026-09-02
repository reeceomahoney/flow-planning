import numpy as np

from flow_planning.plan import retime, rrt_connect, shortcut, smooth_pinned


def test_rrt_around_wall():
    rng = np.random.default_rng(0)
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    def valid(q):
        q = np.atleast_2d(q)
        return ~((np.abs(q[:, 0]) < 0.1) & (q[:, 1] < 0.6))

    q_s, q_g = np.array([-0.8, 0.0]), np.array([0.8, 0.0])
    path = rrt_connect(q_s, q_g, valid, lo, hi, rng, step=0.2)
    assert path is not None and np.allclose(path[0], q_s) and np.allclose(path[-1], q_g)
    path = shortcut(path, valid, rng)
    q = retime(path, 40)
    assert q.shape == (40, 2) and valid(q).all() and q[:, 1].max() > 0.6
    sm = smooth_pinned(q)
    assert np.allclose(sm[0], q_s) and np.allclose(sm[-1], q_g)
