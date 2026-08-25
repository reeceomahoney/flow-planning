import numpy as np

from flow_planning.bend import carry_segment, fk, solve_seq
from flow_planning.kinematics import build_piper_chain


def test_carry_segment_is_the_inner_closed_run():
    g = np.array([0.0] * 20 + [8.0] * 10 + [0.0] * 40 + [8.0] * 10 + [0.0] * 20)
    assert carry_segment(g, 1.5) == (30, 70)
    assert carry_segment(np.full(50, 8.0), 1.5) is None


def test_traj_seeded_ik_reproduces_the_demo():
    chain = build_piper_chain("cpu")
    t = np.linspace(0.0, 1.0, 12)[:, None]
    q = np.array([[0.3, 1.2, -0.8, 0.4, -0.6, 0.2]]) * t + np.array(
        [[-0.2, 0.4, -0.3, -0.1, -0.2, 0.5]]
    )
    ee, quat = fk(chain, q, np.zeros(3))
    out = solve_seq(chain, ee[:, None], quat[:, None], q[:, None], 4, 0.05, np.zeros(3))
    assert np.abs(out[:, 0] - q).max() < 1e-5
