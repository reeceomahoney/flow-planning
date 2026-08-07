import numpy as np

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
