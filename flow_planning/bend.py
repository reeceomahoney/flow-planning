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


def carry_segment(gripper, closed_thresh):
    closed = np.concatenate([[False], gripper < closed_thresh, [False]])
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    inner = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if a > 0]
    inner = [s for s in inner if s[1] < len(gripper)]
    return max(inner, key=lambda s: s[1] - s[0]) if inner else None


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


def solve_seq(chain, ee_t, quat_t, seed0, iters, damping, base, limits=None):
    """(T, B, 3/4) world targets + (B, 7) initial joints -> (T, B, 7).

    Damped least squares, batched over copies and SEQUENTIAL over frames: each
    frame warm-starts from the last. Independent solves hop posture families
    and are unusably jerky."""

    limits = limits or chain.get_joint_limits()
    lo, hi = (torch.tensor(x, device=chain.device) for x in limits)
    eye = damping * torch.eye(6, device=chain.device)
    p = torch.as_tensor(ee_t - base, dtype=torch.float32, device=chain.device)
    q = torch.as_tensor(
        quat_t[..., [3, 0, 1, 2]], dtype=torch.float32, device=chain.device
    )
    rot = quaternion_to_matrix(q)
    seed = torch.as_tensor(seed0, dtype=torch.float32, device=chain.device)
    traj = seed.dim() == 3  # (T, B, n): carry the delta from a reference path
    th = seed[0] if traj else seed
    out = torch.empty(p.shape[:2] + (th.shape[-1],), device=chain.device)
    for t in range(len(p)):
        if traj and t > 0:
            th = seed[t] + (th - seed[t - 1])
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


OBJ0 = 18
SEG_ZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LABEL_DIM = 8


def obj_slice(k):
    return slice(OBJ0 + 9 * k, OBJ0 + 9 * k + 3)


def held_object(obs, close, opened, n_obj):
    ee = obs[close:opened, EE] - obs[close, EE]
    best, best_err = None, 0.05
    for k in range(n_obj):
        p = obs[close:opened, obj_slice(k)] - obs[close, obj_slice(k)]
        err = np.abs(p - ee).mean()
        if err < best_err and np.linalg.norm(p[-1]) > 0.02:
            best, best_err = k, err
    return best


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of_rot6d(r6):
    return float(np.arctan2(r6[1], r6[0]))


def yaw_of_quat(q):
    from scipy.spatial.transform import Rotation as R

    m = R.from_quat(q).as_matrix()
    return float(np.arctan2(m[1, 0], m[0, 0]))


def shift_path(T, segs, m, dbs, dps):
    knots, vals = [0], [np.zeros(3)]
    for (c, o), db, dp in zip(segs, dbs, dps):
        for t, v in [(c - m, db), (c + m, db), (o - m, dp), (o + m, dp)]:
            knots.append(t)
            vals.append(v)
    knots.append(T - 1)
    vals.append(vals[-1])
    k = np.array(knots)
    for i in range(1, len(k)):
        k[i] = max(k[i], k[i - 1] + 1)
    keep = k < T
    k, v = k[keep], np.stack(vals)[keep]
    t = np.arange(T)
    return np.stack([np.interp(t, k, v[:, j]) for j in range(3)], axis=1)


def bulge_delta(ee, s0, s1, phi, theta, mode="additive"):
    d = bend_delta(ee, s0, s1, phi, theta)
    if d.any() and mode == "additive":
        d[s0:s1] += ee[s0:s1] - np.linspace(ee[s0], ee[s1 - 1], s1 - s0)
    return d


def descend_delta(ee, s0, s1, h):
    d = np.zeros_like(ee)
    n = s1 - s0
    if h < 1e-3 or n < 4:
        return d
    s = np.linspace(0.0, 1.0, n)
    d[s0:s1, 2] = h * np.sin(np.pi * s**2.5)
    return d


def grasp_rotation(ee, quat, w0, close, opened, w1, m, centre, alpha):
    from scipy.spatial.transform import Rotation as R

    T = len(ee)
    a1 = max(close - m, w0 + 1)
    centre = np.array([centre[0], centre[1], 0.0])
    oh = ee[close] - centre
    oh[2] = 0.0
    t = np.arange(T)
    ramp = np.clip((t - w0) / (a1 - w0), 0.0, 1.0)
    if w1 is not None:
        decay = np.clip(1 - (t - opened) / max(w1 - opened, 1), 0.0, 1.0)
        ramp = np.where(t > opened, decay, ramp)
    new_ee = ee.copy()
    off = rotz(alpha) @ oh - oh
    for i in range(w0, T):
        if i < a1:
            new_ee[i] = centre + rotz(ramp[i] * alpha) @ (ee[i] - centre)
        else:
            new_ee[i] = ee[i] + ramp[i] * off
    yaw = wrap(alpha)
    if np.isclose(abs(yaw), np.pi):
        yaw = -np.pi
    rot = R.from_euler("z", (ramp * yaw)[:, None]) * R.from_quat(quat)
    assert isinstance(rot, R)
    return new_ee, rot.as_quat().astype(np.float32)


def grasp_width(x, k, a_eff):
    from scipy.spatial.transform import Rotation as R

    c = x["segs"][k][0]
    ax = R.from_quat(x["quat"][c]).as_matrix()[:2, 1]
    ax = rotz(a_eff)[:2, :2] @ ax
    return 2 * float(np.abs(ax) @ x["half"][k])


def seg_options(alphas, phis, n_theta, modes, descents=(0.0,)):
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    arcs = [(0.0, 0.0)] + [(p, th) for p in phis for th in thetas]
    transit = [(0.0, 0.0, 0.0)] + [
        (p, th, float(i)) for i, _ in enumerate(modes) for p in phis for th in thetas
    ]
    return [
        (a, *ap, *tr) if dz == 0.0 else (a, *ap, *tr, dz)
        for a in alphas
        for ap in arcs
        for tr in transit
        for dz in descents
    ]


def encode_label(combo, n_seg):
    out = np.zeros(LABEL_DIM * n_seg, np.float32)
    for k, (a, pa, ta, pt, tt, md, *rest) in enumerate(combo[:n_seg]):
        r = np.radians(a)
        out[LABEL_DIM * k : LABEL_DIM * (k + 1)] = [
            np.sin(r),
            1 - np.cos(r),
            pa * np.cos(ta),
            pa * np.sin(ta),
            pt * np.cos(tt),
            pt * np.sin(tt),
            md,
            rest[0] if rest else 0.0,
        ]
    return out


def build(x, combo, m, modes):
    """x: demo dict (segs, held, obs, ee, quat, dyaw, half, dbs, dps, ends).

    Returns None if a rotation widens the grasp beyond the demo's."""
    segs = x["segs"]
    for k, (a, *_) in enumerate(combo):
        if x["held"][k] is None:
            continue
        a_eff = wrap(np.radians(a) + x["dyaw"][k])
        if grasp_width(x, k, a_eff) > grasp_width(x, k, x["dyaw"][k]) + 0.01:
            return None
    ee, quat = x["ee"].copy(), x["quat"].copy()
    objs = [
        x["obs"][:, obj_slice(j)].copy() if j is not None else None for j in x["held"]
    ]
    w0 = 0
    fam = set()
    base_shift = shift_path(len(x["ee"]), segs, m, x["dbs"], x["dps"])
    expected = [
        e + d for e, d, j in zip(x["ends"], x["dps"], x["held"]) if j is not None
    ]
    for k, ((c, o), (a, pa, ta, pt, tt, mode, *rest)) in enumerate(zip(segs, combo)):
        dz = rest[0] if rest else 0.0
        a_eff = wrap(np.radians(a) + x["dyaw"][k]) if x["held"][k] is not None else 0.0
        if abs(a_eff) > 1e-3:
            centre = x["obs"][0, obj_slice(x["held"][k])]
            w1 = segs[k + 1][0] - m if k + 1 < len(segs) else None
            ee, quat = grasp_rotation(ee, quat, w0, c, o, w1, m, centre, a_eff)
        if a:
            fam.add("rot")
        if pa:
            ee = ee + bulge_delta(ee, w0 + m if k else 1, c - m, pa * np.pi, ta)
            fam.add("app")
        if dz:
            ee = ee + descend_delta(ee, w0 + m if k else 1, c, dz)
            fam.add("desc")
        if pt:
            d = bulge_delta(ee, c + m, o - m, pt * np.pi, tt, modes[int(mode)])
            ee = ee + d
            objs[k] = objs[k] + d
            fam.add("tra")
        w0 = o
    return dict(
        fam="+".join(sorted(fam)) or "none",
        combo=combo,
        ee_t=ee + base_shift,
        quat=quat,
        objs=[o + base_shift if o is not None else None for o in objs],
        expected=expected,
    )
