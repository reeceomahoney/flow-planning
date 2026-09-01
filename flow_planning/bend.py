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


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def track_delta(ee, s0, s1, targets, switches, tau, desc):
    d = np.zeros_like(ee)
    label = np.zeros((len(ee), 2), np.float32)
    if s1 - s0 < 4:
        return d, label
    up = np.array([0.0, 0.0, 1.0])
    chord = ee[s1 - 1] - ee[s0]
    chord = chord / np.linalg.norm(chord)
    vert = up - chord[2] * chord
    vert /= np.linalg.norm(vert)
    basis = np.stack([np.cross(chord, vert), vert])
    bounds = [s0, *switches, s1]
    for k, t in enumerate(targets):
        label[bounds[k] : bounds[k + 1]] = t
    o = v = np.zeros(3)
    for i in range(s0, s1):
        a = (label[i] @ basis - o) / tau**2 - 2.0 * v / tau
        v = v + a
        o = o + v
        d[i] = o
    env = smoothstep((s1 - 1 - np.arange(s0, s1)) / desc)
    d[s0:s1] *= env[:, None]
    return d, label


def fk(chain, q, base):  # (T, 7) joints -> world EE pos, quat xyzw
    th = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=chain.device)
    m = chain.forward_kinematics(th).get_matrix()
    pos = m[:, :3, 3].cpu().numpy() + base
    quat = matrix_to_quaternion(m[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
    return pos, quat


def solve_seq(chain, ee_t, quat_t, seed0, iters, damping, base, limits=None, ns=None):
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
    if ns is not None:
        ns_t, ns_w = ns
        ns_t = torch.as_tensor(ns_t, dtype=torch.float32, device=chain.device)
        ns_w = torch.as_tensor(ns_w, dtype=torch.float32, device=chain.device)
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
            if ns is not None and float(ns_w[t].max()) > 0:
                w = ns_w[t]
                dq = (w[..., None] if w.dim() else w) * (ns_t[t] - th)
                dq = (
                    dq
                    - (jt @ torch.linalg.solve(jac @ jt + eye, (jac @ dq[..., None])))[
                        ..., 0
                    ]
                )
                th = th + dq
            th = th.clamp(lo, hi)
        out[t] = th
    return out.cpu().numpy()


def tpad(a, tm):  # (T, ...) -> (tm, ...) repeating the last frame
    return np.concatenate([a, np.repeat(a[-1:], tm - len(a), axis=0)])


OBJ0 = 18
SEG_ZERO = (0.0, 0.0, 0.0, 0.0, 0.0)
LABEL_DIM = 6


def obj_slice(k):
    return slice(OBJ0 + 9 * k, OBJ0 + 9 * k + 3)


def held_segments(obs, gripper, n_obj):
    segs = grasp_segments(gripper)
    held = [held_object(obs, c, o, n_obj) for c, o in segs]
    keep = [(s, j) for s, j in zip(segs, held) if j is not None]
    return [s for s, _ in keep], [j for _, j in keep]


def held_object(obs, close, opened, n_obj):
    ee = obs[close:opened, EE] - obs[close, EE]
    best, best_err = None, 0.05
    for k in range(n_obj):
        p = obs[close:opened, obj_slice(k)] - obs[close, obj_slice(k)]
        err = np.abs(p - ee).mean()
        if err < best_err and np.linalg.norm(p[-1]) > 0.02:
            best, best_err = k, err
    return best


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


def bulge_delta(ee, s0, s1, phi, theta):
    d = bend_delta(ee, s0, s1, phi, theta)
    if d.any():
        d[s0:s1] += ee[s0:s1] - np.linspace(ee[s0], ee[s1 - 1], s1 - s0)
    return d


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hand_yaw(alpha, q7):
    y = ((alpha + np.pi / 2) % np.pi) - np.pi / 2
    opts = [y, y - np.pi, y + np.pi]
    return min(opts, key=lambda v: abs(q7 - v))


def grasp_rotation(
    ee, quat, w0, close, opened, w1, m, centre, alpha, yaw=None, lead=None
):
    from scipy.spatial.transform import Rotation as R

    T = len(ee)
    a1 = max(close - m, w0 + 1)
    if lead is not None:
        w0 = max(w0, a1 - lead)
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
    if yaw is None:
        yaw = wrap(alpha)
    rot = R.from_euler("z", (ramp * yaw)[:, None]) * R.from_quat(quat)
    assert isinstance(rot, R)
    return new_ee, rot.as_quat().astype(np.float32)


def grasp_width(x, k, a):
    from scipy.spatial.transform import Rotation as R

    c = x["segs"][k][0]
    ax = R.from_quat(x["quat"][c]).as_matrix()[:2, 1]
    ax = rotz(a)[:2, :2] @ ax
    return 2 * float(np.abs(ax) @ x["half"][k])


def seg_options(rng, phi_range, n_phi, n_theta, alphas=(0.0,)):
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    arcs = [(0.0, 0.0)] + [
        (float(rng.uniform(*phi_range)), th) for th in thetas for _ in range(n_phi)
    ]
    return [(a, *ap, *tr) for a in alphas for ap in arcs for tr in arcs]


def encode_label(combo, n_seg):
    out = np.zeros(LABEL_DIM * n_seg, np.float32)
    for k, (a, pa, ta, pt, tt) in enumerate(combo[:n_seg]):
        r = np.radians(a)
        out[LABEL_DIM * k : LABEL_DIM * (k + 1)] = [
            np.sin(r),
            1 - np.cos(r),
            pa * np.cos(ta),
            pa * np.sin(ta),
            pt * np.cos(tt),
            pt * np.sin(tt),
        ]
    return out


def build(x, combo, m):
    segs = x["segs"]
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
    for k, ((c, o), (a, pa, ta, pt, tt)) in enumerate(zip(segs, combo)):
        j = x["held"][k]
        if a and j is not None:
            ra = np.radians(a)
            if grasp_width(x, k, ra) > grasp_width(x, k, 0.0) + 0.01:
                return None
            w1 = segs[k + 1][0] - m if k + 1 < len(segs) else None
            yaw = hand_yaw(ra, float(x["act"][c, 6]))
            ee, quat = grasp_rotation(
                ee, quat, w0, c, o, w1, m, x["obs"][0, obj_slice(j)], ra, yaw, 3 * m
            )
            fam.add("rot")
        if pa:
            ee = ee + bulge_delta(ee, w0 + m if k else 1, c - m, pa * np.pi, ta)
            fam.add("app")
        if pt:
            d = bulge_delta(ee, c + m, o - m, pt * np.pi, tt)
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
