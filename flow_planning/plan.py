import numpy as np


def steps(a, b, res):
    n = max(2, int(np.ceil(np.abs(b - a).max() / res)) + 1)
    return a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]


def rrt_connect(
    q_s, q_g, valid, lo, hi, rng, iters=2000, step=0.2, res=0.05, local=0.7
):
    if valid(steps(q_s, q_g, res)).all():
        return np.stack([q_s, q_g])
    nodes = [[np.asarray(q_s, float)], [np.asarray(q_g, float)]]
    par = [[-1], [-1]]

    def grow(k, target, greedy):
        arr = np.stack(nodes[k])
        i = int(np.abs(arr - target).max(1).argmin())
        q = arr[i]
        while True:
            d = target - q
            dist = float(np.abs(d).max())
            q_new = target if dist <= step else q + d * (step / dist)
            if not valid(steps(q, q_new, res)[1:]).all():
                return i, False
            nodes[k].append(q_new)
            par[k].append(i)
            i, q = len(nodes[k]) - 1, q_new
            if dist <= step:
                return i, True
            if not greedy:
                return i, False

    def trace(k, i):
        out = []
        while i >= 0:
            out.append(nodes[k][i])
            i = par[k][i]
        return out

    for it in range(iters):
        a = it % 2
        if rng.random() < local:
            u = rng.random()
            q_rand = q_s + u * (q_g - q_s) + rng.normal(0.0, 0.5, len(q_s))
            q_rand = np.clip(q_rand, lo, hi)
        else:
            q_rand = rng.uniform(lo, hi)
        ia, _ = grow(a, q_rand, False)
        ib, reached = grow(1 - a, nodes[a][ia], True)
        if reached:
            i0, i1 = (ia, ib) if a == 0 else (ib, ia)
            return np.stack(trace(0, i0)[::-1] + trace(1, i1)[1:])
    return None


def shortcut(path, valid, rng, res=0.05, iters=100):
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if valid(steps(path[i], path[j], res)).all():
            path = path[: i + 1] + path[j:]
    return np.stack(path)


def retime(path, n):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, path[:, j]) for j in range(path.shape[1])], 1)


def smooth_pinned(q, passes=3):
    q = q.copy()
    for _ in range(passes):
        q[1:-1] = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
    return q
