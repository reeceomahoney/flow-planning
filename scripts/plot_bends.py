import json
from pathlib import Path

import draccus
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from augment import Config

from flow_planning.bend import EE, bend_delta, transit_segments

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPISODE = 0
COPIES = 8


def root(repo):
    return Path.home() / ".cache/huggingface/lerobot" / repo


def load(root, episode):
    t = pq.read_table(
        root / "data",
        columns=["observation.state", "action"],
        filters=[("episode_index", "==", episode)],
    )
    return tuple(
        np.stack(t[c].to_numpy(zero_copy_only=False))
        for c in ("observation.state", "action")
    )


def plot(trajs, path):
    cols = int(np.ceil(np.sqrt(len(trajs))))
    rows = int(np.ceil(len(trajs) / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    for i, (title, traj) in enumerate(trajs.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.plot(*traj.T, "o-", lw=1, ms=3)
        ax.scatter(*traj[0], c="g", s=40)
        ax.scatter(*traj[-1], c="r", s=40)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title)
        span = np.ptp(traj, 0)
        ax.set_box_aspect(np.maximum(span, 0.25 * span.max()))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


@draccus.wrap()
def main(cfg: Config):
    r = root(cfg.dst_repo)
    fps = json.loads((r / "meta/info.json").read_text())["fps"]
    obs, act = load(r, EPISODE)

    ee = obs[:, EE]
    close, opened = transit_segments(act[:, 7])
    margin = round(cfg.bend_margin * fps)
    print(f"{len(ee)} frames, close {close}, open {opened}, margin {margin}")

    rng = np.random.default_rng(cfg.seed)
    grid = {"original": ee}
    for _ in range(COPIES):
        phi = np.pi * rng.uniform(cfg.bend_min, cfg.bend_max)
        theta = rng.uniform(0.0, np.pi)
        d = bend_delta(ee, close + margin, opened - margin, phi, theta)
        grid[f"{phi / np.pi:.2f}pi, {theta / np.pi:.2f}pi"] = ee + d
    print(plot(grid, "outputs/bends.png"))


if __name__ == "__main__":
    main()
