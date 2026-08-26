import matplotlib.pyplot as plt

EPS = [64, 128, 256, 768]
# success, CAR per (arm, regime, test); None = not run yet
R = {
    ("DetAug", "frozen", "wall"): [
        (0.578, 0.92),
        (0.711, 0.89),
        (0.547, 0.90),
        (0.594, 0.91),
    ],
    ("DetAug", "frozen", "bunny"): [
        (0.906, 1.00),
        (0.922, 1.00),
        (0.656, 0.98),
        (0.859, 0.98),
    ],
    ("DemoGen-PC", "frozen", "wall"): [
        (0.391, 0.66),
        (0.586, 0.81),
        (0.625, 0.83),
        (0.617, 0.78),
    ],
    ("DemoGen-PC", "frozen", "bunny"): [
        (0.523, 0.95),
        (0.797, 0.98),
        (0.898, 0.98),
        (0.750, 0.94),
    ],
    ("DetAug", "full", "wall"): [(0.852, 0.95), (0.891, 0.95), None, None],
    ("DetAug", "full", "bunny"): [(0.898, 0.98), (0.898, 0.99), None, None],
    ("DemoGen-PC", "full", "wall"): [
        (0.555, 0.63),
        (0.930, 0.97),
        (0.953, 0.97),
        (0.914, 0.98),
    ],
    ("DemoGen-PC", "full", "bunny"): [
        (0.844, 0.96),
        (0.977, 1.00),
        (0.984, 1.00),
        (0.992, 1.00),
    ],
}
COLOR = {"DetAug": "#2a78d6", "DemoGen-PC": "#eb6834"}
STYLE = {"full": "-", "frozen": "--"}

fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
for j, test in enumerate(("wall", "bunny")):
    for i, (metric, name) in enumerate(
        ((0, "task success"), (1, "collision avoidance"))
    ):
        ax = axes[i, j]
        for arm in ("DetAug", "DemoGen-PC"):
            for regime in ("full", "frozen"):
                pts = [(e, v[metric]) for e, v in zip(EPS, R[(arm, regime, test)]) if v]
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    STYLE[regime],
                    color=COLOR[arm],
                    lw=2,
                    marker="o",
                    ms=5,
                    label=f"{arm}, {regime}" if (i, j) == (0, 0) else None,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(EPS, [str(e) for e in EPS])
        ax.set_ylim(0.3 if metric == 0 else 0.55, 1.02)
        ax.grid(True, color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_title(
                {
                    "wall": "0.25 m wall (in-dist. for DemoGen)",
                    "bunny": "bunny mesh (unseen)",
                }[test]
            )
        if j == 0:
            ax.set_ylabel(name)
        if i == 1:
            ax.set_xlabel("fine-tune episodes")
fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    "Grafting obstacle avoidance onto a free-space policy (128 eval eps, 1 seed)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("outputs/ft_results.png", dpi=160)
print("saved outputs/ft_results.png")
