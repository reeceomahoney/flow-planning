import matplotlib.pyplot as plt

EPS = [16, 32, 64, 128]
R = {
    ("DetAug", "wall"): [
        (0.445, 0.664),
        (0.438, 0.898),
        (0.570, 0.891),
        (0.547, 0.883),
    ],
    ("DetAug", "wall-dy"): [
        (0.391, 0.609),
        (0.320, 0.867),
        (0.445, 0.938),
        (0.375, 0.898),
    ],
    ("DetAug", "wall-dx"): [
        (0.531, 0.711),
        (0.461, 0.859),
        (0.656, 0.898),
        (0.602, 0.898),
    ],
    ("DemoGen-PC", "wall"): [
        (0.281, 0.570),
        (0.586, 0.797),
        (0.453, 0.797),
        (0.641, 0.820),
    ],
    ("DemoGen-PC", "wall-dy"): [
        (0.133, 0.477),
        (0.508, 0.852),
        (0.297, 0.688),
        (0.539, 0.828),
    ],
    ("DemoGen-PC", "wall-dx"): [
        (0.297, 0.633),
        (0.578, 0.836),
        (0.453, 0.773),
        (0.766, 0.891),
    ],
}
COLOR = {"DetAug": "#2a78d6", "DemoGen-PC": "#eb6834"}
TITLE = {
    "wall": "wall, centred (DemoGen train position)",
    "wall-dy": "wall shifted 8 cm toward goal",
    "wall-dx": "wall shifted 10 cm along x",
}

fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey="row")
for j, test in enumerate(("wall", "wall-dy", "wall-dx")):
    for i, (metric, name) in enumerate(
        ((0, "task success"), (1, "collision avoidance"))
    ):
        ax = axes[i, j]
        for arm in ("DetAug", "DemoGen-PC"):
            ys = [v[metric] for v in R[(arm, test)]]
            ax.plot(EPS, ys, "-", color=COLOR[arm], lw=2, marker="o", ms=5, label=arm)
        ax.set_xscale("log", base=2)
        ax.set_xticks(EPS, [str(e) for e in EPS])
        ax.set_ylim(0.1 if metric == 0 else 0.4, 1.0)
        ax.grid(True, color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_title(TITLE[test], fontsize=10)
        if j == 0:
            ax.set_ylabel(name)
        if i == 1:
            ax.set_xlabel("augmented episodes (+64 originals)")
axes[0, 0].legend(frameon=False, loc="lower right")
fig.suptitle(
    "Frozen-backbone graft, 0.25 m wall (128 eval eps, 1 seed; free space 0.98+)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("outputs/ft2_results.png", dpi=160)
print("saved outputs/ft2_results.png")
