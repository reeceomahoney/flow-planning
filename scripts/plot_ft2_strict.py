import matplotlib.pyplot as plt

EPS = [16, 32, 64, 128]
# (strict success = goal reached with no obstacle contact, lenient = incl. collisions)
R = {
    ("DetAug", "wall"): [
        (0.445, 0.445),
        (0.438, 0.438),
        (0.570, 0.586),
        (0.547, 0.547),
    ],
    ("DetAug", "wall-dy"): [
        (0.391, 0.391),
        (0.320, 0.320),
        (0.445, 0.445),
        (0.375, 0.375),
    ],
    ("DetAug", "wall-dx"): [
        (0.531, 0.531),
        (0.461, 0.461),
        (0.656, 0.664),
        (0.602, 0.602),
    ],
    ("DemoGen-PC", "wall"): [
        (0.281, 0.281),
        (0.586, 0.586),
        (0.453, 0.453),
        (0.641, 0.664),
    ],
    ("DemoGen-PC", "wall-dy"): [
        (0.133, 0.133),
        (0.508, 0.508),
        (0.297, 0.297),
        (0.539, 0.539),
    ],
    ("DemoGen-PC", "wall-dx"): [
        (0.297, 0.297),
        (0.578, 0.578),
        (0.453, 0.453),
        (0.766, 0.773),
    ],
}
COLOR = {"DetAug": "#2a78d6", "DemoGen-PC": "#eb6834"}
TITLE = {
    "wall": "wall, centred (DemoGen train position)",
    "wall-dy": "wall shifted 8 cm toward goal",
    "wall-dx": "wall shifted 10 cm along x",
}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, test in zip(axes, ("wall", "wall-dy", "wall-dx")):
    for arm in ("DetAug", "DemoGen-PC"):
        strict = [v[0] for v in R[(arm, test)]]
        lenient = [v[1] for v in R[(arm, test)]]
        ax.plot(EPS, lenient, "--", color=COLOR[arm], lw=1.2, alpha=0.5)
        ax.plot(EPS, strict, "-", color=COLOR[arm], lw=2, marker="o", ms=5, label=arm)
    ax.set_xscale("log", base=2)
    ax.set_xticks(EPS, [str(e) for e in EPS])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(TITLE[test], fontsize=10)
    ax.set_xlabel("augmented episodes (+64 originals)")
axes[0].set_ylabel("strict task success (no obstacle contact)")
axes[0].plot([], [], "--", color="#666666", lw=1.2, label="incl. collisions")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Frozen-backbone graft, 0.25 m wall (128 eval eps, 1 seed)", fontsize=11)
fig.tight_layout()
fig.savefig("outputs/ft2_strict.png", dpi=160)
print("saved outputs/ft2_strict.png")
