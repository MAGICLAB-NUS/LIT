import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE, LIT = "#C97B3A", "#2A5FB0"
INK, INK2, GRID = "#1a1a19", "#52514e", "#e6e5e1"

groups = [
    ("LIBERO-Plus  ·  Overall (7-axis mean, 10,030 tasks)", [
        ("π0.5",      68.97, 79.67),
        ("MolmoAct2", 63.62, 71.92),
        ("FAST-WAM",  51.44, 60.63),
        ("ImageWAM",  83.02, 86.89),
    ]),
    ("Real robot  ·  OOD conditions (3 tasks, MolmoAct2)", [
        ("Lighting",    53.3, 70.0),
        ("Camera",      30.0, 46.7),
        ("Distractors", 50.0, 63.3),
    ]),
]

plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 11})
fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.9), sharex=True,
                         gridspec_kw={"height_ratios": [4, 3], "hspace": 0.55})
fig.patch.set_facecolor("white")

for ax, (title, rows) in zip(axes, groups):
    ax.set_facecolor("white")
    dec = 2 if rows[0][1] % 1 and len(rows) == 4 else 1
    fmt = lambda v, d=dec: f"{v:.{d}f}"
    ax.text(103.5, len(rows) - 0.45, "Δ", ha="left", va="center", color=INK2, fontsize=10, clip_on=False)
    n = len(rows)
    ys = list(range(n))[::-1]
    for y, (name, b, l) in zip(ys, rows):
        ax.plot([b, l], [y, y], color=LIT, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([b], [y], s=64, color=BASE, zorder=3, edgecolor="white", linewidths=1.5)
        ax.scatter([l], [y], s=64, color=LIT,  zorder=3, edgecolor="white", linewidths=1.5)
        ax.text(b - 1.4, y, fmt(b), ha="right", va="center", color=INK2, fontsize=10)
        ax.text(l + 1.4, y, fmt(l), ha="left",  va="center", color=INK,  fontsize=10)
        ax.text(103.5, y, "+" + fmt(l - b), ha="left", va="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=11)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=8)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2, length=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

axes[-1].set_xlim(20, 100)
for ax in axes: ax.set_clip_on(False)
axes[-1].set_xticks(range(20, 101, 10))
axes[-1].set_xlabel("Success rate (%)", color=INK2, fontsize=10.5)

handles = [Line2D([], [], marker="o", ls="", ms=8, color=BASE, label="Base"),
           Line2D([], [], marker="o", ls="", ms=8, color=LIT,  label="LIT (ours)")]
axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
               frameon=False, fontsize=10.5, handletextpad=0.4, columnspacing=1.4)

out = "docs/static/images/results_strip"
fig.savefig(out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.15)
print("ok")
