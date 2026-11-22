"""Charts for the technical document.

Numbers from the repo: the deck, PHASE_IMPROVEMENT_PLAN, vla/README.

Palette: reference categorical slots 1-2 (blue, orange), neutral text inks, recessive grid.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def finish(fig, ax, name):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=220)
    plt.close(fig)


# 1. Speed-up: planner output for the 12-object sort vs the ideal 1/N, with the binding limit
arms = [1, 2, 3, 4]
plan = [120, 64, 54, 54]
ideal = [120, 60, 40, 30]
limit = ["arms", "arms", "input bin", "input bin"]
fig, ax = plt.subplots(figsize=(6.4, 3.1))
x = range(len(arms))
bars = ax.bar(
    [i - 0.19 for i in x], plan, width=0.36, color=BLUE, label="Our plan (planner output)"
)
ax.bar([i + 0.19 for i in x], ideal, width=0.36, color=ORANGE, label="Ideal: 1 / N")
for i, (p, lim) in enumerate(zip(plan, limit, strict=True)):
    ax.text(i - 0.19, p + 3, f"{p} s", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(i, -24, f"limit: {lim}", ha="center", va="top", fontsize=8.5, color=INK2)
ax.set_xticks(list(x), [f"{a} arm{'s' if a > 1 else ''}" for a in arms])
ax.set_ylabel("Predicted finish (s)")
ax.set_ylim(0, 140)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title(
    "Sorting 12 objects: more arms help until a shared area binds",
    loc="left",
    fontsize=10.5,
    color=INK,
)
fig.subplots_adjust(bottom=0.24)
finish(fig, ax, "chart_speedup")

# 2. Real two-arm run: measured batch vs computed bounds (5 boxes, 22 Nov)
labels = [
    "One arm alone\n(computed)",
    "Two arms, perfect split\n(computed)",
    "Bowl, one arm at a time\n(computed)",
    "Best real two-arm run\n(measured, 09:20)",
]
vals = [145, 72, 75, 106]
cols = [BLUE, BLUE, BLUE, ORANGE]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ax.barh(range(4)[::-1], vals, color=cols, height=0.55)
for yy, v in zip(range(4)[::-1], vals, strict=True):
    ax.text(v + 2, yy, f"{'about ' if v != 106 else ''}{v} s", va="center", fontsize=9, color=INK)
ax.set_yticks(list(range(4))[::-1], labels, fontsize=8.8)
ax.set_xlabel("Time for 5 boxes (s)")
ax.set_xlim(0, 175)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_title(
    "Two real SO-101 arms, 5 boxes: measured against the bounds",
    loc="left",
    fontsize=10.5,
    color=INK,
)
fig.tight_layout()
fig.savefig(f"{OUT}/chart_real_run.png", dpi=220)
plt.close(fig)

# 3. X-VLA training loss on the 41 two-arm demos (log scale)
steps = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
loss = [0.79, 0.041, 0.022, 0.012, 0.007, 0.004, 0.004, 0.003]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(steps, loss, color=BLUE, linewidth=2, marker="o", markersize=4)
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Training loss (log)")
ax.annotate(
    "0.003",
    (7000, 0.003),
    textcoords="offset points",
    xytext=(0, 9),
    ha="center",
    fontsize=9,
    color=INK,
)
ax.annotate("0.79", (100, 0.79), textcoords="offset points", xytext=(6, -4), fontsize=9, color=INK)
ax.set_title("X-VLA, 41 two-arm demos", loc="left", fontsize=10, color=INK)
finish(fig, ax, "chart_xvla_loss")
print("charts done")
