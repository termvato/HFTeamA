"""Story diagram: see -> split -> run -> arms -> screen, with the plug-in / failure re-plan loop."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = sys.argv[1]
INK, INK2, LINE = "#0b0b0b", "#52514e", "#9a9893"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FILL = "#f4f7fc"

fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40.5)
ax.axis("off")

boxes = [  # x, title, lines
    (
        1,
        "1  See",
        [
            "Top camera + VLM",
            "(Qwen3-VL) names",
            "each object, picks",
            "its bin; ArUco maps",
            "pixels to the table",
        ],
    ),
    (
        21,
        "2  Split",
        [
            "Planner gives each",
            "step to an arm that",
            "can reach it; predicts",
            "the finish time and",
            "names the limit",
        ],
    ),
    (
        41,
        "3  Run",
        [
            "Executor dispatches",
            "steps, one arm at a",
            "time in a shared area;",
            "retries, heartbeat,",
            "re-plans on events",
        ],
    ),
    (
        61,
        "4  Arms",
        [
            "execute(task) ->",
            "ok | failed | timeout",
            "| arm_down",
            "real (LeRobot), sim",
            "(MuJoCo) or fake",
        ],
    ),
    (
        81,
        "5  Show",
        [
            "Live screen: count-",
            "down, what-if for",
            "1-4 arms, plan vs",
            "actual, the limit",
            "in words",
        ],
    ),
]
W, Y, H = 18, 14, 25
for x, title, lines in boxes:
    ax.add_patch(
        FancyBboxPatch(
            (x, Y),
            W,
            H,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=FILL,
            edgecolor=BLUE,
            linewidth=1.2,
        )
    )
    ax.text(x + 1.2, Y + H - 2.6, title, fontsize=10.5, weight="bold", color=INK, va="top")
    for k, ln in enumerate(lines):
        ax.text(x + 1.2, Y + H - 7.0 - k * 3.2, ln, fontsize=7.6, color=INK2, va="top")
for x, _, _ in boxes[:-1]:
    ax.add_patch(
        FancyArrowPatch(
            (x + W + 0.4, Y + H / 2),
            (x + 20 - 0.4, Y + H / 2),
            arrowstyle="-|>",
            mutation_scale=11,
            color=LINE,
            linewidth=1.3,
        )
    )

# the plug-and-produce loop: arm events go back to the planner
ax.add_patch(
    FancyBboxPatch(
        (41, 1.2),
        38,
        7.6,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#fff4ee",
        edgecolor=ORANGE,
        linewidth=1.2,
    )
)
ax.text(
    60,
    6.6,
    "Arm plugged in, removed or failed",
    fontsize=8.6,
    weight="bold",
    color=INK,
    ha="center",
    va="center",
)
ax.text(
    60,
    3.3,
    "found by its USB serial, homes slowly, joins",
    fontsize=7.6,
    color=INK2,
    ha="center",
    va="center",
)
ax.add_patch(
    FancyArrowPatch(
        (41, 5),
        (30, 13.6),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="-|>",
        mutation_scale=11,
        color=ORANGE,
        linewidth=1.5,
    )
)
ax.text(
    29.5,
    6.0,
    "re-plan the rest, live:\nrunning steps stay put",
    fontsize=7.6,
    color=INK,
    ha="right",
    va="center",
)
ax.add_patch(
    FancyArrowPatch(
        (70, 13.6), (70, 9.4), arrowstyle="-|>", mutation_scale=10, color=ORANGE, linewidth=1.3
    )
)
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/flow.png", dpi=240)
print("flow done")
