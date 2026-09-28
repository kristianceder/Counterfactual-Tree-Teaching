"""Fig. 2, the supervision loop, drawn at the paper's text width (5.5 in) so the sizes below are the
printed sizes: 8 pt box titles, 7 pt edge labels. Three rows, three columns, every edge horizontal or
vertical:

    control loop  | Controller -> Report builder
    supervisor    | Recovery skills <- Decision tree <-> LLM teacher
    experience    | Counterfactual replay -> Experience memory <- Outcome judge

Only the edges that need telling apart or naming carry a label (miss / new rule, retire / in context,
route progress); the rest are read off the boxes they join. "Experience memory" is the code's
`ExperienceMemory`: the rules with the judge's scores and the replay's labels, shown to the teacher in
context on every call.

    python scripts/analysis/fig_loop.py        # -> results/figures/fig_loop.{pdf,png}

Uses Nimbus Roman (the class's Times) when the URW base-35 fonts are installed.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import tyro  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.normpath(os.path.join(HERE, "..", "..", "results", "figures"))
FONTS = "/usr/share/fonts/opentype/urw-base35"
for f in ("NimbusRoman-Regular.otf", "NimbusRoman-Bold.otf", "NimbusRoman-Italic.otf"):
    if os.path.exists(os.path.join(FONTS, f)):
        font_manager.fontManager.addfont(os.path.join(FONTS, f))
plt.rcParams.update({"font.family": "Nimbus Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42})

W, H = 5.5, 1.84                  # inches: the text width, and the height the three rows need
BW, BH = 1.36, 0.34               # box size (one line of text per box)
COL = {"A": 0.72, "B": 2.60, "C": 4.50}
ROW = {1: 1.58, 2: 0.98, 3: 0.30}
LANE = 5.36                       # the right-hand lane the judge's input runs down
EDGE = "#3a3a38"
LABEL = dict(fontsize=7, color="#3a3a38")
FILL = {"tree": "#e3edfb", "llm": "#fdeede", "judge": "#e2f4e7", "scores": "#fbf5d3", "replay": "#ececea",
        "plain": "#ffffff"}

BOXES = {  # name: (column, row, title, fill)
    "mpc": ("A", 1, "Controller", "plain"),
    "report": ("B", 1, "Report builder", "plain"),
    "skills": ("A", 2, "Recovery skills", "plain"),
    "tree": ("B", 2, "Decision tree", "tree"),
    "llm": ("C", 2, "LLM teacher", "llm"),
    "replay": ("A", 3, "Counterfactual replay", "replay"),
    "scores": ("B", 3, "Experience memory", "scores"),
    "judge": ("C", 3, "Outcome judge", "judge"),
}


def centre(name):
    c, r = BOXES[name][:2]
    return COL[c], ROW[r]


def side(name, where, offset=0.0):
    """A point on a box's edge: 'n', 's', 'e', 'w', shifted along the edge by `offset` inches."""
    x, y = centre(name)
    return {"n": (x + offset, y + BH / 2), "s": (x + offset, y - BH / 2),
            "e": (x + BW / 2, y + offset), "w": (x - BW / 2, y + offset)}[where]


def arrow(ax, points, label=None, at=None, ha="center", va="bottom", rotation=0):
    """A polyline through `points` with an arrowhead at the end; the label sits at `at`."""
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=7, color=EDGE,
                                 lw=0.8, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(*at, label, ha=ha, va=va, rotation=rotation, zorder=3, **LABEL)


def main(pdf: str = os.path.join(FIGURES, "fig_loop.pdf"), png: str = os.path.join(FIGURES, "fig_loop.png")) -> None:
    """Write the figure to PDF and PNG."""
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for name, (c, r, title, fill) in BOXES.items():
        x, y = centre(name)
        ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=FILL[fill], edgecolor=EDGE, lw=0.8, zorder=1))
        ax.text(x, y, title, ha="center", va="center", fontsize=8, fontweight="bold", color="#0b0b0b", zorder=3)

    g = 0.06  # label gap from a line
    # control loop: snapshot to the report builder, reference back from the recovery skills
    arrow(ax, [side("mpc", "e"), side("report", "w")])
    arrow(ax, [side("report", "s"), side("tree", "n")])
    # supervisor: the tree answers, or asks the teacher, whose answer becomes a rule
    arrow(ax, [side("tree", "e", 0.08), side("llm", "w", 0.08)], "miss", at=((COL["B"] + COL["C"]) / 2, ROW[2] + 0.08 + g * 0.6))
    arrow(ax, [side("llm", "w", -0.08), side("tree", "e", -0.08)], "new rule",
          at=((COL["B"] + COL["C"]) / 2, ROW[2] - 0.08 - g * 0.6), va="top")
    arrow(ax, [side("tree", "w"), side("skills", "e")])
    arrow(ax, [side("skills", "n"), side("mpc", "s")])
    # experience: recoveries to the replay, labels and verdicts onto the rules' scores
    arrow(ax, [side("skills", "s"), side("replay", "n")])
    arrow(ax, [side("replay", "e"), side("scores", "w")])
    arrow(ax, [side("judge", "w"), side("scores", "e")])
    arrow(ax, [side("scores", "n", -0.3), side("tree", "s", -0.3)], "retire", at=(COL["B"] - 0.3 - g, (ROW[2] + ROW[3]) / 2),
          ha="right", va="center")
    mid = ROW[3] + BH / 2 + 0.11   # low in the gap, clear of the "new rule" label above it
    arrow(ax, [side("scores", "n", 0.3), (COL["B"] + 0.3, mid), (COL["C"], mid), side("llm", "s")],
          "in context, every call", at=((COL["B"] + 0.3 + COL["C"]) / 2 + 0.22, mid + 0.03))
    # the judge reads route progress from every report
    arrow(ax, [side("report", "e"), (LANE, ROW[1]), (LANE, ROW[3]), side("judge", "e")],
          "route progress", at=((COL["B"] + BW / 2 + LANE) / 2, ROW[1] + g))

    os.makedirs(os.path.dirname(os.path.abspath(pdf)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print("wrote", os.path.normpath(pdf), "and", os.path.normpath(png))


if __name__ == "__main__":
    tyro.cli(main)
