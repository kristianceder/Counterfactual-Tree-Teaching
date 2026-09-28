"""The scenario grid (appendix figure), one staged situation per panel, from the method's first run.

    python scripts/analysis/fig_scenario_grid.py     # -> results/figures/fig_scenario_grid.{pdf,png}

Four layouts of `method_r1` that stage exactly one situation each, with no patrolling pedestrians,
late enough in the stream that the tree answers every tick: a wall across the planned route
(episode 174, seed 273: replan at the first tick, goal), a gate (115, seed 214: one wait held until
the pedestrian has left, goal), a sealed passage (126, seed 225: the replan finds no route, a human is
called) and a breakdown (196, seed 295: a wait, then a human called once the blocker has stood 30 s).
Each panel draws the 3x3 blocks, the staged obstacle the planner was not told about, the planned
route (dashed), the route after a replan (dotted), the driven path shaded by time, pedestrian paths
(thin) with their positions at the first recovery, and where each recovery was ordered.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(os.environ.get("RUNS", os.path.join(REPO, "results", "runs")), "method_r1")
PDF = os.path.join(REPO, "results", "figures", "fig_scenario_grid.pdf")
PNG = os.path.join(REPO, "results", "figures", "fig_scenario_grid.png")

PANELS = [  # episode file, title
    (174, "(a) wall across the route: replan"),
    (115, "(b) gate: wait until the pedestrian has passed"),
    (126, "(c) sealed passage: no route, call a human"),
    (196, "(d) breakdown: wait, then call a human"),
]
BLOCK = "#3a3a38"        # the blocks the planner knows
STAGED = "#eb6834"       # a staged obstacle the planned route runs through: the planner was not told about it
KNOWN = "#9c9b97"        # a staged obstacle the planner knows (the walls that make a gate or seal a detour)
PED = "#e34948"
ROUTE = "#52514e"
PATH = LinearSegmentedColormap.from_list("path", ["#b7d3f2", "#2a78d6", "#0b3d7a"])   # one hue, light -> dark
MARK = {  # recovery -> (marker, colour, size, label)
    "REPLAN_ROUTE": ("*", "#eda100", 15, "REPLAN_ROUTE ordered"),
    "WAIT": ("D", "#4a3aa7", 8, "WAIT ordered"),
    "REQUEST_HUMAN": ("X", "#0b0b0b", 10, "REQUEST_HUMAN ordered"),
    "REORIENT": ("^", "#1baf7a", 9, "REORIENT ordered"),
}


def is_block(poly: np.ndarray) -> bool:
    """The 3x3 blocks are 4 m squares; anything else is a staged obstacle."""
    w, h = poly[:, 0].max() - poly[:, 0].min(), poly[:, 1].max() - poly[:, 1].min()
    return abs(w - 4.0) < 0.05 and abs(h - 4.0) < 0.05


def crosses(poly: np.ndarray, route: np.ndarray) -> bool:
    """Whether the planned route passes through the obstacle (its bounding box, sampled every 5 cm)."""
    x0, x1, y0, y1 = poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()
    for a, b in zip(route[:-1], route[1:]):
        n = max(2, int(np.hypot(*(b - a)) / 0.05))
        for t in np.linspace(0, 1, n):
            x, y = a + t * (b - a)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
    return False


def draw(ax, rec: dict, title: str):
    m, frames = rec["meta"], rec["frames"]
    b = np.array(m["boundary"])
    ax.add_patch(Polygon(b, closed=True, fill=False, edgecolor="#8a8987", linewidth=1.0))
    r0 = np.array(m["initial_route"])
    for o in m["obstacles"]:
        p = np.array(o)
        if is_block(p):
            ax.add_patch(Polygon(p, closed=True, facecolor=BLOCK, edgecolor="none", zorder=2))
        elif crosses(p, r0):
            ax.add_patch(Polygon(p, closed=True, facecolor=STAGED, edgecolor="none", zorder=2,
                                 label="staged obstacle, unknown to the planner"))
        else:
            ax.add_patch(Polygon(p, closed=True, facecolor=KNOWN, edgecolor="none", zorder=2,
                                 label="staged walls the planner knows"))
    ax.plot(r0[:, 0], r0[:, 1], "--", color=ROUTE, linewidth=1.1, zorder=3, label="planned route")
    first = [f["route"] for f in frames if f.get("route")]
    later = [r for r in first if r != m["initial_route"]]
    if later:
        r1 = np.array(later[0])
        ax.plot(r1[:, 0], r1[:, 1], ":", color=ROUTE, linewidth=1.4, zorder=3, label="route after the replan")
    n_ped = max((len(f["dyn"]) for f in frames), default=0)
    for k in range(n_ped):
        pts = np.array([f["dyn"][k] for f in frames if len(f["dyn"]) > k])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=PED, linewidth=0.7, alpha=0.45, zorder=2,
                label="pedestrian paths" if k == 0 else None)
    xy = np.array([[f["x"], f["y"]] for f in frames])
    t = np.arange(len(xy)) / max(1, len(xy) - 1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=t, cmap=PATH, s=5, zorder=4, linewidths=0, vmin=0, vmax=1)
    recs = [r for r in rec.get("recoveries", []) if not str(r.get("detail", "")).startswith("follow-up")]
    if recs:
        f0 = frames[min(int(recs[0]["step"]), len(frames) - 1)]
        for d in f0["dyn"]:
            ax.add_patch(plt.Circle(d, m.get("dynamic_obstacle_radius", 0.8), facecolor=PED, alpha=0.55,
                                    edgecolor="none", zorder=5))
    seen, placed = set(), []
    for r in recs:
        if r["method"] not in MARK:
            continue
        mk, col, size, lab = MARK[r["method"]]
        f = frames[min(int(r["step"]), len(frames) - 1)]
        x, y = f["x"], f["y"]
        while any(np.hypot(x - px, y - py) < 0.7 for px, py in placed):   # do not hide an earlier marker
            x, y = x + 0.75, y + 0.75
        placed.append((x, y))
        ax.plot(x, y, marker=mk, markersize=size, color=col, markeredgecolor="white",
                markeredgewidth=0.8, zorder=7, linestyle="none", label=None if r["method"] in seen else lab)
        seen.add(r["method"])
    ax.plot(*m["start"], marker="o", color="#2a78d6", markersize=7, markeredgecolor="white", markeredgewidth=0.8,
            zorder=6, linestyle="none", label="start")
    ax.plot(*m["goal"], marker="P", color="#1baf7a", markersize=9, markeredgecolor="white", markeredgewidth=0.8,
            zorder=6, linestyle="none", label="goal")
    s = rec["summary"]
    ending = {"goal": "goal", "human_requested": "human called"}.get(s["termination"], s["termination"])
    short = {"REPLAN_ROUTE": "replan", "WAIT": "wait", "REQUEST_HUMAN": "human", "REORIENT": "reorient"}
    steps = ", ".join(f"{short[r['method']]} {r['step']}" for r in recs if r["method"] in MARK)
    ax.set_xlabel(f"seed {s['seed']}, steps: {steps}; {ending} at {s['steps']}", fontsize=6.5, color="#52514e", labelpad=2)
    ax.set_aspect("equal")
    ax.set_xlim(b[:, 0].min() - 0.4, b[:, 0].max() + 0.4)
    ax.set_ylim(b[:, 1].min() - 0.4, b[:, 1].max() + 0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8.5, loc="left")
    return sc


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 7.0))
    handles, labels = {}, []
    for ax, (ep, title) in zip(axes.flat, PANELS):
        rec = json.load(open(os.path.join(DATA, f"episode_{ep:03d}.json")))
        sc = draw(ax, rec, title)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h; labels.append(lab)
    fig.legend([handles[k] for k in labels], labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.46, -0.005), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 0.93, 1), h_pad=1.0, w_pad=0.6)
    cax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("time (share of the episode)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    os.makedirs(os.path.dirname(PDF), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(PNG)), exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=170, bbox_inches="tight")
    print("wrote", PDF, "and", PNG)


if __name__ == "__main__":
    main()
