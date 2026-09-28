"""Fig. 3: what changes over the 200-layout learning stream, per arm.

    python scripts/analysis/fig_heldout_curve.py     # -> results/figures/fig_heldout_curve.{pdf,png}

The method (experience with replay), distillation alone and experience without replay, three runs
each. Two panels over the learning episode:
  (a) the tree saved every 25 episodes (the checkpoint interval), run frozen (no model, nothing
      inserted or retired) on the 20 held-out layouts, with the scripted rules on the same layouts
      dotted;
  (b) teacher calls per episode, a centred 3-episode moving average on a logarithmic episode axis.
Each run is a thin line and the mean over runs a thick one. Only complete runs are drawn.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("RUNS", os.path.join(REPO, "results", "runs"))
FIGURES = os.path.join(REPO, "results", "figures")
PDF = os.path.join(FIGURES, "fig_heldout_curve.pdf")
PNG = os.path.join(FIGURES, "fig_heldout_curve.png")
BLOCK, EPISODES, HELDOUT = 25, 200, 20
RULES_HELDOUT = "scripted_heldout"    # the scripted rules on the 20 held-out layouts
RULES = dict(color="#6b6b67", lw=1.0, ls=":", zorder=0)
REPS = ("_r1", "_r2", "_r3")

ARMS = [  # run name without the repeat marker, label, colour, line style, marker
    ("method{}", "method: experience with replay", "#2a78d6", "-", "s"),
    ("experience_noreplay{}", "experience without replay", "#1baf7a", "-.", "^"),
    ("distillation{}", "distillation alone", "#eb6834", "--", "o"),
]


def ended_right(s: dict) -> bool:
    return bool(s["success"]) if s.get("success") is not None else s["termination"] == "goal"


def episodes(run: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, run, "episode_*.json")))]


CALLS_WINDOW = 3   # episodes in the centred moving average of panel (b), per run


def calls_per_episode(eps: list[dict], window: int = CALLS_WINDOW) -> tuple[list[int], list[float]]:
    """Teacher calls in each episode, smoothed with a centred moving average of `window` episodes
    (shorter at the ends), so the panel shares its scale with fig_permap's calls per episode."""
    raw = [sum(1 for d in e["decisions"] if d["source"] == "llm") for e in eps]
    half = window // 2
    ys = [sum(raw[max(0, i - half):i + half + 1]) / len(raw[max(0, i - half):i + half + 1]) for i in range(len(raw))]
    return list(range(1, len(raw) + 1)), ys


def heldout(run: str) -> tuple[list[int], list[int]]:
    xs, ys = [], []
    for c in range(BLOCK, EPISODES + 1, BLOCK):
        eps = episodes(f"{run}_heldout_ep{c:04d}")
        if len(eps) == HELDOUT:
            xs.append(c)
            ys.append(sum(ended_right(e["summary"]) for e in eps))
    return xs, ys


def mean_series(series: list[tuple[list, list]]) -> tuple[list, list]:
    xs = sorted({x for s in series for x in s[0]})
    ys = []
    for x in xs:
        vals = [y for s in series for x2, y in zip(*s) if x2 == x]
        ys.append(sum(vals) / len(vals))
    return xs, ys


def main(pdf: str = PDF, png: str = PNG) -> None:
    """Write the figure to PDF and PNG."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.6, "grid.color": "#e4e4e0", "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.05))
    ax_h, ax_c = axes
    for pattern, label, colour, ls, marker in ARMS:
        held, call_s, used = [], [], []
        for r in REPS:
            run = pattern.format(r)
            eps = episodes(run)
            if len(eps) != EPISODES:
                continue
            used.append(run)
            call_s.append(calls_per_episode(eps))
            hx, hy = heldout(run)
            if hx:
                held.append((hx, hy))
        if not used:
            continue
        print(f"{label}: {len(used)} runs ({', '.join(used)}), held-out checkpoints {[len(h[0]) for h in held]}")
        style = dict(color=colour, ls=ls, marker=marker, mfc="white", mew=1.1)
        for ax, series in ((ax_h, held), (ax_c, call_s)):
            if not series:
                continue
            many = len(series) > 1
            st = {**style, "marker": ""} if ax is ax_c else style   # per-episode lines carry no markers
            for xs, ys in series:
                ax.plot(xs, ys, lw=0.7 if many else 1.4, ms=2.5 if many else 4.2, alpha=0.35 if many else 1, **st)
            if many:
                ax.plot(*mean_series(series), lw=1.8, ms=4.5, **st)
        ax_h.plot([], [], lw=1.4, ms=4.2, label=label, **style)
    ax_h.plot([], [], label="scripted rules", **RULES)

    for ax in axes:
        ax.set_xlim(0, 205)
        ax.set_xticks([25, 75, 125, 175])
        ax.set_xlabel("learning episode")
        ax.grid(True, axis="y")
    ax_h.axhline(sum(ended_right(e["summary"]) for e in episodes(RULES_HELDOUT)), **RULES)
    ax_h.set_ylim(15.7, 20.3)   # every checkpoint ends 16-19 right
    ax_h.set_yticks([16, 17, 18, 19, 20])
    ax_h.set_title("(a) frozen tree, 20 held-out layouts", loc="left", fontsize=8)
    ax_h.set_ylabel("layouts ending right")
    # linear calls on a logarithmic episode axis: the first episodes, where the calls are, get the
    # room, and the tail to episode 200 stays on the panel
    ax_c.set_xscale("log")
    ax_c.set_xlim(1, 200)
    ax_c.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax_c.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax_c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_c.set_ylim(0, 15)
    ax_c.set_yticks([0, 5, 10, 15])
    ax_c.set_title("(b) teacher calls per episode", loc="left", fontsize=8)
    handles, labels = ax_h.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
    os.makedirs(os.path.dirname(os.path.abspath(pdf)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print("wrote", os.path.normpath(pdf), "and", os.path.normpath(png))


if __name__ == "__main__":
    tyro.cli(main)
