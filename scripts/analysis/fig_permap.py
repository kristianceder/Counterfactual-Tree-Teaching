"""Fig. 4: self-improvement on a repeated layout (the per-map runs).

    python scripts/analysis/fig_permap.py     # -> results/figures/fig_permap.{pdf,png}, results/tables/permap.md

`permap_{seed}_r{rep}` is one layout six times in a row, the tree grown from empty in the method's
configuration; `permap_{seed}_r{rep}_frozen` is that tree frozen, no model, for one more episode.
Four panels over the episode on the same layout:
  (a) teacher calls per episode;
  (b) share of ticks answered by a rule, with the frozen episode at F (its misses go unanswered);
  (c) decision age, control cycles from report to the verdict acting, over every tick;
  (d) runs ending right, out of all layouts x repeats, with the frozen episode at F.
Thin lines are layouts (mean over their repeats), the thick line the mean over layouts. Only complete
runs count (six learning episodes; one frozen). The table gives, per layout and episode, calls, tree
share, age, ending and steps, and the recoveries of the frozen episode.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("RUNS", os.path.join(REPO, "results", "runs"))
FIGURES = os.path.join(REPO, "results", "figures")
TABLES = os.path.join(REPO, "results", "tables")
EPISODES = 6
COLOUR = "#2a78d6"       # the method's hue in fig_heldout_curve
FROZEN_X = EPISODES + 1  # where the frozen episode is drawn, labelled F


def ended_right(s: dict) -> bool:
    return bool(s["success"]) if s.get("success") is not None else s["termination"] == "goal"


def episodes(run: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, run, "episode_*.json")))]


def measure(e: dict) -> dict:
    dec = e["decisions"]
    src = collections.Counter(d["source"] for d in dec)
    recs = [r["method"] for r in e["recoveries"] if not str(r.get("detail", "")).startswith("follow-up")]
    return dict(calls=src["llm"], ticks=len(dec), share=src["tree"] / len(dec) if dec else float("nan"),
                age=st.mean(d["acted_step"] - d["report_step"] for d in dec) if dec else float("nan"),
                right=ended_right(e["summary"]), steps=e["summary"]["steps"], end=e["summary"]["termination"],
                unanswered=src["unanswered"], recoveries=recs)


def load() -> dict[int, dict[int, dict]]:
    """seed -> repeat -> {"learn": [6 measures], "frozen": measure}, complete runs only."""
    runs: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(DATA, "permap_*"))):
        m = re.match(r"permap_(\d+)_r(\d+)(_frozen)?$", os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        seed, rep, frozen = int(m.group(1)), int(m.group(2)), bool(m.group(3))
        eps = episodes(os.path.basename(d))
        if frozen and len(eps) == 1:
            runs[seed].setdefault(rep, {})["frozen"] = measure(eps[0])
        elif not frozen and len(eps) == EPISODES:
            runs[seed].setdefault(rep, {})["learn"] = [measure(e) for e in eps]
    return {s: {r: v for r, v in reps.items() if "learn" in v} for s, reps in runs.items()}


def kinds(seed: int) -> str:
    """The layout's staged situations, from its first recording."""
    for d in sorted(glob.glob(os.path.join(DATA, f"permap_{seed}_r*"))):
        f = sorted(glob.glob(os.path.join(d, "episode_*.json")))
        layout = json.load(open(f[0]))["meta"].get("layout") if f else None
        if layout:
            return " + ".join(x["kind"] for x in layout["situations"]) + (" (human)" if layout["needs_human"] else "")
    return "?"


def table(runs: dict) -> str:
    lines = ["# Per-map self-improvement, GPT-6 Luna teacher", "",
             "Each cell: teacher calls / tree share / decision age (cycles) / ending right of the repeats / steps, "
             "mean over the layout's complete repeats. F: the tree after episode 6, frozen, no model.", "",
             "| layout | situations | repeats | " + " | ".join(f"ep {i}" for i in range(1, EPISODES + 1)) + " | F |",
             "|---|---|---|" + "---|" * (EPISODES + 1)]
    for seed, reps in sorted(runs.items()):
        cells = []
        for i in range(EPISODES):
            ms = [r["learn"][i] for r in reps.values()]
            cells.append(f"{st.mean(m['calls'] for m in ms):.1f} / {100 * st.mean(m['share'] for m in ms):.0f}% / "
                         f"{st.mean(m['age'] for m in ms):.1f} / {sum(m['right'] for m in ms)}/{len(ms)} / "
                         f"{st.mean(m['steps'] for m in ms):.0f}")
        fr = [r["frozen"] for r in reps.values() if "frozen" in r]
        cells.append(f"{100 * st.mean(m['share'] for m in fr):.0f}% / {sum(m['right'] for m in fr)}/{len(fr)} / "
                     f"{st.mean(m['steps'] for m in fr):.0f}; " + "; ".join(",".join(m["recoveries"]) or "-" for m in fr)
                     if fr else "-")
        lines.append(f"| {seed} | {kinds(seed)} | {len(reps)} | " + " | ".join(cells) + " |")
    lines += ["", "Over layouts (mean of the layout means; runs ending right out of all runs):", "",
              "| episode | calls | tree share | age | right | steps |", "|---|---|---|---|---|---|"]
    for i in range(EPISODES):
        per = [[r["learn"][i] for r in reps.values()] for reps in runs.values()]
        mean = lambda k: st.mean(st.mean(m[k] for m in ms) for ms in per)
        lines.append(f"| {i + 1} | {mean('calls'):.2f} | {100 * mean('share'):.0f}% | {mean('age'):.2f} | "
                     f"{sum(m['right'] for ms in per for m in ms)}/{sum(len(ms) for ms in per)} | {mean('steps'):.0f} |")
    fr = [r["frozen"] for reps in runs.values() for r in reps.values() if "frozen" in r]
    if fr:
        lines.append(f"| F | 0 | {100 * st.mean(m['share'] for m in fr):.0f}% | 1.00 | "
                     f"{sum(m['right'] for m in fr)}/{len(fr)} | {st.mean(m['steps'] for m in fr):.0f} |")
    total = sum(m["calls"] for reps in runs.values() for r in reps.values() for m in r["learn"])
    lines += ["", f"Teacher calls in all: {total} over {sum(len(r) for r in runs.values())} runs."]
    return "\n".join(lines) + "\n"


def main(pdf: str = os.path.join(FIGURES, "fig_permap.pdf"), png: str = os.path.join(FIGURES, "fig_permap.png")) -> None:
    """Write the figure to PDF and PNG, and the table."""
    runs = load()
    if not runs:
        raise SystemExit(f"no complete per-map runs under {DATA}")
    md = table(runs)
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "permap.md"), "w").write(md)
    print(md)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a86",
                         "axes.linewidth": 0.6, "grid.color": "#e4e4e0", "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95))
    its = list(range(1, EPISODES + 1))
    n_runs = sum(len(r) for r in runs.values())
    for ax, key, title, fmt in ((axes[0], "calls", "(a) teacher calls per episode", "{:.1f}"),
                                (axes[1], "share", "(b) ticks answered by the tree", "{:.0%}"),
                                (axes[2], "age", "(c) decision age (cycles)", "{:.1f}")):
        per_map = [[st.mean(r["learn"][i][key] for r in reps.values()) for i in range(EPISODES)] for reps in runs.values()]
        for ys in per_map:
            ax.plot(its, ys, color=COLOUR, alpha=0.3, lw=0.8)
        mean = [st.mean(ys[i] for ys in per_map) for i in range(EPISODES)]
        ax.plot(its, mean, color=COLOUR, lw=1.8, marker="s", ms=3.8, mfc="white", mew=1.1)
        ax.text(its[0] + 0.15, mean[0], fmt.format(mean[0]), va="bottom", ha="left", fontsize=6.5, color="#52514e")
        below = key == "share"   # above the line it would sit on the frozen marker at F
        ax.text(its[-1] + (0 if below else 0.15), mean[-1] - (0.05 if below else 0), fmt.format(mean[-1]),
                va="top" if below else "bottom", ha="center" if below else "left", fontsize=6.5, color="#52514e")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_ylim(0, None)
    # (b): the frozen tree's own coverage, misses unanswered, at F.
    fr = [[r["frozen"]["share"] for r in reps.values() if "frozen" in r] for reps in runs.values()]
    fr = [st.mean(v) for v in fr if v]
    if fr:
        axes[1].plot([FROZEN_X] * len(fr), fr, "o", color=COLOUR, alpha=0.3, ms=2.5, mew=0)
        axes[1].plot([FROZEN_X], [st.mean(fr)], "s", color=COLOUR, ms=3.8, mfc=COLOUR)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    # (d): runs ending right, all layouts and repeats together.
    right = [sum(r["learn"][i]["right"] for reps in runs.values() for r in reps.values()) for i in range(EPISODES)]
    axes[3].plot(its, right, color=COLOUR, lw=1.8, marker="s", ms=3.8, mfc="white", mew=1.1)
    fr_right = [r["frozen"]["right"] for reps in runs.values() for r in reps.values() if "frozen" in r]
    if fr_right:
        axes[3].plot([FROZEN_X], [sum(fr_right)], "s", color=COLOUR, ms=3.8, mfc=COLOUR)
    axes[3].axhline(n_runs, color="#8a8a86", lw=0.8, ls=":", zorder=0)
    axes[3].set_ylim(0, n_runs + 1)
    axes[3].set_title(f"(d) runs ending right, of {n_runs}", loc="left", fontsize=7.5)
    for ax in axes:
        with_f = ax in (axes[1], axes[3])
        ax.set_xticks(its + ([FROZEN_X] if with_f else []))
        ax.set_xticklabels([str(i) for i in its] + (["F"] if with_f else []))
        ax.set_xlim(0.5, FROZEN_X + 0.6)
        ax.set_xlabel("episode on the layout")
        ax.grid(True, axis="y")
    fig.tight_layout(w_pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(pdf)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print("wrote", os.path.normpath(pdf), "and", os.path.normpath(png))


if __name__ == "__main__":
    tyro.cli(main)
