"""Replay a recorded decision-tree / LLM recovery run.

Draws what `failure_monitor.recorder` captured, never the live simulation, so watching a run
cannot change it. One figure, four panels:

    +--------------------------+----------------------------------+
    |                          | decision panel: who decided the  |
    |   map: robot, trail,     | last tick (tree / LLM / breaker),|
    |   obstacles, route,      | failure flagged, recovery, rule, |
    |   tracked reference,     | hold in progress, judge verdicts |
    |   holds and recoveries   +----------------------------------+
    |                          | episode timeline: every tick     |
    |                          | coloured by who decided, failures|
    |                          | flagged, holds, tree edits       |
    |                          +----------------------------------+
    |                          | learning curve: tree vs LLM      |
    |                          | decisions per episode so far     |
    +--------------------------+----------------------------------+

Map styling: red obstacles with dashed inflated outlines, a black boundary, a red robot with a
heading marker.

Three ways to use it (see `scripts/visualize_run.py` for the CLI):

    ReplayFigure(run, episodes).interactive(episode)   window with play/pause and a step slider
    save_video(run, episodes, episode, path)           MP4 (ffmpeg) or GIF per episode
    summary_figure(run, episodes, path)                static figure across all episodes
"""
from __future__ import annotations

import math
import shutil
import textwrap
from bisect import bisect_right
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button, Slider

from failure_monitor.recorder import (EpisodeRecording, SOURCE_BREAKER, SOURCE_LLM, SOURCE_TREE,
                                      SOURCE_UNANSWERED, load_run)

SOURCE_COLOURS = {SOURCE_TREE: "#2ca02c", SOURCE_LLM: "#ff7f0e", SOURCE_BREAKER: "#9467bd",
                  SOURCE_UNANSWERED: "#7f7f7f"}
FAILURE_COLOURS = {"STUCK": "#d62728", "OSCILLATION": "#bcbd22", "REVERSE_TRACKING": "#8c564b",
                   "COLLISION": "#000000", "TIMEOUT": "#7f7f7f", "SOLVER_DEADLINE_MISS": "#17becf"}
RECOVERY_COLOURS = {"REPLAN_ROUTE": "#1f77b4", "WAIT": "#bcbd22", "RESUME_ROUTE": "#17becf",
                    "REORIENT": "#8c564b", "REQUEST_HUMAN": "#e377c2", "CONTINUE": "#7f7f7f"}
JUDGE_STYLE = {  # kind -> (marker, colour, label)
    "good": ("^", "#2ca02c", "judged good"), "bad": ("v", "#d62728", "judged bad"),
    "retired": ("X", "#000000", "rule retired"),
    "insert": ("+", "#1f77b4", "rule learned"), "refine": ("+", "#1f77b4", "rule learned"),
    "replace": ("+", "#1f77b4", "rule learned"), "contradict": ("+", "#9467bd", "contradicted"),
}
FLASH_STEPS = 15  # how long a verdict or new rule stays highlighted in the decision panel
COUNTED_SOURCES = (SOURCE_TREE, SOURCE_LLM, SOURCE_BREAKER, SOURCE_UNANSWERED)


def is_follow_up(recovery: dict) -> bool:
    """A recovery the loop ran on its own as the second half of an earlier one (none do in this repository)."""
    return recovery.get("detail", "").startswith("follow-up")


def action_categories(episodes) -> list[str]:
    """Recovery methods ordered anywhere in the run, CONTINUE excluded, in a fixed display order."""
    seen = {d["method"] for e in episodes for d in e.decisions if d["method"] != "CONTINUE"}
    return [m for m in RECOVERY_COLOURS if m in seen] + sorted(seen - set(RECOVERY_COLOURS))


def failure_categories(episodes) -> list[str]:
    """Failure modes flagged anywhere in the run, in a fixed display order."""
    seen = {d["failure_mode"] for e in episodes for d in e.decisions if d.get("failure_mode")}
    return [m for m in FAILURE_COLOURS if m in seen] + sorted(seen - set(FAILURE_COLOURS))


def _failure_colour(mode: str | None) -> str:
    return FAILURE_COLOURS.get(mode or "", "#d62728")


def source_label(source: str, teacher: str) -> str:
    """Display name for who decided. The teacher is only called "LLM" when it is one."""
    if source == SOURCE_LLM:
        return "LLM" if teacher == "llm" else f"teacher ({teacher})"
    return {SOURCE_TREE: "decision tree", SOURCE_BREAKER: "loop breaker",
            SOURCE_UNANSWERED: "tree miss (no teacher)"}.get(source, source)


@dataclass
class _Episode:
    """An episode recording with the lookups a frame needs precomputed."""
    rec: EpisodeRecording
    steps: list[int]
    routes: list[list[list[float]]]
    decision_steps: list[int]
    note_steps: list[int]
    judge_steps: list[int]

    @classmethod
    def build(cls, rec: EpisodeRecording) -> "_Episode":
        routes, current = [], rec.meta.get("initial_route", [])
        for frame in rec.frames:
            if frame.get("route"):
                current = frame["route"]
            routes.append(current)
        rec.decisions.sort(key=lambda d: d["acted_step"])
        rec.notes.sort(key=lambda n: n["step"])
        rec.judgements.sort(key=lambda j: j["step"])
        return cls(rec=rec, steps=[f["step"] for f in rec.frames], routes=routes,
                   decision_steps=[d["acted_step"] for d in rec.decisions],
                   note_steps=[n["step"] for n in rec.notes],
                   judge_steps=[j["step"] for j in rec.judgements])

    def decisions_until(self, step: int) -> list[dict]:
        return self.rec.decisions[:bisect_right(self.decision_steps, step)]

    def judgements_until(self, step: int) -> list[dict]:
        return self.rec.judgements[:bisect_right(self.judge_steps, step)]


class ReplayFigure:
    """The four-panel replay figure for one run. Switch episodes with `show_episode`, move in
    time with `draw(frame_index)`."""

    def __init__(self, run: dict, episodes: list[EpisodeRecording], figsize=(17, 11)):
        self.run = run
        self.episodes = [_Episode.build(e) for e in episodes]
        self.teacher = run.get("meta", {}).get("teacher", "llm")
        self.fig = plt.figure(figsize=figsize)
        grid = GridSpec(4, 2, figure=self.fig, width_ratios=[1.25, 1], height_ratios=[1.0, 0.8, 1, 1],
                        left=0.04, right=0.98, top=0.96, bottom=0.08, wspace=0.12, hspace=0.85)
        self.ax_map = self.fig.add_subplot(grid[:, 0])
        self.ax_text = self.fig.add_subplot(grid[0, 1])
        counters = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, 1], wspace=0.75)
        self.ax_actions = self.fig.add_subplot(counters[0, 0])
        self.ax_failures = self.fig.add_subplot(counters[0, 1])
        self.ax_timeline = self.fig.add_subplot(grid[2, 1])
        self.ax_curve = self.fig.add_subplot(grid[3, 1])
        self.ax_text.axis("off")
        self.index = 0
        self.frame = 0
        self._build_curve()
        self._build_counters()

    # -- episode setup -----------------------------------------------------------------------

    @property
    def ep(self) -> _Episode:
        return self.episodes[self.index]

    def show_episode(self, index: int) -> None:
        self.index = max(0, min(index, len(self.episodes) - 1))
        self.frame = 0
        self._build_map()
        self._build_timeline()
        self._build_text()

    def _build_map(self) -> None:
        ax, meta = self.ax_map, self.ep.rec.meta
        ax.cla()
        boundary = meta.get("boundary", [])
        if boundary:
            ax.add_patch(Polygon(boundary, closed=True, fill=False, edgecolor="k", linewidth=1.5))
            xs, ys = zip(*boundary)
            pad = 0.5
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        for obstacle in meta.get("inflated_obstacles", []):
            ax.add_patch(Polygon(obstacle, closed=True, fill=False, edgecolor="r", linestyle="--",
                                 linewidth=0.8, alpha=0.6))
        for obstacle in meta.get("obstacles", []):
            ax.add_patch(Polygon(obstacle, closed=True, facecolor="#f4a3a3", edgecolor="r", linewidth=1.2))
        route0 = meta.get("initial_route", [])
        if route0:
            ax.plot(*zip(*route0), ":", color="#999999", linewidth=1, label="initial route")
        goal = meta.get("goal")
        if goal:
            ax.plot(goal[0], goal[1], "*", color="gold", markeredgecolor="k", markersize=18, label="goal", zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        self.route_line, = ax.plot([], [], "--", color="#1f77b4", linewidth=1.5, label="intended route")
        self.trail_line, = ax.plot([], [], "-", color="#1f3b73", linewidth=1.2, alpha=0.7, label="driven path")
        self.ref_line, = ax.plot([], [], "x-", color="#2ca02c", markersize=3, linewidth=1, label="tracked reference")
        radius = meta.get("robot_radius", 0.5)
        self.hold_ring = Circle((0, 0), radius * 1.9, fill=False, linewidth=3, visible=False, zorder=9)
        self.robot = Circle((0, 0), radius, color="r", zorder=10)
        self.heading, = ax.plot([], [], "-", color="lightsalmon", linewidth=3, zorder=11)
        ax.add_patch(self.hold_ring)
        ax.add_patch(self.robot)
        n_dyn = max((len(f["dyn"]) for f in self.ep.rec.frames), default=0)
        dyn_r = meta.get("dynamic_obstacle_radius", 0.8)
        self.dyn = [Circle((0, 0), dyn_r, facecolor="#c9b3e6", edgecolor="#6a3d9a", alpha=0.85,
                           visible=False, zorder=8) for _ in range(n_dyn)]
        for patch in self.dyn:
            ax.add_patch(patch)
        ax.plot([], [], "o", color="#c9b3e6", markeredgecolor="#6a3d9a", label="dynamic obstacle")
        # Recovery markers, revealed as they happen.
        self.recovery_marks = []
        for rec in self.ep.rec.recoveries:
            k = self._frame_at(rec["step"])
            f = self.ep.rec.frames[k]
            colour = RECOVERY_COLOURS.get(rec["method"], "k")
            mark, = ax.plot(f["x"], f["y"], "D", color=colour, markeredgecolor="k", markersize=8,
                            visible=False, zorder=7)
            label = ax.annotate(rec["method"].replace("_", " ").lower(), (f["x"], f["y"]),
                                xytext=(6, 6), textcoords="offset points", fontsize=8, color=colour,
                                visible=False, zorder=7)
            self.recovery_marks.append((rec["step"], mark, label))
        self.banner = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center", va="bottom",
                              fontsize=13, fontweight="bold")
        self.hold_text = ax.text(0.01, 0.01, "", transform=ax.transAxes, ha="left", va="bottom",
                                 fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    def _build_timeline(self) -> None:
        ax, ep = self.ax_timeline, self.ep
        ax.cla()
        max_step = ep.rec.meta.get("max_steps", ep.steps[-1] if ep.steps else 1)
        rows = {"decided by": 3, "failure flagged": 2, "recovery / hold": 1, "tree": 0}
        ax.set_yticks(list(rows.values()), list(rows.keys()), fontsize=9)
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(0, max_step)
        ax.set_xlabel("control step", fontsize=9)
        ax.set_title("This episode", fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)

        self.tl_artists = []  # (step, artist) revealed once the cursor passes step
        for d in ep.rec.decisions:
            s = d["acted_step"]
            colour = SOURCE_COLOURS.get(d["source"], "k")
            # Hollow: an order dropped unexecuted because its evidence had vanished when it landed.
            a = (ax.scatter([s], [3], s=40, facecolors="none", edgecolors=colour, linewidths=1.5, zorder=3)
                 if d.get("expired") else ax.scatter([s], [3], s=28, color=colour, zorder=3))
            self.tl_artists.append((s, a))
            if d.get("failure_mode"):
                b = ax.scatter([d["report_step"]], [2], s=40, marker="s",
                               color=_failure_colour(d["failure_mode"]), zorder=3)
                self.tl_artists.append((s, b))
        # Holds: contiguous runs of frames with a hold label.
        start, label = None, None
        for f in ep.rec.frames + [{"step": None, "hold": None}]:
            method = f["hold"].split(":")[0] if f.get("hold") else None
            if method != label:
                if label is not None and start is not None:
                    # Row 1 of a -0.6..3.6 axis, +-0.25, in axes fractions.
                    span = ax.axvspan(start, last, ymin=(0.75 + 0.6) / 4.2, ymax=(1.25 + 0.6) / 4.2,
                                      color=RECOVERY_COLOURS.get(label, "k"), alpha=0.5)
                    self.tl_artists.append((start, span))
                start, label = f["step"], method
            if f["step"] is not None:
                last = f["step"]
        for r in ep.rec.recoveries:
            m = ax.scatter([r["step"]], [1], s=46, marker="D", color=RECOVERY_COLOURS.get(r["method"], "k"),
                           edgecolors="k", zorder=4)
            self.tl_artists.append((r["step"], m))
        for j in ep.rec.judgements:
            marker, colour, _ = JUDGE_STYLE.get(j["kind"], (".", "k", j["kind"]))
            if j["kind"] == "reinforce":
                continue
            m = ax.scatter([j["step"]], [0], s=50, marker=marker, color=colour, zorder=4)
            self.tl_artists.append((j["step"], m))
        for _, artist in self.tl_artists:
            artist.set_visible(False)
        self.cursor = ax.axvline(0, color="k", linewidth=1)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=source_label(s, self.teacher))
                   for s, c in SOURCE_COLOURS.items() if any(d["source"] == s for d in ep.rec.decisions)]
        handles += [plt.Line2D([], [], marker=m, linestyle="", color=c, label=l)
                    for m, c, l in {v[2]: v for v in JUDGE_STYLE.values()}.values()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8,
                  frameon=False, handletextpad=0.2, columnspacing=1.0)

    def _build_curve(self) -> None:
        ax = self.ax_curve
        ax.cla()
        n = len(self.episodes)
        xs = list(range(1, n + 1))
        self.bar_tree = ax.bar(xs, [0] * n, color=SOURCE_COLOURS[SOURCE_TREE], label="decision tree")
        self.bar_llm = ax.bar(xs, [0] * n, color=SOURCE_COLOURS[SOURCE_LLM],
                              label=source_label(SOURCE_LLM, self.teacher))
        self.bar_breaker = ax.bar(xs, [0] * n, color=SOURCE_COLOURS[SOURCE_BREAKER], label="loop breaker")
        ticks_max = max((len(e.rec.decisions) for e in self.episodes), default=1)
        ax.set_ylim(0, ticks_max * 1.3 + 1)
        ax.set_xlim(0.4, n + 0.6)
        ax.set_xticks(xs)
        ax.set_xlabel("episode", fontsize=9)
        ax.set_ylabel("decisions", fontsize=9)
        self.share_text = [ax.text(x, 0, "", ha="center", va="bottom", fontsize=8) for x in xs]
        self.goal_text = [ax.text(x, 0, "", ha="center", va="bottom", fontsize=12,
                                  fontweight="bold") for x in xs]
        # No legend: the timeline panel above uses the same colours and already names them.
        ax.set_title("Who decides, per episode (colours as in the timeline)", fontsize=10, loc="left")

    def _build_counters(self) -> None:
        """Two stacked horizontal bar charts: recoveries ordered and failure modes flagged, each
        bar split by who decided. Categories and scale are fixed for the whole run, so episodes
        can be compared by eye as the replay moves between them."""
        episodes = [e.rec for e in self.episodes]
        self.counter_panels = []
        for ax, cats, key in ((self.ax_actions, action_categories(episodes), "method"),
                              (self.ax_failures, failure_categories(episodes), "failure_mode")):
            ax.cla()
            peak = max((sum(1 for d in e.decisions if d.get(key) in cats) and
                        max(sum(1 for d in e.decisions if d.get(key) == c) for c in cats)
                        for e in episodes), default=0) if cats else 0
            ax.set_xlim(0, max(1, peak) * 1.55)
            ys = list(range(len(cats)))
            ax.set_yticks(ys, [c.replace("_", " ").lower() for c in cats], fontsize=8)
            ax.set_ylim(-0.6, max(len(cats), 1) - 0.4)
            ax.invert_yaxis()
            ax.tick_params(axis="x", labelsize=7)
            bars = {src: ax.barh(ys, [0] * len(cats), color=SOURCE_COLOURS[src], height=0.65)
                    for src in COUNTED_SOURCES} if cats else {}
            labels = [ax.text(0, y, "", va="center", fontsize=8) for y in ys]
            if not cats:
                ax.text(0.5, 0.5, "none in this run", transform=ax.transAxes, ha="center", fontsize=9,
                        color="#7f7f7f")
                ax.set_yticks([])
            self.counter_panels.append((ax, cats, key, bars, labels))

    def _draw_counters(self, decisions: list[dict], step: int) -> None:
        earlier = [d for e in self.episodes[:self.index] for d in e.rec.decisions]
        continues = sum(1 for d in decisions if d["method"] == "CONTINUE")
        nominal = sum(1 for d in decisions if not d.get("failure_mode"))
        titles = (f"Actions executed (+CONTINUE {continues})",
                  f"Failure modes flagged (+nominal {nominal})")
        for (ax, cats, key, bars, labels), title in zip(self.counter_panels, titles):
            ax.set_title(title, fontsize=9, loc="left")
            for y, cat in enumerate(cats):
                left = 0
                for src in COUNTED_SOURCES:
                    # Dropped orders were never executed, so they are not counted as actions (their
                    # failure flag still is: the report did say it).
                    n = sum(1 for d in decisions if d.get(key) == cat and d["source"] == src
                            and not (key == "method" and d.get("expired")))
                    bars[src][y].set_x(left)
                    bars[src][y].set_width(n)
                    left += n
                run_total = left + sum(1 for d in earlier if d.get(key) == cat
                                       and not (key == "method" and d.get("expired")))
                labels[y].set_position((left + 0.15, y))
                labels[y].set_text(f"{left}  (run {run_total})" if self.index else f"{left}")

    def _build_text(self) -> None:
        self.ax_text.cla()
        self.ax_text.axis("off")
        kw = dict(transform=self.ax_text.transAxes, ha="left", va="top")
        self.t_header = self.ax_text.text(0, 1.0, "", fontsize=12, fontweight="bold", **kw)
        self.t_source = self.ax_text.text(0, 0.84, "", fontsize=13, fontweight="bold", color="white",
                                          bbox=dict(boxstyle="round", facecolor="grey"), **kw)
        self.t_body = self.ax_text.text(0, 0.68, "", fontsize=10, family="monospace", **kw)
        self.t_flash = self.ax_text.text(0, 0.14, "", fontsize=10, fontweight="bold", **kw)

    # -- per-frame drawing -------------------------------------------------------------------

    def _frame_at(self, step: int) -> int:
        return max(0, min(bisect_right(self.ep.steps, step) - 1, len(self.ep.steps) - 1))

    def draw(self, frame_index: int) -> None:
        ep = self.ep
        if not ep.rec.frames:
            return
        self.frame = k = max(0, min(frame_index, len(ep.rec.frames) - 1))
        f = ep.rec.frames[k]
        step = f["step"]
        meta = ep.rec.meta

        # map
        trail = ep.rec.frames[:k + 1]
        self.trail_line.set_data([p["x"] for p in trail], [p["y"] for p in trail])
        route = ep.routes[k]
        self.route_line.set_data([p[0] for p in route], [p[1] for p in route])
        self.ref_line.set_data([p[0] for p in f["ref"]], [p[1] for p in f["ref"]])
        self.robot.center = (f["x"], f["y"])
        r = meta.get("robot_radius", 0.5)
        self.heading.set_data([f["x"], f["x"] + 1.6 * r * math.cos(f["heading"])],
                              [f["y"], f["y"] + 1.6 * r * math.sin(f["heading"])])
        for patch, centre in zip(self.dyn, f["dyn"]):
            patch.center = tuple(centre)
            patch.set_visible(True)
        for patch in self.dyn[len(f["dyn"]):]:
            patch.set_visible(False)
        for s, mark, label in self.recovery_marks:
            mark.set_visible(s <= step)
            label.set_visible(s <= step)
        hold = f.get("hold")
        if hold:
            colour = RECOVERY_COLOURS.get(hold.split(":")[0], "k")
            self.hold_ring.center = (f["x"], f["y"])
            self.hold_ring.set_edgecolor(colour)
            self.hold_ring.set_visible(True)
            self.hold_text.set_text(f"recovery in progress: {hold}")
            self.hold_text.set_visible(True)
        else:
            self.hold_ring.set_visible(False)
            self.hold_text.set_visible(False)

        decisions = ep.decisions_until(step)
        last = decisions[-1] if decisions else None
        flagged = next((d for d in reversed(decisions)
                        if d.get("failure_mode") and step - d["acted_step"] <= FLASH_STEPS), None)
        if flagged:
            self.banner.set_text(f"{flagged['failure_mode']} flagged by "
                                 f"{source_label(flagged['source'], self.teacher)} -> {flagged['method']}"
                                 + (" (dropped: stale)" if flagged.get("expired") else ""))
            self.banner.set_color(_failure_colour(flagged["failure_mode"]))
        else:
            self.banner.set_text("no failure flagged" if last else "warming up (no assessment yet)")
            self.banner.set_color("#2ca02c" if last else "#7f7f7f")

        # decision panel
        n_ep, total = len(self.episodes), self.ep.rec.summary
        end = ""
        if k == len(ep.rec.frames) - 1 and total:
            end = f"  |  ended: {total.get('termination', '?')}"
        self.t_header.set_text(f"Episode {self.index + 1}/{n_ep}  |  step {step}/{meta.get('max_steps', '?')}"
                               f"  |  t = {step * meta.get('ts', 0.2):.1f} s{end}")
        if last:
            self.t_source.set_text(f" last decision: {source_label(last['source'], self.teacher)} ")
            self.t_source.get_bbox_patch().set_facecolor(SOURCE_COLOURS.get(last["source"], "grey"))
            lines = [f"failure flagged : {last['failure_mode'] or 'none (nominal)'}",
                     f"action          : {last['method']}" + ("  (breaker override)" if last.get("overridden") else "")
                     + (f"  DROPPED: {last['expired']}" if last.get("expired") else "")]
            if last["source"] == SOURCE_TREE and last.get("rule"):
                lines.append(f"rule            : {last['rule']}")
            elif last["source"] == SOURCE_LLM:
                lines.append(f"latency         : {last['latency_s']:.2f} s "
                             f"(report @{last['report_step']} -> acted @{last['acted_step']})")
            if last.get("rationale") and last["source"] != SOURCE_TREE:
                lines += textwrap.wrap("why             : " + last["rationale"], 64,
                                       subsequent_indent=" " * 18)[:3]
            counts = {s: sum(d["source"] == s for d in decisions) for s in SOURCE_COLOURS}
            lines.append("")
            lines.append(f"this episode    : tree {counts[SOURCE_TREE]}  |  "
                         f"{source_label(SOURCE_LLM, self.teacher)} {counts[SOURCE_LLM]}"
                         + (f"  |  breaker {counts[SOURCE_BREAKER]}" if counts[SOURCE_BREAKER] else ""))
            self.t_body.set_text("\n".join(lines))
        else:
            self.t_source.set_text(" no decision yet ")
            self.t_source.get_bbox_patch().set_facecolor("grey")
            self.t_body.set_text("")
        recent = [j for j in ep.judgements_until(step)
                  if step - j["step"] <= FLASH_STEPS and j["kind"] != "reinforce"]
        if recent:
            j = recent[-1]
            marker, colour, label = JUDGE_STYLE.get(j["kind"], (".", "k", j["kind"]))
            flash = f"{label.upper()}: {j['rule']} -> {j['action']}" + (f"  ({j['text']})" if j.get("text") else "")
            self.t_flash.set_text("\n".join(textwrap.wrap(flash, 78)[:2]))
            self.t_flash.set_color(colour)
        else:
            self.t_flash.set_text("")

        # timeline
        self.cursor.set_xdata([step, step])
        for s, artist in self.tl_artists:
            artist.set_visible(s <= step)

        self._draw_counters(decisions, step)

        # learning curve: finished episodes in full, the current one up to the cursor
        for i, e in enumerate(self.episodes):
            if i < self.index:
                shown = e.rec.decisions
            elif i == self.index:
                shown = decisions
            else:
                shown = []
            tree = sum(d["source"] == SOURCE_TREE for d in shown)
            llm = sum(d["source"] == SOURCE_LLM for d in shown)
            brk = sum(d["source"] == SOURCE_BREAKER for d in shown)
            self.bar_tree[i].set_height(tree)
            self.bar_llm[i].set_y(tree)
            self.bar_llm[i].set_height(llm)
            self.bar_breaker[i].set_y(tree + llm)
            self.bar_breaker[i].set_height(brk)
            total_n = tree + llm + brk
            self.share_text[i].set_position((i + 1, total_n + 0.3))
            short = "LLM" if self.teacher == "llm" else "teacher"
            self.share_text[i].set_text(f"{100 * llm / total_n:.0f}% {short}" if total_n else "")
            self.goal_text[i].set_position((i + 1, total_n + 0.3 + 0.08 * self.ax_curve.get_ylim()[1]))
            done = i < self.index or (i == self.index and k == len(ep.rec.frames) - 1)
            if done and e.rec.summary:
                ok = e.rec.summary.get("success")
                self.goal_text[i].set_text("✓" if ok else "✗")
                self.goal_text[i].set_color("#2ca02c" if ok else "#d62728")
            else:
                self.goal_text[i].set_text("")

    # -- interactive window ------------------------------------------------------------------

    def interactive(self, episode: int = 0, fps: float = 10.0) -> None:
        """Window with a step slider, play/pause, and keys: space play/pause, left/right step,
        n/p next/previous episode."""
        self.show_episode(episode)
        ax_slider = self.fig.add_axes([0.08, 0.02, 0.62, 0.025])
        ax_play = self.fig.add_axes([0.73, 0.015, 0.06, 0.035])
        ax_prev = self.fig.add_axes([0.80, 0.015, 0.08, 0.035])
        ax_next = self.fig.add_axes([0.89, 0.015, 0.08, 0.035])
        slider = Slider(ax_slider, "frame", 0, max(1, len(self.ep.rec.frames) - 1), valinit=0, valstep=1)
        play = Button(ax_play, "play")
        prev = Button(ax_prev, "prev episode")
        nxt = Button(ax_next, "next episode")
        state = {"playing": False}

        def refresh(value):
            self.draw(int(value))
            self.fig.canvas.draw_idle()

        def set_episode(index):
            self.show_episode(index)
            slider.valmax = max(1, len(self.ep.rec.frames) - 1)
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            slider.set_val(0)
            refresh(0)

        def toggle(_=None):
            state["playing"] = not state["playing"]
            play.label.set_text("pause" if state["playing"] else "play")

        def tick():
            if state["playing"]:
                if self.frame >= len(self.ep.rec.frames) - 1:
                    if self.index < len(self.episodes) - 1:
                        set_episode(self.index + 1)
                    else:
                        toggle()
                else:
                    slider.set_val(self.frame + 1)

        def key(event):
            if event.key == " ":
                toggle()
            elif event.key == "right":
                slider.set_val(min(self.frame + 1, slider.valmax))
            elif event.key == "left":
                slider.set_val(max(self.frame - 1, 0))
            elif event.key == "n":
                set_episode(self.index + 1)
            elif event.key == "p":
                set_episode(self.index - 1)

        slider.on_changed(refresh)
        play.on_clicked(toggle)
        prev.on_clicked(lambda _: set_episode(self.index - 1))
        nxt.on_clicked(lambda _: set_episode(self.index + 1))
        self.fig.canvas.mpl_connect("key_press_event", key)
        timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        timer.add_callback(tick)
        timer.start()
        refresh(0)
        self._widgets = (slider, play, prev, nxt, timer)  # keep references alive
        plt.show()


# -- video and summary ---------------------------------------------------------------------------

def _writer(path: str, fps: float):
    """ffmpeg for .mp4 when it can be found (system, or the `imageio-ffmpeg` package), else a GIF."""
    if path.endswith(".mp4"):
        if not animation.writers.is_available("ffmpeg") and not shutil.which("ffmpeg"):
            try:
                import imageio_ffmpeg  # type: ignore
                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                gif = path[:-4] + ".gif"
                print(f"  ffmpeg not found (install it, or `uv pip install imageio-ffmpeg`); writing {gif} instead")
                return animation.PillowWriter(fps=fps), gif
        return animation.FFMpegWriter(fps=fps, bitrate=2400), path
    return animation.PillowWriter(fps=fps), path


def save_video(run: dict, episodes: list[EpisodeRecording], episode: int, path: str,
               fps: float = 10.0, stride: int = 1, dpi: int = 90) -> str:
    """Render one episode to `path` (.mp4 or .gif). Earlier episodes appear, finished, in the
    learning-curve panel. Returns the path actually written."""
    figure = ReplayFigure(run, episodes)
    figure.show_episode(episode)
    frames = list(range(0, len(figure.ep.rec.frames), max(1, stride)))
    if frames and frames[-1] != len(figure.ep.rec.frames) - 1:
        frames.append(len(figure.ep.rec.frames) - 1)
    frames += [frames[-1]] * int(fps)  # hold the last frame for a second
    writer, path = _writer(path, fps)
    anim = animation.FuncAnimation(figure.fig, figure.draw, frames=frames, blit=False)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(figure.fig)
    return path


def summary_figure(run: dict, episodes: list[EpisodeRecording], path: str | None = None):
    """Static figure across all episodes: who decided, LLM share and decision age, episode outcome,
    and what happened to the tree. Saved to `path` if given; the figure is returned."""
    n = len(episodes)
    xs = list(range(1, n + 1))
    teacher = run.get("meta", {}).get("teacher", "llm")
    count = lambda e, s: sum(d["source"] == s for d in e.decisions)  # noqa: E731
    tree = [count(e, SOURCE_TREE) for e in episodes]
    llm = [count(e, SOURCE_LLM) for e in episodes]
    brk = [count(e, SOURCE_BREAKER) for e in episodes]
    share = [100 * l / (t + l + b) if (t + l + b) else 0 for t, l, b in zip(tree, llm, brk)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(f"Learning across {n} episode(s), teacher: {teacher if teacher != 'llm' else run['meta'].get('model') or 'LLM'}",
                 fontsize=13)

    ax = axes[0, 0]
    ax.bar(xs, tree, color=SOURCE_COLOURS[SOURCE_TREE], label="decision tree")
    ax.bar(xs, llm, bottom=tree, color=SOURCE_COLOURS[SOURCE_LLM], label=source_label(SOURCE_LLM, teacher))
    ax.bar(xs, brk, bottom=[t + l for t, l in zip(tree, llm)], color=SOURCE_COLOURS[SOURCE_BREAKER], label="loop breaker")
    ax.set_title("Who decides each assessment tick")
    ax.set_xlabel("episode")
    ax.set_ylabel("decisions")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(xs, share, "o-", color=SOURCE_COLOURS[SOURCE_LLM], label="share decided by teacher [%]")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of decisions")
    ax.set_xlabel("episode")
    ax2 = ax.twinx()
    age = [e.summary.get("mean_age_steps", 0) for e in episodes]
    ax2.plot(xs, age, "s--", color="#555555", label="mean decision age [control steps]")
    ax2.set_ylabel("control steps")
    ax2.set_ylim(0, max(age + [1]) * 1.3)
    ax.set_title("Dependence on the teacher, and decision latency")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

    ax = axes[1, 0]
    steps = [e.summary.get("steps", len(e.frames)) for e in episodes]
    colours = ["#2ca02c" if e.summary.get("success") else "#d62728" for e in episodes]
    ax.bar(xs, steps, color=colours)
    for x, e, s in zip(xs, episodes, steps):
        ax.text(x, s + 3, e.summary.get("termination", ""), ha="center", fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(0, max(steps + [1]) * 1.35)
    ax.set_title("Episode outcome (green = goal reached)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")

    ax = axes[1, 1]
    stat = lambda e, key: e.summary.get("tree", {}).get(key, 0)  # noqa: E731
    ax.plot(xs, [e.summary.get("active_rules", 0) for e in episodes], "o-", color="#1f77b4", label="active rules (end of episode)")
    width = 0.27
    ax.bar([x - width for x in xs], [stat(e, "judged_good") for e in episodes], width, color="#2ca02c", label="judged good")
    ax.bar(xs, [stat(e, "judged_bad") for e in episodes], width, color="#d62728", label="judged bad")
    ax.bar([x + width for x in xs], [stat(e, "retirements") for e in episodes], width, color="#333333", label="retired")
    ax.set_title("The tree: size and outcome verdicts")
    ax.set_xlabel("episode")
    ax.legend(fontsize=8)

    for ax, cats, key, colours, title in (
            (axes[2, 0], action_categories(episodes), "method", RECOVERY_COLOURS,
             "Recoveries executed per episode (CONTINUE not counted)"),
            (axes[2, 1], failure_categories(episodes), "failure_mode", FAILURE_COLOURS,
             "Failure modes flagged per episode (assessment ticks)")):
        bottom = [0] * n
        for cat in cats:
            heights = [sum(1 for d in e.decisions if d.get(key) == cat
                           and not (key == "method" and d.get("expired"))) for e in episodes]
            ax.bar(xs, heights, bottom=bottom, color=colours.get(cat, "#7f7f7f"),
                   label=cat.replace("_", " ").lower())
            bottom = [b + h for b, h in zip(bottom, heights)]
        for x, total_n in zip(xs, bottom):
            ax.text(x, total_n + 0.2, str(total_n), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(bottom + [1]) * 1.25)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode")
        ax.set_ylabel("ticks")
        if cats:
            ax.legend(fontsize=8)

    for a in axes.flat:
        a.set_xticks(xs)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def open_run(directory: str) -> tuple[dict, list[EpisodeRecording]]:
    """`load_run`, re-exported so callers only need this module."""
    return load_run(directory)
