"""Side-by-side replays of one layout under different deciders, played in sync: the supplementary
video's scenes 1-6 (scene 7 is `scripts/visualize_run.py`; `render_scenes.sh` renders all seven).

    python scripts/videos/compare_video.py luna312            # held-out layout, four deciders
    python scripts/videos/compare_video.py breakdown422       # unseen breakdown, both final trees
    python scripts/videos/compare_video.py rules121           # also rules246, rules280, rules162
    python scripts/videos/compare_video.py luna312 --still 280   # one frame (step 280) as PNG
    python scripts/videos/compare_video.py readme312 --speed 4 --fast 8 --dpi 60 --gif 960   # the README's GIF

Each panel replays its own recording of the same seed from results/runs (or `RUNS`), so the layout and
the pedestrians are the same and only the decisions differ. Every panel shows the same control step
(0.2 s each); a panel whose episode has ended holds its last frame and says how it ended. Playback is
2x real time while any episode that ends before the step budget is still running, then 4x once only
the ones running out the budget are left (--speed, --fast); the label under the clock says which. The
map uses the colours of fig_scenario_grid.py: blocks dark, a staged obstacle the planned route runs
through (the planner was not told about it) orange, staged walls the planner knows grey.
Writes videos/<preset>.mp4 (1920x1080, H.264) and its last frame as videos/<preset>.png at the
repository root. Needs ffmpeg from imageio-ffmpeg (`uv sync --extra viz`).

A preset with `show_ref: True` also draws the reference the MPC is tracking (the 20 points of its 4 s
horizon, as recorded), which shows a recovery that sends the robot backwards before the robot moves.
`notes: {panel index: [(first step, last step, text), ...]}` puts a caption above that panel's robot
over those steps ("|" breaks the line).
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("RUNS", os.path.join(REPO, "results", "runs"))
OUT = os.path.join(REPO, "videos")

# Seeds 100-299 are episodes 1-200 of every learning run, 300-319 episodes 1-20 of a held-out
# evaluation and 400-499 episodes 1-100 of an unseen one: seed s is episode s - 99, s - 299 or s - 399.
RULES = ("scripted_learning", "Scripted rules",
         "Hand-written tests: replan once the robot has stood still for 4 s, wait once stuck behind a pedestrian.")
METHOD = ("method_r1", "The method, while learning",
          "Decision tree taught by GPT-6 Luna over the layouts so far, with experience; Luna answers the misses.")


def rules_vs_method(seed: int, what: str, title: str | None = None) -> dict:
    """A learning layout: the scripted rules above, the method's first learning run below."""
    ep = seed - 99
    return dict(
        title=title or f"Learning layout {seed}: the rules fail, the method does not",
        subtitle=f"Seed {seed}, episode {ep}: {what}",
        layout="stack",
        panels=[(RULES[0], ep, RULES[1], RULES[2], False), (METHOD[0], ep, METHOD[1], METHOD[2], True)])


PRESETS = {
    # On held-out seed 312 the LLM alone runs out of time and the tree it taught reaches the goal. The
    # LLM alone's RESUME_ROUTE on a robot already driving re-attaches the route at the nearest waypoint,
    # 4.6 m behind it, so the robot reverses; the tree has no RESUME_ROUTE rule.
    "luna312": dict(
        title="GPT-6 Luna alone, and the tree it taught",
        subtitle="Held-out layout (seed 312), none of them learned on it: the robot starts facing away from its "
                 "route, an unknown wall blocks it, and two gates close.",
        layout="side",   # 2x2, text beside each map
        show_ref=True,   # the reference flips behind the LLM alone's robot one step after its RESUME_ROUTE (54.2 s)
        # step 271: RESUME_ROUTE acts; the route is trimmed at waypoint (9.86, 16.49), 4.59 m behind the robot
        notes={2: [(271, 305, "the resumed route starts|4.6 m behind the robot")]},
        panels=[  # run, episode (1-based), name, what it is, is the method
            ("no_recovery_heldout", 13, "No supervisor",
             "The MPC tracker alone, with no recovery layer.", False),
            ("scripted_heldout", 13, "Scripted rules",
             "Hand-written tests: replan once the robot has stood still for 4 s.", False),
            ("luna_heldout_r1", 13, "GPT-6 Luna alone",
             "Asked every 2 s over the API; an answer acts when it arrives. No worked examples in the prompt.",
             False),
            ("method_r1_heldout_ep0200", 13, "Decision tree taught by Luna",
             "Grown from Luna's answers (same prompt) over 200 other layouts, with experience, then frozen. "
             "No LLM on board.", True),
        ]),
    # The final trees of the method and of distillation alone on unseen seed 422, frozen. Both wait from
    # the first tick; once the obstacle has stood 30 s the method's tree calls a human and distillation's
    # `no/no/dynamic -> WAIT` keeps waiting until the watchdog stops the controller.
    "breakdown422": dict(
        title="A breakdown, with and without experience",
        subtitle="Unseen layout (seed 422), trees taught by GPT-6 Luna, frozen, no LLM on board: an obstacle "
                 "parks in the only passage and never leaves.",
        layout="stack",  # 1x2, text above and below each map
        panels=[
            ("distillation_r1_unseen", 23, "Distillation alone",
             "Tree grown from Luna's answers over 200 layouts, then frozen; Luna saw only the status report.",
             False),
            ("method_r1_unseen", 23, "Distillation with experience (the method)",
             "The same, but Luna was also shown the rules learned so far and how earlier recoveries turned out.",
             True),
        ]),
    # The README's GIF: the paper's headline on one held-out layout. The MPC alone stands at the wall
    # until the step budget runs out; the frozen tree replans at its first tick, waits out a gate and
    # reaches the goal, with no LLM on board.
    "readme312": dict(
        title="The MPC alone, and the tree an LLM taught",
        subtitle="Held-out layout (seed 312): the robot starts facing away from its route, a wall the planner "
                 "does not know about blocks it, and two gates close.",
        layout="stack",
        panels=[
            ("no_recovery_heldout", 13, "MPC alone",
             "The trajectory tracker with no recovery layer.", False),
            ("method_r1_heldout_ep0200", 13, "MPC + decision tree (our method)",
             "Taught by GPT-6 Luna over 200 other layouts, then frozen. No LLM on board.", True),
        ]),
    # The three learning layouts the scripted rules end wrong and the method ends right, and 162, the one
    # the method ends wrong in all three runs and the rules end right.
    "rules121": rules_vs_method(121, "a wall, then a gate. Same two recoveries on both sides; the tree replans 6 s sooner, and that is the margin."),
    "rules246": rules_vs_method(246, "a wall, then a gate with a pedestrian. The tree replans 8 s sooner; the rules' robot meets the pedestrian in the gate."),
    "rules280": rules_vs_method(280, "a wall, then a gate. The tree replans at once and turns to the route; the rules replan 18 s later and stand behind a pedestrian."),
    "rules162": rules_vs_method(162, "a wall, then a gate. The tree replans 12 s sooner but holds twice at the gate, then 20 s more up top for a pedestrian on the route.",
                                title="Learning layout 162: the method fails, the rules do not"),
}

TS = 0.2                 # control period, s
BLOCK = "#3a3a38"        # the blocks the planner knows
STAGED = "#eb6834"       # a staged obstacle the planned route runs through: the planner was not told about it
KNOWN = "#9c9b97"        # staged walls the planner knows (the walls that make a gate or seal a detour)
PED = "#e34948"
ROUTE = "#52514e"
TRAIL = "#2a78d6"
REF = "#d63384"          # the MPC's reference over its horizon (presets with show_ref)
ROBOT = "#0b3d7a"
GOAL = "#1baf7a"
GOOD, BAD, INK, MUTED = "#1a7f45", "#c62828", "#1d1d1b", "#6b6a66"
FAST_INK = "#b35900"     # the speed label while fast-forwarding
METHOD_INK = "#0b3d7a"
MARK = {  # recovery -> (marker, colour, size, map label)
    "REPLAN_ROUTE": ("*", "#eda100", 20, "replan"),
    "WAIT": ("D", "#4a3aa7", 11, "wait"),
    "REQUEST_HUMAN": ("X", "#0b0b0b", 14, "human called"),
    "REORIENT": ("^", "#1baf7a", 12, "turn"),
    "RESUME_ROUTE": ("s", "#17becf", 10, "resume route"),
}
ACT = {"REPLAN_ROUTE": "replan", "WAIT": "wait", "REQUEST_HUMAN": "call a human", "REORIENT": "turn to the route",
       "RESUME_ROUTE": "resume the route", "BACK_OFF": "back off", "CONTINUE": "carry on"}


def is_block(poly: np.ndarray) -> bool:
    """The 3x3 blocks are 4 m squares; anything else is a staged obstacle."""
    w, h = poly[:, 0].max() - poly[:, 0].min(), poly[:, 1].max() - poly[:, 1].min()
    return abs(w - 4.0) < 0.05 and abs(h - 4.0) < 0.05


def crosses(poly: np.ndarray, route: np.ndarray) -> bool:
    """Whether the planned route passes through the obstacle (its bounding box, sampled every 5 cm)."""
    x0, x1, y0, y1 = poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()
    for a, b in zip(route[:-1], route[1:]):
        for t in np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / 0.05))):
            x, y = a + t * (b - a)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
    return False


class Panel:
    """One decider: its map on `ax_map`, its name, what it is and its live status on the text axes."""

    def __init__(self, ax_map, ax_head, ax_status, run: str, episode: int, name: str, blurb: str,
                 method: bool, wrap: int, status_wrap: int, show_ref: bool = False, notes=()):
        self.show_ref = show_ref
        self.notes = [(a, b, text.replace("|", "\n")) for a, b, text in notes]
        rec = json.load(open(os.path.join(DATA, run, f"episode_{episode:03d}.json")))
        self.teacher = json.load(open(os.path.join(DATA, run, "run.json")))["meta"].get("teacher")
        self.frames, self.meta, self.summary = rec["frames"], rec["meta"], rec["summary"]
        self.decisions = sorted(rec["decisions"], key=lambda d: d["acted_step"])
        self.recoveries = [r for r in rec["recoveries"] if not str(r.get("detail", "")).startswith("follow-up")]
        self.no_supervisor = all(d["source"] == "unanswered" for d in self.decisions) and not self.recoveries
        self.llm_on_board = self.teacher == "llm"
        self.frozen_tree = self.teacher == "none" and any(d["source"] == "tree" for d in self.decisions)
        self.status_wrap = status_wrap
        self.ax = ax_map
        self.present = set()   # legend entries this panel draws
        self._draw_static()
        self._draw_dynamic()
        ink = METHOD_INK if method else INK
        for ax in (ax_head, ax_status):
            ax.axis("off")
        ax_head.text(0, 1, name, transform=ax_head.transAxes, ha="left", va="top", fontsize=17,
                     fontweight="bold", color=ink)
        ax_head.text(0, 0.62, "\n".join(textwrap.wrap(blurb, wrap)), transform=ax_head.transAxes, ha="left",
                     va="top", fontsize=11.5, color=MUTED, linespacing=1.3)
        self.status = ax_status.text(0, 1, "", transform=ax_status.transAxes, ha="left", va="top",
                                     fontsize=13, color=INK, linespacing=1.55)

    # -- the map ---------------------------------------------------------------------------------

    def _draw_static(self) -> None:
        ax, m = self.ax, self.meta
        b = np.array(m["boundary"])
        ax.add_patch(Polygon(b, closed=True, fill=False, edgecolor="#8a8987", linewidth=1.0))
        r0 = np.array(m["initial_route"])
        for o in m["obstacles"]:
            p = np.array(o)
            kind = "blocks" if is_block(p) else "unknown" if crosses(p, r0) else "known"
            colour = {"blocks": BLOCK, "unknown": STAGED, "known": KNOWN}[kind]
            ax.add_patch(Polygon(p, closed=True, facecolor=colour, edgecolor="none", zorder=2))
            self.present.add(kind)
        ax.plot(r0[:, 0], r0[:, 1], ":", color=ROUTE, linewidth=1.4, alpha=0.8, zorder=3)
        ax.plot(*m["start"], marker="o", color=TRAIL, markersize=9, markeredgecolor="white",
                markeredgewidth=1.0, zorder=4, linestyle="none")
        ax.plot(*m["goal"], marker="P", color=GOAL, markersize=16, markeredgecolor="white",
                markeredgewidth=1.0, zorder=6, linestyle="none")
        ax.set_aspect("equal")
        ax.set_xlim(b[:, 0].min() - 0.4, b[:, 0].max() + 0.4)
        ax.set_ylim(b[:, 1].min() - 0.4, b[:, 1].max() + 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _draw_dynamic(self) -> None:
        ax, m = self.ax, self.meta
        self.routes, current = [], m["initial_route"]
        for f in self.frames:
            if f.get("route"):
                current = f["route"]
            self.routes.append(current)
        self.route_line, = ax.plot([], [], "--", color=ROUTE, linewidth=1.8, zorder=3)
        self.trail, = ax.plot([], [], "-", color=TRAIL, linewidth=2.4, zorder=4, solid_capstyle="round")
        if self.show_ref:
            self.ref_line, = ax.plot([], [], "-", color=REF, linewidth=3.2, alpha=0.85, zorder=6,
                                     solid_capstyle="round")
            self.present.add("ref")
        n_ped = max((len(f["dyn"]) for f in self.frames), default=0)
        if n_ped:
            self.present.add("pedestrian")
        self.peds = [ax.add_patch(Circle((0, 0), m.get("dynamic_obstacle_radius", 0.8), facecolor=PED, alpha=0.6,
                                         edgecolor="none", zorder=5, visible=False)) for _ in range(n_ped)]
        rr = m.get("robot_radius", 0.5)
        self.hold_ring = ax.add_patch(Circle((0, 0), rr * 1.9, fill=False, linewidth=2.5, zorder=8, visible=False))
        self.robot = ax.add_patch(Circle((0, 0), rr, facecolor=ROBOT, edgecolor="white", linewidth=1.2, zorder=9))
        self.heading, = ax.plot([], [], "-", color="white", linewidth=2.2, zorder=10, solid_capstyle="round")
        # Recovery markers, one per method, outcome and place: a replan repeated where the robot stands is
        # one star with a count. Labels of markers near each other are stacked so that none hides another.
        self.groups = []   # dicts: method, ok, x, y, steps, marker, label
        for r in self.recoveries:
            if r["method"] not in MARK:
                continue
            f = self.frames[min(int(r["step"]), len(self.frames) - 1)]
            ok = bool(r.get("success", True))
            g = next((g for g in self.groups if g["method"] == r["method"] and g["ok"] == ok
                      and np.hypot(g["x"] - f["x"], g["y"] - f["y"]) < 0.7), None)
            if g is None:
                mk, col, size, _ = MARK[r["method"]]
                x, y = f["x"], f["y"]
                while any(np.hypot(x - h["x"], y - h["y"]) < 0.7 for h in self.groups):
                    x, y = x + 0.8, y + 0.8
                near = sum(1 for h in self.groups if np.hypot(x - h["x"], y - h["y"]) < 3.0)
                marker, = ax.plot(x, y, marker=mk, markersize=size, color=col, markeredgecolor="white",
                                  markeredgewidth=1.0, zorder=7, linestyle="none", visible=False)
                label = ax.annotate("", (x, y), xytext=(13, 8 - 24 * near), textcoords="offset points",
                                    fontsize=11.5, color=INK, zorder=11, visible=False,
                                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none",
                                              alpha=0.9))
                g = dict(method=r["method"], ok=ok, x=x, y=y, steps=[], marker=marker, label=label)
                self.groups.append(g)
                self.present.add(r["method"])
            g["steps"].append(int(r["step"]))
        self.badge = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=15,
                             fontweight="bold", color="white", zorder=12,
                             bbox=dict(boxstyle="round,pad=0.35", facecolor=GOOD, edgecolor="none"))
        self.badge.set_visible(False)
        self.note = ax.text(0, 0, "", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=INK,
                            zorder=12, visible=False, linespacing=1.3,
                            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=REF, alpha=0.92))

    # -- per frame -------------------------------------------------------------------------------

    def who(self, source: str) -> str:
        if source == "llm":
            return "the LLM" if self.teacher == "llm" else "the rules"
        return {"tree": "the tree", "breaker": "the loop breaker", "unanswered": "no rule"}.get(source, source)

    def ending(self) -> tuple[bool, str, str]:
        """(right?, badge text, what it means)"""
        s = self.summary
        t = s["steps"] * TS
        term = s["termination"]
        if term == "human_requested":
            kinds = {sit["kind"] for sit in (self.meta.get("layout") or {}).get("situations", [])}
            why = ("the obstacle in the only passage never leaves" if "breakdown" in kinds and "sealed" not in kinds
                   else "no route exists")
            if s.get("needs_human"):
                return True, f"Called a human at {t:.1f} s", f"the right call, as {why}"
            return False, f"Called a human at {t:.1f} s", "the wrong call, as there was a way through"
        return {"goal": (True, f"Goal reached at {t:.1f} s", "goal reached"),
                "timeout": (False, f"Out of time at {t:.0f} s", "the step budget ran out"),
                "watchdog": (False, f"Watchdog stop at {t:.1f} s", "the controller could not keep up"),
                "collision": (False, f"Collision at {t:.1f} s", "collision"),
                }.get(term, (bool(s.get("success")), f"{term} at {t:.1f} s", term))

    def draw(self, step: int) -> None:
        last = len(self.frames) - 1
        k = min(step, last)
        f = self.frames[k]
        trail = self.frames[:k + 1]
        self.trail.set_data([p["x"] for p in trail], [p["y"] for p in trail])
        if self.show_ref:
            ref = f.get("ref") or []
            self.ref_line.set_data([p[0] for p in ref], [p[1] for p in ref])
        route = self.routes[k]
        self.route_line.set_data([p[0] for p in route], [p[1] for p in route])
        self.robot.center = (f["x"], f["y"])
        note = next((text for a, b, text in self.notes if a <= k <= b), None)
        self.note.set_visible(note is not None)
        if note is not None:
            self.note.set_position((f["x"], f["y"] + 1.3))
            self.note.set_text(note)
        rr = self.meta.get("robot_radius", 0.5)
        self.heading.set_data([f["x"], f["x"] + 0.95 * rr * np.cos(f["heading"])],
                              [f["y"], f["y"] + 0.95 * rr * np.sin(f["heading"])])
        for patch, centre in zip(self.peds, f["dyn"]):
            patch.center = tuple(centre)
            patch.set_visible(True)
        for patch in self.peds[len(f["dyn"]):]:
            patch.set_visible(False)
        hold = (f.get("hold") or "").split(":")[0]
        self.hold_ring.set_visible(bool(hold))
        if hold:
            self.hold_ring.center = (f["x"], f["y"])
            self.hold_ring.set_edgecolor(MARK.get(hold, (None, "#4a3aa7"))[1])
        so_far = {}   # "replan (no route)" -> count, over all of that recovery's markers, in order of first use
        for g in self.groups:
            n = sum(1 for s in g["steps"] if s <= k)
            g["marker"].set_visible(n > 0)
            g["label"].set_visible(n > 0)
            text = MARK[g["method"]][3] + ("" if g["ok"] else ": no route")
            g["label"].set_text(text + (f"  ×{n}" if n > 1 else ""))
            if n:
                what = ACT.get(g["method"], g["method"]) + ("" if g["ok"] else " (no route)")
                so_far[what] = so_far.get(what, 0) + n

        # the status block
        lines = []
        if k >= last:
            ok, badge, meaning = self.ending()
            lines.append(f"episode over: {meaning}")
        elif hold:   # "REPLAN_ROUTE: turning to face the route", "WAIT: holding (5 steps left)", ...
            what = f["hold"].split(":", 1)[-1].strip()
            lines.append("robot: " + ("turning to face the new route" if hold == "REPLAN_ROUTE" else what))
        elif abs(f["v"]) < 0.05:
            n = 0
            while k - n >= 0 and abs(self.frames[k - n]["v"]) < 0.05:
                n += 1
            lines.append(f"robot: standing still for {n * TS:.1f} s")
        else:
            lines.append(f"robot: {'reversing' if f['v'] < 0 else 'moving'} at {abs(f['v']):.1f} m/s")
        done = [d for d in self.decisions if d["acted_step"] <= k]
        orders = [d for d in done if d["method"] != "CONTINUE"]
        if self.no_supervisor:
            lines.append("no recovery layer: nothing is ordered")
        elif orders:
            d = orders[-1]
            took = (f" (its answer took {d['latency_s']:.1f} s)" if d["source"] == "llm" and self.llm_on_board
                    else "")
            lines.append(f"last order: {ACT.get(d['method'], d['method'])}, by {self.who(d['source'])} "
                         f"at {d['acted_step'] * TS:.1f} s{took}")
        else:
            lines.append("no recovery ordered yet")
        if so_far:
            lines.append("recoveries so far: " + ", ".join(what + (f" ×{n}" if n > 1 else "")
                                                             for what, n in so_far.items()))
        if self.llm_on_board:
            lines.append(f"LLM calls so far: {sum(1 for d in done if d['source'] == 'llm')}")
        elif self.frozen_tree:
            lines.append("LLM calls: none, no LLM on board")
        self.status.set_text("\n".join(textwrap.fill(line, self.status_wrap, subsequent_indent="   ")
                                       for line in lines))
        if k >= last:
            ok, badge, _ = self.ending()
            self.badge.set_text(("✓ " if ok else "✗ ") + badge)
            self.badge.get_bbox_patch().set_facecolor(GOOD if ok else BAD)
        self.badge.set_visible(k >= last)


def build(preset: dict):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.025, 0.965, preset["title"], ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.025, 0.918, preset["subtitle"], ha="left", va="top", fontsize=13, color=MUTED)
    clock = fig.text(0.975, 0.965, "", ha="right", va="top", fontsize=17, color=INK, family="monospace")
    speed = fig.text(0.862, 0.958, "", ha="right", va="top", fontsize=12, color=MUTED)   # left of the clock
    panels = []
    specs = preset["panels"]
    show_ref, notes = preset.get("show_ref", False), preset.get("notes", {})
    if preset["layout"] == "side":       # 2x2: map left, text right, per cell
        cw, ch, y0 = 0.5, 0.405, 0.08
        for i, (run, ep, name, blurb, method) in enumerate(specs):
            col, row = i % 2, i // 2
            left, bottom = col * cw, y0 + (1 - row) * ch
            ax_map = fig.add_axes([left + 0.012, bottom + 0.01, 0.225, ch - 0.02])
            ax_head = fig.add_axes([left + 0.245, bottom + ch * 0.62, cw - 0.26, ch * 0.36])
            ax_status = fig.add_axes([left + 0.245, bottom + 0.02, cw - 0.26, ch * 0.58])
            panels.append(Panel(ax_map, ax_head, ax_status, run, ep, name, blurb, method, wrap=44, status_wrap=38,
                                show_ref=show_ref, notes=notes.get(i, ())))
    else:                                # 1xN: name above, map, status below
        n = len(specs)
        cw = 1.0 / n
        for i, (run, ep, name, blurb, method) in enumerate(specs):
            left = i * cw
            ax_head = fig.add_axes([left + 0.03, 0.765, cw - 0.06, 0.12])
            ax_map = fig.add_axes([left + 0.03, 0.235, cw - 0.06, 0.52])
            ax_status = fig.add_axes([left + 0.03, 0.075, cw - 0.06, 0.15])
            panels.append(Panel(ax_map, ax_head, ax_status, run, ep, name, blurb, method, wrap=80, status_wrap=70,
                                show_ref=show_ref, notes=notes.get(i, ())))
    present = set().union(*(p.present for p in panels))
    entries = [("blocks", Patch(facecolor=BLOCK, label="blocks")),
               ("unknown", Patch(facecolor=STAGED, label="obstacle unknown to the planner")),
               ("known", Patch(facecolor=KNOWN, label="walls the planner knows")),
               ("", Line2D([], [], linestyle=":", color=ROUTE, linewidth=1.4, label="planned route")),
               ("", Line2D([], [], linestyle="--", color=ROUTE, linewidth=1.8, label="current route")),
               ("", Line2D([], [], color=TRAIL, linewidth=2.4, label="driven path")),
               ("ref", Line2D([], [], color=REF, linewidth=3.2, alpha=0.85, label="MPC reference, next 4 s")),
               ("pedestrian", Line2D([], [], marker="o", linestyle="none", color=PED, alpha=0.6, markersize=11,
                                     label="pedestrian")),
               ("", Line2D([], [], marker="P", linestyle="none", color=GOAL, markersize=12, label="goal"))]
    entries += [(meth, Line2D([], [], marker=mk, linestyle="none", color=col, markeredgecolor="white",
                              markersize=min(size, 14), label=lab)) for meth, (mk, col, size, lab) in MARK.items()]
    handles = [h for key, h in entries if not key or key in present]
    ncol = len(handles) if len(handles) <= 9 else (len(handles) + 1) // 2   # past nine it overflows one row
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.012 if ncol == len(handles) else 0.004), handlelength=1.8,
               columnspacing=1.3, handletextpad=0.5)
    return fig, panels, clock, speed


def playback(panels: list[Panel], fps: float, speed: float, fast: float, hold_start: float,
             hold_end: float) -> list[tuple[int, float]]:
    """(control step, playback speed) per video frame. `speed` while any episode that ends before the step
    budget is still running and for one second of video after the last of them has ended, `fast` from
    there on, when only the ones running out the budget are left."""
    n_steps = max(len(p.frames) for p in panels)
    ends = [len(p.frames) - 1 for p in panels if p.summary["termination"] != "timeout"]
    switch = max(ends) + int(round(speed / TS)) if ends else n_steps
    frames = [(0, speed)] * int(hold_start * fps)
    t = 0.0
    while True:
        k = min(int(round(t)), n_steps - 1)
        now = speed if k < switch else fast
        frames.append((k, now))
        if k >= n_steps - 1:
            break
        t += now / (fps * TS)   # control steps per video frame at this speed
    return frames + [(n_steps - 1, frames[-1][1])] * int(hold_end * fps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--fps", type=float, default=10.0, help="video frames per second")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed (x real time) while episodes that "
                    "end before the step budget are still running")
    ap.add_argument("--fast", type=float, default=4.0, help="playback speed once only timeouts are left")
    ap.add_argument("--still", type=int, default=None, help="write one PNG of this control step and stop")
    ap.add_argument("--hold-start", type=float, default=1.5, help="seconds on step 0 before playing")
    ap.add_argument("--hold-end", type=float, default=4.0, help="seconds on the last frame")
    ap.add_argument("--dpi", type=float, default=120, help="pixels per inch of the 16x9 in figure (120: 1920x1080)")
    ap.add_argument("--gif", type=int, default=0, help="also write videos/<preset>.gif, this many pixels wide "
                    "(ffmpeg palette, from the MP4)")
    args = ap.parse_args()
    fig, panels, clock, speed_label = build(PRESETS[args.preset])

    def update(step: int, speed: float) -> None:
        clock.set_text(f"t = {step * TS:5.1f} s")
        fast = speed != args.speed
        speed_label.set_text(f"▶▶ fast-forward: {speed:g}× real time" if fast
                             else f"played at {speed:g}× real time")
        speed_label.set_color(FAST_INK if fast else MUTED)
        speed_label.set_fontweight("bold" if fast else "normal")
        speed_label.set_fontsize(14 if fast else 12)
        for p in panels:
            p.draw(step)

    os.makedirs(OUT, exist_ok=True)
    if args.still is not None:
        update(args.still, args.speed)
        path = os.path.join(OUT, f"{args.preset}_step{args.still:03d}.png")
        fig.savefig(path, dpi=args.dpi)
        print("wrote", path)
        return
    import imageio_ffmpeg  # type: ignore
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=args.fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
    schedule = playback(panels, args.fps, args.speed, args.fast, args.hold_start, args.hold_end)
    path = os.path.join(OUT, f"{args.preset}.mp4")
    with writer.saving(fig, path, dpi=args.dpi):
        for step, speed in schedule:
            update(step, speed)
            writer.grab_frame()
    fig.savefig(os.path.join(OUT, f"{args.preset}.png"), dpi=args.dpi)
    switch = next((i for i, (_, s) in enumerate(schedule) if s != args.speed), None)
    print(f"wrote {path} ({len(schedule)} frames, {len(schedule) / args.fps:.1f} s"
          + (f"; {args.fast:g}x from {switch / args.fps:.1f} s, step {schedule[switch][0]}" if switch else "")
          + f") and {args.preset}.png")
    if args.gif:
        import subprocess
        gif = os.path.join(OUT, f"{args.preset}.gif")
        palette = (f"fps={args.fps:g},scale={args.gif}:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path, "-filter_complex", palette, "-loop", "0", gif],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
