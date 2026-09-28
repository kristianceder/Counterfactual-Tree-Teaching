"""Counterfactual replays of a recorded run, after the fact: was each of its recoveries needed?

    python scripts/replay_run.py results/runs/scripted_learning               # every episode
    python scripts/replay_run.py results/runs/scripted_learning --episode 22  # one episode

This is how the scripted rules' learning stream was labelled for the paper (Table 2's "needed" and
the rules' row of the counterfactual table): the rules run no replays of their own, so their
recording was replayed afterwards with the settings the method's loop uses. `scripts/run.py
--simulate` does the same between episodes while learning (`counterfactual.ExperienceReplayer`),
except that it labels at most six recoveries per episode and skips the breaker's orders.

Per episode, two things:

1. **Check.** Re-run the episode with `ReplayAnalyzer` answering every tick from the recording.
   It should match the recording frame for frame; where it does not (a solve near the wall-clock
   cap returned a different control), the first step it drifts from is noted, and labels past that
   step are marked `replay_clean_to_override: false`. If even the decisions differ, nothing is
   labelled.
2. **Counterfactuals.** For every recorded recovery (anything but CONTINUE that did not expire),
   re-run the episode with that one verdict replaced by CONTINUE; ticks after it are answered
   NOMINAL for `--grace` steps and by the scripted rules after that. The ending next to the
   recorded one gives the label (`counterfactual.label`): needed, unnecessary, harmful or unclear.

Writes `counterfactual_epNNN.json` into the run directory, next to the recording. Replays run one at
a time, since the solver is capped by wall time and a loaded machine drifts sooner.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np
import tyro  # type: ignore

from failure_monitor.counterfactual import GraceThenFallback, compare, label, run_replay
from failure_monitor.recorder import EpisodeRecorder
from failure_monitor.recovery import RecoveryMethod
from failure_monitor.replay import ReplayAnalyzer
from failure_monitor.scripted import ScriptedTeacher


@dataclasses.dataclass
class Args:
    run_dir: tyro.conf.Positional[str]
    """A `--record` directory: `run.json` plus `episode_NNN.json`."""
    episode: int = 0
    """Which episode (1-based); 0 replays them all."""
    grace: int = 50
    """Control steps after the replaced verdict answered NOMINAL before the scripted rules take over.
    0 would hand over at once, and rules that know the replaced action re-order it a tick later: the
    replay would measure a delay, not the recovery's absence."""
    slack: int = 10
    """Steps a replay may be slower than the recording and still count as `unnecessary`."""
    position_tol: float = 0.02
    """Metres of drift between replay and recording that still count as an exact reproduction."""


def goal_distance(frames: list[dict], goal: list[float], step: int) -> float | None:
    if step >= len(frames):
        return None
    f = frames[step]
    return float(np.hypot(f["x"] - goal[0], f["y"] - goal[1]))


def replay_episode(args: Args, index: int, recording: dict) -> dict:
    summary = recording["summary"]
    print(f"\n=== {args.run_dir} episode {index} (seed {summary.get('seed')}): recorded "
          f"{summary.get('termination')} in {summary.get('steps')} steps, {len(recording['decisions'])} decisions ===")
    recorded = {"termination": summary.get("termination"), "steps": summary.get("steps"),
                "success": summary.get("success")}
    out: dict = {"run_dir": args.run_dir, "episode": index, "seed": summary.get("seed"), "after": "scripted",
                 "grace": args.grace, "slack": args.slack, "recorded": recorded}

    started = time.perf_counter()
    analyzer = ReplayAnalyzer(recording)
    rec = EpisodeRecorder()
    result = run_replay(recording, analyzer, rec)
    rec.finish(termination=result.termination, steps=result.steps)
    check = compare(recording, rec, args.position_tol)
    check["replay"] = analyzer.summary()
    check["wall_s"] = round(time.perf_counter() - started, 1)
    check["exact"] = (check["decisions_match"] and check["max_drift_m"] <= args.position_tol
                      and check["termination_recorded"] == check["termination_replayed"]
                      and check["steps_recorded"] == check["steps_replayed"])
    out["check"] = check
    drift_from = "" if check["first_drift_step"] is None else f" (from step {check['first_drift_step']})"
    print(f"  check: {'EXACT' if check['exact'] else 'DIVERGED'} -- drift max {check['max_drift_m']:.4f} m{drift_from}, "
          f"termination {check['termination_replayed']} vs {check['termination_recorded']}, "
          f"steps {check['steps_replayed']} vs {check['steps_recorded']}, "
          f"decisions {'match' if check['decisions_match'] else 'DIFFER'}, {check['wall_s']} s")
    if not check["decisions_match"]:
        print("  counterfactuals skipped: the replay did not reproduce the decisions")
        out["counterfactuals"] = []
        return out

    first_drift = check["first_drift_step"]
    recoveries = [d for d in recording["decisions"]
                  if d["method"] != RecoveryMethod.CONTINUE.value and not d.get("expired")]
    goal = recording["meta"]["goal"]
    rows = []
    for d in recoveries:
        step, acted = int(d["report_step"]), int(d["acted_step"])
        fallback = GraceThenFallback(ScriptedTeacher(), step + args.grace) if args.grace > 0 else ScriptedTeacher()
        analyzer = ReplayAnalyzer(recording, override={step: RecoveryMethod.CONTINUE}, fallback=fallback)
        rec = EpisodeRecorder()
        started = time.perf_counter()
        result = run_replay(recording, analyzer, rec)
        rec.finish(termination=result.termination, steps=result.steps)
        cf = {"termination": result.termination, "steps": result.steps, "success": bool(result.success),
              "wall_s": round(time.perf_counter() - started, 1), "replay": analyzer.summary(),
              "goal_distance_at_horizon": goal_distance(rec.recording.frames, goal, acted + 25)}
        row = {"report_step": step, "acted_step": acted, "source": d.get("source"),
               # only as good as the unchanged replay up to the replaced verdict
               "replay_clean_to_override": first_drift is None or step <= first_drift,
               "failure_mode": d.get("failure_mode"), "method": d["method"], "rule": d.get("rule"),
               "overridden": d.get("overridden"),
               "recorded_goal_distance_at_horizon": goal_distance(recording["frames"], goal, acted + 25),
               "counterfactual": cf, "label": label(recorded, cf, args.slack)}
        rows.append(row)
        print(f"  {d['method']:<13} @{step:>4}: without it {cf['termination']}@{cf['steps']} "
              f"(recorded {recorded['termination']}@{recorded['steps']}) -> {row['label']}"
              + ("" if row["replay_clean_to_override"] else "  (replay had drifted before this step)"))
    out["counterfactuals"] = rows
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    out["label_counts"] = counts
    return out


def main(args: Args) -> None:
    run = json.load(open(os.path.join(args.run_dir, "run.json")))
    for entry in run["episodes"]:
        index = int(entry["index"])
        if args.episode and index != args.episode:
            continue
        with open(os.path.join(args.run_dir, entry["file"])) as f:
            out = replay_episode(args, index, json.load(f))
        path = os.path.join(args.run_dir, f"counterfactual_ep{index:03d}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
