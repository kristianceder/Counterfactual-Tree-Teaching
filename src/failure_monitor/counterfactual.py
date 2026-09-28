"""Counterfactual replays: was a recovery needed?

The judge in `tree.py` scores a recovery on whether the robot got going again afterwards, which a
robot that was fine all along also does. The replay answers the other half. The recorded episode is
re-run with `replay.ReplayAnalyzer` answering every tick from the recording, except that one
recovery is replaced by CONTINUE; ticks after that step are answered NOMINAL for `grace` steps and
by the scripted rules after that. Its ending, next to the recorded one, gives the label:

- `unnecessary`: without it the episode still ended right, no later than it did with it;
- `needed`: without it the episode ended wrong, or right but more than `slack` steps later;
- `harmful`: without it the episode ended right where the recording did not, or sooner;
- `unclear`: both ended wrong.

"Ended right" is reaching the goal, or calling a human on a layout that needs one. `needed` is an
upper bound on the recovery's own contribution: every later recovery is changed too.

Every replay starts with the unchanged episode, as an exactness check; a replay is trusted only up
to the first step it drifts from the recording (the solver is capped by wall time, see
`replay.py`), and labels past that step are marked not `clean`.

`ExperienceReplayer` runs the replays in worker processes between episodes (`scripts/run.py
--simulate`); the labels go into the experience table and onto the rules (`TreePolicy.note_counterfactual`).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from .recorder import EpisodeRecorder
from .recovery import RecoveryMethod
from .replay import ReplayAnalyzer


def run_replay(recording: dict, analyzer: ReplayAnalyzer, recorder: EpisodeRecorder | None = None):
    """Re-run the episode `recording` describes with `analyzer` answering its ticks."""
    from .episode import run_episode, seed_everything
    meta, summary = recording["meta"], recording["summary"]
    seed_everything(int(summary["seed"]))
    return run_episode(analyzer, scenario_option=int(meta.get("scenario_option", 1)),
                       max_steps=int(meta["max_steps"]), assess_hz=float(meta["assess_hz"]),
                       recorder=recorder, watchdog_steps=int(meta["watchdog_steps"]),
                       watchdog_margin=int(meta["watchdog_margin"]))


def compare(recording: dict, replay: EpisodeRecorder, tol: float = 0.02) -> dict:
    """How closely a replay reproduced the recording."""
    a = recording["frames"]
    b = replay.recording.frames
    n = min(len(a), len(b))
    drift = [float(np.hypot(a[i]["x"] - b[i]["x"], a[i]["y"] - b[i]["y"])) for i in range(n)]
    first_drift = next((i for i, d in enumerate(drift) if d > tol), None)
    rec_dec = [(d["report_step"], d["acted_step"], d["method"]) for d in recording["decisions"]]
    rep_dec = [(d["report_step"], d["acted_step"], d["method"]) for d in replay.recording.decisions]
    return {
        "frames_recorded": len(a), "frames_replayed": len(b),
        "max_drift_m": max(drift) if drift else 0.0,
        "first_drift_step": first_drift,
        "termination_recorded": recording["summary"].get("termination"),
        "termination_replayed": replay.recording.summary.get("termination"),
        "steps_recorded": recording["summary"].get("steps"),
        "steps_replayed": replay.recording.summary.get("steps"),
        "decisions_match": rec_dec == rep_dec,
        "decisions_recorded": len(rec_dec), "decisions_replayed": len(rep_dec),
    }


class GraceThenFallback:
    """Answer NOMINAL/CONTINUE until `until_step`, then hand every tick to `fallback`."""

    def __init__(self, fallback, until_step: int):
        self.fallback, self.until_step = fallback, until_step

    def assess_situation(self, context, candidates=None, evidence=None, stall_threshold=None):
        from .llm import SituationAssessment
        if int(getattr(context, "step", 0) or 0) <= self.until_step:
            return SituationAssessment(method=RecoveryMethod.CONTINUE, rationale="", raw_response="grace:nominal",
                                       failure_mode=None)
        return self.fallback.assess_situation(context, candidates, evidence=evidence,
                                              stall_threshold=stall_threshold)


def _ended_right(outcome: dict) -> bool:
    if outcome.get("success") is None:
        return outcome["termination"] == "goal"
    return bool(outcome["success"])


def label(recorded: dict, cf: dict, slack: int) -> str:
    """The label for one replay; `recorded`/`cf` carry `termination`, `steps` and `success`."""
    r_goal, c_goal = _ended_right(recorded), _ended_right(cf)
    if r_goal and c_goal:
        if cf["steps"] <= recorded["steps"] + slack:
            return "unnecessary" if cf["steps"] >= recorded["steps"] - slack else "harmful"
        return "needed"
    if r_goal and not c_goal:
        return "needed"
    if not r_goal and c_goal:
        return "harmful"
    return "unclear"


def _replay(job: tuple[dict, int | None, int]) -> dict:
    """One replay in a worker: unchanged if `step` is None (the exactness check), else with the
    decision on the report from `step` replaced by CONTINUE."""
    recording, step, grace = job
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .scripted import ScriptedTeacher

    override, fallback = None, None
    if step is not None:
        scripted = ScriptedTeacher()
        fallback = GraceThenFallback(scripted, step + grace) if grace > 0 else scripted
        override = {step: RecoveryMethod.CONTINUE}
    analyzer = ReplayAnalyzer(recording, override=override, fallback=fallback)
    recorder = EpisodeRecorder()
    started = time.perf_counter()
    result = run_replay(recording, analyzer, recorder)
    recorder.finish(termination=result.termination, steps=result.steps)
    out = {"step": step, "termination": result.termination, "steps": result.steps,
           "success": bool(result.success), "overridden": analyzer.summary().get("overridden", 0),
           "wall_s": round(time.perf_counter() - started, 1)}
    if step is None:
        out["check"] = compare(recording, recorder)
    return out


class ExperienceReplayer:
    """A pool of replay workers kept alive across episodes. `label_episode` blocks until the
    episode's replays are done.

    Args:
        workers: replay processes.
        grace: steps after the replaced decision answered NOMINAL before the scripted rules take
            over. 0 would hand over at once, and rules that know the replaced action would simply
            re-order it a tick later: the replay would measure a delay, not the recovery's absence.
        slack: steps a replay may be slower than the recording and still count as `unnecessary`.
        max_recoveries: recoveries labelled per episode (the first ones).
    """

    def __init__(self, workers: int = 3, grace: int = 50, slack: int = 10, max_recoveries: int = 6):
        self.grace, self.slack, self.max_recoveries = grace, slack, max_recoveries
        self.pool = mp.get_context("spawn").Pool(max(1, workers))
        self.stats = {"episodes": 0, "replays": 0, "labelled": 0, "skipped_inexact": 0,
                      "never_reached": 0, "wall_s": 0.0}

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def label_episode(self, recording: dict, table=None, verbose: bool = True) -> list[dict]:
        """Counterfactual labels for the recoveries of one recorded episode, filed in `table`."""
        started = time.perf_counter()
        # Not the breaker's: it is a rule, so the replay re-issues the same order on the next tick.
        recoveries = [d for d in recording["decisions"]
                      if d["method"] != "CONTINUE" and not d.get("expired") and d.get("signature")
                      and not d.get("overridden") and d.get("source") != "breaker"]
        recoveries = recoveries[:self.max_recoveries]
        self.stats["episodes"] += 1
        if not recoveries:
            return []
        jobs = [(recording, None, self.grace)]
        jobs += [(recording, int(d["report_step"]), self.grace) for d in recoveries]
        results = self.pool.map(_replay, jobs)
        self.stats["replays"] += len(jobs)
        check, replays = results[0]["check"], results[1:]

        summary = recording["summary"]
        recorded = {"termination": summary.get("termination"), "steps": summary.get("steps"),
                    "success": summary.get("success")}
        rows = []
        if not check["decisions_match"]:
            self.stats["skipped_inexact"] += 1
            if verbose:
                print("  [experience] the episode did not replay its own decisions; nothing labelled")
        else:
            for d, r in zip(recoveries, replays):
                if not r["overridden"]:
                    self.stats["never_reached"] += 1
                    continue
                cf = {"termination": r["termination"], "steps": r["steps"], "success": r["success"]}
                row = {"report_step": d["report_step"], "method": d["method"], "source": d.get("source"),
                       "signature": d["signature"], "label": label(recorded, cf, self.slack),
                       "without_it": f"{r['termination']}@{r['steps']}",
                       "clean": check["first_drift_step"] is None or d["report_step"] <= check["first_drift_step"]}
                rows.append(row)
                if table is not None:
                    table.record_counterfactual(d["signature"], d["method"], row["label"])
            self.stats["labelled"] += len(rows)
        wall = time.perf_counter() - started
        self.stats["wall_s"] += wall
        if verbose:
            ended = f"{recorded['termination']}@{recorded['steps']}"
            print(f"  [experience] {len(jobs)} replays in {wall:.0f} s (recorded {ended}): "
                  + (", ".join(f"{r['method']}@{r['report_step']} {r['label']} (without it {r['without_it']})"
                               for r in rows) or "no labels"))
        return rows
