"""Record what a run did, step by step, so it can be watched afterwards.

Rendering while the control loop runs is not free: every frame delays the loop, which changes
where an LLM verdict lands, which changes the episode. So a run records instead, and
`visualizer.recovery_replay` draws the recording later, as often as wanted, without touching the
result.

A run directory holds one `episode_NNN.json` per episode and a `run.json` with the per-episode
summary that the learning curve is drawn from. Everything is plain JSON with floats rounded to
millimetres, a few hundred kilobytes per episode.

Stdlib only, like `tree.py`: the harness can import it without pulling in anything heavy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

# Who answered an assessment tick. Read off `SituationAssessment.raw_response`, which
# `TreeAnalyzer` stamps for every answer that did not come from the model.
SOURCE_TREE = "tree"
SOURCE_LLM = "llm"
SOURCE_BREAKER = "breaker"
SOURCE_UNANSWERED = "unanswered"   # a tree miss with no teacher loaded (answered NOMINAL/CONTINUE)


def decision_source(raw_response: str | None) -> str:
    """Classify a decision by who made it. See `tree_analyzer.TreeAnalyzer._raw`."""
    raw = raw_response or ""
    if raw == "breaker":
        return SOURCE_BREAKER
    if raw == "tree:miss":
        return SOURCE_UNANSWERED
    if raw.startswith("tree:"):
        return SOURCE_TREE
    return SOURCE_LLM


def _r(value: Any) -> Any:
    """Round floats (also inside lists/tuples) to 3 decimals for compact JSON. Numpy scalars are
    converted to plain Python numbers on the way, so callers can pass `np.float32` positions."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        value = value.item()
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, (list, tuple)):
        return [_r(v) for v in value]
    if isinstance(value, dict):
        return {k: _r(v) for k, v in value.items()}
    return value


def _points(points: Iterable[Iterable[float]] | None) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in (points or [])]


@dataclass
class EpisodeRecording:
    """Everything one episode needs to be replayed. Built by `run_episode` through `EpisodeRecorder`."""
    meta: dict = field(default_factory=dict)
    """Static scene and settings: boundary, obstacles, goal, radii, control period, seed, ..."""
    frames: list[dict] = field(default_factory=list)
    """One per control cycle: pose, speed, dynamic obstacles, tracked reference, route, hold."""
    asks: list[dict] = field(default_factory=list)
    """Assessment ticks: the step a report was taken and handed to the analyzer."""
    decisions: list[dict] = field(default_factory=list)
    """Verdicts acted on: report step, acted step, source, failure mode, recovery, rule/rationale."""
    recoveries: list[dict] = field(default_factory=list)
    """Recoveries actually executed."""
    notes: list[dict] = field(default_factory=list)
    """Loop events worth showing: hold released, waiting for clear, skipped tick, escalation."""
    judgements: list[dict] = field(default_factory=list)
    """Tree edits and outcome verdicts during the episode (see `TreePolicy.timeline`)."""
    summary: dict = field(default_factory=dict)
    """Filled in at the end: termination, success, steps, tree stats, timing."""

    def to_dict(self) -> dict:
        return {"meta": self.meta, "frames": self.frames, "asks": self.asks,
                "decisions": self.decisions, "recoveries": self.recoveries, "notes": self.notes,
                "judgements": self.judgements, "summary": self.summary}

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeRecording":
        return cls(**{k: data.get(k, [] if k not in ("meta", "summary") else {})
                      for k in ("meta", "frames", "asks", "decisions", "recoveries", "notes",
                                "judgements", "summary")})


class EpisodeRecorder:
    """The handle `run_episode` writes to. Every method is cheap (appends a small dict)."""

    def __init__(self) -> None:
        self.recording = EpisodeRecording()
        self._last_route: list[list[float]] | None = None

    def scene(self, *, boundary, obstacles, inflated_obstacles, goal, start, initial_route,
              robot_radius: float, dynamic_obstacle_radius: float, ts: float, max_steps: int,
              **extra: Any) -> None:
        self.recording.meta.update(_r({
            "boundary": _points(boundary),
            "obstacles": [_points(o) for o in obstacles],
            "inflated_obstacles": [_points(o) for o in inflated_obstacles],
            "goal": list(goal), "start": list(start),
            "initial_route": _points(initial_route),
            "robot_radius": robot_radius, "dynamic_obstacle_radius": dynamic_obstacle_radius,
            "ts": ts, "max_steps": max_steps, **extra,
        }))

    def frame(self, *, step: int, position, heading: float, speed: float, angular_speed: float,
              dynamic_obstacles, reference, route_ahead, hold: str | None) -> None:
        self.recording.frames.append({
            "step": step, "x": round(float(position[0]), 3), "y": round(float(position[1]), 3),
            "heading": round(float(heading), 3), "v": round(float(speed), 3),
            "w": round(float(angular_speed), 3),
            "dyn": _points(dynamic_obstacles), "ref": _points(reference),
            # The route only changes on a replan; store it only when it differs from the last frame's.
            "route": _points(route_ahead) if self._route_changed(route_ahead) else None,
            "hold": hold,
        })

    def _route_changed(self, route) -> bool:
        current = _points(route)
        if self._last_route == current:
            return False
        self._last_route = current
        return True

    def ask(self, step: int) -> None:
        self.recording.asks.append({"step": step})

    def decision(self, *, report_step: int, acted_step: int, raw_response: str | None,
                 failure_mode: str | None, method: str, rationale: str, latency_s: float,
                 overridden: bool = False, expired: str | None = None,
                 signature: list[str] | None = None) -> None:
        source = decision_source(raw_response)
        self.recording.decisions.append({
            # The report's `tree.signature`, where the caller has one: what lets a counterfactual
            # label found later be filed under the situation the decision was made in.
            "signature": signature,
            "report_step": report_step, "acted_step": acted_step, "source": source,
            "rule": raw_response[len("tree:"):] if source == SOURCE_TREE else None,
            "failure_mode": failure_mode, "method": method, "rationale": rationale,
            "latency_s": round(float(latency_s), 3), "overridden": overridden,
            "expired": expired,  # why the order was dropped unexecuted, or None
        })

    def recovery(self, *, step: int, method: str, success: bool, detail: str) -> None:
        self.recording.recoveries.append({"step": step, "method": method, "success": success,
                                          "detail": detail})

    def note(self, step: int, text: str) -> None:
        self.recording.notes.append({"step": step, "text": text})

    def finish(self, **summary: Any) -> None:
        self.recording.summary.update(_r(summary))


class RunRecorder:
    """A directory of episode recordings plus `run.json`, the per-episode summary."""

    def __init__(self, directory: str, **run_meta: Any) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.run: dict = {"meta": _r(run_meta), "episodes": []}

    def new_episode(self) -> EpisodeRecorder:
        return EpisodeRecorder()

    def save_episode(self, index: int, recorder: EpisodeRecorder) -> str:
        path = os.path.join(self.directory, f"episode_{index:03d}.json")
        with open(path, "w") as f:
            json.dump(recorder.recording.to_dict(), f, separators=(",", ":"))
        self.run["episodes"].append({"index": index, "file": os.path.basename(path),
                                     **recorder.recording.summary})
        with open(os.path.join(self.directory, "run.json"), "w") as f:
            json.dump(self.run, f, indent=1)
        return path


def load_run(directory: str) -> tuple[dict, list[EpisodeRecording]]:
    """Read a run directory back: `(run.json contents, [episode recordings in order])`."""
    with open(os.path.join(directory, "run.json")) as f:
        run = json.load(f)
    episodes = []
    for entry in run["episodes"]:
        with open(os.path.join(directory, entry["file"])) as f:
            episodes.append(EpisodeRecording.from_dict(json.load(f)))
    return run, episodes
