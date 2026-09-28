"""Answer assessment ticks from a recording instead of a model, so an episode can be re-run
exactly -- and then re-run with one decision changed.

The layout and the moving obstacles repeat given the seed, and the teacher's verdicts land on
whichever control cycle its wall-clock latency put them on -- which is what this module puts back.
**The MPC does not repeat exactly.** `max_solver_time_micros` (500 ms, baked into the solver at
build time) caps a solve by the wall clock, so how far it has converged when it returns depends on
what else the machine was doing. A solve sitting near that budget can return a different control
from one run to the next, and the episode diverges from there. So the exactness check a replay
starts with is necessary but not sufficient, and every counterfactual label is approximate.
`recorder.py` stores every verdict
with the step its report was taken on and the step it was acted on, which is all that is needed
to put the same verdicts back on the same steps: `ReplayAnalyzer` looks each tick's report step
up in the recording and hands back the recorded verdict, tagged with the recorded latency in
control cycles so `AsyncFailureAnalyzer` releases it where the recording did.

That makes a counterfactual cheap. Override one decision -- typically a recovery replaced by
CONTINUE -- and the episode diverges from that step on, with no model loaded and no snapshot of
the solver to restore. What the robot does instead is the answer to "was that recovery needed?",
which the outcome judge in `tree.py` structurally cannot see (a robot that was fine all along
scores `good`). Ticks after the divergence have no recorded answer; they are answered by
`fallback` if one is given (a `ScriptedTeacher`, say) and nominally otherwise.

Stdlib-only, like `recorder.py` and `tree.py`. Used by `counterfactual.py`.
"""
from __future__ import annotations

from typing import Any

from .llm import SituationAssessment
from .recovery import RecoveryMethod
from .types import FailureMode

NOMINAL = "replay:nominal"
"""`raw_response` of a tick the recording has no answer for (after the divergence)."""


class ReplayAnalyzer:
    """Answers `run_episode`'s assessment ticks from a recording, as a teacher would.

    Args:
        recording: one `episode_NNN.json` as a dict (see `recorder.EpisodeRecording.to_dict`).
        override: `{report_step: RecoveryMethod}` -- verdicts to replace. From the first
            overridden step on, the episode is treated as diverged: later recorded verdicts are
            not replayed (they were drawn from a robot that is no longer where this one is).
        fallback: an object with `assess_situation(context, candidates, evidence=, stall_threshold=)`
            to answer diverged ticks; `None` answers them NOMINAL/CONTINUE.
    """

    def __init__(self, recording: dict[str, Any], override: dict[int, RecoveryMethod] | None = None,
                 fallback: Any = None):
        self.by_report_step: dict[int, dict] = {}
        for d in recording.get("decisions", []):
            # One verdict per report step; a report step never has two in a recording.
            self.by_report_step[int(d["report_step"])] = d
        self.override = {int(k): v for k, v in (override or {}).items()}
        self.fallback = fallback
        self.diverged_at: int | None = None
        """First step answered by something other than the recording, or `None` if it replayed
        to the end."""
        self.replayed = 0
        self.overridden = 0
        self.fallback_calls = 0
        self.answers: list[dict] = []
        """What each tick was answered with and from where, for the runner's report."""

    # -- the analyzer interface --------------------------------------------------------------

    def assess_situation(self, context, candidates=None, evidence=None, stall_threshold=None):
        step = int(getattr(context, "step", 0) or 0)
        recorded = self.by_report_step.get(step)
        if step in self.override:
            self.overridden += 1
            if self.diverged_at is None:
                self.diverged_at = step
            # Keep the recorded latency so the *timing* of the tick is unchanged and only its
            # content differs. A fresh tick would otherwise land on a different step, and the
            # comparison would be with a robot supervised on a different clock.
            latency = self._latency(recorded, step)
            method = self.override[step]
            decision = self._assessment(method, None, "replay:override", latency)
            self._log(step, "override", decision)
            return decision
        if recorded is not None and self.diverged_at is None:
            self.replayed += 1
            mode = recorded.get("failure_mode")
            failure_mode = FailureMode(mode.lower()) if mode else None
            decision = self._assessment(RecoveryMethod(recorded["method"]), failure_mode,
                                        self._raw(recorded), self._latency(recorded, step))
            self._log(step, "replay", decision)
            return decision
        if self.diverged_at is None:
            self.diverged_at = step
        if self.fallback is not None:
            self.fallback_calls += 1
            decision = self.fallback.assess_situation(context, candidates, evidence=evidence,
                                                      stall_threshold=stall_threshold)
            setattr(decision, "replay_latency_steps", 0)
            self._log(step, "fallback", decision)
            return decision
        decision = self._assessment(RecoveryMethod.CONTINUE, None, NOMINAL, 0)
        self._log(step, "nominal", decision)
        return decision

    # -- helpers -----------------------------------------------------------------------------

    @staticmethod
    def _latency(recorded: dict | None, step: int) -> int:
        if recorded is None:
            return 0
        return max(0, int(recorded["acted_step"]) - step)

    @staticmethod
    def _raw(recorded: dict) -> str:
        """Reproduce `recorder.decision_source`'s classification so a recording of the replay
        reads the same as the original."""
        source = recorded.get("source")
        if source == "breaker":
            return "breaker"
        if source == "tree":
            return "tree:" + (recorded.get("rule") or "replay")
        if source == "unanswered":
            return "tree:miss"
        return "replay:llm"

    @staticmethod
    def _assessment(method: RecoveryMethod, failure_mode: FailureMode | None, raw: str,
                    latency_steps: int) -> SituationAssessment:
        decision = SituationAssessment(method=method, rationale="", raw_response=raw,
                                       failure_mode=failure_mode)
        setattr(decision, "replay_latency_steps", latency_steps)
        return decision

    def _log(self, step: int, source: str, decision) -> None:
        mode = getattr(decision, "failure_mode", None)
        self.answers.append({"step": step, "source": source, "method": decision.method.value,
                             "failure_mode": mode.value if mode else None,
                             "latency_steps": getattr(decision, "replay_latency_steps", 0)})

    def summary(self) -> dict:
        return {"replayed": self.replayed, "overridden": self.overridden,
                "fallback_calls": self.fallback_calls, "diverged_at": self.diverged_at}
