"""Runs the decision-maker's call off the control loop's thread, so asking the teacher does not stop
the robot.

`AsyncFailureAnalyzer` submits each assessment to a single worker thread and hands the loop an
`LLMReply` back at `report step + ceil(latency_s / ts)`: the control cycle a robot running at the
real control period would have received it on, measured from the call this run actually made. The
simulation itself is not paced to real time (an unrendered control cycle takes a few milliseconds
against the 200 ms it represents), so a reply taken the instant the thread returns would land
wherever that ratio happened to put it.

On each step with a call outstanding the wrapper compares robot time since the report against the
wall time the call has had. While the robot is less than one control period ahead there is still
simulation to overlap with the call; past that, simulating further would carry the robot beyond
the step the answer is due on, so the wrapper waits out the rest. The answer is therefore acted on
at most one control period after the step it was due on.

Only one call is ever in flight: a second report taken while the first is out would be answered
against a robot that has already moved on. The loop skips that tick and asks again at the next,
and `dropped` counts the skipped ticks.

A tree hit (`tree_analyzer.TreeAnalyzer`) goes through the same path and costs one control cycle; a
replayed verdict (`replay.ReplayAnalyzer`) carries its recorded latency (`replay_latency_steps`) so
it is released on the step the recording released it on.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import SituationAssessment
    from .report import SituationContext


@dataclass
class LLMRequest:
    """One assessment asked for: the report, the step it describes, and when the clock started."""
    step: int
    context: "SituationContext"
    stall: int | None
    """`recovery_cycles_without_progress` as the report showed it: the recovery-loop breaker judges
    the answer against the number the decision-maker was shown."""
    submitted_at: float = field(default_factory=time.perf_counter)


@dataclass
class LLMReply:
    """One assessment come back, and what it cost to get it."""
    decision: "SituationAssessment"
    request: LLMRequest
    latency_s: float
    """Wall-clock seconds the call took."""
    latency_steps: int
    """`latency_s` as a whole number of control cycles."""
    release_step: int
    """The step the reply becomes actionable on."""
    delivered_step: int = -1
    """The step it was actually acted on."""

    @property
    def age_steps(self) -> int:
        """Control cycles between the report being taken and its answer being acted on."""
        return self.delivered_step - self.request.step


class AsyncFailureAnalyzer:
    """Wraps an analyzer so one call at a time runs on a worker thread.

        llm = AsyncFailureAnalyzer(analyzer, ts=config.ts)
        for i in range(max_steps):
            ...
            reply = llm.poll(i)               # a verdict asked for some steps ago, or None
            if reply is not None:
                execute(reply.decision)
            if assess_due and not llm.busy:
                llm.submit(lambda: analyzer.assess_situation(context, ...),
                           step=i, context=context, stall=context.recovery_cycles_without_progress)
        llm.close()
    """

    def __init__(self, analyzer, ts: float):
        self.analyzer = analyzer
        self.ts = ts
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="failure-llm")
        self._future: Future | None = None
        self._request: LLMRequest | None = None
        self._held: LLMReply | None = None   # answered, waiting for its release step
        self.calls = 0
        self.total_llm_s = 0.0
        self.blocking_s = 0.0
        """Wall time the control loop waited for an answer."""
        self.latencies_s: list[float] = []
        self.ages_steps: list[int] = []
        self.dropped = 0
        """Ticks that wanted an assessment while one was still in flight."""
        self._skipped_this_call = 0

    @property
    def busy(self) -> bool:
        """Whether a call is outstanding: still running, or answered but not yet released."""
        return self._request is not None or self._held is not None

    def skip(self) -> bool:
        """Record that a tick wanted an assessment while `busy`. True the first time for the call
        currently in flight."""
        self.dropped += 1
        self._skipped_this_call += 1
        return self._skipped_this_call == 1

    def submit(self, call: Callable[[], "SituationAssessment"], *, step: int,
               context: "SituationContext", stall: int | None) -> None:
        """Ask. `call` is a zero-argument thunk returning a `SituationAssessment`; the answer
        arrives later from `poll`."""
        if self.busy:
            raise RuntimeError("a call is already in flight -- check `busy` before submitting")
        self._request = LLMRequest(step=step, context=context, stall=stall)
        self._skipped_this_call = 0
        self._future = self._executor.submit(self._timed, call)

    def poll(self, step: int) -> LLMReply | None:
        """The reply due to be acted on at step `step`, or `None`. An exception raised in the
        worker surfaces here, on the control loop's thread."""
        if self._held is None and self._future is not None:
            request = self._request
            if not self._future.done():
                robot_elapsed = (step - request.step) * self.ts
                wall_elapsed = time.perf_counter() - request.submitted_at
                if robot_elapsed < wall_elapsed + self.ts:
                    return None
                blocked_at = time.perf_counter()
                self._future.result()
                self.blocking_s += time.perf_counter() - blocked_at
            decision, elapsed = self._future.result()
            self._future, self._request = None, None
            self._held = self._reply(request, decision, elapsed)
        if self._held is None or step < self._held.release_step:
            return None
        reply, self._held = self._held, None
        reply.delivered_step = step
        self.ages_steps.append(reply.age_steps)
        return reply

    def close(self) -> None:
        """Shut the worker down, waiting out a call still running. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._future = self._request = self._held = None

    @property
    def mean_latency_s(self) -> float:
        return sum(self.latencies_s) / len(self.latencies_s) if self.latencies_s else 0.0

    @property
    def mean_age_steps(self) -> float:
        return sum(self.ages_steps) / len(self.ages_steps) if self.ages_steps else 0.0

    @staticmethod
    def _timed(call: Callable[[], "SituationAssessment"]) -> tuple["SituationAssessment", float]:
        started = time.perf_counter()
        decision = call()
        return decision, time.perf_counter() - started

    def _reply(self, request: LLMRequest, decision: "SituationAssessment", elapsed: float) -> LLMReply:
        self.calls += 1
        self.total_llm_s += elapsed
        self.latencies_s.append(elapsed)
        latency_steps = math.ceil(elapsed / self.ts) if self.ts > 0 else 0
        replay_steps = getattr(decision, "replay_latency_steps", None)
        if replay_steps is not None:
            latency_steps = int(replay_steps)
        return LLMReply(decision=decision, request=request, latency_s=elapsed,
                        latency_steps=latency_steps, release_step=request.step + latency_steps)
