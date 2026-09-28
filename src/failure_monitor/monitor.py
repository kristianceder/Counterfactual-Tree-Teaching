from __future__ import annotations

from .types import FailureEvent, FailureMode


class VerdictLog:
    """The failure verdicts made during an episode (by the teacher, the tree or the breaker),
    edge-triggered: the robot called STUCK on five consecutive ticks is one failure, not five.
    The history is reported back to the decision-maker as `prior_failures_this_episode`."""

    def __init__(self):
        self.history: list[FailureEvent] = []
        self._active: set[FailureMode] = set()

    def record(self, mode: FailureMode | None, step: int, message: str = "") -> FailureEvent | None:
        """Record one tick's verdict (`None` is nominal and clears whatever was active). Returns the
        new event if this verdict starts a failure, else `None`."""
        if mode is None:
            self._active.clear()
            return None
        if mode in self._active:
            return None
        # A verdict names exactly one mode, so a different one supersedes rather than adds to it.
        self._active = {mode}
        event = FailureEvent(mode, step, message or f"LLM assessed the robot as {mode.value}.")
        self.history.append(event)
        return event

    @property
    def prior(self) -> list[str]:
        """The history in the report's form, e.g. ["stuck@88"]."""
        return [f"{e.mode.value}@{e.step}" for e in self.history]

    def failure_modes(self) -> set[FailureMode]:
        return {event.mode for event in self.history}
