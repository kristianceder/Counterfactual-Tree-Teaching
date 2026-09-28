"""Adapter that lets a grown `tree.TreePolicy` answer the loop's assessment ticks, escalating to the
teacher only when the tree has nothing to say.

`episode.run_episode` asks whatever it is given as `analyzer` for an assessment each tick, and
`async_llm.AsyncFailureAnalyzer` delivers the answer at the control cycle a real robot would have
had it. `TreeAnalyzer` offers the same `assess_situation` as the teacher, so the loop runs on a
learned policy without knowing it. A tree hit returns in microseconds and is acted on the next
control cycle; an escalation costs exactly what a teacher call costs.

The policy is mutated by every tick (stats, pending judgements, new rules), so it wants one call at
a time, which `AsyncFailureAnalyzer` guarantees. While a teacher call is out, the loop skips
assessment ticks rather than queueing them, so the tree does not answer during an escalation
either: it is the absence of escalations that gets the robot decided upon every tick.
"""
from __future__ import annotations

import inspect
from typing import Any

from .llm import SituationAssessment, human_grounds
from .recovery import RecoveryMethod
from .tree import FEATURE_NAMES, Leaf, TreePolicy
from .types import FailureMode


class TreeAnalyzer:
    """Answers from a `TreePolicy` first and a teacher second.

    Args:
        policy: the tree to consult, and to grow with whatever the teacher says.
        teacher: what to escalate to -- `openai_teacher.OpenAITeacher`, `scripted.ScriptedTeacher`,
            or anything with the same `assess_situation`. `None` freezes the tree: misses are
            answered nominally and counted as `TreeStats.unanswered`, which is how a learned policy
            is evaluated on its own, with no model at all.
        verbose: print each escalation and each tree edit as it happens.
        learn: whether a teacher's answer is folded back into the tree.
        memory: an `experience.ExperienceMemory`, or `None`. With one, every escalation hands the
            teacher what has been learned so far as `assess_situation(experience=...)`. A teacher
            whose `assess_situation` does not take that argument (the scripted one) is not given it.
        on_teacher: called as `on_teacher(context, decision, experience)` after every teacher answer
            (the run's teacher log).
    """

    def __init__(self, policy: TreePolicy, teacher: Any | None = None, *, verbose: bool = False,
                 learn: bool = True, memory: Any | None = None, on_teacher: Any | None = None):
        self.policy = policy
        self.teacher = teacher
        self.verbose = verbose
        self.learn_enabled = learn
        self.memory = memory
        self.on_teacher = on_teacher
        self.last_experience: str | None = None
        self._teacher_takes_experience = False
        if teacher is not None and memory is not None:
            try:
                params = inspect.signature(teacher.assess_situation).parameters
                self._teacher_takes_experience = ("experience" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))
            except (TypeError, ValueError):
                pass

    def start_episode(self) -> None:
        self.policy.start_episode()

    def end_episode(self, termination: str, needs_human: bool | None = None) -> None:
        """Settle the judgements still pending on how the episode ended (`TreePolicy.end_episode`).
        `needs_human` is what a REQUEST_HUMAN is judged on."""
        self.policy.end_episode(termination, needs_human)

    def assess_situation(self, context, candidates: dict[RecoveryMethod, str],
                         evidence: dict[FailureMode, str] | None = None,
                         stall_threshold: int | None = None) -> SituationAssessment:
        """Both halves of the question, from the tree if it knows. In order of precedence:

        1. **The recovery-loop breaker**, once `recovery_cycles_without_progress` has reached
           `stall_threshold` (and no hold is running): REQUEST_HUMAN where the report meets that
           action's conditions, REPLAN_ROUTE otherwise -- the rule the teacher's prompt states and
           the loop enforces. Applied here, it spends neither a teacher call nor a tree rule on a
           tick whose answer is fixed, and keeps the tree from learning rules that only held
           because the robot was under a rule at the time.
        2. **The tree**, when a trusted rule covers the report.
        3. **The teacher**, whose answer is then folded in as a rule.
        """
        self.policy.observe(context)

        stall = getattr(context, "recovery_cycles_without_progress", None)
        # A hold still running is left to finish: the count reaches the threshold the moment the
        # third WAIT starts, and a replan three steps into it would cancel the hold.
        held = bool(getattr(context, "holding_position", False))
        if stall_threshold is not None and stall is not None and stall >= stall_threshold and not held:
            demanded = RecoveryMethod.REQUEST_HUMAN if human_grounds(context) else RecoveryMethod.REPLAN_ROUTE
            self.policy.note_breaker(context)
            return SituationAssessment(
                method=demanded,
                rationale=(f"Recovery-loop breaker: recovery_cycles_without_progress is {stall}, at "
                           f"or beyond the {stall_threshold}-cycle limit, so the answer is fixed "
                           f"({demanded.value}) regardless of policy."),
                raw_response="breaker",
                failure_mode=None,
            )

        leaf = self.policy.decide(context)
        if leaf is not None:
            return SituationAssessment(method=leaf.action, rationale=self._rationale(leaf),
                                       raw_response=self._raw(leaf), failure_mode=leaf.mode)

        if self.teacher is None:
            # Nothing to ask: a nominal reading and a no-op, the safe direction to be wrong in.
            self.policy.note_unanswered()
            if self.verbose:
                print(f"  [tree] no rule for step {getattr(context, 'step', '?')} and no teacher "
                      f"-- answering NOMINAL/CONTINUE")
            return SituationAssessment(method=RecoveryMethod.CONTINUE,
                                       rationale="No rule covers this report and no teacher is loaded.",
                                       raw_response="tree:miss", failure_mode=None)

        if self.verbose:
            print(f"  [tree] no rule for step {getattr(context, 'step', '?')} -- asking the teacher")
        extra = {}
        self.last_experience = None
        if self._teacher_takes_experience:
            self.last_experience = self.memory.for_report(context)
            extra["experience"] = self.last_experience
            if self.verbose:
                print("  [tree] experience given to the teacher:\n    "
                      + self.last_experience.replace("\n", "\n    "))
        decision = self.teacher.assess_situation(context, candidates, evidence=evidence,
                                                 stall_threshold=stall_threshold, **extra)
        if self.on_teacher is not None:
            self.on_teacher(context, decision, self.last_experience)
        self._learn(context, decision.method, decision.failure_mode)
        return decision

    def close(self) -> None:
        """Release the teacher, if it holds anything. Safe to call twice."""
        closer = getattr(self.teacher, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "TreeAnalyzer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _learn(self, context, method: RecoveryMethod, mode: FailureMode | None) -> None:
        if not self.learn_enabled:
            return
        leaf, what = self.policy.learn(context, method, mode)
        if self.verbose:
            verdict = mode.value.upper() if mode is not None else "NOMINAL"
            print(f"  [tree] {what}: {'/'.join(leaf.path)} -> {verdict}/{method.value}")

    @staticmethod
    def _rationale(leaf: Leaf) -> str:
        """Why the robot is about to do this, in the report's own field names, and where the rule
        was learned."""
        conditions = ", ".join(f"{name}={value}"
                               for name, value in zip(FEATURE_NAMES, leaf.path)) or "(any report)"
        return (f"Decision-tree rule [{conditions}], learned from the {leaf.source} at episode "
                f"{leaf.learned_episode} step {leaf.learned_step} "
                f"(support {leaf.support}, {leaf.good} ok / {leaf.bad} bad).")

    @staticmethod
    def _raw(leaf: Leaf) -> str:
        """Stands in for the model's raw reply, so a recording says which rule decided."""
        return "tree:" + "/".join(leaf.path)
