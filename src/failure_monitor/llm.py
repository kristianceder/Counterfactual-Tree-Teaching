"""The teacher's prompt, and how its answer is read back.

`FailureAnalyzer.assess_situation` asks the decision-maker both halves of the question on every
assessment tick -- is the robot in a failure state, and what should it do -- from a routine status
report (`report.SituationContext`) and the failure-mode evidence (`report.monitor_evidence`). The
model call itself is `complete()`, which a subclass implements: `openai_teacher.OpenAITeacher` sends
it to GPT-6 Luna over the OpenAI API.

The prompt is one long fixed instruction block, then the experience text (`experience.py`, when
the run shows the teacher what has been learned), then the report. `complete()` is handed the fixed
head and the changing tail separately, so a provider can cache the head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .recovery import RecoveryMethod
from .types import FailureMode

if TYPE_CHECKING:
    from .report import SituationContext


DEFAULT_MONITOR_SYSTEM_PROMPT = (
    "You are the online failure monitor for a mobile robot's autonomous trajectory planner (a "
    "receding-horizon MPC controller, optionally hybridized with a DRL policy). Once per second, "
    "on a fixed schedule, you are handed a JSON status report describing what the robot is doing "
    "right now. Nothing has flagged these reports -- there is no separate failure detector, you "
    "are it -- so most of them describe a robot that is perfectly fine, and calling a healthy "
    "robot broken is as much of a mistake as missing a real failure. Read the report against the "
    "evidence you are given for each failure mode, decide whether the robot is actually in one of "
    "them, and say what it should do next."
)
"""The system prompt. The robot's controller in this repository is the plain MPC; the text is kept as
the recorded runs' teacher read it."""

REPORT_MARK = "\n\nStatus report (JSON):\n"
"""What the report follows in the user prompt."""


EXPERIENCE_GUIDE = (
    "You have handled this robot before. Below, ahead of the status report, is what came of it. "
    "Use it: the instructions above are general, this is what actually happened on this robot.\n"
    "- RULES YOU HAVE ALREADY TAUGHT are your own earlier answers, which the robot's decision tree "
    "now applies without asking you. Each says how often it was used and how it was judged "
    "afterwards: 'good' means the robot got going again after the action, 'bad' means it did not "
    "(or, for CONTINUE, that the report already met a failure test and nothing was done). A RETIRED "
    "rule was judged bad repeatedly: you are being asked again because that answer did not work, so "
    "do not give it again unless the report differs in a way that matters. Stay consistent with "
    "rules that were judged good.\n"
    "- WHAT HAPPENED IN SIMILAR SITUATIONS scores actions in situations like this one. 'judged "
    "good/bad' is the same judgement as above. 'replayed without it' is a simulation of the same "
    "episode with that one action left out: 'needed' means the episode went worse without it, "
    "'unnecessary' that it ended just as well without it (the action only cost time), 'harmful' "
    "that it ended better without it. An action that is mostly 'needed' and judged good is the one "
    "to repeat; one that is mostly 'unnecessary' or 'harmful' should make you lean towards "
    "CONTINUE or a different action.\n"
    "- Situations are written in the decision tree's bins: hold = holding_position; stopped = "
    "stopped_for_s (no: under 1 s, brief: under 4 s, long: 4 s or more); blocked = static if "
    "static_obstacle_blocking_path, else dynamic if dynamic_obstacle_blocking_path, else clear; "
    "route = none once goal_reachable is false; contact = touch / near / clear from the obstacle "
    "gaps; reversing = reversing_for_s (brief: 1 s or more, long: 3 s or more); heading = "
    "route_heading_error_deg (on: under 30, off: under 90, away: 90 or more); blocker = "
    "blocking_obstacle_stationary_s (moving: under 2 s, parked: under 30 s, abandoned: 30 s or "
    "more); deadline = miss when the solver deadline test is met.\n"
    "- How to weigh it. Find the recorded situation that matches the current report on the features "
    "that matter (a difference marked 'has changed the right answer before' is NOT a match; a "
    "difference in heading or contact usually is). If an action there was judged good or 'needed' "
    "two or more times and never bad or 'harmful', take that action -- even where the general "
    "pairing in the instructions above would suggest another one, because those pairings are rules "
    "of thumb and this is what worked on this robot in this situation. If the record is thin (one "
    "case), mixed, or there is no matching situation, judge the report on the instructions above."
)
"""How to read the text `experience.ExperienceMemory` produces. Only in the prompt when that text
is (`assess_situation(experience=...)`)."""


def human_grounds(context) -> bool:
    """Whether a report meets one of REQUEST_HUMAN's two conditions as its description states them.
    Only used to word the recovery-loop hard rule, which is already a rule rather than a judgement."""
    if getattr(context, "goal_reachable", None) is False:
        return True
    waited = any(str(r).startswith("WAIT@") for r in (getattr(context, "recoveries_this_episode", None) or []))
    return waited and (getattr(context, "blocking_obstacle_stationary_s", None) or 0.0) >= 30.0


def split_prompt(user: str, experience: str | None) -> tuple[str, str]:
    """The head every call of a run shares, and the tail that changes (the experience text, then
    the report). A call where the recovery-loop hard rule is spliced in gets a head of its own.
    The head's trailing blank line moves to the tail, so `head + tail == user`."""
    cut = user.rindex(REPORT_MARK)
    if experience is not None:
        assert user[cut - len(experience):cut] == experience, "experience text is not where assess_situation puts it"
        cut -= len(experience)
    head = user[:cut].rstrip()
    return head, user[len(head):]


@dataclass
class LLMConfig:
    """What `FailureAnalyzer` needs to know about the model it calls."""
    model_name: str = "gpt-6-luna"
    temperature: float = 0.0
    max_output_tokens: int = 64
    """The two answer lines are ~15 tokens."""
    system_prompt: str = DEFAULT_MONITOR_SYSTEM_PROMPT


@dataclass
class SituationAssessment:
    """One answer: the recovery to execute, and the failure mode diagnosed (`None` is nominal, the
    expected answer on most ticks). Reported as given, even when the two disagree."""
    method: RecoveryMethod
    rationale: str
    raw_response: str
    failure_mode: FailureMode | None = None

    @property
    def failure_message(self) -> str:
        if self.failure_mode is None:
            return "Nominal: no failure detected."
        return self.rationale or f"LLM assessed the robot as {self.failure_mode.value}."


class FailureAnalyzer:
    """Builds the teacher's prompt and parses its answer; `complete()` is the model call.

    Stateless apart from `last_prompt_tokens`/`last_cached_tokens`, which a subclass may set.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_prompt_tokens: int | None = None
        self.last_cached_tokens: int | None = None

    def complete(self, system_prompt: str, head: str, tail: str) -> str:
        """One completion for the user prompt `head + tail` (see `split_prompt`)."""
        raise NotImplementedError

    def assess_situation(self, context: "SituationContext", candidates: dict[RecoveryMethod, str],
                         evidence: dict[FailureMode, str], stall_threshold: int | None = None,
                         experience: str | None = None) -> SituationAssessment:
        """Ask whether the robot is in a failure state and what it should do now.

        Args:
            candidates: the recoveries to choose between (`recovery.recovery_candidates()`).
            evidence: the failure modes and their evidence (`report.monitor_evidence()`).
            stall_threshold: once `recovery_cycles_without_progress` reaches it, a hard rule is
                added to the prompt: REPLAN_ROUTE, or REQUEST_HUMAN where the report meets that
                action's conditions. The caller's recovery-loop breaker enforces the same rule.
            experience: what has been learned, rendered by `experience.ExperienceMemory`, or None.

        Falls back to a nominal CONTINUE if the reply names neither a mode nor an action.
        """
        judging = (
            "Every mode above opens with its decisive test. Check that test against the report's "
            "actual numbers before naming a mode: if the test is not met, that mode is not the "
            "answer however alarming the other fields look, and if no test is met the answer is "
            "NONE. Where more than one test is met, name the one describing what the robot is "
            "physically doing wrong. A mode you named a moment ago has to pass its test again "
            "now -- judge the report in front of you.")
        hold = (
            "holding_position: true means a WAIT or REORIENT is deliberately keeping the robot "
            "still, recovery_hold_steps_remaining says how much of it is left, and "
            "recoveries_this_episode lists what has been tried already. A hold in progress is not "
            "a failure. Check whether the one already running is working before ordering another: "
            "a negative route_heading_error_change_5s means a REORIENT is turning the robot onto "
            "its route, and a blocking obstacle that has started moving again means a WAIT has "
            "done its job. Re-issuing a recovery that is already working restarts its hold from "
            "the beginning and spends more of the step budget to repeat what is already "
            "happening -- answer CONTINUE and let it finish.")
        modes = "\n".join(f"- {mode.value.upper()} -- {desc}" for mode, desc in evidence.items())
        mode_names = ", ".join(mode.value.upper() for mode in evidence)
        options = "\n".join(f"- {method.value}: {desc}" for method, desc in candidates.items())
        method_names = ", ".join(method.value for method in candidates)

        hard_rule = ""
        stall = context.recovery_cycles_without_progress
        if stall_threshold is not None and stall is not None and stall >= stall_threshold:
            if human_grounds(context):
                # Offered merely as an exception, REQUEST_HUMAN lost to the rule it was the
                # exception to, and the replan cancelled the hold the robot was under.
                hard_rule = (
                    f"\nHard rule, overriding everything else above: recovery_cycles_without_progress "
                    f"is {stall}, at or beyond the {stall_threshold}-cycle limit, and the report meets "
                    f"REQUEST_HUMAN's conditions (goal_reachable: false, or a blocking obstacle that "
                    f"has not moved for 30 s or more with WAIT already tried). Nothing the robot can "
                    f"do by itself is going to work. Your ACTION MUST be REQUEST_HUMAN.\n")
            else:
                hard_rule = (
                    f"\nHard rule, overriding everything else above: recovery_cycles_without_progress is "
                    f"{stall}, at or beyond the {stall_threshold}-cycle limit. Whatever has been tried "
                    f"is not working. Your ACTION MUST be REPLAN_ROUTE this time -- even if the report "
                    f"otherwise looks clear, or a dynamic obstacle still appears to be blocking the path.\n"
                    "The single exception is REQUEST_HUMAN, and only if the report meets one of the "
                    "two conditions in its description (goal_reachable: false, or a blocking obstacle "
                    "that has not moved for 30 s or more with WAIT already tried).\n")
        margin = context.watchdog_margin
        human = (
            "Then check whether the robot can still finish by itself. goal_reachable: false means "
            "a REPLAN_ROUTE has already found that no route to the goal exists; it is a fact about "
            "permanent obstacles and nothing the robot does changes it, so the ACTION is "
            "REQUEST_HUMAN, not another REPLAN_ROUTE. If the field is absent, a route exists as "
            "far as anyone knows. Likewise blocking_obstacle_stationary_s of 30 or more with a WAIT "
            "already in recoveries_this_episode: the thing in the way has been left there, and the "
            "ACTION is REQUEST_HUMAN, not another WAIT. Below 30 it has only paused, and "
            "REQUEST_HUMAN would be a false alarm.\n\n"
            "Then check the controller watchdog. If solver_deadline_misses_last_12s is 10 or more "
            "and holding_position is false, the robot is on its way to being stopped for good, "
            "whatever else the report says: FAILURE is SOLVER_DEADLINE_MISS, and the ACTION is WAIT "
            "if dynamic_obstacle_blocking_path is true, REPLAN_ROUTE if static_obstacle_blocking_path "
            "is true. "
            "With one of those flags true CONTINUE is wrong for that report. With NEITHER true the "
            "road is open and the solver is only slow: the ACTION is CONTINUE while "
            f"watchdog_steps_remaining is above {margin}, and WAIT at "
            f"{margin} or below."
            " (REQUEST_HUMAN's own conditions, where that action is offered, still come first.)\n\n")
        instruction = (
            f"Reply with exactly two lines and nothing else, no explanation:\n"
            f"FAILURE: <NONE, or one of {mode_names}>\n"
            f"ACTION: <one of {method_names}>"
        )
        memory = ""
        if experience is not None:
            # The static guide stays in the cached head; the experience text changes only when the
            # tree or the table does; the report changes every call.
            guide = EXPERIENCE_GUIDE.replace(
                "deadline = miss when the solver deadline test is met.",
                "deadline = miss when the solver deadline test is met, critical when "
                f"watchdog_steps_remaining is also {margin} or less.")
            memory = f"\n\n{guide}\n\n{experience}"
        prompt = (
            f"Assess the robot's situation from the status report below. It was taken on a fixed "
            f"schedule, not because anything went wrong, so start from the assumption that the robot "
            f"is fine and only depart from it if the fields say otherwise.\n\n"
            f"Step 1 -- is the robot in a failure state? Answer NONE if it is making normal progress "
            f"toward its goal, or name exactly one of these:\n{modes}\n"
            f"{judging}\n\n"
            f"Before either step, check whether a recovery you ordered earlier is still running. "
            f"{hold}\n\n"
            f"{human}"
            f"Step 2 -- what should the robot do now? Choose exactly one:\n{options}\n"
            f"Answer CONTINUE whenever the robot is fine, or when the problem is real but none of the "
            f"other methods addresses it -- CONTINUE is the correct answer for most reports.\n"
            f"{hard_rule}\n"
            f"{instruction}"
            f"{memory}"
        )
        user = f"{prompt}{REPORT_MARK}{context.to_json()}"
        head, tail = split_prompt(user, experience)
        raw = self.complete(self.config.system_prompt, head, tail)

        failure_mode = self._parse_failure_mode(self._answer_line(raw, "FAILURE"), evidence)
        method = self._parse_method(self._answer_line(raw, "ACTION") or raw, candidates)
        return SituationAssessment(method=method, rationale="", raw_response=raw, failure_mode=failure_mode)

    @staticmethod
    def _answer_line(raw: str, prefix: str) -> str | None:
        """The text after `PREFIX:` on the line that starts with it, or `None` if the model didn't
        use the requested format. Case- and punctuation-tolerant (a small model likes to bold the
        prefix or wrap the answer in quotes) but deliberately not a free-text search -- that's the
        caller's fallback to make, and it differs per field."""
        for line in raw.splitlines():
            stripped = line.strip().lstrip("*# -")
            if stripped.upper().startswith(prefix.upper()):
                # Original casing, not the uppercased form used to match: this same text is the
                # model's written justification when the WHY line is asked for.
                return stripped[len(prefix):].strip(" :\t-*\"'")
        return None

    @staticmethod
    def _parse_failure_mode(text: str | None, evidence: dict["FailureMode", str]) -> "FailureMode | None":
        """The `FailureMode` named in `text`, or `None` for a nominal assessment.

        `None` is also what an unparseable answer maps to, which is the safe direction to be wrong
        in: it means the caller takes the model at "nothing is flagged" and keeps monitoring, rather
        than inventing a failure episode out of a malformed reply."""
        if not text:
            return None
        upper = text.upper()
        if "NONE" in upper:
            return None
        # Longest names first, so one mode name contained in another cannot shadow it.
        for mode in sorted(evidence, key=lambda m: len(m.value), reverse=True):
            if mode.value.upper() in upper:
                return mode
        return None

    @staticmethod
    def _parse_method(raw: str, candidates: dict[RecoveryMethod, str]) -> RecoveryMethod:
        lines = raw.strip().splitlines()
        first_line = lines[0].upper() if lines else ""
        # Longest names first, so one method name contained in another cannot shadow it.
        ordered = sorted(candidates, key=lambda m: len(m.value), reverse=True)
        for method in ordered:
            if method.value in first_line:
                return method
        # Fall back to a full-text search, in case the model buried the answer
        # (e.g. behind a "thinking" preamble it wasn't asked to suppress).
        upper = raw.upper()
        for method in ordered:
            if method.value in upper:
                return method
        return RecoveryMethod.CONTINUE
