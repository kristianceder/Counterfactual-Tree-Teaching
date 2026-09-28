"""The scripted rules: a deterministic decision-maker that applies the failure-mode tests and the
recovery pairings directly, instead of asking a model to.

Two uses. As a teacher (`--teacher scripted`) it exercises the whole pipeline in seconds with no
API key, and it is the "scripted rules" baseline. As a reference (`ReferenceShadow`) it scores the
decisions of whatever is answering the ticks, without changing them: on the scenario grid these are
the rules verified to be the ones that work, so agreement with them measures decision quality
where the episode's ending cannot (a WAIT into a running hold only restarts it, the breaker makes
the human call, and what does fail is often the MPC). It is also the fallback that answers ticks
after the changed decision in a counterfactual replay (`counterfactual.py`).
"""
from __future__ import annotations

from .llm import SituationAssessment
from .recovery import RecoveryMethod
from .types import FailureMode


class ScriptedTeacher:
    """Applies, in order: the rules for calling a human (`_human_rule`), the rules for a solver
    running down the controller watchdog (`_watchdog_rule`), and the failure-mode tests with their
    recovery pairings (`_rule`)."""

    name = "scripted"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.calls = 0

    def assess_situation(self, context, candidates=None, evidence=None,
                         stall_threshold=None) -> SituationAssessment:
        self.calls += 1
        mode, method, why = (self._human_rule(context) or self._watchdog_rule(context) or self._rule(context))
        return SituationAssessment(method=method, rationale=why,
                                   raw_response=f"scripted:{mode.value if mode else 'none'}",
                                   failure_mode=mode)

    @staticmethod
    def _human_rule(c) -> tuple[FailureMode | None, RecoveryMethod, str] | None:
        """REQUEST_HUMAN's two conditions, plus the two consequences of blockages that *only*
        waiting clears: a hold that has done its job is ended (RESUME_ROUTE) rather than sat out,
        and an obstacle that has merely paused is waited for rather than replanned against. `None`
        falls through."""
        def value(name, default=None):
            got = getattr(c, name, None)
            return default if got is None else got

        stuck = (FailureMode.STUCK if value("stopped_for_s", 0.0) >= 4.0 and not value("holding_position", False)
                 and value("goal_distance", 0.0) > 0.3 else None)
        if value("goal_reachable") is False:
            return (stuck, RecoveryMethod.REQUEST_HUMAN,
                    "A replan has already found that no route to the goal exists; nothing the robot can do changes that.")
        # What can be waited for is what the robot has actually come up against, not anything
        # anywhere along the route.
        gap = value("dynamic_obstacle_gap")
        blocked_dynamic = bool(value("dynamic_obstacle_blocking_path", False)) and (gap is None or gap <= 3.0)
        blocked_static = bool(value("static_obstacle_blocking_path", False))
        if value("holding_position", False):
            if (value("previous_recovery") == "WAIT" and not blocked_dynamic and not blocked_static
                    and value("recovery_hold_steps_remaining", 0) >= 10
                    and abs(value("route_heading_error_deg", 0.0)) < 30.0):
                return (None, RecoveryMethod.RESUME_ROUTE,
                        "The obstacle the robot was waiting for has gone; pick the route back up now.")
            return None
        if blocked_dynamic and not blocked_static:
            # Not conditioned on STUCK: between holds the robot presses the obstacle again, which
            # reads as a grinding solver rather than a stopped robot. It only has to be getting nowhere.
            waited = sum(1 for r in value("recoveries_this_episode", []) if str(r).startswith("WAIT@"))
            if (value("blocking_obstacle_stationary_s", 0.0) >= 30.0 and waited >= 1
                    and value("goal_progress_last_5s", 0.0) < 0.2):
                return (stuck, RecoveryMethod.REQUEST_HUMAN,
                        "The blocking obstacle has not moved for 30 s and waiting has been tried: it is not going to leave.")
            if stuck is not None:
                return (stuck, RecoveryMethod.WAIT,
                        "A dynamic obstacle holds the route and no replan can see it; wait for it to move on.")
        return None

    @staticmethod
    def _watchdog_rule(c) -> tuple[FailureMode | None, RecoveryMethod, str] | None:
        """SOLVER_DEADLINE_MISS as `report.deadline_miss_evidence` states it."""
        counted = getattr(c, "solver_deadline_misses_last_12s", None)
        missing = (getattr(c, "solver_deadline_miss_streak", 0) or 0) >= 5 or (counted is not None and counted >= 10)
        if getattr(c, "holding_position", False) or not missing:
            return None
        if getattr(c, "static_obstacle_blocking_path", False):
            # Pressed against a wall the route runs through, the robot creeps rather than stops, so
            # STUCK's four seconds may never start while the watchdog runs down.
            return (FailureMode.SOLVER_DEADLINE_MISS, RecoveryMethod.REPLAN_ROUTE,
                    "The solver is grinding on a reference through a static obstacle; only a new route removes it.")
        gap = getattr(c, "dynamic_obstacle_gap", None)
        if getattr(c, "dynamic_obstacle_blocking_path", False) and (gap is None or gap <= 3.0):
            return (FailureMode.SOLVER_DEADLINE_MISS, RecoveryMethod.WAIT,
                    "The solver is grinding on a reference through a dynamic obstacle; a hold takes the conflict away.")
        # Nothing in the way: the solver is slow on an open road and recovers as the robot drives
        # on, unless the watchdog is nearly out of room (`watchdog_margin`).
        margin, left = getattr(c, "watchdog_margin", None), getattr(c, "watchdog_steps_remaining", None)
        if margin is not None and left is not None and counted is not None and counted >= 10:
            if left <= margin:
                return (FailureMode.SOLVER_DEADLINE_MISS, RecoveryMethod.WAIT,
                        "Nothing blocks the route, but the watchdog is nearly out of room; a hold brings the count down.")
            if getattr(c, "dynamic_obstacle_blocking_path", False):
                return None  # an obstacle further down the route: `_rule`'s answer
            return (FailureMode.SOLVER_DEADLINE_MISS, RecoveryMethod.CONTINUE,
                    "Nothing blocks the route: the solver is only slow on an open road and recovers as the robot "
                    "drives on, and the watchdog still has room. A hold would cost five seconds for nothing.")
        return None

    @staticmethod
    def _rule(c) -> tuple[FailureMode | None, RecoveryMethod, str]:
        """The failure-mode tests in the order the evidence lists them: a recovery already running
        first, then what the robot is physically doing wrong, then the solver."""
        def value(name, default=None):
            got = getattr(c, name, None)
            return default if got is None else got

        if value("holding_position", False):
            return None, RecoveryMethod.CONTINUE, "A recovery is already holding the robot; let it finish."

        gaps = [g for g in (value("static_obstacle_gap"), value("dynamic_obstacle_gap")) if g is not None]
        if gaps and min(gaps) <= 0.0:
            return (FailureMode.COLLISION, RecoveryMethod.REPLAN_ROUTE,
                    "Footprint is touching something; route around it.")

        if value("stopped_for_s", 0.0) >= 4.0 and value("goal_distance", 0.0) > 0.3:
            if value("static_obstacle_blocking_path", False):
                return (FailureMode.STUCK, RecoveryMethod.REPLAN_ROUTE,
                        "Stopped against a static obstacle on the route: the map, not the moment, is wrong.")
            if value("dynamic_obstacle_blocking_path", False):
                if value("blocking_obstacle_stationary_s", 0.0) >= 2.0:
                    return (FailureMode.STUCK, RecoveryMethod.REPLAN_ROUTE,
                            "The blocking obstacle has stopped moving; waiting it out is not working.")
                return (FailureMode.STUCK, RecoveryMethod.WAIT,
                        "A moving obstacle is across the route; let it clear.")
            return (FailureMode.STUCK, RecoveryMethod.REPLAN_ROUTE,
                    "Stopped with nothing named as blocking; replan rather than sit there.")

        if value("reversing_for_s", 0.0) >= 3.0:
            return (FailureMode.REVERSE_TRACKING, RecoveryMethod.REORIENT,
                    "Driving the route backwards; turn to face along it.")

        if value("heading_reversals_last_5s", 0) >= 4 and value("goal_progress_last_5s", 1.0) < 0.2:
            return (FailureMode.OSCILLATION, RecoveryMethod.REPLAN_ROUTE,
                    "Moving but getting nowhere; take a different route.")

        if value("solver_deadline_miss_streak", 0) >= 5:
            return (FailureMode.SOLVER_DEADLINE_MISS, RecoveryMethod.CONTINUE,
                    "A timing fault, not a navigation one: no route change fixes it.")

        return None, RecoveryMethod.CONTINUE, "Nothing in the report meets a failure test."


class ReferenceShadow:
    """Wraps whatever answers the ticks and notes, without touching its answer, what the scripted
    rules would have said about the same report ("agrees" means the same ACTION).

    Per episode (`take()`): ticks, how many agreed, the same split by who answered (tree, teacher,
    breaker), recoveries ordered, recoveries the reference would not have ordered, and the
    commonest disagreements. Pure bookkeeping: the wrapped analyzer is called exactly as before.
    """

    def __init__(self, inner):
        self.inner = inner
        self.reference = ScriptedTeacher()
        self.rows: list[tuple[str, str, str]] = []   # (who answered, reference action, given action)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def assess_situation(self, context, candidates, evidence=None, stall_threshold=None):
        decision = self.inner.assess_situation(context, candidates, evidence=evidence, stall_threshold=stall_threshold)
        wanted = self.reference.assess_situation(context).method.value
        raw = str(getattr(decision, "raw_response", "") or "")
        who = "tree" if raw.startswith("tree:") else "breaker" if raw == "breaker" else "teacher"
        self.rows.append((who, wanted, decision.method.value))
        return decision

    def take(self) -> dict:
        """This episode's tally, and start the next."""
        rows, self.rows = self.rows, []
        by_source: dict[str, list[int]] = {}
        differs: dict[str, int] = {}
        for who, wanted, given in rows:
            cell = by_source.setdefault(who, [0, 0])
            cell[0] += wanted == given
            cell[1] += 1
            if wanted != given:
                differs[f"{wanted} -> {given}"] = differs.get(f"{wanted} -> {given}", 0) + 1
        ordered = [r for r in rows if r[2] != "CONTINUE"]
        return {"ticks": len(rows), "agree": sum(w == g for _, w, g in rows), "by_source": by_source,
                "recoveries_ordered": len(ordered),
                "recoveries_reference_would_not_order": sum(1 for _, w, g in ordered if w == "CONTINUE"),
                "recoveries_reference_wanted_but_missing": sum(1 for _, w, g in rows if w != "CONTINUE" and g == "CONTINUE"),
                "differs": dict(sorted(differs.items(), key=lambda kv: -kv[1])[:6])}
