r"""A decision tree grown from the teacher's own answers, so a situation the teacher has already
ruled on is answered without asking it again.

A `TreePolicy` answers an assessment tick from a tree of learned rules when it recognises the
situation, and escalates to the teacher when it does not -- inserting the teacher's answer as a new
rule on the way back, so the next occurrence is free. The teacher is consulted about novel
situations; the tree is the controller, and it is what the robot actually runs on.

    tick -> signature(report) -> tree lookup --hit--> action          (microseconds)
                                            \--miss--> teacher -> action, and insert a rule

Three choices make this learning rather than caching:

1. **What counts as "the same situation"** -- `signature()` discretises a `SituationContext`
   into a tuple of bins, on the same thresholds `report.FAILURE_MODE_EVIDENCE` states in prose.
   Continuous reports never repeat exactly; binned ones do.
2. **How a rule generalises** -- a new rule is stored against the first `init_depth` features
   only, and is deepened *only when it is contradicted* (`DecisionTree.learn`), so the tree grows
   toward exactly the distinctions the teacher turned out to care about.
3. **What happens when a rule is wrong** -- a rule that orders a recovery is judged a few
   seconds later on whether the robot got going again (`TreePolicy.observe`), and a rule that
   answers "nothing is wrong" about a report already meeting a failure test is marked wrong on the
   spot (`TreePolicy.decide`). A chain of WAITs is one wait, judged where it ends (`_start_scoring`,
   `note_breaker`), and a REQUEST_HUMAN ends the episode, so it is judged on whether a human was
   needed (`end_episode`). Counterfactual replay labels are filed on the rules too
   (`note_counterfactual`). A rule that keeps failing is retired, which sends the next matching
   tick back to the teacher.

Dependency-free on purpose: stdlib plus this package's two enums. Reports are read field by field
with `getattr`, so the tree can be exercised against hand-built reports with no MPC, no map and no
model (`scripts/self_test.py`). `tree_analyzer.py` is the adapter that plugs it into the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from typing import Any, Callable, Iterable

from .recovery import RecoveryMethod
from .types import FailureMode


# -- thresholds ---------------------------------------------------------------------------
# Every number here is one the prompt already commits to: the `test:` clause of the matching
# `report.FAILURE_MODE_EVIDENCE` entry, which states it in prose. Change both together.
STUCK_STOPPED_S = 4.0          # FailureMode.STUCK: stopped_for_s >= 4, holding_position false
BRIEF_STOP_S = 1.0             # below this the robot is simply driving; not a distinction any test makes
REVERSING_S = 3.0              # FailureMode.REVERSE_TRACKING: reversing_for_s >= 3
BRIEF_REVERSE_S = 1.0          # as BRIEF_STOP_S: separates "a reversing manoeuvre" from "noise"
OSCILLATION_REVERSALS = 4      # FailureMode.OSCILLATION: heading_reversals_last_5s >= 4
PROGRESS_M = 0.2               # FailureMode.OSCILLATION: goal_progress_last_5s < 0.2 is "getting nowhere"
CONTACT_M = 0.0                # FailureMode.COLLISION: static/dynamic gap <= 0 is contact
NEAR_MISS_M = 0.5              # `contact=near`: a dynamic obstacle's footprint within 0.5 m
CLOSING_MPS = 0.1              # `approach=yes` (the report carries no closing speed here, so it reads `?`)
DEADLINE_STREAK = 5            # FailureMode.SOLVER_DEADLINE_MISS: solver_deadline_miss_streak >= 5
WATCHDOG_EVIDENCE_MISSES = 10  # ... or solver_deadline_misses_last_12s >= 10 (report.WATCHDOG_EVIDENCE_MISSES)
STEPS_PER_M = 5                # FailureMode.TIMEOUT: roughly 1 m of progress per 5 control steps
PARKED_OBSTACLE_S = 2.0        # blocking_obstacle_stationary_s past which waiting it out stops being the answer
ABANDONED_OBSTACLE_S = 30.0    # ... and past which it is not coming back to life: REQUEST_HUMAN's second condition
ON_ROUTE_DEG = 30.0            # route_heading_error_deg below which the robot is pointed along its route
OFF_ROUTE_DEG = 90.0           # ... and above which it is pointed away from it rather than merely off it

# Not a failure test, and not a feature: the one number `TreePolicy._judge` scores a recovery on.
RECOVERED_PROGRESS_M = 1.0     # route_progress_m gained over one judgement window that counts as "got going again";
                               # the same 1 m `SituationBuilder.progress_stall_tol` calls real progress

UNKNOWN = "?"
"""Bin for a field the report did not carry.

Reports are sparse by construction: `SituationContext.to_dict` drops `None`s, and every rolling
statistic is absent until the motion window has filled. A missing field is therefore ordinary, and it is its own bin rather
than being folded into the nearest real one -- a rule learned about a robot whose
`stopped_for_s` was genuinely 0 should not fire on one whose report never said."""


# -- features -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """One column of a signature: a name, and how to read a report into one of a few bins.

    `bins` is documentation, not logic -- `of()` decides -- but it is written out because it is
    what `DecisionTree.render` prints and what a stored tree carries in its header so an old file
    cannot be loaded against a changed feature set (see `TreePolicy.load`).
    """
    name: str
    bins: tuple[str, ...]
    of: Callable[[Any], str]

    def __call__(self, context: Any) -> str:
        return self.of(context)


def _num(context: Any, field_name: str) -> float | None:
    """A numeric report field, or `None` if absent. Tolerates a `bool` (Python's `bool` is an
    `int`, and `holding_position` is the one field where that would silently pass)."""
    value = getattr(context, field_name, None)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _band(value: float | None, edges: Iterable[tuple[float, str]], above: str) -> str:
    """`value` placed in the first band whose upper edge it falls under, else `above`."""
    if value is None:
        return UNKNOWN
    for edge, name in edges:
        if value < edge:
            return name
    return above


def _holding(context: Any) -> str:
    held = getattr(context, "holding_position", None)
    return UNKNOWN if held is None else ("yes" if held else "no")


def _stopped(context: Any) -> str:
    return _band(_num(context, "stopped_for_s"),
                 [(BRIEF_STOP_S, "no"), (STUCK_STOPPED_S, "brief")], "long")


def _blocked(context: Any) -> str:
    """What, if anything, the report says is sitting on the robot's route.

    Static wins over dynamic when both are set: they call for opposite recoveries (a wall is
    replanned around, a pedestrian is waited out), and a static obstacle is the one that will
    still be there in ten seconds.
    """
    static = getattr(context, "static_obstacle_blocking_path", None)
    dynamic = getattr(context, "dynamic_obstacle_blocking_path", None)
    if static:
        return "static"
    if dynamic:
        return "dynamic"
    if static is None and dynamic is None:
        return UNKNOWN
    return "clear"


def _route(context: Any) -> str:
    """Whether a route to the goal exists at all, as far as the report knows.

    `none` only once a REPLAN_ROUTE has searched the static map and come back empty
    (`SituationContext.goal_reachable: false`); everything else is `ok`, including a report from a
    caller that never sets the field. It is what separates the two reports a robot at a wall
    produces -- "stopped, static obstacle on the route" before the replan and the very same line
    after a replan that found nothing -- which call for REPLAN_ROUTE and REQUEST_HUMAN
    respectively. Without it the second report matches the first one's rule and the tree replans
    against the same map until the step budget runs out.
    """
    return "none" if getattr(context, "goal_reachable", None) is False else "ok"


def _contact(context: Any) -> str:
    """`touch` if either footprint-edge gap is at or below 0 (COLLISION), `near` if the *dynamic*
    gap is under `NEAR_MISS_M`, else `clear`.

    Near is dynamic-only on purpose. Static gaps of 0.4-0.9 m are routine on the grid (the robot
    drives alongside blocks), no failure test reads a small static gap, and a bin that lit up for
    every block the robot passed would split the tree on a distinction the teacher is never told
    to make."""
    static, dynamic = _num(context, "static_obstacle_gap"), _num(context, "dynamic_obstacle_gap")
    if static is None and dynamic is None:
        return UNKNOWN
    if min(g for g in (static, dynamic) if g is not None) <= CONTACT_M:
        return "touch"
    if dynamic is not None and dynamic < NEAR_MISS_M:
        return "near"
    return "clear"


def _closing(context: Any) -> str:
    """Whether the nearest dynamic obstacle is coming at the robot. The report in this repository
    carries no `dynamic_obstacle_closing_speed`, so this always reads `?`; the column is kept so the
    feature header, and every stored tree, stays the same."""
    speed = _num(context, "dynamic_obstacle_closing_speed")
    if speed is None:
        return UNKNOWN
    return "yes" if speed >= CLOSING_MPS else "no"


def _reversing(context: Any) -> str:
    return _band(_num(context, "reversing_for_s"),
                 [(BRIEF_REVERSE_S, "no"), (REVERSING_S, "brief")], "long")


def _wobble(context: Any) -> str:
    reversals = _num(context, "heading_reversals_last_5s")
    if reversals is None:
        return UNKNOWN
    return "yes" if reversals >= OSCILLATION_REVERSALS else "no"


def _progress(context: Any) -> str:
    return _band(_num(context, "goal_progress_last_5s"),
                 [(0.0, "back"), (PROGRESS_M, "stalled")], "forward")


def _heading(context: Any) -> str:
    error = _num(context, "route_heading_error_deg")
    return _band(None if error is None else abs(error),
                 [(ON_ROUTE_DEG, "on"), (OFF_ROUTE_DEG, "off")], "away")


def _blocker(context: Any) -> str:
    """Whether the thing in the way is going somewhere.

    A dynamic obstacle that has been stationary for a while is, for recovery purposes, a wall:
    WAIT is the right answer to a pedestrian crossing and the wrong one to a parked trolley, and
    nothing else in the report distinguishes them.
    """
    stationary = _num(context, "blocking_obstacle_stationary_s")
    if stationary is None:
        return UNKNOWN
    return _band(stationary, [(PARKED_OBSTACLE_S, "moving"), (ABANDONED_OBSTACLE_S, "parked")], "abandoned")


def _deadline(context: Any) -> str:
    streak = _num(context, "solver_deadline_miss_streak")
    if streak is None:
        return UNKNOWN
    # The watchdog counts misses over a window, so the count can climb while the streak keeps being
    # broken by the odd fast solve.
    misses = _num(context, "solver_deadline_misses_last_12s")
    counted = misses is not None and misses >= WATCHDOG_EVIDENCE_MISSES
    # "Missing, with room left" and "missing, nearly out of it" (`watchdog_margin`) are answered
    # differently on an open road, and only there: binned for every report, the split would divide
    # each WAIT-at-a-gate situation in two.
    margin, left = _num(context, "watchdog_margin"), _num(context, "watchdog_steps_remaining")
    if (counted and margin is not None and left is not None and left <= margin
            and _blocked(context) == "clear"):
        return "critical"
    return "miss" if streak >= DEADLINE_STREAK or counted else "ok"


def _budget(context: Any) -> str:
    """Whether the step budget still covers the distance left, at the evidence's rule of thumb
    of about 1 m per 5 control steps."""
    remaining = _num(context, "steps_remaining")
    distance = _num(context, "goal_distance")
    if remaining is None or distance is None:
        return UNKNOWN
    return "tight" if remaining < STEPS_PER_M * distance else "ok"


FEATURES: tuple[Feature, ...] = (
    Feature("hold", ("yes", "no", UNKNOWN), _holding),
    Feature("stopped", ("no", "brief", "long", UNKNOWN), _stopped),
    Feature("blocked", ("static", "dynamic", "clear", UNKNOWN), _blocked),
    # Straight after `blocked`, which it qualifies: the same wall is a REPLAN_ROUTE while a route
    # exists and a REQUEST_HUMAN once one does not. Placed late it would only be reached after a
    # contradiction had already split the rule on whatever unrelated feature happened to differ first.
    Feature("route", ("ok", "none"), _route),
    Feature("contact", ("touch", "near", "clear", UNKNOWN), _contact),
    # Always `?` here (see `_closing`); kept so the header, and every stored tree, is unchanged.
    Feature("approach", ("yes", "no", UNKNOWN), _closing),
    Feature("reversing", ("no", "brief", "long", UNKNOWN), _reversing),
    Feature("wobble", ("yes", "no", UNKNOWN), _wobble),
    Feature("progress", ("back", "stalled", "forward", UNKNOWN), _progress),
    Feature("heading", ("on", "off", "away", UNKNOWN), _heading),
    Feature("blocker", ("moving", "parked", "abandoned", UNKNOWN), _blocker),
    Feature("deadline", ("ok", "miss", "critical", UNKNOWN), _deadline),
    Feature("budget", ("ok", "tight", UNKNOWN), _budget),
)
"""The signature's columns, in the order the tree is allowed to split on them.

The order is the prompt's own checking order, not an information-gain ranking, and that is the
point: a tree that splits on whatever separates the training data best is unreadable next to the
instructions the teacher was given, and unreadable is the one thing this arrangement is supposed
not to be. So `hold` comes first because `assess_situation`'s prompt opens by telling the model
that a recovery already running is not a failure; `stopped` second because it is the decisive
test for STUCK; the physical modes before the solver ones, exactly as `FAILURE_MODE_EVIDENCE`
orders them.

Ordering matters to behaviour as well as readability. `init_depth` decides how many of these a
brand-new rule is conditioned on, and splitting only ever deepens: a feature early in this tuple is
one the tree will condition on *before it has any evidence that it matters*, one late in it is only
consulted after a contradiction proves it does. And because a rule's conditions are a *prefix* of
this order, `evidence_depth` forces any rule naming a mode down to the last feature that mode's test
reads -- so a mode evidenced late here produces specific rules that generalise little. `TIMEOUT` is
the extreme case: `budget` is last, so a rule naming it is conditioned on everything.
"""

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)


MODE_EVIDENCE_FEATURES: dict[FailureMode, tuple[str, ...]] = {
    FailureMode.STUCK: ("hold", "stopped"),
    FailureMode.COLLISION: ("contact",),
    FailureMode.REVERSE_TRACKING: ("reversing",),
    FailureMode.OSCILLATION: ("wobble", "progress"),
    FailureMode.SOLVER_DEADLINE_MISS: ("deadline",),
    FailureMode.TIMEOUT: ("budget",),
}
"""Which `FEATURES` carry the evidence for each failure mode, straight off its `test:` clause in
`FAILURE_MODE_EVIDENCE`.

A rule that names a mode has to be conditioned on the evidence for it, or it fires in situations
where that evidence is absent: a REORIENT rule stored above `reversing` reads "the robot is moving
and nothing is in the way -> turn it on the spot". And the outcome check cannot catch it, because a
needless recovery on a healthy robot looks exactly like a successful one.
"""


ACTION_EVIDENCE_FEATURES: dict[RecoveryMethod, tuple[str, ...]] = {
    RecoveryMethod.REQUEST_HUMAN: ("route",),
}
"""The same requirement for an action whose grounds are not any failure mode's test.

REQUEST_HUMAN is ordered on a robot that is STUCK, and STUCK's evidence ends at `stopped`: a rule
stored there reads "stopped for long -> call a human" and ends the next episode at the first
wall, where the answer is a replan. What makes it a human's problem is that no route exists, so
the rule has to sit at least as deep as `route`. Where the report it was learned on still has a
route (`route=ok`), its ground can only be the other one -- a blocker that has not moved for
`ABANDONED_OBSTACLE_S` -- and the rule is forced down to `blocker` as well (`evidence_depth`'s
`sig`). A false alarm costs the mission, so this is not left to refinement.
"""


def evidence_depth(mode: FailureMode | None, action: RecoveryMethod | None = None,
                   sig: tuple[str, ...] | None = None) -> int:
    """How deep a rule naming `mode` (and ordering `action`) has to sit for its evidence to be
    inside its own conditions.

    0 for a nominal verdict, which claims no mode and therefore needs no particular field to
    support it.
    """
    features = tuple(MODE_EVIDENCE_FEATURES.get(mode) or ()) if mode is not None else ()
    features += ACTION_EVIDENCE_FEATURES.get(action, ()) if action is not None else ()
    if (action is RecoveryMethod.REQUEST_HUMAN and sig is not None
            and sig[FEATURE_NAMES.index("route")] == "ok"):
        features += ("blocker",)
    if not features:
        return 0
    return max(FEATURE_NAMES.index(name) for name in features) + 1


CHECKABLE_MODES: tuple[FailureMode, ...] = (
    FailureMode.COLLISION, FailureMode.STUCK, FailureMode.REVERSE_TRACKING, FailureMode.OSCILLATION,
)
"""Modes whose `test:` clause can be evaluated against a report here, in priority order: the ones
that are unambiguous in the report's own fields and describe something the robot is physically
doing wrong. `TIMEOUT` is a budget projection rather than a present fact, and the solver mode is
judged on the prompt's own terms. Order matters only to `evidence_of_failure`, which names one.
"""


def meets_test(context: Any, mode: FailureMode) -> bool:
    """Whether this report satisfies `mode`'s decisive test, as `FAILURE_MODE_EVIDENCE` states it.

    `True` for any mode outside `CHECKABLE_MODES` -- not because the robot is in it, but because
    this report cannot say either way, and a caller asking the question should not be told "no" by
    something that never looked.
    """
    if mode is FailureMode.COLLISION:
        return _contact(context) == "touch"
    if mode is FailureMode.STUCK:
        stopped = _num(context, "stopped_for_s")
        goal_distance = _num(context, "goal_distance")
        return (not bool(getattr(context, "holding_position", False))
                and stopped is not None and stopped >= STUCK_STOPPED_S
                and (goal_distance is None or goal_distance > 0.3))
    if mode is FailureMode.REVERSE_TRACKING:
        return _reversing(context) == "long"
    if mode is FailureMode.OSCILLATION:
        reversals = _num(context, "heading_reversals_last_5s")
        progress = _num(context, "goal_progress_last_5s")
        return (reversals is not None and reversals >= OSCILLATION_REVERSALS
                and progress is not None and progress < PROGRESS_M)
    return True


def evidence_of_failure(context: Any) -> FailureMode | None:
    """The failure mode this report *already* evidences on its own decisive test, or `None`.

    This is not a detector and does not replace one -- nothing calls it to decide what the robot
    should do. It exists so `TreePolicy` can score a rule that answered "nothing is wrong"
    about a report that plainly said otherwise, which is the one mistake in that direction a policy
    can be held to without predicting the future.

    While a recovery hold is running, only `COLLISION` counts. `assess_situation`'s prompt tells
    the model a running recovery is not a failure, and the robot's motion during a WAIT or REORIENT
    hold is the recovery's doing rather than something to diagnose: a REORIENT turning on the spot
    can read as seconds of negative speed. `meets_test` stays literal to the `test:` clauses.
    Contact is never excused.
    """
    holding = bool(getattr(context, "holding_position", False))
    for mode in CHECKABLE_MODES:
        if holding and mode is not FailureMode.COLLISION:
            continue
        if meets_test(context, mode):
            return mode
    return None


def signature(context: Any) -> tuple[str, ...]:
    """The full discretised form of one report -- one bin per `FEATURES` entry, in order.

    Two reports with the same signature are "the same situation" as far as this policy is
    concerned. That is the whole modelling assumption, and the honest way to read it is as a
    resolution limit: everything this tuple cannot express, the tree cannot condition on, and a
    teacher that keeps giving different answers to one signature shows up as a contradiction
    rather than as silently unstable behaviour (see `DecisionTree.learn`).
    """
    return tuple(feature(context) for feature in FEATURES)


# -- the tree -----------------------------------------------------------------------------

@dataclass
class Leaf:
    """One learned rule: "a report whose signature starts with `path` gets `action`".

    The stats are the audit trail, and each is counted for a different reader. `support` is how
    many times a teacher independently gave this answer for a matching report -- evidence *for*
    the rule. `fires` is how often the robot has actually run on it. `good`/`bad` are what
    happened next, judged by `TreePolicy.observe`. `conflicts` counts teacher answers that
    contradicted it at full signature width, i.e. disagreements this feature set cannot explain.
    """
    path: tuple[str, ...]
    action: RecoveryMethod
    mode: FailureMode | None
    signature: tuple[str, ...]
    """The full signature of the report that most recently taught this leaf. Kept because
    refinement needs to know where the *old* rule sits in feature space, not just the prefix it
    was stored at -- see `DecisionTree.learn`."""
    source: str = "llm"                  # who taught it: "llm", or whatever a caller names its teacher
    learned_episode: int = 0
    learned_step: int = 0
    support: int = 1
    episodes: int = 1
    """Distinct episodes whose teacher answers gave this rule its current answer -- what
    `TreePolicy.min_support` counts. `support` alone counts answers, and answers from one episode
    are not independent: the next tick of the same approach, two seconds later, would confirm a
    rule the teacher answers differently in every later episode."""
    fires: int = 0
    good: int = 0
    bad: int = 0
    conflicts: int = 0
    unevidenced: bool = False
    """Whether the teacher named a mode this report's own decisive test does not support.

    Recorded, never overruled. The premise of the whole arrangement is that the model can rule on
    situations no threshold was written for, so a diagnosis the thresholds disagree with is not
    automatically wrong -- but it is the one thing about a learned rule worth knowing before
    trusting it, because the tree will repeat it indefinitely and silently.

    For example `hold=no/stopped=brief -> STUCK/REPLAN_ROUTE`: stuck after one to four seconds
    stopped, when the evidence states the test as four seconds. See `describe()`, which prints this,
    and `TreeStats.unevidenced`."""
    replacements: int = 0
    """How many times a retired rule at this exact path has been re-taught from scratch.

    A path that keeps being retired and re-taught is one where the outcome check and the teacher
    disagree and no feature separates the cases -- the same statement about `FEATURES` that
    `conflicts` makes, arrived at from the other direction. `TreePolicy.decide` stops trusting the
    path once this reaches `conflict_limit`, rather than letting it flip between answers forever."""
    retired: bool = False
    """Set once `bad` reaches `TreePolicy.retire_after`. A retired leaf is kept rather than
    deleted: it still answers `render()` and the stored file, so a run's record shows what was
    tried and dropped, but `TreePolicy.decide` treats it as a miss and sends the tick back to the
    teacher, whose answer replaces it."""
    cf_needed: int = 0
    cf_not_needed: int = 0
    """Counterfactual replays of recoveries this rule covers (`TreePolicy.note_counterfactual`):
    the episode re-run with that one recovery replaced by CONTINUE ended wrong (`needed`), or ended
    as well or sooner (`unnecessary`, `harmful`). The judge's `good`/`bad` cannot tell these apart:
    it asks whether the robot got going after the recovery, and a robot that was never in trouble
    always does."""
    note: str = ""

    @property
    def depth(self) -> int:
        return len(self.path)

    def describe(self) -> str:
        """One line, as `render()` and a decision's rationale print it."""
        verdict = self.mode.value.upper() if self.mode is not None else "NOMINAL"
        stats = (f"support {self.support} in {self.episodes} ep, fires {self.fires}, "
                 f"{self.good} ok / {self.bad} bad")
        if self.conflicts:
            stats += f", {self.conflicts} contradicted"
        if self.replacements:
            stats += f", re-taught {self.replacements}x"
        if self.unevidenced:
            stats += ", UNEVIDENCED"
        state = " RETIRED" if self.retired else ""
        return f"{verdict} -> {self.action.value} [{stats}]{state}"


class DecisionTree:
    """The learned rules, as a proper tree over `FEATURES` in order.

    Leaves are stored flat, keyed by the signature prefix they match, under one invariant: **no
    leaf's path is a proper prefix of another leaf's path**. That makes the flat dict a genuine
    tree -- the leaves are its frontier -- and lookup unambiguous: at most one stored path can be
    a prefix of any given signature. It also means the file on disk is a flat list of rules that
    reads top to bottom, rather than a nest of dictionaries.

    Growth is conflict-driven. A rule enters shallow (`init_depth` features) and is deepened only
    when a teacher contradicts it, at the first feature that actually separates the two reports.
    So the tree ends up conditioned on the distinctions that turned out to matter, and nothing
    else, without ever being told in advance which those are.
    """

    def __init__(self, init_depth: int = 2):
        if not 1 <= init_depth <= len(FEATURES):
            raise ValueError(f"init_depth must be in 1..{len(FEATURES)}, got {init_depth}")
        self.init_depth = init_depth
        self.leaves: dict[tuple[str, ...], Leaf] = {}

    # -- reading ---------------------------------------------------------------------------

    def match(self, sig: tuple[str, ...]) -> Leaf | None:
        """The one leaf governing `sig`, retired or not, or `None` if nothing covers it.

        Shortest path first only as an implementation detail -- the no-proper-prefix invariant
        means at most one can match, so the order is about stopping early, not about precedence.
        """
        for depth in range(1, len(sig) + 1):
            leaf = self.leaves.get(sig[:depth])
            if leaf is not None:
                return leaf
        return None

    def __len__(self) -> int:
        return len(self.leaves)

    @property
    def active(self) -> list[Leaf]:
        return [leaf for leaf in self.leaves.values() if not leaf.retired]

    # -- writing ---------------------------------------------------------------------------

    def learn(self, sig: tuple[str, ...], action: RecoveryMethod, mode: FailureMode | None,
              *, source: str = "llm", episode: int = 0, step: int = 0,
              unevidenced: bool = False) -> tuple[Leaf, str]:
        """Fold one teacher answer into the tree. Returns the governing leaf and what happened:
        `"insert"`, `"reinforce"`, `"refine"`, `"replace"` or `"contradict"`.

        The cases, in the order they are tested:

        - **No leaf matches** -- store a new one at `init_depth`, the shallowest the policy is
          willing to generalise from; deeper where the tree has already been split there (see
          `_insert_depth`), and deeper again if the mode it names needs it (`evidence_depth`), so a
          rule always carries the evidence for its own diagnosis inside its conditions.
          (`"insert"`)
        - **A matching leaf agrees** -- raise its support and re-date it. Nothing about the tree's
          shape changes; the rule simply has more evidence behind it. (`"reinforce"`)
        - **A matching leaf is retired** -- it has already been judged wrong, so the teacher was
          asked precisely to correct it. If this report differs from the one that rule was last
          taught on, deepen exactly as below, leaving the failed rule to keep escalating on its own
          branch (`"refine"`); if the two are identical, overwrite it in place, resetting the stats
          but counting the replacement and keeping a note of what it used to say. (`"replace"`)
        - **A matching leaf disagrees** -- deepen. Find the first feature on which the old rule's
          signature and this one differ, and replace the single leaf with two, both at that depth.
          Any third bin of that feature is now uncovered and will escalate, which is correct: the
          tree has just learned the feature matters, not what every value of it implies.
          (`"refine"`)
        - **... and they differ on no feature at all** -- the teacher has given two answers to one
          signature. There is no split that separates them, so the newer answer wins (it is the
          more recent evidence about a world that may have moved on) and `conflicts` records the
          disagreement. A leaf that keeps contradicting itself is a statement about this feature
          set, not about the teacher, and `TreePolicy` escalates it permanently once
          `conflict_limit` is reached. (`"contradict"`)
        """
        leaf = self.match(sig)
        if leaf is None:
            depth = min(max(self._insert_depth(sig), evidence_depth(mode, action, sig)), len(sig))
            new = Leaf(path=sig[:depth], action=action, mode=mode, signature=sig,
                       source=source, learned_episode=episode, learned_step=step,
                       unevidenced=unevidenced)
            self.leaves[new.path] = new
            return new, "insert"

        if leaf.retired:
            # The teacher was asked precisely because this rule had been judged wrong. Where the
            # report differs from the one the rule was last taught on, that difference is the more
            # useful thing to record: splitting leaves the failed rule governing its own corner
            # (still retired, so it keeps escalating) and gives the new answer its own, rather than
            # handing one over-general path back and forth between two answers forever. Replacing
            # in place is then reserved for what it actually means -- the same situation, a
            # different answer, because the last one did not work.
            split = _first_difference(leaf.signature, sig)
            if split is None:
                was = leaf.describe()
                # In place -- unless the new answer needs evidence the old path stops short of
                # (see `_deep_enough`), in which case it goes as deep as that evidence.
                path = self._deep_enough(leaf.path, sig, mode, action)
                new = Leaf(path=path, action=action, mode=mode, signature=sig, source=source,
                           learned_episode=episode, learned_step=step, unevidenced=unevidenced,
                           replacements=leaf.replacements + 1,
                           note=f"replaced a retired rule (was {was})")
                del self.leaves[leaf.path]
                self.leaves[path] = new
                return new, "replace"
            depth = min(max(split + 1, evidence_depth(mode, action, sig)), len(sig))
            del self.leaves[leaf.path]
            kept = replace(leaf, path=leaf.signature[:depth])
            new = Leaf(path=sig[:depth], action=action, mode=mode, signature=sig, source=source,
                       learned_episode=episode, learned_step=step, unevidenced=unevidenced,
                       note=f"split from a retired rule at {'/'.join(leaf.path)} "
                            f"on {FEATURE_NAMES[split]}")
            self.leaves[kept.path] = kept
            self.leaves[new.path] = new
            return new, "refine"

        if leaf.action is action and leaf.mode is mode:
            leaf.support += 1
            if episode != leaf.learned_episode:
                leaf.episodes += 1
            leaf.signature = sig
            leaf.learned_episode, leaf.learned_step = episode, step
            return leaf, "reinforce"

        split = _first_difference(leaf.signature, sig)
        if split is None:
            # Same discretised situation, different answer: nothing to split on.
            path = self._deep_enough(leaf.path, sig, mode, action)
            if path != leaf.path:
                del self.leaves[leaf.path]
                leaf.path = path
                self.leaves[path] = leaf
            leaf.conflicts += 1
            leaf.action, leaf.mode, leaf.unevidenced = action, mode, unevidenced
            # The support counted confirmations of the answer just overwritten, not this one.
            leaf.support = 1
            leaf.episodes = 1
            leaf.signature = sig
            leaf.learned_episode, leaf.learned_step = episode, step
            return leaf, "contradict"

        depth = min(max(split + 1, evidence_depth(mode, action, sig)), len(sig))
        del self.leaves[leaf.path]
        kept = replace(leaf, path=leaf.signature[:depth])
        new = Leaf(path=sig[:depth], action=action, mode=mode, signature=sig, source=source,
                   learned_episode=episode, learned_step=step, unevidenced=unevidenced,
                   note=f"split from {'/'.join(leaf.path)} on {FEATURE_NAMES[split]}")
        self.leaves[kept.path] = kept
        self.leaves[new.path] = new
        return new, "refine"

    # -- inspection ------------------------------------------------------------------------

    def _insert_depth(self, sig: tuple[str, ...]) -> int:
        """Where a new rule for `sig` can live: `init_depth`, or deeper where the tree has already
        been split.

        Inserting at a fixed depth is wrong wherever a refinement has already happened there. The
        shallow node a refinement leaves behind is an *internal* node, and a new leaf placed on it
        would be a proper prefix of the rules underneath -- breaking the invariant `match` relies
        on and shadowing every one of them, since lookup takes the shallowest path that fits.

        Otherwise an over-general rule retired and split one level down would be re-created at the
        original shallow path by the next unmatched report, hiding the refinement again.

        So a new rule descends as far as the existing structure goes and enters as a sibling at the
        point its signature falls off the tree. The tree's own shape is the record of how finely
        this region has already had to be carved, and a newcomer respects it.
        """
        depth = max(1, self.init_depth)
        while any(len(path) > depth and path[:depth] == sig[:depth] for path in self.leaves):
            depth += 1
        return min(depth, len(sig))

    def invariant_violations(self) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Pairs of stored paths where one is a proper prefix of the other, which must never
        happen: the shallower would shadow the deeper in `match`, silently disabling it.

        Exposed rather than asserted internally so a runner can assert on a real grown tree, and so a
        run against a hand-edited file can be sanity-checked.
        """
        paths = sorted(self.leaves)
        return [(shallow, deep) for i, shallow in enumerate(paths) for deep in paths[i + 1:]
                if len(deep) > len(shallow) and deep[:len(shallow)] == shallow]

    @staticmethod
    def _deep_enough(path: tuple[str, ...], sig: tuple[str, ...], mode: FailureMode | None,
                     action: RecoveryMethod) -> tuple[str, ...]:
        """`path`, or `sig` cut at the depth the answer's evidence needs if `path` is shorter.

        `insert` and `refine` respect `evidence_depth`, and so must the two branches that give an
        *existing* path a new answer (a retired rule replaced, a rule contradicted): a depth-3
        REPLAN_ROUTE rule re-answered REQUEST_HUMAN on a report whose `route` was `none` would
        otherwise keep its three conditions and call a human at every wall. Moving the rule deeper
        uncovers the rest of the old path, which is correct: what it said about those reports is
        exactly what has just been found wrong, so they go back to the teacher.
        """
        need = min(evidence_depth(mode, action, sig), len(sig))
        return path if len(path) >= need else tuple(sig[:need])

    def render(self) -> str:
        """The whole policy as an indented tree, one line per rule.

        This is the artefact the arrangement exists to produce: the robot's actual decision
        policy, in the report's own field names, on one screen. Print it after a run, diff it
        between runs, or set it beside the scripted rules.
        """
        if not self.leaves:
            return "(empty tree -- every tick escalates to the teacher)"
        lines: list[str] = []

        def walk(prefix: tuple[str, ...], indent: str) -> None:
            leaf = self.leaves.get(prefix)
            if leaf is not None:
                lines.append(f"{indent}{leaf.describe()}")
                return
            depth = len(prefix)
            if depth >= len(FEATURES):
                return
            children = sorted({path[depth] for path in self.leaves
                               if len(path) > depth and path[:depth] == prefix})
            for value in children:
                lines.append(f"{indent}{FEATURE_NAMES[depth]}={value}")
                walk(prefix + (value,), indent + "  ")

        walk((), "")
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "features": list(FEATURE_NAMES),
            "init_depth": self.init_depth,
            "rules": [
                {
                    "path": list(leaf.path),
                    "action": leaf.action.value,
                    "mode": leaf.mode.value if leaf.mode is not None else None,
                    "signature": list(leaf.signature),
                    "source": leaf.source,
                    "learned_episode": leaf.learned_episode,
                    "learned_step": leaf.learned_step,
                    "support": leaf.support,
                    "episodes": leaf.episodes,
                    "fires": leaf.fires,
                    "good": leaf.good,
                    "bad": leaf.bad,
                    "conflicts": leaf.conflicts,
                    "unevidenced": leaf.unevidenced,
                    "replacements": leaf.replacements,
                    "retired": leaf.retired,
                    "note": leaf.note,
                    # Only where a run filed replay labels, so every other stored tree is unchanged.
                    **({"cf_needed": leaf.cf_needed, "cf_not_needed": leaf.cf_not_needed}
                       if leaf.cf_needed or leaf.cf_not_needed else {}),
                }
                # Sorted so a stored tree is diffable between runs: the JSON changes only where
                # the policy did, not because a dict happened to iterate differently.
                for leaf in sorted(self.leaves.values(), key=lambda l: l.path)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionTree":
        """Rebuild a tree from `to_dict` output.

        Refuses a file whose `features` header doesn't match this module's, rather than loading
        it and quietly misreading every path: the paths are positional, so a feature inserted or
        reordered since the file was written silently changes what every stored rule means.
        """
        stored = tuple(data.get("features", ()))
        if stored != FEATURE_NAMES:
            raise ValueError(
                "this tree was grown against a different feature set and its rules would be "
                f"misread:\n  stored:  {list(stored)}\n  current: {list(FEATURE_NAMES)}\n"
                "Grow a fresh tree (delete the file, or pass a new --tree path) rather than "
                "editing this one by hand."
            )
        tree = cls(init_depth=int(data.get("init_depth", 2)))
        for rule in data.get("rules", ()):
            mode = rule.get("mode")
            leaf = Leaf(
                path=tuple(rule["path"]),
                action=RecoveryMethod(rule["action"]),
                mode=FailureMode(mode) if mode else None,
                signature=tuple(rule.get("signature", rule["path"])),
                source=rule.get("source", "llm"),
                learned_episode=int(rule.get("learned_episode", 0)),
                learned_step=int(rule.get("learned_step", 0)),
                support=int(rule.get("support", 1)),
                episodes=int(rule.get("episodes", 1)),
                fires=int(rule.get("fires", 0)),
                good=int(rule.get("good", 0)),
                bad=int(rule.get("bad", 0)),
                conflicts=int(rule.get("conflicts", 0)),
                unevidenced=bool(rule.get("unevidenced", False)),
                replacements=int(rule.get("replacements", 0)),
                retired=bool(rule.get("retired", False)),
                cf_needed=int(rule.get("cf_needed", 0)),
                cf_not_needed=int(rule.get("cf_not_needed", 0)),
                note=rule.get("note", ""),
            )
            tree.leaves[leaf.path] = leaf
        return tree


def _first_difference(a: tuple[str, ...], b: tuple[str, ...]) -> int | None:
    """Index of the first feature on which two signatures differ, or `None` if they never do."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


# -- the policy ---------------------------------------------------------------------------

@dataclass
class TreeStats:
    """What the policy did over a run. Every field is a count of ticks or of tree edits, so the
    headline numbers a run is judged on (`hit_rate`, and `escalations` against `ticks`) can be
    recomputed from a stored file rather than trusted from a log line."""
    ticks: int = 0                # assessments the policy was asked for
    hits: int = 0                 # ... answered from the tree
    escalations: int = 0          # ... sent to the teacher
    confirmations: int = 0        # ... of those, sent because the covering rule was not yet confirmed (see min_support)
    unanswered: int = 0           # ... that missed with no teacher available (answered NOMINAL/CONTINUE)
    breaker: int = 0              # ... decided by the recovery-loop breaker rule instead of either
    inserts: int = 0
    reinforces: int = 0
    refines: int = 0
    replaces: int = 0
    contradictions: int = 0
    judged_good: int = 0
    judged_bad: int = 0
    unevidenced: int = 0            # teacher answers naming a mode the report's own test did not support
    undecided: int = 0            # fired rules whose outcome never became clear before the defer cap
    orphaned: int = 0             # judgements whose rule was re-taught to a different action before they came due
    retirements: int = 0

    @property
    def hit_rate(self) -> float:
        """Share of assessments answered without the teacher. The learning curve is this number
        per episode; it starts at 0 on an empty tree and is the headline result of a run."""
        decided = self.hits + self.escalations + self.unanswered
        return self.hits / decided if decided else 0.0

    def delta(self, before: "TreeStats") -> "TreeStats":
        """This snapshot minus an earlier one -- per-episode numbers out of a running total."""
        return TreeStats(**{name: getattr(self, name) - getattr(before, name)
                            for name in self.__dataclass_fields__})

    def snapshot(self) -> "TreeStats":
        return replace(self)

    def __str__(self) -> str:
        return (f"{self.ticks} ticks: {self.hits} from the tree, {self.escalations} to the teacher "
                f"({100 * self.hit_rate:.0f}% hit rate); "
                f"rules +{self.inserts} new, {self.reinforces} reinforced, {self.refines} refined, "
                f"{self.replaces} replaced, {self.contradictions} contradicted, "
                f"{self.confirmations} confirmation asks; "
                f"outcomes {self.judged_good} ok / {self.judged_bad} bad "
                f"({self.undecided} unresolved, {self.orphaned} orphaned), {self.retirements} retired"
                + (f"; {self.unevidenced} diagnosis(es) the report's own test did not support"
                   if self.unevidenced else ""))


_STAT_FOR_EDIT = {"insert": "inserts", "reinforce": "reinforces", "refine": "refines",
                  "replace": "replaces", "contradict": "contradictions"}
"""`DecisionTree.learn`'s outcome -> the `TreeStats` counter that records it."""


_CONTINUES_A_WAIT = (RecoveryMethod.WAIT, RecoveryMethod.RESUME_ROUTE, RecoveryMethod.REQUEST_HUMAN)
"""Actions that carry a pending WAIT forward instead of superseding it (see `TreePolicy._start_scoring`)."""


@dataclass
class _Pending:
    """A recovery that has been ordered and is waiting to be judged by what the robot did next.

    Only rules that actually order something are queued here. A rule answering NOMINAL/CONTINUE
    executes nothing, so there is no effect to wait for -- it is scored immediately instead, and
    only for the mistake it can be held to (see `TreePolicy.decide`).
    """
    path: tuple[str, ...]
    signature: tuple[str, ...]
    """The full signature of the report the recovery was ordered on. If the rule at `path` has
    been split by the time the judgement comes due, this finds the rule that now governs that
    exact situation."""
    step: int
    action: RecoveryMethod
    window_start: int
    """Step the judgement window opened: the order step, moved forward to the first tick after any
    hold the recovery put the robot in. A WAIT is judged on what the robot does once it is let go,
    not on the 25 cycles it was told to stand still."""
    progress_at_start: float | None
    """`route_progress_m` when the window opened. `None` if the report did not carry it, in which
    case the recovery is dropped uncredited rather than guessed at."""
    held: bool = False
    """Whether a hold has been seen since the order, i.e. `window_start` still has to move."""


class TreePolicy:
    """The tree plus everything that changes it: when to trust it, how a teacher's answer is
    folded in, and how a rule that fired is scored against what the robot did next.

    This is the object a run owns. `tree_analyzer.TreeAnalyzer` is a thin adapter that gives it
    the shape the pipeline expects; a caller that wants to drive it directly needs only three
    methods, in this order, once per assessment tick:

        policy.observe(context)              # judge whatever fired earlier, retire what failed
        leaf = policy.decide(context)        # a usable rule, or None -- ask the teacher
        policy.learn(context, method, mode)  # ... and fold the teacher's answer back in

    Args:
        tree: an existing `DecisionTree` to carry on growing; a fresh empty one by default.
        init_depth: how many features a brand-new rule is conditioned on -- at least, since it
            enters deeper wherever the tree has already been split there (see
            `DecisionTree._insert_depth`).
        horizon_steps: length of the window a recovery is judged over, in control cycles, counted
            from when it was ordered or, if it held the robot, from when the hold ended. The
            default is one `MOTION_WINDOW_S` at the 0.2 s control period: long enough for a robot
            that has got going to cover `RECOVERED_PROGRESS_M` at a fraction of its cruising speed.
        max_defer_steps: a recovery whose robot is still being held this many cycles after it was
            ordered is dropped uncredited rather than judged.
        retire_after: bad outcomes before a rule stops being trusted.
        conflict_limit: full-width contradictions -- or re-teachings of one retired path (see
            `Leaf.replacements`) -- before a rule is permanently deferred to the teacher. Reaching
            it either way is a statement about `FEATURES`: the teacher, or the outcome check, is
            distinguishing situations this signature cannot. The policy stops pretending otherwise
            instead of flip-flopping between the two answers forever.
        source: what to record as the teacher's name on rules learned through this policy.
        min_support: distinct episodes whose teacher answers a rule needs behind it before the
            tree answers from it alone (see `Leaf.episodes` for why episodes, not answers).
            With 1, the default, a rule is trusted from the moment it is taught. It is then never
            asked about again, because only misses reach the teacher, so a minority first answer
            can be frozen in. With 2, a matching report goes back to the teacher once more: an
            agreeing answer confirms the rule, and a disagreeing one refines or contradicts it.
        cf_retire_after: counterfactual replays labelling a recovery rule's actions "unnecessary"
            or "harmful" before the rule is retired, provided "needed" is at most
            `cf_needed_share` of its labels (`note_counterfactual`). `None` only counts them.
    """

    def __init__(self, tree: DecisionTree | None = None, *, init_depth: int = 2,
                 horizon_steps: int = 25, max_defer_steps: int = 75, retire_after: int = 2,
                 conflict_limit: int = 3, source: str = "llm", min_support: int = 1,
                 cf_retire_after: int | None = None, cf_needed_share: float = 0.1):
        self.tree = tree if tree is not None else DecisionTree(init_depth=init_depth)
        self.cf_retire_after = cf_retire_after
        """Replays saying "not needed" before a recovery rule is retired on them, provided `needed`
        is at most `cf_needed_share` of its labels. `None` (the default) only counts. See
        `note_counterfactual`."""
        self.cf_needed_share = cf_needed_share
        self.horizon_steps = horizon_steps
        self.max_defer_steps = max_defer_steps
        self.retire_after = retire_after
        self.conflict_limit = conflict_limit
        self.source = source
        self.min_support = max(1, min_support)
        self.stats = TreeStats()
        self.episode = 0
        self.events: list[str] = []
        """Human-readable log of every change to the tree, in order, across the whole run. This is
        what a write-up quotes: "the rule the robot ran on at episode 3 was learned at episode 1,
        step 88, refined at episode 2 and retired at episode 5"."""
        self.listeners: list = []
        """Callables `(signature, action_name, verdict, why)` told of every judge verdict, booked or
        orphaned, as it is made -- how `experience.ExperienceTable` is kept up to date online without
        this module knowing it exists."""
        self.timeline: list[dict] = []
        """The same history as `events`, structured, plus the `good` verdicts `events` leaves out:
        one dict per edit (`insert`, `refine`, ...), verdict (`good`/`bad`) or `retired`, with
        episode, step, rule path, action and a short text. What a replay draws; not persisted."""
        self._pending: list[_Pending] = []
        self._last_step = -1

    # -- episode bookkeeping ---------------------------------------------------------------

    def start_episode(self) -> None:
        """Begin a new episode: the tree carries over, the pending judgements do not.

        A rule that fired in the last seconds of an episode has no next-few-seconds to be judged
        on -- the robot arrived, or ran out of budget -- and crediting it against the *next*
        episode's opening cycles would be judging it on a different robot in a different place.
        Those simply go uncredited.
        """
        self.episode += 1
        self._pending.clear()
        self._last_step = -1

    # -- 1. judging what already fired -----------------------------------------------------

    def observe(self, context: Any) -> None:
        """Score recoveries ordered earlier against what the robot has done since, and retire the
        rules that keep failing. Call once per assessment tick, before `decide`.

        This is the half that makes the arrangement learn from mistakes rather than merely cache
        answers, and it needs no new plumbing anywhere in the control loop: the policy is handed a
        fresh report every tick anyway, so the evidence for "did that recovery help?" arrives on
        its own. A caller that skips `observe` still gets a working cache -- and nothing that can
        ever correct itself.
        """
        step = int(getattr(context, "step", self._last_step + 1) or 0)
        if step < self._last_step:
            # The step counter went backwards, so this is a new episode nobody announced. Treat it
            # as one rather than judging last episode's rules against this one's opening cycles.
            self.start_episode()
        self._last_step = step

        holding = bool(getattr(context, "holding_position", False))
        still_pending: list[_Pending] = []
        no_route = _route(context) == "none"
        for pending in self._pending:
            if (no_route and pending.action is RecoveryMethod.REPLAN_ROUTE
                    and pending.signature[FEATURE_NAMES.index("route")] == "ok"):
                # The replan came back with "no route exists". It gained no ground because there is
                # none to gain, and finding that out is the most useful thing it could have done.
                # A replan ordered *after* the report already said `route=none` is judged as usual.
                self.stats.undecided += 1
                self._note("unjudged", step, pending.path, pending.action.value, "the replan found that no route exists")
                continue
            if holding:
                # Whatever is holding the robot -- this recovery or one ordered since -- the robot is
                # not free to make progress, so the window has not opened yet.
                pending.held = True
                if step - pending.step <= self.max_defer_steps:
                    still_pending.append(pending)
                else:
                    self.stats.undecided += 1
                continue
            if pending.held:
                pending.held = False
                pending.window_start = step
                pending.progress_at_start = _num(context, "route_progress_m")
            if step < pending.window_start + self.horizon_steps:
                still_pending.append(pending)
                continue
            verdict, why = self._judge(pending, context)
            if verdict is None:
                self.stats.undecided += 1
                continue
            self._credit(pending, verdict, step, why)
        self._pending = still_pending

    def end_episode(self, termination: str, needs_human: bool | None = None) -> None:
        """Settle the judgements still pending when an episode ends, on how it ended.

        Reaching the goal is the clearest "the robot got going again" there is, and it is exactly
        the case the window cannot see: a recovery ordered in the last few seconds of a successful
        episode has no window left to be measured over. A collision is its opposite, and ends the
        episode before the report that would show the contact is ever built. Any other ending (a
        timeout, an escalation, a solver error) says nothing about a recovery ordered moments
        before it, so those judgements go uncredited, as they did before this was called at all.

        A REQUEST_HUMAN is the exception, because it *is* the ending: there is no motion after it
        to measure. It is judged on `needs_human` -- whether anything the robot could have done
        would have reached the goal -- which is what the person who answers the call finds out on
        arrival, and which the scenario grid knows by construction (`ScenarioLayout.needs_human`).
        `None` leaves it uncredited.

        Optional: a caller that never calls it gets `start_episode`'s behaviour, which drops them.
        """
        # A watchdog stop is a collision's equal here: the robot was lost while this recovery was
        # the thing in charge of it.
        verdict = {"goal": "good", "collision": "bad", "watchdog": "bad"}.get(termination)
        step = max(self._last_step, 0)
        for pending in self._pending:
            if pending.action is RecoveryMethod.REQUEST_HUMAN:
                if needs_human is None:
                    self.stats.undecided += 1
                else:
                    self._credit(pending, "good" if needs_human else "bad", step,
                                 why="a human was needed" if needs_human else "false alarm: the goal was reachable")
            elif verdict is None:
                self.stats.undecided += 1
            else:
                self._credit(pending, verdict, step, why=f"episode ended: {termination}")
        self._pending.clear()

    def _credit(self, pending: _Pending, verdict: str, step: int, why: str = "") -> None:
        """Book a verdict against the rule that ordered `pending`, retiring it if it is spent."""
        # The action was taken in that situation and this is how it went, whatever has become of
        # the rule since: listeners hear of it even if the verdict ends up orphaned.
        self._tell(pending.signature, pending.action, verdict, why)
        leaf = self.tree.leaves.get(pending.path)
        if leaf is None:
            # The rule was split while this judgement was pending, so its path moved. The rule
            # that now governs the same situation still deserves the credit, as long as it
            # still orders the same action.
            leaf = self.tree.match(pending.signature)
        if leaf is None or leaf.action is not pending.action:
            # Re-taught to something else in the meantime: this outcome describes an action
            # the tree no longer takes. Counted, rather than silently dropped.
            self.stats.orphaned += 1
            return
        self._note(verdict, step, leaf.path, pending.action.value, why)
        if verdict == "good":
            leaf.good += 1
            self.stats.judged_good += 1
        else:
            leaf.bad += 1
            self.stats.judged_bad += 1
            self._log(f"[ep {self.episode} step {step}] {'/'.join(pending.path)} "
                      f"-> {pending.action.value} did not help ({leaf.bad}/{self.retire_after})"
                      + (f" -- {why}" if why else ""))
            self._retire_if_spent(leaf, step)

    def _retire_if_spent(self, leaf: Leaf, step: int) -> None:
        """Stop trusting a rule that has gone bad `retire_after` times. The rule stays in the tree
        (and in the stored file) as a record of what was tried; `decide` simply stops answering
        from it, so the next matching tick goes back to the teacher and `learn` replaces it."""
        if leaf.bad >= self.retire_after and not leaf.retired:
            leaf.retired = True
            self.stats.retirements += 1
            self._note("retired", step, leaf.path, leaf.action.value, f"after {leaf.bad} bad outcomes")
            self._log(f"[ep {self.episode} step {step}] retired {'/'.join(leaf.path)} "
                      f"-> {leaf.action.value}: next match goes back to the teacher")

    def note_counterfactual(self, sig: tuple[str, ...], action: RecoveryMethod, label: str,
                            step: int = 0) -> Leaf | None:
        """File one counterfactual label (`counterfactual.py`: the episode re-run with this recovery
        replaced by CONTINUE) against the rule that governs `sig`, and retire the rule if the replays
        keep saying the recovery was not needed. Returns the rule, if any.

        The second opinion the route-progress judge cannot give: after a hold on an open road the
        robot does drive on, so the judge books it good, while replayed without the hold it was not
        needed.

        The bar is deliberately lopsided. `harmful` mostly means "25 steps sooner without it",
        `needed` means the episode is lost without it, so a rule is only retired when `needed` is
        rare (`cf_needed_share`, one label in ten) over enough labels (`cf_retire_after`). Retired,
        the rule behaves as any other retired rule: the next match goes back to the teacher, whose
        answer replaces it, and `conflict_limit` stops a path that keeps being re-taught the same
        way. Labels are filed whoever made the decision -- the teacher's answer and the tree's are
        the same rule. `unclear` labels are ignored."""
        leaf = self.tree.match(tuple(sig))
        if leaf is None or leaf.retired or leaf.action is not action or label not in ("needed", "unnecessary", "harmful"):
            return None
        if label == "needed":
            leaf.cf_needed += 1
        else:
            leaf.cf_not_needed += 1
        total = leaf.cf_needed + leaf.cf_not_needed
        if (self.cf_retire_after is not None and leaf.cf_not_needed >= self.cf_retire_after
                and leaf.cf_needed <= self.cf_needed_share * total):
            leaf.retired = True
            self.stats.retirements += 1
            why = f"replayed without it, not needed in {leaf.cf_not_needed} of {total}"
            self._note("retired", step, leaf.path, leaf.action.value, why)
            self._log(f"[ep {self.episode} step {step}] retired {'/'.join(leaf.path)} "
                      f"-> {leaf.action.value}: {why}; next match goes back to the teacher")
        return leaf

    def _judge(self, pending: _Pending, context: Any) -> tuple[str | None, str]:
        """`("good" | "bad", why)` for one recovery whose window has run, or `(None, why)` when the
        report cannot say. Called `horizon_steps` after the window opened (see `_Pending.window_start`).

        The question is the one a recovery exists to answer -- did the robot get going again -- and
        it is measured along the route rather than read off the failure tests:

        - in contact with an obstacle: **bad**.
        - gained at least `RECOVERED_PROGRESS_M` of `route_progress_m` over the window: **good**.
        - otherwise: **bad**.

        Not on the failure tests at the end of the window, which would blame the recovery for
        whatever happened next (a pedestrian arriving after the robot had got going), and not on
        progress toward the goal, which a detour moves away from. Progress along the route being
        driven counts the detour for what it is.
        """
        if meets_test(context, FailureMode.COLLISION):
            return "bad", "in contact with an obstacle"
        gained = self._gained(pending, context)
        if gained is None:
            return None, "no route progress measured"
        if gained >= RECOVERED_PROGRESS_M:
            return "good", f"{gained:.1f} m along the route"
        return "bad", f"only {gained:.1f} m along the route in {self.horizon_steps} steps"

    @staticmethod
    def _gained(pending: _Pending, context: Any) -> float | None:
        """Route progress since `pending`'s window opened, or `None` if either end is unmeasured."""
        now = _num(context, "route_progress_m")
        if now is None or pending.progress_at_start is None:
            return None
        return now - pending.progress_at_start

    # -- 2. deciding -----------------------------------------------------------------------

    def decide(self, context: Any) -> Leaf | None:
        """The rule governing this report, or `None` to ask the teacher.

        A leaf is returned only if it is trusted: not retired by a bad outcome, not past
        `conflict_limit` full-width contradictions, not past the same limit in re-teachings of
        one path (`Leaf.replacements`), and confirmed by teacher answers from at least
        `min_support` distinct episodes (`Leaf.episodes`).

        Returning it counts as a fire, and starts whichever scoring applies. A rule that orders a
        recovery is queued for `observe` to judge once the robot has had a motion window to respond.
        A rule answering NOMINAL executes nothing, so there is nothing to wait for: it is scored
        here and now, and only for the one mistake it can fairly be held to.
        """
        self.stats.ticks += 1
        sig = signature(context)
        leaf = self.tree.match(sig)
        if (leaf is None or leaf.retired or leaf.conflicts >= self.conflict_limit
                or leaf.replacements >= self.conflict_limit):
            self.stats.escalations += 1
            return None
        if leaf.episodes < self.min_support:
            # Covered, but not yet confirmed: ask again. The answer comes back through `learn`,
            # which reinforces the rule if it agrees and refines or contradicts it if not.
            self.stats.escalations += 1
            self.stats.confirmations += 1
            return None
        self.stats.hits += 1
        leaf.fires += 1
        self._start_scoring(leaf, context)
        return leaf

    def _start_scoring(self, leaf: Leaf, context: Any) -> None:
        """Begin judging an action this rule stands for, taken on `context` -- whether the tree
        answered from the rule or the teacher's answer just created or confirmed it.

        Both are scored: scoring only tree hits would leave every first occurrence unjudged.
        """
        step = int(getattr(context, "step", 0) or 0)

        if leaf.action is RecoveryMethod.CONTINUE:
            # Nothing is executed, so there is no effect to wait for. The one mistake such a rule
            # can be held to is doing nothing about a report that already met a failure test --
            # misreading the evidence in front of it, rather than failing to predict what happened
            # next. Scoring it on the latter would retire the most valuable rule in the tree ("the
            # robot is driving, leave it alone") the first time a robot drove into trouble five
            # seconds after being correctly left alone.
            #
            # That holds whatever mode the rule names. A STUCK/CONTINUE rule on a robot that meets
            # a failure test has done nothing about it, exactly as a NOMINAL one has; exempting
            # rules that name a mode meant they could never be judged at all. The solver modes are
            # safe here, because `evidence_of_failure` only reports the physical ones.
            missed = evidence_of_failure(context)
            if missed is not None:
                self._tell(signature(context), leaf.action, "bad",
                           f"did nothing about a report meeting the {missed.value.upper()} test")
                leaf.bad += 1
                self.stats.judged_bad += 1
                self._note("bad", step, leaf.path, leaf.action.value,
                           f"did nothing about a report meeting the {missed.value.upper()} test")
                verdict = leaf.mode.value.upper() if leaf.mode is not None else "NOMINAL"
                self._log(f"[ep {self.episode} step {step}] {'/'.join(leaf.path)} -> {verdict}/CONTINUE "
                          f"did nothing about a report that meets the {missed.value.upper()} test "
                          f"({leaf.bad}/{self.retire_after})")
                self._retire_if_spent(leaf, step)
            return

        # Whatever is still pending has been superseded: a new recovery replaces it, hold included,
        # so its window ends here. Left pending, its window would keep being pushed back by the
        # holds of the recoveries that replaced it.
        carried: list[_Pending] = []
        for pending in self._pending:
            if pending.action is RecoveryMethod.WAIT and leaf.action in _CONTINUES_A_WAIT:
                # A WAIT followed by another WAIT is one wait, not a failed one and a retry: the
                # obstacle is still there, which is the only thing a hold of fixed length can find
                # out. Judged as superseded, the first hold at every gate that stayed shut longer
                # than one hold was booked bad -- 45 bad to 22 good over 40 scenario-grid episodes,
                # 8 WAIT rules retired, in layouts where WAIT was the one action that worked. The
                # chain is judged where it ends: on the progress after its last hold, or as
                # superseded if something other than a WAIT takes over. (A WAIT loop at a wall ends
                # in the breaker's replan, which does not come through here: `note_breaker` books
                # that chain bad.) Two other actions
                # continue a wait rather than replace it. RESUME_ROUTE ends the hold early *because*
                # the obstacle has gone, so the WAIT it cuts short is judged with it, on the progress
                # that follows (booked as superseded, that was 18 of 30 bad WAIT verdicts in a run
                # whose gate episodes all ended right). And REQUEST_HUMAN's own condition is that
                # WAIT has been tried: those holds are left unjudged when the episode ends there,
                # rather than held against a rule that is right at every gate.
                carried.append(pending)
                continue
            gained = self._gained(pending, context)
            if pending.held or gained is None or gained < RECOVERED_PROGRESS_M:
                shown = "while still holding" if pending.held or gained is None else f"after only {gained:.1f} m"
                self._credit(pending, "bad", step, f"superseded by {leaf.action.value} {shown}")
            else:
                self._credit(pending, "good", step)
        self._pending = carried

        self._pending.append(_Pending(
            path=leaf.path,
            signature=signature(context),
            step=step,
            action=leaf.action,
            window_start=step,
            progress_at_start=_num(context, "route_progress_m"),
        ))

    def note_unanswered(self) -> None:
        """Record that a miss went unanswered because there was no teacher to ask (see
        `tree_analyzer.TreeAnalyzer`, which then falls back to a nominal reading). Counted
        separately from an escalation so a frozen-tree run's coverage is not flattered by the
        ticks it had no answer for."""
        self.stats.escalations -= 1
        self.stats.unanswered += 1

    def note_breaker(self, context: Any = None) -> None:
        """Record a tick decided by the recovery-loop breaker rather than by tree or teacher, and
        settle the WAITs it is taking over from where the report already says they were wrong.

        The breaker's action never goes through `_start_scoring` (nothing is learned from a tick
        whose answer was fixed), so it supersedes nothing: a WAIT still pending is judged afterwards
        on the progress the breaker's replan produces. With a chain of WAITs judged as one, that
        would credit a whole loop of waiting at a wall with the detour that ended it. So when the
        breaker fires with `static_obstacle_blocking_path` true, the pending WAITs are booked bad
        here: a hold never moves a wall, which is the one thing WAIT's own description rules out.
        Against a purely dynamic blockage they stay pending -- a gate still shut after three holds
        is not evidence that waiting for it was wrong.
        """
        self.stats.ticks += 1
        self.stats.breaker += 1
        if context is None or not getattr(context, "static_obstacle_blocking_path", False):
            return
        step = int(getattr(context, "step", self._last_step) or 0)
        waits = [p for p in self._pending if p.action is RecoveryMethod.WAIT]
        self._pending = [p for p in self._pending if p.action is not RecoveryMethod.WAIT]
        for pending in waits:
            self._credit(pending, "bad", step, "the breaker took over with a static obstacle on the route")

    # -- 3. learning -----------------------------------------------------------------------

    def learn(self, context: Any, action: RecoveryMethod, mode: FailureMode | None,
              *, source: str | None = None) -> tuple[Leaf, str]:
        """Fold a teacher's answer for this report into the tree. Returns `(leaf, what_happened)`,
        with `what_happened` as `DecisionTree.learn` defines it."""
        step = int(getattr(context, "step", 0) or 0)
        # Does the report the teacher was shown actually satisfy the test for the mode it named?
        # Recorded, never overruled -- see `Leaf.unevidenced` for why this is worth knowing and why
        # it is not worth acting on.
        unevidenced = mode is not None and not meets_test(context, mode)
        if unevidenced:
            self.stats.unevidenced += 1
            self._log(f"[ep {self.episode} step {step}] the {self.source} called this "
                      f"{mode.value.upper()}, which this report's own test does not support")
        leaf, what = self.tree.learn(signature(context), action, mode, unevidenced=unevidenced,
                                     source=source or self.source, episode=self.episode, step=step)
        counter = _STAT_FOR_EDIT[what]
        setattr(self.stats, counter, getattr(self.stats, counter) + 1)
        self._note(what, step, leaf.path, action.value,
                   mode.value.upper() if mode is not None else "NOMINAL")
        if what != "reinforce":
            verdict = mode.value.upper() if mode is not None else "NOMINAL"
            self._log(f"[ep {self.episode} step {step}] {what}: {'/'.join(leaf.path)} "
                      f"-> {verdict}/{action.value}")
        # The teacher's answer is acted on too, so it is judged like a tree hit, and the credit
        # goes to the rule that now holds it.
        self._start_scoring(leaf, context)
        return leaf, what

    def _tell(self, sig: tuple[str, ...], action: RecoveryMethod, verdict: str, why: str = "") -> None:
        for listener in self.listeners:
            listener(sig, action.value, verdict, why)

    def _log(self, line: str) -> None:
        self.events.append(line)

    def _note(self, kind: str, step: int, path: tuple[str, ...], action: str, text: str) -> None:
        self.timeline.append({"episode": self.episode, "step": step, "kind": kind,
                              "rule": "/".join(path), "action": action, "text": text})

    # -- persistence -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        data = self.tree.to_dict()
        data["version"] = 1
        data["episodes"] = self.episode
        data["stats"] = {name: getattr(self.stats, name) for name in TreeStats.__dataclass_fields__}
        data["events"] = self.events
        return data

    def save(self, path: str) -> None:
        """Write the tree, its running stats and its edit log to `path` as JSON.

        Written whole and atomically (a temporary file in the same directory, then a rename), so
        an interrupted run -- Ctrl-C during a sweep is routine -- leaves the previous tree intact
        rather than a half-written file that the next run refuses to load.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TreePolicy":
        """Carry on from a stored tree. `kwargs` go to `__init__`, so the learning parameters are
        the caller's to set per run; only the rules themselves come out of the file."""
        with open(path) as f:
            data = json.load(f)
        policy = cls(tree=DecisionTree.from_dict(data), **kwargs)
        policy.episode = int(data.get("episodes", 0))
        policy.events = list(data.get("events", ()))
        for name, value in (data.get("stats") or {}).items():
            if name in TreeStats.__dataclass_fields__:
                setattr(policy.stats, name, int(value))
        return policy

    @classmethod
    def open(cls, path: str | None, *, reset: bool = False, **kwargs) -> "TreePolicy":
        """`load` if `path` names an existing file, a fresh policy otherwise -- the form a runner
        wants, where the same command should work on the first run and the tenth."""
        if path and not reset and os.path.exists(path):
            return cls.load(path, **kwargs)
        return cls(**kwargs)
