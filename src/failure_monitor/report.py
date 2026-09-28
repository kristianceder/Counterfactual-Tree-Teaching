"""The status report the decision-maker reads on every assessment tick, and the evidence each failure
mode is judged on.

The report is built on a fixed clock whether or not anything looks wrong: nothing has flagged it,
so it carries the raw measurements a failure verdict needs (how long the robot has been stopped,
how far it got, what is on its route, how the solver is keeping up) rather than any verdict.
`FAILURE_MODE_EVIDENCE` is the other half: the decisive test for each failure mode, written against
these field names, which is what the teacher is told and what the decision tree bins on
(`tree.signature`).

`SituationBuilder.update()` is cheap and runs every control cycle; `build()` does the geometry and
runs only on an assessment tick.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from shapely.geometry import LineString, Point, Polygon  # type: ignore

from .types import FailureMode, RobotSnapshot


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_from(route: list[tuple[float, float]], position: tuple[float, float]) -> list[tuple[float, float]]:
    """The part of `route` still ahead of `position`: the point on the route nearest the robot,
    followed by every waypoint beyond it."""
    if len(route) < 2:
        return list(route)
    line = LineString(route)
    along = line.project(Point(position))
    start = line.interpolate(along)
    ahead = [(start.x, start.y)]
    travelled = 0.0
    for a, b in zip(route, route[1:]):
        travelled += _dist(a, b)
        if travelled > along:
            ahead.append(b)
    return ahead


# Eight 45-degree sectors for `clearance_by_direction`, starting at the robot's heading and going
# counter-clockwise, named so the reader can tell at a glance which way "blocked at 0.4 m" is.
_SECTOR_NAMES = ('ahead', 'ahead_left', 'left', 'behind_left',
                 'behind', 'behind_right', 'right', 'ahead_right')


FAILURE_MODE_EVIDENCE: dict[FailureMode, str] = {
    FailureMode.STUCK: (
        "test: stopped_for_s of 4 or more, with holding_position false and goal_distance above 0.3. "
        "The robot has come to a standstill it cannot drive its way out of -- the classic MPC local "
        "optimum, where the receding horizon has gone into a concave pocket and has nothing left to "
        "trade off against; static_obstacle_blocking_path: true confirms what it is jammed against. "
        "Stopping for a second or two to let something pass is ordinary driving, not this. Neither is "
        "a robot standing still because a WAIT or REORIENT told it to, which is what "
        "holding_position: true means. Judge it on stopped_for_s rather than path_length_last_5s: "
        "that one says the same thing but saturates, so past five seconds a brief pause and a robot "
        "pinned for half a minute read exactly alike."
    ),
    FailureMode.OSCILLATION: (
        "test: heading_reversals_last_5s of 4 or more, with goal_progress_last_5s below 0.2. "
        "The robot is moving but getting nowhere -- wavering or zig-zagging, typically at the mouth "
        "of an obstacle or against something it keeps swerving around. The signature is a large "
        "path_length_last_5s (it is driving) against a small net_displacement_last_5s (it is not "
        "getting anywhere), which is what separates this from STUCK, where it is not moving at all."
    ),
    FailureMode.REVERSE_TRACKING: (
        "test: reversing_for_s of 3 or more. "
        "The robot is driving its route backwards instead of turning around to face the way it needs "
        "to go. For this unicycle both track the same path at almost the same cost, so the solver can "
        "settle into reverse at a third of the forward speed and never come out of it on its own. "
        "reversing_for_s counts unbroken seconds of negative robot_speed, not counting time a recovery "
        "held the robot (a BACK_OFF reverses on purpose), so 0 -- or anything below 3 -- means this is not the failure, whatever route_heading_error_deg reads. A large heading "
        "error on its own only says the robot is pointed away from its route, which is equally what a "
        "turn already correcting it looks like partway through; route_heading_error_change_5s is what "
        "tells those two apart."
    ),
    FailureMode.COLLISION: (
        "test: static_obstacle_gap or dynamic_obstacle_gap at or below 0. "
        "The robot's footprint is touching or overlapping something. Both gaps are measured from the "
        "footprint edge (robot_radius is already accounted for), so 0 is contact and a negative value "
        "is overlap. A small positive gap is a near miss, not a collision."
    ),
    FailureMode.TIMEOUT: (
        "test: steps_remaining too small to cover goal_distance -- at roughly 1 m of progress per 5 "
        "steps, 8 m still to go needs on the order of 40 steps left. "
        "The episode's budget is about to run out with the robot short of the goal. Waiting is what "
        "usually causes this: each WAIT spends about 25 steps."
    ),
}
"""What each `FailureMode` looks like in the report's fields. `SOLVER_DEADLINE_MISS` is added by
`monitor_evidence`, whose text depends on the controller watchdog's limit.

Each entry opens with a `test:` clause -- the decisive numeric condition, in the report's own field
names -- before explaining what the mode means, so it can be checked at a glance against the JSON.
The modes that describe what the robot is physically doing wrong come first and the solver mode
last: a model has to name exactly one, and one shown the solver mode first tends to name it and stop
looking. The threshold constants at the top of `tree.py` mirror these tests; change both together.
(The sentence about BACK_OFF, a recovery not offered here, is kept as the teacher read it.)"""


WATCHDOG_WINDOW_STEPS = 60
"""The controller watchdog counts missed deadlines over this many control cycles (12 s). A window
rather than an unbroken run: a replan resets the reference and the next solve is fast, so a streak
restarts from zero while the solver is in fact still grinding."""
WATCHDOG_STEPS = 45
"""Missed deadlines within `WATCHDOG_WINDOW_STEPS` that stop the controller for good (the episode
ends as "watchdog"). Twelve seconds and 45 misses leave a supervisor room: a tick every 2 s, a
verdict a second or two later, and 2 s for the evidence to build."""
WATCHDOG_EVIDENCE_MISSES = 10
"""`solver_deadline_misses_last_12s` at or above which SOLVER_DEADLINE_MISS's test holds
(`tree.py` mirrors it)."""
WATCHDOG_MARGIN_STEPS = 20
"""With this many misses or fewer left, a robot missing deadlines on an open road is held (WAIT)
rather than left to drive on: one assessment period plus a model call is how long a robot that
misses every solve runs on before a WAIT ordered now takes hold. Carried on the report as
`watchdog_margin` so the prompt, the scripted rules and the tree's `deadline` bin draw the same
line."""


def deadline_miss_evidence(watchdog_steps: int = WATCHDOG_STEPS, margin: int = WATCHDOG_MARGIN_STEPS,
                           window_steps: int = WATCHDOG_WINDOW_STEPS) -> str:
    """SOLVER_DEADLINE_MISS's evidence for a controller with a watchdog.

    A robot pressing its reference through a dynamic obstacle keeps the solver at its time cap, and
    a WAIT hold pins the reference to the current pose, which takes the conflict away; a reference
    through a wall is removed by a replan. With nothing on the route, the solver is only slow for a
    few seconds after a replan or a sharp turn, and the answer depends on the room left (`margin`).
    """
    return (
        f"test: solver_deadline_misses_last_12s of {WATCHDOG_EVIDENCE_MISSES} or more, with "
        f"holding_position false. "
        f"A safety watchdog stops the controller for good once {watchdog_steps} of the last "
        f"{window_steps} solves have run over the control period, and that ends the mission. "
        "solver_deadline_misses_last_12s is that count, and watchdog_steps_remaining is how many more "
        "misses the watchdog will tolerate; the count only falls as slow solves age out of the "
        "window. Judge this mode on the count. solver_deadline_miss_streak is no substitute: a single "
        "fast solve resets the streak to 0 while the robot is still grinding, so a streak of 0 or 1 "
        "next to a count of 25 is a robot a few seconds from being stopped, not a healthy one. With holding_position true it is not this mode, whatever the number says: a "
        "hold is the remedy, the count recovers while it runs, and ordering anything else cancels it. "
        "The cause is nearly always the solver grinding on a reference it cannot follow. With "
        "dynamic_obstacle_blocking_path: true the robot is being asked to drive through something it "
        "must avoid: the ACTION is WAIT, because a hold pins the reference to where the robot already "
        "is and the solve time falls back within a step. REPLAN_ROUTE does not help there, it cannot "
        "see a dynamic obstacle and hands back the same conflict. With "
        "static_obstacle_blocking_path: true it is the other way round: the route runs through a wall, "
        "holding still leaves it there, and REPLAN_ROUTE removes the conflict -- do not wait for "
        "stopped_for_s to reach STUCK's 4 s, a robot pressed against a wall creeps rather than stops. "
        "All of that is for a robot with something on its route. With NEITHER blocking flag true the "
        "road is open and the solver is only slow, typically for a few seconds after a replan or a "
        "sharp turn; it recovers as the robot drives on, and a hold there costs five seconds of "
        f"standing still. Then the ACTION depends on the room left: with watchdog_steps_remaining above "
        f"{margin} it is CONTINUE (still name the mode); at {margin} or below it is WAIT, because a hold "
        "is the one thing that brings the count down before the watchdog acts. With a blocking flag "
        "true, CONTINUE is never the answer: doing nothing is what runs the count down."
    )


def monitor_evidence(watchdog_steps: int = WATCHDOG_STEPS, watchdog_margin: int = WATCHDOG_MARGIN_STEPS
                     ) -> dict[FailureMode, str]:
    """The failure modes offered to the decision-maker and the evidence for each."""
    return {**FAILURE_MODE_EVIDENCE,
            FailureMode.SOLVER_DEADLINE_MISS: deadline_miss_evidence(watchdog_steps, watchdog_margin)}


MOTION_WINDOW_S = 5.0
"""Length of the rolling window the motion and solver statistics are measured over."""

MAX_PROGRESS_STEP_M = 1.0
"""Largest one-cycle change in route still ahead that `route_progress_m` accepts as driven. Several
times what the robot covers in a control period; anything bigger is the projection jumping between
legs of the route."""


@dataclass
class SituationContext:
    """One assessment tick's status report, meant to be handed to the decision-maker as-is.

    Every field is optional: `to_dict`/`to_json` drop unset fields (and empty lists) rather than
    report a misleading zero for something that was never measured. The order of the fields is the
    order of the JSON the teacher reads.
    """
    step: int = 0
    prior_failures_this_episode: list[str] = field(default_factory=list)
    """Earlier verdicts this episode, e.g. ["stuck@88"] (`monitor.VerdictLog`)."""
    steps_remaining: int | None = None
    """Control cycles left in the episode's step budget. A WAIT hold is ~25 steps, so holding out
    for a blockage is a very different proposition with 200 steps left than with 30."""

    # -- kinematic state --
    position: tuple[float, float] | None = None
    heading: float | None = None
    goal_distance: float | None = None
    goal_bearing_deg: float | None = None
    """Signed angle (-180..180 deg) from where the robot is facing to the goal: positive to its left.
    Shares the robot-relative frame of `clearance_by_direction`, because the two are useful
    together; distinct from `route_heading_error_deg`, which is about the route being tracked."""
    robot_speed: float | None = None
    """Commanded linear speed (m/s), signed: negative means the robot is driving backwards."""
    route_heading_error_deg: float | None = None
    """Absolute angle (0-180 deg) between where the robot is facing and where the reference it is
    tracking leads; while a hold collapses that reference to a point, where the intended route
    leads. ~180 with a negative speed is a robot driving its route in reverse (a REORIENT case, not
    a REPLAN_ROUTE one)."""

    # -- planner health --
    solver_time_ms: float | None = None                  # wall-clock time the last MPC solve took
    mpc_realtime: bool | None = None                      # whether that solve finished within the control period
    progress_last_3s: float | None = None               # net decrease in goal_distance; negative = moving away
    reference_replans_last_5s: int | None = None         # how often the local reference shifted materially
    reference_direction_changes: int | None = None       # heading reversals along the current reference

    # -- obstacle situation (distances from the robot's centre) --
    min_obstacle_distance: float | None = None
    dynamic_obstacle_blocking_path: bool | None = None
    """Whether a dynamic obstacle is within 1.3 m of the robot's intended route."""
    static_obstacle_blocking_path: bool | None = None
    """Whether the intended route runs into a static obstacle's real geometry -- the route was
    planned without that obstacle (a hidden wall), so the route itself has to change."""
    blocking_obstacle_speed: float | None = None
    """Speed (m/s) of the dynamic obstacle on the route that is nearest the robot, estimated by
    finite differences between control cycles."""
    blocking_obstacle_stationary_s: float | None = None
    """How long (s) that obstacle has been standing still. A pedestrian stopped for 8 s is not about
    to clear the path; one parked for 30 s has been left there."""

    # -- surroundings, in the robot's own frame --
    clearance_by_direction: dict[str, float] | None = None
    """How far (m) the robot could travel in each of eight 45-degree directions relative to its
    heading before hitting permanent geometry (static obstacles and the map boundary), capped at
    `SituationBuilder.sector_scan_range`. Structural only: says whether a detour has room to go."""

    # -- recovery-loop state --
    previous_recovery: str | None = None
    previous_recovery_success: bool | None = None
    recovery_cycles_without_progress: int | None = None
    """Recovery attempts since the robot last made more than `progress_stall_tol` of progress along
    its route, or since the last REPLAN_ROUTE. The recovery-loop breaker acts on it."""
    goal_reachable: bool | None = None
    """False once a REPLAN_ROUTE this episode found no route to the goal through the static map;
    unset until then, never True. A fact about permanent geometry, so it stays false."""

    # -- motion over the last MOTION_WINDOW_S --
    path_length_last_5s: float | None = None
    """Arc length (m) the robot drove over the last 5 s."""
    net_displacement_last_5s: float | None = None
    """Straight-line distance (m) between where the robot was 5 s ago and where it is now."""
    goal_progress_last_5s: float | None = None
    """Net decrease (m) in `goal_distance` over the window; negative while driving a detour."""
    heading_reversals_last_5s: int | None = None
    """How many times the robot's turn direction flipped over the window (turns above noise only)."""
    reversing_for_s: float | None = None
    """How long (s) the robot has been commanding a negative speed without interruption, not
    counting time a recovery held it."""
    stopped_for_s: float | None = None
    """How long (s) the robot has been at a standstill without interruption, unbounded -- unlike
    `path_length_last_5s`, which saturates after five seconds."""
    route_heading_error_change_5s: float | None = None
    """Change in `route_heading_error_deg` over the last 5 s: negative while turning onto the route,
    which is what says a REORIENT under way is working."""

    # -- contact --
    robot_radius: float | None = None                  # footprint radius (m) the two gaps below already account for
    static_obstacle_gap: float | None = None
    """Distance (m) from the robot's footprint edge to the nearest static obstacle: 0 is contact,
    negative is overlap."""
    dynamic_obstacle_gap: float | None = None
    """Same, for the nearest dynamic obstacle (its own footprint radius included)."""

    # -- solver health --
    solver_deadline_miss_streak: int | None = None     # consecutive steps whose solve overran the control period
    worst_solver_time_ms_last_5s: float | None = None
    control_period_ms: float | None = None             # the deadline the two fields above are measured against
    solver_deadline_misses_last_12s: int | None = None
    """Solves over the control period in the last `WATCHDOG_WINDOW_STEPS` cycles: what the
    controller watchdog counts."""
    watchdog_steps_remaining: int | None = None
    """Missed deadlines left before the watchdog stops the robot for good."""
    watchdog_margin: int | None = None
    """`WATCHDOG_MARGIN_STEPS`, carried for `tree.signature` and the prompt. Not in the report the
    model reads (`REPORT_EXCLUDED`): the number is in its instructions."""

    # -- episode budget --
    step_budget: int | None = None

    # -- what the recovery loop is currently doing to the robot --
    holding_position: bool | None = None
    """True while a WAIT or REORIENT (or a replan turning to face its new route) is holding the
    robot in place -- otherwise a hold working as intended looks like a STUCK robot."""
    recovery_hold_steps_remaining: int | None = None   # steps left on that hold before it lapses on its own
    steps_since_last_recovery: int | None = None
    recoveries_this_episode: list[str] = field(default_factory=list)
    """Every recovery executed so far, in order, e.g. "REPLAN_ROUTE@88:ok"."""

    # -- for scoring recoveries (`tree.TreePolicy`), never shown to the model --
    route_progress_m: float | None = None
    """Distance (m) gained along the intended route so far this episode: an odometer that adds each
    cycle's decrease in route still ahead, and nothing on a cycle where the route was replaced. Only
    differences between two reports mean anything. Counts a detour as progress while it is driven,
    which is what "did that recovery get the robot going again" needs (`tree.TreePolicy._judge`)."""

    REPORT_EXCLUDED = frozenset({"route_progress_m", "watchdog_margin"})
    """Fields `to_dict` (and so the prompt) leaves out. Still readable as attributes."""

    def to_dict(self) -> dict:
        """A plain dict with unset fields, empty lists/dicts and `REPORT_EXCLUDED` dropped, and floats
        rounded to 3 decimals."""
        def _round(v):
            if isinstance(v, float):
                return round(v, 3) + 0.0  # +0.0 folds -0.0 to 0.0, which reads oddly in a prompt
            if isinstance(v, (list, tuple)):
                return type(v)(_round(x) for x in v)
            if isinstance(v, dict):
                return {k: _round(x) for k, x in v.items()}
            return v
        return {k: _round(v) for k, v in asdict(self).items()
                if v is not None and v != [] and v != {} and k not in self.REPORT_EXCLUDED}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SituationBuilder:
    """Keeps the rolling state a `SituationContext` needs and builds one on demand."""

    def __init__(self, ts: float, max_steps: int, robot_radius: float, dynamic_obstacle_radius: float,
                 watchdog_steps: int = WATCHDOG_STEPS, watchdog_margin: int = WATCHDOG_MARGIN_STEPS,
                 obstacle_block_margin: float = 1.3, static_obstacle_block_margin: float = 0.5,
                 progress_stall_tol: float = 1.0, sector_scan_range: float = 10.0,
                 stationary_speed_tol: float = 0.05, min_reversal_angle: float = 0.05,
                 reverse_speed_tol: float = -0.05, stopped_speed_tol: float = 0.05):
        """
        Args:
            ts: control period (s).
            max_steps: the episode's step budget.
            robot_radius, dynamic_obstacle_radius: footprint radii (m) for the two gap fields.
            watchdog_steps: the controller watchdog's limit (see `WATCHDOG_STEPS`).
            watchdog_margin: see `WATCHDOG_MARGIN_STEPS`.
            obstacle_block_margin: distance (m) from the route inside which a dynamic obstacle's
                centre counts as blocking it: an obstacle's radius plus the robot's, plus slack.
            static_obstacle_block_margin: distance (m) from the route inside which a static
                obstacle's real polygon counts as blocking it (about the robot's radius).
            progress_stall_tol: progress (m) along the route that counts as real progress for
                `recovery_cycles_without_progress`.
            sector_scan_range: how far (m) `clearance_by_direction` looks.
            stationary_speed_tol: estimated obstacle speed (m/s) below which it counts as stopped,
                above the noise floor of the finite-difference estimate.
            min_reversal_angle: per-step heading change (rad) below which a turn is solver noise.
            reverse_speed_tol: commanded speed (m/s) below which the robot counts as reversing.
            stopped_speed_tol: commanded speed magnitude (m/s) below which it counts as stopped.
        """
        self.ts = ts
        self.max_steps = max_steps
        self.robot_radius = robot_radius
        self.dynamic_obstacle_radius = dynamic_obstacle_radius
        self.watchdog_steps = watchdog_steps
        self.watchdog_margin = watchdog_margin
        self.obstacle_block_margin = obstacle_block_margin
        self.static_obstacle_block_margin = static_obstacle_block_margin
        self.progress_stall_tol = progress_stall_tol
        self.sector_scan_range = sector_scan_range
        self.stationary_speed_tol = stationary_speed_tol
        self.min_reversal_angle = min_reversal_angle
        self.reverse_speed_tol = reverse_speed_tol
        self.stopped_speed_tol = stopped_speed_tol

        self._goal_dists: deque[float] = deque(maxlen=max(1, round(3.0 / ts)))
        self._replans: deque[bool] = deque(maxlen=max(1, round(5.0 / ts)))
        self._last_reference_path: list[tuple[float, float]] | None = None
        self._last_dynamic_obstacles: list[tuple[float, float]] | None = None
        self._dyn_obstacle_velocity: list[tuple[float, float]] = []
        self._dyn_obstacle_still_steps: list[int] = []

        self._previous_recovery: str | None = None
        self._previous_recovery_success: bool | None = None
        self._stall_cycles = 0
        self._stall_anchor: float | None = None
        self._recovery_ever_attempted = False
        self._no_route = False

        self._watchdog_misses: deque[bool] = deque(maxlen=WATCHDOG_WINDOW_STEPS)
        window = max(2, round(MOTION_WINDOW_S / ts))
        self._window_positions: deque[tuple[float, float]] = deque(maxlen=window)
        self._window_headings: deque[float] = deque(maxlen=window)
        self._window_goal_dists: deque[float] = deque(maxlen=window)
        self._window_solver_times: deque[float] = deque(maxlen=window)
        self._window_heading_errors: deque[float] = deque(maxlen=window)
        self._deadline_miss_streak = 0
        self._reverse_run = 0
        self._stopped_run = 0

        self._recoveries: list[str] = []
        self._last_recovery_step: int | None = None
        self._hold_until_step: int | None = None

        self._route_progress = 0.0
        self._progress_route: tuple[tuple[float, float], ...] | None = None
        self._progress_remaining: float | None = None

    @property
    def evidence(self) -> dict[FailureMode, str]:
        """The failure modes to offer the decision-maker, matching what this builder reports."""
        return monitor_evidence(self.watchdog_steps, self.watchdog_margin)

    @property
    def watchdog_tripped(self) -> bool:
        """Whether `watchdog_steps` deadlines were missed within the last `WATCHDOG_WINDOW_STEPS` cycles."""
        return sum(self._watchdog_misses) >= self.watchdog_steps

    # -- what the recovery loop tells the builder --

    def record_recovery_outcome(self, method: str, success: bool, step: int, cooldown_steps: int = 0,
                                counts: bool = True) -> None:
        """A recovery was executed at `step`. `cooldown_steps` is how long it holds the robot (WAIT,
        REORIENT, a replan turning first); `counts=False` keeps it out of
        `recovery_cycles_without_progress` (a WAIT for an obstacle that is still visibly passing).
        A REPLAN_ROUTE earns a clean slate: it changed the route itself."""
        self._previous_recovery = method
        self._previous_recovery_success = success
        self._recovery_ever_attempted = True
        if method == 'REPLAN_ROUTE':
            self._stall_cycles = 0
            self._stall_anchor = None
        elif counts:
            self._stall_cycles += 1
        self._recoveries.append(f"{method}@{step}:{'ok' if success else 'failed'}")
        self._last_recovery_step = step
        self._hold_until_step = step + cooldown_steps if cooldown_steps > 0 else None

    def record_no_route(self) -> None:
        """A REPLAN_ROUTE found no route to the goal: every later report says `goal_reachable: false`."""
        self._no_route = True

    def end_hold(self, step: int) -> None:
        """A hold ended before its cooldown ran out (a turn that finished, a WAIT released early).
        From `step` on, `holding_position` reads false."""
        if self._hold_until_step is not None and step < self._hold_until_step:
            self._hold_until_step = step

    def hold_for(self, step: int, steps: int) -> None:
        """Extend the current hold to `steps` steps from `step` (a WAIT kept past its cooldown)."""
        self._hold_until_step = step + steps

    # -- per-cycle bookkeeping --

    def update(self, snapshot: RobotSnapshot) -> None:
        """Cheap per-step bookkeeping; call every control cycle."""
        self._goal_dists.append(_dist(snapshot.position, snapshot.goal))

        replanned = False
        if snapshot.reference_path and self._last_reference_path:
            n = min(len(snapshot.reference_path), len(self._last_reference_path))
            mean_shift = sum(
                _dist(snapshot.reference_path[i], self._last_reference_path[i]) for i in range(n)
            ) / n
            replanned = mean_shift > 0.1  # m; a genuine reroute rather than per-step drift of the same path
        self._replans.append(replanned)
        if snapshot.reference_path:
            self._last_reference_path = snapshot.reference_path

        # Obstacle velocities by finite differences; positional index is stable frame to frame.
        velocity_measured = bool(
            snapshot.dynamic_obstacles and self._last_dynamic_obstacles
            and len(snapshot.dynamic_obstacles) == len(self._last_dynamic_obstacles)
        )
        if velocity_measured:
            self._dyn_obstacle_velocity = [
                ((cur[0] - prev[0]) / self.ts, (cur[1] - prev[1]) / self.ts)
                for cur, prev in zip(snapshot.dynamic_obstacles, self._last_dynamic_obstacles)
            ]
        else:
            self._dyn_obstacle_velocity = [(0.0, 0.0) for _ in (snapshot.dynamic_obstacles or [])]
        self._last_dynamic_obstacles = snapshot.dynamic_obstacles

        # How long each dynamic obstacle has been standing still, in consecutive steps. Unmeasured
        # frames (no previous frame to difference against) leave the counts alone.
        if len(self._dyn_obstacle_still_steps) != len(self._dyn_obstacle_velocity):
            self._dyn_obstacle_still_steps = [0] * len(self._dyn_obstacle_velocity)
        if velocity_measured:
            for i, vel in enumerate(self._dyn_obstacle_velocity):
                if math.hypot(*vel) < self.stationary_speed_tol:
                    self._dyn_obstacle_still_steps[i] += 1
                else:
                    self._dyn_obstacle_still_steps[i] = 0

        self._window_positions.append(snapshot.position)
        self._window_headings.append(snapshot.heading)
        self._window_goal_dists.append(_dist(snapshot.position, snapshot.goal))

        if snapshot.solver_time_ms is not None:
            self._window_solver_times.append(snapshot.solver_time_ms)
            self._watchdog_misses.append(snapshot.solver_time_ms > self.ts * 1000)
            self._deadline_miss_streak = (
                self._deadline_miss_streak + 1 if snapshot.solver_time_ms > self.ts * 1000 else 0
            )

        speed = snapshot.speed if snapshot.speed is not None else (
            snapshot.action[0] if snapshot.action is not None else None)
        holding = self._hold_until_step is not None and snapshot.step < self._hold_until_step
        if holding:
            # Motion a recovery ordered is not evidence of a failure: the runs start from zero
            # again once the hold ends.
            self._reverse_run = 0
            self._stopped_run = 0
        elif speed is not None:
            self._reverse_run = self._reverse_run + 1 if speed < self.reverse_speed_tol else 0
            self._stopped_run = self._stopped_run + 1 if abs(speed) < self.stopped_speed_tol else 0

        # Per step, so route_heading_error_change_5s spans a real 5 seconds of turning.
        heading_error = self._heading_error(snapshot)
        if heading_error is not None:
            self._window_heading_errors.append(heading_error)

        self._update_route_progress(snapshot)

    def dynamic_obstacle_on_route(self, snapshot: RobotSnapshot) -> bool | None:
        """`dynamic_obstacle_blocking_path`, without building a report."""
        return self._dynamic_obstacle_blocking_path(snapshot)

    def blocking_obstacle_stationary_s(self, snapshot: RobotSnapshot) -> float | None:
        """`blocking_obstacle_stationary_s`, without building a report."""
        return self._blocking_obstacle_motion(snapshot)[1]

    # -- the report --

    def build(self, snapshot: RobotSnapshot, prior_failures: list[str]) -> SituationContext:
        """The report for this tick. `snapshot` is this cycle's, already passed to `update()`;
        `prior_failures` are the episode's earlier verdicts (`VerdictLog.prior`)."""
        # Re-anchor the stall count the moment real progress happens.
        progress = self._stall_progress(snapshot)
        if self._stall_anchor is None or progress - self._stall_anchor > self.progress_stall_tol:
            self._stall_anchor = progress
            self._stall_cycles = 0

        progress_last_3s = None
        if len(self._goal_dists) == self._goal_dists.maxlen:
            progress_last_3s = self._goal_dists[0] - self._goal_dists[-1]
        mpc_realtime = None
        if snapshot.solver_time_ms is not None:
            mpc_realtime = snapshot.solver_time_ms <= self.ts * 1000
        robot_speed = snapshot.speed
        if robot_speed is None and snapshot.action is not None:
            robot_speed = snapshot.action[0]   # signed on purpose: the sign is the reverse-tracking signal
        blocking_speed, blocking_stationary_s = self._blocking_obstacle_motion(snapshot)
        static_gap, dynamic_gap = self._obstacle_gaps(snapshot)
        holding = self._hold_until_step is not None and snapshot.step < self._hold_until_step

        return SituationContext(
            step=snapshot.step,
            prior_failures_this_episode=prior_failures,
            steps_remaining=max(0, self.max_steps - 1 - snapshot.step),
            position=snapshot.position,
            heading=snapshot.heading,
            goal_distance=_dist(snapshot.position, snapshot.goal),
            goal_bearing_deg=self._goal_bearing_deg(snapshot),
            robot_speed=robot_speed,
            route_heading_error_deg=self._heading_error(snapshot),
            solver_time_ms=snapshot.solver_time_ms,
            mpc_realtime=mpc_realtime,
            progress_last_3s=progress_last_3s,
            reference_replans_last_5s=sum(self._replans) if self._replans else None,
            reference_direction_changes=self._reference_direction_changes(snapshot.reference_path),
            min_obstacle_distance=self._min_obstacle_distance(snapshot),
            dynamic_obstacle_blocking_path=self._dynamic_obstacle_blocking_path(snapshot),
            static_obstacle_blocking_path=self._static_obstacle_blocking_path(snapshot),
            blocking_obstacle_speed=blocking_speed,
            blocking_obstacle_stationary_s=blocking_stationary_s,
            clearance_by_direction=self._clearance_by_direction(snapshot),
            previous_recovery=self._previous_recovery,
            previous_recovery_success=self._previous_recovery_success,
            recovery_cycles_without_progress=self._stall_cycles if self._recovery_ever_attempted else None,
            goal_reachable=False if self._no_route else None,
            path_length_last_5s=self._path_length(),
            net_displacement_last_5s=(_dist(self._window_positions[0], self._window_positions[-1])
                                      if len(self._window_positions) >= 2 else None),
            goal_progress_last_5s=(self._window_goal_dists[0] - self._window_goal_dists[-1]
                                   if len(self._window_goal_dists) >= 2 else None),
            heading_reversals_last_5s=self._heading_reversals(),
            reversing_for_s=self._reverse_run * self.ts,
            stopped_for_s=self._stopped_run * self.ts,
            route_heading_error_change_5s=(self._window_heading_errors[-1] - self._window_heading_errors[0]
                                           if len(self._window_heading_errors) >= 2 else None),
            robot_radius=self.robot_radius,
            static_obstacle_gap=static_gap,
            dynamic_obstacle_gap=dynamic_gap,
            solver_deadline_miss_streak=self._deadline_miss_streak,
            watchdog_steps_remaining=max(0, self.watchdog_steps - sum(self._watchdog_misses)),
            watchdog_margin=self.watchdog_margin,
            solver_deadline_misses_last_12s=sum(self._watchdog_misses),
            worst_solver_time_ms_last_5s=max(self._window_solver_times) if self._window_solver_times else None,
            control_period_ms=self.ts * 1000,
            step_budget=self.max_steps,
            holding_position=holding,
            recovery_hold_steps_remaining=(self._hold_until_step - snapshot.step) if holding else None,
            steps_since_last_recovery=(snapshot.step - self._last_recovery_step
                                       if self._last_recovery_step is not None else None),
            recoveries_this_episode=list(self._recoveries),
            route_progress_m=self._route_progress if self._progress_remaining is not None else None,
        )

    # -- measurements --

    def _stall_progress(self, snapshot: RobotSnapshot) -> float:
        """Progress for `recovery_cycles_without_progress`: along the route (`route_progress_m`)
        once a route is being measured, so driving a detour counts; before that, the negated
        straight-line distance to the goal."""
        if self._progress_remaining is None:
            return -_dist(snapshot.position, snapshot.goal)
        return self._route_progress

    def _update_route_progress(self, snapshot: RobotSnapshot) -> None:
        """Advance `route_progress_m` by this cycle's decrease in route still ahead. A cycle whose
        route differs from the last one's contributes nothing, and neither does a jump larger than
        `MAX_PROGRESS_STEP_M` (the projection snapping to another leg of the route)."""
        if len(snapshot.route_ahead) < 2:
            return
        route = tuple((float(x), float(y)) for x, y in snapshot.route_ahead)
        line = LineString(route)
        remaining = line.length - line.project(Point(snapshot.position))
        if route == self._progress_route and self._progress_remaining is not None:
            gained = self._progress_remaining - remaining
            if abs(gained) <= MAX_PROGRESS_STEP_M:
                self._route_progress += gained
        self._progress_route = route
        self._progress_remaining = remaining

    @staticmethod
    def _route_heading_error_deg(snapshot: RobotSnapshot, min_lookahead: float = 0.5) -> float | None:
        """Angle between the robot's heading and the direction the reference it is actually tracking
        leads (falling back to the intended route). The local reference comes first on purpose:
        a robot tracking a reference that leads behind it, in reverse, reads ~180 against the
        reference but ~0 against its intended route. Points nearer than `min_lookahead` are skipped:
        the bearing to a point the robot stands on is noise."""
        route = snapshot.reference_path or snapshot.route_ahead
        if not route:
            return None
        target = next((p for p in route if _dist(snapshot.position, p) >= min_lookahead), None)
        if target is None:
            return None
        bearing = math.atan2(target[1]-snapshot.position[1], target[0]-snapshot.position[0])
        error = math.atan2(math.sin(bearing-snapshot.heading), math.cos(bearing-snapshot.heading))
        return abs(math.degrees(error))

    def _heading_error(self, snapshot: RobotSnapshot) -> float | None:
        """`route_heading_error_deg`. A hold collapses the tracked reference onto the robot, which
        leaves nothing to measure against, so fall back to the intended route from where the robot
        is along it -- what a REORIENT is turning toward."""
        error = self._route_heading_error_deg(snapshot)
        if error is None and snapshot.route_ahead:
            error = self._route_heading_error_deg(
                replace(snapshot, reference_path=[], route_ahead=_route_from(snapshot.route_ahead, snapshot.position)))
        return error

    @staticmethod
    def _goal_bearing_deg(snapshot: RobotSnapshot) -> float | None:
        dx, dy = snapshot.goal[0] - snapshot.position[0], snapshot.goal[1] - snapshot.position[1]
        if math.hypot(dx, dy) < 1e-6:
            return None
        bearing = math.atan2(dy, dx)
        error = math.atan2(math.sin(bearing-snapshot.heading), math.cos(bearing-snapshot.heading))
        return math.degrees(error)

    def _blocking_obstacle_motion(self, snapshot: RobotSnapshot) -> tuple[float | None, float | None]:
        """Speed (m/s) and stationary time (s) of the dynamic obstacle on the route nearest the robot
        -- the one the robot has come up against -- or (None, None) if nothing is on the route (the
        same test as `dynamic_obstacle_blocking_path`, so the fields never contradict it)."""
        path = snapshot.route_ahead or snapshot.reference_path
        if (not path or len(path) < 2 or not snapshot.dynamic_obstacles
                or len(snapshot.dynamic_obstacles) != len(self._dyn_obstacle_velocity)):
            return None, None
        line = LineString(path)
        distances = [line.distance(Point(obs)) for obs in snapshot.dynamic_obstacles]
        idx = min(range(len(distances)), key=distances.__getitem__)
        if distances[idx] >= self.obstacle_block_margin:
            return None, None
        on_route = [i for i, d in enumerate(distances) if d < self.obstacle_block_margin]
        idx = min(on_route, key=lambda i: _dist(snapshot.position, snapshot.dynamic_obstacles[i]))
        speed = math.hypot(*self._dyn_obstacle_velocity[idx])
        stationary_s = None
        if idx < len(self._dyn_obstacle_still_steps):
            stationary_s = self._dyn_obstacle_still_steps[idx] * self.ts
        return speed, stationary_s

    def _clearance_by_direction(self, snapshot: RobotSnapshot) -> dict[str, float] | None:
        """Distance to the nearest permanent geometry in each of eight robot-relative sectors.
        Wedges rather than rays, so a gap too narrow for the robot never reads as open."""
        geometries: list = [Polygon(obs) for obs in snapshot.obstacles]
        if snapshot.boundary and len(snapshot.boundary) >= 3:
            # As a ring: it is the wall that blocks, not the free space inside it.
            geometries.append(LineString(list(snapshot.boundary) + [snapshot.boundary[0]]))
        if not geometries:
            return None
        origin = Point(snapshot.position)
        reach = self.sector_scan_range
        count = len(_SECTOR_NAMES)
        half_width = math.pi / count
        clearance: dict[str, float] = {}
        for i, name in enumerate(_SECTOR_NAMES):
            centre = snapshot.heading + i * 2 * math.pi / count
            arc = [
                (snapshot.position[0] + reach*math.cos(a), snapshot.position[1] + reach*math.sin(a))
                for a in (centre - half_width + k * (2*half_width) / 8 for k in range(9))
            ]
            wedge = Polygon([snapshot.position, *arc])
            nearest = reach
            for geom in geometries:
                overlap = wedge.intersection(geom)
                if not overlap.is_empty:
                    nearest = min(nearest, overlap.distance(origin))
            clearance[name] = nearest
        return clearance

    def _min_obstacle_distance(self, snapshot: RobotSnapshot) -> float | None:
        point = Point(snapshot.position)
        dists = [Polygon(obs).distance(point) for obs in snapshot.obstacles]
        dists += [_dist(snapshot.position, obs) for obs in snapshot.dynamic_obstacles]
        return min(dists) if dists else None

    def _dynamic_obstacle_blocking_path(self, snapshot: RobotSnapshot) -> bool | None:
        # The intended route rather than the tracked reference, which a WAIT collapses to a point.
        path = snapshot.route_ahead or snapshot.reference_path
        if not path or len(path) < 2 or not snapshot.dynamic_obstacles:
            return None
        line = LineString(path)
        return any(line.distance(Point(obs)) < self.obstacle_block_margin for obs in snapshot.dynamic_obstacles)

    def _static_obstacle_blocking_path(self, snapshot: RobotSnapshot) -> bool | None:
        path = snapshot.route_ahead or snapshot.reference_path
        if not path or len(path) < 2 or not snapshot.obstacles:
            return None
        line = LineString(path)
        return any(line.distance(Polygon(obs)) < self.static_obstacle_block_margin for obs in snapshot.obstacles)

    @staticmethod
    def _reference_direction_changes(reference_path: list[tuple[float, float]], min_angle: float = 0.2) -> int | None:
        """Heading reversals along the shape of the current reference path."""
        if not reference_path or len(reference_path) < 3:
            return None
        headings = [math.atan2(b[1]-a[1], b[0]-a[0]) for a, b in zip(reference_path, reference_path[1:])]
        deltas = [math.atan2(math.sin(b-a), math.cos(b-a)) for a, b in zip(headings, headings[1:])]
        return sum(
            1 for d1, d2 in zip(deltas, deltas[1:])
            if d1*d2 < 0 and abs(d1) > min_angle and abs(d2) > min_angle
        )

    def _path_length(self) -> float | None:
        if len(self._window_positions) < 2:
            return None
        points = list(self._window_positions)
        return sum(_dist(a, b) for a, b in zip(points, points[1:]))

    def _heading_reversals(self) -> int | None:
        if len(self._window_headings) < 3:
            return None
        headings = list(self._window_headings)
        deltas = [math.atan2(math.sin(b-a), math.cos(b-a)) for a, b in zip(headings, headings[1:])]
        return sum(
            1 for d1, d2 in zip(deltas, deltas[1:])
            if d1*d2 < 0 and abs(d1) > self.min_reversal_angle and abs(d2) > self.min_reversal_angle
        )

    def _obstacle_gaps(self, snapshot: RobotSnapshot) -> tuple[float | None, float | None]:
        """Signed clearance (m) from the footprint edge to the nearest static and dynamic obstacle."""
        static_gap = None
        if snapshot.obstacles:
            point = Point(snapshot.position)
            static_gap = min(Polygon(o).distance(point) for o in snapshot.obstacles) - self.robot_radius
        dynamic_gap = None
        if snapshot.dynamic_obstacles:
            dynamic_gap = (min(_dist(snapshot.position, o) for o in snapshot.dynamic_obstacles)
                           - self.robot_radius - self.dynamic_obstacle_radius)
        return static_gap, dynamic_gap
