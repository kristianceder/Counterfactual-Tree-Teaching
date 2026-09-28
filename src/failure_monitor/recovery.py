"""The recovery actions the decision-maker chooses between, and what each one does to the live
`TrajectoryGenerator`.

One self-contained `RecoveryStrategy` per `RecoveryMethod`: its `description` is shown to the
decision-maker verbatim (see `recovery_candidates`), its `execute()` acts on the controller. The
MPC/path-planning imports are deferred into `execute()` so importing `failure_monitor` does not
need the compiled solver.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from shapely.geometry import Point, Polygon  # type: ignore

if TYPE_CHECKING:
    from mpc_traj_tracker import TrajectoryGenerator
    from path_planning import GeometricMap


def _nearest_free_point(point: tuple[float, float], obstacles: list[Polygon], boundary: Polygon,
                         max_radius: float = 1.0, radius_step: float = 0.05, directions: int = 16) -> tuple[float, float]:
    """If `point` touches or overlaps an obstacle (or sits outside `boundary`),
    search an expanding ring of nearby points for one that's clear.

    `VisibilityPathFinder` refuses a start/end point that isn't strictly inside
    free space -- but a robot the MPC has pinned right against its own inflated
    safety margin (exactly the STUCK scenario this module exists for) sits
    *touching* that margin's boundary by construction, which shapely's
    `intersects()` (unlike a strict interior test) counts as "in the obstacle".
    Returns `point` unchanged if no clear point is found within `max_radius`,
    letting the caller's own `ValueError` handling take over.
    """
    def is_free(p: Point) -> bool:
        return boundary.contains(p) and not any(o.intersects(p) for o in obstacles)

    if is_free(Point(point)):
        return point
    radius = radius_step
    while radius <= max_radius:
        for k in range(directions):
            angle = 2 * math.pi * k / directions
            candidate = Point(point[0] + radius * math.cos(angle), point[1] + radius * math.sin(angle))
            if is_free(candidate):
                return (candidate.x, candidate.y)
        radius += radius_step
    return point


def _remaining_route(route: list[tuple[float, float]],
                      position: tuple[float, float]) -> list[tuple[float, float]]:
    """Trim `route` to the portion from the waypoint nearest `position` onward -- resuming a
    known route should pick up from here, not restart at its original beginning, which by the
    time a resume is actually decided on may be well behind the robot (e.g. the start of a
    REPLAN_ROUTE from several steps/a WAIT hold ago). Falls back to a direct line to the
    route's own endpoint if trimming would leave fewer than 2 points (e.g. the robot is
    already essentially at the end) -- `set_ref_trajectory`'s node-walking needs at least 2.
    """
    if len(route) < 2:
        return route
    idx = min(range(len(route)), key=lambda i: math.hypot(route[i][0]-position[0], route[i][1]-position[1]))
    remaining = route[idx:]
    return remaining if len(remaining) >= 2 else [position, route[-1]]


def _bearing_along(points, position: tuple[float, float], min_lookahead: float) -> float | None:
    """Bearing (rad) from `position` to the first of `points` at least `min_lookahead` away, or
    `None` if there is none.

    The same computation `SituationBuilder._route_heading_error_deg` reports as
    `route_heading_error_deg`, so a turn aimed with this corrects exactly the angle the report
    showed; the tangent stored at the reference's current index can disagree with it by tens of
    degrees.
    """
    for p in points:
        if math.hypot(p[0]-position[0], p[1]-position[1]) >= min_lookahead:
            return math.atan2(p[1]-position[1], p[0]-position[0])
    return None


def _start_turn(traj_gen: "TrajectoryGenerator", heading: float) -> float:
    """Hold position and turn on the spot toward `heading`. Returns the angle (rad) to correct.

    Shared by REORIENT and by REPLAN_ROUTE's turn-first case. The hold ends as soon as the robot
    is aligned (`run_episode` checks `TrajectoryGenerator.heading_override_error` every cycle), so
    a strategy's `cooldown_steps` is only a cap.
    """
    traj_gen.holding = True
    traj_gen.speed_ref_override = [0.0] * traj_gen.N_hor
    traj_gen.heading_override = heading
    current = float(traj_gen.state[2])
    return abs(math.atan2(math.sin(heading-current), math.cos(heading-current)))


TURN_ALIGNED_RAD = math.radians(10.0)
"""A turn-in-place hold counts as done once the heading is within this of its target. Callers
releasing the hold early read it from here, so the strategies and the loop agree on "aligned"."""


class RecoveryMethod(Enum):
    """The recovery actions the decision-maker can choose between."""
    REPLAN_ROUTE = "REPLAN_ROUTE"  # recompute the reference path around the (now known) obstacle
    WAIT = "WAIT"                  # hold position for a while, e.g. for a transient obstruction to clear
    RESUME_ROUTE = "RESUME_ROUTE"  # pick the already-known route back up from here, no recomputation
    REORIENT = "REORIENT"          # turn on the spot to face along the route, then carry on
    REQUEST_HUMAN = "REQUEST_HUMAN"  # stop and hand the situation to a human supervisor; ends autonomous operation
    CONTINUE = "CONTINUE"          # no corrective action; keep tracking the current reference


@dataclass
class RecoveryOutcome:
    """What actually happened when a `RecoveryStrategy` was run."""
    method: RecoveryMethod
    success: bool
    detail: str
    cooldown_steps: int = 0
    """If >0, how many control cycles the recovery holds the robot (WAIT, REORIENT, a replan that
    turns to face its new route first) before the hold lapses on its own."""
    no_route: bool = False
    """Set by `ReplanRouteStrategy` when the static map has no route to the goal at all. The caller
    reports it (`SituationBuilder.record_no_route`, `goal_reachable: false`) and leaves calling for
    help to the decision-maker."""
    new_route: list[tuple[float, float]] | None = None
    """The new intended route, if this recovery changed it (REPLAN_ROUTE). The caller feeds it into
    `RobotSnapshot.route_ahead`, so obstacle-ahead checks keep looking at where the robot is going
    even while a later WAIT freezes the tracked reference to a point."""


@dataclass
class RecoveryContext:
    """Everything a `RecoveryStrategy` might need to act, gathered by the caller each time a
    recovery is executed."""
    traj_gen: "TrajectoryGenerator"
    goal: tuple[float, float]
    geo_map: "GeometricMap | None" = None  # needed by REPLAN_ROUTE; other strategies ignore it
    route_ahead: list[tuple[float, float]] | None = None
    """The robot's current *intended* route (see `RecoveryOutcome.new_route`/`RobotSnapshot.route_ahead`),
    if the caller tracks one -- needed by RESUME_ROUTE to pick it back up without recomputing it via
    REPLAN_ROUTE's visibility-graph search. `None`/empty and RESUME_ROUTE just fails cleanly
    (`RecoveryOutcome.success=False`) rather than guessing a route."""


class RecoveryStrategy(ABC):
    """One self-contained class per recovery action. `description` is shown to the decision-maker
    verbatim -- guidance on *when* to pick this method, not just what it does."""
    method: RecoveryMethod
    description: str

    @abstractmethod
    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        ...


class ReplanRouteStrategy(RecoveryStrategy):
    """Recompute the reference path from the robot's current position to the
    goal via the same visibility-graph planner that generated the *initial*
    reference (`path_planning.LocalPathPlanner`) -- but against `ctx.geo_map`,
    which (unlike the environment's own initial-reference search) already
    includes every static obstacle the MPC's own collision constraints are
    built from, whether or not it was considered "visible" when the very
    first reference was planned. That gap -- an obstacle the MPC must avoid
    but the reference planner never routed around -- is exactly what traps
    the robot at a hidden wall.
    """
    method = RecoveryMethod.REPLAN_ROUTE
    description = (
        "Recompute the route to the goal from scratch around the obstacle (an expensive visibility-graph "
        "search over *static* obstacles only -- it has no notion of dynamic ones, so it may simply "
        "recompute a route that runs right back through wherever a dynamic obstacle currently happens to "
        "be). Use this when the *route itself* is wrong -- static_obstacle_blocking_path: true is the "
        "clear, direct signal for this: the route was planned straight through real obstacle geometry, so "
        "it genuinely cannot work no matter how long the robot waits or how many times it resumes. "
        "mpc_feasible: false (the solver failing to converge) is corroborating evidence of the same thing. "
        "Also the right call if recovery_cycles_without_progress keeps climbing (2+) AND "
        "static_obstacle_blocking_path or mpc_feasible points at a real route problem, not just a dynamic "
        "obstacle that hasn't cleared yet -- see WAIT's description for why a *purely* dynamic blockage "
        "(dynamic_obstacle_blocking_path: true, static_obstacle_blocking_path: false) should not be "
        "escalated to REPLAN_ROUTE by cycle count alone: this search cannot see that obstacle, so it is not "
        "a reliable way to get around one that's only in the way right now. If the route is still fine and "
        "the robot has simply paused (e.g. after a WAIT), use RESUME_ROUTE instead -- much cheaper, and "
        "doesn't throw away a route that was already correct. "
        "clearance_by_direction says whether there is anywhere for a new route to go: it gives the open "
        "distance (m) in eight directions relative to the robot's own heading, counting permanent "
        "geometry only (static obstacles and walls). Read it together with goal_bearing_deg, the "
        "direction of the goal in that same frame: an open sector on the side the goal lies is room for "
        "the detour this search would find, while a robot whose only open sectors are behind it is in a "
        "pocket, and the new route will have to lead back out the way it came before making progress "
        "again -- correct, but expect it to look like going the wrong way at first."
    )

    def __init__(self, mode: str = 'work', turn_first_above_deg: float = 90.0, turn_hold_steps: int = 40,
                 min_lookahead: float = 0.5):
        """
        Args:
            mode: `TrajectoryGenerator` work mode ("safe"/"work"/"super") to build the new
                reference trajectory's node spacing at -- see `execute()`.
            turn_first_above_deg: if the new route leaves in a direction more than this far from
                where the robot is facing, turn on the spot to face it before driving. Otherwise
                the MPC takes the cheaper option of tracking the route in reverse. On the randomized
                grid that happened after 4 of 10 replans, for 4-9 s at a time, and the resulting
                REVERSE_TRACKING report then got the replan itself judged a failure.
            turn_hold_steps: cap on that turn's hold, same as `ReorientStrategy.hold_steps`. It
                ends earlier once the robot is aligned.
            min_lookahead: as `ReorientStrategy`: how far along the new route to look for its
                direction.
        """
        self.mode = mode
        self.turn_first_above_deg = turn_first_above_deg
        self.turn_hold_steps = turn_hold_steps
        self.min_lookahead = min_lookahead

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        if ctx.geo_map is None:
            return RecoveryOutcome(self.method, success=False, detail="No map available to replan against.")

        import networkx as nx  # deferred, see module docstring
        from path_planning import LocalPathPlanner  # deferred, see module docstring

        obstacle_polys = [Polygon(o) for o in ctx.geo_map.processed_obstacle_list]
        boundary_poly = Polygon(ctx.geo_map.processed_boundary_coords)
        position = (float(ctx.traj_gen.state[0]), float(ctx.traj_gen.state[1]))
        position = _nearest_free_point(position, obstacle_polys, boundary_poly)
        try:
            new_path = LocalPathPlanner(ctx.geo_map).get_ref_path(position, ctx.goal)
        except ValueError as e:
            # A positional edge case (start/end pinned in an obstacle or outside the boundary) --
            # may well resolve itself next step as the robot's own state changes, so this is left
            # as an ordinary retryable failure, not a no-route finding.
            return RecoveryOutcome(self.method, success=False, detail=f"Replanning failed: {e}")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # NetworkXNoPath: start and goal are both in the graph but disconnected from each
            # other. NodeNotFound: start or goal has *no* visibility edges at all (e.g. fully
            # enclosed), so it was never even added as a graph node. Either way this means the
            # visibility graph -- built from every *static* obstacle, which by definition never
            # moves -- has no route between here and the goal: full stop, not just right now but
            # ever. Retrying (REPLAN_ROUTE again, WAIT, RESUME_ROUTE) cannot change a graph-
            # connectivity fact. Stop the robot here rather than leaving it tracking its (already
            # broken) reference, and report the finding: the report then says goal_reachable: false.
            ctx.traj_gen.holding = True
            ctx.traj_gen.speed_ref_override = [0.0] * ctx.traj_gen.N_hor
            return RecoveryOutcome(
                self.method, success=False, no_route=True,
                detail="No route to the goal exists in the visibility graph -- the static obstacles "
                       "fully block every path. Stopping in place; this needs a human supervisor, "
                       "not another recovery attempt.",
            )

        # `set_ref_trajectory` spaces the new trajectory's nodes using whatever
        # `base_speed` is currently set (see `TrajectoryGenerator.get_global_ref_traj`)
        # -- e.g. still whatever "safe"/low-speed mode was active while the robot was
        # stuck. If a *faster* mode is used to track it afterwards (as run_episode
        # does once recovered), that mismatch leaves the local reference horizon
        # spanning far less distance than the now-higher commanded speed expects,
        # which reads as a short/near-stationary reference and a robot that
        # overshoots it and has to back up. Setting the mode here first keeps the
        # node spacing consistent with the speed it will actually be tracked at.
        ctx.traj_gen.set_work_mode(self.mode)
        ctx.traj_gen.set_ref_trajectory(new_path)
        new_route = [(float(p.x), float(p.y)) for p in new_path]
        detail = (f"Replanned a {len(new_path)}-waypoint route to the goal via the visibility graph, "
                  f"now accounting for the obstacle that trapped the robot.")

        # A new route that leaves behind the robot gets tracked in reverse unless the robot turns
        # to face it first. See `turn_first_above_deg`.
        heading = _bearing_along(new_route, (float(ctx.traj_gen.state[0]), float(ctx.traj_gen.state[1])),
                                 self.min_lookahead)
        current = float(ctx.traj_gen.state[2])
        if heading is not None:
            off = math.degrees(abs(math.atan2(math.sin(heading-current), math.cos(heading-current))))
            if off > self.turn_first_above_deg:
                _start_turn(ctx.traj_gen, heading)
                return RecoveryOutcome(
                    self.method, success=True,
                    detail=f"{detail} The new route leaves {off:.0f} deg from the robot's heading, so it "
                           f"turns to face it first (holding up to {self.turn_hold_steps} steps).",
                    cooldown_steps=self.turn_hold_steps,
                    new_route=new_route,
                )
        return RecoveryOutcome(self.method, success=True, detail=detail, new_route=new_route)


class WaitStrategy(RecoveryStrategy):
    """Hold the current position for `hold_steps` control cycles -- appropriate
    for a transient obstruction (e.g. a dynamic obstacle currently blocking the
    path) that's expected to clear on its own, unlike a structural one (see
    `ReplanRouteStrategy` for that).

    Deliberately does *not* touch the robot's intended route to the goal --
    only `TrajectoryGenerator.holding`/`speed_ref_override`, which affect what's
    tracked moment-to-moment (see `execute()`). `RecoveryOutcome.new_route` is left
    `None` here for exactly that reason: the route this recovery leaves behind
    (whatever `ReplanRouteStrategy` last set, or the original one) is still correct,
    it's just paused, and a caller building reports should keep checking
    obstacles against that real route (`RobotSnapshot.route_ahead`), not wherever the
    robot happens to be holding -- otherwise the very act of waiting blinds the next
    `dynamic_obstacle_blocking_path` check to the obstacle it's waiting on.
    """
    method = RecoveryMethod.WAIT
    description = (
        "Hold the current position and wait. Use this when the blockage looks transient -- "
        "dynamic_obstacle_blocking_path: true AND static_obstacle_blocking_path: false (or absent) -- a "
        "dynamic obstacle currently in the way, but the route itself is fine and doesn't need to change. "
        "If static_obstacle_blocking_path is true, the route runs through real obstacle geometry -- WAIT "
        "will never fix that no matter how long you hold, use REPLAN_ROUTE instead. Repeated WAITs against "
        "the same dynamic obstacle are normal and fine, even as recovery_cycles_without_progress climbs -- "
        "do NOT switch to REPLAN_ROUTE on cycle count alone while the blockage is purely dynamic "
        "(dynamic_obstacle_blocking_path: true, static_obstacle_blocking_path: false): REPLAN_ROUTE's "
        "visibility-graph search only sees static obstacles, so it has no way to route around a dynamic "
        "one and may simply recompute a route straight back through this exact spot. Only move off WAIT "
        "once static_obstacle_blocking_path becomes true or mpc_feasible becomes false -- a real sign the "
        "route itself, not just this moment, is the problem. Once the blockage has actually cleared, "
        "prefer RESUME_ROUTE over waiting further. "
    )
    parked_advice = (
        "Before holding again, check that the obstacle is actually going to move: "
        "blocking_obstacle_speed is how fast the thing in the way is travelling, and "
        "blocking_obstacle_stationary_s is how long it has been standing still. A moving obstacle "
        "(nonzero speed) will clear on its own and is worth waiting for; one that has been stationary "
        "for several seconds has stopped, not paused, and is functionally a static obstacle no amount of "
        "further waiting will move -- keep that in mind against steps_remaining, since each WAIT spends "
        "about 25 steps of the episode's remaining budget, and waiting out a parked obstacle simply "
        "converts the blockage into a timeout."
    )
    """The last part of `description` without REQUEST_HUMAN on offer; `recovery_candidates` swaps in
    `parked_advice_with_human`, which is what the decision-maker reads. Kept as text so the offered
    description is assembled exactly as it was when the recorded runs were made."""
    parked_advice_with_human = (
        "How long to go on waiting is read off blocking_obstacle_stationary_s, the time the thing in "
        "the way has been standing still. Up to about 25 s it has paused, not stopped: a vehicle "
        "halted in a doorway, a gate that is shut, a pedestrian who has stopped to talk. Keep waiting "
        "-- repeated WAITs are right here, REPLAN_ROUTE cannot see a dynamic obstacle and returns the "
        "route the robot already has, and it is far too early for REQUEST_HUMAN. At 30 s or more, with "
        "a WAIT already in recoveries_this_episode, it is not going to move and the answer is "
        "REQUEST_HUMAN rather than another hold."
    )
    solver_advice = (
        " WAIT is also the fix for a solver that is missing its deadlines against a dynamic obstacle "
        "(SOLVER_DEADLINE_MISS with dynamic_obstacle_blocking_path: true): the robot is being asked to "
        "drive through something it must avoid, and a hold pins the reference to where the robot "
        "already is, so the solve time falls back within a step. Do not answer CONTINUE to that "
        "report: the watchdog counts every missed deadline and ends the mission when "
        "watchdog_steps_remaining reaches 0."
    )
    description = description + parked_advice

    def __init__(self, hold_steps: int = 25):
        self.hold_steps = hold_steps

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        # `holding=True` makes `get_local_ref_traj()` re-anchor to the robot's *current* state
        # every call, not a fixed point recorded once here -- see its docstring for why a
        # one-shot snapshot caused a reverse correction the first time this was tried. Also
        # stop the *speed* reference from still pulling forward: run_step()'s default speed_ref
        # is a function of distance to the (unmoved) final goal, so left alone it stays a
        # nonzero target the whole time this hold is active, fighting the position hold instead
        # of agreeing with it and settling on a small drifting compromise rather than a genuine
        # stop. See TrajectoryGenerator.run_step()/get_local_ref_traj().
        ctx.traj_gen.holding = True
        ctx.traj_gen.speed_ref_override = [0.0] * ctx.traj_gen.N_hor
        return RecoveryOutcome(
            self.method, success=True,
            detail=f"Holding position for up to {self.hold_steps} steps.",
            cooldown_steps=self.hold_steps,
        )


class ResumeRouteStrategy(RecoveryStrategy):
    """Pick the already-known route (`ctx.route_ahead` -- the original route, or the result of
    the last `ReplanRouteStrategy` run) back up from the robot's current position, with no
    recomputation. This is the missing middle ground between `WaitStrategy` (freeze in place
    forever, one cooldown at a time) and `ReplanRouteStrategy` (an expensive visibility-graph
    search that also discards whatever route was already correct): once a transient blockage
    has actually cleared, or a WAIT's cooldown has simply run out with nothing left blocking the
    path, this is the cheap way to un-pause and get moving again along a route that was never
    actually wrong.
    """
    method = RecoveryMethod.RESUME_ROUTE
    description = (
        "Resume the already-known route (the original one, or the last REPLAN_ROUTE's result) from the "
        "robot's current position -- no recomputation. Use this once a transient blockage that caused an "
        "earlier WAIT has actually cleared, or a WAIT's hold just ran out and nothing is blocking the path "
        "anymore: dynamic_obstacle_blocking_path false AND static_obstacle_blocking_path false (or absent) "
        "together are exactly that signal. If static_obstacle_blocking_path is true, resuming will just "
        "drive the robot right back into the same obstacle -- pick REPLAN_ROUTE instead, even though it's "
        "more expensive. Same if recovery_cycles_without_progress is already 2 or more despite a clear "
        "report right now -- a route that keeps leading back into trouble needs to actually change, not "
        "just get re-attached to again."
    )

    def __init__(self, mode: str = 'work'):
        """
        Args:
            mode: `TrajectoryGenerator` work mode ("safe"/"work"/"super") to build the resumed
                reference trajectory's node spacing at -- see `ReplanRouteStrategy` for why this
                matters (same reasoning applies here).
        """
        self.mode = mode

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        if not ctx.route_ahead or len(ctx.route_ahead) < 2:
            return RecoveryOutcome(self.method, success=False,
                                    detail="No known route to resume (route_ahead unset or too short).")

        position = (float(ctx.traj_gen.state[0]), float(ctx.traj_gen.state[1]))
        remaining = _remaining_route(ctx.route_ahead, position)

        # Same reasoning as ReplanRouteStrategy: set the mode before rebuilding the trajectory,
        # so node spacing matches the speed it's actually tracked at afterwards.
        ctx.traj_gen.set_work_mode(self.mode)
        ctx.traj_gen.set_ref_trajectory(remaining)
        return RecoveryOutcome(
            self.method, success=True,
            detail=f"Resumed the existing route ({len(remaining)} waypoints from here to the goal), "
                   f"no replanning needed.",
            # Not new_route: this is *the same* intended route, just re-attached to -- nothing for
            # RobotSnapshot.route_ahead to update.
        )


class ReorientStrategy(RecoveryStrategy):
    """Turn on the spot until the robot faces along its route, then let it drive on.

    The missing action for a failure that is about *orientation*, not *routing*.
    `REPLAN_ROUTE`/`RESUME_ROUTE` both answer "which way should I go"; `WAIT`
    answers "should I go at all right now". None of them can fix a robot whose
    route is perfectly good and which is simply pointed the wrong way along it --
    the state the MPC lands in when it settles into tracking a route in reverse (for a
    unicycle both directions track the same path at almost the same cost).

    Mechanism: hold position (`holding`, so cross-track error is zero by
    construction) with the speed reference zeroed like `WaitStrategy`, but set
    `TrajectoryGenerator.heading_override` to the route's own direction. That
    leaves the heading cost as the only term with a gradient, so the solver spends
    the hold rotating the robot in place rather than translating it. Once it is
    facing the right way, forward tracking is unambiguously the cheaper option and
    the reverse branch stops being reachable by warm start -- which is what makes
    this a real fix rather than another nudge of `qtheta`.

    Deliberately does not touch the route itself: `RecoveryOutcome.new_route` stays
    `None`, exactly like `WaitStrategy`, because the intended route was never the
    problem.
    """
    method = RecoveryMethod.REORIENT
    description = (
        "Turn the robot on the spot to face along its route, without moving or changing the route. "
        "Use this when the route is fine but the robot is pointed the wrong way along it -- the direct "
        "signal is a large route_heading_error_deg (roughly 120 or more, i.e. facing sideways-to-"
        "backwards relative to where the route leads), especially together with a negative robot_speed, "
        "which together mean the robot is driving its route in reverse instead of turning around. That "
        "combination will not improve on its own and no amount of re-routing fixes it: REPLAN_ROUTE "
        "would just hand back a route the robot is still facing away from, and RESUME_ROUTE would "
        "re-attach to the same one. Prefer this over REPLAN_ROUTE whenever "
        "static_obstacle_blocking_path is false and the real problem is which way the robot is facing "
        "rather than where the route goes."
    )

    def __init__(self, hold_steps: int = 40, min_lookahead: float = 0.5):
        """
        Args:
            hold_steps: How long to hold while turning. A half-turn at the default `ang_vel_max`
                (0.5 rad/s) takes pi/0.5 ~ 6.3 s ~ 32 steps at a 0.2 s control period, so the
                default leaves headroom for the worst case (a full 180 deg reversal) plus
                acceleration limits. The caller re-evaluates when this cooldown elapses.
            min_lookahead: Ignore route waypoints nearer than this when working out which way the
                route actually leads -- the bearing to a point the robot is standing on is noise.
                Matches `SituationBuilder._route_heading_error_deg`, so the angle this
                strategy corrects is the same one the report showed the model.
        """
        self.hold_steps = hold_steps
        self.min_lookahead = min_lookahead

    def _reference_direction(self, ctx: RecoveryContext) -> float | None:
        """The direction the reference the MPC is currently tracking leads, measured the way the
        report measures it (see `_bearing_along`).

        Read off the global reference from `idx_ref` onward, which is what the MPC's local
        reference is cut from. Read-only on purpose: `get_local_ref_traj()` would give a similar
        answer but also advances `idx_ref` as a side effect, which a recovery has no business
        doing.
        """
        traj = getattr(ctx.traj_gen, 'ref_traj', None)
        if traj is None or len(traj) == 0:
            return None
        arr = traj.numpy()
        idx = min(max(int(getattr(ctx.traj_gen, 'idx_ref', 0)), 0), len(arr)-1)
        position = (float(ctx.traj_gen.state[0]), float(ctx.traj_gen.state[1]))
        return _bearing_along(((float(x), float(y)) for x, y in arr[idx:, :2]), position, self.min_lookahead)

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        position = (float(ctx.traj_gen.state[0]), float(ctx.traj_gen.state[1]))
        desired = self._reference_direction(ctx)
        if desired is None:
            # Fall back to the intended route's own bearing. Deliberately second: after a
            # REPLAN_ROUTE out of a dead end, the reference legitimately leads *back* the way the
            # robot came, while `route_ahead`'s next waypoints can still read as roughly straight
            # ahead -- measuring against those says "0 deg to correct" at the exact moment the
            # robot is reversing, and turns this strategy into a 40-step no-op (observed).
            route = ctx.route_ahead
            target = next((p for p in route
                           if math.hypot(p[0]-position[0], p[1]-position[1]) >= self.min_lookahead),
                          None) if route else None
            if target is None:
                return RecoveryOutcome(self.method, success=False,
                                        detail="No reference or route direction available to turn toward.")
            desired = math.atan2(target[1]-position[1], target[0]-position[0])

        error = _start_turn(ctx.traj_gen, desired)
        return RecoveryOutcome(
            self.method, success=True,
            detail=f"Turning in place to face along the route: {math.degrees(error):.0f} deg "
                   f"to correct, holding up to {self.hold_steps} steps.",
            cooldown_steps=self.hold_steps,
        )


class ContinueStrategy(RecoveryStrategy):
    """No corrective action. A deliberate no-op is a real decision (e.g. the
    flagged condition looks transient/benign already, or none of the other
    candidates fit) rather than a missing one, so it's a first-class method.
    """
    method = RecoveryMethod.CONTINUE
    description = (
        "Take no corrective action and keep tracking the current reference. Use this only if the report "
        "doesn't actually call for a route change or a pause."
    )

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        return RecoveryOutcome(self.method, success=True, detail="No corrective action taken.", cooldown_steps=10)


class RequestHumanStrategy(RecoveryStrategy):
    """Stop in place and hand the situation to a human supervisor. Terminal: the episode ends as
    "human_requested", and whether the call was right is a fact about the layout
    (`ScenarioLayout.needs_human`). It has the two ways of being wrong every other action has:
    ordered against a blockage that would have cleared (a false alarm, which costs the mission) or
    not ordered against one that never will.
    """
    method = RecoveryMethod.REQUEST_HUMAN
    description = (
        "Stop the robot and call a human supervisor. This ends autonomous operation: the mission is "
        "handed over and no other recovery runs after it, so it is only right when nothing the robot "
        "can do by itself will get it to the goal. Two situations qualify. (1) goal_reachable: false -- "
        "a REPLAN_ROUTE has already searched the map of permanent obstacles and found no route to the "
        "goal at all. Replanning again searches the same map and fails the same way, and waiting does "
        "not move a wall: call the human. (2) The only way forward is held by an obstacle that is not "
        "going to leave: dynamic_obstacle_blocking_path: true with blocking_obstacle_stationary_s of "
        "30 s or more, after WAIT has already been tried (see recoveries_this_episode) and with no "
        "open detour in clearance_by_direction. A pedestrian or vehicle that pauses in a doorway "
        "stands still for up to 20 s and then moves on; something that has not moved for 30 s or more has "
        "broken down or been left there. Do NOT use this for a blockage that has only just started, "
        "for a robot that is merely slow, or while goal_reachable is true and "
        "static_obstacle_blocking_path is true (that is a REPLAN_ROUTE: a route exists and has not "
        "been tried). A false alarm costs the whole mission, so when a shorter remedy has not been "
        "tried yet, try it first."
    )

    def execute(self, ctx: RecoveryContext) -> RecoveryOutcome:
        # Stopped the same way ReplanRouteStrategy stops a robot it has no route for.
        ctx.traj_gen.holding = True
        ctx.traj_gen.speed_ref_override = [0.0] * ctx.traj_gen.N_hor
        return RecoveryOutcome(self.method, success=True,
                               detail="Stopped in place and called a human supervisor.")


DEFAULT_STRATEGIES: dict[RecoveryMethod, RecoveryStrategy] = {
    s.method: s for s in [ReplanRouteStrategy(), WaitStrategy(), ResumeRouteStrategy(),
                          ReorientStrategy(), RequestHumanStrategy(), ContinueStrategy()]
}


def recovery_candidates(watchdog_margin: int = 20) -> dict[RecoveryMethod, str]:
    """The recoveries offered to the decision-maker, each with the description it is shown.

    REQUEST_HUMAN is listed first, and the two recoveries it takes over from say where they stop
    applying: offered last and unqualified, it was never chosen against `goal_reachable: false`.
    WAIT's description is completed with what to do about a solver missing its deadlines, and where
    a hold on an open road stops being worth it (`watchdog_margin`, see `report.WATCHDOG_MARGIN_STEPS`).
    """
    offered = {method: strategy.description for method, strategy in DEFAULT_STRATEGIES.items()}
    offered[RecoveryMethod.WAIT] += WaitStrategy.solver_advice
    offered[RecoveryMethod.WAIT] += (
        " On an open road (SOLVER_DEADLINE_MISS with neither blocking flag true) a hold is only the "
        f"last resort: WAIT there once watchdog_steps_remaining is {watchdog_margin} or less, and "
        "CONTINUE above that."
    )
    offered = {RecoveryMethod.REQUEST_HUMAN: offered.pop(RecoveryMethod.REQUEST_HUMAN), **offered}
    offered[RecoveryMethod.REPLAN_ROUTE] += (
        " One exception overrides all of the above: goal_reachable: false means a REPLAN_ROUTE has "
        "already searched this same map of permanent obstacles and found no route at all. Running it "
        "again searches the same map and finds the same nothing, however clearly "
        "static_obstacle_blocking_path says the route is blocked. Never pick REPLAN_ROUTE while "
        "goal_reachable is false; that is REQUEST_HUMAN's case."
    )
    offered[RecoveryMethod.WAIT] = offered[RecoveryMethod.WAIT].replace(
        WaitStrategy.parked_advice, WaitStrategy.parked_advice_with_human)
    return offered


RECOVERY_METHOD_DESCRIPTIONS: dict[RecoveryMethod, str] = recovery_candidates()


class RecoveryManager:
    """Looks up and runs the `RecoveryStrategy` for a chosen `RecoveryMethod`."""
    def __init__(self, strategies: dict[RecoveryMethod, RecoveryStrategy] | None = None):
        self.strategies = dict(strategies) if strategies is not None else dict(DEFAULT_STRATEGIES)

    def execute(self, method: RecoveryMethod, ctx: RecoveryContext) -> RecoveryOutcome:
        # Reset all three unconditionally, not just for non-WAIT methods: `WaitStrategy`/
        # `ReorientStrategy` set them back themselves, right after this call, if one of them is
        # what's actually chosen -- so a stale hold/zero-speed/heading override from a *previous*
        # WAIT or REORIENT never silently survives into a REPLAN_ROUTE/RESUME_ROUTE/CONTINUE
        # decision (which all expect the normal distance-to-goal speed_ref and an actual
        # reference path, not a held point or a turn-in-place target).
        ctx.traj_gen.speed_ref_override = None
        ctx.traj_gen.holding = False
        ctx.traj_gen.heading_override = None
        strategy = self.strategies.get(method)
        if strategy is None:
            return RecoveryOutcome(method, success=False, detail=f"No strategy registered for {method}.")
        return strategy.execute(ctx)
