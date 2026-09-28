"""One closed-loop episode: the MPC drives the robot through a scenario-grid layout, and on a fixed
clock a decision-maker (the tree, the teacher, the scripted rules, or a recording) is asked whether
the robot has failed and what to do; its answer is executed against the live controller.

    every control cycle:  world.step -> MPC solve -> SituationBuilder.update
    every assessment:     report -> AsyncFailureAnalyzer -> (some cycles later) recovery strategy

Imports the MPC stack and the simulator, so it needs the compiled solver (scripts/build_solver.py).
"""
from __future__ import annotations

import functools
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .async_llm import AsyncFailureAnalyzer
from .llm import SituationAssessment, human_grounds
from .monitor import VerdictLog
from .recorder import EpisodeRecorder
from .recovery import (TURN_ALIGNED_RAD, RecoveryContext, RecoveryManager, RecoveryMethod,
                       recovery_candidates)
from .report import MOTION_WINDOW_S, WATCHDOG_MARGIN_STEPS, WATCHDOG_STEPS, WATCHDOG_WINDOW_STEPS, SituationBuilder
from .tree import ABANDONED_OBSTACLE_S, signature
from .types import RobotSnapshot

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MPC_CONFIG = os.path.join(REPO, "config", "mpc_default.yaml")

BREAKER_THRESHOLD = 3
"""`recovery_cycles_without_progress` at which the recovery-loop breaker applies: the prompt states
a hard rule (REPLAN_ROUTE, or REQUEST_HUMAN where its conditions are met) and the loop overrides an
answer that ignores it."""
WAIT_RELEASE_STEPS = 3
"""Control cycles in a row with no dynamic obstacle on the route before a WAIT ordered against one
is released early. One cycle would release on an obstacle that merely brushed the margin."""
HOLD_RECHECK_STEPS = 5
"""A WAIT whose obstacle is still in the way when its hold runs out is kept this many more cycles
(1 s) and checked again."""
HOLD_KEEP_GAP_M = 3.0
"""... "in the way" being on the route and within this footprint-edge gap of the robot."""
DYN_OBS_SIZE = 0.8 + 0.8


def seed_everything(seed: int) -> None:
    """Seed the global random generators; the scenario-grid layout is drawn from `random`."""
    random.seed(seed)
    np.random.seed(seed)


def _predict(last_pos: list, current_pos: list, steps: int = 20) -> list:
    """Constant-velocity prediction of a dynamic obstacle, as the MPC's dynamic constraints take it."""
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    return [[current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1]
            for i in range(steps)]


@dataclass
class RecoveryRecord:
    """One recovery executed during an episode."""
    step: int
    method: RecoveryMethod
    success: bool
    detail: str
    rationale: str = ""
    diagnosis: str = ""

    def __str__(self) -> str:
        status = 'ok' if self.success else 'FAILED'
        chose = f"{self.diagnosis} -> {self.method.value}" if self.diagnosis else self.method.value
        return f"[step {self.step}] {chose} ({status}): {self.detail}"


@dataclass
class EpisodeTiming:
    """What one episode cost in wall time, and how late verdicts were acted on."""
    wall_s: float
    llm_s: float
    blocking_s: float
    calls: int
    skipped: int
    mean_latency_s: float
    mean_age_steps: float
    sim_s: float = 0.0

    def __str__(self) -> str:
        return (f"{self.wall_s:.1f} s wall for {self.sim_s:.1f} s simulated; {self.calls} decisions costing "
                f"{self.llm_s:.1f} s ({self.mean_latency_s:.2f} s each), {self.blocking_s:.1f} s of it waited "
                f"for; mean decision age {self.mean_age_steps:.1f} steps; {self.skipped} tick(s) skipped")


@dataclass
class EpisodeResult:
    """Outcome of one `run_episode()`.

    `termination` is why the episode ended: "goal", "collision", "human_requested" (REQUEST_HUMAN
    was executed), "watchdog" (too many missed deadlines), "mpc_error" (the solver raised) or
    "timeout" (the step budget ran out). `success` means "reached the goal" on a layout with a way
    through and "called a human" on one without (`needs_human`).
    """
    verdicts: VerdictLog
    success: bool
    steps: int
    recoveries: list[RecoveryRecord] = field(default_factory=list)
    timing: EpisodeTiming | None = None
    termination: str = "timeout"
    needs_human: bool = False


def run_episode(analyzer, scenario_option: int = 1, max_steps: int = 500, verbose: bool = False,
                assess_hz: float = 0.5, recorder: EpisodeRecorder | None = None,
                watchdog_steps: int = WATCHDOG_STEPS, watchdog_margin: int = WATCHDOG_MARGIN_STEPS) -> EpisodeResult:
    """Run one episode on a scenario-grid layout drawn from the global `random` state (seed it
    first with `seed_everything`).

    Args:
        analyzer: what answers the assessment ticks -- anything with `assess_situation(context,
            candidates, evidence=, stall_threshold=)`: a `TreeAnalyzer`, a `ScriptedTeacher`, a
            `ReplayAnalyzer`. Called on a worker thread and acted on at the control cycle a real
            robot would have received the answer (`async_llm`).
        scenario_option: the scenario grid's option (1: the mixed stream; see `simulation.OPTIONS`).
        max_steps: step budget (0.2 s each).
        assess_hz: assessments per second of simulated time.
        recorder: receives a compact per-step trace for replay and for the counterfactual labels.
        watchdog_steps: missed deadlines within `WATCHDOG_WINDOW_STEPS` that end the episode.
        watchdog_margin: see `report.WATCHDOG_MARGIN_STEPS`.

    A few rules of the loop, all in service of the recoveries doing what their descriptions say:

    - CONTINUE is never executed (it would cancel a running hold), and a hold lapses when its
      cooldown runs out; a turn in place is released as soon as the robot is aligned.
    - A WAIT ordered against a dynamic obstacle on the route is released early once the obstacle
      has been off the route for `WAIT_RELEASE_STEPS` cycles, does not count towards
      `recovery_cycles_without_progress` while the obstacle has stood less than
      `tree.ABANDONED_OBSTACLE_S` (repeated holds at a gate are not a recovery loop), and is kept
      past its cooldown while that obstacle is still on the route within `HOLD_KEEP_GAP_M`
      (re-checked every `HOLD_RECHECK_STEPS`) -- until it has stood 30 s, when the decision-maker
      must see a robot that is not holding in order to call a human.
    - The recovery-loop breaker overrides an answer that ignores the hard rule in the prompt, but
      not while a hold is running.
    - A REPLAN_ROUTE that finds no route does not end the episode: the report says
      `goal_reachable: false` and calling a human is left to the decision-maker.
    - The controller watchdog reads measured wall time, so where it trips depends on the machine.
    """
    from mpc_traj_tracker import MPCConfig, TrajectoryGenerator
    from simulation import DYNAMIC_OBSTACLE_RADIUS, World, draw

    layout = draw(scenario_option)
    world = World(layout)
    needs_human = bool(layout.needs_human)

    config = MPCConfig.from_yaml(MPC_CONFIG)
    traj_gen = TrajectoryGenerator(config, motion_model=None)
    geo_map = world.geometric_map()
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list)
    robot_radius = world.robot.cfg.RADIUS

    verdicts = VerdictLog()
    context_builder = SituationBuilder(
        ts=config.ts, max_steps=max_steps, robot_radius=robot_radius,
        dynamic_obstacle_radius=DYNAMIC_OBSTACLE_RADIUS, watchdog_steps=watchdog_steps,
        watchdog_margin=watchdog_margin)
    candidates = recovery_candidates(watchdog_margin)
    evidence = context_builder.evidence
    assess_period = max(1, round(1.0 / (assess_hz * config.ts)))
    # No assessment until the report's rolling window is full: a verdict drawn from half a window.
    assess_warmup = round(MOTION_WINDOW_S / config.ts)
    recovery_manager = RecoveryManager()
    recovered_speed_mode = False    # set once a REPLAN_ROUTE/RESUME_ROUTE succeeds: back to 'work' speed
    cooldown_until: int | None = None
    recent_dyn_gaps: deque = deque(maxlen=max(2, round(1.0 / config.ts) + 1))  # nearest dynamic gap, last 1 s
    dyn_obstacle_list: list = []
    recoveries: list[RecoveryRecord] = []
    escalated = False

    init_state = np.array([*world.robot.position, world.robot.angle])
    goal_state = np.array([*world.goal.position, 0])
    goal_xy = tuple(world.goal.position)
    ref_path = list(world.path.coords)
    traj_gen.load_init_state(init_state, goal_state)
    traj_gen.set_work_mode(mode='work')
    traj_gen.set_ref_trajectory(ref_path)
    # The intended route, kept apart from the tracked reference (which a WAIT collapses to a point).
    route_ahead: list[tuple[float, float]] = [(float(x), float(y)) for x, y in ref_path]
    last_ordered: str | None = None
    if recorder is not None:
        recorder.scene(boundary=geo_map.boundary_coords, obstacles=geo_map.obstacle_list,
                       inflated_obstacles=geo_map.processed_obstacle_list, goal=goal_xy,
                       start=init_state[:2].tolist(), initial_route=route_ahead,
                       robot_radius=robot_radius, dynamic_obstacle_radius=DYNAMIC_OBSTACLE_RADIUS,
                       ts=config.ts, max_steps=max_steps, assess_hz=assess_hz,
                       scenario_option=scenario_option, watchdog_steps=watchdog_steps,
                       watchdog_margin=watchdog_margin, layout=layout.describe())

    def note(step: int, text: str) -> None:
        if recorder is not None:
            recorder.note(step, text)

    def hold_label(step: int) -> str | None:
        """What the running recovery is doing to the robot right now, for the recorder."""
        if last_ordered is None:
            return None
        if traj_gen.heading_override is not None:
            return f"{last_ordered}: turning to face the route"
        if traj_gen.holding and cooldown_until is not None:
            return f"{last_ordered}: holding ({max(0, cooldown_until - step)} steps left)"
        return None

    last_dyn_obstacle_list = None
    chosen_ref_traj = None
    done = False
    step_reached = 0
    termination = "timeout"
    wait_release_armed = False  # the running hold is a WAIT for a dynamic obstacle on the route
    hold_check_armed = False    # ... and gets checked before it is let go
    hold_blocked, hold_stationary_s, hold_kept_steps = False, None, 0
    off_route_steps = 0

    llm = AsyncFailureAnalyzer(analyzer, ts=config.ts)

    def apply_reply(reply, step: int) -> bool:
        """Act on one verdict: record the diagnosis, run the recovery-loop breaker over it, and
        execute the chosen recovery. Returns True if the episode should stop here. The diagnosis is
        dated to the report it was drawn from, the recovery happens now."""
        nonlocal route_ahead, cooldown_until, recovered_speed_mode, escalated
        nonlocal wait_release_armed, off_route_steps, hold_check_armed, hold_blocked, hold_kept_steps
        nonlocal last_ordered
        decision = reply.decision
        answered = reply.decision
        context = reply.request.context
        stall = reply.request.stall
        failure_mode = decision.failure_mode
        diagnosis = failure_mode.value.upper() if failure_mode is not None else "NOMINAL"
        flagged = verdicts.record(failure_mode, reply.request.step, decision.failure_message)
        if flagged is not None and verbose:
            print(f"  {flagged}")
        # Not into a hold still running: the count reaches the threshold the moment the third WAIT
        # starts, and an override three steps into it decides the question the hold was buying time
        # to ask (wait on, or call for help?). The tick after the hold ends is soon enough.
        held = bool(getattr(context, "holding_position", False))
        if (stall is not None and stall >= BREAKER_THRESHOLD and not held
                and decision.method not in (RecoveryMethod.REPLAN_ROUTE, RecoveryMethod.REQUEST_HUMAN)):
            demanded = RecoveryMethod.REQUEST_HUMAN if human_grounds(context) else RecoveryMethod.REPLAN_ROUTE
            if verbose:
                print(f"  --- Recovery loop breaker: {decision.method.value} despite the hard rule "
                      f"(recovery_cycles_without_progress={stall}); overriding to {demanded.value} ---")
            decision = SituationAssessment(
                method=demanded,
                rationale=f"Overridden by the recovery-loop breaker: recovery_cycles_without_progress "
                          f"reached {stall} and the hard rule demanded {demanded.value}, but "
                          f"{decision.method.value} was chosen (said: {decision.rationale!r}).",
                raw_response=decision.raw_response,
            )
        if verbose:
            when = (f"report from step {reply.request.step}, acted on at step {step}"
                    if reply.age_steps else f"step {step}")
            print(f"  --- decision ({when}, {reply.latency_s:.2f} s): {diagnosis} -> {decision.method.value} ---")
            if decision.rationale:
                print("  " + decision.rationale.replace("\n", "\n  "))
        if recorder is not None:
            recorder.decision(report_step=reply.request.step, acted_step=step,
                              raw_response=answered.raw_response,
                              failure_mode=diagnosis if diagnosis != "NOMINAL" else None,
                              method=decision.method.value, rationale=decision.rationale,
                              latency_s=reply.latency_s, overridden=decision is not answered,
                              expired=None, signature=list(signature(context)))
        # CONTINUE is a no-op, and executing it would clear a hold that is still meant to run.
        if decision.method is RecoveryMethod.CONTINUE:
            return False
        outcome = recovery_manager.execute(
            decision.method,
            RecoveryContext(traj_gen=traj_gen, goal=goal_xy, geo_map=geo_map, route_ahead=route_ahead))
        hold_info = {'step': step, 'cooldown_steps': outcome.cooldown_steps}
        waiting_for_dynamic = (outcome.method is RecoveryMethod.WAIT
                               and bool(getattr(context, 'dynamic_obstacle_blocking_path', False))
                               and not getattr(context, 'static_obstacle_blocking_path', False))
        still_pausing = (getattr(context, 'blocking_obstacle_stationary_s', None) or 0.0) < ABANDONED_OBSTACLE_S
        wait_release_armed, off_route_steps = waiting_for_dynamic, 0
        hold_check_armed = waiting_for_dynamic
        hold_blocked, hold_kept_steps = False, 0
        if waiting_for_dynamic and still_pausing:
            hold_info['counts'] = False
        context_builder.record_recovery_outcome(outcome.method.value, outcome.success, **hold_info)
        recoveries.append(RecoveryRecord(step=step, method=outcome.method, success=outcome.success,
                                         detail=outcome.detail, rationale=decision.rationale,
                                         diagnosis=diagnosis))
        if outcome.new_route is not None:
            route_ahead = outcome.new_route
        last_ordered = outcome.method.value
        if recorder is not None:
            recorder.recovery(step=step, method=outcome.method.value, success=outcome.success,
                              detail=outcome.detail)
        cooldown_until = step + outcome.cooldown_steps if outcome.cooldown_steps else None
        if outcome.success and decision.method in (RecoveryMethod.REPLAN_ROUTE, RecoveryMethod.RESUME_ROUTE):
            recovered_speed_mode = True
        if verbose:
            print(f"  --- recovery ({'ok' if outcome.success else 'FAILED'}): {outcome.detail}")
        if outcome.no_route:
            # Reported, not acted on: the next report says goal_reachable: false.
            context_builder.record_no_route()
            note(step, "no route to the goal: reported as goal_reachable=false")
            return False
        if outcome.method is RecoveryMethod.REQUEST_HUMAN:
            escalated = True
            verdict = "needed" if needs_human else "NOT needed: a false alarm"
            note(step, f"HUMAN REQUESTED ({verdict})")
            if verbose:
                print(f"  --- HUMAN REQUESTED: robot stopped in place. On this layout a human was {verdict} ---")
            return True
        return False

    episode_started = time.perf_counter()
    for i in range(max_steps):
        step_reached = i

        if traj_gen.heading_override is not None and abs(traj_gen.heading_override_error()) < TURN_ALIGNED_RAD:
            # A turn in place is done as soon as the robot faces the right way.
            traj_gen.holding = False
            traj_gen.speed_ref_override = None
            traj_gen.heading_override = None
            if cooldown_until is not None:
                cooldown_until = i
            context_builder.end_hold(i)
            note(i, "turn complete, hold released")
            if verbose:
                print(f"  --- step {i}: turn complete, releasing the hold ---")

        if (hold_check_armed and cooldown_until is not None and i >= cooldown_until and traj_gen.holding
                and hold_blocked and (hold_stationary_s is None or hold_stationary_s < ABANDONED_OBSTACLE_S)):
            # What this WAIT was ordered for is still in the way: letting go now would only hand
            # the MPC a route through it. Keep the hold and look again shortly.
            if hold_kept_steps == 0:
                note(i, "WAIT kept past its hold: the obstacle is still on the route")
                if verbose:
                    print(f"  --- step {i}: the obstacle is still on the route, keeping the WAIT ---")
            hold_kept_steps += HOLD_RECHECK_STEPS
            cooldown_until = i + HOLD_RECHECK_STEPS
            context_builder.hold_for(i, HOLD_RECHECK_STEPS)
        if cooldown_until is not None and i >= cooldown_until:
            if hold_check_armed and hold_kept_steps:
                note(i, f"WAIT let go after {hold_kept_steps} extra steps: "
                        + ("the obstacle has stood too long to be waited for" if hold_blocked
                           else "the obstacle is no longer in the way"))
            hold_check_armed = False
            # A hold lapses when its cooldown runs out.
            traj_gen.holding = False
            traj_gen.speed_ref_override = None
            traj_gen.heading_override = None
            cooldown_until = None

        dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in world.dynamic_obstacles]
        if dyn_obstacle_list:
            recent_dyn_gaps.append(min(math.hypot(traj_gen.state[0]-p[0], traj_gen.state[1]-p[1])
                                       for p in dyn_obstacle_list)
                                   - robot_radius - DYNAMIC_OBSTACLE_RADIUS)
        else:
            recent_dyn_gaps.clear()
        if last_dyn_obstacle_list is None:
            last_dyn_obstacle_list = dyn_obstacle_list
        dyn_obstacle_pred_list = [_predict(last_dyn_obstacle_list[j], dyn_obs)
                                  for j, dyn_obs in enumerate(dyn_obstacle_list)]
        last_dyn_obstacle_list = dyn_obstacle_list

        world.set_robot_state(traj_gen.state[:2], traj_gen.state[2], traj_gen.last_action[0], traj_gen.last_action[1])
        done = world.step()
        if dyn_obstacle_list:
            traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
        chosen_ref_traj, *_ = traj_gen.get_local_ref_traj()

        # 'safe' (low reference speed) until a replan or resume has succeeded.
        mpc_mode = 'work' if recovered_speed_mode else 'safe'
        try:
            mpc_output = traj_gen.get_action(chosen_ref_traj, mode=mpc_mode)
        except Exception as e:
            if verbose:
                print(f'MPC fails: {e}')
            termination = "mpc_error"
            break
        if mpc_output is None:   # goal reached (TrajectoryGenerator.check_termination_condition)
            termination = "goal"
            break
        action, pred_states, cost = mpc_output

        snapshot = RobotSnapshot(
            step=i,
            position=(float(traj_gen.state[0]), float(traj_gen.state[1])),
            heading=float(traj_gen.state[2]),
            goal=goal_xy,
            action=(float(action[0]), float(action[1])),
            solver_time_ms=traj_gen.solver_time_timelist[-1] if traj_gen.solver_time_timelist else None,
            obstacles=geo_map.obstacle_list,
            speed=float(action[0]),
            reference_path=[(float(p[0]), float(p[1])) for p in chosen_ref_traj],
            route_ahead=route_ahead,
            dynamic_obstacles=[(float(p[0]), float(p[1])) for p in dyn_obstacle_list],
            boundary=geo_map.boundary_coords,
        )
        context_builder.update(snapshot)
        if hold_check_armed and traj_gen.holding:
            # Read every cycle, so the check at the hold's end has this cycle's answer.
            hold_blocked = (bool(context_builder.dynamic_obstacle_on_route(snapshot))
                            and bool(recent_dyn_gaps) and recent_dyn_gaps[-1] <= HOLD_KEEP_GAP_M)
            hold_stationary_s = context_builder.blocking_obstacle_stationary_s(snapshot)
        if wait_release_armed and traj_gen.holding and cooldown_until is not None and i < cooldown_until:
            # The obstacle this WAIT was for has left the route: the wait is over.
            on_route = context_builder.dynamic_obstacle_on_route(snapshot)
            off_route_steps = 0 if on_route else off_route_steps + 1
            if off_route_steps >= WAIT_RELEASE_STEPS:
                wait_release_armed = False
                note(i, f"WAIT released early: the obstacle has left the route ({cooldown_until - i} steps of the hold unused)")
                if verbose:
                    print(f"  --- step {i}: the obstacle has left the route, releasing the WAIT "
                          f"({cooldown_until - i} steps early) ---")
                cooldown_until = i
                context_builder.end_hold(i)
        if recorder is not None:
            recorder.frame(step=i, position=snapshot.position, heading=snapshot.heading,
                           speed=float(action[0]), angular_speed=float(action[1]),
                           dynamic_obstacles=snapshot.dynamic_obstacles,
                           reference=snapshot.reference_path, route_ahead=route_ahead,
                           hold=hold_label(i))
        if context_builder.watchdog_tripped:
            # Checked before this step's verdict is applied: a WAIT landing now is one step too late.
            termination = "watchdog"
            note(i, "WATCHDOG: too many missed deadlines, controller stopped")
            if verbose:
                print(f"  --- WATCHDOG: {watchdog_steps} of the last {WATCHDOG_WINDOW_STEPS} solves ran over "
                      f"the {config.ts * 1000:.0f} ms control period -- controller stopped ---")
            break

        reply = llm.poll(i)
        if reply is not None and apply_reply(reply, i):
            termination = "human_requested"
            break

        # Purely a clock: asking only while something already looks wrong would smuggle a detector
        # back in through the trigger, and asking through a hold is what lets a WAIT be ended early.
        assess_due = i >= assess_warmup and (i - assess_warmup) % assess_period == 0
        if assess_due and llm.busy:
            # One call in flight at a time: skip the tick and ask again at the next.
            if llm.skip():
                note(i, "assessment skipped: previous call still out")
                if verbose:
                    print(f"  (assessment due at step {i}, but the previous one is still out -- skipped)")
            assess_due = False
        if assess_due:
            context = context_builder.build(snapshot, verdicts.prior)
            if verbose:
                print(f"  --- status report (step {i}) ---")
                print("  " + context.to_json().replace("\n", "\n  "))
            call = functools.partial(analyzer.assess_situation, context, candidates,
                                     evidence=evidence, stall_threshold=BREAKER_THRESHOLD)
            if recorder is not None:
                recorder.ask(i)
            llm.submit(call, step=i, context=context, stall=context.recovery_cycles_without_progress)

        if done:
            termination = "collision" if world.collided else "goal"
            break

    wall_s = time.perf_counter() - episode_started
    timing = EpisodeTiming(wall_s=wall_s, llm_s=llm.total_llm_s, blocking_s=llm.blocking_s,
                           calls=llm.calls, skipped=llm.dropped, mean_latency_s=llm.mean_latency_s,
                           mean_age_steps=llm.mean_age_steps, sim_s=(step_reached + 1) * config.ts)
    llm.close()

    success = escalated if needs_human else (world.reached_goal and not escalated)
    if recorder is not None:
        recorder.finish(termination=termination, success=success, steps=step_reached + 1,
                        needs_human=needs_human, llm_calls=timing.calls,
                        mean_age_steps=timing.mean_age_steps, mean_latency_s=timing.mean_latency_s,
                        wall_s=wall_s)
    if verbose:
        if termination == "human_requested":
            outcome_str = f"called a human supervisor ({'needed' if needs_human else 'a false alarm'})"
        else:
            outcome_str = 'reached the goal' if success else f'did not reach the goal ({termination})'
        modes = [m.value for m in verdicts.failure_modes()]
        print(f"  -> finished after {step_reached+1} steps, {outcome_str}. "
              f"Failure modes named: {modes if modes else 'none'}")
        print(f"  -> {timing}")

    return EpisodeResult(verdicts=verdicts, success=success, steps=step_reached + 1, recoveries=recoveries,
                         timing=timing, termination=termination, needs_human=needs_human)
