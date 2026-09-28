"""Logic checks on the decision tree, its judge, the scripted rules and the experience memory,
against hand-built reports. No MPC, no map, no model; about a second.

    python scripts/self_test.py
"""
import json
import os
import sys
import tempfile

from failure_monitor import (FailureMode, RECOVERY_METHOD_DESCRIPTIONS, RecoveryMethod, ReferenceShadow,
                             ScriptedTeacher, SituationAssessment, SituationContext, TreeAnalyzer, TreePolicy,
                             evidence_of_failure, monitor_evidence, recovery_candidates, signature)
from failure_monitor.tree import DecisionTree, FEATURE_NAMES


def report(**fields) -> SituationContext:
    """A `SituationContext` with only the fields a check cares about set.

    The real class rather than a stand-in on purpose: `tree.signature` reads reports by field
    name, so building these from the actual dataclass is what stops a renamed or mistyped field
    from turning into a feature that silently reads `?` forever.
    """
    # route_progress_m defaults to a standstill: a check that wants a recovery credited says so.
    base = dict(step=0, goal_distance=5.0, holding_position=False, route_progress_m=0.0)
    return SituationContext(**{**base, **fields})


def _check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def self_test(tmp_dir: str) -> bool:
    """Checks on the learning core, cheapest and most isolated first. Returns True if all pass."""
    ok = True
    print("\n=== 1. binning: a report becomes a signature ===")
    driving = report(stopped_for_s=0.0, goal_progress_last_5s=1.2, route_heading_error_deg=5.0)
    stuck = report(stopped_for_s=9.0, static_obstacle_blocking_path=True,
                   goal_progress_last_5s=0.0, static_obstacle_gap=0.4)
    sig_driving, sig_stuck = signature(driving), signature(stuck)
    named = dict(zip(FEATURE_NAMES, sig_stuck))
    ok &= _check("a driving robot bins as not stopped", sig_driving[1] == "no", sig_driving[1])
    ok &= _check("a pinned robot bins as stopped=long", named["stopped"] == "long", named["stopped"])
    ok &= _check("... blocked by something static", named["blocked"] == "static", named["blocked"])
    ok &= _check("... and not in contact with it", named["contact"] == "clear", named["contact"])
    ok &= _check("an absent field bins as unknown, not as zero",
                 dict(zip(FEATURE_NAMES, signature(report())))["approach"] == "?", signature(report()))

    print("\n=== 2. growth: insert, reinforce, refine ===")
    tree = DecisionTree(init_depth=2)
    leaf, what = tree.learn(sig_stuck, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("a first answer inserts a shallow rule", what == "insert" and leaf.depth == 2,
                 f"{what}, depth {leaf.depth}")
    _, what = tree.learn(sig_stuck, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("the same answer again reinforces rather than duplicating",
                 what == "reinforce" and len(tree) == 1, f"{what}, {len(tree)} rules")

    # Same first two features (not holding, stopped long), different blocker: a moving obstacle
    # should be waited out, not replanned around. The tree has no way to know that until told.
    sig_wait = signature(report(stopped_for_s=9.0, dynamic_obstacle_blocking_path=True,
                                blocking_obstacle_stationary_s=0.0, goal_progress_last_5s=0.0))
    leaf, what = tree.learn(sig_wait, RecoveryMethod.WAIT, FailureMode.STUCK)
    ok &= _check("a contradiction deepens the tree instead of overwriting",
                 what == "refine" and len(tree) == 2, f"{what}, {len(tree)} rules")
    ok &= _check("... splitting on the feature that actually separates them",
                 leaf.path[:2] == sig_stuck[:2] and leaf.path[2] == "dynamic",
                 f"split at {FEATURE_NAMES[len(leaf.path) - 1]}")
    ok &= _check("both rules stay reachable",
                 tree.match(sig_stuck).action is RecoveryMethod.REPLAN_ROUTE
                 and tree.match(sig_wait).action is RecoveryMethod.WAIT)
    ok &= _check("a third value of the split feature is now uncovered (escalates)",
                 tree.match(signature(report(stopped_for_s=9.0, goal_progress_last_5s=0.0))) is None)

    print("\n=== 3. contradiction the features cannot explain ===")
    flip = DecisionTree(init_depth=2)
    flip.learn(sig_stuck, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    leaf, what = flip.learn(sig_stuck, RecoveryMethod.WAIT, FailureMode.STUCK)
    ok &= _check("identical signatures with different answers are recorded, not split",
                 what == "contradict" and leaf.conflicts == 1 and len(flip) == 1,
                 f"{what}, {leaf.conflicts} conflict(s)")

    print("\n=== 4. outcomes: a rule that does not help is retired ===")
    policy = TreePolicy(init_depth=2, horizon_steps=5, retire_after=2, source="scripted")
    # The teacher's own WAIT is acted on too, so it is judged exactly like a tree hit: one bad
    # outcome from the teacher's occurrence, one from the rule firing afterwards.
    policy.learn(report(step=0, stopped_for_s=9.0, static_obstacle_blocking_path=True),
                 RecoveryMethod.WAIT, FailureMode.STUCK)
    policy.observe(report(step=6, stopped_for_s=15.0, static_obstacle_blocking_path=True,
                          goal_progress_last_5s=0.0))
    ok &= _check("the teacher's own answer is judged, not just later tree hits",
                 policy.stats.judged_bad == 1, f"{policy.stats.judged_bad} bad")
    for episode_step in (10,):
        # Fire the rule, then hand it a report from a few cycles later in which the robot is still
        # pinned: WAIT did not help, and nothing is holding it to excuse the standstill.
        fired = report(step=episode_step, stopped_for_s=9.0, static_obstacle_blocking_path=True)
        policy.observe(fired)
        ok &= _check(f"the rule fires at step {episode_step}", policy.decide(fired) is not None)
        after = report(step=episode_step + 6, stopped_for_s=15.0, static_obstacle_blocking_path=True,
                       goal_progress_last_5s=0.0)
        policy.observe(after)
    retired = [leaf for leaf in policy.tree.leaves.values() if leaf.retired]
    ok &= _check("two bad outcomes retire it", len(retired) == 1 and policy.stats.judged_bad == 2,
                 f"{policy.stats.judged_bad} bad")
    ok &= _check("a retired rule stops answering (the tick goes back to the teacher)",
                 policy.decide(report(step=99, stopped_for_s=9.0,
                                      static_obstacle_blocking_path=True)) is None)
    replacement = report(step=100, stopped_for_s=9.0, static_obstacle_blocking_path=True)
    leaf, what = policy.learn(replacement, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("the teacher's answer for the same situation replaces it in place",
                 what == "replace" and not leaf.retired
                 and leaf.action is RecoveryMethod.REPLAN_ROUTE, what)

    # A retired rule re-taught on a *different* report should split rather than hand one
    # over-general path back and forth between two answers.
    thrash = TreePolicy(init_depth=2, horizon_steps=5, retire_after=1)
    thrash.learn(report(step=0, reversing_for_s=0.0), RecoveryMethod.CONTINUE, None)
    reversing = report(step=10, reversing_for_s=5.0, goal_progress_last_5s=0.0)
    thrash.observe(reversing)
    thrash.decide(reversing)   # answers NOMINAL for a report meeting the REVERSE_TRACKING test
    leaf, what = thrash.learn(reversing, RecoveryMethod.REORIENT, FailureMode.REVERSE_TRACKING)
    ok &= _check("a retired rule re-taught on a different report splits instead of flip-flopping",
                 what == "refine" and len(thrash.tree) == 2, f"{what}, {len(thrash.tree)} rules")
    ok &= _check("... leaving the failed rule retired on its own branch",
                 any(l.retired for l in thrash.tree.leaves.values())
                 and thrash.tree.match(signature(reversing)).action is RecoveryMethod.REORIENT)

    print("\n=== 5. a good outcome is credited ===")
    rewarded = TreePolicy(init_depth=2, horizon_steps=5)
    rewarded.learn(report(step=0, stopped_for_s=9.0, static_obstacle_blocking_path=True),
                   RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    fired = report(step=10, stopped_for_s=9.0, static_obstacle_blocking_path=True)
    rewarded.observe(fired)
    rewarded.decide(fired)
    rewarded.observe(report(step=16, stopped_for_s=0.0, goal_progress_last_5s=1.4, route_progress_m=1.4))
    ok &= _check("driving again afterwards credits the rule", rewarded.stats.judged_good == 1,
                 f"{rewarded.stats.judged_good} good")

    print("\n=== 5b. a nominal rule is scored on the report it was shown, not on the future ===")
    # The rule that matters most in any grown tree is "the robot is driving, leave it alone". It
    # must survive a robot that gets into trouble later, and must not survive waving through a
    # report that already meets a failure test.
    nominal = TreePolicy(init_depth=2, horizon_steps=5, retire_after=2)
    nominal.learn(report(step=0, stopped_for_s=0.0), RecoveryMethod.CONTINUE, None)
    driving = report(step=10, stopped_for_s=0.0, goal_progress_last_5s=1.1)
    nominal.observe(driving)
    nominal.decide(driving)
    nominal.observe(report(step=40, stopped_for_s=9.0, goal_progress_last_5s=0.0))
    ok &= _check("a correct nominal call is not punished for a later failure",
                 nominal.stats.judged_bad == 0, f"{nominal.stats.judged_bad} bad")
    ok &= _check("evidence_of_failure reads the decisive tests off a report",
                 evidence_of_failure(report(stopped_for_s=9.0)) is FailureMode.STUCK
                 and evidence_of_failure(report(static_obstacle_gap=-0.1)) is FailureMode.COLLISION
                 and evidence_of_failure(report(reversing_for_s=4.0)) is FailureMode.REVERSE_TRACKING
                 and evidence_of_failure(driving) is None)
    # Same kind of rule, but over-general: conditioned on `hold` alone (init_depth=1), so it also
    # answers NOMINAL for a report that plainly meets the STUCK test. That is the mistake the
    # immediate check exists for, and an over-general rule is exactly how the tree produces one.
    waved = TreePolicy(init_depth=1, horizon_steps=5, retire_after=2)
    waved.learn(report(step=0, stopped_for_s=0.0), RecoveryMethod.CONTINUE, None)
    for step in (10, 20):
        pinned = report(step=step, stopped_for_s=9.0, goal_progress_last_5s=0.0)
        waved.observe(pinned)
        ok &= _check(f"the over-general rule answers the pinned report at step {step}",
                     waved.decide(pinned) is not None)
    ok &= _check("waving through a failure the report already showed is marked wrong",
                 waved.stats.judged_bad == 2, f"{waved.stats.judged_bad} bad")
    ok &= _check("... and retires the rule", any(l.retired for l in waved.tree.leaves.values()))

    print("\n=== 5e. scoring edge cases ===")
    # A robot turning on the spot during a REORIENT hold can read as seconds of negative speed.
    # That is the recovery's doing, and the prompt says a running recovery is not a failure.
    held_reversing = report(step=0, holding_position=True, reversing_for_s=5.0)
    ok &= _check("reversing during a recovery hold is not evidence of failure",
                 evidence_of_failure(held_reversing) is None)
    ok &= _check("... but contact during a hold still is",
                 evidence_of_failure(report(step=0, holding_position=True, dynamic_obstacle_gap=-0.01))
                 is FailureMode.COLLISION)
    held = TreePolicy(init_depth=2, retire_after=1)
    held.learn(held_reversing, RecoveryMethod.CONTINUE, None)
    ok &= _check("... so a nominal answer during a hold is not punished for it",
                 held.stats.judged_bad == 0, f"{held.stats.judged_bad} bad")

    # A rule that names a failure but does nothing about it used to be exempt from scoring entirely.
    named = TreePolicy(init_depth=2, retire_after=1)
    named.learn(report(step=0, reversing_for_s=5.0), RecoveryMethod.CONTINUE, FailureMode.STUCK)
    ok &= _check("a mode+CONTINUE rule doing nothing about evident failure is marked wrong",
                 named.stats.judged_bad == 1, f"{named.stats.judged_bad} bad")

    # A judgement whose rule is split while it is pending goes to whichever rule now governs its
    # situation -- if that rule still orders the same action -- and is counted otherwise.
    split = TreePolicy(init_depth=2, horizon_steps=5, retire_after=5)
    pinned = report(step=0, stopped_for_s=9.0, static_obstacle_blocking_path=True)
    split.learn(pinned, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    # Split with a nominal answer: a second *recovery* would supersede (and so settle) the pending one.
    split.learn(report(step=1, stopped_for_s=0.0, dynamic_obstacle_blocking_path=True,
                       static_obstacle_blocking_path=False),
                RecoveryMethod.CONTINUE, None)   # splits the REPLAN rule
    split.observe(report(step=6, stopped_for_s=0.0, goal_progress_last_5s=1.0, route_progress_m=1.0))
    governing = split.tree.match(signature(pinned))
    ok &= _check("a judgement survives its rule being split, credited to the rule now governing it",
                 governing.good == 1 and split.stats.orphaned == 0,
                 f"{governing.good} good, {split.stats.orphaned} orphaned")
    orphan = TreePolicy(init_depth=2, horizon_steps=5, retire_after=5)
    orphan.learn(pinned, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    orphan.tree.leaves[orphan.tree.match(signature(pinned)).path].action = RecoveryMethod.WAIT
    orphan.observe(report(step=6, stopped_for_s=0.0, goal_progress_last_5s=1.0, route_progress_m=1.0))
    ok &= _check("... and counted as orphaned if that rule now orders something else",
                 orphan.stats.orphaned == 1, f"{orphan.stats.orphaned} orphaned")

    print("\n=== 5h. a recovery is judged on getting the robot going, not on what happens next ===")
    wall = dict(stopped_for_s=0.0, static_obstacle_blocking_path=True, static_obstacle_gap=0.4)
    # A REPLAN_ROUTE at a wall, and 30 steps later the robot is driving its new route with a
    # pedestrian 0.45 m away. That is not the replan's doing.
    passerby = TreePolicy(init_depth=3, horizon_steps=25, retire_after=1)
    passerby.learn(report(step=25, route_progress_m=10.0, **wall), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    crossing = report(step=55, route_progress_m=12.5, robot_speed=0.54, dynamic_obstacle_gap=0.45)
    passerby.observe(crossing)
    ok &= _check("a pedestrian arriving after the robot got going is not blamed on the recovery",
                 passerby.stats.judged_good == 1 and passerby.stats.judged_bad == 0,
                 f"{passerby.stats.judged_good} good / {passerby.stats.judged_bad} bad")
    # But a recovery replaced before the robot got anywhere is bad, and does not wait for the
    # replacement's hold to end: a chain of turns must not share the progress made after the last.
    chain = TreePolicy(init_depth=3, horizon_steps=25, retire_after=5)
    near = dict(reversing_for_s=5.0)
    chain.learn(report(step=25, route_progress_m=5.0, **near), RecoveryMethod.REORIENT, FailureMode.REVERSE_TRACKING)
    chain.observe(report(step=35, route_progress_m=4.0, holding_position=True))
    chain.learn(report(step=45, route_progress_m=4.0, holding_position=True, **near),
                RecoveryMethod.REORIENT, FailureMode.REVERSE_TRACKING)
    ok &= _check("a recovery superseded while its hold still runs is bad", chain.stats.judged_bad == 1,
                 f"{chain.stats.judged_bad} bad")
    chain.observe(report(step=55, route_progress_m=3.0, holding_position=True))   # the second one's own hold
    chain.observe(report(step=75, route_progress_m=3.0))                          # released: its window opens
    chain.observe(report(step=100, route_progress_m=8.0))
    ok &= _check("... and only the last of the chain is credited with the progress after it",
                 (chain.stats.judged_good, chain.stats.judged_bad) == (1, 1),
                 f"{chain.stats.judged_good} good / {chain.stats.judged_bad} bad")

    # Not getting anywhere is still bad, and contact is bad whatever the odometer says.
    idle = TreePolicy(init_depth=3, horizon_steps=25, retire_after=5)
    idle.learn(report(step=0, route_progress_m=4.0, **wall), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    idle.observe(report(step=25, route_progress_m=4.6))
    ok &= _check("a recovery the robot gains under 1 m after is bad", idle.stats.judged_bad == 1,
                 f"{idle.stats.judged_bad} bad")
    idle.learn(report(step=30, route_progress_m=4.6, **wall), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    idle.observe(report(step=55, route_progress_m=8.0, static_obstacle_gap=-0.05))
    ok &= _check("... and contact is bad however far it got", idle.stats.judged_bad == 2,
                 f"{idle.stats.judged_bad} bad")

    # A WAIT is judged on what the robot does once released, however long the hold lasted.
    waited = TreePolicy(init_depth=3, horizon_steps=25, retire_after=5)
    pinned_by_crosser = dict(stopped_for_s=6.0, dynamic_obstacle_blocking_path=True, blocking_obstacle_stationary_s=0.0)
    waited.learn(report(step=0, route_progress_m=2.0, **pinned_by_crosser), RecoveryMethod.WAIT, FailureMode.STUCK)
    for step in (10, 20):
        waited.observe(report(step=step, route_progress_m=2.0, holding_position=True))
    waited.observe(report(step=30, route_progress_m=2.1))          # released: the window opens here
    waited.observe(report(step=50, route_progress_m=2.9))          # 20 steps in: not due
    ok &= _check("a hold moves the window to its end rather than eating into it",
                 waited.stats.judged_good + waited.stats.judged_bad == 0)
    waited.observe(report(step=55, route_progress_m=3.3))
    ok &= _check("... and the released robot's progress is what is scored", waited.stats.judged_good == 1,
                 f"{waited.stats.judged_good} good / {waited.stats.judged_bad} bad")
    stayed = TreePolicy(init_depth=3, horizon_steps=25, max_defer_steps=40)
    stayed.learn(report(step=0, **pinned_by_crosser), RecoveryMethod.WAIT, FailureMode.STUCK)
    for step in (10, 30, 50):
        stayed.observe(report(step=step, holding_position=True))
    ok &= _check("a robot still held past max_defer_steps goes uncredited",
                 stayed.stats.undecided == 1 and not stayed._pending, f"{stayed.stats.undecided} undecided")

    # The end of an episode settles what the window never reached.
    ends = {}
    for termination in ("goal", "collision", "timeout"):
        late = TreePolicy(init_depth=3, horizon_steps=25, retire_after=5)
        late.start_episode()
        late.learn(report(step=380, route_progress_m=30.0, **wall), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
        late.end_episode(termination)
        ends[termination] = (late.stats.judged_good, late.stats.judged_bad, late.stats.undecided)
    ok &= _check("an episode ending at the goal credits the last recovery; a collision debits it; "
                 "a timeout leaves it uncredited",
                 ends == {"goal": (1, 0, 0), "collision": (0, 1, 0), "timeout": (0, 0, 1)}, str(ends))

    print("\n=== 5f. a rule is confirmed before it is trusted (min_support) ===")
    # A teacher's first answer for a robot driving at a wall on its route is NOMINAL, and every
    # later answer for the same situation is STUCK/REPLAN.
    confirm = TreePolicy(init_depth=3, min_support=2)
    confirm.start_episode()
    far = report(step=25, stopped_for_s=0.0, static_obstacle_blocking_path=True, static_obstacle_gap=2.3)
    near = report(step=35, stopped_for_s=0.0, static_obstacle_blocking_path=True, static_obstacle_gap=2.3)
    confirm.learn(far, RecoveryMethod.CONTINUE, None)
    ok &= _check("a rule taught once does not answer yet: the next match is asked again",
                 confirm.decide(near) is None and confirm.stats.confirmations == 1,
                 f"{confirm.stats.confirmations} confirmation ask(s)")
    leaf, what = confirm.learn(near, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("... and a disagreeing answer replaces the unconfirmed one",
                 what == "contradict" and leaf.action is RecoveryMethod.REPLAN_ROUTE and leaf.support == 1,
                 f"{what}, support {leaf.support}")
    ok &= _check("... which itself still needs confirming", confirm.decide(near) is None)
    confirm.learn(report(step=45, stopped_for_s=0.0, static_obstacle_blocking_path=True, static_obstacle_gap=2.3),
                  RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("an agreeing answer from the same episode does not confirm it",
                 confirm.decide(report(step=55, stopped_for_s=0.0, static_obstacle_blocking_path=True,
                                       static_obstacle_gap=2.3)) is None)
    confirm.start_episode()
    later = report(step=25, stopped_for_s=0.0, static_obstacle_blocking_path=True, static_obstacle_gap=2.3)
    confirm.learn(later, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("... one from another episode does, and from then on the tree answers",
                 confirm.decide(later) is not None and confirm.tree.match(signature(later)).episodes == 2)
    ok &= _check("min_support=1 keeps the old behaviour: trusted as soon as taught",
                 (lambda p: (p.learn(far, RecoveryMethod.CONTINUE, None), p.decide(near))[1] is not None)(
                     TreePolicy(init_depth=3)))

    print("\n=== 5c. a new rule never shadows an existing refinement ===")
    # The bug this guards: a refinement leaves an internal node behind, and a later unmatched
    # report inserting at `init_depth` would land on it -- hiding every rule underneath, because
    # `match` takes the shallowest path that fits.
    shadow = TreePolicy(init_depth=2, horizon_steps=5, retire_after=1)
    shadow.learn(report(step=0, reversing_for_s=0.0), RecoveryMethod.CONTINUE, None)
    reversing = report(step=10, reversing_for_s=5.0, goal_progress_last_5s=0.0)
    shadow.observe(reversing)
    shadow.decide(reversing)
    shadow.learn(reversing, RecoveryMethod.REORIENT, FailureMode.REVERSE_TRACKING)  # refines
    deep = len(shadow.tree.match(signature(reversing)).path)
    # A different report that also starts `hold=no / stopped=no` and matches no rule: it must not
    # be stored back at the shallow path the refinement just split.
    novel = report(step=50, reversing_for_s=0.0, dynamic_obstacle_blocking_path=True,
                   heading_reversals_last_5s=6, goal_progress_last_5s=0.0)
    leaf, what = shadow.learn(novel, RecoveryMethod.REPLAN_ROUTE, FailureMode.OSCILLATION)
    ok &= _check("a new rule in a refined region enters below the split",
                 what == "insert" and leaf.depth > 2, f"{what}, depth {leaf.depth}")
    ok &= _check("... so the refinement is still reachable",
                 len(shadow.tree.match(signature(reversing)).path) == deep
                 and shadow.tree.match(signature(reversing)).action is RecoveryMethod.REORIENT)
    ok &= _check("no stored path shadows another", shadow.tree.invariant_violations() == [],
                 str(shadow.tree.invariant_violations()))

    print("\n=== 5d. a rule carries the evidence for the mode it names ===")
    # A rule naming REVERSE_TRACKING must be conditioned through `reversing` (feature 4), or it
    # fires on robots that are not reversing at all ("moving with a clear path -> turn on the
    # spot") -- while scoring good outcomes, because a needless recovery leaves no failure evident.
    from failure_monitor.tree import evidence_depth
    ok &= _check("evidence_depth puts each mode below its own test",
                 evidence_depth(FailureMode.REVERSE_TRACKING) > FEATURE_NAMES.index("reversing")
                 and evidence_depth(FailureMode.STUCK) > FEATURE_NAMES.index("stopped")
                 and evidence_depth(None) == 0,
                 f"REVERSE_TRACKING -> {evidence_depth(FailureMode.REVERSE_TRACKING)}")
    deep_enough = DecisionTree(init_depth=2)
    leaf, _ = deep_enough.learn(signature(report(reversing_for_s=5.0, goal_progress_last_5s=0.0)),
                                RecoveryMethod.REORIENT, FailureMode.REVERSE_TRACKING)
    ok &= _check("a REVERSE_TRACKING rule is conditioned on `reversing`, not stored above it",
                 leaf.depth > FEATURE_NAMES.index("reversing")
                 and dict(zip(FEATURE_NAMES, leaf.path))["reversing"] == "long",
                 f"depth {leaf.depth}")
    ok &= _check("... so it does not fire on a robot that is not reversing",
                 deep_enough.match(signature(report(goal_progress_last_5s=1.0))) is None)

    print("\n=== 5e. a diagnosis the report does not support is recorded, not overruled ===")
    # A teacher calling a robot STUCK after 1-4 s stopped, when the evidence states the test as 4 s
    # and says a brief pause is ordinary driving.
    # The tree keeps the rule -- the premise is that the model can see what no threshold covers --
    # but flags it, because otherwise it is repeated indefinitely and silently.
    from failure_monitor.tree import meets_test
    unsupported = TreePolicy(init_depth=2, source="teacher")
    brief = report(step=85, stopped_for_s=2.5)
    leaf, _ = unsupported.learn(brief, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("a STUCK call below the 4 s test is flagged unevidenced",
                 leaf.unevidenced and unsupported.stats.unevidenced == 1
                 and "UNEVIDENCED" in leaf.describe())
    ok &= _check("... and the rule is still learned, not discarded",
                 unsupported.tree.match(signature(brief)) is leaf
                 and leaf.action is RecoveryMethod.REPLAN_ROUTE)
    supported = TreePolicy(init_depth=2, source="teacher")
    pinned = report(step=90, stopped_for_s=9.0)
    leaf, _ = supported.learn(pinned, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    ok &= _check("a supported diagnosis is not flagged",
                 not leaf.unevidenced and supported.stats.unevidenced == 0)
    ok &= _check("meets_test reads each mode's own test, not the priority order",
                 meets_test(report(reversing_for_s=5.0), FailureMode.REVERSE_TRACKING)
                 and not meets_test(report(reversing_for_s=0.5), FailureMode.REVERSE_TRACKING)
                 and meets_test(report(stopped_for_s=9.0), FailureMode.STUCK))

    print("\n=== 6. persistence ===")
    path = os.path.join(tmp_dir, "self_test_tree.json")
    policy.save(path)
    reloaded = TreePolicy.load(path)
    ok &= _check("a saved tree reloads with the same rules",
                 reloaded.tree.to_dict() == policy.tree.to_dict(), f"{len(reloaded.tree)} rules")
    ok &= _check("... and the same episode count", reloaded.episode == policy.episode)
    with open(path) as f:
        data = json.load(f)
    data["features"] = ["something", "else"]
    with open(path, "w") as f:
        json.dump(data, f)
    try:
        TreePolicy.load(path)
        ok &= _check("a tree grown against different features is refused", False)
    except ValueError:
        ok &= _check("a tree grown against different features is refused", True)

    print("\n=== 7. the adapter: miss escalates, hit does not ===")
    teacher = ScriptedTeacher()
    analyzer = TreeAnalyzer(TreePolicy(init_depth=2, source="scripted"), teacher=teacher)
    first = report(step=30, stopped_for_s=9.0, static_obstacle_blocking_path=True,
                   goal_progress_last_5s=0.0)
    decision = analyzer.assess_situation(first, RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("the first occurrence asks the teacher",
                 teacher.calls == 1 and decision.method is RecoveryMethod.REPLAN_ROUTE,
                 f"{teacher.calls} call(s) -> {decision.method.value}")
    again = report(step=200, stopped_for_s=11.0, static_obstacle_blocking_path=True,
                   goal_progress_last_5s=0.0)
    decision = analyzer.assess_situation(again, RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("the same situation later is answered by the tree",
                 teacher.calls == 1 and decision.method is RecoveryMethod.REPLAN_ROUTE
                 and decision.raw_response.startswith("tree:"),
                 f"{teacher.calls} call(s), {decision.raw_response}")
    ok &= _check("the breaker answers without asking anyone",
                 analyzer.assess_situation(report(step=210, recovery_cycles_without_progress=3),
                                           RECOVERY_METHOD_DESCRIPTIONS,
                                           stall_threshold=3).raw_response == "breaker")
    frozen = TreeAnalyzer(TreePolicy(), teacher=None)
    verdict = frozen.assess_situation(report(step=5, stopped_for_s=9.0), RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("with no teacher a miss answers nominally rather than raising",
                 verdict.failure_mode is None and verdict.method is RecoveryMethod.CONTINUE
                 and frozen.policy.stats.unanswered == 1)

    print("\n=== 8. REQUEST_HUMAN and the watchdog ===")
    wall = dict(stopped_for_s=9.0, static_obstacle_blocking_path=True, goal_progress_last_5s=0.0)
    before, after = report(step=25, **wall), report(step=45, goal_reachable=False, **wall)
    named = lambda r: dict(zip(FEATURE_NAMES, signature(r)))
    ok &= _check("a replan that found no route shows up as route=none, and only then",
                 named(before)["route"] == "ok" and named(after)["route"] == "none")
    human = TreePolicy(init_depth=2, source="scripted")
    human.learn(before, RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    leaf, _ = human.learn(after, RecoveryMethod.REQUEST_HUMAN, FailureMode.STUCK)
    ok &= _check("a REQUEST_HUMAN rule is conditioned on the route, however shallow the tree inserts",
                 leaf.depth > FEATURE_NAMES.index("route") and named(after)["route"] in leaf.path,
                 "/".join(leaf.path))
    ok &= _check("... so the same wall with a route still open keeps its replan",
                 human.tree.match(signature(before)).action is RecoveryMethod.REPLAN_ROUTE)
    left_there = report(step=65, stopped_for_s=0.4, dynamic_obstacle_blocking_path=True,
                        static_obstacle_blocking_path=False, blocking_obstacle_stationary_s=41.0)
    leaf, _ = human.learn(left_there, RecoveryMethod.REQUEST_HUMAN, None)
    ok &= _check("a human call with a route still open is conditioned on the blocker being abandoned",
                 leaf.depth > FEATURE_NAMES.index("blocker") and "abandoned" in leaf.path, "/".join(leaf.path))
    ok &= _check("... so it cannot fire at a gate that has only been shut a few seconds",
                 human.tree.match(signature(report(step=65, stopped_for_s=0.4, dynamic_obstacle_blocking_path=True,
                                                   static_obstacle_blocking_path=False,
                                                   blocking_obstacle_stationary_s=6.0))) is None)
    for needed, verdict in ((True, "good"), (False, "bad")):
        judged = TreePolicy(init_depth=2, source="scripted")
        judged.start_episode()
        rule, _ = judged.learn(after, RecoveryMethod.REQUEST_HUMAN, FailureMode.STUCK)
        judged.end_episode("human_requested", needs_human=needed)
        ok &= _check(f"a human call is judged {verdict} when a human was {'needed' if needed else 'not needed'}",
                     (rule.good, rule.bad) == ((1, 0) if needed else (0, 1)), f"{rule.good} ok / {rule.bad} bad")

    gate = dict(stopped_for_s=6.0, dynamic_obstacle_blocking_path=True, static_obstacle_blocking_path=False)
    aware = ScriptedTeacher()
    paused = report(blocking_obstacle_stationary_s=10.0, **gate)
    ok &= _check("an obstacle that has merely paused is waited for",
                 aware.assess_situation(paused).method is RecoveryMethod.WAIT)
    abandoned = report(blocking_obstacle_stationary_s=35.0, recoveries_this_episode=["WAIT@75:ok"], **gate)
    ok &= _check("... one that has not moved for 30 s after a WAIT is a human's problem",
                 aware.assess_situation(abandoned).method is RecoveryMethod.REQUEST_HUMAN
                 and named(abandoned)["blocker"] == "abandoned")
    ok &= _check("... and never before a WAIT has been tried",
                 aware.assess_situation(report(blocking_obstacle_stationary_s=35.0, **gate)).method
                 is RecoveryMethod.WAIT)
    ok &= _check("no route is a human's problem whatever else the report says",
                 aware.assess_situation(after).method is RecoveryMethod.REQUEST_HUMAN)
    cleared = report(holding_position=True, previous_recovery="WAIT", recovery_hold_steps_remaining=18,
                     dynamic_obstacle_blocking_path=False, static_obstacle_blocking_path=False,
                     route_heading_error_deg=4.0)
    ok &= _check("a WAIT whose obstacle has gone is ended with RESUME_ROUTE",
                 aware.assess_situation(cleared).method is RecoveryMethod.RESUME_ROUTE)
    grinding = report(stopped_for_s=0.0, solver_deadline_miss_streak=2, solver_deadline_misses_last_12s=16,
                      dynamic_obstacle_blocking_path=True, static_obstacle_blocking_path=False)
    ok &= _check("a watchdog running down counts as a deadline miss even with the streak broken",
                 named(grinding)["deadline"] == "miss"
                 and named(report(solver_deadline_miss_streak=2))["deadline"] == "ok")
    ok &= _check("... and against a dynamic blockage the answer is WAIT",
                 aware.assess_situation(grinding).method is RecoveryMethod.WAIT)
    chain = TreePolicy(init_depth=2, source="scripted")
    chain.start_episode()
    shut = dict(stopped_for_s=6.0, dynamic_obstacle_blocking_path=True, static_obstacle_blocking_path=False)
    first, _ = chain.learn(report(step=25, route_progress_m=5.0, **shut), RecoveryMethod.WAIT, FailureMode.STUCK)
    chain.observe(report(step=35, holding_position=True, route_progress_m=5.0))
    chain.observe(report(step=55, route_progress_m=5.0, **shut))
    chain.learn(report(step=55, route_progress_m=5.0, **shut), RecoveryMethod.WAIT, FailureMode.STUCK)
    ok &= _check("a WAIT followed by another WAIT is not booked as a failure",
                 (first.good, first.bad) == (0, 0), f"{first.good} ok / {first.bad} bad")
    chain.observe(report(step=65, holding_position=True, route_progress_m=5.0))
    chain.observe(report(step=85, route_progress_m=5.2))
    chain.observe(report(step=115, route_progress_m=9.0))
    ok &= _check("... the chain is judged once it ends, and both holds share the verdict",
                 (first.good, first.bad) == (2, 0), f"{first.good} ok / {first.bad} bad")
    walled = TreePolicy(init_depth=2, source="scripted")
    walled.start_episode()
    wrong, _ = walled.learn(report(step=25, route_progress_m=5.0, **shut), RecoveryMethod.WAIT, FailureMode.STUCK)
    walled.observe(report(step=35, holding_position=True, route_progress_m=5.0))
    walled.learn(report(step=55, route_progress_m=5.0, stopped_for_s=9.0, goal_progress_last_5s=0.0,
                        static_obstacle_blocking_path=True, static_obstacle_gap=0.1), RecoveryMethod.REPLAN_ROUTE,
                 FailureMode.STUCK)
    wrong = walled.tree.match(signature(report(**shut)))   # the split moved the rule; same rule, new path
    ok &= _check("... and a WAIT chain something else has to take over from is still a failure",
                 wrong.action is RecoveryMethod.WAIT and wrong.bad == 1, f"{wrong.good} ok / {wrong.bad} bad")

    looped = TreePolicy(init_depth=2, source="scripted")
    looped.start_episode()
    at_wall = dict(stopped_for_s=9.0, static_obstacle_blocking_path=True, goal_progress_last_5s=0.0)
    idle, _ = looped.learn(report(step=25, route_progress_m=5.0, **at_wall), RecoveryMethod.WAIT, FailureMode.STUCK)
    looped.observe(report(step=35, holding_position=True, route_progress_m=5.0))
    looped.learn(report(step=55, route_progress_m=5.0, **at_wall), RecoveryMethod.WAIT, FailureMode.STUCK)
    looped.note_breaker(report(step=85, route_progress_m=5.0, **at_wall))
    looped.observe(report(step=125, route_progress_m=12.0))     # the breaker's replan got the robot going
    ok &= _check("a WAIT loop at a wall is not credited with the detour the breaker's replan produced",
                 (idle.good, idle.bad) == (0, 2), f"{idle.good} ok / {idle.bad} bad")
    gated = TreePolicy(init_depth=2, source="scripted")
    gated.start_episode()
    patient, _ = gated.learn(report(step=25, route_progress_m=5.0, **shut), RecoveryMethod.WAIT, FailureMode.STUCK)
    gated.note_breaker(report(step=85, route_progress_m=5.0, **shut))
    ok &= _check("... while one at a gate still shut is left pending when the breaker fires",
                 (patient.good, patient.bad) == (0, 0) and len(gated._pending) == 1)

    print("\n=== 8b. a new answer on an old path, and a replan that finds no route ===")
    no_route = dict(stopped_for_s=9.0, static_obstacle_blocking_path=True, goal_progress_last_5s=0.0,
                    goal_reachable=False)
    wall_open = dict(stopped_for_s=9.0, static_obstacle_blocking_path=True, goal_progress_last_5s=0.0)
    for how in ("replace", "contradict"):
        reused = TreePolicy(init_depth=3, retire_after=1, source="scripted")
        reused.start_episode()
        old_rule, _ = reused.learn(report(step=25, **no_route), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
        if how == "replace":
            old_rule.retired = True
        leaf, what = reused.learn(report(step=45, **no_route), RecoveryMethod.REQUEST_HUMAN, FailureMode.STUCK)
        ok &= _check(f"a REQUEST_HUMAN that comes in by '{how}' is still conditioned on the route",
                     what == how and len(leaf.path) > FEATURE_NAMES.index("route")
                     and leaf.path[FEATURE_NAMES.index("route")] == "none", f"{what}: {'/'.join(leaf.path)}")
        ok &= _check("... and does not answer for a wall that still has a way round",
                     reused.tree.match(signature(report(**wall_open))) is None)
    found = TreePolicy(init_depth=3, retire_after=2, source="scripted")
    found.start_episode()
    replan, _ = found.learn(report(step=25, route_progress_m=5.0, **wall_open), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    found.observe(report(step=35, route_progress_m=5.0, **no_route))
    found.observe(report(step=75, route_progress_m=5.0, **no_route))
    ok &= _check("a replan that came back with 'no route exists' is not judged on the ground it did not gain",
                 (replan.good, replan.bad) == (0, 0) and not found._pending, f"{replan.good} ok / {replan.bad} bad")
    again, _ = found.learn(report(step=85, route_progress_m=5.0, **no_route), RecoveryMethod.REPLAN_ROUTE, FailureMode.STUCK)
    found.observe(report(step=125, route_progress_m=5.0, **no_route))
    ok &= _check("... but replanning again once the report says so is judged like any other recovery",
                 again.bad == 1, f"{again.good} ok / {again.bad} bad")

    print("\n=== 8c. the reference shadow: decisions scored where they are made ===")

    class _Fixed:
        def __init__(self, method, raw):
            self.method, self.raw = method, raw
        def assess_situation(self, context, candidates, evidence=None, stall_threshold=None):
            return SituationAssessment(method=self.method, rationale="", raw_response=self.raw, failure_mode=None)
    gate_report = report(step=75, stopped_for_s=6.0, dynamic_obstacle_blocking_path=True,
                         static_obstacle_blocking_path=False, dynamic_obstacle_gap=0.4)
    driving_report = report(step=85, stopped_for_s=0.0)
    watched = ReferenceShadow(_Fixed(RecoveryMethod.WAIT, "tree:no/long/dynamic"))
    given = watched.assess_situation(gate_report, RECOVERY_METHOD_DESCRIPTIONS)
    watched.assess_situation(driving_report, RECOVERY_METHOD_DESCRIPTIONS)
    tally = watched.take()
    ok &= _check("the shadow passes the wrapped answer through untouched", given.method is RecoveryMethod.WAIT)
    ok &= _check("it scores each tick against the scripted rules and says who answered",
                 (tally["ticks"], tally["agree"]) == (2, 1) and tally["by_source"] == {"tree": [1, 2]}, str(tally))
    ok &= _check("... counts the recovery the rules would not have ordered",
                 tally["recoveries_ordered"] == 2 and tally["recoveries_reference_would_not_order"] == 1
                 and tally["differs"] == {"CONTINUE -> WAIT": 1})
    ok &= _check("... and starts clean for the next episode", watched.take()["ticks"] == 0)

    print("\n=== 9. experience: what the teacher is shown of what has been learned ===")
    from failure_monitor.experience import ExperienceMemory, ExperienceTable, situation_key
    exp_policy = TreePolicy(init_depth=3, retire_after=2, source="scripted")
    exp_table = ExperienceTable()
    exp_policy.listeners.append(exp_table.on_verdict)
    exp_policy.start_episode()
    at_gate = dict(stopped_for_s=6.0, dynamic_obstacle_blocking_path=True, static_obstacle_blocking_path=False,
                   blocking_obstacle_stationary_s=8.0, route_heading_error_deg=5.0)
    exp_policy.learn(report(step=25, route_progress_m=5.0, **at_gate), RecoveryMethod.WAIT, FailureMode.STUCK)
    exp_policy.observe(report(step=35, holding_position=True, route_progress_m=5.0))
    exp_policy.observe(report(step=55, route_progress_m=5.1))
    exp_policy.observe(report(step=85, route_progress_m=8.0))
    gate_sig = signature(report(**at_gate))
    entry = exp_table.entries.get((situation_key(gate_sig), "WAIT"))
    ok &= _check("a judge verdict reaches the experience table as it is made",
                 entry is not None and (entry.good, entry.bad) == (1, 0), str(exp_table.to_dict()["entries"]))
    exp_table.record_counterfactual(gate_sig, "WAIT", "needed")
    exp_table.record_counterfactual(gate_sig, "WAIT", "needed")
    exp_table.record_counterfactual(gate_sig, "REPLAN_ROUTE", "unnecessary")
    longer = report(**{**at_gate, "blocking_obstacle_stationary_s": 35.0})
    text = ExperienceMemory(exp_policy.tree, exp_table, "both").for_report(longer)
    ok &= _check("the memory names the rule covering the report and how it was judged",
                 "The rule that covers this report" in text and "ACTION WAIT" in text and "1 good / 0 bad" in text)
    ok &= _check("... and the nearest recorded situation, with what differs and the counterfactual labels",
                 "blocker=parked (now abandoned)" in text and "2 needed" in text and "1 unnecessary" in text)
    ok &= _check("a situation with nothing like it on record says so rather than showing another's scores",
                 "nothing similar" in ExperienceMemory(exp_policy.tree, exp_table, "table").for_report(
                     report(holding_position=True, reversing_for_s=5.0, route_heading_error_deg=150.0)))
    table_file = os.path.join(tmp_dir, "experience.json")
    exp_table.save(table_file)
    ok &= _check("the table survives a save and a load",
                 ExperienceTable.open(table_file).to_dict() == exp_table.to_dict())

    class _Listens:
        def __init__(self):
            self.seen = "never called"
        def assess_situation(self, context, candidates=None, evidence=None, stall_threshold=None, experience=None):
            self.seen = experience
            return SituationAssessment(method=RecoveryMethod.CONTINUE, rationale="", raw_response="x", failure_mode=None)
    listens = _Listens()
    asked = TreeAnalyzer(TreePolicy(init_depth=2), teacher=listens,
                         memory=ExperienceMemory(exp_policy.tree, exp_table, "both"))
    asked.assess_situation(report(step=5, **at_gate), RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("a teacher that takes `experience` is given it", isinstance(listens.seen, str) and "WAIT" in listens.seen)
    plain_teacher = ScriptedTeacher()
    TreeAnalyzer(TreePolicy(init_depth=2), teacher=plain_teacher,
                 memory=ExperienceMemory(exp_policy.tree, exp_table, "both")).assess_situation(
        report(step=5, **at_gate), RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("... and one that does not is asked the way it always was", plain_teacher.calls == 1)
    listens.seen = "never called"
    TreeAnalyzer(TreePolicy(init_depth=2), teacher=listens).assess_situation(
        report(step=5, **at_gate), RECOVERY_METHOD_DESCRIPTIONS)
    ok &= _check("without a memory the teacher's prompt is untouched", listens.seen is None)

    print("\n=== 10. an open road with a high miss count, and replay labels that retire a rule ===")
    open_road = dict(solver_deadline_miss_streak=1, solver_deadline_misses_last_12s=18,
                     static_obstacle_blocking_path=False, dynamic_obstacle_blocking_path=False)
    roomy = report(**open_road, watchdog_steps_remaining=27)
    ok &= _check("the evidence and WAIT's description both name the margin",
                 "above 20 it is CONTINUE" in monitor_evidence()[FailureMode.SOLVER_DEADLINE_MISS]
                 and "20 or less" in recovery_candidates()[RecoveryMethod.WAIT]
                 and list(recovery_candidates())[0] is RecoveryMethod.REQUEST_HUMAN)
    ok &= _check("a report without the margin bins as `miss`, and the margin never reaches the prompt",
                 dict(zip(FEATURE_NAMES, signature(report(**open_road, watchdog_steps_remaining=3))))["deadline"] == "miss"
                 and "watchdog_margin" not in report(**open_road, watchdog_margin=20).to_dict())
    tight = report(**open_road, watchdog_steps_remaining=12, watchdog_margin=20)
    ok &= _check("... the deadline bin splits at it",
                 dict(zip(FEATURE_NAMES, signature(report(**open_road, watchdog_steps_remaining=27, watchdog_margin=20))))["deadline"] == "miss"
                 and dict(zip(FEATURE_NAMES, signature(tight)))["deadline"] == "critical")
    ok &= _check("... on an open road only: a robot at a gate bins as it always did",
                 dict(zip(FEATURE_NAMES, signature(report(**{**open_road, 'dynamic_obstacle_blocking_path': True},
                                                          watchdog_steps_remaining=12, watchdog_margin=20))))["deadline"] == "miss")
    margin_teacher = ScriptedTeacher()
    ok &= _check("... and the scripted rules hold the robot only below it",
                 margin_teacher.assess_situation(roomy).method is RecoveryMethod.CONTINUE
                 and margin_teacher.assess_situation(report(**open_road, watchdog_steps_remaining=12)).method is RecoveryMethod.CONTINUE
                 and margin_teacher.assess_situation(tight).method is RecoveryMethod.WAIT)

    cf_policy = TreePolicy(init_depth=3, cf_retire_after=5)
    idle_wait, gate_wait = roomy, report(**{**open_road, 'dynamic_obstacle_blocking_path': True}, dynamic_obstacle_gap=1.0)
    for ctx in (idle_wait, gate_wait):
        cf_policy.learn(ctx, RecoveryMethod.WAIT, FailureMode.SOLVER_DEADLINE_MISS)
    for n in range(12):
        cf_policy.note_counterfactual(signature(gate_wait), RecoveryMethod.WAIT, "needed" if n % 3 == 0 else "harmful")
    for n in range(4):
        cf_policy.note_counterfactual(signature(idle_wait), RecoveryMethod.WAIT, "harmful")
    cf_policy.note_counterfactual(signature(idle_wait), RecoveryMethod.WAIT, "unclear")
    cf_policy.note_counterfactual(signature(idle_wait), RecoveryMethod.REPLAN_ROUTE, "harmful")
    ok &= _check("four replays saying 'not needed' are not yet enough; 'unclear' and another action's labels do not count",
                 cf_policy.decide(idle_wait) is not None and cf_policy.tree.match(signature(idle_wait)).cf_not_needed == 4)
    cf_policy.note_counterfactual(signature(idle_wait), RecoveryMethod.WAIT, "unnecessary")
    ok &= _check("the fifth retires the rule: the next match goes back to the teacher",
                 cf_policy.decide(idle_wait) is None and cf_policy.tree.match(signature(idle_wait)).retired)
    ok &= _check("a rule needed in a third of its replays is kept, however many say 'harmful'",
                 cf_policy.decide(gate_wait) is not None and cf_policy.tree.match(signature(gate_wait)).cf_not_needed == 8)
    counting = TreePolicy(init_depth=3)
    counting.learn(idle_wait, RecoveryMethod.WAIT, FailureMode.SOLVER_DEADLINE_MISS)
    for n in range(20):
        counting.note_counterfactual(signature(idle_wait), RecoveryMethod.WAIT, "harmful")
    ok &= _check("off by default: labels are counted and nothing is retired", counting.decide(idle_wait) is not None)
    ok &= _check("the counts survive a save and a load; a tree without any is stored as before",
                 DecisionTree.from_dict(counting.tree.to_dict()).match(signature(idle_wait)).cf_not_needed == 20
                 and all("cf_needed" not in rule for rule in TreePolicy(init_depth=2).tree.to_dict()["rules"])
                 and "cf_needed" not in json.dumps(analyzer.policy.tree.to_dict()))

    print("\n=== the tree these checks grew ===")
    print("  " + analyzer.policy.tree.render().replace("\n", "\n  "))
    return bool(ok)


if __name__ == "__main__":
    ok = self_test(tempfile.mkdtemp(prefix="failure_monitor_tree_"))
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
