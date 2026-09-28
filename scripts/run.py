"""Run episodes on the scenario grid with a decision tree carried across them.

    python scripts/run.py --teacher llm --episodes 200 --seed 100 --reset --memory both --simulate \\
        --cf-retire-after 10 --checkpoint-every 25 --tree runs/method.json --record runs/method

On every assessment tick the tree answers if a trusted rule covers the report; otherwise the teacher
is asked and its answer is folded into the tree. `--teacher llm` is GPT-6 Luna over the OpenAI API
(needs OPENAI_API_KEY), `--teacher scripted` the scripted rules (no API, the pipeline in minutes),
`--teacher none` freezes the tree and answers misses nominally (how a grown tree is evaluated).

The tree is stored as JSON (`--tree`) and reloaded on the next run, so learning accumulates across
invocations; `--reset` starts from an empty tree. `--record DIR` keeps every episode for
`scripts/visualize_run.py` and the analysis scripts. Next to the tree go the experience table
(`<tree>.experience.json`), the teacher log (`<tree>.jsonl`: every teacher call with its report) and
the API usage log (`<tree>.api.jsonl`: latency, tokens, cost per call).
"""
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from typing import Literal

import tyro  # type: ignore

from failure_monitor import ReferenceShadow, RecoveryMethod, ScriptedTeacher, TreeAnalyzer, TreePolicy, signature  # noqa: E402


@dataclass
class EpisodeRow:
    """One episode's line in the learning curve."""
    episode: int
    ticks: int
    hits: int
    escalations: int
    unanswered: int
    success: bool
    steps: int
    rules: int
    mean_age_steps: float
    wall_s: float
    termination: str
    agree: int
    shadow_ticks: int
    recoveries: int
    unwanted: int

    @property
    def hit_rate(self) -> float:
        """Share of assessments the tree answered (misses with no teacher count against it)."""
        decided = self.hits + self.escalations + self.unanswered
        return self.hits / decided if decided else 0.0


@dataclass
class Args:
    """Run episodes on the scenario grid with a decision tree carried across them.

    Args:
        episodes: How many episodes to run, with the tree carried across them.
        teacher: Who answers a tick the tree has no rule for: "llm" (the OpenAI model), "scripted"
            (the scripted rules) or "none" (frozen tree; misses answered NOMINAL/CONTINUE).
        model: OpenAI model name for `--teacher llm`.
        temperature: Sampling temperature for the model.
        seed: Seed of the first episode; episode k gets `seed + k` (see `vary_seed`).
        vary_seed: Give each episode its own layout. Off repeats one layout (`seed`) every episode.
        scenario: Scenario-grid option: 1 is the mixed stream; 2-7 stage one situation each (wall,
            gate, sealed, breakdown, reversed, traffic).
        max_steps: Step budget per episode (0.2 s each).
        tree: Where the tree is stored. The experience table, teacher log and API usage log go next
            to it, named after it.
        reset: Start from an empty tree (and experience table) instead of loading the stored one.
        save: Write the tree back after every episode.
        checkpoint_every: Also save a copy of the tree every N episodes (`<tree>.epNNNN.json`), to be
            evaluated frozen afterwards. 0 saves none.
        learn: Fold the teacher's answers into the tree. Off keeps the tree as it is (with an empty
            tree and a teacher: the teacher alone).
        init_depth: Features a brand-new rule is conditioned on before anything contradicts it.
        min_support: Episodes whose teacher answers a rule needs before the tree answers alone.
        horizon_steps: Control cycles a recovery is judged over, from when it (or its hold) ends.
        retire_after: Bad judge verdicts before a rule is retired. The default never retires: the
            judge scores, and only replay labels (`cf_retire_after`) retire.
        cf_retire_after: With `--simulate`: retire a recovery rule once this many replays labelled
            its recoveries not needed, provided `needed` is at most one label in ten. None only
            counts the labels.
        memory: What the teacher is shown of what has been learned, ahead of each report: "both"
            (the nearest rules with their scores, and the scored situation-action table) or "off".
        simulate: Between episodes, replay the episode just driven once per recovery with that
            recovery replaced by CONTINUE, and file the labels (needed / unnecessary / harmful) in the
            experience table and on the rules.
        sim_workers: Replay processes for `--simulate`.
        sim_grace: Steps after the replaced decision answered NOMINAL in a replay before the scripted
            rules take over.
        record: Directory to record every episode into. Empty records nothing.
        api_budget_usd: Stop the run once the teacher has cost this much.
        verbose: Print each report, decision and tree edit.
        show: Print the stored tree and its history, and exit.
    """
    episodes: int = 5
    teacher: Literal["llm", "scripted", "none"] = "scripted"
    model: str = "gpt-6-luna"
    temperature: float = 0.0
    seed: int = 100
    vary_seed: bool = True
    scenario: int = 1
    max_steps: int = 500
    tree: str = "runs/tree.json"
    reset: bool = False
    save: bool = True
    checkpoint_every: int = 0
    learn: bool = True
    init_depth: int = 3
    min_support: int = 2
    horizon_steps: int = 25
    retire_after: int = 1_000_000
    cf_retire_after: int | None = None
    memory: Literal["off", "both"] = "off"
    simulate: bool = False
    sim_workers: int = 4
    sim_grace: int = 50
    record: str = ""
    api_budget_usd: float = 2.0
    verbose: bool = False
    show: bool = False


def build_teacher(args: Args, stem: str):
    if args.teacher == "llm":
        from failure_monitor.llm import LLMConfig
        from failure_monitor.openai_teacher import OpenAITeacher
        usage_log = stem + ".api.jsonl"
        print(f"=== Teacher: {args.model} over the OpenAI API, temperature {args.temperature:g}; usage -> {usage_log} ===")
        return OpenAITeacher(LLMConfig(model_name=args.model, temperature=args.temperature),
                             usage_log=usage_log, max_usd=args.api_budget_usd), args.model
    if args.teacher == "scripted":
        return ScriptedTeacher(), "scripted"
    return None, "none"


def run_episodes(args: Args) -> tuple[list[EpisodeRow], TreePolicy, list[str]]:
    from failure_monitor.episode import run_episode, seed_everything
    from failure_monitor.experience import ExperienceMemory, ExperienceTable
    from failure_monitor.recorder import EpisodeRecorder, RunRecorder

    stem = os.path.splitext(args.tree)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.tree)), exist_ok=True)
    teacher, source = build_teacher(args, stem)
    policy = TreePolicy.open(args.tree, reset=args.reset, init_depth=args.init_depth,
                             horizon_steps=args.horizon_steps, retire_after=args.retire_after,
                             min_support=args.min_support, source=source, cf_retire_after=args.cf_retire_after)
    print(f"=== Tree: {args.tree} -- {len(policy.tree)} rule(s) carried in, {policy.episode} episode(s) of history ===")
    if teacher is None and not len(policy.tree):
        print("  NOTE: --teacher none with an empty tree: every tick is answered NOMINAL/CONTINUE, which is "
              "the plain MPC with no recovery.")

    table, memory, replayer = None, None, None
    table_path = stem + ".experience.json"
    if args.memory == "both" or args.simulate:
        table = ExperienceTable.open(table_path, reset=args.reset)
        policy.listeners.append(table.on_verdict)      # the judge's verdicts, online
        print(f"=== Experience table: {table_path} -- {len(table.entries)} situation-action pair(s) carried in ===")
    if args.memory == "both":
        memory = ExperienceMemory(policy.tree, table, mode="both")
    if args.simulate:
        from failure_monitor.counterfactual import ExperienceReplayer
        replayer = ExperienceReplayer(workers=args.sim_workers, grace=args.sim_grace)

    episode_index = 0
    teacher_log = stem + ".jsonl" if teacher is not None else None

    def log_teacher(context, decision, experience):
        with open(teacher_log, "a") as f:
            f.write(json.dumps({
                "episode": episode_index, "step": int(getattr(context, "step", 0) or 0),
                "signature": list(signature(context)),
                "mode": decision.failure_mode.value if decision.failure_mode else None,
                "method": decision.method.value, "report": context.to_dict(),
                **({"experience": experience} if experience else {}),
            }) + "\n")

    run_recorder = None
    if args.record:
        run_recorder = RunRecorder(args.record, teacher=args.teacher, model=args.model if args.teacher == "llm" else None,
                                   scenario=args.scenario, seed=args.seed, vary_seed=args.vary_seed, tree=args.tree,
                                   init_depth=args.init_depth, min_support=args.min_support,
                                   retire_after=args.retire_after, cf_retire_after=args.cf_retire_after,
                                   learn=args.learn, rules_at_start=len(policy.tree), memory=args.memory,
                                   simulate=args.simulate, episodes_of_history_at_start=policy.episode)

    rows: list[EpisodeRow] = []
    events_before = len(policy.events)
    with TreeAnalyzer(policy, teacher=teacher, verbose=args.verbose, learn=args.learn, memory=memory,
                      on_teacher=log_teacher if teacher_log else None) as analyzer:
        # The shadow scores every tick against the scripted rules; it never changes an answer.
        shadow = ReferenceShadow(analyzer)
        for index in range(args.episodes):
            episode_index = index
            analyzer.start_episode()
            stats_before = policy.stats.snapshot()
            seed = args.seed + index if args.vary_seed else args.seed
            seed_everything(seed)
            print(f"\n=== Episode {index + 1}/{args.episodes} (seed {seed}) ===")
            recorder = run_recorder.new_episode() if run_recorder is not None else None
            if recorder is None and replayer is not None:
                recorder = EpisodeRecorder()   # the replays need the episode's trace
            timeline_before = len(policy.timeline)
            result = run_episode(shadow, scenario_option=args.scenario, max_steps=args.max_steps,
                                 verbose=args.verbose, recorder=recorder)
            analyzer.end_episode(result.termination, result.needs_human)
            delta = policy.stats.delta(stats_before)
            quality = shadow.take()
            if recorder is not None:
                recorder.finish(reference=quality)
                recorder.recording.judgements.extend(policy.timeline[timeline_before:])
                recorder.finish(episode=policy.episode, seed=seed, rules=len(policy.tree),
                                active_rules=len(policy.tree.active), tree=dataclasses.asdict(delta))
                if replayer is not None:
                    labels = replayer.label_episode(recorder.recording.to_dict(), table)
                    recorder.finish(counterfactual=[{k: v for k, v in row.items() if k != "signature"}
                                                    for row in labels])
                    # The same labels, told to the rules themselves (they retire a rule only with
                    # `--cf-retire-after`). Rows past the replay's first drift are left out.
                    before = policy.stats.retirements
                    for row in labels:
                        if row.get("clean", True):
                            policy.note_counterfactual(tuple(row["signature"]), RecoveryMethod(row["method"]),
                                                       row["label"], step=int(row["report_step"]))
                    if policy.stats.retirements > before:
                        recorder.recording.judgements.extend(policy.timeline[-(policy.stats.retirements - before):])
                        print(f"  [experience] {policy.stats.retirements - before} rule(s) retired on replay labels")
                if run_recorder is not None:
                    run_recorder.save_episode(index + 1, recorder)
            if table is not None:
                table.save(table_path)
            rows.append(EpisodeRow(
                episode=policy.episode, ticks=delta.ticks, hits=delta.hits, escalations=delta.escalations,
                unanswered=delta.unanswered, success=result.success, steps=result.steps, rules=len(policy.tree),
                mean_age_steps=result.timing.mean_age_steps if result.timing else 0.0,
                wall_s=result.timing.wall_s if result.timing else 0.0, termination=result.termination,
                agree=quality["agree"], shadow_ticks=quality["ticks"],
                recoveries=quality["recoveries_ordered"], unwanted=quality["recoveries_reference_would_not_order"]))
            print(f"  -> {result.termination} in {result.steps} steps; tree: {delta}")
            if args.save:
                policy.save(args.tree)
            if args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
                ckpt = f"{stem}.ep{index + 1:04d}.json"
                policy.save(ckpt)
                print(f"  checkpoint: {ckpt}")
    if replayer is not None:
        print(f"\n=== Between-episode replays: {json.dumps(replayer.stats)} ===")
        replayer.close()
    if table is not None:
        print(f"=== Experience table: {len(table.entries)} situation-action pair(s), {table.verdicts} judge "
              f"verdict(s), {table.simulations} counterfactual label(s); saved to {table_path} ===")
    return rows, policy, policy.events[events_before:]


def print_curve(rows: list[EpisodeRow]) -> None:
    """How much of the deciding the tree took over, episode by episode."""
    print("\n=== Learning curve ===")
    frozen = any(row.unanswered for row in rows)
    miss_head = f"  {'miss':>4}" if frozen else ""
    print(f"  {'ep':>3}  {'ticks':>5}  {'tree':>5}  {'llm':>4}{miss_head}  {'hit%':>5}  "
          f"{'rules':>5}  {'age':>5}  {'steps':>5}  {'wall_s':>7}  {'agree':>6}  {'recov':>5}  {'unwanted':>8}  ended")
    for row in rows:
        miss = f"  {row.unanswered:>4}" if frozen else ""
        agree = 100 * row.agree / row.shadow_ticks if row.shadow_ticks else 100
        ended = ("right" if row.success else "wrong") + f" ({row.termination})"
        print(f"  {row.episode:>3}  {row.ticks:>5}  {row.hits:>5}  {row.escalations:>4}{miss}  "
              f"{100 * row.hit_rate:>4.0f}%  {row.rules:>5}  {row.mean_age_steps:>5.1f}  "
              f"{row.steps:>5}  {row.wall_s:>7.1f}  {agree:>5.0f}%  {row.recoveries:>5}  {row.unwanted:>8}  {ended}")
    ticks, agree = sum(r.shadow_ticks for r in rows), sum(r.agree for r in rows)
    print(f"  agree: ticks whose action matched the scripted rules ({agree}/{ticks}). recov: recoveries ordered; "
          f"unwanted: those the rules would not have ordered ({sum(r.unwanted for r in rows)} of "
          f"{sum(r.recoveries for r in rows)}).\n  age: mean control cycles between a report and its verdict "
          f"being acted on (1 is a tree hit).")
    if frozen:
        print("  miss: ticks no rule covered, with no teacher to ask -- answered NOMINAL/CONTINUE.")


def show(path: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"No tree at {path}.")
    policy = TreePolicy.load(path)
    print(f"=== {path} ===\n  {len(policy.tree)} rule(s) over {policy.episode} episode(s) of history\n")
    print(policy.tree.render())
    flagged = [leaf for leaf in policy.tree.leaves.values() if leaf.unevidenced]
    print(f"\n  {len(policy.tree.active)} active, {len(policy.tree) - len(policy.tree.active)} retired, "
          f"{len(flagged)} unevidenced (a diagnosis the report's own test did not support).")
    print(f"\n=== How it was learned ===\n  {policy.stats}")
    if policy.events:
        print("\n=== Every change ever made to it ===")
        for line in policy.events:
            print(f"  {line}")


def main(args: Args) -> None:
    if args.show:
        show(args.tree)
        return
    rows, policy, events = run_episodes(args)
    print_curve(rows)
    print("\n=== The policy the robot is running on ===")
    print("  " + policy.tree.render().replace("\n", "\n  "))
    print(f"\n  {len(policy.tree.active)} active rule(s), {len(policy.tree) - len(policy.tree.active)} retired, "
          f"over {policy.episode} episode(s) of history.")
    if events:
        print("\n=== What changed in the tree this run ===")
        for line in events:
            print(f"  {line}")
    print(f"\n=== Totals ===\n  {policy.stats}")
    if args.save:
        print(f"  saved to {args.tree}")
    right = sum(1 for row in rows if row.success)
    called = sum(1 for row in rows if row.success and row.termination == "human_requested")
    print(f"\n  ended right in {right}/{len(rows)} episode(s)" + (f" ({called} by rightly calling a human)" if called else ""))
    # A path that is a proper prefix of another would silently disable every rule beneath it.
    violations = policy.tree.invariant_violations()
    if violations:
        sys.exit(f"the grown tree has shadowed rules: {violations}")


if __name__ == "__main__":
    # The replay workers are started with spawn, which re-imports this module: keep the guard.
    main(tyro.cli(Args))
