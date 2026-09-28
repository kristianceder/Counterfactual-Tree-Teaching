"""Every number in the paper's tables and results text, counted from the runs under results/runs/.

    python scripts/analysis/paper_tables.py     # -> results/tables/paper_tables.md

Counted from the recordings (`episode_*.json`), the tree files and the API usage logs; nothing is
copied from another summary. Runs that do not exist (yet) are listed as missing and left out, so the
script can be re-run while a repeat is still going. Model calls are decisions whose source is the
model. Replay labels are the learning runs' own `--simulate` labels (`summary.counterfactual`), and
for the scripted rules, which run no replays of their own, the labels `scripts/replay_run.py` wrote
next to their recording (`counterfactual_epNNN.json`); either is counted only where the replay was
clean up to the recovery's own step.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RUNS", os.path.normpath(os.path.join(HERE, "..", "..", "results", "runs")))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "tables"))
REPS = ("_r1", "_r2", "_r3")

# Held-out layouts, seeds 300-319 (tab:main). The frozen tree is the method arm's after 200 layouts.
MAIN = [("No recovery", ["no_recovery_heldout"]),
        ("Scripted rules", ["scripted_heldout"]),
        ("LLM alone, real time", [f"luna_heldout{r}" for r in REPS]),
        ("Frozen tree, no model", [f"method{r}_heldout_ep0200" for r in REPS])]
# Learning layouts, seeds 100-299 (tab:method).
ARMS = [("Scripted rules, no learning", ["scripted_learning"]),
        ("Distillation alone", [f"distillation{r}" for r in REPS]),
        ("Experience, no replays", [f"experience_noreplay{r}" for r in REPS]),
        ("Experience with replays (method)", [f"method{r}" for r in REPS])]
UNSEEN = [("Scripted rules", ["scripted_unseen"]),
          ("Distillation alone", [f"distillation{r}_unseen" for r in REPS]),
          ("Experience, no replays", [f"experience_noreplay{r}_unseen" for r in REPS]),
          ("Experience with replays (method)", [f"method{r}_unseen" for r in REPS])]
SLACK = 10


def episodes(run: str) -> list[dict]:
    d = os.path.join(DATA, run)
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "episode_*.json")))]


def right(s: dict) -> bool:
    return bool(s["success"]) if s.get("success") is not None else s["termination"] == "goal"


def by_seed(eps: list[dict]) -> dict[int, dict]:
    return {e["summary"]["seed"]: e for e in eps}


def stats(eps: list[dict]) -> dict:
    n = len(eps)
    term = collections.Counter(e["summary"]["termination"] for e in eps)
    need = [e for e in eps if e["summary"].get("needs_human")]
    goal_steps = [e["summary"]["steps"] for e in eps if e["summary"]["termination"] == "goal"]
    dec = [d for e in eps for d in e["decisions"]]
    llm = [d for d in dec if d["source"] == "llm"]
    tree = [d for d in dec if d["source"] == "tree"]
    ages = [d["acted_step"] - d["report_step"] for d in dec]
    model = any(e["meta"].get("model") for e in eps) or any(d.get("latency_s") for d in llm if d.get("latency_s"))
    ref = [e["summary"].get("reference") or {} for e in eps]
    ticks, agree = sum(r.get("ticks", 0) for r in ref), sum(r.get("agree", 0) for r in ref)
    ordered = sum(r.get("recoveries_ordered", 0) for r in ref)
    unwanted = sum(r.get("recoveries_reference_would_not_order", 0) for r in ref)
    recs = [r for e in eps for r in e["recoveries"] if not str(r.get("detail", "")).startswith("follow-up")]
    methods = collections.Counter(r["method"] for r in recs)
    first, last = eps[:25], eps[-25:]
    calls = lambda part: sum(1 for e in part for d in e["decisions"] if d["source"] == "llm") / max(len(part), 1)
    return dict(n=n, right=sum(right(e["summary"]) for e in eps), goal=term["goal"],
                human=sum(e["summary"]["termination"] == "human_requested" for e in need), need=len(need),
                false_alarm=sum(e["summary"]["termination"] == "human_requested" and not e["summary"].get("needs_human") for e in eps),
                timeout=term["timeout"], collision=term["collision"], watchdog=term["watchdog"],
                steps=st.mean(goal_steps) if goal_steps else float("nan"),
                calls=len(llm) / n if model else float("nan"), calls_first=calls(first) if model else float("nan"),
                calls_last=calls(last) if model else float("nan"), llm_total=len(llm),
                age=st.mean(ages) if ages else float("nan"), tree_share=len(tree) / len(dec) if dec and tree else float("nan"),
                agree=agree / ticks if ticks and (llm or tree) else float("nan"),
                unwanted=unwanted / ordered if ordered else float("nan"),
                recoveries=len(recs), per_ep=len(recs) / n, methods=dict(methods))


def cf_stream(eps: list[dict]) -> dict:
    c = collections.Counter()
    for e in eps:
        s = e["summary"]
        for x in s.get("counterfactual") or []:
            c["recoveries"] += 1
            if not x.get("clean", True):
                continue
            c["clean"] += 1
            c[x["label"]] += 1
            c[f"{x['label']}:{x['method']}"] += 1
            if x["label"] == "harmful":
                c["harmful_outcome" if not right(s) else "harmful_steps"] += 1
    return dict(c)


def cf_files(run: str) -> dict:
    """`cf_stream` for a run replayed after the fact (`scripts/replay_run.py`), from its files."""
    c = collections.Counter()
    for f in sorted(glob.glob(os.path.join(DATA, run, "counterfactual_ep*.json"))):
        d = json.load(open(f))
        for x in d.get("counterfactuals") or []:
            c["recoveries"] += 1
            if not x.get("replay_clean_to_override", True):
                continue
            c["clean"] += 1
            c[x["label"]] += 1
            c[f"{x['label']}:{x['method']}"] += 1
            if x["label"] == "harmful":
                c["harmful_outcome" if not right(d["recorded"]) else "harmful_steps"] += 1
    return dict(c)


def replay_labels(run: str, eps: list[dict]) -> dict:
    """The run's replay labels: its own, or failing that the ones written after the fact."""
    return cf_stream(eps) or cf_files(run)


def tree_info(run: str) -> dict:
    path = os.path.join(DATA, f"{run}.json")
    if not os.path.exists(path):
        return {}
    t = json.load(open(path))
    retired = [e for e in t["events"] if "] retired " in e]
    by_label = [e for e in retired if "replayed without it" in e]
    return dict(rules=len(t["rules"]), retired_judge=len(retired) - len(by_label), retired_label=len(by_label),
                label_events=[e.split(":")[0] for e in by_label])


def heldout_curve(run: str) -> list[int]:
    out = []
    for c in range(25, 201, 25):
        eps = episodes(f"{run}_heldout_ep{c:04d}")
        if len(eps) == 20:
            out.append(sum(right(e["summary"]) for e in eps))
    return out


def api(run: str) -> dict:
    path = os.path.join(DATA, f"{run}.api.jsonl")
    if not os.path.exists(path):
        return {}
    rows = [json.loads(line) for line in open(path)]
    ok = [r for r in rows if r["ok"]]
    prompt = [r["usage"]["input_tokens"] + r["usage"]["cache_creation_input_tokens"] + r["usage"]["cache_read_input_tokens"]
              for r in ok]
    return dict(calls=len(ok), failed=len(rows) - len(ok), latency=st.median(r["total_s"] for r in ok) if ok else float("nan"),
                usd=sum(r["cost_usd"] for r in rows), prompt=st.mean(prompt) if ok else float("nan"),
                cached=st.mean(r["usage"]["cache_read_input_tokens"] for r in ok) if ok else float("nan"))


def mcnemar(a: dict[int, dict], b: dict[int, dict]) -> tuple[int, int, float]:
    """Layouts right only under `a`, only under `b`, and the exact two-sided p on the discordant pairs."""
    seeds = sorted(set(a) & set(b))
    x = sum(right(a[s]["summary"]) and not right(b[s]["summary"]) for s in seeds)
    y = sum(right(b[s]["summary"]) and not right(a[s]["summary"]) for s in seeds)
    n, k = x + y, min(x, y)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def steps_diff(a: dict[int, dict], b: dict[int, dict], boot: int = 10000) -> tuple[float, float, float, int]:
    """Mean steps to goal of `b` minus `a` on layouts both reach the goal, with a 95% bootstrap interval."""
    both = [s for s in set(a) & set(b) if a[s]["summary"]["termination"] == "goal" and b[s]["summary"]["termination"] == "goal"]
    d = [b[s]["summary"]["steps"] - a[s]["summary"]["steps"] for s in both]
    if not d:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(0)
    means = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(boot))
    return st.mean(d), means[int(0.025 * boot)], means[int(0.975 * boot)], len(d)


def agg(vals: list[float], fmt: str = "{:.0f}") -> str:
    v = [x for x in vals if x == x]
    if not v:
        return "--"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(st.mean(v))} ± {fmt.format(st.stdev(v))} ({', '.join(fmt.format(x) for x in v)})"


def kinds(eps: list[dict]) -> dict[str, list[int]]:
    """Per staged situation: [layouts holding it, of them ended right]."""
    out = collections.defaultdict(lambda: [0, 0])
    for e in eps:
        for k in {s["kind"] for s in e["meta"]["layout"].get("situations", [])}:
            out[k][0] += 1
            out[k][1] += right(e["summary"])
    return dict(out)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    md = ["# Paper numbers from the GPT-6 Luna runs", "",
          "Generated by `scripts/analysis/paper_tables.py`. Mean ± s.d. (per-run values) where a",
          "row has repeats. Missing runs are listed at the end.", ""]
    missing = []

    def load(runs, size):
        """The runs that are complete (`size` episodes); an unfinished one is reported, not counted."""
        got = []
        for r in runs:
            eps = episodes(r)
            if len(eps) == size:
                got.append((r, eps))
            else:
                missing.append(f"{r} ({len(eps)} of {size} episodes)" if eps else r)
        return got

    # ---- held-out 20
    md += ["## Held-out layouts (seeds 300-319)", "",
           "| arrangement | runs | right /20 | goal / human / timeout / collision / watchdog | steps to goal | model calls / ep | age (cycles) | tree share | agree with rules | recoveries / ep |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, runs in MAIN:
        got = load(runs, 20)
        if not got:
            continue
        S = [stats(e) for _, e in got]
        md.append(f"| {name} | {len(S)} | {agg([s['right'] for s in S])} | "
                  + " / ".join(agg([s[k] for s in S]) for k in ("goal", "human", "timeout", "collision", "watchdog"))
                  + f" | {agg([s['steps'] for s in S])} | {agg([s['calls'] for s in S], '{:.1f}')} | {agg([s['age'] for s in S], '{:.1f}')} | "
                  f"{agg([100 * s['tree_share'] for s in S], '{:.0f}')}% | {agg([100 * s['agree'] for s in S], '{:.0f}')}% | {agg([s['per_ep'] for s in S], '{:.1f}')} |")
    md.append("")

    # ---- learning 200
    md += ["## Learning layouts (seeds 100-299), 200 each", "",
           "| arm | runs | right /200 | human, needed (of 49) | timeout / watchdog / collision | steps to goal | recoveries | waits | needed | calls / ep (first 25 / last 25) | rules (retired by labels) | frozen held-out, mean over 8 checkpoints | agree with rules | unwanted recoveries |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    arm_eps = {}
    for name, runs in ARMS:
        got = load(runs, 200)
        if not got:
            continue
        arm_eps[name] = got
        S, C, T, H = [], [], [], []
        for run, eps in got:
            S.append(stats(eps)); C.append(replay_labels(run, eps)); T.append(tree_info(run)); H.append(heldout_curve(run))
        needed = [c["needed"] / c["clean"] for c in C if c.get("clean")]
        curves = [st.mean(h) for h in H if len(h) == 8]
        md.append(f"| {name} | {len(S)} | {agg([s['right'] for s in S])} | {agg([s['human'] for s in S])} | "
                  + " / ".join(agg([s[k] for s in S]) for k in ("timeout", "watchdog", "collision"))
                  + f" | {agg([s['steps'] for s in S])} | {agg([s['recoveries'] for s in S])} | "
                  + f"{agg([s['methods'].get('WAIT', 0) for s in S])} | "
                  + (f"{agg([100 * x for x in needed])}%" if needed else "--")
                  + f" | {agg([s['calls_first'] for s in S], '{:.1f}')} / {agg([s['calls_last'] for s in S], '{:.1f}')} | "
                  + (f"{agg([t['rules'] for t in T])} ({agg([t['retired_label'] for t in T])})" if T and T[0] else "--")
                  + f" | {agg(curves, '{:.1f}')} | {agg([100 * s['agree'] for s in S], '{:.0f}')}% | {agg([100 * s['unwanted'] for s in S], '{:.0f}')}% |")
    md.append("")
    md += ["### Frozen held-out per checkpoint (25 ... 200), per run", ""]
    for name, got in arm_eps.items():
        for run, _ in got:
            h = heldout_curve(run)
            if h:
                md.append(f"- {name}, `{run}`: {', '.join(map(str, h))} (mean {st.mean(h):.1f})")
    md.append("")

    # ---- unseen 100
    md += ["## Unseen layouts (seeds 400-499), final tree frozen, no model", "",
           "| arm | runs | right /100 | human, needed (of 28) | timeout / watchdog / collision | steps to goal |", "|---|---|---|---|---|---|"]
    unseen_eps = {}
    for name, runs in UNSEEN:
        got = load(runs, 100)
        if not got:
            continue
        unseen_eps[name] = got
        S = [stats(e) for _, e in got]
        md.append(f"| {name} | {len(S)} | {agg([s['right'] for s in S])} | {agg([s['human'] for s in S])} | "
                  + " / ".join(agg([s[k] for s in S]) for k in ("timeout", "watchdog", "collision")) + f" | {agg([s['steps'] for s in S])} |")
    md.append("")

    # ---- replay labels
    md += ["## Replay labels on the learning streams (the loop's own `--simulate`; the scripted rules replayed after the fact)", "",
           "| arm | runs | recoveries | clean | needed | unnecessary | harmful (outcome / steps) | unclear |", "|---|---|---|---|---|---|---|---|"]
    for name, got in arm_eps.items():
        C = [replay_labels(r, e) for r, e in got]
        C = [c for c in C if c.get("clean")]
        if not C:
            continue
        sh = lambda k: agg([100 * c.get(k, 0) / c["clean"] for c in C])
        md.append(f"| {name} | {len(C)} | {agg([c['recoveries'] for c in C])} | {agg([100 * c['clean'] / c['recoveries'] for c in C])}% | "
                  f"{sh('needed')}% | {sh('unnecessary')}% | {sh('harmful_outcome')}% / {sh('harmful_steps')}% | {sh('unclear')}% |")
        by_method = collections.Counter()
        for c in C:
            for k, v in c.items():
                if ":" in k:
                    by_method[k] += v
        md.append(f"|  by label:method, pooled | | | | " + ", ".join(f"{k} {v}" for k, v in sorted(by_method.items())) + " | | | |")
    md.append("")

    # ---- per staged situation, learning
    md += ["## Learning layouts by staged situation (layouts holding it / ended right), pooled over runs", ""]
    for name, got in arm_eps.items():
        K = collections.defaultdict(lambda: [0, 0])
        for _, eps in got:
            for k, (n, r) in kinds(eps).items():
                K[k][0] += n; K[k][1] += r
        md.append(f"- {name} ({len(got)} runs): " + ", ".join(f"{k} {r}/{n}" for k, (n, r) in sorted(K.items())))
    md.append("")

    # ---- paired comparisons
    md += ["## Paired comparisons on the same layouts (right only under A / only under B, exact McNemar p)", ""]
    def pairs(table, a, b, label):
        if a not in table or b not in table:
            return
        for ra, ea in table[a]:
            for rb, eb in table[b]:
                x, y, p = mcnemar(by_seed(ea), by_seed(eb))
                d, lo, hi, n = steps_diff(by_seed(eb), by_seed(ea))
                md.append(f"- {label}: `{ra}` vs `{rb}`: {x} / {y}, p = {p:.3f}; steps to goal A - B {d:+.0f} (95% {lo:+.0f} to {hi:+.0f}, {n} layouts)")
    pairs(arm_eps, "Experience with replays (method)", "Distillation alone", "learning, method vs distillation")
    pairs(arm_eps, "Experience with replays (method)", "Experience, no replays", "learning, method vs no replays")
    pairs(arm_eps, "Experience with replays (method)", "Scripted rules, no learning", "learning, method vs rules")
    pairs(arm_eps, "Distillation alone", "Scripted rules, no learning", "learning, distillation vs rules")
    pairs(arm_eps, "Experience, no replays", "Distillation alone", "learning, no replays vs distillation")
    pairs(arm_eps, "Experience, no replays", "Scripted rules, no learning", "learning, no replays vs rules")
    pairs(unseen_eps, "Experience with replays (method)", "Distillation alone", "unseen, method vs distillation")
    pairs(unseen_eps, "Experience with replays (method)", "Scripted rules", "unseen, method vs rules")
    pairs(unseen_eps, "Experience with replays (method)", "Experience, no replays", "unseen, method vs no replays")
    pairs(unseen_eps, "Distillation alone", "Scripted rules", "unseen, distillation vs rules")
    pairs(unseen_eps, "Experience, no replays", "Scripted rules", "unseen, no replays vs rules")
    md.append("")

    # ---- the rule the replays retire
    md += ["## Retirements by replay labels (episode, rule)", ""]
    for name, got in arm_eps.items():
        for run, _ in got:
            t = tree_info(run)
            if t.get("label_events"):
                md.append(f"- `{run}`: " + "; ".join(t["label_events"]))
    md += ["", "## The broad `no/no/dynamic -> WAIT` rule in the final trees", ""]
    for name, got in arm_eps.items():
        for run, _ in got:
            p = os.path.join(DATA, f"{run}.json")
            if not os.path.exists(p):
                continue
            for r in json.load(open(p))["rules"]:
                if tuple(r.get("path", [])) == ("no", "no", "dynamic"):
                    md.append(f"- `{run}`: {r.get('action')} fires {r.get('fires')}, judged {r.get('good')} good / {r.get('bad')} bad, retired {r.get('retired')}")
    md.append("")

    # ---- latency and cost
    md += ["## Teacher calls, latency and cost", ""]
    for name, runs in MAIN[2:3] + ARMS[1:]:
        for run in runs:
            a = api(run)
            if a:
                md.append(f"- `{run}`: {a['calls']} calls ({a['failed']} failed attempts), median {a['latency']:.2f} s, "
                          f"${a['usd']:.3f}; prompt {a['prompt']:.0f} tokens per call, {a['cached']:.0f} of them cached")
    md.append("")
    if missing:
        md += ["## Missing (not run yet)", ""] + [f"- `{m}`" for m in missing]
    open(os.path.join(OUT, "paper_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
