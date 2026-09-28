"""What the robot has learned so far, in a form the teacher can be shown.

The decision tree (`tree.py`) is a cache: it answers the ticks it recognises and sends the rest to
the model, and the model answers each of those from the instructions alone, as though it had never
seen this robot before. Everything the arrangement has found out in the meantime -- which rules
exist, how the judge scored them, what a replay says would have happened otherwise -- stays on the
tree's side of the fence. This module carries it across, as two blocks of plain text that go into
the teacher's prompt ahead of the report (`FailureAnalyzer.assess_situation(experience=...)`):

1. **The rules already learned** (`render_rules`). The tree is the record of every decision the
   teacher has made and what became of it, so it doubles as a memory: the rules nearest the current
   report, each with its fires and its good/bad count, retired ones included -- a rule that was
   tried and judged bad twice is the most useful thing in the list.

2. **A table of scored situation-action pairs** (`ExperienceTable`). Keyed on a coarse situation
   (`KEY_FEATURES`, a subset of the tree's own bins) and an action, it counts two kinds of evidence:

   - *real*: the judge's verdict on an action the robot actually took (`TreePolicy` reports every
     verdict to its `listeners`), updated online, while the episode runs;
   - *counterfactual*: the label the same action gets when the episode is replayed in simulation
     with that one decision replaced by CONTINUE -- `needed`, `unnecessary`, `harmful` or
     `unclear` -- added between episodes (`counterfactual.ExperienceReplayer`). This is the half the judge cannot supply: "the robot got going
     afterwards" is also true of a robot that was fine all along, so a needless recovery is judged
     good, and only the replay says it was needless. Simulation is the multiplier here: every
     episode the robot drives is driven again, once per recovery, while it is not driving.

Stdlib-only, like `tree.py` and `recorder.py`: no shapely, no MPC. The table is a JSON file next to
the tree. Nothing here decides anything -- it only describes.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .tree import FEATURE_NAMES, DecisionTree, Leaf, signature

KEY_FEATURES: tuple[str, ...] = ("hold", "stopped", "blocked", "route", "contact", "reversing",
                                 "heading", "blocker", "deadline")
"""The bins a situation is filed under. The tree's own features minus the four that describe the
moment rather than the predicament (`approach` is always unknown here, and `wobble`, `progress`
and `budget` flicker from tick to tick): two stalls at the same gate should land in the
same row even if one report caught the robot creeping and the other caught it still."""

_KEY_INDEX = tuple(FEATURE_NAMES.index(name) for name in KEY_FEATURES)


def situation_key(sig: Iterable[str]) -> tuple[str, ...]:
    sig = tuple(sig)
    return tuple(sig[i] for i in _KEY_INDEX)


def _similarity(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x == y)


CF_LABELS = ("needed", "unnecessary", "harmful", "unclear")


@dataclass
class Entry:
    """Everything known about one action in one situation."""
    good: int = 0                      # judge verdicts on the real robot
    bad: int = 0
    why_bad: Counter = field(default_factory=Counter)
    counterfactual: Counter = field(default_factory=Counter)   # CF_LABELS -> count

    @property
    def replayed(self) -> int:
        return sum(self.counterfactual.values())

    def to_dict(self) -> dict:
        return {"good": self.good, "bad": self.bad, "why_bad": dict(self.why_bad),
                "counterfactual": dict(self.counterfactual)}

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(good=d.get("good", 0), bad=d.get("bad", 0), why_bad=Counter(d.get("why_bad", {})),
                   counterfactual=Counter(d.get("counterfactual", {})))


class ExperienceTable:
    """Scored (situation, action) pairs. See the module docstring."""

    def __init__(self) -> None:
        self.entries: dict[tuple[tuple[str, ...], str], Entry] = {}
        self.verdicts = 0
        self.simulations = 0

    # -- writing ---------------------------------------------------------------------------

    def _entry(self, sig: Iterable[str], action: str) -> Entry:
        return self.entries.setdefault((situation_key(sig), action), Entry())

    def on_verdict(self, sig: tuple[str, ...], action: str, verdict: str, why: str = "") -> None:
        """A judge verdict on something the robot did. Signature of `TreePolicy.listeners`."""
        entry = self._entry(sig, action)
        self.verdicts += 1
        if verdict == "good":
            entry.good += 1
        else:
            entry.bad += 1
            if why:
                entry.why_bad[why[:80]] += 1

    def record_counterfactual(self, sig: Iterable[str], action: str, label: str) -> None:
        """The label a counterfactual replay gave `action` in this situation (one of `CF_LABELS`)."""
        if label not in CF_LABELS:
            raise ValueError(f"counterfactual label must be one of {CF_LABELS}, got {label!r}")
        self._entry(sig, action).counterfactual[label] += 1
        self.simulations += 1

    # -- reading ---------------------------------------------------------------------------

    def _grouped(self) -> dict[tuple[str, ...], dict[str, Entry]]:
        grouped: dict[tuple[str, ...], dict[str, Entry]] = {}
        for (key, action), entry in self.entries.items():
            grouped.setdefault(key, {})[action] = entry
        return grouped

    def feature_weights(self) -> tuple[float, ...]:
        """How much each of `KEY_FEATURES` has mattered so far, from 1 (not at all) to 5: its
        information gain on the action that did best in each stored situation.

        Retrieval by plain feature count treats `blocker=parked` vs `abandoned` like `heading=on` vs
        `off`. The first decides between WAIT and REQUEST_HUMAN and the second decides nothing, and
        plain counting would show a robot at a 31 s breakdown two WAIT situations (one feature off:
        the blocker) ahead of the REQUEST_HUMAN one (one feature off: the heading). The table
        already knows which features separate its own answers, so it is asked.
        """
        rows = []
        for key, actions in self._grouped().items():
            score = {a: e.good + e.counterfactual["needed"] - e.bad - e.counterfactual["harmful"]
                        - 0.5 * e.counterfactual["unnecessary"] for a, e in actions.items()}
            best = max(score, key=score.get)
            rows.append((key, best, sum(e.good + e.bad + e.replayed for e in actions.values()) or 1))
        total = sum(n for _, _, n in rows)

        def entropy(counts: Counter) -> float:
            n = sum(counts.values())
            return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0

        overall = Counter()
        for _, best, n in rows:
            overall[best] += n
        base = entropy(overall)
        if not rows or base == 0.0:
            return tuple(1.0 for _ in KEY_FEATURES)
        weights = []
        for i in range(len(KEY_FEATURES)):
            by_value: dict[str, Counter] = {}
            for key, best, n in rows:
                by_value.setdefault(key[i], Counter())[best] += n
            conditional = sum(sum(c.values()) / total * entropy(c) for c in by_value.values())
            weights.append(1.0 + 4.0 * max(0.0, base - conditional) / base)
        return tuple(weights)

    def situations_like(self, sig: Iterable[str], max_situations: int = 4,
                        min_similarity: int | None = None) -> list[tuple[tuple[str, ...], int, dict[str, Entry]]]:
        """The stored situations nearest `sig`: `(key, features in common, {action: Entry})`,
        nearest first. `hold` has to agree -- a robot being held and a robot that is not are
        different questions however alike the rest of the report is."""
        here = situation_key(sig)
        weights = self.feature_weights()
        floor = 0.6 if min_similarity is None else min_similarity
        ranked = []
        for key, actions in self._grouped().items():
            if key[0] != here[0]:
                continue
            share = sum(w for w, a, b in zip(weights, key, here) if a == b) / sum(weights)
            if share >= floor:
                evidence = sum(e.good + e.bad + e.replayed for e in actions.values())
                ranked.append((round(share, 3), evidence, key, actions))
        ranked.sort(key=lambda r: (-r[0], -r[1]))
        return [(key, _similarity(key, here), actions) for _, _, key, actions in ranked[:max_situations]]

    def render(self, sig: Iterable[str], max_situations: int = 4) -> str:
        """The table as the prompt shows it, nearest situation first. Empty string if nothing
        stored resembles the report."""
        here = situation_key(sig)
        weights = self.feature_weights()
        matters = sorted(weights)[len(weights) // 2]      # above the median weight: has separated answers before
        blocks = []
        for n, (key, same, actions) in enumerate(self.situations_like(sig, max_situations), 1):
            differs = [f"{name}={value} (now {now})" + (" -- a feature that has changed the right answer before"
                                                         if w > matters and w > 1.5 else "")
                       for name, value, now, w in zip(KEY_FEATURES, key, here, weights) if value != now]
            head = (f"{n}. " + ", ".join(f"{name}={value}" for name, value in zip(KEY_FEATURES, key)
                                         if value not in ("?",)))
            head += ("  [same as the current report]" if same == len(KEY_FEATURES)
                     else f"  [differs from the current report in: {'; '.join(differs)}]")
            lines = [head]
            for action, e in sorted(actions.items(), key=lambda kv: -(kv[1].good + kv[1].bad + kv[1].replayed)):
                parts = []
                if e.good or e.bad:
                    judged = f"done on the robot {e.good + e.bad}x: judged {e.good} good / {e.bad} bad"
                    if e.bad and e.why_bad:
                        judged += f" ({e.why_bad.most_common(1)[0][0]})"
                    parts.append(judged)
                if e.replayed:
                    parts.append(f"replayed without it {e.replayed}x: " + ", ".join(
                        f"{e.counterfactual[label]} {label}" for label in CF_LABELS if e.counterfactual[label]))
                lines.append(f"   - {action}: " + "; ".join(parts))
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    # -- persistence -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"key_features": list(KEY_FEATURES), "verdicts": self.verdicts, "simulations": self.simulations,
                "entries": [{"situation": list(key), "action": action, **entry.to_dict()}
                            for (key, action), entry in sorted(self.entries.items())]}

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def open(cls, path: str, reset: bool = False) -> "ExperienceTable":
        table = cls()
        if reset or not path or not os.path.exists(path):
            return table
        data = json.load(open(path))
        if tuple(data.get("key_features", ())) != KEY_FEATURES:
            raise ValueError(f"{path} was written with different KEY_FEATURES; start a new table.")
        table.verdicts, table.simulations = data.get("verdicts", 0), data.get("simulations", 0)
        for row in data.get("entries", []):
            table.entries[(tuple(row["situation"]), row["action"])] = Entry.from_dict(row)
        return table


def _conditions(leaf: Leaf) -> str:
    return ", ".join(f"{name}={value}" for name, value in zip(FEATURE_NAMES, leaf.path)) or "(any report)"


def _rule_line(leaf: Leaf) -> str:
    verdict = leaf.mode.value.upper() if leaf.mode is not None else "NONE"
    stats = f"used {leaf.fires}x"
    if leaf.good or leaf.bad:
        stats += f", judged {leaf.good} good / {leaf.bad} bad"
    if leaf.retired:
        stats += "; RETIRED -- it kept being judged bad, so do not simply repeat it"
    elif leaf.conflicts:
        stats += f"; you have answered this situation {leaf.conflicts + 1} different ways"
    return f"[{_conditions(leaf)}] -> FAILURE {verdict}, ACTION {leaf.action.value}  ({stats})"


def render_rules(tree: DecisionTree, sig: tuple[str, ...], max_rules: int = 8,
                 weights: dict[str, float] | None = None) -> str:
    """The learned rules nearest `sig`, as the prompt shows them.

    Nearest means: agrees with the report on `hold`, then on as large a share of its own conditions
    as possible, then most used. The rule that actually covers the report (if one does -- the
    teacher is also asked to confirm an unconfirmed rule, and to replace a retired one) is named
    first and separately."""
    weights = weights or {}
    w = [weights.get(name, 1.0) for name in FEATURE_NAMES]
    covering = tree.match(sig)
    ranked = []
    for leaf in tree.leaves.values():
        if leaf is covering or not leaf.path or leaf.path[0] != sig[0]:
            continue
        share = (sum(wi for wi, a, b in zip(w, leaf.path, sig) if a == b) / sum(w[:len(leaf.path)]))
        ranked.append((round(share, 3), leaf.retired, leaf.fires, leaf))
    ranked.sort(key=lambda r: (-r[0], -r[1], -r[2]))
    lines, seen = [], set()
    if covering is not None:
        lines.append("The rule that covers this report: " + _rule_line(covering))
    for _, _, _, leaf in ranked:
        # One line per distinct lesson: rules that give the same answer and agree on every feature
        # that has mattered are the same lesson learned at different moments.
        lesson = (leaf.mode, leaf.action, leaf.retired,
                  tuple(v for v, wi in zip(leaf.path, w) if wi > 1.5))
        if lesson in seen:
            continue
        seen.add(lesson)
        lines.append("- " + _rule_line(leaf))
        if len(lines) >= max_rules + (covering is not None):
            break
    return "\n".join(lines)


MEMORY_MODES = ("off", "tree", "table", "both")


class ExperienceMemory:
    """Builds the `experience` text for one report. `TreeAnalyzer` holds one and asks it on every
    escalation; with `mode="off"` it is never constructed and the prompt is what it always was."""

    def __init__(self, tree: DecisionTree, table: ExperienceTable | None, mode: str = "both",
                 max_rules: int = 8, max_situations: int = 4):
        if mode not in MEMORY_MODES or mode == "off":
            raise ValueError(f"memory mode must be one of {MEMORY_MODES[1:]}, got {mode!r}")
        self.tree, self.table, self.mode = tree, table, mode
        self.max_rules, self.max_situations = max_rules, max_situations

    def for_report(self, context: Any) -> str:
        sig = signature(context)
        parts = []
        if self.mode in ("tree", "both"):
            weights = (dict(zip(KEY_FEATURES, self.table.feature_weights())) if self.table is not None else None)
            rules = render_rules(self.tree, sig, self.max_rules, weights)
            parts.append("RULES YOU HAVE ALREADY TAUGHT (nearest to this report first):\n"
                         + (rules or "(none yet)"))
        if self.mode in ("table", "both") and self.table is not None:
            table = self.table.render(sig, self.max_situations)
            parts.append("WHAT HAPPENED IN SIMILAR SITUATIONS (most similar first):\n"
                         + (table or "(nothing similar on record yet)"))
        return "\n\n".join(parts)
