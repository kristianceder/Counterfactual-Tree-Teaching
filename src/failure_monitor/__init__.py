"""Agentic failure recovery for an MPC-driven robot: a decision tree distilled from a hosted LLM
teacher decides, on a fixed clock, whether the robot has failed and which recovery to execute.

    report.py          the status report and the failure-mode evidence
    llm.py             the teacher's prompt and answer parsing; openai_teacher.py the model call
    tree.py            the decision tree, its judge and its retirement rules
    tree_analyzer.py   tree first, teacher on a miss
    experience.py      what the teacher is shown of what has been learned
    recovery.py        the recovery actions, executed against the live MPC
    scripted.py        the scripted rules (baseline, reference and replay fallback)
    episode.py         one closed-loop episode on the scenario grid
    counterfactual.py  replays that label each recovery needed / unnecessary / harmful
    replay.py, recorder.py, async_llm.py, monitor.py, types.py   supporting pieces

Importing the package does not need the compiled MPC solver; `episode.run_episode` does.
"""
from .types import FailureMode, FailureEvent, RobotSnapshot
from .monitor import VerdictLog
from .report import (SituationContext, SituationBuilder, FAILURE_MODE_EVIDENCE, monitor_evidence,
                     MOTION_WINDOW_S)
from .recovery import (RecoveryMethod, RecoveryOutcome, RecoveryContext, RecoveryStrategy, DEFAULT_STRATEGIES,
                       RECOVERY_METHOD_DESCRIPTIONS, recovery_candidates, RecoveryManager)
from .llm import LLMConfig, FailureAnalyzer, SituationAssessment
from .async_llm import AsyncFailureAnalyzer, LLMRequest, LLMReply
from .tree import (DecisionTree, Leaf, TreePolicy, TreeStats, Feature, FEATURES, FEATURE_NAMES, signature,
                   evidence_of_failure, evidence_depth, meets_test, MODE_EVIDENCE_FEATURES, CHECKABLE_MODES)
from .tree_analyzer import TreeAnalyzer
from .scripted import ScriptedTeacher, ReferenceShadow

__all__ = [
    'FailureMode', 'FailureEvent', 'RobotSnapshot', 'VerdictLog',
    'SituationContext', 'SituationBuilder', 'FAILURE_MODE_EVIDENCE', 'monitor_evidence', 'MOTION_WINDOW_S',
    'RecoveryMethod', 'RecoveryOutcome', 'RecoveryContext', 'RecoveryStrategy', 'DEFAULT_STRATEGIES',
    'RECOVERY_METHOD_DESCRIPTIONS', 'recovery_candidates', 'RecoveryManager',
    'LLMConfig', 'FailureAnalyzer', 'SituationAssessment',
    'AsyncFailureAnalyzer', 'LLMRequest', 'LLMReply',
    'DecisionTree', 'Leaf', 'TreePolicy', 'TreeStats', 'TreeAnalyzer', 'Feature', 'FEATURES', 'FEATURE_NAMES',
    'signature', 'evidence_of_failure', 'evidence_depth', 'meets_test', 'MODE_EVIDENCE_FEATURES',
    'CHECKABLE_MODES', 'ScriptedTeacher', 'ReferenceShadow',
]
