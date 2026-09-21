"""Benchmark scoring and the run harness (ARCHITECTURE.md section 3).

This is the top layer. It reads the dataset through ``src/benchmark/schema.py``
and drives a pipeline through its public ``ask``; nothing below it imports
anything from here, which is what keeps the harness reusable against the hybrid
and agentic systems of Phases 4 and 5 without either of them knowing it exists.
"""

from .benchmark import (
    BenchmarkRun,
    QuestionRun,
    QueryableSystem,
    load_results,
    run_benchmark,
    run_benchmark_cli,
    score_result,
    write_results,
)
from .judge import JUDGE_PROTOCOL_VERSION, JudgeVerdict, LLMJudge
from .metrics import (
    ERROR_CATEGORIES,
    ERROR_TAXONOMY_VERSION,
    SCORING_RULES_VERSION,
    QuestionScore,
    classify_error,
    score_answer,
    score_citations,
    score_retrieval,
    summarize,
)

__all__ = [
    "BenchmarkRun",
    "ERROR_CATEGORIES",
    "ERROR_TAXONOMY_VERSION",
    "JUDGE_PROTOCOL_VERSION",
    "JudgeVerdict",
    "LLMJudge",
    "QuestionRun",
    "QuestionScore",
    "QueryableSystem",
    "SCORING_RULES_VERSION",
    "classify_error",
    "load_results",
    "run_benchmark",
    "run_benchmark_cli",
    "score_answer",
    "score_citations",
    "score_result",
    "score_retrieval",
    "summarize",
    "write_results",
]
