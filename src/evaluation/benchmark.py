"""The benchmark run harness (EVALUATION_PROTOCOL.md sections 6, 23, 27).

Benchmark mode, as section 6 defines it: no manual intervention, latency
measured, metrics calculated, results stored. The harness indexes each document
once and asks every question against it, because re-parsing a 42-page PDF per
question would make the run dominated by the one cost that has nothing to do
with what is being measured.

Three properties this file is built around:

* **It is system-agnostic.** It drives anything satisfying :class:`QueryableSystem`
  -- an ``ask`` returning something shaped like a ``RAGResult``. The system is
  whatever ``build_pipeline(config)`` builds, and the stages a system ran are
  read from its own result metadata, so nothing here names a particular
  system. Phase 4 found two places Phase 3 had assumed the dense baseline (the
  default factory, and taxonomy stages that no caller ever set); both now read
  the system instead. Nothing below the evaluation layer imports this module.
* **It does not change answer behaviour** (ARCHITECTURE.md section 26). It calls
  ``ask`` exactly as a user would and measures what comes back. The retrieval
  probe used for Recall@10 is a separate read-only call.
* **It checkpoints as it goes.** EXPERIMENT_PLAN.md section 4 lists an expiring
  Colab session as a likely risk with "checkpoint per-question results as they
  are produced, not at the end" as its mitigation, so the per-question file is
  rewritten after every question rather than once at the end.

Outputs are the four files EVALUATION_PROTOCOL.md section 27 names, in the
directory it names them in.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from ..benchmark.schema import (
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_QUESTIONS_PATH,
    Question,
    load_questions,
)
from ..benchmark.validate import validate_benchmark
from ..config import RAGConfig
from ..errors import BenchmarkValidationError, RAGError
from ..generation.abstention import ABSTENTION_PATTERNS_VERSION, is_abstention
from ..pipeline import RAGResult, build_pipeline
from . import statistics
from .metrics import (
    ERROR_TAXONOMY_VERSION,
    RECALL_KS,
    SCORING_RULES_VERSION,
    STAGE_CONTEXT_SELECTION,
    STAGE_GENERATION,
    STAGE_RETRIEVAL,
    VERIFICATION_NOT_RUN,
    AbstentionScore,
    AnswerScore,
    CitationScore,
    QuestionScore,
    RetrievalScore,
    classify_error,
    is_failure,
    score_answer,
    score_citations,
    score_retrieval,
    summarize,
)

CONFIG_FILENAME = "config.json"
RESULTS_FILENAME = "results.json"
PER_QUESTION_FILENAME = "per_question.json"
SUMMARY_FILENAME = "summary.csv"

#: EVALUATION_PROTOCOL.md section 11. Recorded in every run, and set before the
#: run rather than assumed.
SEED = 42

#: EVALUATION_PROTOCOL.md section 4. Bump when questions, answers, evidence
#: pages or documents change; results from two versions are not comparable.
BENCHMARK_VERSION = "1.0"

#: Depth the retrieval probe ranks to. BENCHMARK_SPEC.md section 10 asks for
#: Recall@10 and MRR@10; the baseline generates from top_k=5.
RETRIEVAL_METRIC_DEPTH = max(RECALL_KS)


@runtime_checkable
class QueryableSystem(Protocol):
    """What the harness needs from a system under test.

    Deliberately small. ``retrieve`` is optional: a system without it is scored
    on the chunks its ``ask`` returned, with the shallower depth recorded rather
    than hidden.
    """

    name: str

    def index(self, pdf_path: Any) -> Any: ...

    def ask(self, question: str, top_k: int | None = None) -> RAGResult: ...


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionRun:
    """One question's result and its score, before either is written out."""

    question: Question
    score: QuestionScore
    record: dict[str, Any]


def _failed_run(question: Question, system: str, exc: BaseException, latency: float) -> QuestionRun:
    """Record a question whose run raised, without aborting the benchmark.

    ARCHITECTURE.md section 24 wants explicit failures; EVALUATION_PROTOCOL.md
    section 23 has a category for them. A harness that dies on question 30 of 76
    produces no result at all, which is strictly worse than a result with one
    question marked as a runtime failure.
    """
    answerable = not question.is_unanswerable
    score = QuestionScore(
        question_id=question.id,
        question_type=question.question_type,
        split=question.split,
        answerable=answerable,
        retrieval=RetrievalScore(scored=False, excluded_reason="run_failed"),
        answer=AnswerScore(
            correct=0.0,
            overlap=0.0,
            numeric_agreement=False,
            faithfulness_numeric=None,
            faithfulness_token=0.0,
            unsupported=False,
        ),
        citation=CitationScore(scored=False),
        abstention=AbstentionScore(answerable=answerable, abstained=False),
        verification_status=VERIFICATION_NOT_RUN,
        error=f"{type(exc).__name__}: {exc}",
    )
    score = _finalize(score, stages=(), evidence_pages=())
    record = {
        "question_id": question.id,
        "system": system,
        "reference_answer": question.answer,
        "question": question.question,
        "answer": "",
        "retrieved_chunks": [],
        "citations": [],
        "evidence_pages": [],
        "verification_status": VERIFICATION_NOT_RUN,
        "latency": round(latency, 4),
        "metrics": score.to_record(),
    }
    return QuestionRun(question=question, score=score, record=record)


def _finalize(
    score: QuestionScore,
    stages: Sequence[str],
    evidence_pages: Sequence[int],
    retrieved_pages: Sequence[int] | None = None,
    reranking_lost_gold: bool = False,
    verification_lost_answer: bool = False,
) -> QuestionScore:
    """Attach the failure verdict and its single category."""
    failed = is_failure(score)
    category = (
        classify_error(
            score,
            stages=stages,
            reranking_lost_gold=reranking_lost_gold,
            evidence_pages=evidence_pages,
            retrieved_pages=retrieved_pages,
            verification_lost_answer=verification_lost_answer,
        )
        if failed
        else None
    )
    return QuestionScore(
        question_id=score.question_id,
        question_type=score.question_type,
        split=score.split,
        answerable=score.answerable,
        retrieval=score.retrieval,
        answer=score.answer,
        citation=score.citation,
        abstention=score.abstention,
        verification_status=score.verification_status,
        error=score.error,
        error_category=category,
        failed=failed,
    )


#: The stages a system is taken to have run when it declares none: retrieve,
#: assemble context, generate. Baseline A's shape.
DEFAULT_STAGES: tuple[str, ...] = (
    STAGE_RETRIEVAL,
    STAGE_CONTEXT_SELECTION,
    STAGE_GENERATION,
)


def score_result(
    question: Question,
    result: RAGResult,
    system: str,
    retrieved_for_metrics: Sequence[Any] | None = None,
    stages: Sequence[str] | None = None,
    verification_status: str | None = None,
    latency_s: float | None = None,
) -> QuestionRun:
    """Score one answered question and build its section 20 record.

    Abstention is read from the answer text by ``is_abstention`` rather than
    from ``result.abstained``. BENCHMARK_SPEC.md section 13.2 requires exactly
    that: detecting abstention from the system's own flag would let a system be
    graded on its self-report, and the two are cross-checked in the record so a
    disagreement is visible rather than silently resolved in the system's
    favour.

    The system under test declares its stages in ``result.metadata["stages"]``
    and, when it reranks, the pages of the top-k it handed the reranker in
    ``metadata["pre_rerank_pages"]``. Both are read from the result rather
    than from knowledge of any particular system, which is what keeps this
    function system-agnostic (DD-038, DD-044).

    The same holds for verification (DD-054): the status is the system's own
    ``metadata["verification_status"]`` -- ``not_run`` for a system that
    reports none -- and a system that verifies reports the draft it verified
    as ``metadata["draft_answer"]``, which is scored here with the same rules
    as the answer to tell a verifier that lost a correct answer from one that
    caught a wrong one. Phase 3 hard-wired ``not_run``, which no verifying
    system could have survived.
    """
    if stages is None:
        stages = tuple(result.metadata.get("stages") or DEFAULT_STAGES)
    if verification_status is None:
        verification_status = str(
            result.metadata.get("verification_status") or VERIFICATION_NOT_RUN
        )
    abstained = is_abstention(result.answer)
    ranked = list(retrieved_for_metrics if retrieved_for_metrics is not None else result.retrieved)

    retrieval = score_retrieval(question, ranked)
    answer = score_answer(
        question,
        answer=result.answer,
        abstained=abstained,
        evidence_text=result.evidence_text,
        has_citations=bool(result.citations),
    )
    citation = score_citations(
        question,
        answer=result.answer,
        cited_pages=result.cited_pages,
        resolved=len(result.citations),
        dropped_markers=len(result.metadata.get("dropped_markers") or ()),
        fabricated_page_mentions=len(
            result.metadata.get("fabricated_page_mentions") or ()
        ),
    )
    score = QuestionScore(
        question_id=question.id,
        question_type=question.question_type,
        split=question.split,
        answerable=not question.is_unanswerable,
        retrieval=retrieval,
        answer=answer,
        citation=citation,
        abstention=AbstentionScore(
            answerable=not question.is_unanswerable, abstained=abstained
        ),
        verification_status=verification_status,
    )
    # Blame follows the answer path, not the deeper probe used for Recall@10.
    answer_path_pages = sorted({p for r in result.retrieved for p in r.pages})
    score = _finalize(
        score,
        stages=stages,
        evidence_pages=result.evidence_pages,
        retrieved_pages=answer_path_pages,
        reranking_lost_gold=reranking_lost_gold(
            question, answer_path_pages, result.metadata.get("pre_rerank_pages")
        ),
        verification_lost_answer=verification_lost_answer(question, result, answer),
    )

    metrics = score.to_record()
    metrics["self_reported_abstention"] = result.abstained
    metrics["abstention_agrees_with_self_report"] = result.abstained == abstained
    metrics["retrieval_metric_depth"] = len(ranked)

    record = result.to_record(
        question_id=question.id,
        system=system,
        reference_answer=question.answer,
        metrics=metrics,
        verification_status=verification_status,
        latency_s=latency_s,
    )
    return QuestionRun(question=question, score=score, record=record)


def reranking_lost_gold(
    question: Question,
    answer_path_pages: Sequence[int],
    pre_rerank_pages: Sequence[int] | None,
) -> bool:
    """Did the reranker push every gold page off the answer path? (DD-044)

    True when a gold page was in the top-k the reranker was handed -- what the
    generator would have seen with the reranker off -- and no gold page is in
    the top-k it returned. ``None`` means the system reported no pre-rerank
    ranking, so nothing can be blamed on a reranker.
    """
    if pre_rerank_pages is None:
        return False
    gold = set(question.evidence_pages)
    return bool(gold & set(pre_rerank_pages)) and not (gold & set(answer_path_pages))


def verification_lost_answer(
    question: Question, result: RAGResult, final: AnswerScore
) -> bool:
    """Did verification turn a correct draft into a wrong answer? (DD-054)

    ``metadata["draft_answer"]`` is the answer a verifying system generated
    before verification acted on it -- what "Verification OFF" returns. It is
    scored with the same rule as the final answer; True when the draft is
    correct and the final answer is not. A system that reports no draft cannot
    have lost one.
    """
    draft = result.metadata.get("draft_answer")
    if not isinstance(draft, str) or draft == result.answer:
        return False
    draft_score = score_answer(
        question,
        answer=draft,
        abstained=is_abstention(draft),
        evidence_text=result.evidence_text,
        has_citations=True,
    )
    return draft_score.correct >= 1.0 and final.correct < 1.0


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRun:
    """Everything one benchmark produced."""

    system: str
    runs: list[QuestionRun] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    index_stats: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    started: float = 0.0
    judge: Any = None
    #: Records carried over from an interrupted run rather than answered now.
    resumed: int = 0

    @property
    def scores(self) -> list[QuestionScore]:
        return [run.score for run in self.runs]

    @property
    def records(self) -> list[dict[str, Any]]:
        return [run.record for run in self.runs]

    def summary(self) -> dict[str, Any]:
        result = summarize(self.scores)
        result["system"] = self.system
        result.update(latency_stats(self.latencies))
        result["index_seconds_total"] = round(
            sum(float(s.get("parse_s", 0)) + float(s.get("chunk_s", 0))
                + float(s.get("embed_index_s", 0)) for s in self.index_stats),
            3,
        )
        result["documents_indexed"] = len(self.index_stats)
        if self.resumed:
            # The index figures above cover this session only; the carried-over
            # records were answered by an earlier one (DD-064).
            result["resumed_records"] = self.resumed
        # EVALUATION_PROTOCOL.md section 28's per-system costs, read from what
        # each record reports -- not an agentic special case.
        result.update(run_cost_stats(self.records))
        # BENCHMARK_SPEC.md section 15 asks for peak GPU memory. Nothing here
        # measures it when no CUDA device is present, and reporting 0.0 would
        # claim a measurement that was not taken.
        result["peak_vram_gb"] = _peak_vram_gb()
        # EVALUATION_PROTOCOL.md 19.1: a judged figure never appears without
        # the disagreement rate that says how far to trust it.
        result.update(
            self.judge.summary() if self.judge is not None else {"judge_enabled": False}
        )
        return result


def run_cost_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval iterations, model calls and parse failures, from the records.

    Each figure is over the records that report it, with that count beside it,
    and None when none do: Baselines A and B report no iterations or model
    calls, and a 0 for them would be a measurement nobody took. Verification
    statuses are counted for every system, since every record has one.
    """

    def reported(key: str) -> list[float]:
        return [
            float(r[key])
            for r in records
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
        ]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    iterations = reported("retrieval_iterations")
    calls = reported("model_calls")
    decisions = sum(reported("llm_decisions"))
    failures = sum(reported("llm_parse_failures"))
    refined = [r for r in records if isinstance(r.get("refinement_changed_answer"), bool)]
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("verification_status") or VERIFICATION_NOT_RUN)
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "retrieval_iterations_mean": mean(iterations),
        "retrieval_iterations_max": max(iterations) if iterations else None,
        "n_reporting_retrieval_iterations": len(iterations),
        "model_calls_mean": mean(calls),
        "model_calls_max": max(calls) if calls else None,
        "n_reporting_model_calls": len(calls),
        "diagnostic_model_calls_total": sum(reported("diagnostic_model_calls")) if calls else None,
        "llm_decisions_total": int(decisions) if calls else None,
        "llm_parse_failures_total": int(failures) if calls else None,
        "llm_parse_failure_rate": round(failures / decisions, 4) if decisions else None,
        "n_refined": len(refined) if calls else None,
        "n_refinement_changed_answer": (
            sum(1 for r in refined if r["refinement_changed_answer"]) if calls else None
        ),
        "verification_status_counts": dict(sorted(statuses.items())),
    }


def latency_stats(latencies: Sequence[float]) -> dict[str, Any]:
    """Mean, median and P95 (EVALUATION_PROTOCOL.md section 15).

    "Avoid relying only on average latency" -- so the average is never the only
    figure here.
    """
    if not latencies:
        return {
            "latency_mean_s": None,
            "latency_median_s": None,
            "latency_p95_s": None,
            "latency_max_s": None,
        }
    ordered = sorted(latencies)
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    # Nearest-rank P95: with 76 questions an interpolated percentile invents a
    # latency no query had.
    p95_index = min(count - 1, max(0, math.ceil(0.95 * count) - 1))
    return {
        "latency_mean_s": round(sum(ordered) / count, 4),
        "latency_median_s": round(median, 4),
        "latency_p95_s": round(ordered[p95_index], 4),
        "latency_max_s": round(ordered[-1], 4),
    }


def _peak_vram_gb() -> float | None:
    """Peak CUDA memory, or None when there is no GPU to measure."""
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    except Exception:
        return None


def _seed_everything(seed: int = SEED) -> None:
    """EVALUATION_PROTOCOL.md section 11, applied rather than documented."""
    import random  # noqa: PLC0415

    random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import numpy  # noqa: PLC0415

        numpy.random.seed(seed)
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
    except Exception:
        pass


def build_config_snapshot(
    config: RAGConfig,
    system: str,
    questions_path: Path,
    documents_dir: Path,
    n_questions: int,
    split: str | None,
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The reproducibility record (EVALUATION_PROTOCOL.md sections 3, 4, 11, 13).

    "No benchmark result should exist without a corresponding configuration."
    Fields the dense baseline has no value for are written as ``null`` rather
    than omitted, because a reader comparing this against a Phase 4 run needs to
    see that the reranker was absent, not that the key was forgotten.
    """
    described = config.describe()
    return {
        "system": system,
        "benchmark_version": BENCHMARK_VERSION,
        "questions_path": str(questions_path),
        "documents_dir": str(documents_dir),
        "n_questions": n_questions,
        "split": split,
        "random_seed": SEED,
        "scoring_rules_version": SCORING_RULES_VERSION,
        "error_taxonomy_version": ERROR_TAXONOMY_VERSION,
        "abstention_patterns_version": ABSTENTION_PATTERNS_VERSION,
        # Section 3's minimum configuration, in its own order.
        "generation_model": config.generation.model_id,
        "generation_model_revision": None,
        "quantization": config.generation.quantization,
        "embedding_model": config.embedding.model_id,
        "embedding_model_revision": None,
        "reranker_model": described["reranker_model"],
        "reranker_enabled": described["reranker_enabled"],
        "chunking_strategy": config.chunking.strategy,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "retrieval_strategy": config.retrieval.strategy,
        "retrievers": described["retrievers"],
        "fusion": described["fusion"],
        "rrf_k": described["rrf_k"],
        "candidate_k": described["candidate_k"],
        "top_k": config.retrieval.top_k,
        # The depth the reranker ranks, or -- with it off -- the depth of the
        # fused pool passed through unchanged (DD-046).
        "rerank_k": described["rerank_candidates"],
        "retrieval_metric_depth": RETRIEVAL_METRIC_DEPTH,
        # The cap in force; null for a baseline, which has no loop (DD-050).
        "max_retrieval_iterations": described["max_retrieval_iterations"],
        "planner_enabled": described["planner_enabled"],
        "planner": described["planner"],
        "evidence_controller_enabled": described["evidence_controller_enabled"],
        "evidence_controller": described["evidence_controller"],
        "sufficiency_threshold": described["sufficiency_threshold"],
        "refinement_enabled": described["refinement_enabled"],
        "verification_enabled": described["verification_enabled"],
        "verifier": described["verifier"],
        "temperature": config.generation.temperature,
        "max_new_tokens": config.generation.max_new_tokens,
        "judge": judge or {"enabled": False, "reason": "not requested"},
        "config_fingerprint": config.fingerprint(),
        "config": described,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def run_benchmark(
    questions: Sequence[Question],
    documents_dir: str | Path,
    system_factory=None,
    config: RAGConfig | None = None,
    out_dir: str | Path | None = None,
    judge: Any = None,
    progress=None,
    checkpoint_dir: str | Path | None = None,
    completed: Sequence[dict[str, Any]] = (),
) -> BenchmarkRun:
    """Answer every question and score it.

    Questions are grouped by document so each PDF is parsed, chunked and indexed
    once. That is not only for speed: re-indexing per question would make the
    reported index time meaningless and would let a per-question index
    difference leak into the answers.

    ``completed`` holds records from an interrupted run (DD-064). Their
    questions are not asked again: each record is kept, in the position its
    question would have taken, and a document whose questions are all done is
    not indexed. ``checkpoint_dir`` receives ``per_question.json`` after every
    question, without writing the other three files.
    """
    config = (config or RAGConfig.default()).validate()
    documents_dir = Path(documents_dir)
    _seed_everything()

    factory = system_factory or (lambda: build_pipeline(config))
    probe_system = factory()
    run = BenchmarkRun(
        system=getattr(probe_system, "name", "system"),
        started=time.time(),
        judge=judge,
    )

    def _checkpoint(question_run: QuestionRun) -> None:
        """Persist what is finished before attempting the next question."""
        target = checkpoint_dir if checkpoint_dir is not None else out_dir
        if target is not None:
            write_checkpoint(target, run.records)
        if progress:
            progress(question_run)

    done = {record["question_id"]: record for record in completed}
    by_document: dict[str, list[Question]] = {}
    for question in questions:
        by_document.setdefault(question.document, []).append(question)

    for document, group in by_document.items():
        if all(question.id in done for question in group):
            for question in group:
                _keep(run, question, done[question.id])
            continue
        pdf_path = documents_dir / document
        try:
            system = factory()
            system.index(pdf_path)
            run.index_stats.append(dict(getattr(system, "index_stats", {})))
        except (RAGError, OSError) as exc:
            # One unreadable document must not cost the other 60 questions.
            for question in group:
                run.runs.append(_failed_run(question, run.system, exc, 0.0))
                _checkpoint(run.runs[-1])
            continue

        for question in group:
            if question.id in done:
                _keep(run, question, done[question.id])
                continue
            started = time.perf_counter()
            try:
                result = system.ask(question.question)
                latency = time.perf_counter() - started
                ranked = _probe(system, question.question, result)
                question_run = score_result(
                    question,
                    result,
                    system=run.system,
                    retrieved_for_metrics=ranked,
                    latency_s=latency,
                )
                if judge is not None:
                    judge.annotate(question, result, question_run.record)
                run.latencies.append(latency)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                question_run = _failed_run(
                    question, run.system, exc, time.perf_counter() - started
                )
            run.runs.append(question_run)
            _checkpoint(question_run)

    run.config = build_config_snapshot(
        config,
        system=run.system,
        questions_path=Path(DEFAULT_QUESTIONS_PATH),
        documents_dir=documents_dir,
        n_questions=len(questions),
        split=None,
        judge=judge.describe() if judge is not None else None,
    )
    if out_dir is not None:
        write_results(out_dir, run)
    return run


def _keep(run: BenchmarkRun, question: Question, record: dict[str, Any]) -> None:
    """Carry a record from an interrupted run into this one, unchanged."""
    run.runs.append(
        QuestionRun(question=question, score=QuestionScore.from_record(record), record=record)
    )
    if record.get("latency") is not None:
        run.latencies.append(float(record["latency"]))
    run.resumed += 1


def load_completed(
    out_dir: str | Path, config: RAGConfig, questions: Sequence[Question]
) -> list[dict[str, Any]]:
    """The records an interrupted run left in ``out_dir``, checked (DD-064).

    Refuses, rather than mixing two experiments in one file, when a stored
    record was produced by a different configuration or answers a question
    outside the current selection. A record whose run raised is dropped, so the
    question is asked again: a crash is usually the session that died, not the
    question.
    """
    path = Path(out_dir) / PER_QUESTION_FILENAME
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = {question.id for question in questions}
    fingerprint = config.fingerprint()
    kept: list[dict[str, Any]] = []
    for record in records:
        if record.get("metrics", {}).get("error_category") == "system-runtime":
            continue
        stored = record.get("config_fingerprint")
        if stored != fingerprint:
            raise BenchmarkValidationError(
                f"Refusing to resume {path}: question {record['question_id']} was "
                f"answered with config fingerprint {stored}, but this run's is "
                f"{fingerprint}. Resume with the same flags as the interrupted "
                "run, or write to a new --out directory."
            )
        if record["question_id"] not in selected:
            raise BenchmarkValidationError(
                f"Refusing to resume {path}: question {record['question_id']} is "
                "not in this run's selection. Resume with the same --questions, "
                "--split and --limit as the interrupted run, or write to a new "
                "--out directory."
            )
        kept.append(record)
    return kept


def _probe(system: Any, question: str, result: RAGResult) -> Sequence[Any]:
    """Rank to the metric depth without touching the answer.

    Falls back to the chunks ``ask`` returned for a system with no probe, in
    which case the shallower depth shows up in the record's ``capped_ks``
    rather than being passed off as a depth-10 figure.
    """
    retrieve = getattr(system, "retrieve", None)
    if retrieve is None:
        return result.retrieved
    try:
        return retrieve(question, RETRIEVAL_METRIC_DEPTH)
    except RAGError:
        return result.retrieved


# ---------------------------------------------------------------------------
# Storage (EVALUATION_PROTOCOL.md section 27)
# ---------------------------------------------------------------------------

#: BENCHMARK_SPEC.md section 20's summary table, in its order.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "system",
    "accuracy_all",
    "accuracy_answerable",
    "faithfulness",
    "citation_any_correct",
    "citation_precision",
    "recall_at_5",
    "full_recall_at_5",
    "mrr_at_10",
    "abstention_accuracy",
    "false_answer_rate",
    "over_abstention_rate",
    "unsupported_answer_rate",
    "latency_median_s",
    "latency_p95_s",
    "peak_vram_gb",
    # EVALUATION_PROTOCOL.md section 28, for any system that reports them.
    "retrieval_iterations_mean",
    "model_calls_mean",
)


def write_results(out_dir: str | Path, run: BenchmarkRun) -> dict[str, Path]:
    """Write the four files section 27 names, and return their paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = run.summary()

    paths = {
        "config": directory / CONFIG_FILENAME,
        "results": directory / RESULTS_FILENAME,
        "per_question": directory / PER_QUESTION_FILENAME,
        "summary": directory / SUMMARY_FILENAME,
    }
    _write_json(paths["config"], run.config)
    _write_json(
        paths["results"],
        {
            "summary": summary,
            # EVALUATION_PROTOCOL.md 26.1: every headline figure carries a 95% CI.
            "confidence_intervals": statistics.headline_intervals(run.records),
            "by_question_type": summarize_by(run.scores, "question_type"),
            "by_split": summarize_by(run.scores, "split"),
            "failures": [
                {
                    "question_id": s.question_id,
                    "question_type": s.question_type,
                    "category": s.error_category,
                    "error": s.error,
                }
                for s in run.scores
                if s.failed
            ],
        },
    )
    _write_json(paths["per_question"], run.records)
    _write_summary_csv(paths["summary"], summary)
    return paths


def write_checkpoint(out_dir: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    """Rewrite the per-question file mid-run (EXPERIMENT_PLAN.md section 4)."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PER_QUESTION_FILENAME
    _write_json(path, list(records))
    return path


def summarize_by(scores: Sequence[QuestionScore], key: str) -> dict[str, Any]:
    """Break the summary down by question type or split.

    BENCHMARK_SPEC.md section 21 makes this the point of the exercise: the
    benchmark succeeds only if it reveals *where* the system works and fails,
    which one aggregate number cannot do.
    """
    groups: dict[str, list[QuestionScore]] = {}
    for score in scores:
        groups.setdefault(str(getattr(score, key)), []).append(score)
    return {name: summarize(group) for name, group in sorted(groups.items())}


#: Attempts at replacing a result file another process briefly holds open.
_WRITE_ATTEMPTS = 5


def _write_json(path: Path, payload: Any) -> None:
    """Write a result file atomically, retrying a transient lock (DD-064).

    The checkpoint rewrites ``per_question.json`` after every question, so a
    run opens it hundreds of times. On Windows, and on a Drive-backed Colab
    folder, another process (an indexer, a sync client, an editor) can hold it
    for a moment and the open fails. Writing a sibling file and renaming it
    over the target also means a crash mid-write can never leave a truncated
    file for ``--resume`` to trip over.
    """
    text = json.dumps(payload, indent=2, default=str) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    for attempt in range(_WRITE_ATTEMPTS):
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
            return
        except OSError:
            if attempt == _WRITE_ATTEMPTS - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerow({column: summary.get(column) for column in SUMMARY_COLUMNS})


def load_results(out_dir: str | Path) -> dict[str, Any]:
    """Read back everything :func:`write_results` wrote."""
    directory = Path(out_dir)
    with (directory / SUMMARY_FILENAME).open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    return {
        "config": json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8")),
        "results": json.loads(
            (directory / RESULTS_FILENAME).read_text(encoding="utf-8")
        ),
        "per_question": json.loads(
            (directory / PER_QUESTION_FILENAME).read_text(encoding="utf-8")
        ),
        "summary_csv": summary_rows,
    }


COMPARISON_FILENAME = "comparison.json"


def restrict_to_split(loaded: dict[str, Any], split: str) -> dict[str, Any]:
    """One split's records of a stored run, without re-running it.

    A run over every question contains each split's records unchanged -- the
    stack is deterministic and each question is answered independently -- so
    filtering is the same measurement as a run with ``--split``, minus another
    look at the evaluation set. A run over the *other* split cannot be
    restricted and is refused. The copy's config records the split it now
    holds and the filtering, so the comparison's section 10 check still sees
    one split on both sides.
    """
    config = dict(loaded.get("config") or {})
    stored = config.get("split")
    if stored not in (None, split):
        raise statistics.ComparisonError(
            f"This run holds only the {stored!r} split; it cannot be restricted "
            f"to {split!r}."
        )
    records = [
        r for r in loaded.get("per_question") or [] if (r.get("metrics") or {}).get("split") == split
    ]
    if not records:
        raise statistics.ComparisonError(f"No {split!r} records in this run.")
    config["split"] = split
    config["restricted_from_split"] = stored
    config["n_questions"] = len(records)
    return {**loaded, "config": config, "per_question": records}


def compare_runs(
    baseline_dir: str | Path,
    system_dir: str | Path,
    out_path: str | Path | None = None,
    allow_confounded: bool = False,
    split: str | None = None,
) -> dict[str, Any]:
    """Paired comparison of two stored runs (DD-028), written as JSON.

    Reads both result sets as written, so a stored baseline is compared without
    being re-run or rewritten. Defaults to ``<system_dir>/comparison.json``.
    With ``split``, both runs are first restricted to that split's questions
    (:func:`restrict_to_split`).
    """
    baseline, system = load_results(baseline_dir), load_results(system_dir)
    if split:
        baseline, system = restrict_to_split(baseline, split), restrict_to_split(system, split)
    comparison = statistics.compare(
        baseline,
        system,
        baseline_label=str(baseline_dir),
        system_label=str(system_dir),
        allow_confounded=allow_confounded,
    )
    if split:
        comparison["split"] = split
    path = Path(out_path) if out_path else Path(system_dir) / COMPARISON_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, comparison)
    comparison["written_to"] = str(path)
    return comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_benchmark_cli(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    out_dir: str | Path = "results/baseline",
    config: RAGConfig | None = None,
    split: str | None = None,
    limit: int | None = None,
    judge: Any = None,
    skip_validation: bool = False,
    progress=None,
    resume: bool = False,
) -> BenchmarkRun:
    """Load, validate, run, score and store, in that order.

    Validation is not optional by default. A malformed dataset produces
    plausible-looking numbers, which BENCHMARK_SPEC.md calls the worst failure
    mode an evaluation harness has; ``require_valid_benchmark`` already exists
    for exactly this and refusing to start beats explaining the numbers later.

    ``per_question.json`` is rewritten after every question. With ``resume``,
    the records already in ``out_dir`` are kept and only the remaining
    questions are asked (DD-064).
    """
    questions_path = Path(questions_path)
    documents_dir = Path(documents_dir)

    if not skip_validation:
        validation = validate_benchmark(questions_path, documents_dir)
        if not validation.ok:
            raise BenchmarkValidationError(
                "Refusing to benchmark an invalid dataset. Fix these first, or "
                "pass skip_validation=True if you know what you are measuring:"
                f"\n{validation.report()}"
            )

    questions = load_questions(questions_path)
    if split:
        questions = [q for q in questions if q.split == split]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise BenchmarkValidationError(
            f"No questions selected from {questions_path} "
            f"(split={split!r}, limit={limit!r})."
        )

    config = config or RAGConfig.default()
    completed: list[dict[str, Any]] = []
    if resume:
        if judge is not None:
            raise BenchmarkValidationError(
                "--resume cannot be combined with --judge: the judge's summary "
                "would cover only the questions answered in this session."
            )
        completed = load_completed(out_dir, config.validate(), questions)
    run = run_benchmark(
        questions,
        documents_dir=documents_dir,
        config=config,
        judge=judge,
        progress=progress,
        checkpoint_dir=out_dir,
        completed=completed,
    )
    run.config = build_config_snapshot(
        config,
        system=run.system,
        questions_path=questions_path,
        documents_dir=documents_dir,
        n_questions=len(questions),
        split=split,
        judge=judge.describe() if judge is not None else None,
    )
    write_results(out_dir, run)
    return run
