"""Confidence intervals and paired tests (EVALUATION_PROTOCOL.md 26, DD-028).

Two procedures, both fixed by section 26.1 and implemented as written there:

* **Every headline figure carries a 95% CI** -- percentile bootstrap over
  questions: resample the per-question scores with replacement 2,000 times and
  take the 2.5th and 97.5th percentiles of the resampled means.
* **Every system-vs-baseline claim is a paired test.** For each question
  ``d_i = score_system(i) - score_baseline(i)``; bootstrap the mean of ``d``
  2,000 times; report the mean difference, its 95% CI and a two-sided p-value.
  If the CI includes zero the verdict is "no significant difference", whatever
  the sign of the point estimate.

The p-value is the one the percentile interval implies (DD-047): twice the
smaller tail of the bootstrap distribution beyond zero, with the usual +1
correction so a finite resample never reports an impossible p = 0. Identical
systems give exactly 1.0; the smallest reportable value is 2 / 2001.

Everything here reads **stored per-question records** (``per_question.json``),
not in-memory score objects. That is what lets a new run be compared against
``results/baseline/`` without re-running or rewriting it -- the baseline is the
fixed comparison point, and a comparison that needed to regenerate it would
not be comparing against the thing that was published.

This module depends on the metric definitions and never on the harness that
drives runs; ``benchmark.py`` depends on it, not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics import (
    ERROR_CATEGORIES,
    FULL_RECALL_REQUIRED_TYPES,
    SCORE_COMPATIBLE_VERSIONS,
    SCORING_RULES_VERSION,
)

#: EVALUATION_PROTOCOL.md 26.1: "2,000 times". Not changed (DD-047).
BOOTSTRAP_RESAMPLES = 2000
#: EVALUATION_PROTOCOL.md section 11. The same seed for every metric, so every
#: metric over the same questions is resampled with the same draws.
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95

NO_SIGNIFICANT_DIFFERENCE = "no significant difference"
SIGNIFICANT_IMPROVEMENT = "significant improvement"
SIGNIFICANT_REGRESSION = "significant regression"
NOT_COMPUTABLE = "not computable"


# ---------------------------------------------------------------------------
# Per-question metric vectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How one headline figure is read out of a per-question record.

    ``value`` returns None when the question is outside the metric's
    denominator, which is how the three distinct denominators of
    EVALUATION_PROTOCOL.md 21.1 survive into the intervals: an unanswerable
    question is not a 0.0 in Recall@5, it is not in Recall@5 at all.
    """

    name: str
    value: Callable[[Mapping[str, Any]], float | None]
    higher_is_better: bool = True
    note: str = ""


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("metrics") or {}


def _section(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _metrics(record).get(name) or {}


def _retrieval(key: str) -> Callable[[Mapping[str, Any]], float | None]:
    def read(record: Mapping[str, Any]) -> float | None:
        section = _section(record, "retrieval")
        return section.get(key) if section.get("scored") else None

    return read


def _multi_full_recall(record: Mapping[str, Any]) -> float | None:
    if _metrics(record).get("question_type") not in FULL_RECALL_REQUIRED_TYPES:
        return None
    return _retrieval("full_recall_at_5")(record)


def _correct(answerable_only: bool) -> Callable[[Mapping[str, Any]], float | None]:
    def read(record: Mapping[str, Any]) -> float | None:
        metrics = _metrics(record)
        if answerable_only and not metrics.get("answerable"):
            return None
        return metrics.get("correct")

    return read


def _multi_correct(record: Mapping[str, Any]) -> float | None:
    if _metrics(record).get("question_type") not in FULL_RECALL_REQUIRED_TYPES:
        return None
    return _metrics(record).get("correct")


def _answer(key: str) -> Callable[[Mapping[str, Any]], float | None]:
    def read(record: Mapping[str, Any]) -> float | None:
        value = _section(record, "answer").get(key)
        return None if value is None else float(value)

    return read


def _citation(key: str) -> Callable[[Mapping[str, Any]], float | None]:
    def read(record: Mapping[str, Any]) -> float | None:
        section = _section(record, "citation")
        return section.get(key) if section.get("scored") else None

    return read


def _abstention(key: str, answerable: bool) -> Callable[[Mapping[str, Any]], float | None]:
    def read(record: Mapping[str, Any]) -> float | None:
        section = _section(record, "abstention")
        if bool(section.get("answerable")) != answerable:
            return None
        return 1.0 if section.get(key) else 0.0

    return read


def _latency(record: Mapping[str, Any]) -> float | None:
    # The harness's latency figures cover questions that ran; a question that
    # raised has no comparable latency.
    if _metrics(record).get("error"):
        return None
    value = record.get("latency")
    return None if value is None else float(value)


def _model_calls(record: Mapping[str, Any]) -> float | None:
    # Baselines report no model calls (DD-055): absent, not zero.
    value = record.get("model_calls")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


#: The headline figures, with the same denominators ``metrics.summarize`` uses.
#: A test holds the means of these vectors to the summary's figures.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("accuracy_all", _correct(answerable_only=False)),
    MetricSpec("accuracy_answerable", _correct(answerable_only=True)),
    MetricSpec(
        "accuracy_multi_hop_comparison",
        _multi_correct,
        note="multi_hop + comparison questions only (RQ3, DD-056)",
    ),
    MetricSpec("recall_at_1", _retrieval("recall_at_1")),
    MetricSpec("recall_at_3", _retrieval("recall_at_3")),
    MetricSpec("recall_at_5", _retrieval("recall_at_5")),
    MetricSpec("recall_at_10", _retrieval("recall_at_10")),
    MetricSpec("full_recall_at_5", _retrieval("full_recall_at_5")),
    MetricSpec("full_recall_at_10", _retrieval("full_recall_at_10")),
    MetricSpec("full_recall_at_5_multi_hop_comparison", _multi_full_recall),
    MetricSpec("mrr_at_10", _retrieval("mrr_at_10")),
    MetricSpec("ndcg_at_10", _retrieval("ndcg_at_10"), note="binary relevance (DD-040)"),
    MetricSpec("faithfulness", _answer("faithfulness_token")),
    MetricSpec("citation_any_correct", _citation("any_correct")),
    MetricSpec(
        "citation_precision",
        _citation("precision"),
        note="over questions where something was cited (DD-042)",
    ),
    MetricSpec("abstention_accuracy", _abstention("abstained", answerable=False)),
    MetricSpec(
        "false_answer_rate",
        _abstention("false_answer", answerable=False),
        higher_is_better=False,
    ),
    MetricSpec(
        "over_abstention_rate",
        _abstention("over_abstention", answerable=True),
        higher_is_better=False,
    ),
    MetricSpec(
        "unsupported_answer_rate", _answer("unsupported"), higher_is_better=False
    ),
    MetricSpec(
        "latency_s",
        _latency,
        higher_is_better=False,
        note="mean seconds per question; a cost, not a quality figure",
    ),
    MetricSpec(
        "model_calls_mean",
        _model_calls,
        higher_is_better=False,
        note=(
            "generation-model invocations per question (DD-055); a cost, not a "
            "quality figure; only systems that report it"
        ),
    ),
)
METRICS_BY_NAME: dict[str, MetricSpec] = {spec.name: spec for spec in METRICS}

#: Declared before the Phase 4 run (DD-048), per section 26.2: a component
#: improving the metric it plausibly *should* improve is evidence; one isolated
#: significant result among many is a hypothesis.
PRIMARY_METRICS: dict[str, dict[str, list[str]]] = {
    "RQ1": {
        "question": ["Does hybrid retrieval improve evidence retrieval over dense alone?"],
        "primary": ["recall_at_5"],
        "secondary": ["mrr_at_10", "full_recall_at_5_multi_hop_comparison"],
    },
    "RQ2": {
        "question": [
            "Does reranking improve final answer quality enough to justify its cost?"
        ],
        "primary": ["accuracy_all", "faithfulness"],
        "cost": ["latency_s"],
        "secondary": ["recall_at_5", "mrr_at_10"],
    },
    # Declared before the Phase 5 run (DD-056), with the floors that bound what
    # the offline run can show stated there rather than after the result.
    "RQ3": {
        "question": [
            "Does adaptive/agentic retrieval improve difficult and multi-hop questions?"
        ],
        "primary": [
            "full_recall_at_5_multi_hop_comparison",
            "accuracy_multi_hop_comparison",
        ],
        "cost": ["latency_s"],
        "secondary": ["accuracy_all", "recall_at_5", "mrr_at_10"],
    },
    "RQ4": {
        "question": ["Does evidence verification reduce unsupported answers?"],
        "primary": ["unsupported_answer_rate"],
        "secondary": ["false_answer_rate", "faithfulness", "citation_precision"],
    },
    # EXPERIMENT_PLAN.md section 3: the agentic system is kept only if one of
    # these improves with a CI excluding zero, against Baseline B (DD-056).
    "keep_or_drop": {
        "question": [
            "Does the agentic pipeline improve accuracy, faithfulness, citation "
            "correctness or abstention over Baseline B?"
        ],
        "primary": [
            "accuracy_all",
            "faithfulness",
            "citation_any_correct",
            "abstention_accuracy",
        ],
    },
    # Declared before any Phase 6 run (DD-058). Each ablation arm is compared
    # against the full agentic system (results/agentic/) as arm - reference; a
    # component "helps" when removing it makes one of its own metrics
    # significantly worse. "attribution" metrics are read on every arm to find
    # the cause of Phase 5's regression.
    "ablation:planner_off": {
        "question": ["Does the planner's decomposition retrieve both hops?"],
        "primary": ["full_recall_at_5_multi_hop_comparison", "accuracy_multi_hop_comparison"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:hybrid_off": {
        "question": ["Inside the agentic system, does hybrid retrieval beat dense alone?"],
        "primary": ["recall_at_5", "mrr_at_10"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:reranker_off_agentic": {
        "question": ["Inside the agentic system, does reranking improve ranking and answers?"],
        "primary": ["mrr_at_10", "accuracy_all"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:refinement_off": {
        "question": ["Does a second retrieval round recover evidence the first missed?"],
        "primary": ["over_abstention_rate", "full_recall_at_5"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:evidence_controller_off": {
        "question": ["Does the evidence controller refuse unanswerable questions?"],
        "primary": ["abstention_accuracy", "over_abstention_rate"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:verification_off": {
        "question": ["Does verification reduce unsupported answers?"],
        "primary": ["unsupported_answer_rate", "faithfulness"],
        "cost": ["latency_s", "model_calls_mean"],
    },
    "ablation:attribution": {
        "question": ["Which component caused Phase 5's faithfulness and citation regression?"],
        "primary": ["faithfulness", "citation_any_correct"],
    },
    # Dev only (DD-058): the threshold is chosen on accuracy (all), ties broken
    # by citation any-correct, then over-abstention, then nearness to 0.5.
    "threshold_selection": {
        "question": ["Which agents.sufficiency_threshold, chosen on --split dev?"],
        "primary": ["accuracy_all"],
        "secondary": ["citation_any_correct", "over_abstention_rate"],
    },
}


def per_question_vectors(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[MetricSpec] = METRICS,
) -> dict[str, dict[str, float]]:
    """``{metric: {question_id: value}}``, each metric over its own denominator."""
    vectors: dict[str, dict[str, float]] = {spec.name: {} for spec in metrics}
    for record in records:
        question_id = str(record["question_id"])
        for spec in metrics:
            value = spec.value(record)
            if value is not None:
                vectors[spec.name][question_id] = float(value)
    return vectors


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


def _resampled_means(
    values: np.ndarray, resamples: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[indices].mean(axis=1)


def _percentiles(means: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> dict[str, Any]:
    """Percentile bootstrap CI of the mean (EVALUATION_PROTOCOL.md 26.1)."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    low, high = _percentiles(_resampled_means(data, resamples, seed), level)
    return {
        "n": int(data.size),
        "mean": _round(data.mean()),
        "ci_low": _round(low),
        "ci_high": _round(high),
    }


def paired_bootstrap(
    baseline: Mapping[str, float],
    system: Mapping[str, float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Paired bootstrap of ``mean(system - baseline)`` over shared questions.

    Pairs on the question ids present in both mappings -- the questions inside
    the metric's denominator for both systems. Sorted ids make the pairing, and
    so the draws, independent of record order.
    """
    shared = sorted(set(baseline) & set(system))
    if not shared:
        return {
            "n": 0,
            "baseline_mean": None,
            "system_mean": None,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "verdict": NOT_COMPUTABLE,
        }
    a = np.asarray([baseline[q] for q in shared], dtype=float)
    b = np.asarray([system[q] for q in shared], dtype=float)
    differences = b - a
    means = _resampled_means(differences, resamples, seed)
    low, high = _percentiles(means, level)

    # Two-sided, from the same bootstrap distribution the CI is read from, so
    # the two never disagree about which side of zero the result is on.
    below = (int(np.sum(means <= 0.0)) + 1) / (resamples + 1)
    above = (int(np.sum(means >= 0.0)) + 1) / (resamples + 1)
    p_value = min(1.0, 2.0 * min(below, above))

    if low <= 0.0 <= high:
        verdict = NO_SIGNIFICANT_DIFFERENCE
    else:
        better = (low > 0.0) == higher_is_better
        verdict = SIGNIFICANT_IMPROVEMENT if better else SIGNIFICANT_REGRESSION

    return {
        "n": len(shared),
        "baseline_mean": _round(a.mean()),
        "system_mean": _round(b.mean()),
        "mean_difference": _round(differences.mean()),
        "ci_low": _round(low),
        "ci_high": _round(high),
        "p_value": _round(p_value),
        "questions_changed": int(np.count_nonzero(differences)),
        "verdict": verdict,
    }


def headline_intervals(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A 95% CI for every headline figure of one run."""
    vectors = per_question_vectors(records)
    return {
        "method": method_description(),
        "metrics": {
            spec.name: bootstrap_ci(list(vectors[spec.name].values()))
            for spec in METRICS
        },
    }


def method_description() -> dict[str, Any]:
    return {
        "procedure": "percentile bootstrap over questions (EVALUATION_PROTOCOL.md 26.1)",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "ci_level": CI_LEVEL,
        "ci": "2.5th and 97.5th percentiles of the resampled means",
        "p_value": (
            "two-sided: min(1, 2 * min((#means<=0 + 1), (#means>=0 + 1)) / "
            "(resamples + 1)) over the paired bootstrap distribution (DD-047)"
        ),
        "reporting_rule": (
            "if the 95% CI of the difference includes zero, the result is "
            "'no significant difference', regardless of the point estimate"
        ),
        "source": "per_question.json records as stored",
    }


# ---------------------------------------------------------------------------
# Comparing two stored runs
# ---------------------------------------------------------------------------

#: EVALUATION_PROTOCOL.md section 10: held constant between compared systems.
#: A difference here means the comparison measures more than one change.
HELD_CONSTANT: tuple[str, ...] = (
    "benchmark_version",
    "split",
    "generation_model",
    "quantization",
    "embedding_model",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "temperature",
    "max_new_tokens",
)
#: What a retrieval-engineering comparison is allowed to vary.
VARIED: tuple[str, ...] = (
    "system",
    "retrieval_strategy",
    "retrievers",
    "fusion",
    "rrf_k",
    "candidate_k",
    "reranker_enabled",
    "reranker_model",
    "rerank_k",
    # The agentic system's switches (DD-050): what Phase 5 and 6 vary.
    "max_retrieval_iterations",
    "planner_enabled",
    "planner",
    "evidence_controller_enabled",
    "evidence_controller",
    "sufficiency_threshold",
    "refinement_enabled",
    "verification_enabled",
    "verifier",
)


class ComparisonError(ValueError):
    """The two runs cannot be compared as they stand."""


def scores_comparable(version_a: str | None, version_b: str | None) -> bool:
    """Were the per-question scores of both runs computed by the same formulas?"""
    if version_a == version_b:
        return True
    pair = {version_a, version_b}
    return any(pair == {SCORING_RULES_VERSION, old} for old in SCORE_COMPATIBLE_VERSIONS)


def compare(
    baseline: Mapping[str, Any],
    system: Mapping[str, Any],
    baseline_label: str = "baseline",
    system_label: str = "system",
    allow_confounded: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired comparison of two loaded result sets (``load_results`` output)."""
    base_config = baseline.get("config") or {}
    sys_config = system.get("config") or {}
    base_records = baseline.get("per_question") or []
    sys_records = system.get("per_question") or []

    base_ids = {str(r["question_id"]) for r in base_records}
    sys_ids = {str(r["question_id"]) for r in sys_records}
    if base_ids != sys_ids:
        raise ComparisonError(
            "The two runs answered different questions "
            f"({len(base_ids - sys_ids)} only in the baseline, "
            f"{len(sys_ids - base_ids)} only in the system). A paired test "
            "needs the same questions on both sides."
        )

    versions = {
        "baseline": base_config.get("scoring_rules_version"),
        "system": sys_config.get("scoring_rules_version"),
    }
    if not scores_comparable(versions["baseline"], versions["system"]):
        raise ComparisonError(
            f"Scoring rules {versions['baseline']!r} and {versions['system']!r} "
            "are not declared score-compatible (metrics.SCORE_COMPATIBLE_VERSIONS)."
        )

    mismatched = {
        key: {"baseline": base_config.get(key), "system": sys_config.get(key)}
        for key in HELD_CONSTANT
        if base_config.get(key) != sys_config.get(key)
    }
    if mismatched and not allow_confounded:
        raise ComparisonError(
            "EVALUATION_PROTOCOL.md section 10: these must be held constant "
            f"between compared systems and are not: {sorted(mismatched)}. Pass "
            "allow_confounded=True to compare anyway; the result will say so."
        )

    base_vectors = per_question_vectors(base_records)
    sys_vectors = per_question_vectors(sys_records)
    contrasts: dict[str, Any] = {}
    for spec in METRICS:
        contrast = paired_bootstrap(
            base_vectors[spec.name],
            sys_vectors[spec.name],
            resamples=resamples,
            seed=seed,
            higher_is_better=spec.higher_is_better,
        )
        contrast["higher_is_better"] = spec.higher_is_better
        contrast["baseline_ci"] = bootstrap_ci(
            list(base_vectors[spec.name].values()), resamples, seed
        )
        contrast["system_ci"] = bootstrap_ci(
            list(sys_vectors[spec.name].values()), resamples, seed
        )
        if spec.note:
            contrast["note"] = spec.note
        contrasts[spec.name] = contrast

    def categories(loaded: Mapping[str, Any]) -> dict[str, int]:
        summary = (loaded.get("results") or {}).get("summary") or {}
        found = summary.get("error_categories") or {}
        return {name: int(found.get(name, 0)) for name in ERROR_CATEGORIES}

    return {
        "baseline": baseline_label,
        "system": system_label,
        "baseline_system_name": base_config.get("system"),
        "system_system_name": sys_config.get("system"),
        "n_questions": len(base_ids),
        "method": method_description(),
        "fair_comparison": {
            "held_constant": list(HELD_CONSTANT),
            "mismatched": mismatched,
            "confounded": bool(mismatched),
            "varied": {
                key: {"baseline": base_config.get(key), "system": sys_config.get(key)}
                for key in VARIED
                if base_config.get(key) != sys_config.get(key)
            },
        },
        "versions": {
            "scoring_rules": versions,
            "error_taxonomy": {
                "baseline": base_config.get("error_taxonomy_version"),
                "system": sys_config.get("error_taxonomy_version"),
            },
            "abstention_patterns": {
                "baseline": base_config.get("abstention_patterns_version"),
                "system": sys_config.get("abstention_patterns_version"),
            },
        },
        "multiple_comparisons": {
            "n_comparisons": len(contrasts),
            "note": (
                "EVALUATION_PROTOCOL.md 26.2: at a 5% threshold roughly one in "
                "twenty comparisons looks significant by chance. Read the "
                "primary metrics first; an isolated significant secondary "
                "result is a hypothesis."
            ),
            "n_significant": sum(
                1
                for c in contrasts.values()
                if c["verdict"] in (SIGNIFICANT_IMPROVEMENT, SIGNIFICANT_REGRESSION)
            ),
        },
        "primary_metrics": PRIMARY_METRICS,
        "contrasts": contrasts,
        "error_categories": {
            "baseline": categories(baseline),
            "system": categories(system),
        },
    }
