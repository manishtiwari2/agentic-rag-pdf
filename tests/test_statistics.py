"""Tests for the DD-028 statistics, against answers known before running them.

A bootstrap is easy to get subtly wrong -- resampling the wrong axis, pairing
by position instead of by question, reading the CI off the raw scores -- and
every such mistake still produces plausible-looking intervals. So the method is
checked on inputs whose answer is known analytically before it is trusted on
real data.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from src.evaluation.benchmark import load_results
from src.evaluation.statistics import (
    BOOTSTRAP_RESAMPLES,
    METRICS,
    NO_SIGNIFICANT_DIFFERENCE,
    SIGNIFICANT_IMPROVEMENT,
    SIGNIFICANT_REGRESSION,
    ComparisonError,
    bootstrap_ci,
    compare,
    paired_bootstrap,
    per_question_vectors,
)

BASELINE = pathlib.Path("results/baseline")
needs_baseline = pytest.mark.skipif(
    not (BASELINE / "per_question.json").exists(), reason="results/baseline absent"
)


def as_map(values):
    return {f"q{i:03d}": float(v) for i, v in enumerate(values)}


class TestKnownAnswers:
    def test_zero_differences_collapse_the_interval_and_give_p_one(self):
        scores = as_map([0, 1, 1, 0, 1] * 15)
        result = paired_bootstrap(scores, dict(scores))
        assert result["mean_difference"] == 0.0
        assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0
        assert result["p_value"] == 1.0
        assert result["verdict"] == NO_SIGNIFICANT_DIFFERENCE

    def test_a_constant_difference_collapses_to_that_constant(self):
        a = as_map([0.0] * 40)
        b = as_map([0.25] * 40)
        result = paired_bootstrap(a, b)
        assert result["ci_low"] == pytest.approx(0.25)
        assert result["ci_high"] == pytest.approx(0.25)
        # The smallest p-value 2,000 resamples can report, never an impossible 0.
        assert result["p_value"] == pytest.approx(2 / (BOOTSTRAP_RESAMPLES + 1), abs=1e-6)
        assert result["verdict"] == SIGNIFICANT_IMPROVEMENT

    def test_symmetric_differences_match_the_normal_approximation(self):
        """d_i = +/-1 in equal numbers: mean 0, sd 1, so the 95% CI of the mean
        is +/- 1.96 / sqrt(n) to within bootstrap noise."""
        n = 400
        a = as_map([0.0] * n)
        b = as_map([1.0, -1.0] * (n // 2))
        result = paired_bootstrap(a, b)
        half_width = 1.96 / math.sqrt(n)
        assert result["ci_low"] == pytest.approx(-half_width, abs=0.02)
        assert result["ci_high"] == pytest.approx(half_width, abs=0.02)
        assert result["verdict"] == NO_SIGNIFICANT_DIFFERENCE
        assert result["p_value"] > 0.5

    def test_a_regression_in_a_lower_is_better_metric_is_named_as_one(self):
        a = as_map([0.0] * 30)
        b = as_map([1.0] * 30)
        assert (
            paired_bootstrap(a, b, higher_is_better=False)["verdict"]
            == SIGNIFICANT_REGRESSION
        )

    def test_the_unpaired_interval_matches_the_normal_approximation(self):
        values = [0.0, 1.0] * 200
        result = bootstrap_ci(values)
        half_width = 1.96 * 0.5 / math.sqrt(len(values))
        assert result["mean"] == 0.5
        assert result["ci_low"] == pytest.approx(0.5 - half_width, abs=0.01)
        assert result["ci_high"] == pytest.approx(0.5 + half_width, abs=0.01)

    def test_it_is_reproducible(self):
        rng = np.random.default_rng(0)
        a = as_map(rng.integers(0, 2, 76))
        b = as_map(rng.integers(0, 2, 76))
        assert paired_bootstrap(a, b) == paired_bootstrap(a, b)

    def test_pairing_is_by_question_not_by_position(self):
        a = {"q1": 0.0, "q2": 1.0}
        b = {"q2": 1.0, "q1": 0.0}  # same scores, different order
        assert paired_bootstrap(a, b)["mean_difference"] == 0.0

    def test_no_shared_questions_is_not_computable(self):
        assert paired_bootstrap({}, {})["verdict"] == "not computable"
        assert bootstrap_ci([])["mean"] is None


@needs_baseline
class TestAgainstTheStoredBaseline:
    @pytest.fixture(scope="class")
    def baseline(self):
        return load_results(BASELINE)

    def test_baseline_a_against_itself_is_no_significant_difference(self, baseline):
        comparison = compare(baseline, baseline)
        for name, contrast in comparison["contrasts"].items():
            if name == "model_calls_mean":
                # Baseline A reports no model calls (DD-055): absent, not zero.
                assert contrast["n"] == 0 and contrast["verdict"] == "not computable"
                continue
            assert contrast["verdict"] == NO_SIGNIFICANT_DIFFERENCE, name
            assert contrast["ci_low"] <= 0.0 <= contrast["ci_high"], name
            assert contrast["mean_difference"] == 0.0, name
            assert contrast["p_value"] == 1.0, name
        assert comparison["multiple_comparisons"]["n_significant"] == 0
        assert not comparison["fair_comparison"]["confounded"]

    def test_the_vectors_reproduce_the_stored_summary(self, baseline):
        """Each metric's denominator is the one ``summarize`` used: the vector
        means equal the published figures (records round to 4 dp)."""
        summary = baseline["results"]["summary"]
        vectors = per_question_vectors(baseline["per_question"])
        for spec in METRICS:
            if spec.name == "latency_s":
                continue
            if spec.name not in summary:
                # Added after results/baseline/ was written (DD-056); the same
                # check runs against a fresh summary in tests/test_agentic.py.
                continue
            values = list(vectors[spec.name].values())
            if summary[spec.name] is None:
                # A figure the baseline does not report, such as retrieval
                # iterations (DD-055): null in the summary, and no vector.
                assert values == [], spec.name
                continue
            assert sum(values) / len(values) == pytest.approx(summary[spec.name], abs=5e-5), spec.name
        assert len(vectors["recall_at_5"]) == summary["n_retrieval_scored"]
        assert len(vectors["abstention_accuracy"]) == summary["n_unanswerable"]
        assert len(vectors["over_abstention_rate"]) == summary["n_answerable"]

    def test_a_different_question_set_is_refused(self, baseline):
        fewer = dict(baseline, per_question=baseline["per_question"][:-1])
        with pytest.raises(ComparisonError, match="different questions"):
            compare(baseline, fewer)

    def test_a_confounded_comparison_is_refused_unless_asked_for(self, baseline):
        other = dict(baseline, config=dict(baseline["config"], generation_model="other/model"))
        with pytest.raises(ComparisonError, match="held constant"):
            compare(baseline, other)
        allowed = compare(baseline, other, allow_confounded=True)
        assert allowed["fair_comparison"]["confounded"]
        assert "generation_model" in allowed["fair_comparison"]["mismatched"]

    def test_incompatible_scoring_rules_are_refused(self, baseline):
        other = dict(baseline, config=dict(baseline["config"], scoring_rules_version="1999"))
        with pytest.raises(ComparisonError, match="score-compatible"):
            compare(baseline, other)
