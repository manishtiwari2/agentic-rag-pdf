"""Tests for Phase 6: the ablation arms and the stored-results reports.

Each checks a claim an ablation study could satisfy in appearance only:

* **Each ablation flag changes exactly one configuration field** against the
  reference system (EVALUATION_PROTOCOL.md section 10: change only the
  component under test). An arm that silently changed a second field would
  measure two things and report one.
* **A split restriction is the same records, not a re-run**, and a run over
  the other split is refused rather than quietly compared.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from src.cli import _build_config, main
from src.errors import ConfigurationError
from src.evaluation.benchmark import load_results, restrict_to_split
from src.evaluation.statistics import ComparisonError, compare

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTIC = ROOT / "results" / "agentic"
HYBRID = ROOT / "results" / "hybrid"
needs_results = pytest.mark.skipif(
    not (AGENTIC / "per_question.json").exists() or not (HYBRID / "per_question.json").exists(),
    reason="stored Phase 4/5 results not present",
)


def _config(*flags: str):
    import argparse

    parser = argparse.ArgumentParser()
    from src.cli import _add_system

    _add_system(parser)
    for name in ("--offline", "--low-memory"):
        parser.add_argument(name, action="store_true")
    parser.add_argument("--strategy")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    return _build_config(parser.parse_args(["--offline", *flags]))


def _flatten(config) -> dict[str, object]:
    out: dict[str, object] = {}
    for section, values in dataclasses.asdict(config).items():
        for key, value in values.items():
            out[f"{section}.{key}"] = value
    return out


def _changed(reference, arm) -> dict[str, tuple[object, object]]:
    a, b = _flatten(reference), _flatten(arm)
    return {key: (a[key], b[key]) for key in a if a[key] != b[key]}


class TestEachAblationFlagChangesOneField:
    REFERENCE = ("--system", "agentic")

    @pytest.mark.parametrize(
        "flag, field, value",
        [
            ("--no-planner", "agents.planner_enabled", False),
            ("--no-hybrid", "retrieval.retrievers", ("dense",)),
            ("--no-rerank", "reranking.enabled", False),
            ("--no-refinement", "agents.refinement_enabled", False),
            ("--no-evidence-controller", "agents.evidence_controller_enabled", False),
            ("--no-verification", "agents.verification_enabled", False),
        ],
    )
    def test_the_arm_differs_from_the_reference_in_its_field_alone(self, flag, field, value):
        changed = _changed(_config(*self.REFERENCE), _config(*self.REFERENCE, flag))
        assert changed == {field: (_flatten(_config(*self.REFERENCE))[field], value)}

    def test_the_threshold_flag_sets_the_threshold_alone(self):
        changed = _changed(
            _config(*self.REFERENCE),
            _config(*self.REFERENCE, "--sufficiency-threshold", "0.3"),
        )
        assert changed == {"agents.sufficiency_threshold": (0.5, 0.3)}

    def test_the_iteration_flag_sets_the_cap_alone(self):
        changed = _changed(
            _config(*self.REFERENCE), _config(*self.REFERENCE, "--max-iterations", "3")
        )
        assert changed == {"agents.max_retrieval_iterations": (2, 3)}

    def test_every_arm_validates(self):
        for flag in (
            "--no-planner", "--no-hybrid", "--no-rerank", "--no-refinement",
            "--no-evidence-controller", "--no-verification",
        ):
            _config(*self.REFERENCE, flag).validate()

    def test_hybrid_off_is_refused_on_the_dense_baseline(self):
        with pytest.raises(ConfigurationError, match="--no-hybrid"):
            _config("--system", "dense", "--no-hybrid")

    def test_the_threshold_is_refused_outside_the_agentic_system(self):
        with pytest.raises(ConfigurationError, match="--sufficiency-threshold"):
            _config("--system", "hybrid", "--sufficiency-threshold", "0.3")

    def test_the_arm_config_records_the_retrievers_and_threshold(self):
        described = _config(*self.REFERENCE, "--no-hybrid", "--sufficiency-threshold", "0.4").describe()
        assert described["retrievers"] == ["dense"]
        assert described["sufficiency_threshold"] == 0.4


@needs_results
class TestSplitRestriction:
    @pytest.fixture(scope="class")
    def agentic(self):
        return load_results(AGENTIC)

    def test_restriction_keeps_that_splits_records_unchanged(self, agentic):
        restricted = restrict_to_split(agentic, "eval")
        expected = [r for r in agentic["per_question"] if r["metrics"]["split"] == "eval"]
        assert restricted["per_question"] == expected
        assert restricted["config"]["split"] == "eval"
        assert restricted["config"]["restricted_from_split"] is None
        assert agentic["config"]["split"] is None  # the stored run is not mutated

    def test_a_run_over_the_other_split_is_refused(self, agentic):
        dev = restrict_to_split(agentic, "dev")
        with pytest.raises(ComparisonError, match="cannot be restricted"):
            restrict_to_split(dev, "eval")

    def test_restricted_runs_compare_as_one_split(self, agentic):
        hybrid = load_results(HYBRID)
        comparison = compare(restrict_to_split(hybrid, "eval"), restrict_to_split(agentic, "eval"))
        assert comparison["n_questions"] == len(restrict_to_split(agentic, "eval")["per_question"])
        assert "split" not in comparison["fair_comparison"]["mismatched"]

    def test_compare_runs_cli_accepts_a_split(self, tmp_path, capsys):
        out = tmp_path / "c.json"
        code = main([
            "compare-runs", "--baseline", str(HYBRID), "--system", str(AGENTIC),
            "--split", "dev", "--out", str(out),
        ])
        assert code == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# results/final/ (src/evaluation/report.py)
# ---------------------------------------------------------------------------

from src.evaluation import report  # noqa: E402

FINAL = ROOT / "results" / "final"


def _contrast(diff, low, high, higher_is_better=True):
    return {"difference": diff, "ci_low": low, "ci_high": high, "higher_is_better": higher_is_better}


class TestTheCostRule:
    """DD-058, on synthetic contrasts, so the rule is checked independently of
    what the offline runs happened to produce."""

    def test_worse_without_reads_the_harmful_side(self):
        assert report._worse_without(_contrast(-0.1, -0.2, -0.01))
        assert not report._worse_without(_contrast(-0.1, -0.2, 0.0))  # touches zero
        assert report._worse_without(_contrast(0.1, 0.01, 0.2, higher_is_better=False))
        assert not report._worse_without(_contrast(None, None, None))

    def test_better_without_is_the_mirror(self):
        assert report._better_without(_contrast(0.1, 0.01, 0.2))
        assert report._better_without(_contrast(-0.1, -0.2, -0.01, higher_is_better=False))
        assert not report._better_without(_contrast(0.1, 0.0, 0.2))

    def test_a_reversed_comparison_is_flipped_without_negative_zero(self):
        stored = {"mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.1, "baseline_mean": 0.5,
                  "system_mean": 0.5, "n": 10, "higher_is_better": True}
        flipped = report._oriented(stored, -1)
        assert (flipped["difference"], flipped["ci_low"], flipped["ci_high"]) == (0.0, -0.1, 0.0)
        assert str(flipped["difference"]) == "0.0"


@pytest.mark.skipif(not (ROOT / "results" / "ablations" / "planner_off").exists(),
                    reason="Phase 6 results not present")
class TestTheFinalReport:
    @pytest.fixture(scope="class")
    def eval_table(self):
        return report.final_table(ROOT, split="eval")

    def test_the_headline_is_the_eval_split(self, eval_table):
        assert eval_table["split"] == "eval"
        assert {row["n_questions"] for row in eval_table["rows"]} == {43}
        assert len(eval_table["rows"]) == len(report.FINAL_SYSTEMS)

    def test_peak_vram_is_null_offline_never_zero(self, eval_table):
        for row in eval_table["rows"]:
            assert row["peak_vram_gb"] is None

    def test_the_baselines_report_no_iterations_rather_than_one(self, eval_table):
        dense = eval_table["rows"][0]
        assert dense["retrieval_iterations_mean"]["mean"] is None
        assert dense["model_calls_mean"]["mean"] is None

    def test_a_one_split_run_is_left_out_of_the_every_question_table(self):
        table = report.final_table(ROOT, split=None)
        assert [o["directory"] for o in table["omitted"]] == [report.FINAL_SYSTEMS[-1][1]]
        assert {row["n_questions"] for row in table["rows"]} == {76}

    def test_seven_arms_are_counted(self):
        assert report.ablation_report(ROOT)["n_arms_counted"] == 7

    def test_the_committed_report_is_what_the_stored_results_produce(self, tmp_path):
        """results/final/ is generated, never hand-edited: regenerating it from
        the stored runs reproduces the committed files exactly."""
        if not (FINAL / "REPORT.md").exists():
            pytest.skip("results/final/ not generated")
        paths = report.write_report(ROOT, tmp_path)
        for name in ("final_table.json", "ablations.json", "error_analysis.json", "REPORT.md"):
            assert (tmp_path / name).read_text(encoding="utf-8") == (FINAL / name).read_text(
                encoding="utf-8"
            ), name
        assert set(paths) == {"final_table", "ablations", "error_analysis", "markdown"}


class TestTheStackNote:
    """The report says which stack it measured, read from the runs (DD-069).

    ``final-table --root model_stack`` must not print the offline caveat over
    model-stack numbers, nor an offline report claim the model stack.
    """

    def test_the_stored_runs_are_labelled_offline(self):
        assert report.stack_note(ROOT) == report.OFFLINE_NOTE

    def test_a_model_stack_root_names_its_models(self, tmp_path):
        run = tmp_path / "results" / "agentic"
        run.mkdir(parents=True)
        (run / "config.json").write_text(
            '{"generation_model": "Qwen/Qwen3-4B-Instruct-2507", "quantization": "4bit",'
            ' "embedding_model": "BAAI/bge-m3", "reranker_model": "BAAI/bge-reranker-v2-m3",'
            ' "planner": "llm", "evidence_controller": "llm", "verifier": "llm"}',
            encoding="utf-8",
        )
        note = report.stack_note(tmp_path)
        assert note.startswith("Model stack: Qwen/Qwen3-4B-Instruct-2507 (4bit)")
        assert "BAAI/bge-m3" in note and "One greedy run" in note
        assert "Offline" not in note

    def test_the_tuned_arm_is_the_one_chosen_on_dev(self):
        assert report.TUNED_DIRECTORY == "results/experiments/threshold/eval_0.6"
        assert (ROOT / report.TUNED_DIRECTORY / "per_question.json").exists()
