"""Dev parameter studies: the table is generated and the rule is code (DD-071)."""

from __future__ import annotations

import csv
import json
import pathlib

import pytest

from src.cli import main
from src.evaluation import statistics, sweeps

N = 30
SPEC = sweeps.STUDIES["chunking"]


def _record(qid: int, recall: float, mrr: float, correct: bool, qtype: str = "factual") -> dict:
    return {
        "question_id": f"q{qid:03d}",
        "latency": 0.01,
        "metrics": {
            "question_type": qtype,
            "split": "dev",
            "answerable": True,
            "correct": correct,
            "retrieval": {
                "scored": True,
                "recall_at_1": recall,
                "recall_at_5": recall,
                "recall_at_10": recall,
                "full_recall_at_5": recall,
                "mrr_at_10": mrr,
            },
        },
    }


def _write_arm(
    root: pathlib.Path,
    strategy: str,
    size: int,
    overlap: int,
    recall_hits: int,
    mrr_hits: int,
    correct_hits: int,
    chunks: int = 100,
    split: str = "dev",
) -> pathlib.Path:
    """An arm whose first ``*_hits`` questions score 1 and the rest 0."""
    directory = root / f"{strategy}_{size}_{overlap}"
    directory.mkdir(parents=True)
    records = [
        _record(
            i,
            recall=1.0 if i < recall_hits else 0.0,
            mrr=1.0 if i < mrr_hits else 0.0,
            correct=i < correct_hits,
            qtype="multi_hop" if i < 3 else "factual",
        )
        for i in range(N)
    ]
    config = {
        "split": split,
        "generation_model": "local/scripted-extractive",
        "scoring_rules_version": "2026-09-23.2",
        "chunking_strategy": strategy,
        "chunk_size": size,
        "chunk_overlap": overlap,
    }
    summary = {"chunks_total": chunks, "indexed_chars_total": chunks * size, "latency_p95_s": 0.01}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "results.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
    (directory / "per_question.json").write_text(json.dumps(records), encoding="utf-8")
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["system"])
    return directory


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "chunking"
    # Incumbent: 15 of 30 retrieved, 15 ranked first, 5 correct.
    _write_arm(base, "structure", 1200, 150, 15, 15, 5)
    return base


class TestTheRule:
    def test_the_incumbent_stays_when_no_challenger_is_significant(self, root):
        _write_arm(root, "fixed", 1200, 150, 16, 15, 5)
        study = sweeps.run_study(SPEC, root)
        assert study["qualified"] == []
        assert study["chosen"] == "structure_1200_150"

    def test_a_significant_primary_gain_with_no_regression_is_chosen(self, root):
        _write_arm(root, "fixed", 800, 150, 28, 28, 5)
        study = sweeps.run_study(SPEC, root)
        assert study["qualified"] == ["fixed_800_150"]
        assert study["chosen"] == "fixed_800_150"

    def test_a_guard_regression_disqualifies(self, root):
        # Recall up, but the questions it ranks first fall from 15 to 2.
        _write_arm(root, "fixed", 800, 150, 28, 2, 5)
        study = sweeps.run_study(SPEC, root)
        assert study["qualified"] == []
        assert study["chosen"] == "structure_1200_150"

    def test_ties_go_to_recall_then_mrr_then_fewer_chunks(self, root):
        _write_arm(root, "fixed", 400, 0, 28, 28, 5, chunks=300)
        _write_arm(root, "fixed", 800, 0, 28, 28, 5, chunks=150)
        _write_arm(root, "fixed", 1600, 0, 27, 28, 5, chunks=50)
        study = sweeps.run_study(SPEC, root)
        assert set(study["qualified"]) == {"fixed_400_0", "fixed_800_0", "fixed_1600_0"}
        assert study["chosen"] == "fixed_800_0"

    def test_a_reference_arm_is_never_chosen(self, root):
        _write_arm(root, "fixed", 800, 150, 28, 28, 5)
        spec = sweeps.StudySpec(**{**SPEC.__dict__, "references": ("fixed_800_150",)})
        study = sweeps.run_study(spec, root)
        assert study["chosen"] == "structure_1200_150"

    def test_an_eval_arm_is_refused(self, root):
        _write_arm(root, "fixed", 800, 150, 15, 15, 5, split="eval")
        with pytest.raises(statistics.ComparisonError, match="dev only"):
            sweeps.run_study(SPEC, root)

    def test_a_missing_incumbent_is_refused(self, tmp_path):
        _write_arm(tmp_path, "fixed", 800, 150, 15, 15, 5)
        with pytest.raises(statistics.ComparisonError, match="incumbent"):
            sweeps.run_study(SPEC, tmp_path)


class TestTheTable:
    def test_values_are_read_from_the_stored_run(self, root):
        study = sweeps.run_study(SPEC, root)
        values = study["rows"][0]["values"]
        assert values["recall_at_5"] == pytest.approx(0.5)
        assert values["chunks_total"] == 100
        # The three multi_hop questions are the first three, all retrieved.
        assert values["full_recall_at_5_multi_hop"] == pytest.approx(1.0)

    def test_render_names_the_stack_and_the_choice_and_is_deterministic(self, root):
        _write_arm(root, "fixed", 1200, 150, 15, 15, 5)
        study = sweeps.run_study(SPEC, root)
        text = sweeps.render(study)
        assert "Offline stand-in stack" in text
        assert "**Chosen:** `structure_1200_150`" in text
        assert sweeps.render(sweeps.run_study(SPEC, root)) == text

    def test_structure_versus_fixed_is_counted_per_question_type(self, root):
        # Fixed retrieves 2 fewer multi_hop and 0 more factual questions.
        _write_arm(root, "fixed", 1200, 150, 15, 15, 5)
        study = sweeps.run_study(SPEC, root)
        tally = study["strategy_by_type"]["tally"]["recall_at_5"]
        assert tally["multi_hop"] == {"wins": 0, "ties": 3, "losses": 0}
        assert sum(tally["factual"].values()) == N - 3

    def test_the_cli_writes_both_files(self, root, capsys):
        assert main(["sweep-table", "--study", "chunking", "--root", str(root)]) == 0
        assert (root / sweeps.TABLE_FILENAME).exists()
        assert json.loads((root / sweeps.TABLE_JSON_FILENAME).read_text())["chosen"]
        assert "Chosen: structure_1200_150" in capsys.readouterr().out
