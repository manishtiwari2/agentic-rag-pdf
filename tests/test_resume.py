"""Resumable benchmark runs (DD-064).

A Colab session that dies 50 minutes into an arm must not cost the arm. The
guarantee tested here is the strong one: an interrupted run, resumed, stores
exactly what an uninterrupted run stores, record for record, apart from the
timings that cannot be equal.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from src.config import RAGConfig
from src.errors import BenchmarkValidationError
from src.evaluation.benchmark import run_benchmark_cli

BENCHMARK = pathlib.Path("benchmark/questions.json")
#: q001-q008 are doc1.pdf and q009-q010 doc2.pdf, so an interruption after the
#: ninth question leaves one document finished and one half done.
LIMIT = 10

pytestmark = pytest.mark.skipif(
    not BENCHMARK.exists(), reason="benchmark/questions.json is not present"
)


def _config() -> RAGConfig:
    config = RAGConfig.offline()
    return dataclasses.replace(
        config, retrieval=dataclasses.replace(config.retrieval, strategy="agentic")
    )


def _strip(value):
    """Drop the timing fields, which differ between any two runs."""
    if isinstance(value, dict):
        return {
            k: _strip(v)
            for k, v in value.items()
            if k != "latency" and not k.endswith("_s")
        }
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _records(out: pathlib.Path) -> list[dict]:
    return json.loads((out / "per_question.json").read_text(encoding="utf-8"))


class _Interrupt(Exception):
    pass


def _interrupt_after(n: int):
    seen = []

    def progress(_question_run):
        seen.append(1)
        if len(seen) == n:
            raise _Interrupt

    return progress


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    run_benchmark_cli(out_dir=out, config=_config(), limit=LIMIT, skip_validation=True)
    return out


class TestResume:
    def test_the_cli_checkpoints_after_every_question(self, tmp_path):
        with pytest.raises(_Interrupt):
            run_benchmark_cli(
                out_dir=tmp_path, config=_config(), limit=LIMIT, skip_validation=True,
                progress=_interrupt_after(3),
            )
        assert [r["question_id"] for r in _records(tmp_path)] == ["q001", "q002", "q003"]
        assert not (tmp_path / "results.json").exists()

    def test_a_resumed_run_equals_an_uninterrupted_one(self, tmp_path, uninterrupted):
        with pytest.raises(_Interrupt):
            run_benchmark_cli(
                out_dir=tmp_path, config=_config(), limit=LIMIT, skip_validation=True,
                progress=_interrupt_after(9),
            )
        asked = []
        run = run_benchmark_cli(
            out_dir=tmp_path, config=_config(), limit=LIMIT, resume=True,
            skip_validation=True, progress=lambda qr: asked.append(qr.question.id),
        )
        assert asked == ["q010"]
        # doc1.pdf was finished before the interruption, so only doc2.pdf is
        # indexed again.
        assert run.summary()["documents_indexed"] == 1
        assert run.summary()["resumed_records"] == 9
        assert _strip(_records(tmp_path)) == _strip(_records(uninterrupted))
        stored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        full = json.loads((uninterrupted / "results.json").read_text(encoding="utf-8"))
        for key in ("accuracy_all", "faithfulness", "citation_any_correct", "recall_at_5"):
            assert stored["summary"][key] == pytest.approx(full["summary"][key], abs=1e-4)

    def test_a_different_configuration_is_refused(self, tmp_path):
        with pytest.raises(_Interrupt):
            run_benchmark_cli(
                out_dir=tmp_path, config=_config(), limit=LIMIT, skip_validation=True,
                progress=_interrupt_after(2),
            )
        other = _config()
        other = dataclasses.replace(
            other, chunking=dataclasses.replace(other.chunking, chunk_size=900)
        )
        with pytest.raises(BenchmarkValidationError, match="fingerprint"):
            run_benchmark_cli(
                out_dir=tmp_path, config=other, limit=LIMIT, resume=True, skip_validation=True
            )

    def test_a_record_outside_the_selection_is_refused(self, tmp_path):
        with pytest.raises(_Interrupt):
            run_benchmark_cli(
                out_dir=tmp_path, config=_config(), limit=LIMIT, skip_validation=True,
                progress=_interrupt_after(2),
            )
        with pytest.raises(BenchmarkValidationError, match="selection"):
            run_benchmark_cli(
                out_dir=tmp_path, config=_config(), split="eval", limit=3, resume=True,
                skip_validation=True,
            )

    def test_resume_with_nothing_stored_is_a_fresh_run(self, tmp_path):
        run = run_benchmark_cli(
            out_dir=tmp_path, config=_config(), limit=2, resume=True, skip_validation=True
        )
        assert len(run.records) == 2 and "resumed_records" not in run.summary()
