"""The README's results table is generated, never typed (issue #36)."""

from __future__ import annotations

import pathlib

import pytest

from src.evaluation.report import README_BEGIN, README_END, headline_section, splice_readme

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "results" / "final" / "REPORT.md"


def test_the_readme_block_matches_the_stored_report():
    if not REPORT.exists():  # pragma: no cover - a checkout without results
        pytest.skip("results/final/REPORT.md is not present")
    readme = README.read_bytes().decode("utf-8")
    report = REPORT.read_text(encoding="utf-8")
    assert splice_readme(readme, report) == readme, (
        "README.md's results block is stale: run `python -m src.cli readme-table`."
    )


def test_the_block_carries_the_stack_label():
    if not REPORT.exists():  # pragma: no cover
        pytest.skip("results/final/REPORT.md is not present")
    assert "stand-in stack" in headline_section(REPORT.read_text(encoding="utf-8"))


def test_splicing_replaces_only_the_marked_block():
    report = "# R\n\n### Eval split - headline\n\n| a |\n\n### Next\n\nnot this\n"
    readme = f"intro\n{README_BEGIN}\nold\n{README_END}\noutro\n"
    spliced = splice_readme(readme, report)
    assert spliced == f"intro\n{README_BEGIN}\n| a |\n{README_END}\noutro\n"
