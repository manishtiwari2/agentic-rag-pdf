"""Stale and conflicting sources (DD-074): versions, detection, policy."""

from __future__ import annotations

import dataclasses

import pymupdf
import pytest

from src.config import ConflictConfig, RAGConfig
from src.versioning import (
    FILENAME,
    PDF_METADATA,
    UNKNOWN,
    USER,
    compare_versions,
    resolve_version,
)
from src.pipeline import build_pipeline

QUESTION = "How many customers did the authority serve?"


def _pdf(path, customers: str, date: str | None = None):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 520, 760),
        f"Water Supply\n\nThe authority served {customers} customers across the region during the year.",
        fontsize=11,
    )
    doc.set_metadata({"creationDate": date} if date else {})
    doc.save(path)
    return path


def _pipeline(enabled: bool = True):
    base = RAGConfig.offline()
    return build_pipeline(
        dataclasses.replace(
            base,
            retrieval=dataclasses.replace(base.retrieval, strategy="hybrid"),
            conflicts=ConflictConfig(enabled=enabled, detector="rules"),
        )
    )


class TestVersionResolution:
    def test_user_beats_metadata_beats_filename(self):
        meta = {"pdf_creation_date": "D:20240215093000+05'30'"}
        assert resolve_version("report_2023.pdf", meta, "2025-03 FY25").source == USER
        assert resolve_version("report_2023.pdf", meta).date == "2024-02-15"
        assert resolve_version("report_2023.pdf", meta).source == PDF_METADATA
        named = resolve_version("report_2023-06.pdf")
        assert (named.date, named.precision, named.source) == ("2023-06", "month", FILENAME)
        assert resolve_version("v20240.pdf").source == UNKNOWN

    def test_a_label_alone_orders_nothing(self):
        version = resolve_version("memo.pdf", None, "v2")
        assert (version.label, version.date, version.source) == ("v2", None, UNKNOWN)

    def test_comparison_is_at_the_shared_precision_and_refuses_unknown(self):
        assert compare_versions(resolve_version("a_2024.pdf"), resolve_version("b_2024-06.pdf")) == 0
        assert compare_versions(resolve_version("a_2025.pdf"), resolve_version("b_2024-06.pdf")) == 1
        assert compare_versions(resolve_version("a_2025.pdf"), resolve_version("memo.pdf")) is None


class TestPolicy:
    def test_the_newer_source_answers_and_both_are_cited(self, tmp_path):
        old = _pdf(tmp_path / "old.pdf", "398,500", "D:20240215000000")
        new = _pdf(tmp_path / "new.pdf", "412,000", "D:20250215000000")
        pipeline = _pipeline()
        pipeline.index_documents([old, new])
        result = pipeline.ask(QUESTION)
        assert result.metadata["conflict_policy"] == "prefer_newer"
        assert "412,000" in result.answer and "supersedes" in result.answer
        assert len({c.document_id for c in result.citations}) == 2
        assert not result.abstained

    def test_an_undated_disagreement_refuses_and_shows_both(self, tmp_path):
        new = _pdf(tmp_path / "new.pdf", "412,000", "D:20250215000000")
        memo = _pdf(tmp_path / "memo.pdf", "405,000")
        pipeline = _pipeline()
        pipeline.index_documents([new, memo])
        result = pipeline.ask(QUESTION)
        assert result.metadata["conflict_policy"] == "refused"
        assert result.abstained
        assert "412,000" in result.answer and "405,000" in result.answer

    def test_a_user_date_can_settle_an_undated_document(self, tmp_path):
        new = _pdf(tmp_path / "new.pdf", "412,000", "D:20250215000000")
        memo = _pdf(tmp_path / "memo.pdf", "405,000")
        pipeline = _pipeline()
        pipeline.index_documents([new, memo], versions={"memo.pdf": "2025-06-01"})
        result = pipeline.ask(QUESTION)
        assert result.metadata["conflict_policy"] == "prefer_newer"
        assert "405,000" in result.answer.split("Note:")[0]

    def test_agreeing_sources_are_not_a_conflict(self, tmp_path):
        a = _pdf(tmp_path / "a.pdf", "412,000", "D:20240215000000")
        b = _pdf(tmp_path / "b.pdf", "412,000", "D:20250215000000")
        pipeline = _pipeline()
        pipeline.index_documents([a, b])
        assert pipeline.ask(QUESTION).metadata["conflict_policy"] == "none"

    def test_off_by_default_adds_nothing_to_the_result(self, tmp_path):
        old = _pdf(tmp_path / "old.pdf", "398,500", "D:20240215000000")
        new = _pdf(tmp_path / "new.pdf", "412,000", "D:20250215000000")
        pipeline = _pipeline(enabled=False)
        pipeline.index_documents([old, new])
        metadata = pipeline.ask(QUESTION).metadata
        assert "conflicts" not in metadata and "conflict_policy" not in metadata


@pytest.mark.parametrize("offline", [True])
def test_the_chat_app_turns_conflict_checking_on(offline):
    from src.chat.ui import make_config

    assert make_config("hybrid", offline).conflicts.enabled
