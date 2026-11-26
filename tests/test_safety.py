"""Question deadlines, logging, and recoverable parsing errors."""

import json
import logging
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest

from src.chat.session import ChatSession, Rewrite
from src.config import GenerationConfig, IngestionConfig
from src.errors import ConfigurationError, DeadlineExceeded, PDFReadError
from src.generation.backends import HuggingFaceBackend
from src.ingestion.pdfplumber_parser import PdfPlumberParser
from src.pipeline import build_pipeline
from src.request_log import JsonFormatter, request_id, request_scope


def test_model_timeout_keeps_slot_until_native_call_exits(monkeypatch):
    model = HuggingFaceBackend(GenerationConfig(call_timeout_s=0.15))
    release = Event()
    entered = Event()
    calls = []

    def blocked(*args):
        calls.append(args)
        entered.set()
        release.wait(5)
        return "late answer"

    monkeypatch.setattr(model, "_generate", blocked)
    try:
        with pytest.raises(DeadlineExceeded, match="Model call timed out"):
            model.generate("system", "first")
        assert entered.is_set()
        with pytest.raises(DeadlineExceeded):
            model.generate("system", "second")
        assert len(calls) == 1
    finally:
        release.set()
    monkeypatch.setattr(model, "_generate", lambda *_: "ok")
    assert model.generate("system", "third") == "ok"


def test_question_budget_includes_rewrite_and_keeps_history_atomic(offline_config, monkeypatch):
    config = replace(offline_config, generation=replace(offline_config.generation, question_timeout_s=0.15))
    pipeline = build_pipeline(config)
    release = Event()
    finished = Event()

    def slow_rewrite(question, history):
        release.wait(5)
        finished.set()
        return Rewrite(standalone=question, rewriter="test")

    rewriter = Mock()
    rewriter.rewrite.side_effect = slow_rewrite
    session = ChatSession(pipeline, rewriter)
    ask = Mock(side_effect=AssertionError("retrieval started after timeout"))
    monkeypatch.setattr(pipeline, "_ask", ask)
    try:
        with pytest.raises(DeadlineExceeded, match="Question timed out"):
            session.ask("question")
        assert not finished.is_set()
    finally:
        release.set()
    assert finished.wait(2)
    assert session.history == [] and session.turns == 0
    ask.assert_not_called()


def test_json_logs_share_request_id_and_exclude_inputs():
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(json.loads(self.format(record)))

    records = []
    handler = Capture()
    handler.setFormatter(JsonFormatter())
    app = logging.getLogger("src")
    app.addHandler(handler)
    parser = PdfPlumberParser()
    page = Mock(page_number=3)
    page.find_tables.side_effect = ValueError("private document text")
    page.extract_text_lines.side_effect = RuntimeError("private document text")
    try:
        for _ in range(2):
            with pytest.raises(PDFReadError, match="page 3"), request_scope("parse PDF"):
                assert parser._extract_tables(page) == []
                parser._text_lines(page, [])
    finally:
        app.removeHandler(handler)
    assert not request_id.get()
    ids = [record["request_id"] for record in records]
    assert len(ids) == 4 and ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert all(ids) and "private document text" not in str(records)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(ConfigurationError):
        GenerationConfig(call_timeout_s=value)


def test_invalid_page_limit_is_rejected():
    with pytest.raises(ConfigurationError):
        IngestionConfig(max_pages=0)
