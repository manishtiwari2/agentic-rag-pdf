"""Request logging and recoverable PDF parsing errors."""

import json
import logging
from unittest.mock import Mock

import pytest

from src.errors import PDFReadError
from src.ingestion.pdfplumber_parser import PdfPlumberParser
from src.request_log import JsonFormatter, request_id, request_scope


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
