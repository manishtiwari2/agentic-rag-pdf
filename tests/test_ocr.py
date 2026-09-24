"""OCR for pages with no text layer, on by default (DD-067).

A fake engine stands in for Tesseract, so these run on any machine. One test
runs the real engine and is skipped where the Tesseract binary is absent.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import pytest

from src.chunking.base import build_chunker
from src.config import ChunkingConfig, IngestionConfig, RAGConfig
from src.errors import OCRUnavailableError, ScannedPDFError
from src.ingestion.ocr import TesseractOcr
from src.ingestion.parser import build_parser
from src.pipeline import DenseRAGPipeline

from . import pdf_fixtures as pdfs

OCR_TEXT = (
    "3 Results\n\n"
    "The scanned page reports a median latency of 4.2 seconds per question\n"
    "on the T4 GPU, measured over the whole benchmark."
)
BENCHMARK_PDFS = sorted(pathlib.Path("benchmark/documents").glob("*.pdf"))


class FakeOcr:
    name = "test/fake-ocr"

    def __init__(self, text: str = OCR_TEXT) -> None:
        self.text = text
        self.calls = 0

    def image_to_text(self, image) -> str:
        self.calls += 1
        assert image.size[0] > 1000  # rendered at OCR resolution, not a thumbnail
        return self.text


class NeverOcr:
    """Fails the test if a page with a text layer is ever sent to OCR."""

    name = "test/never"

    def image_to_text(self, image) -> str:  # pragma: no cover - must not run
        raise AssertionError("a page with a text layer was OCR'd")


def _parser(parser_name: str, engine, **overrides):
    return build_parser(IngestionConfig(parser=parser_name, **overrides), ocr_engine=engine)


def _snapshot(document):
    return (
        document.document_id,
        document.metadata,
        [(p.page_number, p.text, p.blocks, p.image_count, p.metadata) for p in document.pages],
    )


class TestOnByDefault:
    @pytest.mark.parametrize("preset", ["default", "offline", "low_memory"])
    def test_every_preset_has_ocr_on(self, preset):
        assert getattr(RAGConfig, preset)().ingestion.ocr is True

    def test_describe_records_it(self):
        assert RAGConfig.offline().describe()["ocr"] is True


class TestScannedPages:
    def test_a_fully_scanned_pdf_is_ocrd_page_by_page(self, parser_name):
        engine = FakeOcr()
        document = _parser(parser_name, engine).parse_bytes(pdfs.scanned_pdf(3), "scan.pdf")
        assert engine.calls == 3
        assert document.metadata["ocr_pages"] == [1, 2, 3]
        for page in document.pages:
            assert page.metadata["ocr"] is True and page.metadata["ocr_engine"] == engine.name
            assert "median latency of 4.2 seconds" in page.text
            assert [b.text for b in page.blocks][0] == "3 Results"

    def test_ocrd_chunks_record_it(self, parser_name):
        document = _parser(parser_name, FakeOcr()).parse_bytes(pdfs.scanned_pdf(2), "scan.pdf")
        chunks = build_chunker(ChunkingConfig()).chunk(document)
        assert chunks and all(c.metadata["ocr"] is True for c in chunks)
        assert all(c.to_record()["meta_ocr"] is True for c in chunks)

    def test_only_the_scanned_page_of_a_mixed_pdf_is_ocrd(self, parser_name):
        engine = FakeOcr()
        data = pdfs.mixed_pdf(scanned_pages=(2,), pages=3)
        with_ocr = _parser(parser_name, engine).parse_bytes(data, "mixed.pdf")
        without = _parser(parser_name, None, ocr=False).parse_bytes(data, "mixed.pdf")
        assert engine.calls == 1 and with_ocr.metadata["ocr_pages"] == [2]
        assert "median latency" in with_ocr.page(2).text and not without.page(2).text
        for number in (1, 3):
            assert with_ocr.page(number) == without.page(number)
            assert pdfs.PAGE_MARKER.format(n=number) in with_ocr.page(number).text
        chunks = build_chunker(ChunkingConfig()).chunk(with_ocr)
        assert {c.metadata.get("ocr", False) for c in chunks if 2 in c.pages} == {True}
        assert not any(c.metadata.get("ocr") for c in chunks if 2 not in c.pages)

    def test_ocr_that_reads_nothing_is_still_a_scanned_pdf_error(self, parser_name):
        with pytest.raises(ScannedPDFError, match="OCR could not read it") as excinfo:
            _parser(parser_name, FakeOcr(text="  ")).parse_bytes(pdfs.scanned_pdf(3), "s.pdf")
        assert "ocr_languages" in str(excinfo.value)

    def test_a_scanned_pdf_indexes_through_the_pipeline(self, offline_config, tmp_path):
        path = tmp_path / "scan.pdf"
        path.write_bytes(pdfs.scanned_pdf(2))
        pipeline = DenseRAGPipeline(offline_config)
        pipeline._parser = build_parser(offline_config.ingestion, ocr_engine=FakeOcr())
        pipeline.index(path)
        result = pipeline.ask("What was the median latency on the T4 GPU?")
        assert not result.abstained and "4.2 seconds" in result.answer
        assert result.cited_pages


class TestTextPagesAreNeverOcrd:
    def test_a_text_pdf_parses_identically_with_ocr_on(self, parser_name):
        data = pdfs.structured_pdf()
        on = _parser(parser_name, NeverOcr()).parse_bytes(data, "s.pdf")
        off = _parser(parser_name, None, ocr=False).parse_bytes(data, "s.pdf")
        assert _snapshot(on) == _snapshot(off) and "ocr_pages" not in on.metadata

    @pytest.mark.skipif(not BENCHMARK_PDFS, reason="benchmark PDFs not present")
    @pytest.mark.parametrize("path", BENCHMARK_PDFS, ids=lambda p: p.name)
    def test_the_benchmark_pdfs_parse_identically_with_ocr_on(self, path):
        # So OCR on by default cannot move a stored result on its own.
        on = build_parser(IngestionConfig(), ocr_engine=NeverOcr()).parse(path)
        off = build_parser(IngestionConfig(ocr=False)).parse(path)
        assert _snapshot(on) == _snapshot(off)


class TestMissingTesseract:
    def test_missing_pytesseract_says_how_to_install_it(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        with pytest.raises(OCRUnavailableError) as excinfo:
            TesseractOcr().image_to_text(object())
        message = str(excinfo.value)
        assert "pytesseract package is not installed" in message
        assert "apt-get install -y tesseract-ocr" in message
        assert "pip install pytesseract" in message and "--no-ocr" in message

    def test_a_missing_engine_never_yields_an_empty_document(self, monkeypatch, parser_name):
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        parser = build_parser(IngestionConfig(parser=parser_name))
        with pytest.raises(OCRUnavailableError):
            parser.parse_bytes(pdfs.mixed_pdf(scanned_pages=(2,)), "mixed.pdf")

    @pytest.mark.skipif(shutil.which("tesseract") is not None, reason="Tesseract is installed")
    def test_a_missing_binary_says_how_to_install_it(self):
        pytest.importorskip("pytesseract")
        with pytest.raises(OCRUnavailableError, match="Tesseract binary was not found"):
            TesseractOcr().image_to_text(object())


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="the Tesseract binary is absent")
def test_real_tesseract_reads_a_scanned_page():
    pytest.importorskip("pytesseract")
    text = "Scanned pages are read by optical character recognition."
    document = build_parser(IngestionConfig()).parse_bytes(pdfs.scanned_text_pdf(text), "s.pdf")
    assert document.page(1).metadata["ocr"] is True
    recognised = " ".join(document.page(1).text.split()).lower()
    assert "optical character recognition" in recognised

