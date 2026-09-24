"""Parser behaviour, including the failure modes Phase 1 must handle."""

from __future__ import annotations

import pytest

from src.errors import (
    EmptyDocumentError,
    EncryptedPDFError,
    PDFReadError,
    ScannedPDFError,
)
from src.ingestion.document import Page

from . import pdf_fixtures as pdfs


class TestPageNumbering:
    """1-based page numbers, verified against known page contents."""

    def test_pages_are_numbered_from_one(self, parser):
        document = parser.parse_bytes(pdfs.multipage_pdf(5), "m.pdf")
        assert [p.page_number for p in document.pages] == [1, 2, 3, 4, 5]

    def test_page_content_matches_its_number(self, parser):
        """The off-by-one test: page N must contain marker N and no other."""
        document = parser.parse_bytes(pdfs.multipage_pdf(6), "m.pdf")
        for page in document.pages:
            expected = pdfs.PAGE_MARKER.format(n=page.page_number)
            assert expected in page.text
            others = {
                pdfs.PAGE_MARKER.format(n=n)
                for n in range(1, 7)
                if n != page.page_number
            }
            assert not any(marker in page.text for marker in others)

    def test_page_lookup_is_one_based(self, parser):
        document = parser.parse_bytes(pdfs.multipage_pdf(3), "m.pdf")
        assert pdfs.PAGE_MARKER.format(n=1) in document.page(1).text
        with pytest.raises(KeyError):
            document.page(0)

    def test_page_rejects_zero_based_numbering(self):
        with pytest.raises(ValueError, match="1-based"):
            Page(page_number=0, text="x")


class TestEmptyPages:
    def test_empty_page_is_kept_so_numbering_survives(self, parser):
        document = parser.parse_bytes(pdfs.pdf_with_empty_page(3, 2), "e.pdf")
        assert document.page_count == 3
        assert document.page(2).is_empty
        # Dropping page 2 would renumber page 3, so page 3 must still be page 3.
        assert pdfs.PAGE_MARKER.format(n=3) in document.page(3).text

    def test_non_empty_pages_excludes_the_blank_one(self, parser):
        document = parser.parse_bytes(pdfs.pdf_with_empty_page(3, 2), "e.pdf")
        assert [p.page_number for p in document.non_empty_pages] == [1, 3]


class TestFailureModes:
    def test_scanned_pdf_with_ocr_off_raises_with_an_actionable_message(self, make_parser):
        with pytest.raises(ScannedPDFError) as excinfo:
            make_parser(ocr=False).parse_bytes(pdfs.scanned_pdf(3), "scan.pdf")
        message = str(excinfo.value)
        assert "scanned" in message.lower() or "image-only" in message.lower()
        assert "OCR is switched off" in message  # says why nothing was read
        assert "--no-ocr" in message  # and what to do about it
        assert "3 of 3 pages" in message  # says what it saw

    def test_encrypted_pdf_raises_without_a_password(self, parser):
        with pytest.raises(EncryptedPDFError, match="password"):
            parser.parse_bytes(pdfs.encrypted_pdf(), "enc.pdf")

    def test_encrypted_pdf_opens_with_the_password(self, make_parser):
        parser = make_parser(password="secret")
        document = parser.parse_bytes(pdfs.encrypted_pdf("secret"), "enc.pdf")
        assert "Confidential" in document.page(1).text

    def test_wrong_password_is_reported_as_such(self, make_parser):
        parser = make_parser(password="wrong")
        with pytest.raises(EncryptedPDFError, match="rejected"):
            parser.parse_bytes(pdfs.encrypted_pdf("secret"), "enc.pdf")

    def test_blank_pdf_raises(self, parser):
        with pytest.raises(EmptyDocumentError):
            parser.parse_bytes(pdfs.blank_pdf(2), "blank.pdf")

    def test_missing_file_raises(self, parser, tmp_path):
        with pytest.raises(PDFReadError, match="No such file"):
            parser.parse(tmp_path / "nope.pdf")

    def test_non_pdf_bytes_raise(self, parser):
        with pytest.raises(PDFReadError, match="valid"):
            parser.parse_bytes(b"this is not a pdf", "junk.pdf")


class TestNormalizationThroughTheParser:
    def test_hyphenated_word_is_rejoined_across_lines(self, parser):
        document = parser.parse_bytes(pdfs.hyphenation_pdf(), "h.pdf")
        assert "retrieval" in document.page(1).text
        assert "retrie-" not in document.page(1).text

    def test_capitalised_hyphen_is_preserved(self, parser):
        """`Mixed-\\nCase` is a real hyphenated word, not a broken one."""
        document = parser.parse_bytes(pdfs.hyphenation_pdf(), "h.pdf")
        assert "Mixed-Case" in document.page(1).text

    def test_running_header_and_footer_are_removed(self, parser):
        document = parser.parse_bytes(
            pdfs.multipage_pdf(5, header="ACME Annual Report"), "m.pdf"
        )
        for page in document.pages:
            assert "ACME Annual Report" not in page.text
            assert "Page 1 of 5" not in page.text
        assert "acme annual report" in document.metadata["removed_running_lines"]

    def test_body_text_is_not_removed_as_a_header(self, parser):
        """Every page shares this sentence, but it is not page furniture."""
        document = parser.parse_bytes(
            pdfs.multipage_pdf(5, header="ACME Annual Report"), "m.pdf"
        )
        for page in document.pages:
            assert "The retrieval component scores each candidate" in page.text
            assert pdfs.PAGE_MARKER.format(n=page.page_number) in page.text

    def test_short_documents_keep_their_headers(self, parser):
        """Two repetitions are a coincidence, not a running header."""
        document = parser.parse_bytes(
            pdfs.multipage_pdf(2, header="ACME Annual Report"), "m.pdf"
        )
        assert "ACME Annual Report" in document.page(1).text


class TestDocumentIdentity:
    def test_document_id_is_content_addressed(self, parser):
        data = pdfs.multipage_pdf(3)
        first = parser.parse_bytes(data, "a.pdf")
        second = parser.parse_bytes(data, "b.pdf")
        assert first.document_id == second.document_id

    def test_different_content_gives_a_different_id(self, parser):
        a = parser.parse_bytes(pdfs.multipage_pdf(3), "a.pdf")
        b = parser.parse_bytes(pdfs.multipage_pdf(4), "b.pdf")
        assert a.document_id != b.document_id


class TestStructureExtraction:
    def test_table_becomes_its_own_block(self, structured_document):
        tables = [b for p in structured_document.pages for b in p.blocks if b.is_table]
        assert len(tables) == 1
        assert "| System | Recall | Latency |" in tables[0].text

    def test_table_text_is_not_duplicated_as_prose(self, structured_document):
        page = structured_document.page(2)
        prose = "\n".join(b.text for b in page.blocks if not b.is_table)
        assert "0.83" not in prose

    def test_headings_carry_a_larger_font_size(self, structured_document):
        sizes = {
            b.text: b.font_size
            for p in structured_document.pages
            for b in p.blocks
            if not b.is_table
        }
        assert sizes["1 Introduction"] > sizes[
            next(t for t in sizes if t.startswith("This system answers"))
        ]
