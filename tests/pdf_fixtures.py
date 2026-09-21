"""Synthetic PDFs built at test time.

The benchmark corpus does not exist yet (Phase 0), and checking binary fixtures
into the repository would make it impossible to see what a failing test is
actually looking at. These builders generate small PDFs whose content is known
exactly, which is what makes an assertion like "the chunk containing PAGE 7's
marker reports page_number == 7" meaningful.
"""

from __future__ import annotations

import struct
import zlib

import pymupdf

#: Every page of ``multipage_pdf`` carries this marker with its own number, so a
#: page-attribution error shows up as a mismatch rather than as a vague failure.
PAGE_MARKER = "PAGEMARKER{n:02d}"

_BODY = (
    "The retrieval component scores each candidate passage against the query "
    "and returns the highest scoring passages in rank order. This paragraph "
    "exists to give the page enough text to survive the empty-page check and "
    "to produce a chunk of realistic size."
)


def _new() -> pymupdf.Document:
    return pymupdf.open()


def _write(doc: pymupdf.Document) -> bytes:
    data = doc.tobytes()
    doc.close()
    return data


def multipage_pdf(pages: int = 5, header: str | None = None) -> bytes:
    """A plain multi-page document with a unique marker per page."""
    doc = _new()
    for index in range(pages):
        page = doc.new_page()
        if header is not None:
            # Inside the top margin band, where page furniture lives.
            page.insert_text((72, 36), header, fontsize=9)
        page.insert_text((72, 150), PAGE_MARKER.format(n=index + 1), fontsize=14)
        page.insert_textbox(pymupdf.Rect(72, 170, 520, 380), _BODY, fontsize=11)
        if header is not None:
            page.insert_text((72, 800), f"Page {index + 1} of {pages}", fontsize=9)
    return _write(doc)


def pdf_with_empty_page(total: int = 3, empty_index: int = 2) -> bytes:
    """A document where page ``empty_index`` (1-based) has no text at all."""
    doc = _new()
    for index in range(total):
        page = doc.new_page()
        if index + 1 == empty_index:
            continue
        page.insert_text((72, 150), PAGE_MARKER.format(n=index + 1), fontsize=14)
        page.insert_textbox(pymupdf.Rect(72, 170, 520, 380), _BODY, fontsize=11)
    return _write(doc)


def encrypted_pdf(password: str = "secret") -> bytes:
    """A password-protected document."""
    doc = _new()
    page = doc.new_page()
    page.insert_text((72, 80), "Confidential contents.", fontsize=12)
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw=password,
        user_pw=password,
        permissions=pymupdf.PDF_PERM_ACCESSIBILITY,
    )
    doc.close()
    return data


def _grey_png(width: int = 8, height: int = 8, value: int = 200) -> bytes:
    """A valid greyscale PNG, built here so the fixture needs no image library."""
    raw = b"".join(b"\x00" + bytes([value]) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def scanned_pdf(pages: int = 3) -> bytes:
    """An image-only document: pages carry a picture and no text layer."""
    # An image scaled to fill the page, and no text drawn at all, which is what
    # a scan looks like to a text extractor.
    png = _grey_png()
    doc = _new()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png)
    return _write(doc)


def blank_pdf(pages: int = 2) -> bytes:
    """Pages with neither text nor images."""
    doc = _new()
    for _ in range(pages):
        doc.new_page()
    return _write(doc)


def structured_pdf() -> bytes:
    """A document with headings, paragraphs, a table, and a page-spanning paragraph."""
    doc = _new()

    page1 = doc.new_page()
    page1.insert_text((72, 70), "1 Introduction", fontsize=18)
    page1.insert_textbox(
        pymupdf.Rect(72, 90, 520, 200),
        "This system answers questions about a single uploaded document. "
        "It retrieves passages and cites the pages they came from.",
        fontsize=11,
    )
    page1.insert_text((72, 230), "2 Method", fontsize=18)
    page1.insert_textbox(
        pymupdf.Rect(72, 250, 520, 380),
        "The method embeds every chunk with a bi-encoder and searches the "
        "resulting vectors with an inner-product index. The retrieved passages "
        "are numbered and handed to the generator as evidence, and the sentence "
        "continues onto the following page because it was not",
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_textbox(
        pymupdf.Rect(72, 70, 520, 140),
        "finished before the page break, which is the case that produces "
        "off-by-one citations when a chunker gets page spans wrong.",
        fontsize=11,
    )
    page2.insert_text((72, 170), "3 Results", fontsize=18)
    page2.insert_textbox(
        pymupdf.Rect(72, 190, 520, 260),
        "Accuracy improved on every configuration that was measured.",
        fontsize=11,
    )
    # A ruled table. PyMuPDF's table finder keys off the ruling lines.
    rows = [
        ["System", "Recall", "Latency"],
        ["Dense", "0.71", "4.2 s"],
        ["Hybrid", "0.83", "5.1 s"],
    ]
    top, left, row_h, col_w = 290.0, 72.0, 24.0, 120.0
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                left + c * col_w,
                top + r * row_h,
                left + (c + 1) * col_w,
                top + (r + 1) * row_h,
            )
            page2.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page2.insert_text((rect.x0 + 4, rect.y1 - 8), cell, fontsize=10)

    return _write(doc)


def hyphenation_pdf() -> bytes:
    """A page where a word is split by a hyphen at a line break.

    Ligature handling is unit-tested on strings instead: the base-14 fonts
    available to the fixture builder have no ligature glyph, so a ligature
    written here would come back as a replacement character and the test would
    be checking the fixture rather than the normalizer.
    """
    doc = _new()
    page = doc.new_page()
    page.insert_text((72, 150), "The system performs retrie-", fontsize=12)
    page.insert_text((72, 170), "val over the indexed chunks, and keeps", fontsize=12)
    page.insert_text((72, 190), "Mixed-", fontsize=12)
    page.insert_text((72, 210), "Case hyphens intact.", fontsize=12)
    return _write(doc)
