"""Shared parsing machinery (ARCHITECTURE.md section 5).

Everything here is backend-independent: header and footer stripping, hyphen
joining, page assembly, table rendering, and the checks that turn a scanned or
blank PDF into an explicit error. A concrete parser supplies raw blocks with
geometry; this module turns them into a ``Document``.

The split exists because DD-033 requires the PDF library to be replaceable. If
the post-processing lived inside one backend, swapping the backend would mean
reimplementing the part of ingestion that the page-attribution guarantees
actually depend on.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import IngestionConfig
from ..errors import EmptyDocumentError, PDFReadError, ScannedPDFError
from .document import Block, BlockKind, Document, Page, make_document_id
from .normalization import (
    collapse_blank_lines,
    count_running_keys,
    edge_lines,
    join_hyphenated_linebreaks,
    keep_mask,
    normalize_text,
)


@dataclass
class RawBlock:
    """A block as a backend reports it, before normalization is finished."""

    lines: list[str]
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None
    is_bold: bool = False
    is_table: bool = False
    order: int = 0


@dataclass
class RawPage:
    """One page's raw blocks, plus what the scanned-document check needs."""

    page_number: int
    blocks: list[RawBlock] = field(default_factory=list)
    image_count: int = 0
    height: float = 0.0

    def margin_flags(self, margin_fraction: float) -> list[bool]:
        """One flag per line: does this line sit in the top or bottom margin?

        Returns an empty list when no block reports geometry, which tells the
        caller to fall back to positional detection.
        """
        if self.height <= 0 or not any(b.bbox for b in self.blocks):
            return []
        top = self.height * margin_fraction
        bottom = self.height * (1.0 - margin_fraction)
        flags: list[bool] = []
        for block in self.blocks:
            if block.bbox is None:
                in_margin = False
            else:
                in_margin = block.bbox[3] <= top or block.bbox[1] >= bottom
            flags.extend([in_margin] * len(block.lines))
        return flags


class BaseParser(ABC):
    """Turns raw per-page blocks into a ``Document``.

    Subclasses implement :meth:`_extract` and nothing else. Everything that
    affects page attribution is here, so every backend gets the same guarantees.
    """

    name: str = "base"

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()

    # -- backend hook -------------------------------------------------------

    @abstractmethod
    def _extract(self, data: bytes, source_name: str) -> tuple[list[RawPage], dict]:
        """Return the raw pages and the PDF's own metadata.

        Implementations raise ``EncryptedPDFError`` for a protected file and
        ``PDFReadError`` for anything unreadable.
        """

    # -- public -------------------------------------------------------------

    def parse(self, path: str | os.PathLike[str]) -> Document:
        path = os.fspath(path)
        if not os.path.exists(path):
            raise PDFReadError(f"No such file: {path!r}.")
        if not os.path.isfile(path):
            raise PDFReadError(f"Not a file: {path!r}.")
        with open(path, "rb") as handle:
            data = handle.read()
        if not data:
            raise PDFReadError(f"{os.path.basename(path)} is empty (0 bytes).")
        return self.parse_bytes(data, os.path.basename(path))

    def parse_bytes(self, data: bytes, source_name: str) -> Document:
        raw_pages, pdf_metadata = self._extract(data, source_name)
        pages, removed = self._finalize_pages(raw_pages)
        document = Document(
            document_id=make_document_id(data),
            source_name=source_name,
            pages=tuple(pages),
            metadata={
                "parser": self.name,
                "page_count": len(pages),
                "pdf_title": str(pdf_metadata.get("title") or ""),
                "removed_running_lines": sorted(removed),
                "total_chars": sum(len(p.text) for p in pages),
            },
        )
        self._check_extractable(document, raw_pages)
        return document

    # -- assembly -----------------------------------------------------------

    def _finalize_pages(
        self, raw_pages: list[RawPage]
    ) -> tuple[list[Page], set[str]]:
        """Strip running headers and footers, then build immutable pages."""
        pages_lines = [
            [line for block in page.blocks for line in block.lines]
            for page in raw_pages
        ]
        margin = self.config.header_footer_margin_fraction
        page_flags = [raw.margin_flags(margin) for raw in raw_pages]
        candidates = [
            [line for line, flag in zip(lines, flags) if flag]
            if flags
            else edge_lines(lines, self.config)
            for lines, flags in zip(pages_lines, page_flags)
        ]

        running: set[str] = set()
        if self.config.strip_running_headers:
            running = count_running_keys(candidates, len(raw_pages), self.config)

        pages: list[Page] = []
        for raw, flat_lines, flags in zip(raw_pages, pages_lines, page_flags):
            mask = keep_mask(flat_lines, running, self.config, flags or None)
            cursor = 0
            blocks: list[Block] = []
            for raw_block in raw.blocks:
                span = len(raw_block.lines)
                kept = [
                    line
                    for line, keep in zip(raw_block.lines, mask[cursor : cursor + span])
                    if keep
                ]
                cursor += span
                text = "\n".join(kept)
                if self.config.join_hyphenated_linebreaks and not raw_block.is_table:
                    text = join_hyphenated_linebreaks(text)
                text = text.strip()
                if not text:
                    continue
                blocks.append(
                    Block(
                        text=text,
                        kind=BlockKind.TABLE
                        if raw_block.is_table
                        else BlockKind.UNKNOWN,
                        bbox=raw_block.bbox,
                        font_size=raw_block.font_size,
                        is_bold=raw_block.is_bold,
                        order=len(blocks),
                    )
                )

            page_text = "\n\n".join(b.text for b in blocks)
            if self.config.join_hyphenated_linebreaks:
                # Run again over the assembled page. Extractors often emit the
                # two halves of a hyphenated word as separate blocks, so the
                # per-block pass above never sees the seam.
                page_text = join_hyphenated_linebreaks(page_text)
            page_text = collapse_blank_lines(page_text)

            pages.append(
                Page(
                    page_number=raw.page_number,
                    text=page_text,
                    blocks=tuple(blocks),
                    image_count=raw.image_count,
                    metadata={
                        "block_count": len(blocks),
                        "table_count": sum(1 for b in blocks if b.is_table),
                    },
                )
            )
        return pages, running

    # -- failure boundaries -------------------------------------------------

    def _check_extractable(self, document: Document, raw_pages: list[RawPage]) -> None:
        """Fail loudly on a scanned or blank PDF.

        EXPERIMENT_PLAN.md section 2 makes this a Phase 1 exit criterion: a
        scanned PDF must produce a clear error, not garbage.
        """
        cfg = self.config
        page_count = document.page_count
        if page_count == 0:
            raise EmptyDocumentError(f"{document.source_name} contains no pages.")

        text_empty = [p for p in document.pages if len(p.text) < cfg.min_chars_per_page]
        images_by_page = {raw.page_number: raw.image_count for raw in raw_pages}
        image_only = [p for p in text_empty if images_by_page.get(p.page_number, 0) > 0]

        total_chars = document.total_chars
        if (
            image_only
            and len(image_only) / page_count >= cfg.scanned_page_fraction
            and total_chars < cfg.min_document_chars
        ):
            raise ScannedPDFError(
                f"{document.source_name} appears to be a scanned or image-only "
                f"PDF: {len(image_only)} of {page_count} pages contain images but "
                f"no extractable text layer (the whole document yielded "
                f"{total_chars} characters). This system reads text, not pixels, "
                "so indexing it would produce empty or nonsensical answers. Run "
                "OCR first, for example `ocrmypdf input.pdf output.pdf`, then "
                "ingest the OCR'd file."
            )

        if total_chars == 0:
            raise EmptyDocumentError(
                f"{document.source_name} has {page_count} page(s) but no "
                "extractable text and no images. The file may be corrupt, or its "
                "content may be vector graphics only."
            )


# ---------------------------------------------------------------------------
# Helpers shared by the backends
# ---------------------------------------------------------------------------


def normalize_lines(lines: list[str], config: IngestionConfig) -> list[str]:
    """Normalize each line, dropping the ones that end up empty."""
    out: list[str] = []
    for line in lines:
        cleaned = normalize_text(line, config)
        if cleaned.strip():
            out.append(cleaned)
    return out


def mostly_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    """True when ``inner``'s centre lies within ``outer``."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def render_table(rows: list[list[str | None]]) -> str:
    """Render extracted cells as a pipe table.

    A pipe table keeps each value beside its column header on the same line,
    which matters because the chunk is what a model later reads: whitespace
    alignment does not survive tokenization, and a bare list of cells loses
    which column a number belonged to.
    """
    cleaned = [
        [" ".join((cell or "").split()) for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def sort_blocks_into_reading_order(blocks: list[RawBlock]) -> None:
    """Order blocks top-to-bottom, left-to-right, in place.

    Tables are discovered separately from prose and would otherwise appear
    wherever the backend happened to emit them rather than where they sit on
    the page.
    """
    blocks.sort(
        key=lambda b: (
            round(b.bbox[1], 1) if b.bbox else 0.0,
            round(b.bbox[0], 1) if b.bbox else 0.0,
            b.order,
        )
    )
    for index, block in enumerate(blocks):
        block.order = index
