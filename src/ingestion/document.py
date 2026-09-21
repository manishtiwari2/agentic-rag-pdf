"""The internal document representation (ARCHITECTURE.md section 4).

This is the boundary between the PDF world and everything downstream. Chunking,
retrieval and generation see ``Document``/``Page``/``Block`` and never a PDF
library object, so the parser can be replaced without touching them.

**Page numbers are 1-based throughout.** PDF libraries index from 0 and users
count from 1; the conversion happens exactly once, in the parser, and
``Page.__post_init__`` refuses anything below 1 so a regression fails loudly
instead of shifting every citation in the system by one page.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class BlockKind(str, Enum):
    """Coarse role of a text block.

    The parser emits ``UNKNOWN`` for everything except tables it can detect
    directly. Classification into headings and paragraphs is the chunker's job
    (``chunking/segmentation.py``), so a parser that cannot report font
    information still produces usable blocks.
    """

    UNKNOWN = "unknown"
    TABLE = "table"


@dataclass(frozen=True)
class Block:
    """A contiguous run of text as laid out on the page."""

    text: str
    kind: BlockKind = BlockKind.UNKNOWN
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None
    is_bold: bool = False
    #: Reading order within the page, starting at 0.
    order: int = 0

    @property
    def is_table(self) -> bool:
        return self.kind is BlockKind.TABLE


@dataclass(frozen=True)
class Page:
    """One page of a document.

    ``text`` is the normalized full text of the page and is what the fixed-size
    chunker consumes. ``blocks`` carry layout detail and are what the
    structure-aware chunker consumes. Both describe the same page, so a chunk
    built from either route reports the same page number.
    """

    page_number: int
    text: str
    blocks: tuple[Block, ...] = ()
    #: Images found on the page. Used only for scanned-document detection.
    image_count: int = 0
    char_count_raw: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(
                f"page_number must be 1-based, got {self.page_number}. PDF "
                "libraries index pages from 0; convert once at the parser "
                "boundary rather than anywhere downstream."
            )

    @property
    def is_empty(self) -> bool:
        """True when the page has no usable text after normalization.

        Empty pages are kept in the document rather than dropped. Dropping them
        would renumber every page that follows.
        """
        return not self.text.strip()


@dataclass(frozen=True)
class Document:
    """A parsed PDF."""

    document_id: str
    source_name: str
    pages: tuple[Page, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Page]:
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def non_empty_pages(self) -> tuple[Page, ...]:
        return tuple(p for p in self.pages if not p.is_empty)

    def page(self, page_number: int) -> Page:
        """Return the page with the given 1-based number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        raise KeyError(
            f"Page {page_number} not found in {self.source_name!r} "
            f"(pages 1..{self.page_count})."
        )


def make_document_id(data: bytes) -> str:
    """Deterministic id derived from the file's bytes.

    Content-addressed rather than random so that chunk ids, and therefore saved
    results, are reproducible across runs and sessions.
    """
    return "doc_" + hashlib.sha256(data).hexdigest()[:12]
