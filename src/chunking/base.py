"""The chunk model and the chunker interface.

A ``Chunk`` is the unit that gets embedded, retrieved, shown to the generator
and finally cited. DD-010 requires its metadata to survive all four stages, so
the metadata is part of the type rather than a dictionary that a later stage
might forget to carry forward.

Page attribution is the property most worth protecting. ``pages`` lists every
page the chunk covers, which matters because DD-025 scores retrieval by page and
counts a chunk spanning a page break as a hit for each page it covers.
``page_number`` is the citation anchor -- the first page -- and is derived, not
supplied, so it cannot drift away from ``pages``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import ChunkingConfig
from ..errors import ChunkingError
from ..ingestion.document import Document


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with the metadata needed to cite it."""

    chunk_id: str
    document_id: str
    text: str
    #: Every page this chunk draws text from, ascending. Never empty.
    pages: tuple[int, ...]
    #: Nearest enclosing heading, when the chunker tracks structure.
    section: str | None = None
    #: Position in the document's chunk sequence, from 0.
    index: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pages:
            raise ChunkingError(
                f"Chunk {self.chunk_id} has no page attribution. Every chunk must "
                "record the page it came from (DD-010); a chunk without one "
                "cannot be cited."
            )
        if list(self.pages) != sorted(self.pages):
            raise ChunkingError(
                f"Chunk {self.chunk_id} lists pages out of order: {self.pages}."
            )
        if self.pages[0] < 1:
            raise ChunkingError(
                f"Chunk {self.chunk_id} has page {self.pages[0]}; page numbers are "
                "1-based."
            )

    @property
    def page_number(self) -> int:
        """The citation anchor: the first page this chunk covers."""
        return self.pages[0]

    @property
    def page_span(self) -> tuple[int, int]:
        """First and last page covered, inclusive."""
        return self.pages[0], self.pages[-1]

    @property
    def spans_pages(self) -> bool:
        return self.pages[0] != self.pages[-1]

    def citation_label(self) -> str:
        """Human-readable page reference, e.g. ``page 4`` or ``pages 4-5``."""
        first, last = self.page_span
        return f"page {first}" if first == last else f"pages {first}-{last}"

    def to_record(self) -> dict[str, object]:
        """Flat form for observability logs and saved results."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "pages": list(self.pages),
            "page_span": list(self.page_span),
            "section": self.section,
            "index": self.index,
            "chars": len(self.text),
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }


def make_chunk_id(document_id: str, strategy: str, index: int) -> str:
    """Deterministic chunk identifier.

    Derived from the content-addressed document id, so re-indexing the same PDF
    with the same configuration reproduces the same ids and saved results stay
    comparable across runs.
    """
    return f"{document_id}:{strategy}:{index:05d}"


class Chunker(ABC):
    """Converts a parsed document into retrieval units.

    Implementations must not import the PDF parser: they consume the document
    model only, which is what allows the parser to be swapped.
    """

    #: Short identifier, recorded in chunk ids and in results.
    name: str = "chunker"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

    # -- shared helpers -----------------------------------------------------

    def _build(
        self,
        document: Document,
        index: int,
        text: str,
        pages: tuple[int, ...],
        section: str | None = None,
        **metadata: object,
    ) -> Chunk:
        return Chunk(
            chunk_id=make_chunk_id(document.document_id, self.name, index),
            document_id=document.document_id,
            text=text,
            pages=pages,
            section=section,
            index=index,
            metadata={"strategy": self.name, "source_name": document.source_name,
                      **metadata},
        )

    def _fallback_single_chunk(self, document: Document) -> list[Chunk]:
        """Emit the whole document as one chunk.

        Used when the configured minimum chunk length would otherwise discard
        every chunk of a very short document. Returning nothing would make the
        document silently unanswerable, which is worse than one short chunk.
        """
        text = "\n\n".join(p.text for p in document.non_empty_pages).strip()
        if not text:
            return []
        pages = tuple(p.page_number for p in document.non_empty_pages)
        return [
            self._build(document, 0, text, pages, None, reason="short_document")
        ]


def build_chunker(config: ChunkingConfig) -> Chunker:
    """Construct the chunker named by the configuration (DD-017)."""
    from .fixed import FixedSizeChunker  # noqa: PLC0415 - avoids a cycle
    from .structure import StructureAwareChunker  # noqa: PLC0415

    if config.strategy == "structure":
        return StructureAwareChunker(config)
    if config.strategy == "fixed":
        return FixedSizeChunker(config)
    raise ChunkingError(  # pragma: no cover - guarded by the Literal type
        f"Unknown chunking strategy {config.strategy!r}. "
        "Available: 'structure', 'fixed'."
    )
