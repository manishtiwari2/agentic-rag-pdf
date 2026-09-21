"""Fixed-size chunking: the structure-blind baseline.

DD-009 claims structure-aware chunking beats splitting every N characters, and
EXPERIMENT_PLAN.md requires that claim to be measured rather than assumed. This
is the control condition, so it is deliberately simple: concatenate the pages,
slide a window, snap to word boundaries so tokens are not cut in half.

``section`` is left as ``None`` on purpose. A baseline that consulted headings
to label its chunks would already be half structure-aware, and the comparison it
exists to anchor would no longer measure what it claims to. The field is present
on every chunk, as DD-010 requires; it is simply empty for this strategy.

Page attribution is still exact. The concatenated text carries an offset table,
so a window straddling a page break reports both pages.
"""

from __future__ import annotations

from ..config import ChunkingConfig
from ..ingestion.document import Document
from .base import Chunk, Chunker
from .segmentation import slice_page_offsets

#: Separator inserted between pages in the concatenated buffer.
_PAGE_SEPARATOR = "\n\n"


class FixedSizeChunker(Chunker):
    """Splits the document into overlapping windows of a fixed character size."""

    name = "fixed"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)

    def chunk(self, document: Document) -> list[Chunk]:
        text, page_offsets = _concatenate(document)
        if not text.strip():
            return []

        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Snap back to a word boundary so a token is not cut in half.
                space = text.rfind(" ", start + 1, end)
                newline = text.rfind("\n", start + 1, end)
                boundary = max(space, newline)
                if boundary > start:
                    end = boundary
            piece = text[start:end]
            if piece.strip():
                pages = _pages_for(page_offsets, start, end)
                chunks.append(
                    self._build(
                        document,
                        len(chunks),
                        piece.strip(),
                        pages,
                        None,
                        kind="text",
                        char_span=[start, end],
                        page_span=[pages[0], pages[-1]],
                    )
                )
            if end >= len(text):
                break
            # Step forward by at least one character: with overlap >= size the
            # window would otherwise never advance. ``validate()`` rejects that
            # configuration, and this keeps the loop safe regardless.
            start = max(start + 1, end - overlap)

        kept = [c for c in chunks if len(c.text) >= self.config.min_chunk_chars]
        if not kept:
            return self._fallback_single_chunk(document)
        return _reindex(kept)


def _concatenate(document: Document) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Join the non-empty pages, recording where each page's text starts."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page in document.pages:
        if page.is_empty:
            continue
        if parts:
            cursor += len(_PAGE_SEPARATOR)
            parts.append(_PAGE_SEPARATOR)
        offsets.append((cursor, page.page_number))
        parts.append(page.text)
        cursor += len(page.text)
    return "".join(parts), tuple(offsets)


def _pages_for(
    page_offsets: tuple[tuple[int, int], ...], start: int, end: int
) -> tuple[int, ...]:
    if not page_offsets:
        return (1,)
    sliced = slice_page_offsets(page_offsets, start, end)
    return tuple(sorted({page for _, page in sliced}))


def _reindex(chunks: list[Chunk]) -> list[Chunk]:
    """Renumber after filtering so indices stay contiguous."""
    from .base import make_chunk_id  # noqa: PLC0415 - local to avoid a cycle

    renumbered: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        if chunk.index == index:
            renumbered.append(chunk)
            continue
        renumbered.append(
            Chunk(
                chunk_id=make_chunk_id(
                    chunk.document_id, str(chunk.metadata.get("strategy", "fixed")), index
                ),
                document_id=chunk.document_id,
                text=chunk.text,
                pages=chunk.pages,
                section=chunk.section,
                index=index,
                metadata=chunk.metadata,
            )
        )
    return renumbered
