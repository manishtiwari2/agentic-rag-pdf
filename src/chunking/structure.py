"""Structure-aware chunking (DD-009).

Chunks follow the document's own boundaries instead of a character count:

* a heading starts a new chunk, and leads the chunk it starts, so the retrieved
  passage carries the context a reader would have had;
* a table is emitted as its own chunk, never merged with prose and never split,
  because half a table is worse than no table -- the rows survive without the
  header that says what the numbers mean;
* a paragraph is never split unless it alone exceeds the chunk size, and then
  at a sentence boundary;
* overlap is applied only when a chunk was closed because it filled up. Carrying
  a tail across a heading boundary would label text with the wrong section, and
  the section label is metadata DD-010 requires to stay correct.

Page attribution is exact rather than approximate: each piece of text carries
the offsets at which each page's contribution begins, so a chunk assembled from
the end of page 3 and the start of page 4 reports both.
"""

from __future__ import annotations

from ..config import ChunkingConfig
from ..ingestion.document import Document
from .base import Chunk, Chunker
from .segmentation import (
    Segment,
    SegmentKind,
    first_sentence_boundary_after,
    last_sentence_boundary_before,
    segment_document,
    slice_segment,
)


class StructureAwareChunker(Chunker):
    """Chunks along headings, paragraphs and tables."""

    name = "structure"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)

    def chunk(self, document: Document) -> list[Chunk]:
        segments = segment_document(document, self.config)
        if not segments:
            return self._fallback_single_chunk(document)

        state = _BuildState(self.config)
        for segment in segments:
            if segment.kind is SegmentKind.TABLE:
                state.flush()
                state.emit_table(segment)
                continue
            if segment.kind is SegmentKind.HEADING:
                state.flush()
                state.push_heading(segment)
                continue
            for piece in self._split_oversized(segment):
                state.add(piece)
        state.flush()

        chunks = self._materialize(document, state.pending)
        return chunks or self._fallback_single_chunk(document)

    # -- splitting ----------------------------------------------------------

    def _split_oversized(self, segment: Segment) -> list[Segment]:
        """Split a segment that is too large to ever fit in a chunk.

        Terminates by construction: each step either advances to a sentence
        boundary, to a word boundary, or -- if a single unbroken run exceeds the
        chunk size -- by exactly ``chunk_size`` characters.
        """
        size = self.config.chunk_size
        text = segment.text
        if len(text) <= size:
            return [segment]

        pieces: list[Segment] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = last_sentence_boundary_before(text[start:end], end - start)
                if boundary > 0:
                    end = start + boundary
                else:
                    space = text.rfind(" ", start + 1, end)
                    if space > start:
                        end = space + 1
            if end <= start:  # unbreakable run: cut it
                end = min(start + size, len(text))
            pieces.append(slice_segment(segment, start, end))
            start = end
        return pieces

    # -- assembly -----------------------------------------------------------

    def _materialize(
        self, document: Document, pending: list["_PendingChunk"]
    ) -> list[Chunk]:
        coalesced = _coalesce_small(pending, self.config)
        chunks: list[Chunk] = []
        for index, item in enumerate(coalesced):
            text = item.text()
            if not text.strip():
                continue
            chunks.append(
                self._build(
                    document,
                    len(chunks),
                    text,
                    item.pages(),
                    item.section,
                    kind=item.kind,
                    section_path=item.section_path,
                    page_span=list(item.page_span()),
                )
            )
        return chunks


class _PendingChunk:
    """A chunk under construction, before ids and ordering are assigned."""

    def __init__(
        self,
        pieces: list[Segment],
        section: str | None,
        section_path: str,
        kind: str,
    ) -> None:
        self.pieces = pieces
        self.section = section
        self.section_path = section_path
        self.kind = kind

    def text(self) -> str:
        return "\n\n".join(piece.text.strip() for piece in self.pieces if piece.text.strip())

    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for piece in self.pieces for page in piece.pages}))

    def page_span(self) -> tuple[int, int]:
        pages = self.pages()
        return pages[0], pages[-1]

    @property
    def chars(self) -> int:
        return len(self.text())


class _BuildState:
    """Accumulates segments into pending chunks, tracking the section stack."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.pending: list[_PendingChunk] = []
        self.buffer: list[Segment] = []
        self.buffer_chars = 0
        self.stack: list[tuple[int, str]] = []

    # -- section tracking ---------------------------------------------------

    @property
    def section(self) -> str | None:
        return self.stack[-1][1] if self.stack else None

    @property
    def section_path(self) -> str:
        return " > ".join(title for _, title in self.stack)

    def push_heading(self, segment: Segment) -> None:
        while self.stack and self.stack[-1][0] >= segment.level:
            self.stack.pop()
        self.stack.append((segment.level, segment.text))
        # The heading leads the chunk it opens: a passage retrieved without its
        # heading reads as if it were about whatever preceded it.
        self.buffer.append(segment)
        self.buffer_chars += len(segment.text)

    # -- accumulation -------------------------------------------------------

    def add(self, segment: Segment) -> None:
        addition = len(segment.text)
        if self.buffer and self.buffer_chars + addition > self.config.chunk_size:
            self.flush(size_triggered=True)
        self.buffer.append(segment)
        self.buffer_chars += addition

    def emit_table(self, segment: Segment) -> None:
        text = segment.text
        if len(text) > self.config.max_table_chars:
            text = text[: self.config.max_table_chars].rstrip() + "\n[table truncated]"
            segment = Segment(
                text=text, kind=segment.kind, page_offsets=segment.page_offsets
            )
        self.pending.append(
            _PendingChunk([segment], self.section, self.section_path, "table")
        )

    def flush(self, size_triggered: bool = False) -> None:
        if not self.buffer:
            return
        # A buffer holding nothing but headings is not a chunk; keep the
        # headings so they lead the next chunk that has content.
        if all(piece.kind is SegmentKind.HEADING for piece in self.buffer):
            return

        pieces = list(self.buffer)
        self.pending.append(
            _PendingChunk(pieces, self.section, self.section_path, "text")
        )
        self.buffer = []
        self.buffer_chars = 0

        if size_triggered:
            tail = _overlap_tail(pieces[-1], self.config.chunk_overlap)
            if tail is not None:
                self.buffer.append(tail)
                self.buffer_chars = len(tail.text)


def _overlap_tail(segment: Segment, overlap: int) -> Segment | None:
    """The trailing text of ``segment`` to repeat in the next chunk.

    Sliced from the segment rather than copied, so the repeated text keeps the
    page attribution of where it actually came from.
    """
    if overlap <= 0:
        return None
    text = segment.text
    if not text.strip():
        return None
    if len(text) <= overlap:
        return segment

    start = len(text) - overlap
    boundary = first_sentence_boundary_after(text, start)
    if boundary != -1 and boundary < len(text):
        start = boundary
    else:
        space = text.find(" ", start)
        start = space + 1 if space != -1 else start
    if start >= len(text):
        return None
    return slice_segment(segment, start, len(text))


def _coalesce_small(
    pending: list[_PendingChunk], config: ChunkingConfig
) -> list[_PendingChunk]:
    """Fold undersized chunks into a neighbour instead of discarding them.

    Dropping them would be simpler and would quietly remove content from the
    index -- a one-line answer sitting alone under its own heading is exactly
    the kind of chunk that falls below the threshold.

    Tables are never folded in either direction: merging one back into prose
    would undo the reason it was given its own chunk.
    """
    limit = config.min_chunk_chars
    if limit <= 0:
        return pending

    result: list[_PendingChunk] = []
    for item in pending:
        if item.chars >= limit or item.kind == "table":
            result.append(item)
            continue
        previous = result[-1] if result else None
        if (
            previous is not None
            and previous.kind != "table"
            and previous.chars + item.chars <= config.chunk_size * 1.5
        ):
            previous.pieces.extend(item.pieces)
        else:
            result.append(item)
    return result
