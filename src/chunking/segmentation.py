"""Turn a parsed document into classified, page-attributed segments.

This is the structure analysis the structure-aware chunker runs on, and it is
deterministic code by design (DD-018). Deciding whether a line is a heading is a
layout question -- font size, boldness, numbering, length -- not a semantic one,
so it needs no model, and using one here would make chunking slow, unrepeatable
and impossible to unit test.

Every segment carries ``page_offsets``: the character offset inside the segment
at which each page's contribution starts. That is what lets a chunk built from
part of a segment report exactly the pages that part came from, instead of
inheriting the whole segment's span.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ..config import ChunkingConfig
from ..ingestion.document import Block, Document
from ..ingestion.normalization import join_hyphenated_linebreaks


class SegmentKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"


@dataclass(frozen=True)
class Segment:
    """A classified run of text with exact page attribution."""

    text: str
    kind: SegmentKind
    #: ``(char_offset, page_number)`` pairs, ascending by offset. The first
    #: entry always has offset 0.
    page_offsets: tuple[tuple[int, int], ...]
    #: Heading depth from 1; 0 for everything else.
    level: int = 0

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for _, page in self.page_offsets}))

    @property
    def first_page(self) -> int:
        return self.page_offsets[0][1]


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")
_NAMED_SECTION = re.compile(
    r"^(chapter|section|appendix|part|figure|table)\b[\s:]*\d*", re.IGNORECASE
)
_BULLET = re.compile(r"^\s*(?:[-•‣●·*]|\(?[a-z0-9]{1,3}[.)])\s+")
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z0-9])')
_TERMINAL = (".", "!", "?", ":", ";", '"', "'", ")")

#: Headings are short. Anything longer is a paragraph whatever its font.
_MAX_HEADING_CHARS = 120
_MAX_HEADING_WORDS = 18
#: A block must be this much larger than body text to count as a heading on
#: size alone.
_HEADING_SIZE_RATIO = 1.15


def body_font_size(document: Document) -> float | None:
    """The document's dominant font size, weighted by how much text uses it.

    Weighting by character count rather than by block count matters: a report
    with fifty one-line headings and ten dense paragraphs has more heading
    blocks than body blocks, and an unweighted mode would conclude that the
    heading size *is* the body size and then classify nothing as a heading.
    """
    weights: Counter[float] = Counter()
    for page in document.pages:
        for block in page.blocks:
            if block.font_size is None or block.is_table:
                continue
            weights[round(block.font_size, 1)] += len(block.text)
    if not weights:
        return None
    return weights.most_common(1)[0][0]


def is_heading(block: Block, body_size: float | None) -> bool:
    """Whether a block reads as a section heading."""
    text = block.text.strip()
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if text.count("\n") >= 2:  # headings do not run to three lines
        return False
    words = text.split()
    if len(words) > _MAX_HEADING_WORDS:
        return False

    numbered = bool(_NUMBERED.match(text)) or bool(_NAMED_SECTION.match(text))
    larger = (
        block.font_size is not None
        and body_size is not None
        and block.font_size >= body_size * _HEADING_SIZE_RATIO
    )
    # A line ending in a full stop is a sentence, unless its typography says
    # otherwise loudly enough.
    if text.endswith((".", "!", "?")) and not larger:
        return False
    if larger:
        return True
    if block.is_bold and len(words) <= 12:
        return True
    if numbered and len(words) <= 15:
        return True
    return text.isupper() and 1 <= len(words) <= 12


def heading_level(text: str, block: Block, body_size: float | None) -> int:
    """Depth of a heading, used to maintain the section path."""
    match = _NUMBERED.match(text.strip())
    if match:
        return match.group(1).count(".") + 1
    if (
        block.font_size is not None
        and body_size is not None
        and block.font_size >= body_size * 1.4
    ):
        return 1
    return 2


def is_list(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    bulleted = sum(1 for line in lines if _BULLET.match(line))
    return bulleted >= max(2, len(lines) // 2)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries.

    Deliberately simple. It is used only to choose a split point inside an
    oversized paragraph and to pick an overlap tail, so an occasional bad split
    at an abbreviation costs a slightly awkward chunk boundary, not a wrong
    answer.
    """
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_document(document: Document, config: ChunkingConfig) -> list[Segment]:
    """Classify every block of every page, in reading order."""
    body_size = body_font_size(document)
    segments: list[Segment] = []

    for page in document.pages:
        if page.is_empty:
            continue
        for block in page.blocks:
            segments.extend(_segment_block(block, page.page_number, body_size))

    if config.join_paragraphs_across_pages:
        segments = _join_across_pages(segments)
    return segments


def _segment_block(
    block: Block, page_number: int, body_size: float | None
) -> list[Segment]:
    text = block.text.strip()
    if not text:
        return []

    if block.is_table:
        return [
            Segment(text=text, kind=SegmentKind.TABLE, page_offsets=((0, page_number),))
        ]
    if is_heading(block, body_size):
        return [
            Segment(
                text=" ".join(text.split()),
                kind=SegmentKind.HEADING,
                page_offsets=((0, page_number),),
                level=heading_level(text, block, body_size),
            )
        ]
    if is_list(text):
        return [
            Segment(text=text, kind=SegmentKind.LIST, page_offsets=((0, page_number),))
        ]

    # A block may hold several paragraphs separated by blank lines. Within one
    # paragraph, line breaks are wrapping artifacts and become spaces.
    segments: list[Segment] = []
    for raw in re.split(r"\n\s*\n", text):
        joined = " ".join(raw.split())
        if joined:
            segments.append(
                Segment(
                    text=joined,
                    kind=SegmentKind.PARAGRAPH,
                    page_offsets=((0, page_number),),
                )
            )
    return segments


def _join_across_pages(segments: list[Segment]) -> list[Segment]:
    """Rejoin a paragraph interrupted by a page break.

    This is the case that produces off-by-one citations. Left alone it yields
    two half-sentences attributed to one page each; the second half, retrieved
    on its own, is cited to the page where the sentence ends rather than where
    it started.
    """
    merged: list[Segment] = []
    for segment in segments:
        if (
            merged
            and segment.kind is SegmentKind.PARAGRAPH
            and merged[-1].kind is SegmentKind.PARAGRAPH
            and _continues(merged[-1], segment)
        ):
            merged[-1] = _merge(merged[-1], segment)
            continue
        merged.append(segment)
    return merged


def _continues(previous: Segment, nxt: Segment) -> bool:
    """Does ``nxt`` continue ``previous`` across a page break?"""
    if previous.page_offsets[-1][1] >= nxt.first_page:
        return False  # same page: the blank line between them was deliberate
    tail = previous.text.rstrip()
    head = nxt.text.lstrip()
    if not tail or not head:
        return False
    if tail.endswith(_TERMINAL):
        return False
    return head[0].islower() or tail.endswith("-")


def _merge(previous: Segment, nxt: Segment) -> Segment:
    joined = f"{previous.text}\n{nxt.text}"
    joined = join_hyphenated_linebreaks(joined)
    # The seam was a newline only so the hyphen rule could see it; a paragraph
    # is one line of text.
    offset_shift = len(joined) - len(nxt.text)
    joined = joined.replace("\n", " ")
    return Segment(
        text=joined,
        kind=SegmentKind.PARAGRAPH,
        page_offsets=previous.page_offsets + ((offset_shift, nxt.first_page),),
        level=0,
    )


# ---------------------------------------------------------------------------
# Page attribution for slices
# ---------------------------------------------------------------------------


def last_sentence_boundary_before(text: str, limit: int) -> int:
    """Offset of the last sentence boundary at or before ``limit``, else -1."""
    best = -1
    for match in _SENTENCE_END.finditer(text, 0, limit):
        best = match.end()
    return best


def first_sentence_boundary_after(text: str, position: int) -> int:
    """Offset of the first sentence boundary at or after ``position``, else -1."""
    match = _SENTENCE_END.search(text, position)
    return match.end() if match else -1


def slice_page_offsets(
    page_offsets: tuple[tuple[int, int], ...], start: int, end: int
) -> tuple[tuple[int, int], ...]:
    """Page offsets for the substring ``[start, end)`` of a segment."""
    result: list[tuple[int, int]] = []
    for offset, page in page_offsets:
        if offset <= start:
            if result:
                result[0] = (0, page)
            else:
                result.append((0, page))
        elif offset < end:
            result.append((offset - start, page))
    if not result:
        result.append((0, page_offsets[0][1]))
    return tuple(result)


def slice_segment(segment: Segment, start: int, end: int) -> Segment:
    """A sub-segment carrying exactly the pages its text came from."""
    return Segment(
        text=segment.text[start:end],
        kind=segment.kind,
        page_offsets=slice_page_offsets(segment.page_offsets, start, end),
        level=segment.level,
    )
