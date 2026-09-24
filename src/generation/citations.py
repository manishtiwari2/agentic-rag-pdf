"""Deterministic citation resolution (DD-018).

The model emits markers. Code turns markers into pages. The model never writes a
page number, and no page number it could write would be used if it did.

Out-of-range markers are **dropped, never guessed**. If the model writes ``[C7]``
against a five-item evidence list, the honest reading is that the claim is
unsupported, and there is no defensible way to pick which of the five it meant.
Mapping it to the nearest valid marker, or to the top hit, would manufacture a
citation that points at a page the model never saw a claim from -- which is
exactly the failure the citation system exists to prevent. Dropped markers are
counted and reported so the rate is visible rather than silently absorbed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import EvidenceItem

#: A bracketed group short enough to be a marker list.
_BRACKET = re.compile(r"\[([^\[\]]{1,60})\]")
#: The group's contents must be nothing but marker references, so prose in
#: brackets -- "[see the appendix]" -- is left alone.
_MARKER_GROUP = re.compile(r"^\s*[Cc]?\s*\d+\s*(?:[,;]\s*[Cc]?\s*\d+\s*)*$")
_NUMBER = re.compile(r"\d+")
_PREFIX = re.compile(r"[Cc]")
#: Page references the model invented. It was shown no page numbers, so any of
#: these is fabricated. Counted for reporting, not trusted.
_FABRICATED_PAGE = re.compile(r"\b(?:on|see|in|at|from)?\s*pages?\s*\d+", re.IGNORECASE)

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Citation:
    """A resolved marker, with the page it actually points to."""

    marker: str
    evidence_number: int
    chunk_id: str
    document_id: str
    pages: tuple[int, ...]
    section: str | None = None

    @property
    def page_number(self) -> int:
        return self.pages[0]

    @property
    def page_span(self) -> tuple[int, int]:
        return self.pages[0], self.pages[-1]

    @property
    def label(self) -> str:
        first, last = self.page_span
        return f"page {first}" if first == last else f"pages {first}-{last}"

    def to_record(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "evidence_number": self.evidence_number,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "pages": list(self.pages),
            "page_number": self.page_number,
            "section": self.section,
            "label": self.label,
        }


@dataclass(frozen=True)
class CitationResolution:
    """The outcome of mapping an answer's markers onto evidence."""

    answer: str
    raw_answer: str
    citations: tuple[Citation, ...] = ()
    dropped_markers: tuple[int, ...] = ()
    fabricated_page_mentions: tuple[str, ...] = field(default=())

    @property
    def cited_pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for c in self.citations for page in c.pages}))

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    def to_record(self) -> dict[str, object]:
        return {
            "citations": [c.to_record() for c in self.citations],
            "cited_pages": list(self.cited_pages),
            "dropped_markers": list(self.dropped_markers),
            "fabricated_page_mentions": list(self.fabricated_page_mentions),
        }


def resolve_citations(
    raw_answer: str, evidence: list[EvidenceItem]
) -> CitationResolution:
    """Map the answer's markers to pages, dropping the ones that do not resolve.

    Recognises ``[C1]``, ``[C1, C2]``, ``[C1;C2]``, ``[c1]`` and the bare ``[1]``
    that small models produce when they drop the prefix. Bracketed prose --
    ``[see the appendix]`` -- is left alone.

    A bare group is a marker only if that exact bracketed token is not already
    in the evidence text. Source text carries its own bibliography references
    ("shrinking[3]"), and an answer that copies one is quoting, not citing
    block 3 (DD-062). A prefixed ``[C3]`` never occurs in source text, so it is
    always a marker.
    """
    by_number = {item.number: item for item in evidence}
    source_text = "\n".join(item.text for item in evidence)
    citations: list[Citation] = []
    dropped: list[int] = []
    seen: set[int] = set()

    def rewrite(match: re.Match[str]) -> str:
        inner = match.group(1)
        if not _MARKER_GROUP.match(inner):
            return match.group(0)
        if not _PREFIX.search(inner) and match.group(0) in source_text:
            return match.group(0)
        valid: list[int] = []
        for number in (int(n) for n in _NUMBER.findall(inner)):
            item = by_number.get(number)
            if item is None:
                dropped.append(number)
                continue
            valid.append(number)
            if number in seen:
                continue
            seen.add(number)
            citations.append(
                Citation(
                    marker=item.marker,
                    evidence_number=number,
                    chunk_id=item.chunk_id,
                    document_id=item.chunk.document_id,
                    pages=item.pages,
                    section=item.chunk.section,
                )
            )
        if not valid:
            # Remove the group entirely: leaving a dangling [C7] in the answer
            # shows the reader a citation that resolves to nothing.
            return ""
        return "[" + ", ".join(f"C{n}" for n in valid) + "]"

    cleaned = _BRACKET.sub(rewrite, raw_answer)
    cleaned = _tidy(cleaned)

    return CitationResolution(
        answer=cleaned,
        raw_answer=raw_answer,
        citations=tuple(citations),
        dropped_markers=tuple(dropped),
        fabricated_page_mentions=tuple(_FABRICATED_PAGE.findall(raw_answer)),
    )


def _tidy(text: str) -> str:
    """Repair the spacing left behind by a removed marker group."""
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
