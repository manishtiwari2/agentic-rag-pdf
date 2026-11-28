"""Turn retrieved chunks into a numbered evidence list.

Numbering is positional and stable: the rank-1 chunk is always ``[C1]``. The
mapping from marker to chunk lives here, in code, and is what ``citations.py``
later reads to attach pages.

Two things happen on the way in. Exact duplicates are dropped, because the same
passage appearing twice reads to the model like two independent sources for the
same claim. And the list is trimmed to a character budget, lowest-ranked first,
so a long document cannot push the question out of the model's context window.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..chunking.base import Chunk
from ..config import GenerationConfig
from ..retrieval.retriever import RetrievedChunk
from .prompts import format_evidence_block


@dataclass(frozen=True)
class EvidenceItem:
    """One numbered block of evidence, as the model will see it."""

    number: int
    text: str
    retrieved: RetrievedChunk
    truncated: bool = False

    @property
    def marker(self) -> str:
        return f"C{self.number}"

    @property
    def chunk(self) -> Chunk:
        return self.retrieved.chunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def pages(self) -> tuple[int, ...]:
        return self.chunk.pages

    def to_record(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "truncated": self.truncated,
            "text": self.text,
            **self.retrieved.to_record(),
        }


def build_evidence(
    retrieved: list[RetrievedChunk], config: GenerationConfig
) -> list[EvidenceItem]:
    """Number, deduplicate and budget the retrieved chunks."""
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    used = 0
    strict = config.strict_evidence_budget

    for candidate in retrieved:
        body = candidate.chunk.text.strip() if strict else candidate.chunk.text
        key = " ".join(body.split()).lower()
        if key in seen or (strict and not key):
            continue
        seen.add(key)

        number = len(items) + 1
        overhead = 0
        if strict:
            overhead = len(format_evidence_block(number, "")) + (2 if items else 0)
        available = config.max_context_chars - used - overhead
        if strict and available <= 0:
            break
        limit = min(config.max_evidence_chars, available) if strict and not items else config.max_evidence_chars
        text, truncated = _truncate(body, limit, strict=strict)
        if (strict or items) and len(text) > available:
            break
        used += overhead + len(text)
        items.append(
            EvidenceItem(
                number=number,
                text=text,
                retrieved=candidate,
                truncated=truncated,
            )
        )
    return items


def evidence_blocks(items: list[EvidenceItem]) -> list[tuple[int, str]]:
    """The ``(number, text)`` pairs the prompt builder renders.

    Note what is not returned: no page, no chunk id, no section. That omission
    is the mechanism that stops the model writing a page number.
    """
    return [(item.number, item.text) for item in items]


def _truncate(text: str, limit: int, *, strict: bool) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    suffix = " [...]"
    available = limit - len(suffix) if strict else limit
    if available <= 0:
        return text[:limit].rstrip(), True
    cut = text.rfind(" ", 0, available)
    if cut <= 0:
        cut = available
    return text[:cut].rstrip() + suffix, True
