"""Grounded answer generation (ARCHITECTURE.md section 15).

The generator is the only component that calls a model, and it calls it for one
thing: reading passages and writing a sentence. Everything around that -- which
passages become evidence, what the markers mean, which pages those markers point
to, whether the result counts as an abstention -- is deterministic code
(DD-018).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import GenerationConfig
from ..retrieval.retriever import RetrievedChunk
from .abstention import AbstentionVerdict, detect_abstention
from .backends import LLMBackend, build_backend
from .citations import Citation, CitationResolution, resolve_citations
from .evidence import EvidenceItem, build_evidence, evidence_blocks
from .prompts import ABSTENTION_SENTENCE, SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class GeneratedAnswer:
    """An answer with the evidence and citations behind it."""

    answer: str
    raw_answer: str
    citations: tuple[Citation, ...]
    evidence: tuple[EvidenceItem, ...]
    abstained: bool
    abstention_reason: str
    dropped_markers: tuple[int, ...] = ()
    fabricated_page_mentions: tuple[str, ...] = ()
    backend: str = ""
    latency_s: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def cited_pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for c in self.citations for page in c.pages}))

    def to_record(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "citations": [c.to_record() for c in self.citations],
            "cited_pages": list(self.cited_pages),
            "dropped_markers": list(self.dropped_markers),
            "fabricated_page_mentions": list(self.fabricated_page_mentions),
            "evidence": [e.to_record() for e in self.evidence],
            "backend": self.backend,
            "latency_s": round(self.latency_s, 3),
            **self.metadata,
        }


class GroundedGenerator:
    """Builds the evidence prompt, calls the backend, resolves the citations."""

    def __init__(
        self, config: GenerationConfig | None = None, backend: LLMBackend | None = None
    ) -> None:
        self.config = config or GenerationConfig()
        self.backend = backend or build_backend(self.config)

    def generate(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        evidence = build_evidence(retrieved, self.config)

        if not evidence:
            # Nothing was retrieved, so there is nothing to reason about. The
            # model is not consulted: asking it to answer with no evidence is
            # asking it to answer from memory, which is the failure mode the
            # whole system is built to avoid.
            return self._refusal(
                evidence=(),
                reason="no_evidence",
                latency=time.perf_counter() - started,
            )

        prompt = build_user_prompt(question, evidence_blocks(evidence))
        raw = self.backend.generate(SYSTEM_PROMPT, prompt)

        resolution = resolve_citations(raw, evidence)
        verdict = detect_abstention(resolution.answer)

        if verdict.abstained:
            # A refusal cites nothing. Keeping citations on it would let a
            # citation metric score a system that answered nothing.
            return self._refusal(
                evidence=tuple(evidence),
                reason=verdict.reason,
                latency=time.perf_counter() - started,
                raw=raw,
                resolution=resolution,
            )

        return GeneratedAnswer(
            answer=resolution.answer,
            raw_answer=raw,
            citations=resolution.citations,
            evidence=tuple(evidence),
            abstained=False,
            abstention_reason="",
            dropped_markers=resolution.dropped_markers,
            fabricated_page_mentions=resolution.fabricated_page_mentions,
            backend=self.backend.name,
            latency_s=time.perf_counter() - started,
            metadata={
                "evidence_count": len(evidence),
                "uncited": not resolution.has_citations,
            },
        )

    # -- helpers ------------------------------------------------------------

    def _refusal(
        self,
        evidence: tuple[EvidenceItem, ...],
        reason: str,
        latency: float,
        raw: str | None = None,
        resolution: CitationResolution | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=ABSTENTION_SENTENCE,
            raw_answer=raw if raw is not None else ABSTENTION_SENTENCE,
            citations=(),
            evidence=evidence,
            abstained=True,
            abstention_reason=reason,
            dropped_markers=resolution.dropped_markers if resolution else (),
            fabricated_page_mentions=(
                resolution.fabricated_page_mentions if resolution else ()
            ),
            backend=self.backend.name,
            latency_s=latency,
            metadata={
                "evidence_count": len(evidence),
                "patterns_version": AbstentionVerdict(True).patterns_version,
            },
        )
