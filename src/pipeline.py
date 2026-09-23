"""The two retrieve-then-generate baselines (ARCHITECTURE.md section 19).

Baseline A, dense (``DenseRAGPipeline``):

    parse -> chunk -> embed -> index -> retrieve top-k -> build context -> generate

Baseline B, hybrid (``HybridRAGPipeline``, EVALUATION_PROTOCOL.md section 8):

    parse -> chunk -> dense + lexical retrieval -> RRF fusion -> rerank
          -> top-k -> build context -> generate

Neither has a planner, an evidence controller, a retrieval loop or a verifier.
That is the point. DD-013 requires both baselines to exist before the agentic
system, because "the agentic pipeline is better" is not a claim that can be
checked without systems that lack every part of it -- and Baseline A lacks the
lexical retriever, the fusion and the reranker too, so Baseline B can be
measured against it.

The two share everything except how they rank: the same parser, chunker,
generator, evidence assembly, citation resolution and abstention. EVALUATION_
PROTOCOL.md section 10 asks for exactly that -- change only the component under
test -- and sharing the code rather than copying it is what guarantees it.

The pipeline owns the wiring and nothing else: each stage is implemented in its
own layer and could be swapped through configuration alone.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .chunking.base import Chunk, build_chunker
from .config import RAGConfig
from .errors import ConfigurationError, IndexNotBuiltError
from .generation.citations import Citation
from .generation.evidence import EvidenceItem
from .generation.generator import GeneratedAnswer, GroundedGenerator
from .ingestion.document import Document
from .ingestion.parser import build_parser
from .reranking.reranker import Reranker, build_reranker, rerank
from .retrieval.embeddings import build_embedder
from .retrieval.hybrid import HybridRetriever
from .retrieval.lexical import LexicalRetriever
from .retrieval.retriever import DenseRetriever, RetrievedChunk

#: The stages each system declares in its per-question metadata. The error
#: taxonomy's reserved ``reranking`` category can fire only for a system that
#: declares it ran a reranker (DD-038).
STAGE_RETRIEVAL = "retrieval"
STAGE_RERANKING = "reranking"
STAGE_CONTEXT_SELECTION = "context-selection"
STAGE_GENERATION = "generation"


@dataclass(frozen=True)
class RAGResult:
    """What a question returns: the answer, its citations, and its evidence."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    evidence: tuple[EvidenceItem, ...]
    retrieved: tuple[RetrievedChunk, ...]
    abstained: bool
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def cited_pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for c in self.citations for page in c.pages}))

    @property
    def evidence_text(self) -> str:
        """Everything the generator was shown, as one string.

        What faithfulness is scored against: a claim is grounded only if it
        appears in the passages the model actually received, which is the
        evidence list rather than the wider retrieval.
        """
        return "\n\n".join(item.text for item in self.evidence)

    @property
    def evidence_pages(self) -> tuple[int, ...]:
        return tuple(sorted({page for e in self.evidence for page in e.pages}))

    def to_record(
        self,
        question_id: str | None = None,
        system: str | None = None,
        reference_answer: str | None = None,
        metrics: dict[str, object] | None = None,
        verification_status: str | None = None,
        latency_s: float | None = None,
    ) -> dict[str, object]:
        """Per-question record for observability and error analysis.

        Called bare it is the observability record the CLI prints. Called with
        the benchmark's arguments it is the BENCHMARK_SPEC.md section 20 record
        -- ``question_id, system, answer, reference_answer, retrieved_chunks,
        citations, metrics, latency, verification_status`` -- which is why the
        harness extends this method rather than keeping a serializer of its own:
        two serializers for one result is two things to keep in step, and the
        one further from the data drifts.
        """
        record: dict[str, object] = {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "citations": [c.to_record() for c in self.citations],
            "cited_pages": list(self.cited_pages),
            "retrieved": [r.to_record() for r in self.retrieved],
            "evidence_markers": [e.marker for e in self.evidence],
            **self.metadata,
        }
        if question_id is not None:
            # Section 20 names `retrieved_chunks`; `retrieved` is the same list
            # under the name the rest of the codebase uses, kept so a reader of
            # either document finds what they are looking for.
            record = {
                "question_id": question_id,
                "system": system or self.metadata.get("pipeline", ""),
                "reference_answer": reference_answer,
                **record,
                "retrieved_chunks": record["retrieved"],
                "evidence_pages": list(self.evidence_pages),
                "verification_status": verification_status or "not_run",
                "latency": latency_s
                if latency_s is not None
                else self.metadata.get("latency_s"),
                "metrics": metrics or {},
            }
        return record

    def format(self) -> str:
        """Human-readable rendering for the notebook and the CLI."""
        lines = [self.answer]
        if self.citations:
            lines.append("")
            lines.append("Evidence:")
            for citation in self.citations:
                section = f" - {citation.section}" if citation.section else ""
                lines.append(f"  [{citation.marker}] {citation.label}{section}")
        return "\n".join(lines)


class _RetrieveThenGenerate:
    """Everything the two baselines share: index, ask, and the probe.

    Subclasses supply ``_rank`` -- the only stage that differs -- and the
    retriever it ranks with. ``ask`` merges whatever trace ``_rank`` returns
    into the result's metadata, so a stage a system adds is recorded without a
    change here, and a system that adds none (Baseline A) produces exactly the
    metadata it always did.
    """

    name = "retrieve_then_generate"
    strategy = ""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = (config or RAGConfig.default()).validate()
        if self.config.retrieval.strategy != self.strategy:
            raise ConfigurationError(
                f"{type(self).__name__} runs retrieval.strategy="
                f"{self.strategy!r}, but the configuration says "
                f"{self.config.retrieval.strategy!r}. Use build_pipeline(config) "
                "to get the system the configuration describes."
            )
        self._parser = build_parser(self.config.ingestion)
        self._chunker = build_chunker(self.config.chunking)
        self._retriever = self._build_retriever()
        self._generator = GroundedGenerator(self.config.generation)
        self.document: Document | None = None
        self.chunks: tuple[Chunk, ...] = ()
        self.index_stats: dict[str, object] = {}

    def _build_retriever(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _rank(
        self, question: str, k: int
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- indexing -----------------------------------------------------------

    def index(self, pdf_path: str | os.PathLike[str]) -> Document:
        """Parse, chunk and index a PDF. Errors here are explicit by design."""
        started = time.perf_counter()
        document = self._parser.parse(pdf_path)
        parsed_at = time.perf_counter()

        chunks = self._chunker.chunk(document)
        chunked_at = time.perf_counter()

        self._retriever.index(chunks)
        self.document = document
        self.chunks = tuple(chunks)
        self.index_stats = {
            "document_id": document.document_id,
            "source_name": document.source_name,
            "pages": document.page_count,
            "empty_pages": [p.page_number for p in document.pages if p.is_empty],
            "chars": document.total_chars,
            "chunks": len(chunks),
            "chunks_spanning_pages": sum(1 for c in chunks if c.spans_pages),
            "chunk_strategy": self._chunker.name,
            "index": self._retriever.index_name,
            "parse_s": round(parsed_at - started, 3),
            "chunk_s": round(chunked_at - parsed_at, 3),
            "embed_index_s": round(time.perf_counter() - chunked_at, 3),
            **self.config.describe(),
        }
        return document

    @property
    def is_indexed(self) -> bool:
        return self._retriever.is_indexed

    # -- querying -----------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> RAGResult:
        if not self.is_indexed:
            raise IndexNotBuiltError(
                "No document has been indexed. Call pipeline.index(pdf_path) "
                "before asking a question."
            )

        started = time.perf_counter()
        k = top_k or self.config.retrieval.top_k
        retrieved, trace = self._rank(question, k)
        retrieved_at = time.perf_counter()

        generated = self._maybe_generate(question, retrieved)

        return RAGResult(
            question=question,
            answer=generated.answer,
            citations=generated.citations,
            evidence=generated.evidence,
            retrieved=tuple(retrieved),
            abstained=generated.abstained,
            metadata={
                "pipeline": self.name,
                "retrieval_strategy": self._retriever.strategy,
                "top_k": k,
                "retrieved_chunk_ids": [r.chunk_id for r in retrieved],
                "retrieval_scores": [round(r.score, 6) for r in retrieved],
                "top_score": round(retrieved[0].score, 6) if retrieved else None,
                "abstention_reason": generated.abstention_reason,
                "dropped_markers": list(generated.dropped_markers),
                "fabricated_page_mentions": list(generated.fabricated_page_mentions),
                "backend": generated.backend,
                "retrieval_s": round(retrieved_at - started, 3),
                "generation_s": round(generated.latency_s, 3),
                "latency_s": round(time.perf_counter() - started, 3),
                "config_fingerprint": self.config.fingerprint(),
                **trace,
                **generated.metadata,
            },
        )

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """Rank chunks without generating an answer.

        The benchmark harness uses this to score retrieval at a depth greater
        than the generation context -- BENCHMARK_SPEC.md section 10 asks for
        Recall@10 while the baseline generates from 5, and without a probe the
        two figures would be the same number under two names.

        It reads the index and returns; it touches no state and cannot change
        what ``ask`` produces. ARCHITECTURE.md section 26 requires benchmark
        instrumentation not to change answer behaviour, and a test asserts the
        stronger property that a harness-run result equals a bare ``ask``.
        """
        if not self.is_indexed:
            raise IndexNotBuiltError(
                "No document has been indexed. Call pipeline.index(pdf_path) "
                "before retrieving."
            )
        ranked, _ = self._rank(question, k or self.config.retrieval.top_k)
        return ranked

    def _maybe_generate(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        """Short-circuit to a refusal when retrieval clearly found nothing.

        The score threshold is disabled by default. EXPERIMENT_PLAN.md section 5
        is explicit that untuned thresholds do not belong in the system before
        there is a dev set to tune them on, so out of the box the pipeline only
        short-circuits when retrieval returned nothing at all.
        """
        minimum = self.config.retrieval.min_score
        if minimum is not None and retrieved and retrieved[0].score < minimum:
            retrieved = []
        return self._generator.generate(question, retrieved)


class DenseRAGPipeline(_RetrieveThenGenerate):
    """Baseline A, end to end."""

    name = "baseline_dense"
    strategy = "dense"

    def _build_retriever(self) -> DenseRetriever:
        return DenseRetriever(build_embedder(self.config.embedding), self.config.retrieval)

    def _rank(
        self, question: str, k: int
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        # No trace: Baseline A's per-question metadata is exactly what it was
        # when results/baseline/ was produced, which a test holds it to.
        return self._retriever.retrieve(question, k), {}


class HybridRAGPipeline(_RetrieveThenGenerate):
    """Baseline B: hybrid retrieval, RRF fusion, optional reranking.

    ``reranking.enabled = False`` is the "Reranker OFF" ablation of
    EVALUATION_PROTOCOL.md section 24. With it off, no reranker is built and
    the fused pool is returned unchanged at the same depth -- not re-sorted by
    some neutral score, which would be a different system that looks like the
    same one.
    """

    name = "baseline_hybrid"
    strategy = "hybrid"

    def __init__(
        self, config: RAGConfig | None = None, reranker: Reranker | None = None
    ) -> None:
        super().__init__(config)
        self.reranker: Reranker | None = None
        if self.config.uses_reranker:
            self.reranker = reranker or build_reranker(self.config.reranking)

    def _build_retriever(self) -> HybridRetriever:
        available = {
            "dense": lambda: DenseRetriever(
                build_embedder(self.config.embedding), self.config.retrieval
            ),
            "lexical": lambda: LexicalRetriever(self.config.retrieval),
        }
        return HybridRetriever(
            {name: available[name]() for name in self.config.retrieval.retrievers},
            self.config.retrieval,
            pool_size=self.config.reranking.candidates,
        )

    @property
    def stages(self) -> tuple[str, ...]:
        if self.reranker is not None:
            return (STAGE_RETRIEVAL, STAGE_RERANKING, STAGE_CONTEXT_SELECTION, STAGE_GENERATION)
        return (STAGE_RETRIEVAL, STAGE_CONTEXT_SELECTION, STAGE_GENERATION)

    def _rank(
        self, question: str, k: int
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        """Fuse, optionally rerank, cut to ``k``.

        The pool is fixed by configuration, so the ranking does not depend on
        ``k``: ``ask``'s top 5 is the first five of the harness's depth-10 probe.
        The trace records what the reranker was handed, so the harness can tell
        a gold page the reranker pushed out (``reranking``) from one retrieval
        never surfaced (``retrieval``) -- DD-044.
        """
        candidates = self._retriever.candidates(question)
        pool = list(candidates.fused)
        rerank_started = time.perf_counter()
        ranked = rerank(self.reranker, question, pool) if self.reranker else pool
        rerank_s = time.perf_counter() - rerank_started

        pre_rerank_top = pool[:k]
        trace: dict[str, object] = {
            "stages": list(self.stages),
            "retrievers": list(candidates.by_retriever),
            "fusion": self.config.retrieval.fusion,
            "candidate_pool": len(pool),
            **{
                f"{name}_chunk_ids": [r.chunk_id for r in ranking[:k]]
                for name, ranking in candidates.by_retriever.items()
            },
            "pre_rerank_chunk_ids": [r.chunk_id for r in pre_rerank_top],
            "pre_rerank_pages": _pages(pre_rerank_top),
            "reranker": self.reranker.model_id if self.reranker else None,
            "rerank_s": round(rerank_s, 3),
        }
        return ranked[:k], trace


def _pages(items: Sequence[RetrievedChunk]) -> list[int]:
    return sorted({page for item in items for page in item.pages})


#: The systems a configuration can describe, keyed by ``retrieval.strategy``.
SYSTEMS: dict[str, type[_RetrieveThenGenerate]] = {
    DenseRAGPipeline.strategy: DenseRAGPipeline,
    HybridRAGPipeline.strategy: HybridRAGPipeline,
}


def build_pipeline(config: RAGConfig | None = None) -> _RetrieveThenGenerate:
    """The system ``config`` describes -- Baseline A or B -- chosen by configuration alone."""
    config = config or RAGConfig.default()
    try:
        system = SYSTEMS[config.retrieval.strategy]
    except KeyError:
        raise ConfigurationError(
            f"Unknown retrieval.strategy {config.retrieval.strategy!r}; "
            f"expected one of {sorted(SYSTEMS)}."
        ) from None
    return system(config)
