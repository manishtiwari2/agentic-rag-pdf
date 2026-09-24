"""The two retrieve-then-generate baselines (ARCHITECTURE.md section 19), and
the agentic system built on the second (ARCHITECTURE.md section 20).

Baseline A, dense (``DenseRAGPipeline``):

    parse -> chunk -> embed -> index -> retrieve top-k -> build context -> generate

Baseline B, hybrid (``HybridRAGPipeline``, EVALUATION_PROTOCOL.md section 8):

    parse -> chunk -> dense + lexical retrieval -> RRF fusion -> rerank
          -> top-k -> build context -> generate

Neither baseline has a planner, an evidence controller, a retrieval loop or a
verifier. That is the point. DD-013 requires both baselines to exist before the agentic
system, because "the agentic pipeline is better" is not a claim that can be
checked without systems that lack every part of it -- and Baseline A lacks the
lexical retriever, the fusion and the reranker too, so Baseline B can be
measured against it.

The two share everything except how they rank: the same parser, chunker,
generator, evidence assembly, citation resolution and abstention. EVALUATION_
PROTOCOL.md section 10 asks for exactly that -- change only the component under
test -- and sharing the code rather than copying it is what guarantees it.

The agentic system (``AgenticRAGPipeline``, EVALUATION_PROTOCOL.md section 9)
is Baseline B's retrieval, fusion and reranking driven by an explicit, bounded
loop of planner, evidence controller, refinement and verifier (DD-050). Its
components live in ``src/agents/``; the loop lives here because the pipeline
owns the wiring. Everything else it shares with both baselines, unchanged.

The pipeline owns the wiring and nothing else: each stage is implemented in its
own layer and could be swapped through configuration alone.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .agents.evidence import ABSTAIN, ANSWER, RETRIEVE_AGAIN, EvidenceController, build_evidence_controller
from .agents.planner import Planner, build_planner
from .agents.refinement import MissingTermsRefiner, Refiner
from .agents.state import AgentState, RetrievalRound
from .agents.verifier import (
    STATUS_FAILED as VERIFICATION_FAILED,
    STATUS_NOT_RUN as VERIFICATION_NOT_RUN,
    STATUS_UNAVAILABLE as VERIFICATION_UNAVAILABLE,
    Verification,
    Verifier,
    build_verifier,
    unavailable,
)
from .chunking.base import Chunk, build_chunker
from .config import RAGConfig
from .errors import ConfigurationError, IndexNotBuiltError
from .generation.backends import LLMBackend
from .generation.citations import Citation
from .generation.evidence import EvidenceItem, build_evidence
from .generation.generator import GeneratedAnswer, GroundedGenerator
from .generation.prompts import ABSTENTION_SENTENCE
from .ingestion.document import Document
from .ingestion.parser import build_parser
from .reranking.reranker import Reranker, build_reranker, rerank
from .retrieval.embeddings import build_embedder
from .retrieval.fusion import reciprocal_rank_fusion
from .retrieval.hybrid import Candidates, HybridRetriever
from .retrieval.lexical import LexicalRetriever
from .retrieval.retriever import DenseRetriever, RetrievedChunk

#: The stages each system declares in its per-question metadata. The error
#: taxonomy's reserved ``reranking`` category can fire only for a system that
#: declares it ran a reranker (DD-038).
STAGE_RETRIEVAL = "retrieval"
STAGE_RERANKING = "reranking"
STAGE_CONTEXT_SELECTION = "context-selection"
STAGE_GENERATION = "generation"
#: The agentic system's stages. ``verification`` is the taxonomy's second
#: reserved category, which can fire only for a system declaring it (DD-054).
STAGE_PLANNING = "planning"
STAGE_EVIDENCE_ASSESSMENT = "evidence-assessment"
STAGE_REFINEMENT = "refinement"
STAGE_VERIFICATION = "verification"


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

    @property
    def backend(self) -> LLMBackend:
        """The generation backend, for layers above the pipeline that need the model.

        Shared, never duplicated: a conversation layer rewriting follow-ups asks
        the model the generator already loaded (ARCHITECTURE.md section 21).
        """
        return self._generator.backend

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

    def _pools(
        self, question: str
    ) -> tuple[Candidates, list[RetrievedChunk], list[RetrievedChunk], float]:
        """One query through retrieval, fusion and (optionally) reranking.

        Returns the candidates, the fused pool the reranker was handed, the
        ranking it returned (the pool itself with the reranker off) and the
        reranking time. Baseline B ranks with this once per question; the
        agentic system calls the same code once per query it issues (DD-050).
        """
        candidates = self._retriever.candidates(question)
        pool = list(candidates.fused)
        rerank_started = time.perf_counter()
        ranked = rerank(self.reranker, question, pool) if self.reranker else pool
        return candidates, pool, ranked, time.perf_counter() - rerank_started

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
        candidates, pool, ranked, rerank_s = self._pools(question)

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


class AgenticRAGPipeline(HybridRAGPipeline):
    """The proposed system (EVALUATION_PROTOCOL.md section 9, DD-050).

        query -> planner -> retrieval -> fusion -> reranking -> evidence
              assessment -> optional refinement -> context selection
              -> generation -> verification -> final answer

    Retrieval, fusion and reranking are Baseline B's, called once per query
    through the same ``_pools``; chunking, evidence assembly, generation,
    citation resolution and abstention are the shared layers, unchanged. What
    this class adds is the loop around them and the four components that make
    its decisions, each switchable by configuration alone:

    * ``agents.planner_enabled = False`` -- one query, the question itself;
    * ``agents.evidence_controller_enabled = False`` -- one retrieval round and
      always answer (refinement is triggered only by the controller);
    * ``agents.refinement_enabled = False`` -- one retrieval round; insufficient
      evidence abstains at once;
    * ``agents.verification_enabled = False`` -- the first draft is the answer.

    A switched-off component is never built, and with all four off the system
    answers exactly as Baseline B does. Every loop is capped by configuration
    and the loop, not the component, enforces the cap.
    """

    name = "agentic"
    strategy = "agentic"

    def __init__(
        self,
        config: RAGConfig | None = None,
        reranker: Reranker | None = None,
        planner: Planner | None = None,
        evidence_controller: EvidenceController | None = None,
        refiner: Refiner | None = None,
        verifier: Verifier | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        super().__init__(config, reranker)
        if backend is not None:
            self._generator = GroundedGenerator(self.config.generation, backend=backend)
        # One generation model, shared: the LLM agents ask the model the
        # generator already loaded (ARCHITECTURE.md section 21).
        shared = self._generator.backend
        label = self.config.generation.model_id
        agents = self.config.agents
        self.planner: Planner | None = None
        self.evidence_controller: EvidenceController | None = None
        self.refiner: Refiner | None = None
        self.verifier: Verifier | None = None
        if agents.planner_enabled:
            self.planner = planner or build_planner(agents, shared, label)
        if agents.evidence_controller_enabled:
            self.evidence_controller = evidence_controller or build_evidence_controller(
                agents, shared, label
            )
            if agents.refinement_enabled:
                self.refiner = refiner or MissingTermsRefiner(agents)
        if agents.verification_enabled:
            self.verifier = verifier or build_verifier(agents, shared, label)

    # -- the retrieval loop -------------------------------------------------

    def _merge(self, rankings: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
        """One query's ranking as it is; several fused by RRF (DD-045, DD-051).

        With a single query nothing is re-scored, which is what makes the
        planner-off system's ranking Baseline B's ranking exactly. Sub-query
        rankings are fused in issue order, the original question first, so a
        tie goes to the ranking of the question actually asked.
        """
        if len(rankings) == 1:
            return list(rankings[0])
        fused = reciprocal_rank_fusion(rankings, k=self.config.retrieval.rrf_k)
        return [
            replace(item, source="+".join(dict.fromkeys(item.source.split("+"))))
            for item in fused
        ]

    def _retrieval_loop(self, question: str, k: int, state: AgentState) -> _LoopOutcome:
        """Plan, retrieve, assess and refine, within the configured caps."""
        agents = self.config.agents
        queries = [question]
        if self.planner is not None:
            plan = self.planner.plan(question)
            state.planner = plan.to_record()
            state.count(
                "planner",
                plan.model_calls,
                llm=plan.model_calls > 0,
                parse_failed=plan.parse_failed,
            )
            state.query_type = plan.query_type
            state.retrieval_strategy = plan.retrieval_strategy
            # The cap is re-applied here: an injected planner is trusted no
            # more than an LLM one.
            queries = list(dict.fromkeys(plan.sub_queries or (question,)))[: agents.max_sub_queries]
        state.query_variants = list(queries)

        cap = self.config.max_retrieval_iterations or 1
        rankings: list[list[RetrievedChunk]] = []
        pools: list[list[RetrievedChunk]] = []
        issued: list[str] = []
        pending = list(queries)
        action = ANSWER
        first_context: list[RetrievedChunk] | None = None
        merged: list[RetrievedChunk] = []
        merged_pre: list[RetrievedChunk] = []

        for iteration in range(1, cap + 1):
            state.iteration = iteration
            round_ = RetrievalRound(iteration=iteration, queries=list(pending))
            for query in pending:
                _, pool, ranked, _ = self._pools(query)
                state.reranker_calls += int(self.reranker is not None and bool(pool))
                pools.append(pool)
                rankings.append(ranked)
                issued.append(query)
                round_.candidates[query] = [r.chunk_id for r in pool]
                round_.reranked[query] = [r.chunk_id for r in ranked]
            merged = self._merge(rankings)
            merged_pre = self._merge(pools)
            context = merged[:k]
            if first_context is None:
                first_context = context
            round_.merged_chunk_ids = [r.chunk_id for r in context]
            state.rounds.append(round_)

            if self.evidence_controller is None:
                state.stop_reason = "no_evidence_controller"
                action = ANSWER
                break

            can_retrieve_again = self.refiner is not None and iteration < cap
            decision = self.evidence_controller.assess(
                question, queries, context, can_retrieve_again
            )
            state.count(
                "evidence_controller",
                decision.model_calls,
                llm=decision.model_calls > 0,
                parse_failed=decision.parse_failed,
            )
            round_.evidence_decision = decision.to_record()
            state.evidence_status = "sufficient" if decision.sufficient else "insufficient"

            if decision.next_action == ANSWER:
                state.stop_reason = "sufficient" if decision.sufficient else "controller_answered"
                action = ANSWER
                break
            if decision.next_action == RETRIEVE_AGAIN and can_retrieve_again:
                pending = self.refiner.refine(question, decision, issued)  # type: ignore[union-attr]
                if pending:
                    continue
                state.stop_reason = "no_new_query"
                action = ABSTAIN
                break
            # Abstain -- because the controller said so, or because it asked
            # for a round the budget does not allow. The loop disposes.
            action = ABSTAIN
            if self.refiner is None:
                state.stop_reason = "insufficient_refinement_off"
            elif iteration >= cap:
                state.stop_reason = "max_iterations"
            else:
                state.stop_reason = "controller_abstained"
            break

        state.bounds = {
            "max_retrieval_iterations": cap,
            "iterations_used": state.iteration,
            "hit_cap": action == ABSTAIN and state.stop_reason == "max_iterations",
            "max_sub_queries": agents.max_sub_queries if self.planner is not None else None,
            "sub_queries_truncated": (state.planner or {}).get("truncated_sub_queries", 0)
            if self.planner is not None
            else 0,
            "max_regenerations": agents.max_regenerations if self.verifier is not None else None,
            "regenerations_used": 0,
        }
        return _LoopOutcome(
            ranking=merged,
            pre_rerank=merged_pre,
            action=action,
            first_context=first_context or [],
        )

    # -- asking --------------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """The final merged ranking, without generating (DD-039, DD-050).

        Re-runs the retrieval loop -- plan, rounds, assessment, refinement --
        and nothing after it. It keeps no state, so it cannot change what
        ``ask`` produces, and because the loop assesses the same top-k ``ask``
        does, ``ask``'s context is this ranking's first ``top_k``.
        """
        if not self.is_indexed:
            raise IndexNotBuiltError(
                "No document has been indexed. Call pipeline.index(pdf_path) "
                "before retrieving."
            )
        state = AgentState(query=question)
        outcome = self._retrieval_loop(question, self.config.retrieval.top_k, state)
        return outcome.ranking[: k or self.config.retrieval.top_k]

    def ask(self, question: str, top_k: int | None = None) -> RAGResult:
        if not self.is_indexed:
            raise IndexNotBuiltError(
                "No document has been indexed. Call pipeline.index(pdf_path) "
                "before asking a question."
            )
        started = time.perf_counter()
        k = top_k or self.config.retrieval.top_k
        state = AgentState(query=question)
        outcome = self._retrieval_loop(question, k, state)
        context = outcome.ranking[:k]
        retrieved_at = time.perf_counter()
        state.retrieved_chunk_ids = [r.chunk_id for r in outcome.pre_rerank[:k]]
        state.reranked_chunk_ids = [r.chunk_id for r in context]

        stages = [STAGE_PLANNING] if self.planner is not None else []
        stages.append(STAGE_RETRIEVAL)
        if self.reranker is not None:
            stages.append(STAGE_RERANKING)
        if self.evidence_controller is not None:
            stages.append(STAGE_EVIDENCE_ASSESSMENT)
        if state.iteration > 1:
            stages.append(STAGE_REFINEMENT)
        stages.append(STAGE_CONTEXT_SELECTION)

        abstention_source: str | None = None
        draft: GeneratedAnswer | None = None
        verification_status = VERIFICATION_NOT_RUN
        if outcome.action == ABSTAIN:
            generated = self._refuse(context, "evidence_insufficient")
            abstention_source = "evidence_controller"
        else:
            stages.append(STAGE_GENERATION)
            generated = self._generate(question, context, state)
            draft = generated
            if generated.abstained:
                abstention_source = "generator"
            elif self.verifier is not None:
                stages.append(STAGE_VERIFICATION)
                generated, verification_status = self._verify_loop(
                    question, context, generated, state
                )
                if verification_status == VERIFICATION_FAILED and not generated.abstained:
                    generated = self._refuse(generated.evidence, "verification_failed")
                    abstention_source = "verifier"
                elif generated.abstained:
                    abstention_source = "generator"

        state.selected_context = [e.chunk_id for e in generated.evidence]
        state.draft_answer = draft.answer if draft is not None else None
        state.verification_status = verification_status
        state.citations = [c.to_record() for c in generated.citations]
        state.final = {
            "answer": generated.answer,
            "abstained": generated.abstained,
            "abstention_source": abstention_source,
        }
        refinement_changed = self._refinement_counterfactual(
            question, outcome, draft, context, state
        )

        pre_top = outcome.pre_rerank[:k]
        metadata: dict[str, object] = {
            "pipeline": self.name,
            "retrieval_strategy": self._retriever.strategy,
            "top_k": k,
            "retrieved_chunk_ids": [r.chunk_id for r in context],
            "retrieval_scores": [round(r.score, 6) for r in context],
            "top_score": round(context[0].score, 6) if context else None,
            "abstention_reason": generated.abstention_reason,
            "dropped_markers": list(generated.dropped_markers),
            "fabricated_page_mentions": list(generated.fabricated_page_mentions),
            "backend": generated.backend,
            "retrieval_s": round(retrieved_at - started, 3),
            "generation_s": round(time.perf_counter() - retrieved_at, 3),
            "latency_s": round(time.perf_counter() - started, 3),
            "config_fingerprint": self.config.fingerprint(),
            "stages": stages,
            "retrievers": list(self._retriever.retrievers),
            "fusion": self.config.retrieval.fusion,
            "candidate_pool": len(outcome.ranking),
            # DD-044's counterfactual, across every query issued: what the
            # generator would have been handed with the reranker off.
            "pre_rerank_chunk_ids": [r.chunk_id for r in pre_top],
            "pre_rerank_pages": _pages(pre_top),
            "reranker": self.reranker.model_id if self.reranker else None,
            "query_type": state.query_type,
            "sub_queries": list(state.query_variants),
            "retrieval_iterations": state.iteration,
            "stop_reason": state.stop_reason,
            "abstention_source": abstention_source,
            "verification_status": verification_status,
            "model_calls": state.total_model_calls,
            "diagnostic_model_calls": state.diagnostic_model_calls,
            "reranker_calls": state.reranker_calls,
            "llm_decisions": state.llm_decisions,
            "llm_parse_failures": state.llm_parse_failures,
            "refinement_changed_answer": refinement_changed,
            **generated.metadata,
            "agent_trace": state.to_record(),
        }
        if draft is not None and STAGE_VERIFICATION in stages:
            # What "Verification OFF" would have returned: the harness scores it
            # to tell a verifier that lost a correct answer from one that
            # caught a wrong one (DD-054).
            metadata["draft_answer"] = draft.answer
        return RAGResult(
            question=question,
            answer=generated.answer,
            citations=generated.citations,
            evidence=generated.evidence,
            retrieved=tuple(context),
            abstained=generated.abstained,
            metadata=metadata,
        )

    # -- helpers ---------------------------------------------------------------

    def _generate(
        self,
        question: str,
        context: list[RetrievedChunk],
        state: AgentState,
        diagnostic: bool = False,
    ) -> GeneratedAnswer:
        generated = self._maybe_generate(question, context)
        called = int(bool(generated.evidence))  # no evidence, no model call
        if diagnostic:
            state.diagnostic_model_calls += called
        else:
            state.count("generator", called, llm=False)
            state.generations.append(
                {
                    "context": [r.chunk_id for r in context],
                    "answer": generated.answer,
                    "abstained": generated.abstained,
                    "abstention_reason": generated.abstention_reason,
                    "backend": generated.backend,
                }
            )
        return generated

    def _refuse(
        self, evidence_source: Sequence[RetrievedChunk] | Sequence[EvidenceItem], reason: str
    ) -> GeneratedAnswer:
        """The canonical refusal, keeping the evidence the system had looked at.

        Built here rather than by the generator because the generator was not
        asked -- the evidence controller or the verifier decided -- but it uses
        the one refusal sentence and the shared evidence assembly, so a refusal
        scores exactly as the generator's own would.
        """
        if evidence_source and isinstance(evidence_source[0], EvidenceItem):
            evidence = tuple(evidence_source)  # type: ignore[arg-type]
        else:
            evidence = tuple(build_evidence(list(evidence_source), self.config.generation))  # type: ignore[arg-type]
        return GeneratedAnswer(
            answer=ABSTENTION_SENTENCE,
            raw_answer=ABSTENTION_SENTENCE,
            citations=(),
            evidence=evidence,
            abstained=True,
            abstention_reason=reason,
            backend=self._generator.backend.name,
            metadata={"evidence_count": len(evidence)},
        )

    def _verify_once(
        self, question: str, generated: GeneratedAnswer, state: AgentState
    ) -> Verification:
        try:
            verification = self.verifier.verify(  # type: ignore[union-attr]
                question, generated.answer, generated.evidence
            )
        except Exception as exc:  # noqa: BLE001 - ARCHITECTURE.md section 24
            verification = unavailable(getattr(self.verifier, "name", "verifier"), exc)
        state.count(
            "verifier",
            verification.model_calls,
            llm=verification.model_calls > 0,
            parse_failed=verification.status == VERIFICATION_UNAVAILABLE
            and verification.model_calls > 0,
        )
        state.verifications.append(verification.to_record())
        return verification

    def _verify_loop(
        self,
        question: str,
        context: list[RetrievedChunk],
        generated: GeneratedAnswer,
        state: AgentState,
    ) -> tuple[GeneratedAnswer, str]:
        """Verify; on failure regenerate without the blamed evidence, up to the cap.

        ARCHITECTURE.md section 16's first remedy, bounded by
        ``agents.max_regenerations``. A regeneration drops the chunks the
        unsupported claims cited -- the evidence that did not bear them out --
        and stops early when there is nothing left to drop or nothing left to
        answer from. What remains failed after the last round is refused by
        ``ask``.
        """
        excluded: set[str] = set()
        regenerations = 0
        verification = self._verify_once(question, generated, state)
        while (
            verification.status == VERIFICATION_FAILED
            and regenerations < self.config.agents.max_regenerations
        ):
            by_number = {e.number: e.chunk_id for e in generated.evidence}
            blamed = {by_number[n] for n in verification.blamed_evidence if n in by_number}
            if not blamed - excluded:
                break
            excluded |= blamed
            remaining = [r for r in context if r.chunk_id not in excluded]
            if not remaining:
                break
            regenerations += 1
            generated = self._generate(question, remaining, state)
            if generated.abstained:
                break
            verification = self._verify_once(question, generated, state)
        state.bounds["regenerations_used"] = regenerations
        return generated, verification.status

    def _refinement_counterfactual(
        self,
        question: str,
        outcome: "_LoopOutcome",
        draft: GeneratedAnswer | None,
        context: list[RetrievedChunk],
        state: AgentState,
    ) -> bool | None:
        """Did the second retrieval round change the answer? (Open question 3.)

        Compares the generator's draft from the first round's context with its
        draft from the final context -- the effect of the extra retrieval on
        what the model says, independent of whether the controller or verifier
        then let it through. Diagnostic generations are counted apart from the
        system's model calls and never reach the answer (DD-053). None when
        only one round ran.
        """
        if state.iteration <= 1 or not self.config.agents.record_refinement_counterfactual:
            return None
        first = self._generate(question, outcome.first_context, state, diagnostic=True)
        final = draft if draft is not None else self._generate(question, context, state, diagnostic=True)
        changed = _normalized(first.answer) != _normalized(final.answer)
        state.counterfactual = {
            "first_round_context": [r.chunk_id for r in outcome.first_context],
            "first_round_answer": first.answer,
            "final_context": [r.chunk_id for r in context],
            "final_round_answer": final.answer,
            "changed": changed,
        }
        return changed


@dataclass(frozen=True)
class _LoopOutcome:
    ranking: list[RetrievedChunk]
    pre_rerank: list[RetrievedChunk]
    action: str
    first_context: list[RetrievedChunk]


def _normalized(answer: str) -> str:
    return " ".join((answer or "").split()).lower()


#: The systems a configuration can describe, keyed by ``retrieval.strategy``.
SYSTEMS: dict[str, type[_RetrieveThenGenerate]] = {
    DenseRAGPipeline.strategy: DenseRAGPipeline,
    HybridRAGPipeline.strategy: HybridRAGPipeline,
    AgenticRAGPipeline.strategy: AgenticRAGPipeline,
}


def build_pipeline(config: RAGConfig | None = None) -> _RetrieveThenGenerate:
    """The system ``config`` describes -- Baseline A, B or the agentic system --
    chosen by configuration alone."""
    config = config or RAGConfig.default()
    try:
        system = SYSTEMS[config.retrieval.strategy]
    except KeyError:
        raise ConfigurationError(
            f"Unknown retrieval.strategy {config.retrieval.strategy!r}; "
            f"expected one of {sorted(SYSTEMS)}."
        ) from None
    return system(config)
