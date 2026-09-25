"""Tests for Phase 4: lexical retrieval, fusion, reranking, Baseline B.

Each of these checks a claim a plausible implementation could satisfy in
appearance only:

* **DD-007's justification** -- that lexical retrieval finds an exact identifier
  dense retrieval cannot -- is asserted on this implementation, not assumed.
* **RRF reads ranks, not scores.** An implementation that added normalized
  scores would also be deterministic; it would not survive rescaling.
* **Reranker OFF is the fused ranking, exactly.** Not a re-sort by a neutral
  score, and not with the reranker built and ignored.
* **The ``reranking`` error category fires for a gold page the reranker pushed
  out**, through the real harness, and not for one retrieval never surfaced.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import pytest

from src.benchmark.schema import Question, load_questions
from src.chunking.base import Chunk
from src.config import OVERLAP_RERANKER, RAGConfig, RerankingConfig, RetrievalConfig
from src.errors import ConfigurationError, IndexNotBuiltError, RerankingError, RetrievalError
from src.evaluation.benchmark import run_benchmark_cli, score_result
from src.evaluation.metrics import (
    CONTEXT_SELECTION_FAILURE,
    ERROR_CATEGORIES,
    RERANKING_FAILURE,
    RETRIEVAL_FAILURE,
)
from src.generation.evidence import EvidenceItem
from src.pipeline import (
    STAGE_RERANKING,
    DenseRAGPipeline,
    HybridRAGPipeline,
    RAGResult,
    build_pipeline,
)
from src.reranking.reranker import (
    CrossEncoderReranker,
    TermOverlapReranker,
    build_reranker,
    rerank,
)
from src.retrieval.embeddings import build_embedder
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.lexical import LexicalRetriever, tokenize
from src.retrieval.retriever import DenseRetriever, RetrievedChunk

BENCHMARK = pathlib.Path("benchmark/questions.json")
DOCUMENTS = pathlib.Path("benchmark/documents")
BASELINE = pathlib.Path("results/baseline")
HYBRID = pathlib.Path("results/hybrid")


def chunk(index: int, text: str = "body text", pages: tuple[int, ...] | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"doc:c:{index:05d}",
        document_id="doc",
        text=text,
        pages=pages or (index + 1,),
        index=index,
    )


def ranking(*indices: int, source: str = "dense", scores=None) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=chunk(i),
            score=(scores[n] if scores else 1.0 - n / 100),
            rank=n + 1,
            source=source,
        )
        for n, i in enumerate(indices)
    ]


def hybrid_config(offline_config: RAGConfig, **reranking) -> RAGConfig:
    return dataclasses.replace(
        offline_config,
        retrieval=dataclasses.replace(offline_config.retrieval, strategy="hybrid"),
        reranking=dataclasses.replace(offline_config.reranking, **reranking),
    )


# ---------------------------------------------------------------------------
# Lexical retrieval
# ---------------------------------------------------------------------------

#: A parts catalogue in miniature. Three distractors carry identifiers one digit
#: from the target's and repeat every query word; the target states the exact
#: identifier once, in a short sentence without "torque rating". The last three
#: chunks make "torque", "rating" and "part" as common as they are on a real
#: catalogue page, which is what BM25's IDF is for.
CATALOGUE = (
    "Torque rating for part 884212: the torque rating of this part is 60 Nm. "
    "Check the torque rating before assembly of the part.",
    "Torque rating for part 884213: the torque rating of this part is 75 Nm. "
    "Check the torque rating before assembly of the part.",
    "Torque rating for part 884210: the torque rating of this part is 90 Nm. "
    "Check the torque rating before assembly of the part.",
    "Part 884211 is fastened to 45 Nm.",
    "The torque rating of a part is the torque at which the part is rated.",
    "Every part has a torque rating, stated for a clean, dry thread.",
    "Exceeding the torque rating of a part voids its warranty.",
)
TARGET = 3
IDENTIFIER_QUERY = "What is the torque rating for part 884211?"


class TestLexicalRetrievalJustifiesItself:
    """DD-007: dense retrieval is weak on exact identifiers and numbers.

    The offline embedder is itself partly lexical -- hashed words, character
    4-grams and word pairs -- so this is the honest version of the claim for
    this implementation: its character n-grams give ``884212`` most of the credit
    ``884211`` earns, and its IDF never drives a ubiquitous word to zero, so a
    distractor repeating the query's common words outranks the one chunk with
    the right number. BM25 saturates the repetition, zeroes out the common words
    and matches the identifier as one whole token. Against ``bge-m3`` the effect
    is expected to be stronger, not weaker -- a neural embedder does not see
    digits as a string at all -- but that is not what this test measures.
    """

    @pytest.fixture
    def chunks(self):
        return [chunk(i, text) for i, text in enumerate(CATALOGUE)]

    def test_lexical_ranks_the_exact_identifier_first(self, chunks):
        lexical = LexicalRetriever()
        lexical.index(chunks)
        assert lexical.retrieve(IDENTIFIER_QUERY, 3)[0].chunk.index == TARGET

    def test_dense_alone_does_not(self, chunks):
        config = RAGConfig.offline()
        dense = DenseRetriever(build_embedder(config.embedding), config.retrieval)
        dense.index(chunks)
        ranked = [r.chunk.index for r in dense.retrieve(IDENTIFIER_QUERY, len(chunks))]
        assert ranked[0] != TARGET
        assert ranked[0] in (0, 1, 2), "a near-identical identifier won"
        assert ranked.index(TARGET) >= 3

    def test_fusion_recovers_it_where_dense_alone_missed_it(self, chunks):
        config = RAGConfig.offline()
        dense = DenseRetriever(build_embedder(config.embedding), config.retrieval)
        dense.index(chunks)
        lexical = LexicalRetriever()
        lexical.index(chunks)
        fused = reciprocal_rank_fusion(
            [dense.retrieve(IDENTIFIER_QUERY, 7), lexical.retrieve(IDENTIFIER_QUERY, 7)]
        )
        dense_rank = [r.chunk.index for r in dense.retrieve(IDENTIFIER_QUERY, 7)].index(TARGET)
        fused_rank = [r.chunk.index for r in fused].index(TARGET)
        assert fused_rank < dense_rank


class TestLexicalRetriever:
    def test_numbers_stay_whole(self):
        assert tokenize("Peak 9.4 GB, 1,200 tokens; part 884211") == [
            "peak", "9.4", "gb", "1200", "tokens", "part", "884211",
        ]

    def test_a_decimal_does_not_match_its_digits(self):
        lexical = LexicalRetriever()
        lexical.index([chunk(0, "It read 3.14 volts."), chunk(1, "It read 14 and 3 volts.")])
        assert lexical.retrieve("which reading was 3.14", 2)[0].chunk.index == 0

    def test_a_chunk_sharing_no_term_is_not_returned(self):
        lexical = LexicalRetriever()
        lexical.index([chunk(0, "alpha beta"), chunk(1, "gamma delta")])
        assert [r.chunk.index for r in lexical.retrieve("alpha", 5)] == [0]

    def test_ties_fall_back_to_document_order(self):
        lexical = LexicalRetriever()
        lexical.index([chunk(i, "identical text here") for i in range(4)])
        assert [r.chunk.index for r in lexical.retrieve("identical", 4)] == [0, 1, 2, 3]

    def test_results_carry_the_lexical_source_and_ranks(self):
        lexical = LexicalRetriever()
        lexical.index([chunk(0, "alpha"), chunk(1, "alpha alpha beta")])
        hits = lexical.retrieve("alpha beta", 2)
        assert [h.rank for h in hits] == [1, 2]
        assert {h.source for h in hits} == {"lexical"}

    def test_it_is_deterministic(self):
        chunks = [chunk(i, text) for i, text in enumerate(CATALOGUE)]
        runs = []
        for _ in range(3):
            lexical = LexicalRetriever()
            lexical.index(chunks)
            runs.append([(h.chunk_id, h.score) for h in lexical.retrieve(IDENTIFIER_QUERY, 7)])
        assert runs[0] == runs[1] == runs[2]

    def test_errors_are_explicit(self):
        with pytest.raises(IndexNotBuiltError):
            LexicalRetriever().retrieve("anything")
        with pytest.raises(RetrievalError):
            LexicalRetriever().index([])
        lexical = LexicalRetriever()
        lexical.index([chunk(0, "alpha")])
        with pytest.raises(RetrievalError, match="empty"):
            lexical.retrieve("   ")


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fused_view(items):
    return [(r.chunk_id, r.rank, r.score, r.source) for r in items]


class TestReciprocalRankFusion:
    def test_the_formula(self):
        fused = reciprocal_rank_fusion(
            [ranking(0, 1), ranking(1, 2, source="lexical")], k=60
        )
        by_id = {r.chunk.index: r.score for r in fused}
        assert by_id[1] == pytest.approx(1 / 62 + 1 / 61)
        assert by_id[0] == pytest.approx(1 / 61)
        assert by_id[2] == pytest.approx(1 / 62)
        assert [r.chunk.index for r in fused] == [1, 0, 2]

    def test_it_is_deterministic(self):
        rankings = [ranking(3, 1, 4, 0), ranking(4, 2, 3, source="lexical")]
        first = fused_view(reciprocal_rank_fusion(rankings))
        for _ in range(5):
            assert fused_view(reciprocal_rank_fusion(rankings)) == first

    @pytest.mark.parametrize(
        "transform",
        [
            lambda s, n: s * 1000.0,
            lambda s, n: s / 7.0 - 50.0,
            lambda s, n: (s + 2.0) ** 5,
            lambda s, n: -1.0e9 * n,  # garbage, still in order
            lambda s, n: [0.3, 99.0, -4.0, 1e-9][n],  # garbage, out of order
        ],
    )
    def test_only_the_rankings_matter_not_the_scores(self, transform):
        dense_order, lexical_order = (3, 1, 4, 0), (4, 2, 3, 0)
        reference = reciprocal_rank_fusion(
            [ranking(*dense_order), ranking(*lexical_order, source="lexical")]
        )

        def rescored(order, source):
            items = ranking(*order, source=source)
            return [
                dataclasses.replace(item, score=transform(item.score, n), rank=99 - n)
                for n, item in enumerate(items)
            ]

        fused = reciprocal_rank_fusion(
            [rescored(dense_order, "dense"), rescored(lexical_order, "lexical")]
        )
        assert fused_view(fused) == fused_view(reference)

    def test_an_exact_tie_goes_to_the_higher_priority_list(self):
        # Chunk 0 is (1, 3); chunk 1 is (3, 1). Exactly equal RRF scores, equal
        # best rank; chunk 0 earned its rank 1 in the first (dense) list.
        dense = ranking(0, 2, 1)
        lexical = ranking(1, 2, 0, source="lexical")
        fused = reciprocal_rank_fusion([dense, lexical])
        scores = {r.chunk.index: r.score for r in fused}
        assert scores[0] == scores[1]
        order = [r.chunk.index for r in fused]
        assert order.index(0) < order.index(1)
        # ...and swapping the priority swaps the winner.
        swapped = [r.chunk.index for r in reciprocal_rank_fusion([lexical, dense])]
        assert swapped.index(1) < swapped.index(0)

    def test_a_single_ranking_passes_through_in_order(self):
        single = ranking(5, 2, 7, 1)
        assert [r.chunk.index for r in reciprocal_rank_fusion([single])] == [5, 2, 7, 1]

    def test_the_source_names_every_retriever_that_found_it(self):
        fused = reciprocal_rank_fusion([ranking(0, 1), ranking(1, 2, source="lexical")])
        sources = {r.chunk.index: r.source for r in fused}
        assert sources == {0: "dense", 1: "dense+lexical", 2: "lexical"}

    def test_a_duplicate_within_one_ranking_counts_once(self):
        doubled = ranking(0, 0, 1)
        fused = reciprocal_rank_fusion([doubled])
        assert {r.chunk.index: r.score for r in fused}[0] == pytest.approx(1 / 61)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([ranking(0)], k=0)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


class Fixed:
    """A reranker whose scores are given, to test ``rerank`` alone."""

    model_id = "test/fixed"

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return list(self.scores[: len(passages)])


class Reverse:
    """Scores candidates in reverse of the order it is handed them."""

    model_id = "test/reverse"

    def score(self, query, passages):
        return [float(i) for i in range(len(passages))]


class Exploding:
    model_id = "test/exploding"

    def score(self, query, passages):  # pragma: no cover - must never run
        raise AssertionError("the reranker was called with reranking disabled")


class TestRerank:
    def test_it_orders_by_score_and_renumbers(self):
        items = ranking(0, 1, 2)
        out = rerank(Fixed([0.1, 0.9, 0.5]), "q", items)
        assert [r.chunk.index for r in out] == [1, 2, 0]
        assert [r.rank for r in out] == [1, 2, 3]
        assert [r.score for r in out] == [0.9, 0.5, 0.1]

    def test_ties_keep_the_incoming_order(self):
        out = rerank(Fixed([0.5, 0.5, 0.5]), "q", ranking(4, 2, 9))
        assert [r.chunk.index for r in out] == [4, 2, 9]

    def test_provenance_survives(self):
        items = reciprocal_rank_fusion([ranking(0, 1), ranking(1, source="lexical")])
        out = rerank(Fixed([0.0, 1.0]), "q", items)
        assert {r.chunk.index: r.source for r in out} == {0: "dense", 1: "dense+lexical"}

    def test_a_wrong_number_of_scores_is_an_error(self):
        with pytest.raises(RerankingError):
            rerank(Fixed([0.1]), "q", ranking(0, 1))

    def test_an_empty_pool_is_not_scored(self):
        fixed = Fixed([])
        assert rerank(fixed, "q", []) == []
        assert fixed.calls == 0


class TestTermOverlapReranker:
    def test_it_prefers_the_passage_covering_the_query(self):
        reranker = TermOverlapReranker()
        scores = reranker.score(
            "peak memory during generation",
            ["Latency was measured on a T4.", "Peak memory during generation was 9.4 GB."],
        )
        assert scores[1] > scores[0]

    def test_it_reorders_a_list_bm25_ordered_differently(self):
        # BM25 rewards the rare word repeated; coverage rewards every query word.
        chunks = [
            chunk(0, "zeta zeta zeta"),
            chunk(1, "alpha zeta and several more words of padding text here"),
            chunk(2, "alpha one"),
            chunk(3, "alpha two"),
            chunk(4, "alpha three"),
        ]
        lexical = LexicalRetriever()
        lexical.index(chunks)
        before = lexical.retrieve("alpha zeta", 5)
        after = rerank(TermOverlapReranker(), "alpha zeta", before)
        assert before[0].chunk.index == 0
        assert after[0].chunk.index == 1

    def test_it_is_deterministic(self):
        reranker = TermOverlapReranker()
        passages = list(CATALOGUE)
        assert reranker.score(IDENTIFIER_QUERY, passages) == reranker.score(
            IDENTIFIER_QUERY, passages
        )

    def test_the_configuration_chooses_the_backend(self):
        assert isinstance(
            build_reranker(RerankingConfig(model_id=OVERLAP_RERANKER)), TermOverlapReranker
        )
        # Built lazily: constructing the model-backed reranker loads nothing.
        assert isinstance(build_reranker(RerankingConfig()), CrossEncoderReranker)

    def test_offline_mode_uses_it(self):
        assert RAGConfig.offline().reranking.model_id == OVERLAP_RERANKER


# ---------------------------------------------------------------------------
# Baseline B, end to end
# ---------------------------------------------------------------------------

QUESTIONS = (
    "How are chunks embedded?",
    "What recall did the hybrid system achieve?",
    "What does the system cite?",
    "Did accuracy improve?",
)


class TestHybridPipeline:
    def test_the_configuration_chooses_the_system(self, offline_config):
        assert type(build_pipeline(offline_config)) is DenseRAGPipeline
        assert type(build_pipeline(hybrid_config(offline_config))) is HybridRAGPipeline

    def test_a_system_refuses_the_other_systems_configuration(self, offline_config):
        with pytest.raises(ConfigurationError, match="build_pipeline"):
            DenseRAGPipeline(hybrid_config(offline_config))
        with pytest.raises(ConfigurationError, match="build_pipeline"):
            HybridRAGPipeline(offline_config)

    def test_it_answers_with_citations(self, offline_config, structured_pdf_path):
        pipeline = HybridRAGPipeline(hybrid_config(offline_config))
        pipeline.index(structured_pdf_path)
        result = pipeline.ask("What recall did the hybrid system achieve?")
        assert result.metadata["pipeline"] == "baseline_hybrid"
        assert result.metadata["retrieval_strategy"] == "hybrid"
        assert result.metadata["stages"][1] == STAGE_RERANKING
        assert result.metadata["reranker"] == OVERLAP_RERANKER
        assert result.citations
        assert all(r.source in {"dense", "lexical", "dense+lexical"} for r in result.retrieved)

    def test_the_ranking_does_not_depend_on_k(self, offline_config, structured_pdf_path):
        """The probe's depth-10 ranking is the ranking ``ask`` cut its five from."""
        pipeline = HybridRAGPipeline(hybrid_config(offline_config))
        pipeline.index(structured_pdf_path)
        for question in QUESTIONS:
            asked = [r.chunk_id for r in pipeline.ask(question).retrieved]
            probed = [r.chunk_id for r in pipeline.retrieve(question, 10)]
            assert asked == probed[: len(asked)]

    def test_lexical_only_and_dense_only_arms_are_configuration(
        self, offline_config, structured_pdf_path
    ):
        for retrievers in (("dense",), ("lexical",)):
            config = hybrid_config(offline_config)
            config = dataclasses.replace(
                config, retrieval=dataclasses.replace(config.retrieval, retrievers=retrievers)
            )
            pipeline = HybridRAGPipeline(config)
            pipeline.index(structured_pdf_path)
            result = pipeline.ask("How are chunks embedded?")
            assert result.metadata["retrievers"] == list(retrievers)
            assert {r.source for r in result.retrieved} == {retrievers[0]}


class TestRerankerOffIsTheFusedRanking:
    """EVALUATION_PROTOCOL.md section 24's "Reranker OFF" -- by configuration
    alone, and doing exactly nothing rather than something that looks like it."""

    @pytest.fixture
    def pipeline_off(self, offline_config, structured_pdf_path):
        pipeline = HybridRAGPipeline(
            hybrid_config(offline_config, enabled=False), reranker=Exploding()
        )
        pipeline.index(structured_pdf_path)
        return pipeline

    def test_no_reranker_is_built_or_called(self, pipeline_off):
        assert pipeline_off.reranker is None
        for question in QUESTIONS:
            pipeline_off.ask(question)  # Exploding would raise if it were used

    def test_the_answer_path_is_the_fused_pool_field_for_field(self, pipeline_off):
        k = pipeline_off.config.retrieval.top_k
        for question in QUESTIONS:
            fused = pipeline_off._retriever.candidates(question).fused
            result = pipeline_off.ask(question)
            assert fused_view(result.retrieved) == fused_view(fused[:k])
            assert fused_view(pipeline_off.retrieve(question, 10)) == fused_view(fused[:10])
            assert result.metadata["pre_rerank_chunk_ids"] == [r.chunk_id for r in result.retrieved]

    def test_it_declares_no_reranking_stage(self, pipeline_off):
        result = pipeline_off.ask("How are chunks embedded?")
        assert STAGE_RERANKING not in result.metadata["stages"]
        assert result.metadata["reranker"] is None

    def test_it_matches_a_reranker_that_scores_nothing_apart(
        self, offline_config, structured_pdf_path
    ):
        """A reranker that cannot separate any two candidates leaves fusion's
        order alone (the stable sort), so ON-with-no-signal equals OFF on the
        ranking; ``score`` differs, which is why OFF does not rerank at all."""
        on = HybridRAGPipeline(hybrid_config(offline_config), reranker=Fixed([0.0] * 50))
        off = HybridRAGPipeline(hybrid_config(offline_config, enabled=False))
        for pipeline in (on, off):
            pipeline.index(structured_pdf_path)
        for question in QUESTIONS:
            assert [r.chunk_id for r in on.ask(question).retrieved] == [
                r.chunk_id for r in off.ask(question).retrieved
            ]

    def test_reranker_on_does_reorder(self, offline_config, structured_pdf_path):
        """So the ON/OFF equality tests above are not vacuous."""
        on = HybridRAGPipeline(hybrid_config(offline_config), reranker=Reverse())
        off = HybridRAGPipeline(hybrid_config(offline_config, enabled=False))
        for pipeline in (on, off):
            pipeline.index(structured_pdf_path)
        question = "How are chunks embedded?"
        assert [r.chunk_id for r in on.retrieve(question, 10)] == list(
            reversed([r.chunk_id for r in off.retrieve(question, 10)])
        )


class TestHarnessDoesNotChangeBaselineBAnswers:
    """ARCHITECTURE.md section 26, for Baseline B: the answer the harness scores
    is the answer a bare ``ask`` produces."""

    TIMING_KEYS = frozenset({"retrieval_s", "generation_s", "latency_s", "rerank_s"})

    @pytest.mark.parametrize("enabled", [True, False], ids=["rerank-on", "rerank-off"])
    def test_a_harness_run_result_equals_a_bare_ask(
        self, offline_config, structured_pdf_path, enabled
    ):
        pipeline = HybridRAGPipeline(hybrid_config(offline_config, enabled=enabled))
        pipeline.index(structured_pdf_path)
        question = Question(
            id="q001",
            document="structured.pdf",
            question="What recall did the hybrid system achieve?",
            answer="It reached 0.83 recall.",
            evidence_pages=(2,),
            question_type="numerical",
            difficulty="easy",
            split="dev",
        )

        bare = pipeline.ask(question.question)
        probed = pipeline.retrieve(question.question, 10)
        instrumented = pipeline.ask(question.question)
        scored = score_result(
            question, instrumented, system=pipeline.name, retrieved_for_metrics=probed
        )

        assert instrumented.answer == bare.answer
        assert instrumented.cited_pages == bare.cited_pages
        assert fused_view(instrumented.retrieved) == fused_view(bare.retrieved)
        assert [e.marker for e in instrumented.evidence] == [e.marker for e in bare.evidence]
        strip = lambda m: {k: v for k, v in m.items() if k not in self.TIMING_KEYS}  # noqa: E731
        assert strip(instrumented.metadata) == strip(bare.metadata)
        assert scored.record["answer"] == bare.answer

    def test_the_probe_does_not_disturb_the_index(self, offline_config, structured_pdf_path):
        pipeline = HybridRAGPipeline(hybrid_config(offline_config))
        pipeline.index(structured_pdf_path)
        before = pipeline.ask("How are chunks embedded?")
        pipeline.retrieve("something else entirely", 10)
        after = pipeline.ask("How are chunks embedded?")
        assert before.answer == after.answer
        assert fused_view(before.retrieved) == fused_view(after.retrieved)


# ---------------------------------------------------------------------------
# The `reranking` error category, exercised for real
# ---------------------------------------------------------------------------


def question_on(pages: tuple[int, ...], answer: str = "The figure was 123456.") -> Question:
    """A question the offline system will get wrong, so it needs a category."""
    return Question(
        id="q900",
        document="doc.pdf",
        question="What was the figure?",
        answer=answer,
        evidence_pages=pages,
        question_type="factual",
        difficulty="easy",
        split="dev",
    )


def result_with(
    answer_pages: tuple[tuple[int, ...], ...],
    evidence_pages: tuple[tuple[int, ...], ...] | None = None,
    **metadata,
) -> RAGResult:
    retrieved = tuple(
        RetrievedChunk(chunk=chunk(i, "some passage", pages), score=1.0, rank=i + 1)
        for i, pages in enumerate(answer_pages)
    )
    shown = retrieved if evidence_pages is None else tuple(
        RetrievedChunk(chunk=chunk(50 + i, "some passage", pages), score=1.0, rank=i + 1)
        for i, pages in enumerate(evidence_pages)
    )
    return RAGResult(
        question="What was the figure?",
        answer="The figure was 7. [C1]",
        citations=(),
        evidence=tuple(EvidenceItem(number=i + 1, text="some passage", retrieved=r) for i, r in enumerate(shown)),
        retrieved=retrieved,
        abstained=False,
        metadata={"pipeline": "test", **metadata},
    )


RERANKED_STAGES = ["retrieval", "reranking", "context-selection", "generation"]


class TestRerankingErrorCategory:
    """DD-044. The boundary between ``retrieval`` and ``reranking`` is the
    counterfactual answer path: what the generator would have been handed with
    the reranker off."""

    def test_gold_pushed_out_by_the_reranker_is_a_reranking_failure(self):
        result = result_with(((4,), (5,)), stages=RERANKED_STAGES, pre_rerank_pages=[3, 4])
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == RERANKING_FAILURE

    def test_the_same_record_without_a_declared_reranker_is_a_retrieval_failure(self):
        result = result_with(((4,), (5,)), pre_rerank_pages=[3, 4])
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == RETRIEVAL_FAILURE

    def test_gold_the_reranker_was_never_handed_is_a_retrieval_failure(self):
        """Deeper in the pool, or dropped by fusion: the reranker did not lose
        it, because the reranker-off system would not have had it either."""
        result = result_with(((4,), (5,)), stages=RERANKED_STAGES, pre_rerank_pages=[4, 6])
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == RETRIEVAL_FAILURE

    def test_gold_the_reranker_kept_is_not_a_reranking_failure(self):
        result = result_with(
            ((3,), (5,)), evidence_pages=((5,),), stages=RERANKED_STAGES, pre_rerank_pages=[3]
        )
        run = score_result(question_on((3,)), result, system="test")
        assert run.score.error_category == CONTEXT_SELECTION_FAILURE

    def test_a_multi_page_question_keeping_one_gold_page_was_not_lost(self):
        result = result_with(((4,),), stages=RERANKED_STAGES, pre_rerank_pages=[3, 4])
        run = score_result(question_on((3, 4)), result, system="test")
        assert run.score.error_category != RERANKING_FAILURE

    def test_end_to_end_through_the_real_pipeline(self, offline_config, structured_pdf_path):
        """A real ``HybridRAGPipeline`` whose reranker inverts the pool, scored
        by the real harness, on a question whose gold page is exactly the page
        the pre-rerank top-1 had and the post-rerank top-1 lacks."""
        config = hybrid_config(offline_config)
        config = dataclasses.replace(
            config, retrieval=dataclasses.replace(config.retrieval, top_k=1)
        )
        pipeline = HybridRAGPipeline(config, reranker=Reverse())
        pipeline.index(structured_pdf_path)

        text = "How are chunks embedded with a bi-encoder?"
        result = pipeline.ask(text)
        before = set(result.metadata["pre_rerank_pages"])
        after = {p for r in result.retrieved for p in r.pages}
        gold = tuple(sorted(before - after))
        assert gold, "fixture: inverting the pool must move the top-1 to another page"

        run = score_result(question_on(gold), result, system=pipeline.name)
        assert run.score.failed
        assert run.score.error_category == RERANKING_FAILURE
        assert run.record["metrics"]["error_category"] == RERANKING_FAILURE

        # The same pipeline with the reranker off answers from the gold page.
        off = HybridRAGPipeline(dataclasses.replace(
            config, reranking=dataclasses.replace(config.reranking, enabled=False)
        ))
        off.index(structured_pdf_path)
        off_pages = {p for r in off.ask(text).retrieved for p in r.pages}
        assert set(gold) <= off_pages


# ---------------------------------------------------------------------------
# The real dataset
# ---------------------------------------------------------------------------


#: Scores are stored rounded to 6 decimals, and the embedder's float arithmetic
#: differs in the last bits between platforms (the stored runs were made on
#: Windows; CI runs Linux). A value on a rounding boundary can then differ by one
#: unit in the 6th decimal, and an nDCG in its 16th digit. Everything else --
#: answers, citations, chunk ids, pages, ranks, verdicts -- must match exactly.
FLOAT_TOLERANCE = 2e-6


def _same(new, old) -> bool:
    """Record equality, with floats equal to within ``FLOAT_TOLERANCE``."""
    if isinstance(new, bool) or isinstance(old, bool):
        return type(new) is type(old) and new == old
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return math.isclose(new, old, rel_tol=1e-9, abs_tol=FLOAT_TOLERANCE)
    if isinstance(new, dict) and isinstance(old, dict):
        return new.keys() == old.keys() and all(_same(new[k], old[k]) for k in new)
    if isinstance(new, list) and isinstance(old, list):
        return len(new) == len(old) and all(_same(a, b) for a, b in zip(new, old))
    return new == old


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark/questions.json is not present")
class TestBaselineAIsUnchanged:
    """Phase 4 refactored the pipeline and changed a taxonomy rule. Baseline A
    is the fixed comparison point, so a fresh dense run must reproduce the
    stored ``results/baseline/per_question.json`` on every field except the ones
    that cannot be equal: timings, the config fingerprint (new config fields
    change the hash) and the two version strings. This is the evidence
    ``metrics.SCORE_COMPATIBLE_VERSIONS`` cites."""

    VOLATILE = frozenset({
        "retrieval_s", "generation_s", "latency_s", "latency", "config_fingerprint",
        "scoring_rules_version", "error_taxonomy_version",
    })

    def _strip(self, value):
        if isinstance(value, dict):
            return {k: self._strip(v) for k, v in value.items() if k not in self.VOLATILE}
        if isinstance(value, list):
            return [self._strip(v) for v in value]
        return value

    def test_a_fresh_dense_run_reproduces_the_stored_baseline(self, tmp_path):
        stored = json.loads((BASELINE / "per_question.json").read_text(encoding="utf-8"))
        run_benchmark_cli(
            questions_path=BENCHMARK,
            documents_dir=DOCUMENTS,
            out_dir=tmp_path,
            config=RAGConfig.offline(),
        )
        fresh = json.loads((tmp_path / "per_question.json").read_text(encoding="utf-8"))
        assert len(fresh) == len(stored)
        for old, new in zip(self._strip(stored), self._strip(fresh)):
            assert _same(new, old), (old["question_id"], new, old)


@pytest.fixture(scope="module")
def hybrid_run(tmp_path_factory):
    if not BENCHMARK.exists():  # pragma: no cover
        pytest.skip("benchmark/questions.json is not present")
    out = tmp_path_factory.mktemp("hybrid")
    config = RAGConfig.offline()
    config = dataclasses.replace(
        config, retrieval=dataclasses.replace(config.retrieval, strategy="hybrid")
    )
    run = run_benchmark_cli(
        questions_path=BENCHMARK, documents_dir=DOCUMENTS, out_dir=out, config=config
    )
    return run, out


class TestBaselineBIsUnchanged:
    """Phase 5 changed the pipeline's hybrid ranking code (``_pools``) and the
    taxonomy (DD-054). Baseline B is a fixed comparison point, so a fresh run
    must reproduce ``results/hybrid/per_question.json`` on every field except
    those that cannot be equal. With ``TestBaselineAIsUnchanged`` this is the
    evidence ``metrics.SCORE_COMPATIBLE_VERSIONS["2026-09-23.1"]`` cites."""

    def test_a_fresh_hybrid_run_reproduces_the_stored_baseline_b(self, hybrid_run):
        _, out = hybrid_run
        volatile = TestBaselineAIsUnchanged.VOLATILE | {"rerank_s"}

        def strip(value):
            if isinstance(value, dict):
                return {k: strip(v) for k, v in value.items() if k not in volatile}
            if isinstance(value, list):
                return [strip(v) for v in value]
            return value

        stored = json.loads((HYBRID / "per_question.json").read_text(encoding="utf-8"))
        fresh = json.loads((out / "per_question.json").read_text(encoding="utf-8"))
        assert len(fresh) == len(stored)
        for old, new in zip(strip(stored), strip(fresh)):
            assert _same(new, old), (old["question_id"], new, old)


class TestFullBaselineBSmokeRun:
    """All 76 questions through Baseline B, offline, through the unchanged
    harness entry point -- mirroring Phase 3's run for Baseline A."""

    def test_every_question_is_answered_scored_and_categorised(self, hybrid_run):
        run, _ = hybrid_run
        questions = load_questions(BENCHMARK)
        assert run.system == "baseline_hybrid"
        assert {s.question_id for s in run.scores} == {q.id for q in questions}
        for score in run.scores:
            assert score.error is None, f"{score.question_id}: {score.error}"
            if score.failed:
                assert score.error_category in ERROR_CATEGORIES, score.question_id
            else:
                assert score.error_category is None
        summary = run.summary()
        assert summary["n_retrieval_scored"] == summary["n_answerable"]
        for key in ("recall_at_5", "recall_at_10", "mrr_at_10", "accuracy_all"):
            assert summary[key] is not None, key

    def test_the_four_files_and_the_config_say_what_ran(self, hybrid_run):
        _, out = hybrid_run
        for name in ("config.json", "results.json", "per_question.json", "summary.csv"):
            assert (out / name).exists(), name
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["system"] == "baseline_hybrid"
        assert config["retrieval_strategy"] == "hybrid"
        assert config["retrievers"] == ["dense", "lexical"]
        assert config["reranker_enabled"] is True
        assert config["reranker_model"] == OVERLAP_RERANKER
        assert config["rerank_k"] == 20
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert results["confidence_intervals"]["metrics"]["recall_at_5"]["n"] == 70

    def test_every_record_declares_its_stages_and_pre_rerank_pages(self, hybrid_run):
        run, _ = hybrid_run
        for record in run.records:
            assert record["stages"] == RERANKED_STAGES
            assert isinstance(record["pre_rerank_pages"], list)
