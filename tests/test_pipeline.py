"""End-to-end tests for the dense baseline.

Run entirely on the fallback components: the hashing embedder and the scripted
backend. Both are weak, which is the point -- these tests check that the wiring,
the metadata and the citations are correct, not that the answers are good. Answer
quality is Phase 3's question and needs a benchmark, which does not exist yet.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import ChunkingConfig, RAGConfig, RetrievalConfig
from src.errors import IndexNotBuiltError, ScannedPDFError
from src.generation.prompts import ABSTENTION_SENTENCE
from src.pipeline import DenseRAGPipeline

from . import pdf_fixtures as pdfs


@pytest.fixture
def pipeline(offline_config, structured_pdf_path):
    pipe = DenseRAGPipeline(offline_config)
    pipe.index(structured_pdf_path)
    return pipe


class TestIndexing:
    def test_index_reports_what_it_built(self, pipeline):
        stats = pipeline.index_stats
        assert stats["pages"] == 2
        assert stats["chunks"] > 0
        assert stats["chunk_strategy"] == "structure"
        assert stats["index"] in {"faiss", "numpy"}
        assert stats["config_fingerprint"]

    def test_index_records_the_model_stack_for_reproducibility(self, pipeline):
        stats = pipeline.index_stats
        assert "embedding_model" in stats
        assert "generation_model" in stats
        assert "chunk_size" in stats

    def test_asking_before_indexing_raises(self, offline_config):
        with pytest.raises(IndexNotBuiltError, match="index"):
            DenseRAGPipeline(offline_config).ask("anything")

    def test_scanned_pdf_fails_at_indexing(self, offline_config, tmp_path):
        path = tmp_path / "scan.pdf"
        path.write_bytes(pdfs.scanned_pdf(3))
        with pytest.raises(ScannedPDFError, match="OCR|ocr"):
            DenseRAGPipeline(offline_config).index(path)


class TestAnswering:
    def test_answer_carries_citations_and_evidence(self, pipeline):
        result = pipeline.ask("What recall did the hybrid system achieve?")
        assert not result.abstained
        assert result.answer
        assert result.citations
        assert result.evidence
        assert result.retrieved

    def test_citations_point_at_the_page_the_evidence_came_from(self, pipeline):
        """The table is on page 2, so the citation must say page 2."""
        result = pipeline.ask("What recall did the hybrid system achieve?")
        assert result.cited_pages == (2,)
        for citation in result.citations:
            evidence = next(
                e for e in result.evidence if e.marker == citation.marker
            )
            assert citation.pages == evidence.chunk.pages

    def test_every_citation_traces_back_to_retrieved_evidence(self, pipeline):
        result = pipeline.ask("How are chunks embedded?")
        retrieved_ids = {r.chunk_id for r in result.retrieved}
        for citation in result.citations:
            assert citation.chunk_id in retrieved_ids

    def test_unanswerable_question_abstains(self, pipeline):
        result = pipeline.ask("What is the population of Reykjavik?")
        assert result.abstained
        assert result.answer == ABSTENTION_SENTENCE
        assert result.citations == ()

    def test_result_metadata_supports_debugging(self, pipeline):
        result = pipeline.ask("What recall did the hybrid system achieve?")
        meta = result.metadata
        assert meta["pipeline"] == "baseline_dense"
        assert meta["retrieval_strategy"] == "dense"
        assert len(meta["retrieved_chunk_ids"]) == len(result.retrieved)
        assert len(meta["retrieval_scores"]) == len(result.retrieved)
        assert meta["latency_s"] >= 0
        assert meta["config_fingerprint"]

    def test_result_record_is_serialisable(self, pipeline):
        import json

        record = pipeline.ask("What recall did the hybrid system achieve?").to_record()
        assert json.loads(json.dumps(record, default=str))["citations"]

    def test_formatted_output_shows_pages_not_markers_alone(self, pipeline):
        rendered = pipeline.ask("What recall did the hybrid system achieve?").format()
        assert "page 2" in rendered

    def test_top_k_can_be_overridden_per_question(self, pipeline):
        result = pipeline.ask("What was measured?", top_k=2)
        assert len(result.retrieved) <= 2
        assert result.metadata["top_k"] == 2

    def test_answers_are_deterministic(self, pipeline):
        question = "What recall did the hybrid system achieve?"
        first = pipeline.ask(question)
        second = pipeline.ask(question)
        assert first.answer == second.answer
        assert first.cited_pages == second.cited_pages


class TestPageAttributionEndToEnd:
    """The exit criterion: chunk page numbers verified against the source."""

    @pytest.mark.parametrize("strategy", ["structure", "fixed"])
    def test_cited_page_contains_the_answer_text(self, tmp_path, strategy):
        path = tmp_path / "marked.pdf"
        path.write_bytes(pdfs.multipage_pdf(6))

        config = dataclasses.replace(
            RAGConfig.offline(),
            chunking=ChunkingConfig(strategy=strategy, chunk_size=300, chunk_overlap=60),
            retrieval=RetrievalConfig(top_k=3),
        )
        pipe = DenseRAGPipeline(config)
        document = pipe.index(path)

        result = pipe.ask(f"What does {pdfs.PAGE_MARKER.format(n=4)} refer to?")
        assert result.retrieved
        top = result.retrieved[0]
        assert 4 in top.pages
        # ...and the page the chunk claims really does contain that text.
        for page_number in top.pages:
            page = document.page(page_number)
            assert page.text  # the claimed page is real and non-empty
        assert pdfs.PAGE_MARKER.format(n=4) in "".join(
            document.page(p).text for p in top.pages
        )


class TestStrategySwapping:
    def test_both_chunking_strategies_answer_end_to_end(
        self, structured_pdf_path, offline_config
    ):
        for strategy in ("structure", "fixed"):
            config = dataclasses.replace(
                offline_config,
                chunking=dataclasses.replace(offline_config.chunking, strategy=strategy),
            )
            pipe = DenseRAGPipeline(config)
            pipe.index(structured_pdf_path)
            result = pipe.ask("What recall did the hybrid system achieve?")
            assert result.answer
            assert result.citations
            assert pipe.index_stats["chunk_strategy"] == strategy

    def test_numpy_index_gives_the_same_answer_as_faiss(
        self, structured_pdf_path, offline_config
    ):
        question = "What recall did the hybrid system achieve?"
        answers = []
        for index in ("faiss", "numpy"):
            config = dataclasses.replace(
                offline_config,
                retrieval=dataclasses.replace(offline_config.retrieval, index=index),
            )
            pipe = DenseRAGPipeline(config)
            pipe.index(structured_pdf_path)
            answers.append(pipe.ask(question).answer)
        assert answers[0] == answers[1]


class TestRetrievalThreshold:
    def test_threshold_short_circuits_to_abstention(
        self, structured_pdf_path, offline_config
    ):
        config = dataclasses.replace(
            offline_config,
            retrieval=dataclasses.replace(offline_config.retrieval, min_score=0.99),
        )
        pipe = DenseRAGPipeline(config)
        pipe.index(structured_pdf_path)
        result = pipe.ask("What recall did the hybrid system achieve?")
        assert result.abstained
        assert result.metadata["abstention_reason"] == "no_evidence"
        # The retrieved chunks are still reported, so the decision is auditable.
        assert result.retrieved

    def test_threshold_is_disabled_by_default(self, offline_config):
        assert offline_config.retrieval.min_score is None
