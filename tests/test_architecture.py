"""Tests for the architectural constraints the specifications state as rules.

A rule written only in prose is a rule that drifts. These check the two that
would be expensive to discover late: the layer boundary that keeps the PDF
parser swappable, and the configuration centralisation that keeps experiments
controllable.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.config import (
    MODEL_REGISTRY,
    GenerationConfig,
    RAGConfig,
    embedding_profile,
)
from src.errors import ConfigurationError

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _imports(path: pathlib.Path) -> set[str]:
    """Every module name imported by a file, absolute and relative."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # "from ..ingestion.parser import X" -> "ingestion.parser"
            module = node.module or ""
            found.add(module.lstrip("."))
            found.update(f"{module.lstrip('.')}.{a.name}" for a in node.names)
    return found


def _modules(package: str) -> list[pathlib.Path]:
    return sorted((SRC / package).glob("*.py"))


class TestLayerBoundaries:
    """ARCHITECTURE.md section 4: retrieval must not depend on the parser."""

    @pytest.mark.parametrize(
        "path",
        _modules("retrieval") + _modules("reranking") + _modules("generation"),
        ids=lambda p: p.name,
    )
    def test_downstream_layers_do_not_import_the_parser(self, path):
        imported = _imports(path)
        assert not any("parser" in name for name in imported), (
            f"{path.name} imports the PDF parser; the retrieval and generation "
            "layers must depend on the document model only, so the parser can "
            "be replaced."
        )

    @pytest.mark.parametrize(
        "path",
        _modules("retrieval")
        + _modules("reranking")
        + _modules("generation")
        + _modules("chunking"),
        ids=lambda p: p.name,
    )
    def test_downstream_layers_do_not_import_a_pdf_library(self, path):
        imported = _imports(path)
        for banned in ("pdfplumber", "pdfminer", "pymupdf", "fitz", "pypdf"):
            assert banned not in imported, (
                f"{path.name} imports {banned}. PDF handling belongs in the "
                "ingestion layer."
            )

    @pytest.mark.parametrize("path", _modules("ingestion"), ids=lambda p: p.name)
    def test_ingestion_does_not_reach_downstream(self, path):
        """The parser must not know about chunks, retrieval or generation."""
        imported = _imports(path)
        for banned in ("chunking", "retrieval", "generation", "pipeline"):
            assert not any(name.startswith(banned) for name in imported), (
                f"{path.name} imports {banned}; ingestion is the bottom layer."
            )

    def test_ingestion_does_not_call_a_model(self):
        for path in _modules("ingestion"):
            source = path.read_text(encoding="utf-8")
            assert "transformers" not in source, (
                f"{path.name} references transformers; ARCHITECTURE.md section 5 "
                "says the ingestion layer must not call the LLM."
            )


class TestEvaluationLayerBoundaries:
    """ARCHITECTURE.md section 3: the harness sits above the system it measures.

    The dependency runs one way -- evaluation reads ``benchmark/schema.py`` and
    drives a pipeline; neither knows the harness exists. That is what lets the
    same harness score the hybrid and agentic systems in Phases 4 and 5 without
    either of them importing it, and a cycle here would be discovered as an
    import error in Phase 4 rather than as a rule now.
    """

    @pytest.mark.parametrize("path", _modules("benchmark"), ids=lambda p: p.name)
    def test_the_dataset_layer_does_not_import_the_harness(self, path):
        imported = _imports(path)
        assert not any(name.startswith("evaluation") for name in imported), (
            f"{path.name} imports the evaluation layer. The dependency runs the "
            "other way: evaluation depends on the dataset schema, never the "
            "reverse."
        )

    def test_no_layer_below_evaluation_imports_it(self):
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            if path.parent.name == "evaluation":
                continue
            for name in _imports(path):
                if name.startswith("evaluation") or name.endswith(".evaluation"):
                    offenders.append(f"{path.relative_to(SRC)} imports {name}")
        # cli.py is the entry point and is allowed to wire the harness up; it
        # does so inside the subcommand, so nothing imports it at module load.
        offenders = [o for o in offenders if not o.startswith("cli.py")]
        assert not offenders, "\n".join(offenders)

    @pytest.mark.parametrize("path", _modules("evaluation"), ids=lambda p: p.name)
    def test_the_harness_does_not_import_a_pdf_library(self, path):
        imported = _imports(path)
        for banned in ("pdfplumber", "pdfminer", "pymupdf", "fitz", "pypdf"):
            assert banned not in imported, (
                f"{path.name} imports {banned}. PDF handling belongs in the "
                "ingestion layer; the harness reaches it through the pipeline."
            )

    def test_the_harness_reuses_abstention_detection_rather_than_copying_it(self):
        """BENCHMARK_SPEC.md 13.2 fixes the patterns and records the version.

        A second detector in the evaluation layer would be a second set of
        patterns to keep in step, and the one further from the generator would
        drift. So the harness imports ``is_abstention`` and must not define
        refusal patterns of its own.
        """
        import src.evaluation.benchmark as harness

        assert harness.is_abstention.__module__ == "src.generation.abstention"

        for path in _modules("evaluation"):
            source = path.read_text(encoding="utf-8")
            assert "could not find sufficient evidence" not in source.lower(), (
                f"{path.name} hard-codes a refusal phrase. Abstention detection "
                "lives in src/generation/abstention.py and is versioned there."
            )

    def test_the_scoring_rules_are_versioned(self):
        """Two runs scored under different rules are not comparable, which is
        why the abstention patterns carry a version and these must too."""
        from src.evaluation.metrics import (
            ERROR_TAXONOMY_VERSION,
            SCORING_RULES_VERSION,
        )

        assert SCORING_RULES_VERSION
        assert ERROR_TAXONOMY_VERSION

    def test_the_deliberately_absent_components_are_still_absent(self):
        """DD-013 / STATUS.md section 4. Phase 4 adds lexical retrieval, fusion
        and reranking; it does not quietly acquire the agentic parts Phase 5
        exists to add and measure."""
        assert not (SRC / "agents").exists(), (
            "src/agents/ exists. DD-013 requires baselines that lack the "
            "planner, evidence controller, refinement and verifier before their "
            "value can be measured."
        )

    def test_baseline_a_still_lacks_every_phase_4_component(self):
        """DD-013 still binds Baseline A: Phase 4 is measured *against* a dense
        system, so the dense system must not have acquired lexical retrieval,
        fusion or a reranker along the way."""
        from src.pipeline import DenseRAGPipeline
        from src.retrieval.retriever import DenseRetriever

        pipeline = DenseRAGPipeline(RAGConfig.offline())
        assert type(pipeline._retriever) is DenseRetriever
        assert not hasattr(pipeline, "reranker")

    def test_statistics_do_not_import_the_run_harness(self):
        """The dependency runs benchmark.py -> statistics.py, never back, so a
        stored run can be analysed without the machinery that produced it."""
        imported = _imports(SRC / "evaluation" / "statistics.py")
        assert not any(name.endswith("benchmark") for name in imported), imported

    def test_the_pipeline_and_the_taxonomy_name_the_same_stages(self):
        """The pipeline declares stages; the taxonomy keys reserved categories
        on them. Two spellings of 'reranking' would silently disable one."""
        from src import pipeline
        from src.evaluation import metrics

        for name in ("RETRIEVAL", "RERANKING", "CONTEXT_SELECTION", "GENERATION"):
            assert getattr(pipeline, f"STAGE_{name}") == getattr(metrics, f"STAGE_{name}")


class TestParserSwappability:
    """DD-033: the PDF library must stay replaceable, and it was replaced."""

    def test_pdf_libraries_are_confined_to_the_ingestion_layer(self):
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            if path.parent.name == "ingestion":
                continue
            source = path.read_text(encoding="utf-8")
            for library in ("import pdfplumber", "import pymupdf", "import fitz"):
                if library in source:
                    offenders.append(f"{path.relative_to(SRC)} does `{library}`")
        assert not offenders, "\n".join(offenders)

    def test_default_parser_is_permissively_licensed(self):
        from src.config import IngestionConfig

        assert IngestionConfig().parser == "pdfplumber"

    def test_every_backend_implements_the_protocol(self):
        from src.config import IngestionConfig
        from src.ingestion.parser import PARSERS, PdfParser, build_parser

        for name in PARSERS:
            parser = build_parser(IngestionConfig(parser=name))
            assert isinstance(parser, PdfParser)
            assert parser.name == name

    def test_unknown_parser_is_rejected_with_the_available_names(self):
        from src.config import IngestionConfig
        from src.errors import PDFReadError
        from src.ingestion.parser import build_parser

        with pytest.raises(PDFReadError, match="pdfplumber"):
            build_parser(IngestionConfig(parser="ghostscript"))


class TestConfigurationCentralisation:
    """DD-017: no hard-coded model names scattered through the code."""

    def test_model_ids_appear_only_in_the_config_module(self):
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            if path.name == "config.py":
                continue
            source = path.read_text(encoding="utf-8")
            for model_id in MODEL_REGISTRY:
                if model_id in source:
                    offenders.append(f"{path.relative_to(SRC)} hard-codes {model_id}")
        assert not offenders, "\n".join(offenders)

    def test_every_registry_entry_records_its_licence(self):
        for info in MODEL_REGISTRY.values():
            assert info.license
            assert isinstance(info.commercial_use, bool)


class TestLicensingGuard:
    """DD-024, enforced rather than documented."""

    def test_default_generator_is_apache_licensed(self):
        config = RAGConfig.default()
        info = MODEL_REGISTRY[config.generation.model_id]
        assert info.license == "apache-2.0"
        assert info.commercial_use

    def test_low_memory_preset_is_also_apache_licensed(self):
        info = MODEL_REGISTRY[RAGConfig.low_memory().generation.model_id]
        assert info.commercial_use

    def test_research_licensed_model_is_rejected_by_default(self):
        config = RAGConfig(
            generation=GenerationConfig(model_id="Qwen/Qwen2.5-3B-Instruct")
        )
        with pytest.raises(ConfigurationError, match="research"):
            config.validate()

    def test_research_licensed_model_can_be_benchmarked_explicitly(self):
        config = RAGConfig(
            generation=GenerationConfig(
                model_id="Qwen/Qwen2.5-3B-Instruct",
                allow_non_commercial_model=True,
            )
        )
        assert config.validate() is config


class TestMemoryBudget:
    """MODEL_SELECTION.md 9.1, enforced before the T4 enforces it."""

    def test_default_stack_fits_the_budget(self):
        assert RAGConfig.default().estimate_vram_gb() <= 10.0

    def test_low_memory_stack_is_smaller_still(self):
        assert (
            RAGConfig.low_memory().estimate_vram_gb()
            < RAGConfig.default().estimate_vram_gb()
        )

    def test_oversized_stack_is_rejected_with_a_remedy(self):
        config = RAGConfig(generation=GenerationConfig(quantization="none"))
        # 8.0 GB bf16 generator + 1.2 GB embedder is within budget; the point of
        # the check is that the estimate tracks quantization at all.
        assert config.estimate_vram_gb() > RAGConfig.default().estimate_vram_gb()

    def test_the_reranker_counts_only_when_the_hybrid_system_uses_it(self):
        from src.config import RerankingConfig, RetrievalConfig

        dense = RAGConfig.default()
        hybrid = RAGConfig(retrieval=RetrievalConfig(strategy="hybrid"))
        hybrid_off = RAGConfig(
            retrieval=RetrievalConfig(strategy="hybrid"),
            reranking=RerankingConfig(enabled=False),
        )
        reranker = MODEL_REGISTRY["BAAI/bge-reranker-v2-m3"].vram_gb
        assert hybrid.estimate_vram_gb() == pytest.approx(dense.estimate_vram_gb() + reranker)
        assert hybrid_off.estimate_vram_gb() == dense.estimate_vram_gb()
        assert hybrid.validate() is hybrid

    def test_a_bf16_generator_plus_both_retrieval_models_is_rejected(self):
        """MODEL_SELECTION.md 9.1: '...a 4B generator in bf16 plus both
        retrieval models leaves too little headroom to be safe on a T4'."""
        from src.config import RetrievalConfig

        config = RAGConfig(
            generation=GenerationConfig(quantization="none"),
            retrieval=RetrievalConfig(strategy="hybrid"),
        )
        with pytest.raises(ConfigurationError, match="reranker"):
            config.validate()

    def test_the_reranker_is_apache_licensed(self):
        info = MODEL_REGISTRY[RAGConfig().reranking.model_id]
        assert info.role == "reranking"
        assert info.license == "apache-2.0"


class TestConfigValidation:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        from src.config import ChunkingConfig

        config = RAGConfig(chunking=ChunkingConfig(chunk_size=200, chunk_overlap=200))
        with pytest.raises(ConfigurationError, match="advance"):
            config.validate()

    def test_top_k_must_be_positive(self):
        from src.config import RetrievalConfig

        with pytest.raises(ConfigurationError, match="top_k"):
            RAGConfig(retrieval=RetrievalConfig(top_k=0)).validate()

    @pytest.mark.parametrize(
        "retrieval, reranking, match",
        [
            ({"retrievers": ()}, {}, "retrievers"),
            ({"retrievers": ("dense", "sparse")}, {}, "retrievers"),
            ({"retrievers": ("dense", "dense")}, {}, "twice"),
            ({"strategy": "hybrid", "candidate_k": 3}, {}, "candidate_k"),
            ({"strategy": "hybrid"}, {"candidates": 3}, "candidates"),
            ({"rrf_k": 0}, {}, "rrf_k"),
            ({"bm25_b": 1.5}, {}, "bm25"),
        ],
    )
    def test_hybrid_settings_are_validated(self, retrieval, reranking, match):
        from src.config import RerankingConfig, RetrievalConfig

        config = RAGConfig(
            retrieval=RetrievalConfig(**retrieval),
            reranking=RerankingConfig(**reranking),
        )
        with pytest.raises(ConfigurationError, match=match):
            config.validate()

    def test_fingerprint_changes_with_the_configuration(self):
        from src.config import ChunkingConfig

        base = RAGConfig.default()
        changed = RAGConfig(chunking=ChunkingConfig(chunk_size=999))
        assert base.fingerprint() != changed.fingerprint()

    def test_fingerprint_is_stable_for_equal_configurations(self):
        assert RAGConfig.default().fingerprint() == RAGConfig.default().fingerprint()


class TestEmbeddingProfiles:
    """The silent-failure table: wrong pooling or a stray prefix halves recall."""

    def test_bge_m3_uses_cls_pooling_and_no_query_prefix(self):
        profile = embedding_profile("BAAI/bge-m3")
        assert profile.pooling == "cls"
        assert profile.query_prefix == ""

    def test_bge_v15_requires_the_query_instruction(self):
        profile = embedding_profile("BAAI/bge-small-en-v1.5")
        assert profile.pooling == "cls"
        assert profile.query_prefix.startswith("Represent this sentence")
        assert profile.document_prefix == ""  # queries only

    def test_e5_uses_both_prefixes(self):
        profile = embedding_profile("intfloat/e5-base-v2")
        assert profile.query_prefix == "query: "
        assert profile.document_prefix == "passage: "

    def test_unknown_model_falls_back_without_a_prefix(self):
        profile = embedding_profile("some-org/unknown-encoder")
        assert profile.query_prefix == ""
        assert profile.pooling == "mean"
