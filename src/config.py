"""Central configuration (DD-017).

Every model name, chunk size, top-k and threshold in this project comes from
here. Nothing downstream is allowed to hard-code a model identifier; a test in
``tests/test_config.py`` enforces that by scanning the source tree.

Two hard constraints from the specifications are enforced in code rather than
left as prose:

* DD-024 / MODEL_SELECTION.md 11.2 -- a ``qwen-research`` licensed checkpoint
  may be benchmarked but must never be the default generator. ``validate()``
  raises unless the caller explicitly opts in.
* MODEL_SELECTION.md 9.1 -- the resident model budget on a free-tier T4 is
  10 GB with at least 5 GB of headroom. ``estimate_vram_gb()`` adds up the
  configured stack and ``validate()`` refuses a configuration that busts it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from .errors import ConfigurationError

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

Pooling = Literal["cls", "mean"]


@dataclass(frozen=True)
class ModelInfo:
    """Verified facts about a candidate model.

    Licences were checked against the Hugging Face model cards on 2026-09-20
    (MODEL_SELECTION.md 11.1). ``vram_gb`` figures are estimates from parameter
    counts, superseded by measurement once Phase 3 records real numbers.
    """

    model_id: str
    role: Literal["generation", "embedding", "reranking"]
    license: str
    commercial_use: bool
    vram_gb: float
    note: str = ""


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "Qwen/Qwen3-4B-Instruct-2507": ModelInfo(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        role="generation",
        license="apache-2.0",
        commercial_use=True,
        vram_gb=8.0,
        note="bf16 estimate. Around 2.5 GB at 4-bit NF4.",
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelInfo(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        role="generation",
        license="apache-2.0",
        commercial_use=True,
        vram_gb=3.1,
        note="Low-memory preset.",
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelInfo(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        role="generation",
        license="qwen-research",
        commercial_use=False,
        vram_gb=6.2,
        note="DD-024: research/non-commercial only. Benchmark candidate, never the default.",
    ),
    "BAAI/bge-m3": ModelInfo(
        model_id="BAAI/bge-m3",
        role="embedding",
        license="mit",
        commercial_use=True,
        vram_gb=1.2,
        note="fp16. CLS pooling, no query instruction prefix.",
    ),
    "BAAI/bge-reranker-v2-m3": ModelInfo(
        model_id="BAAI/bge-reranker-v2-m3",
        role="reranking",
        license="apache-2.0",
        commercial_use=True,
        vram_gb=1.2,
        note="Phase 4 (Baseline B). Never loaded by the dense baseline.",
    ),
}

#: Dependency-free backends used when no weights are available.
HASHING_EMBEDDER = "local/hashing-embedder"
SCRIPTED_BACKEND = "local/scripted-extractive"
#: Dependency-free reranker: joint query/passage term coverage (DD-048).
OVERLAP_RERANKER = "local/term-overlap-reranker"
#: Rule-based stand-ins for the agentic system's model-backed decisions
#: (DD-055). Deterministic code, labelled as such wherever a record names the
#: component that decided, so no offline figure is read as a model's.
RULE_PLANNER = "local/rule-based-planner"
RULE_EVIDENCE_CONTROLLER = "local/rule-based-evidence-controller"
RULE_VERIFIER = "local/rule-based-verifier"


@dataclass(frozen=True)
class EmbeddingProfile:
    """How a given embedding family must be called.

    Getting this wrong is silent: BGE-M3 with mean pooling, or a BGE v1.5 model
    without its query instruction, still returns plausible vectors and simply
    retrieves worse. It is a lookup table rather than a guess at the call site.
    """

    pooling: Pooling
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True


#: Keys are matched as case-insensitive substrings of the model id, longest first.
EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    # BGE-M3 is trained without an instruction prefix. Adding one hurts.
    "bge-m3": EmbeddingProfile(pooling="cls", query_prefix=""),
    # BGE v1.5 models require the retrieval instruction, on the query side only.
    "bge-large-en-v1.5": EmbeddingProfile(
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-base-en-v1.5": EmbeddingProfile(
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-small-en-v1.5": EmbeddingProfile(
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "e5-": EmbeddingProfile(
        pooling="mean", query_prefix="query: ", document_prefix="passage: "
    ),
    HASHING_EMBEDDER: EmbeddingProfile(pooling="mean", query_prefix=""),
}

#: Fallback when a model id matches no profile. Mean pooling with no prefix is
#: the least-surprising default, but a caller relying on it should check the
#: model card rather than discover halved recall through a benchmark.
DEFAULT_EMBEDDING_PROFILE = EmbeddingProfile(pooling="mean", query_prefix="")


def embedding_profile(model_id: str) -> EmbeddingProfile:
    """Return the calling convention for ``model_id``."""
    lowered = model_id.lower()
    for key in sorted(EMBEDDING_PROFILES, key=len, reverse=True):
        if key.lower() in lowered:
            return EMBEDDING_PROFILES[key]
    return DEFAULT_EMBEDDING_PROFILE


# ---------------------------------------------------------------------------
# Colab budget (MODEL_SELECTION.md 9.1)
# ---------------------------------------------------------------------------

T4_USABLE_VRAM_GB = 15.0
MODEL_VRAM_BUDGET_GB = 10.0
REQUIRED_HEADROOM_GB = 5.0

#: Multiplier applied to a bf16 estimate for quantized weights.
QUANT_VRAM_FACTOR: dict[str, float] = {"none": 1.0, "8bit": 0.55, "4bit": 0.32}


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionConfig:
    """PDF parsing and text normalization."""

    #: pdfplumber (MIT) is the default. PyMuPDF is AGPL-3.0 and would impose
    #: copyleft on everything downstream; see DD-033.
    parser: Literal["pdfplumber", "pymupdf"] = "pdfplumber"
    password: str | None = None

    # Normalization. Each is individually switchable so its effect on retrieval
    # can be ablated later.
    fix_ligatures: bool = True
    join_hyphenated_linebreaks: bool = True
    strip_running_headers: bool = True
    normalize_quotes: bool = True

    #: A repeated first/last line must appear on at least this fraction of pages
    #: before it is treated as a running header or footer.
    header_footer_min_page_fraction: float = 0.5
    #: ...and on at least this many pages, so a 2-page document is left alone.
    header_footer_min_pages: int = 3
    #: Fraction of page height at the top and bottom treated as the margin band.
    #: Only text lying entirely inside a band can be page furniture. Position is
    #: a far better signal than line order: the first line of body text is also
    #: the "first line of the page", and documents do repeat opening phrases.
    header_footer_margin_fraction: float = 0.12
    #: Leading/trailing line count used only when the parser reports no
    #: geometry, in which case position is unavailable.
    header_footer_scan_lines: int = 2

    # Scanned-document detection.
    #: A page with fewer than this many characters is "text-empty".
    min_chars_per_page: int = 40
    #: Document is treated as scanned when at least this fraction of pages are
    #: text-empty *and* carry images.
    scanned_page_fraction: float = 0.8
    #: ...and the whole document yielded fewer than this many characters.
    min_document_chars: int = 200

    # OCR (DD-067). On in every preset: "any uploaded PDF" includes scanned
    # ones. Only a page with images and no text layer at all is OCR'd, so a
    # PDF with text on every page parses exactly as it would with OCR off.
    ocr: bool = True
    #: Tesseract language codes, joined with "+" (for example ``"eng+deu"``).
    ocr_languages: str = "eng"
    #: Render resolution for OCR, in DPI; 300 is what Tesseract is tuned for.
    ocr_resolution: int = 300


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk geometry.

    Sizes are in **characters**, not tokens: a token-based size would couple the
    chunker to one specific tokenizer, which the swappable-component requirement
    in ARCHITECTURE.md section 4 argues against. Roughly 4 characters per token.

    EXPERIMENT_PLAN.md section 5.1 records that these values have no principled
    justification yet and are to be swept on the dev set, not tuned by feel.
    """

    strategy: Literal["structure", "fixed"] = "structure"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    #: Chunks shorter than this are dropped as noise (page numbers, stray marks).
    min_chunk_chars: int = 40
    #: A table is emitted as its own chunk. Beyond this length it is truncated
    #: rather than split, so a row is never separated from its header.
    max_table_chars: int = 4000
    #: Join a paragraph continuing across a page break into one segment.
    join_paragraphs_across_pages: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = "BAAI/bge-m3"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "float16", "float32"] = "auto"
    batch_size: int = 8
    max_length: int = 512
    #: Dimension of the dependency-free hashing embedder.
    hashing_dim: int = 1024


#: Retrievers a hybrid system may fuse (ARCHITECTURE.md section 8).
RETRIEVERS: tuple[str, ...] = ("dense", "lexical")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    #: "dense" is Baseline A; "hybrid" is Baseline B (EVALUATION_PROTOCOL.md
    #: section 8); "agentic" is Baseline B's retrieval driven by the agent loop
    #: of section 9 (DD-050). The strategy decides which pipeline
    #: ``build_pipeline`` builds, so the system under test is chosen by
    #: configuration alone (DD-017).
    strategy: Literal["dense", "hybrid", "agentic"] = "dense"
    #: Which retrievers a hybrid system fuses, in tie-break priority order
    #: (DD-045). ("dense",) or ("lexical",) are the ablation arms
    #: EVALUATION_PROTOCOL.md section 24 and DD-007 ask for.
    retrievers: tuple[str, ...] = ("dense", "lexical")
    #: Depth each retriever ranks to before fusion (DD-046).
    candidate_k: int = 20
    #: DD-008. The only method so far; a field so a second one is a config
    #: change rather than a code change (ARCHITECTURE.md section 9).
    fusion: Literal["rrf"] = "rrf"
    #: The RRF constant from Cormack et al. (2009); not tuned here (DD-045).
    rrf_k: int = 60
    #: Okapi BM25 parameters, at the conventional values (DD-049).
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    #: "auto" uses FAISS when importable and falls back to the numpy index.
    index: Literal["auto", "faiss", "numpy"] = "auto"
    #: Abstain without calling the generator when the best score falls below
    #: this. ``None`` disables it: EXPERIMENT_PLAN.md forbids introducing an
    #: untuned threshold, so by default the pipeline only short-circuits when
    #: retrieval returns nothing at all.
    min_score: float | None = None


@dataclass(frozen=True)
class RerankingConfig:
    """Cross-encoder reranking (ARCHITECTURE.md section 10, DD-006).

    Read only by the hybrid pipeline. ``enabled=False`` is the "Reranker OFF"
    ablation of EVALUATION_PROTOCOL.md section 24: the fused pool is passed
    through unchanged, at the same depth, so the two arms differ in the
    reranker and nothing else.
    """

    enabled: bool = True
    model_id: str = "BAAI/bge-reranker-v2-m3"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "float16", "float32"] = "auto"
    batch_size: int = 16
    max_length: int = 512
    #: Size of the fused pool handed to the reranker -- and, with the reranker
    #: off, the depth of the ranking returned unchanged (DD-046).
    candidates: int = 20


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    backend: Literal["auto", "transformers", "scripted"] = "auto"
    quantization: Literal["none", "8bit", "4bit"] = "4bit"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    max_new_tokens: int = 384
    #: Greedy decoding. temperature == 0 maps to do_sample=False, never to a
    #: tiny non-zero temperature.
    temperature: float = 0.0
    #: Character budget for the evidence section of the prompt.
    max_context_chars: int = 6000
    #: Longest single evidence block before it is truncated.
    max_evidence_chars: int = 1600
    #: Set True only to benchmark a research-licensed checkpoint (DD-024).
    allow_non_commercial_model: bool = False


AgentBackend = Literal["llm", "rules"]


@dataclass(frozen=True)
class AgentConfig:
    """The agentic system's components (ARCHITECTURE.md sections 11-17).

    Read only by the agentic pipeline. Each component has its own switch, so
    the ablations of EVALUATION_PROTOCOL.md section 24 -- planner OFF,
    refinement OFF, evidence controller OFF, verification OFF -- are
    configuration changes rather than code changes (DD-050). A component that
    is switched off is not built at all.

    ``"llm"`` components ask the generation model, so no second model is
    loaded (ARCHITECTURE.md section 21); ``"rules"`` components are the
    deterministic stand-ins ``RAGConfig.offline()`` uses (DD-055). Every value
    below was set before any agentic result existed and is untuned
    (EXPERIMENT_PLAN.md section 5).
    """

    planner_enabled: bool = True
    planner: AgentBackend = "llm"
    #: Most queries one question is decomposed into, the original included
    #: (DD-051). A cap, because an LLM planner can return any number.
    max_sub_queries: int = 3

    evidence_controller_enabled: bool = True
    evidence_controller: AgentBackend = "llm"
    #: Share of a query's content terms the selected context must contain for
    #: the rule-based controller to call it covered (DD-052).
    sufficiency_threshold: float = 0.5

    refinement_enabled: bool = True
    #: MAX_RETRIEVAL_ITERATIONS (ARCHITECTURE.md section 13, DD-012): total
    #: retrieval rounds, the first included. 1 means no refinement. Open
    #: question 3 in EXPERIMENT_PLAN.md section 5: chosen for latency.
    max_retrieval_iterations: int = 2
    #: When a second round ran, also generate from the first round's context,
    #: so the record says whether iteration 2 changed the answer (DD-053). A
    #: diagnostic: never returned, and not counted as a system model call.
    record_refinement_counterfactual: bool = True

    verification_enabled: bool = True
    verifier: AgentBackend = "llm"
    #: Share of a claim's content terms its cited evidence must contain for the
    #: rule-based verifier to call it supported (DD-054).
    verifier_support_threshold: float = 0.5
    #: Regenerations after a failed verification before abstaining (DD-054).
    max_regenerations: int = 1


@dataclass(frozen=True)
class RAGConfig:
    """The single configuration object for the whole system."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)

    # -- presets ------------------------------------------------------------

    @classmethod
    def default(cls) -> "RAGConfig":
        """Apache-2.0 generator, MIT embedder, 4-bit for T4 headroom."""
        return cls()

    @classmethod
    def low_memory(cls) -> "RAGConfig":
        """Smaller Apache-2.0 generator for constrained sessions (DD-024)."""
        return cls(
            embedding=EmbeddingConfig(batch_size=4),
            generation=GenerationConfig(
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                quantization="4bit",
                max_new_tokens=256,
            ),
        )

    @classmethod
    def offline(cls) -> "RAGConfig":
        """No weights, no GPU, no downloads.

        Uses the hashing embedder, the scripted extractive backend, the
        term-overlap reranker and the rule-based agents. All are real
        implementations rather than mocks: weaker than the model-backed
        components, but they retrieve, rerank, plan, assess, verify and answer
        for real, which is what makes every pipeline testable on CPU.
        """
        return cls(
            embedding=EmbeddingConfig(model_id=HASHING_EMBEDDER, device="cpu"),
            generation=GenerationConfig(
                model_id=SCRIPTED_BACKEND,
                backend="scripted",
                quantization="none",
                device="cpu",
            ),
            reranking=RerankingConfig(model_id=OVERLAP_RERANKER, device="cpu"),
            agents=AgentConfig(
                planner="rules", evidence_controller="rules", verifier="rules"
            ),
        )

    # -- derived ------------------------------------------------------------

    def estimate_vram_gb(self) -> float:
        """Resident VRAM for the configured stack, in GB.

        Excludes activations and KV cache, which is exactly why
        MODEL_SELECTION.md 9.1 reserves headroom on top of this number.
        """
        total = 0.0
        gen = MODEL_REGISTRY.get(self.generation.model_id)
        if gen is not None:
            total += gen.vram_gb * QUANT_VRAM_FACTOR[self.generation.quantization]
        emb = MODEL_REGISTRY.get(self.embedding.model_id)
        if emb is not None:
            total += emb.vram_gb
        if self.uses_reranker:
            reranker = MODEL_REGISTRY.get(self.reranking.model_id)
            if reranker is not None:
                total += reranker.vram_gb
        return round(total, 2)

    @property
    def is_hybrid(self) -> bool:
        """Does the system run hybrid retrieval? Baseline B and the agentic
        system do: the agentic system reuses Baseline B's retrieval, fusion and
        reranking unchanged (DD-050)."""
        return self.retrieval.strategy in ("hybrid", "agentic")

    @property
    def is_agentic(self) -> bool:
        return self.retrieval.strategy == "agentic"

    @property
    def uses_reranker(self) -> bool:
        """Only hybrid retrieval reranks; the dense baseline never loads one."""
        return self.is_hybrid and self.reranking.enabled

    @property
    def max_retrieval_iterations(self) -> int | None:
        """The retrieval-round cap actually in force, or None for a baseline.

        Refinement is triggered only by the evidence controller, so with either
        switched off the system runs exactly one round whatever the configured
        cap says -- and the record says 1, not the unused cap.
        """
        if not self.is_agentic:
            return None
        a = self.agents
        if not (a.refinement_enabled and a.evidence_controller_enabled):
            return 1
        return a.max_retrieval_iterations

    def fingerprint(self) -> str:
        """Stable hash of the whole configuration, recorded with every result."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> dict[str, object]:
        """Reproducibility record (MODEL_SELECTION.md section 15)."""
        gen = MODEL_REGISTRY.get(self.generation.model_id)
        emb = MODEL_REGISTRY.get(self.embedding.model_id)
        rer = MODEL_REGISTRY.get(self.reranking.model_id)
        hybrid = self.is_hybrid
        return {
            "config_fingerprint": self.fingerprint(),
            "generation_model": self.generation.model_id,
            "generation_license": gen.license if gen else "unregistered",
            "quantization": self.generation.quantization,
            "temperature": self.generation.temperature,
            "embedding_model": self.embedding.model_id,
            "embedding_license": emb.license if emb else "unregistered",
            "embedding_pooling": embedding_profile(self.embedding.model_id).pooling,
            "ocr": self.ingestion.ocr,
            "chunk_strategy": self.chunking.strategy,
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "top_k": self.retrieval.top_k,
            # Fields the dense baseline has no value for are null rather than
            # absent, so a reader comparing runs sees the component was absent.
            "retrieval_strategy": self.retrieval.strategy,
            "retrievers": list(self.retrieval.retrievers) if hybrid else ["dense"],
            "fusion": self.retrieval.fusion if hybrid else None,
            "rrf_k": self.retrieval.rrf_k if hybrid else None,
            "candidate_k": self.retrieval.candidate_k if hybrid else None,
            "reranker_enabled": self.uses_reranker,
            "reranker_model": self.reranking.model_id if self.uses_reranker else None,
            "reranker_license": (
                (rer.license if rer else "unregistered") if self.uses_reranker else None
            ),
            "rerank_candidates": self.reranking.candidates if hybrid else None,
            **self._describe_agents(),
            "estimated_vram_gb": self.estimate_vram_gb(),
        }

    def _describe_agents(self) -> dict[str, object]:
        """The agent switches, or null for a system that has no agents -- the
        reranker's precedent: absent, not forgotten."""
        a = self.agents
        agentic = self.is_agentic

        def backend(enabled: bool, kind: str, rule: str) -> str | None:
            if not (agentic and enabled):
                return None
            return self.generation.model_id if kind == "llm" else rule

        return {
            "planner_enabled": a.planner_enabled if agentic else None,
            "planner": backend(a.planner_enabled, a.planner, RULE_PLANNER),
            "max_sub_queries": a.max_sub_queries if agentic and a.planner_enabled else None,
            "evidence_controller_enabled": a.evidence_controller_enabled if agentic else None,
            "evidence_controller": backend(
                a.evidence_controller_enabled, a.evidence_controller, RULE_EVIDENCE_CONTROLLER
            ),
            "sufficiency_threshold": (
                a.sufficiency_threshold if agentic and a.evidence_controller_enabled else None
            ),
            "refinement_enabled": a.refinement_enabled if agentic else None,
            "max_retrieval_iterations": self.max_retrieval_iterations,
            "verification_enabled": a.verification_enabled if agentic else None,
            "verifier": backend(a.verification_enabled, a.verifier, RULE_VERIFIER),
            "verifier_support_threshold": (
                a.verifier_support_threshold if agentic and a.verification_enabled else None
            ),
            "max_regenerations": (
                a.max_regenerations if agentic and a.verification_enabled else None
            ),
        }

    # -- validation ---------------------------------------------------------

    def validate(self) -> "RAGConfig":
        """Raise ``ConfigurationError`` on an inconsistent or disallowed config."""
        c = self.chunking
        if c.chunk_size <= 0:
            raise ConfigurationError("chunking.chunk_size must be positive.")
        if c.chunk_overlap < 0:
            raise ConfigurationError("chunking.chunk_overlap must not be negative.")
        if c.chunk_overlap >= c.chunk_size:
            raise ConfigurationError(
                f"chunking.chunk_overlap ({c.chunk_overlap}) must be smaller than "
                f"chunk_size ({c.chunk_size}); otherwise chunking cannot advance."
            )
        if c.min_chunk_chars < 0:
            raise ConfigurationError("chunking.min_chunk_chars must not be negative.")

        r = self.retrieval
        if r.top_k <= 0:
            raise ConfigurationError("retrieval.top_k must be positive.")
        if not r.retrievers or any(name not in RETRIEVERS for name in r.retrievers):
            raise ConfigurationError(
                f"retrieval.retrievers must be a non-empty selection of "
                f"{list(RETRIEVERS)}, got {list(r.retrievers)}."
            )
        if len(set(r.retrievers)) != len(r.retrievers):
            raise ConfigurationError("retrieval.retrievers lists a retriever twice.")
        if r.strategy == "hybrid":
            if r.candidate_k < r.top_k:
                raise ConfigurationError(
                    f"retrieval.candidate_k ({r.candidate_k}) must be at least "
                    f"top_k ({r.top_k}); fusion cannot return more than it was given."
                )
            if self.reranking.candidates < r.top_k:
                raise ConfigurationError(
                    f"reranking.candidates ({self.reranking.candidates}) must be at "
                    f"least top_k ({r.top_k})."
                )
        if r.rrf_k <= 0:
            raise ConfigurationError("retrieval.rrf_k must be positive.")
        if r.bm25_k1 < 0 or not 0.0 <= r.bm25_b <= 1.0:
            raise ConfigurationError(
                "retrieval.bm25_k1 must be >= 0 and retrieval.bm25_b in [0, 1]."
            )
        if self.reranking.batch_size <= 0:
            raise ConfigurationError("reranking.batch_size must be positive.")

        if self.is_agentic:
            self._validate_agents()

        if self.generation.temperature < 0:
            raise ConfigurationError("generation.temperature must not be negative.")
        if self.generation.max_new_tokens <= 0:
            raise ConfigurationError("generation.max_new_tokens must be positive.")
        if self.generation.max_evidence_chars > self.generation.max_context_chars:
            raise ConfigurationError(
                "generation.max_evidence_chars must not exceed max_context_chars."
            )

        # DD-024: licensing is a hard constraint, not advice.
        gen = MODEL_REGISTRY.get(self.generation.model_id)
        if (
            gen is not None
            and not gen.commercial_use
            and not self.generation.allow_non_commercial_model
        ):
            raise ConfigurationError(
                f"{gen.model_id} is released under '{gen.license}', which permits "
                "research/non-commercial use only, so it must not be the default "
                "generator (DD-024). Use Qwen/Qwen3-4B-Instruct-2507, or "
                "Qwen/Qwen2.5-1.5B-Instruct for the low-memory preset. To "
                "benchmark it deliberately, set "
                "generation.allow_non_commercial_model=True and record the "
                "licence alongside the results."
            )

        # MODEL_SELECTION.md 9.1: fail before the T4 does.
        vram = self.estimate_vram_gb()
        if vram > MODEL_VRAM_BUDGET_GB:
            raise ConfigurationError(
                f"Configured models need an estimated {vram:.1f} GB of VRAM, over "
                f"the {MODEL_VRAM_BUDGET_GB:.0f} GB budget for a free-tier T4 "
                f"({T4_USABLE_VRAM_GB:.0f} GB usable, "
                f"{REQUIRED_HEADROOM_GB:.0f} GB reserved for activations and KV "
                "cache). Use generation.quantization='4bit', switch to "
                "RAGConfig.low_memory(), disable the reranker "
                "(reranking.enabled=False), or load the models sequentially."
            )
        return self

    def _validate_agents(self) -> None:
        a = self.agents
        if a.max_retrieval_iterations < 1:
            raise ConfigurationError(
                "agents.max_retrieval_iterations counts retrieval rounds, the "
                "first included, so it must be at least 1 (1 = no refinement)."
            )
        if a.max_sub_queries < 1:
            raise ConfigurationError(
                "agents.max_sub_queries must be at least 1: the original "
                "question is always the first query."
            )
        if a.max_regenerations < 0:
            raise ConfigurationError("agents.max_regenerations must not be negative.")
        for name in ("sufficiency_threshold", "verifier_support_threshold"):
            value = getattr(a, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"agents.{name} must be in [0, 1], got {value}.")
        scripted = (
            self.generation.backend == "scripted"
            or self.generation.model_id == SCRIPTED_BACKEND
        )
        llm = [
            name
            for name, enabled, kind in (
                ("planner", a.planner_enabled, a.planner),
                ("evidence_controller", a.evidence_controller_enabled, a.evidence_controller),
                ("verifier", a.verification_enabled, a.verifier),
            )
            if enabled and kind == "llm"
        ]
        if scripted and llm:
            raise ConfigurationError(
                f"agents.{', agents.'.join(llm)} = 'llm' needs a generation "
                "model, but the generator is the scripted extractive backend, "
                "which can only answer from evidence -- it cannot plan, assess "
                "or verify. Set them to 'rules' (the rule-based stand-ins, "
                "DD-055), or use RAGConfig.offline()."
            )
