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
        note="Phase 4. Not used by the dense baseline.",
    ),
}

#: Dependency-free backends used when no weights are available.
HASHING_EMBEDDER = "local/hashing-embedder"
SCRIPTED_BACKEND = "local/scripted-extractive"


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


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    #: "auto" uses FAISS when importable and falls back to the numpy index.
    index: Literal["auto", "faiss", "numpy"] = "auto"
    #: Abstain without calling the generator when the best score falls below
    #: this. ``None`` disables it: EXPERIMENT_PLAN.md forbids introducing an
    #: untuned threshold, so by default the pipeline only short-circuits when
    #: retrieval returns nothing at all.
    min_score: float | None = None


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


@dataclass(frozen=True)
class RAGConfig:
    """The single configuration object for the whole system."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0

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

        Uses the hashing embedder and the scripted extractive backend. Both are
        real implementations rather than mocks: weaker than the model-backed
        components, but they retrieve and answer for real, which is what makes
        the pipeline testable on CPU.
        """
        return cls(
            embedding=EmbeddingConfig(model_id=HASHING_EMBEDDER, device="cpu"),
            generation=GenerationConfig(
                model_id=SCRIPTED_BACKEND,
                backend="scripted",
                quantization="none",
                device="cpu",
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
        return round(total, 2)

    def fingerprint(self) -> str:
        """Stable hash of the whole configuration, recorded with every result."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> dict[str, object]:
        """Reproducibility record (MODEL_SELECTION.md section 15)."""
        gen = MODEL_REGISTRY.get(self.generation.model_id)
        emb = MODEL_REGISTRY.get(self.embedding.model_id)
        return {
            "config_fingerprint": self.fingerprint(),
            "generation_model": self.generation.model_id,
            "generation_license": gen.license if gen else "unregistered",
            "quantization": self.generation.quantization,
            "temperature": self.generation.temperature,
            "embedding_model": self.embedding.model_id,
            "embedding_license": emb.license if emb else "unregistered",
            "embedding_pooling": embedding_profile(self.embedding.model_id).pooling,
            "chunk_strategy": self.chunking.strategy,
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "top_k": self.retrieval.top_k,
            "estimated_vram_gb": self.estimate_vram_gb(),
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

        if self.retrieval.top_k <= 0:
            raise ConfigurationError("retrieval.top_k must be positive.")

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
                "RAGConfig.low_memory(), or load the models sequentially."
            )
        return self
