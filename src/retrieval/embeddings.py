"""Embedding backends (ARCHITECTURE.md section 7).

Two implementations behind one interface:

* ``HuggingFaceEmbedder`` -- the real thing, driven by the profile table in
  ``config.py``. BGE-M3 needs CLS pooling and no query instruction; BGE v1.5
  needs the instruction on queries only. Both mistakes are silent: the vectors
  still look fine and recall is simply worse, which is why the convention is
  looked up rather than written at the call site.
* ``HashingEmbedder`` -- no weights, no downloads, no GPU. A real if weak
  retriever rather than a stub: hashed word and character n-grams with
  sublinear term weighting and corpus IDF, which genuinely ranks passages and
  makes the whole pipeline testable on CPU.

The interface hides the model, so swapping one for the other changes nothing
downstream.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import EmbeddingConfig, EmbeddingProfile, embedding_profile
from ..errors import EmbeddingError

Vectors = np.ndarray


@runtime_checkable
class Embedder(Protocol):
    """Converts text into vectors.

    Documents and queries are embedded through separate methods because several
    model families treat them differently, and a single ``embed()`` would make
    that asymmetry impossible to express.
    """

    model_id: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def _l2_normalize(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Dependency-free backend
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
#: Character n-gram size. 4 is short enough to survive inflection
#: ("retrieve"/"retrieval" share three of their n-grams) and long enough to
#: avoid matching on noise.
_NGRAM = 4
#: Character n-grams carry less signal than whole words, so they count for less.
_NGRAM_WEIGHT = 0.4


class HashingEmbedder:
    """Hashed bag-of-n-grams with IDF weighting.

    Not a neural embedder and not pretending to be one: it captures lexical
    overlap and some morphology, and nothing semantic. Its purpose is to let the
    pipeline, and every test of it, run without downloading weights.

    IDF is fitted on the corpus passed to :meth:`embed_documents`. Queries
    embedded afterwards reuse those statistics; a term never seen in the corpus
    gets the maximum weight, since a rare term is the informative kind.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        config = config or EmbeddingConfig()
        self.model_id = config.model_id
        self.dimension = int(config.hashing_dim)
        if self.dimension <= 0:
            raise EmbeddingError("embedding.hashing_dim must be positive.")
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    # -- public -------------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        documents = [list(_features(text)) for text in texts]
        self._fit_idf(documents)
        if not documents:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _l2_normalize(
            np.vstack([self._vectorize(features) for features in documents])
        )

    def embed_query(self, text: str) -> Vectors:
        return _l2_normalize(self._vectorize(list(_features(text))))

    # -- internals ----------------------------------------------------------

    def _fit_idf(self, documents: list[list[tuple[str, float]]]) -> None:
        total = len(documents)
        if total == 0:
            return
        frequency: Counter[str] = Counter()
        for features in documents:
            frequency.update({term for term, _ in features})
        self._idf = {
            term: math.log((total + 1) / (count + 1)) + 1.0
            for term, count in frequency.items()
        }
        self._default_idf = math.log(total + 1) + 1.0

    def _vectorize(self, features: list[tuple[str, float]]) -> Vectors:
        vector = np.zeros((1, self.dimension), dtype=np.float32)
        counts: Counter[str] = Counter()
        weights: dict[str, float] = {}
        for term, weight in features:
            counts[term] += 1
            weights[term] = weight
        for term, count in counts.items():
            # Sublinear term frequency: the tenth occurrence of a word says much
            # less than the first.
            tf = 1.0 + math.log(count)
            idf = self._idf.get(term, self._default_idf) if self._idf else 1.0
            bucket, sign = _hash_bucket(term, self.dimension)
            vector[0, bucket] += sign * tf * idf * weights[term]
        return vector


def _features(text: str) -> Iterable[tuple[str, float]]:
    """Yield ``(term, weight)`` features for a piece of text."""
    tokens = _TOKEN.findall(text.lower())
    for token in tokens:
        yield token, 1.0
        if len(token) > _NGRAM:
            padded = f"<{token}>"
            for i in range(len(padded) - _NGRAM + 1):
                yield padded[i : i + _NGRAM], _NGRAM_WEIGHT
    for first, second in zip(tokens, tokens[1:]):
        yield f"{first}_{second}", 0.8


def _hash_bucket(term: str, dimension: int) -> tuple[int, float]:
    """Stable bucket and sign for a term.

    Python's ``hash`` is salted per process, so it would produce different
    vectors on every run. A fixed FNV-1a keeps indexing reproducible.
    """
    h = 2166136261
    for byte in term.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
    return h % dimension, sign


# ---------------------------------------------------------------------------
# Model-backed backend
# ---------------------------------------------------------------------------


class HuggingFaceEmbedder:
    """Transformer bi-encoder, called according to its family's profile."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.model_id = self.config.model_id
        self.profile: EmbeddingProfile = embedding_profile(self.model_id)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._dimension: int | None = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                f"Embedding model {self.model_id!r} needs `transformers` and "
                "`torch`, which are not installed. Install them, or switch to "
                "the dependency-free backend with RAGConfig.offline()."
            ) from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        if device == "cuda" and self.config.dtype in ("auto", "float16"):
            dtype = torch.float16

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                f"Could not load embedding model {self.model_id!r}: {exc}. Check "
                "the model id and that the weights can be downloaded, or use "
                "RAGConfig.offline() to run without them."
            ) from exc

        self._model.to(device).eval()
        self._device = device
        self._dimension = int(self._model.config.hidden_size)

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    # -- public -------------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        prefixed = [self.profile.document_prefix + text for text in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> Vectors:
        return self._encode([self.profile.query_prefix + text])

    # -- internals ----------------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> Vectors:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        outputs: list[np.ndarray] = []
        batch = max(1, self.config.batch_size)
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                encoded = self._tokenizer(  # type: ignore[misc]
                    list(texts[start : start + batch]),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**encoded).last_hidden_state  # type: ignore[misc]
                pooled = self._pool(hidden, encoded["attention_mask"])
                outputs.append(pooled.float().cpu().numpy())

        matrix = np.vstack(outputs).astype(np.float32)
        return _l2_normalize(matrix) if self.profile.normalize else matrix

    def _pool(self, hidden, attention_mask):
        if self.profile.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Construct the embedder named by the configuration (DD-017)."""
    from ..config import HASHING_EMBEDDER  # noqa: PLC0415

    if config.model_id == HASHING_EMBEDDER:
        return HashingEmbedder(config)
    return HuggingFaceEmbedder(config)
