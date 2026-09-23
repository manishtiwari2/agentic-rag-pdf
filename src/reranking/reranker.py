"""Reranking (ARCHITECTURE.md section 10, DD-006).

    input   query + candidate chunks
    output  the same chunks, re-ranked

A bi-encoder embeds the query and each passage separately and compares the
vectors; a cross-encoder reads the query and the passage *together* and scores
the pair. That is slower -- one forward pass per candidate rather than one per
query -- and more precise, which is why it runs on a short fused pool rather
than the whole index.

Two implementations behind one interface, the same arrangement as the embedder
and the generation backend:

* ``CrossEncoderReranker`` -- ``bge-reranker-v2-m3`` (Apache-2.0), or any
  sequence-classification cross-encoder, loaded lazily through
  ``transformers``.
* ``TermOverlapReranker`` -- no weights, no downloads, no GPU. It scores the
  pair jointly, as a cross-encoder does, but from surface overlap alone:
  the share of the query's distinct content words the passage contains, plus
  half the share of the query's adjacent word pairs it contains. A real if
  weak reranker (DD-048), deliberately a different signal from BM25 -- no IDF,
  no term-frequency saturation, no length normalization -- so it genuinely
  reorders a fused list rather than restating one of its inputs.

The reranker is optional (ARCHITECTURE.md section 10): the hybrid pipeline
builds one only when ``reranking.enabled`` is true, which is what makes the
"Reranker OFF" ablation a configuration change and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol, Sequence, runtime_checkable

from ..config import RerankingConfig
from ..errors import RerankingError
from ..retrieval.retriever import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs. Higher is more relevant."""

    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def rerank(
    reranker: Reranker, query: str, candidates: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Re-order ``candidates`` by the reranker's scores.

    The sort is stable on the incoming rank, so two candidates the reranker
    cannot separate keep the order fusion gave them: the reranker can only move
    a candidate for a reason. ``score`` becomes the reranker's score; ``source``
    keeps naming the retrievers that found the chunk, since reranking changes
    order, not provenance.
    """
    items = list(candidates)
    if not items:
        return []
    scores = reranker.score(query, [item.chunk.text for item in items])
    if len(scores) != len(items):
        raise RerankingError(
            f"Reranker {reranker.model_id!r} returned {len(scores)} scores for "
            f"{len(items)} candidates."
        )
    order = sorted(range(len(items)), key=lambda i: (-float(scores[i]), i))
    return [
        replace(items[i], score=float(scores[i]), rank=rank)
        for rank, i in enumerate(order, start=1)
    ]


# ---------------------------------------------------------------------------
# Dependency-free backend
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:\.\d+)?")
#: Words that say nothing about which passage answers a question. A separate
#: list from the scorer's and the scripted backend's, for the reason
#: ``src/evaluation/metrics.py`` gives: sharing a tokenizer with the component
#: being measured flatters it.
_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been by can did do does for from had has have how i if
    in into is it its of on or that the their them then there these they this those
    to was were what when where which who whom why will with would you your""".split()
)
#: A matching adjacent word pair counts for half a matching word.
_BIGRAM_WEIGHT = 0.5


def _content_tokens(text: str) -> list[str]:
    return [
        t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS
    ]


class TermOverlapReranker:
    """Joint query/passage coverage: what a cross-encoder reads, minus the model."""

    def __init__(self, config: RerankingConfig | None = None) -> None:
        self.config = config or RerankingConfig()
        self.model_id = self.config.model_id

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = _content_tokens(query)
        terms = set(query_tokens)
        bigrams = set(zip(query_tokens, query_tokens[1:]))
        results: list[float] = []
        for passage in passages:
            tokens = _content_tokens(passage)
            present = set(tokens)
            coverage = len(terms & present) / len(terms) if terms else 0.0
            if bigrams:
                pairs = set(zip(tokens, tokens[1:]))
                coverage += _BIGRAM_WEIGHT * len(bigrams & pairs) / len(bigrams)
            results.append(round(coverage, 9))
        return results


# ---------------------------------------------------------------------------
# Model-backed backend
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """A sequence-classification cross-encoder, scored on raw logits."""

    def __init__(self, config: RerankingConfig | None = None) -> None:
        self.config = config or RerankingConfig()
        self.model_id = self.config.model_id
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RerankingError(
                f"Reranker {self.model_id!r} needs `transformers` and `torch`, "
                "which are not installed. Install them, disable the reranker "
                "with reranking.enabled=False, or use RAGConfig.offline()."
            ) from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        if device == "cuda" and self.config.dtype in ("auto", "float16"):
            dtype = torch.float16

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, torch_dtype=dtype
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RerankingError(
                f"Could not load reranker {self.model_id!r}: {exc}. Check the "
                "model id and that the weights can be downloaded, disable the "
                "reranker with reranking.enabled=False, or use "
                "RAGConfig.offline() to run without weights."
            ) from exc

        self._model.to(device).eval()
        self._device = device

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        scores: list[float] = []
        batch = max(1, self.config.batch_size)
        with torch.no_grad():
            for start in range(0, len(passages), batch):
                chunk = list(passages[start : start + batch])
                encoded = self._tokenizer(  # type: ignore[misc]
                    [query] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**encoded).logits  # type: ignore[misc]
                scores.extend(logits.view(-1).float().cpu().tolist())
        return scores


def build_reranker(config: RerankingConfig) -> Reranker:
    """Construct the reranker named by the configuration (DD-017)."""
    from ..config import OVERLAP_RERANKER  # noqa: PLC0415

    if config.model_id == OVERLAP_RERANKER:
        return TermOverlapReranker(config)
    return CrossEncoderReranker(config)
