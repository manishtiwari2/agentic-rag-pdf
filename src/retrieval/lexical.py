"""Lexical retrieval: Okapi BM25 (ARCHITECTURE.md section 8, DD-007).

Dense retrieval matches meaning; this matches strings. ARCHITECTURE.md section 8
names what that buys -- exact terminology, names, identifiers, numbers,
technical phrases -- and DD-007 is the reason it exists at all: an embedder
places ``part 884211`` and ``part 884212`` almost on top of each other, and a
question about one of them needs the retriever that can tell them apart.

Deterministic end to end, with no model anywhere (DD-018). The same ``Chunk``
goes in and the same ``RetrievedChunk`` comes out as for dense retrieval, so
fusion and everything after it cannot tell which retriever produced a
candidate except by its ``source``.

Three choices are recorded in DD-049:

* **Numbers stay whole.** ``9.4`` is one token and ``1,200`` becomes ``1200``;
  splitting on the point would turn a figure into two unrelated integers, and
  numbers are one of the things this retriever is here for.
* **No stemming and no stopword list.** BM25's IDF already drives a word that
  appears everywhere towards zero weight, and a stemmer would blur exactly the
  exact-match signal this retriever contributes.
* **A chunk sharing no term with the query is not returned.** Its BM25 score is
  zero; ranking it would hand fusion a rank decided by document order.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ..chunking.base import Chunk
from ..config import RetrievalConfig
from ..errors import IndexNotBuiltError, RetrievalError
from .retriever import RetrievedChunk

#: A number with an optional decimal part, or an alphanumeric word. Commas
#: inside numbers are removed before tokenizing, so ``1,200`` == ``1200``.
_TOKEN = re.compile(r"\d+(?:\.\d+)*|[a-z0-9]+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens, numbers kept whole."""
    return _TOKEN.findall(_THOUSANDS.sub("", (text or "").lower()))


class LexicalRetriever:
    """BM25 over the chunk texts, with Lucene's non-negative IDF."""

    strategy = "lexical"

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._average_length = 0.0

    # -- indexing -----------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index the chunks, replacing anything indexed before."""
        if not chunks:
            raise RetrievalError(
                "No chunks to index. The document parsed but produced no "
                "retrievable text; check the chunking configuration."
            )
        self._chunks = list(chunks)
        self._term_freqs = [Counter(tokenize(chunk.text)) for chunk in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        total = len(self._chunks)
        self._average_length = sum(self._lengths) / total if total else 0.0

        document_frequency: Counter[str] = Counter()
        for tf in self._term_freqs:
            document_frequency.update(tf.keys())
        # Lucene's variant: log(1 + (N - df + 0.5) / (df + 0.5)). Unlike the
        # original Robertson-Sparck Jones form it never goes negative, so a word
        # in more than half the chunks lowers nothing -- it just stops helping.
        self._idf = {
            term: math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    @property
    def is_indexed(self) -> bool:
        return bool(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    # -- search -------------------------------------------------------------

    def scores(self, query: str) -> list[float]:
        """The BM25 score of every indexed chunk, in document order."""
        k1 = self.config.bm25_k1
        b = self.config.bm25_b
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        average = self._average_length or 1.0
        result: list[float] = []
        for tf, length in zip(self._term_freqs, self._lengths):
            score = 0.0
            norm = k1 * (1.0 - b + b * length / average)
            for term in terms:
                frequency = tf.get(term, 0)
                if frequency:
                    score += self._idf[term] * frequency * (k1 + 1.0) / (frequency + norm)
            result.append(score)
        return result

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if not self._chunks:
            raise IndexNotBuiltError(
                "Nothing has been indexed yet. Call pipeline.index(pdf) before "
                "asking a question."
            )
        if not query.strip():
            raise RetrievalError("The query is empty.")

        k = k or self.config.top_k
        scored = [
            (score, position)
            for position, score in enumerate(self.scores(query))
            if score > 0.0
        ]
        # Ties fall back to document order, so the ranking is a function of the
        # text alone and never of dictionary iteration order.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            RetrievedChunk(
                chunk=self._chunks[position],
                score=score,
                rank=rank,
                source=self.strategy,
            )
            for rank, (score, position) in enumerate(scored[:k], start=1)
        ]
