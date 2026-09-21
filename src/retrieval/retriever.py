"""Dense retrieval (ARCHITECTURE.md section 8).

Phase 2 is dense-only on purpose. Lexical retrieval, fusion and reranking are
Phase 4, and adding them here would leave nothing to compare them against:
DD-013 exists because the value of the extra machinery cannot be measured
without a baseline that lacks it.

Retrieval is deterministic code end to end (DD-018). No model decides the
ranking; a bi-encoder produces vectors and an index sorts them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..chunking.base import Chunk
from ..config import RetrievalConfig
from ..errors import IndexNotBuiltError, RetrievalError
from .embeddings import Embedder
from .vector_store import VectorStore, build_vector_store


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk with its retrieval result attached.

    The chunk is carried whole rather than as an id, so the page metadata DD-010
    protects travels with the passage instead of needing a second lookup that a
    later stage might forget to perform.
    """

    chunk: Chunk
    score: float
    rank: int
    source: str = "dense"

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def pages(self) -> tuple[int, ...]:
        return self.chunk.pages

    def to_record(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "source": self.source,
            **self.chunk.to_record(),
        }


class DenseRetriever:
    """Embeds chunks into a vector index and searches it."""

    strategy = "dense"

    def __init__(
        self,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._store = store
        self._chunks: dict[str, Chunk] = {}

    # -- indexing -----------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed and index the chunks, replacing anything indexed before."""
        if not chunks:
            raise RetrievalError(
                "No chunks to index. The document parsed but produced no "
                "retrievable text; check the chunking configuration."
            )
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        if vectors.shape[0] != len(chunks):
            raise RetrievalError(
                f"Embedder returned {vectors.shape[0]} vectors for "
                f"{len(chunks)} chunks."
            )
        self._store = build_vector_store(self.config, vectors.shape[1])
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._store.add([chunk.chunk_id for chunk in chunks], vectors)

    @property
    def is_indexed(self) -> bool:
        return self._store is not None and len(self._store) > 0

    @property
    def size(self) -> int:
        return len(self._store) if self._store is not None else 0

    @property
    def index_name(self) -> str:
        return self._store.name if self._store is not None else "none"

    # -- search -------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if self._store is None:
            raise IndexNotBuiltError(
                "Nothing has been indexed yet. Call pipeline.index(pdf) before "
                "asking a question."
            )
        if not query.strip():
            raise RetrievalError("The query is empty.")

        k = k or self.config.top_k
        vector = self.embedder.embed_query(query)
        hits = self._store.search(vector, k)
        return [
            RetrievedChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                rank=rank,
                source=self.strategy,
            )
            for rank, (chunk_id, score) in enumerate(hits, start=1)
        ]
