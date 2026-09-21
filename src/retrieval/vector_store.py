"""Vector index with a FAISS implementation and a numpy fallback.

Both store L2-normalized vectors and search by inner product, which for unit
vectors is cosine similarity, so the two are interchangeable and return the same
ranking. The numpy index is exact rather than approximate; on a single PDF the
corpus is a few hundred chunks and a brute-force scan is fast enough that an
approximate index would trade accuracy for nothing.

Ties are broken by insertion order in both implementations. Without that, two
chunks with identical scores could swap places between runs and an ablation
would attribute the difference to whatever was being tested.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import RetrievalConfig
from ..errors import RetrievalError


@runtime_checkable
class VectorStore(Protocol):
    name: str
    dimension: int

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class NumpyVectorStore:
    """Exact inner-product search over a dense matrix."""

    name = "numpy"

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._ids: list[str] = []
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = _validate(ids, vectors, self.dimension)
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, matrix])

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        vector = _as_query(query, self.dimension)
        scores = self._matrix @ vector
        k = min(k, len(self._ids))
        # Stable sort on the negated scores: equal scores keep insertion order.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._ids)


class FaissVectorStore:
    """FAISS ``IndexFlatIP``, with the same semantics as the numpy store."""

    name = "faiss"

    def __init__(self, dimension: int) -> None:
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RetrievalError(
                "FAISS is not installed. Install `faiss-cpu`, or set "
                "retrieval.index='numpy' to use the built-in exact index."
            ) from exc
        self._faiss = faiss
        self.dimension = int(dimension)
        self._index = faiss.IndexFlatIP(self.dimension)
        self._ids: list[str] = []

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = _validate(ids, vectors, self.dimension)
        self._index.add(matrix)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        vector = _as_query(query, self.dimension).reshape(1, -1)
        k = min(k, len(self._ids))
        scores, indices = self._index.search(vector, k)
        results = [
            (self._ids[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]
        # FAISS does not promise a tie order; impose the same one the numpy
        # store uses so the two are interchangeable.
        position = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        results.sort(key=lambda item: (-item[1], position[item[0]]))
        return results

    def __len__(self) -> int:
        return len(self._ids)


def _validate(ids: Sequence[str], vectors: np.ndarray, dimension: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise RetrievalError(
            f"Expected a 2-D array of vectors, got shape {matrix.shape}."
        )
    if matrix.shape[0] != len(ids):
        raise RetrievalError(
            f"Got {len(ids)} ids but {matrix.shape[0]} vectors; every vector must "
            "carry the id of the chunk it came from or citations cannot be "
            "resolved."
        )
    if matrix.shape[1] != dimension:
        raise RetrievalError(
            f"Vectors have dimension {matrix.shape[1]}, but the index was built "
            f"for {dimension}. This usually means the embedding model changed "
            "without the index being rebuilt."
        )
    return np.ascontiguousarray(matrix)


def _as_query(query: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dimension:
        raise RetrievalError(
            f"Query vector has dimension {vector.shape[0]}, index expects "
            f"{dimension}."
        )
    return vector


def build_vector_store(config: RetrievalConfig, dimension: int) -> VectorStore:
    """Construct the index named by the configuration (DD-017).

    ``auto`` prefers FAISS and falls back to numpy, so a missing optional
    dependency degrades performance rather than stopping the pipeline.
    """
    if config.index == "numpy":
        return NumpyVectorStore(dimension)
    if config.index == "faiss":
        return FaissVectorStore(dimension)
    try:
        return FaissVectorStore(dimension)
    except RetrievalError:
        return NumpyVectorStore(dimension)
