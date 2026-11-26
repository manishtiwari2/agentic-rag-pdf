"""A pgvector implementation of the existing dense vector-store protocol."""

from collections.abc import Sequence
from contextlib import nullcontext
from uuid import UUID

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection, Engine

from ..errors import RetrievalError
from ..retrieval.vector_store import _as_query, _validate
from .database import tenant_transaction
from .schema import HNSW_DIMENSION, chunk_embeddings, hnsw_embedding, hnsw_filter


class PgVectorStore:
    name = "pgvector"

    def __init__(
        self, bind: Engine | Connection, user_id: UUID, index_ids: Sequence[UUID], dimension: int,
    ) -> None:
        self._bind = bind
        self._user_id = user_id
        self._indexes = tuple(index_ids)
        self.dimension = dimension

    def _connection(self):
        if isinstance(self._bind, Connection):
            return nullcontext(self._bind)
        return tenant_transaction(self._bind, self._user_id)

    def _scope(self):
        return (
            chunk_embeddings.c.user_id == self._user_id,
            chunk_embeddings.c.index_id.in_(self._indexes),
        )

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = _validate(ids, vectors, self.dimension)
        if len(self._indexes) != 1:
            raise RetrievalError("Adding vectors requires a single document index.")
        if not np.isfinite(matrix).all():
            raise RetrievalError("Document embeddings must be finite.")
        if not ids:
            return
        rows = [
            {"user_id": self._user_id, "index_id": self._indexes[0], "chunk_id": chunk_id,
             "embedding": vector}
            for chunk_id, vector in zip(ids, matrix)
        ]
        with self._connection() as conn:
            conn.execute(chunk_embeddings.insert(), rows)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._indexes:
            return []
        vector = _as_query(query, self.dimension)
        if not np.isfinite(vector).all():
            raise RetrievalError("Query embeddings must be finite.")
        col = chunk_embeddings.c
        with self._connection() as conn:
            if self.dimension == HNSW_DIMENSION:
                distance = hnsw_embedding.max_inner_product(vector)
                conn.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
                approximate = select(col.chunk_id, distance).where(*self._scope(), hnsw_filter)
                rows = conn.execute(approximate.order_by(distance).limit(k)).all()
                if len(rows) == k:
                    return sorted(((key, -float(score)) for key, score in rows), key=lambda hit: (-hit[1], hit[0]))
            # Adding a constant keeps the fallback exact when HNSW returns too few scoped rows.
            distance = col.embedding.max_inner_product(vector)
            exact = select(col.chunk_id, distance).where(*self._scope()).order_by(distance + 0, col.chunk_id)
            return [(key, -float(score)) for key, score in conn.execute(exact.limit(k))]

    def __len__(self) -> int:
        with self._connection() as conn:
            return int(conn.execute(select(func.count()).select_from(chunk_embeddings).where(*self._scope())).scalar_one())
