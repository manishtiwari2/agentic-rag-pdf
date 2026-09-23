"""Hybrid retrieval: dense + lexical, fused (EVALUATION_PROTOCOL.md section 8).

Composes the two retrievers behind the same ``index``/``retrieve`` surface a
single retriever has, so the pipeline treats it as one more retriever. Each
underlying retriever ranks to ``candidate_k``; fusion merges them; the fused
list is cut to the reranker's candidate pool (DD-046).

The pool depth is fixed by configuration, not by the ``k`` a caller asks for.
That is what makes the ranking independent of ``k``: the harness's depth-10
probe and ``ask``'s top 5 read the same ranking, so Recall@10 describes the
list the generator's five came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..chunking.base import Chunk
from ..config import RetrievalConfig
from ..errors import IndexNotBuiltError, RetrievalError
from .fusion import reciprocal_rank_fusion
from .retriever import RetrievedChunk


@dataclass(frozen=True)
class Candidates:
    """The fused pool, plus what each retriever ranked before fusion."""

    fused: tuple[RetrievedChunk, ...]
    by_retriever: dict[str, tuple[RetrievedChunk, ...]]


class HybridRetriever:
    """Several retrievers, each ranking to ``candidate_k``, fused by RRF."""

    strategy = "hybrid"

    def __init__(
        self,
        retrievers: dict[str, object],
        config: RetrievalConfig | None = None,
        pool_size: int | None = None,
    ) -> None:
        if not retrievers:
            raise RetrievalError("A hybrid retriever needs at least one retriever.")
        self.config = config or RetrievalConfig(strategy="hybrid")
        # Insertion order is retriever priority, which is the third fusion
        # tie-break (DD-045).
        self.retrievers = dict(retrievers)
        self.pool_size = pool_size or self.config.candidate_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        for retriever in self.retrievers.values():
            retriever.index(chunks)  # type: ignore[attr-defined]

    @property
    def is_indexed(self) -> bool:
        return all(r.is_indexed for r in self.retrievers.values())  # type: ignore[attr-defined]

    @property
    def index_name(self) -> str:
        names = []
        for name, retriever in self.retrievers.items():
            names.append(getattr(retriever, "index_name", None) or name)
        return "+".join(names)

    def candidates(self, query: str) -> Candidates:
        if not self.is_indexed:
            raise IndexNotBuiltError(
                "Nothing has been indexed yet. Call pipeline.index(pdf) before "
                "asking a question."
            )
        by_retriever = {
            name: tuple(retriever.retrieve(query, self.config.candidate_k))  # type: ignore[attr-defined]
            for name, retriever in self.retrievers.items()
        }
        if self.config.fusion != "rrf":  # pragma: no cover - guarded by Literal
            raise RetrievalError(f"Unknown fusion method {self.config.fusion!r}.")
        fused = reciprocal_rank_fusion(
            list(by_retriever.values()), k=self.config.rrf_k
        )
        return Candidates(fused=tuple(fused[: self.pool_size]), by_retriever=by_retriever)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return list(self.candidates(query).fused[: k or self.config.top_k])
