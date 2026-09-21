"""Embedding and vector-index behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src.chunking.base import Chunk
from src.config import EmbeddingConfig, RetrievalConfig
from src.errors import IndexNotBuiltError, RetrievalError
from src.retrieval.embeddings import HashingEmbedder, build_embedder
from src.retrieval.retriever import DenseRetriever
from src.retrieval.vector_store import (
    FaissVectorStore,
    NumpyVectorStore,
    build_vector_store,
)

CORPUS = [
    "The reranker scores query and passage pairs with a cross-encoder.",
    "Peak GPU memory was measured at 9.4 gigabytes during generation.",
    "Chunks are embedded with a bi-encoder and stored in a vector index.",
    "The benchmark contains seventy-five manually verified questions.",
]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc:test:{i:05d}",
            document_id="doc",
            text=text,
            pages=(i + 1,),
            index=i,
        )
        for i, text in enumerate(CORPUS)
    ]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(EmbeddingConfig(hashing_dim=256))


class TestHashingEmbedder:
    def test_vectors_are_l2_normalized(self, embedder):
        vectors = embedder.embed_documents(CORPUS)
        assert vectors.shape == (len(CORPUS), 256)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_embedding_is_deterministic_across_instances(self):
        """Python's salted hash would make the index differ between runs."""
        a = HashingEmbedder(EmbeddingConfig(hashing_dim=256)).embed_documents(CORPUS)
        b = HashingEmbedder(EmbeddingConfig(hashing_dim=256)).embed_documents(CORPUS)
        np.testing.assert_array_equal(a, b)

    def test_related_text_scores_above_unrelated_text(self, embedder):
        vectors = embedder.embed_documents(CORPUS)
        query = embedder.embed_query("How much GPU memory was used?").reshape(-1)
        scores = vectors @ query
        assert int(np.argmax(scores)) == 1

    def test_morphological_variants_still_match(self, embedder):
        """Character n-grams give some signal where whole words share none.

        Asserted as a comparison rather than against an absolute threshold: the
        magnitude is a property of the hashing scheme, but the ordering is the
        property the n-grams exist to provide.
        """
        embedder.embed_documents(CORPUS)
        a = embedder.embed_query("embedded chunks").reshape(-1)
        variant = embedder.embed_query("chunk embedding").reshape(-1)
        unrelated = embedder.embed_query("licence terms and pricing").reshape(-1)
        assert float(a @ variant) > float(a @ unrelated)
        assert float(a @ variant) > 0.1  # and not merely noise

    def test_empty_corpus_returns_an_empty_matrix(self, embedder):
        assert embedder.embed_documents([]).shape == (0, 256)


class TestVectorStores:
    @pytest.mark.parametrize("store_class", [NumpyVectorStore, FaissVectorStore])
    def test_search_returns_ids_in_score_order(self, store_class, embedder):
        vectors = embedder.embed_documents(CORPUS)
        store = store_class(vectors.shape[1])
        store.add([f"c{i}" for i in range(len(CORPUS))], vectors)
        hits = store.search(embedder.embed_query("cross-encoder reranker"), 3)
        assert len(hits) == 3
        assert hits[0][0] == "c0"
        assert [score for _, score in hits] == sorted(
            (score for _, score in hits), reverse=True
        )

    def test_faiss_and_numpy_agree(self, embedder):
        """The two indexes must be interchangeable, ties included."""
        vectors = embedder.embed_documents(CORPUS)
        ids = [f"c{i}" for i in range(len(CORPUS))]
        numpy_store = NumpyVectorStore(vectors.shape[1])
        faiss_store = FaissVectorStore(vectors.shape[1])
        numpy_store.add(ids, vectors)
        faiss_store.add(ids, vectors)

        query = embedder.embed_query("vector index for chunks")
        numpy_hits = numpy_store.search(query, 4)
        faiss_hits = faiss_store.search(query, 4)
        assert [i for i, _ in numpy_hits] == [i for i, _ in faiss_hits]
        for (_, a), (_, b) in zip(numpy_hits, faiss_hits):
            assert a == pytest.approx(b, abs=1e-5)

    def test_identical_vectors_break_ties_by_insertion_order(self):
        store = NumpyVectorStore(3)
        vector = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        store.add(["first", "second", "third"], np.vstack([vector] * 3))
        hits = store.search(vector, 3)
        assert [chunk_id for chunk_id, _ in hits] == ["first", "second", "third"]

    def test_mismatched_dimension_is_rejected(self):
        store = NumpyVectorStore(4)
        with pytest.raises(RetrievalError, match="dimension"):
            store.add(["a"], np.zeros((1, 8), dtype=np.float32))

    def test_id_count_must_match_vector_count(self):
        store = NumpyVectorStore(4)
        with pytest.raises(RetrievalError, match="ids"):
            store.add(["a", "b"], np.zeros((1, 4), dtype=np.float32))

    def test_empty_index_returns_nothing(self):
        store = NumpyVectorStore(4)
        assert store.search(np.zeros(4, dtype=np.float32), 5) == []

    def test_auto_selects_an_index(self):
        store = build_vector_store(RetrievalConfig(index="auto"), 8)
        assert store.name in {"faiss", "numpy"}


class TestDenseRetriever:
    def test_retrieve_returns_ranked_chunks_with_metadata(self, embedder):
        retriever = DenseRetriever(embedder, RetrievalConfig(top_k=2))
        retriever.index(_chunks())
        hits = retriever.retrieve("how many benchmark questions are there?")
        assert [h.rank for h in hits] == [1, 2]
        assert hits[0].chunk.text.startswith("The benchmark contains")
        # The chunk travels whole, so its pages are available without a lookup.
        assert hits[0].pages == (4,)

    def test_top_k_is_honoured(self, embedder):
        retriever = DenseRetriever(embedder, RetrievalConfig(top_k=5))
        retriever.index(_chunks())
        assert len(retriever.retrieve("memory", k=2)) == 2

    def test_k_larger_than_the_corpus_is_clamped(self, embedder):
        retriever = DenseRetriever(embedder, RetrievalConfig())
        retriever.index(_chunks())
        assert len(retriever.retrieve("memory", k=50)) == len(CORPUS)

    def test_searching_before_indexing_raises(self, embedder):
        with pytest.raises(IndexNotBuiltError, match="index"):
            DenseRetriever(embedder).retrieve("anything")

    def test_indexing_nothing_raises(self, embedder):
        with pytest.raises(RetrievalError, match="No chunks"):
            DenseRetriever(embedder).index([])

    def test_empty_query_raises(self, embedder):
        retriever = DenseRetriever(embedder)
        retriever.index(_chunks())
        with pytest.raises(RetrievalError, match="empty"):
            retriever.retrieve("   ")

    def test_reindexing_replaces_the_previous_corpus(self, embedder):
        retriever = DenseRetriever(embedder)
        retriever.index(_chunks())
        retriever.index(_chunks()[:2])
        assert retriever.size == 2


class TestEmbedderConstruction:
    def test_offline_model_id_selects_the_hashing_embedder(self):
        from src.config import HASHING_EMBEDDER

        embedder = build_embedder(EmbeddingConfig(model_id=HASHING_EMBEDDER))
        assert isinstance(embedder, HashingEmbedder)
