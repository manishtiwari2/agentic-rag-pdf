"""Check production model pins without fetching weights."""

import sys
from dataclasses import replace
from types import SimpleNamespace

from src.api import prefetch_models
from src.api.settings import EMBEDDING_REVISION, RERANKER_REVISION, Settings
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.embeddings import HuggingFaceEmbedder


def test_production_pins_reach_the_model_and_tokenizer(monkeypatch):
    calls = []

    def load(kind):
        def from_pretrained(model_id, **options):
            calls.append((kind, model_id, options["revision"]))
            return model

        return SimpleNamespace(from_pretrained=from_pretrained)

    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4), to=lambda _device: model, eval=lambda: model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32", cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=load("tokenizer"), AutoModel=load("embedder"),
        AutoModelForSequenceClassification=load("reranker"),
    ))
    config = Settings("db", "redis", "test").rag_config()
    assert HuggingFaceEmbedder(config.embedding).dimension == 4
    CrossEncoderReranker(config.reranking)._ensure_loaded()
    assert calls == [
        ("tokenizer", "BAAI/bge-m3", EMBEDDING_REVISION),
        ("embedder", "BAAI/bge-m3", EMBEDDING_REVISION),
        ("tokenizer", "BAAI/bge-reranker-v2-m3", RERANKER_REVISION),
        ("reranker", "BAAI/bge-reranker-v2-m3", RERANKER_REVISION),
    ]
    new_embedder = replace(config, embedding=replace(config.embedding, revision="next"))
    new_reranker = replace(config, reranking=replace(config.reranking, revision="next"))
    assert new_embedder.fingerprint(for_index=True) != config.fingerprint(for_index=True)
    assert new_reranker.fingerprint(for_index=True) == config.fingerprint(for_index=True)
    assert new_reranker.fingerprint() != config.fingerprint()


def test_offline_prefetch_does_not_load_models(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "db")
    monkeypatch.setenv("REDIS_URL", "redis")
    monkeypatch.setenv("RAG_CODE_VERSION", "test")
    monkeypatch.setenv("RAG_OFFLINE", "1")

    def forbidden(_config):
        raise AssertionError("Offline prefetch must not load models")

    monkeypatch.setattr(prefetch_models, "build_embedder", forbidden)
    monkeypatch.setattr(prefetch_models, "build_reranker", forbidden)
    prefetch_models.main()
