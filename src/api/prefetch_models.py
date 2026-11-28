"""Warm the shared model cache before starting the production API and worker."""

from ..reranking.reranker import build_reranker
from ..retrieval.embeddings import build_embedder
from .settings import Settings


def main() -> None:
    settings = Settings.from_env()
    if settings.offline:
        return
    config = settings.rag_config()
    build_embedder(config.embedding).embed_query("startup")
    if config.uses_reranker:
        build_reranker(config.reranking).score("startup", ["startup"])


if __name__ == "__main__":
    main()
