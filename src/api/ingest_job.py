"""Isolated ingestion process that reads one job at a time from the arq worker."""

import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from uuid import UUID


def limit_memory() -> None:
    if os.name == "posix":
        import resource

        limit = int(os.environ.get("JOB_MEMORY_BYTES", 8 * 1024**3))
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    elif os.environ.get("RAG_OFFLINE") != "1":
        raise RuntimeError("Production ingestion requires POSIX memory limits.")


def _process_job(service, owner, job_id) -> int:
    from ..errors import IngestionError, LibraryItemNotFound, StorageError
    from ..request_log import request_scope

    try:
        with request_scope("ingestion job"):
            try:
                service.process(owner, job_id)
            except IngestionError:
                service.fail(owner, job_id, "invalid_pdf")
            except LibraryItemNotFound:
                service.fail(owner, job_id, "document_missing")
            except MemoryError:
                service.fail(owner, job_id, "job_memory_limit")
                return 76
    except StorageError:
        return 75
    return 0


def _serve(replies) -> None:
    from ..request_log import configure_logging
    from ..retrieval.embeddings import build_embedder
    from ..storage.database import build_engine
    from ..storage.library import DocumentLibrary
    from .jobs import UploadJobs
    from .settings import Settings

    configure_logging()
    settings = Settings.from_env()
    engine = build_engine(settings.database_url)
    try:
        config = settings.rag_config()
        size = settings.worker_embedding_batch_size
        embedder = build_embedder(replace(config.embedding, batch_size=size), fit_idf=False)
        library = DocumentLibrary(
            engine, config, settings.code_version, embedder=embedder, embedding_batch_size=size,
        )
        service = UploadJobs(library, settings.storage_bytes, settings.job_timeout_s)
        for line in sys.stdin:
            owner, job_id = (UUID(value) for value in line.split())
            code = _process_job(service, owner, job_id)
            print(f"{job_id} {code}", file=replies, flush=True)
            if code == 76:
                return
    finally:
        engine.dispose()


def main() -> int:
    limit_memory()
    replies = sys.stdout
    with redirect_stdout(sys.stderr):
        _serve(replies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
