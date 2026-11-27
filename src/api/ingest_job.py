"""Isolated ingestion subprocess, invoked only by the arq worker."""

import os
import sys
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


def main() -> int:
    limit_memory()
    from ..errors import IngestionError, LibraryItemNotFound, StorageError
    from ..request_log import configure_logging, request_scope
    from ..storage.database import build_engine
    from ..storage.library import DocumentLibrary
    from .jobs import UploadJobs
    from .settings import Settings

    configure_logging()
    settings = Settings.from_env()
    engine = build_engine(settings.database_url)
    service = UploadJobs(
        DocumentLibrary(engine, settings.rag_config(), settings.code_version),
        settings.storage_bytes, settings.job_timeout_s,
    )
    owner, job_id = (UUID(value) for value in sys.argv[1:])
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
    except StorageError:
        return 75
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
