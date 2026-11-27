"""Redis-backed ingestion worker. Run with arq src.api.worker.WorkerSettings."""

import asyncio
import logging
import os
import signal
import sys
from uuid import UUID

from arq import Retry, cron
from arq.connections import RedisSettings

from ..request_log import configure_logging
from ..storage.database import build_engine
from ..storage.library import DocumentLibrary
from .jobs import QUEUE, UploadJobs, deserialize_job, enqueue, serialize_job
from .settings import Settings

logger = logging.getLogger(__name__)


async def startup(ctx):
    configure_logging()
    settings = Settings.from_env()
    library = DocumentLibrary(build_engine(settings.database_url), settings.rag_config(), settings.code_version)
    ctx["uploads"] = UploadJobs(library, settings.storage_bytes, settings.job_timeout_s)


async def shutdown(ctx):
    if "uploads" in ctx:
        ctx["uploads"].engine.dispose()


async def recover_pending(ctx):
    pending = await asyncio.to_thread(lambda: list(ctx["uploads"].pending()))
    for owner, job_id in pending:
        await enqueue(ctx["redis"], owner, job_id)


async def _stop(process):
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            logger.debug("Ingestion process already exited")
    await process.wait()


async def ingest(ctx, owner_id: str, upload_id: str):
    owner, job_id = UUID(owner_id), UUID(upload_id)
    uploads = ctx["uploads"]
    if not await asyncio.to_thread(uploads.claim, owner, job_id):
        return
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "src.api.ingest_job", owner_id, upload_id,
            start_new_session=os.name == "posix",
        )
    except OSError:
        await asyncio.to_thread(uploads.fail, owner, job_id, "worker_failed")
        return
    try:
        code = await asyncio.wait_for(process.wait(), timeout=uploads.job_timeout_s)
    except asyncio.TimeoutError:
        await _stop(process)
        await asyncio.to_thread(uploads.fail, owner, job_id, "job_timeout")
        return
    except asyncio.CancelledError:
        await _stop(process)
        await asyncio.to_thread(uploads.retry, owner, job_id)
        raise
    if code == 75:
        await asyncio.to_thread(uploads.retry, owner, job_id)
        raise Retry(defer=10)
    if code != 0:
        logger.warning("Ingestion subprocess exited (%s)", code)
        await asyncio.to_thread(uploads.fail, owner, job_id, "worker_failed")


class WorkerSettings:
    functions = [ingest]
    cron_jobs = [cron(recover_pending, second={0, 30}, run_at_startup=True)]
    queue_name = QUEUE
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    on_startup = startup
    on_shutdown = shutdown
    job_serializer = serialize_job
    job_deserializer = deserialize_job
    max_jobs = 1
    job_timeout = int(os.environ.get("JOB_TIMEOUT_S", 240)) + 30
    job_completion_wait = 10
    keep_result = 0
