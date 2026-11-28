"""Redis-backed ingestion worker. Run with arq src.api.worker.WorkerSettings."""

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from uuid import UUID

from arq import Retry, cron
from arq.connections import RedisSettings

from ..request_log import configure_logging
from ..storage.database import build_engine
from ..storage.library import DocumentLibrary
from .jobs import QUEUE, UploadJobs, deserialize_job, enqueue, serialize_job
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionProcess:
    process: asyncio.subprocess.Process | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def startup(ctx):
    configure_logging()
    settings = Settings.from_env()
    library = DocumentLibrary(build_engine(settings.database_url), settings.rag_config(), settings.code_version)
    ctx["uploads"] = UploadJobs(library, settings.storage_bytes, settings.job_timeout_s)
    # arq copies the context dictionary for each job.
    ctx["ingester"] = IngestionProcess()


async def shutdown(ctx):
    try:
        ingester = ctx.pop("ingester", None)
        process = ingester.process if ingester is not None else None
        if process is not None:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                await _stop(process)
    finally:
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


async def _discard_ingester(ingester: IngestionProcess):
    process, ingester.process = ingester.process, None
    if process is not None:
        await _stop(process)


async def _run_job(ingester: IngestionProcess, owner, job_id):
    process = ingester.process
    if process is None or process.returncode is not None:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "src.api.ingest_job",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        ingester.process = process
    process.stdin.write(f"{owner} {job_id}\n".encode("ascii"))
    await process.stdin.drain()
    reply = (await process.stdout.readline()).split()
    if len(reply) != 2 or reply[0] != str(job_id).encode("ascii") or reply[1] not in (b"0", b"75", b"76"):
        raise OSError("The ingestion process did not return a valid job result.")
    return int(reply[1])


async def ingest(ctx, owner_id: str, upload_id: str):
    owner, job_id = UUID(owner_id), UUID(upload_id)
    ingester = ctx["ingester"]
    async with ingester.lock:
        uploads = ctx["uploads"]
        if not await asyncio.to_thread(uploads.claim, owner, job_id):
            return
        try:
            code = await asyncio.wait_for(_run_job(ingester, owner, job_id), timeout=uploads.job_timeout_s)
        except asyncio.TimeoutError:
            await _discard_ingester(ingester)
            await asyncio.to_thread(uploads.fail, owner, job_id, "job_timeout")
            return
        except asyncio.CancelledError:
            await _discard_ingester(ingester)
            await asyncio.to_thread(uploads.retry, owner, job_id)
            raise
        except (OSError, ValueError) as exc:
            logger.warning("Ingestion process failed (%s)", type(exc).__name__)
            await _discard_ingester(ingester)
            await asyncio.to_thread(uploads.fail, owner, job_id, "worker_failed")
            return
        if code == 75:
            await asyncio.to_thread(uploads.retry, owner, job_id)
            raise Retry(defer=10)
        if code == 76:
            await _discard_ingester(ingester)


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
