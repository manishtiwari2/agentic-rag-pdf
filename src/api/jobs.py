"""Durable upload reservations and owner-scoped job status."""

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select

from ..storage.database import tenant_transaction
from ..storage.library import DocumentLibrary, StoredVersion
from ..storage.schema import audit_log, documents, job_uploads, jobs, pdf_files, users
from .errors import ApiError

QUEUE = "rag:ingestion"


def serialize_job(value: dict) -> bytes:
    return json.dumps(value).encode("utf-8")


def deserialize_job(value: bytes) -> dict:
    return json.loads(value)


async def enqueue(redis, owner: UUID, job_id: UUID) -> None:
    await redis.enqueue_job("ingest", str(owner), str(job_id), _job_id=str(job_id), _queue_name=QUEUE)


class UploadJobs:
    def __init__(self, library: DocumentLibrary, storage_bytes: int, job_timeout_s: int) -> None:
        self.library = library
        self.engine = library.engine
        self.storage_bytes = storage_bytes
        self.job_timeout_s = job_timeout_s

    def _claimable(self):
        stale = datetime.now(timezone.utc) - timedelta(seconds=self.job_timeout_s + 60)
        return or_(jobs.c.status == "queued", and_(jobs.c.status == "running", jobs.c.updated_at < stale))

    def submit(
        self, owner: UUID, data: bytes, filename: str, document_id: UUID | None, version: str | None,
    ) -> UUID:
        self.library.parser.validate_bytes(data, filename)
        if not filename or len(filename) > 255 or "\0" in filename or "/" in filename or "\\" in filename:
            raise ApiError(422, "invalid_filename", "Use a filename without directories.")
        if version is not None and (len(version) > 200 or "\0" in version):
            raise ApiError(422, "invalid_version", "Use a version label of up to 200 characters.")
        digest = sha256(data).hexdigest()
        with tenant_transaction(self.engine, owner) as conn:
            self.library._lock_owner(conn, owner)
            if document_id is not None and conn.scalar(select(documents.c.id).where(
                documents.c.user_id == owner, documents.c.id == document_id,
            )) is None:
                raise ApiError(404, "not_found", "The document does not exist.")
            u, j = job_uploads.c, jobs.c
            existing = conn.scalar(select(u.id).select_from(
                job_uploads.join(jobs, and_(u.user_id == j.user_id, u.id == j.id)),
            ).where(
                u.user_id == owner, u.file_sha256 == digest, u.filename == filename,
                u.version_label == version, j.document_id == document_id,
            ).limit(1))
            if existing:
                return existing
            stored = conn.scalar(select(func.coalesce(func.sum(pdf_files.c.size_bytes), 0)).where(
                pdf_files.c.user_id == owner,
            ))
            pending, count = conn.execute(select(func.coalesce(func.sum(job_uploads.c.size_bytes), 0), func.count())
                                          .where(job_uploads.c.user_id == owner)).one()
            if stored + pending + len(data) > self.storage_bytes:
                raise ApiError(413, "storage_quota", "The storage quota would be exceeded.")
            if count >= 10:
                raise ApiError(429, "upload_queue_full", "Wait for an existing upload to finish.")
            job_id = uuid4()
            conn.execute(jobs.insert().values(user_id=owner, id=job_id, document_id=document_id))
            conn.execute(job_uploads.insert().values(
                user_id=owner, id=job_id, file_sha256=digest, filename=filename,
                version_label=version, data=data, size_bytes=len(data),
            ))
            conn.execute(audit_log.insert().values(user_id=owner, action="upload_queued", resource_id=job_id))
            return job_id

    def status(self, owner: UUID, job_id: UUID | None = None):
        c = jobs.c
        query = select(c.id, c.document_id, c.result_version_id, c.status, c.error_code, c.created_at).where(
            c.user_id == owner,
        )
        with tenant_transaction(self.engine, owner) as conn:
            if job_id is None:
                return [dict(row) for row in conn.execute(query.order_by(c.created_at.desc()).limit(100)).mappings()]
            row = conn.execute(query.where(c.id == job_id)).mappings().one_or_none()
            if row is None:
                raise ApiError(404, "not_found", "The job does not exist.")
            return dict(row)

    def claim(self, owner: UUID, job_id: UUID) -> bool:
        with tenant_transaction(self.engine, owner) as conn:
            return conn.scalar(jobs.update().where(
                jobs.c.user_id == owner, jobs.c.id == job_id, self._claimable(),
            ).values(status="running", updated_at=func.now()).returning(jobs.c.id)) is not None

    def pending(self):
        with self.engine.connect() as conn:
            owners = conn.scalars(select(users.c.id)).all()
        for owner in owners:
            with tenant_transaction(self.engine, owner) as conn:
                pending = conn.scalars(select(jobs.c.id).where(jobs.c.user_id == owner, self._claimable())).all()
            yield from ((owner, job_id) for job_id in pending)

    def process(self, owner: UUID, job_id: UUID) -> StoredVersion | None:
        j, u = jobs.c, job_uploads.c
        with tenant_transaction(self.engine, owner) as conn:
            upload = conn.execute(select(u.data, u.filename, u.version_label, j.document_id).select_from(
                job_uploads.join(jobs, and_(j.user_id == u.user_id, j.id == u.id)),
            ).where(u.user_id == owner, u.id == job_id, j.status == "running")).one_or_none()
        if upload is None:
            return None
        return self.library.ingest(
            owner, upload.data, upload.filename, document_id=upload.document_id, version=upload.version_label,
            on_indexed=lambda conn, indexed: self._finish(conn, owner, job_id, version=indexed),
        )

    def retry(self, owner: UUID, job_id: UUID) -> None:
        with tenant_transaction(self.engine, owner) as conn:
            conn.execute(jobs.update().where(
                jobs.c.user_id == owner, jobs.c.id == job_id, jobs.c.status == "running",
            ).values(status="queued", updated_at=func.now()))

    def fail(self, owner: UUID, job_id: UUID, error: str) -> None:
        with tenant_transaction(self.engine, owner) as conn:
            self._finish(conn, owner, job_id, error=error)

    def _finish(self, conn, owner: UUID, job_id: UUID, *, version: StoredVersion | None = None, error=None) -> None:
        values = {"status": "failed", "error_code": error, "updated_at": func.now()}
        if version is not None:
            values.update(status="succeeded", error_code=None, document_id=version.document_id, result_version_id=version.id)
        changed = conn.scalar(jobs.update().where(
            jobs.c.user_id == owner, jobs.c.id == job_id, jobs.c.status == "running",
        ).values(**values).returning(jobs.c.id))
        if changed is None:
            if version is not None:
                raise ApiError(409, "upload_cancelled", "The upload is no longer pending.")
            return
        conn.execute(job_uploads.delete().where(job_uploads.c.user_id == owner, job_uploads.c.id == job_id))
        conn.execute(audit_log.insert().values(user_id=owner, action=f"upload_{values['status']}", resource_id=job_id))
