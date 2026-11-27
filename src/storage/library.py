"""Store and query a user's PDF library through the existing RAG pipeline."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, select
from sqlalchemy.engine import Connection, Engine, RowMapping
from sqlalchemy.dialects.postgresql import insert

from ..chat.session import ChatSession, Turn
from ..chunking.base import Chunk, build_chunker
from ..config import RAGConfig
from ..deadlines import run_question
from ..errors import ConfigurationError, LibraryItemNotFound, PDFReadError, RetrievalError
from ..generation.backends import LLMBackend
from ..ingestion.parser import build_parser
from ..pipeline import RAGResult, build_pipeline
from ..request_log import request_scope
from ..reranking.reranker import Reranker
from ..retrieval.embeddings import Embedder, build_embedder
from ..versioning import DocumentVersion, resolve_version
from .database import tenant_transaction
from .schema import audit_log, chunks, document_indexes, document_versions, documents, pdf_files, users
from .vector_store import PgVectorStore


@dataclass(frozen=True)
class StoredVersion:
    id: UUID
    document_id: UUID
    source_name: str
    page_count: int
    version: DocumentVersion

    @classmethod
    def from_row(cls, row: RowMapping) -> "StoredVersion":
        return cls(
            row["id"], row["document_id"], row["source_name"], row["page_count"],
            DocumentVersion(**row["version_info"]),
        )


@dataclass(frozen=True)
class CitationSource:
    document_id: UUID
    filename: str
    pages: tuple[int, ...]
    text: str
    pdf: bytes
    ocr: bool


class DocumentLibrary:
    def __init__(
        self, engine: Engine, config: RAGConfig, code_version: str, *,
        embedder: Embedder | None = None, backend: LLMBackend | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not code_version.strip():
            raise ConfigurationError("A code version is required for persistent indexing.")
        self.engine = engine
        self.config = config.validate()
        self.code_version = code_version
        self.fingerprint = config.fingerprint()
        self.parser = build_parser(config.ingestion)
        self.chunker = build_chunker(config.chunking)
        # Corpus-fitted IDF would make saved vectors depend on the user's other uploads.
        self.embedder = embedder or build_embedder(config.embedding, fit_idf=False)
        self.backend = backend
        self.reranker = reranker

    @staticmethod
    def _lock_owner(conn: Connection, user_id: UUID) -> None:
        lock = select(users.c.id).where(users.c.id == user_id).with_for_update(key_share=True)
        if conn.execute(lock).scalar_one_or_none() is None:
            raise LibraryItemNotFound("The user does not exist.")

    def _version_query(self, user_id: UUID):
        v, i = document_versions.c, document_indexes.c
        return select(v.id, v.document_id, v.source_name, v.version_info, i.page_count).select_from(
            document_versions.join(document_indexes, and_(v.user_id == i.user_id, v.index_id == i.id))
        ).where(v.user_id == user_id)

    @request_scope("ingest PDF")
    def ingest(
        self, user_id: UUID, data: bytes, source_name: str, *,
        document_id: UUID | None = None, version: str | None = None,
        on_indexed: Callable[[Connection, StoredVersion], None] | None = None,
    ) -> StoredVersion:
        """Publish the PDF, chunks, vectors and version in one transaction."""
        source_name = source_name.replace("\\", "/").rsplit("/", 1)[-1]
        if not source_name or "\0" in source_name or (version is not None and "\0" in version):
            raise PDFReadError("The upload name and version must be valid text.")
        self.parser.validate_bytes(data, source_name)
        digest = sha256(data).hexdigest()
        v, i = document_versions.c, document_indexes.c
        with tenant_transaction(self.engine, user_id) as conn:
            # Serializing writes per owner prevents duplicate GPU work and races with deletion.
            self._lock_owner(conn, user_id)
            if document_id is None:
                document_id = conn.execute(
                    self._version_query(user_id).with_only_columns(v.document_id)
                    .where(i.file_sha256 == digest).order_by(v.created_at, v.id).limit(1)
                ).scalar_one_or_none()
                if document_id is None:
                    document_id = uuid4()
                    conn.execute(documents.insert().values(user_id=user_id, id=document_id, name=source_name))
            elif conn.execute(select(documents.c.id).where(
                documents.c.user_id == user_id, documents.c.id == document_id,
            )).scalar_one_or_none() is None:
                raise LibraryItemNotFound("The document does not exist.")

            index = conn.execute(select(document_indexes).where(
                i.user_id == user_id, i.file_sha256 == digest,
                i.config_fingerprint == self.fingerprint, i.code_version == self.code_version,
            )).mappings().one_or_none()
            if index is None:
                index_id, page_count, creation_date = self._index_pdf(conn, user_id, data, source_name, digest)
            else:
                index_id, page_count, creation_date = index["id"], index["page_count"], index["pdf_creation_date"]
                previous = conn.execute(self._version_query(user_id).where(
                    v.document_id == document_id, v.index_id == index_id,
                )).mappings().one_or_none()
                if previous is not None:
                    stored = StoredVersion.from_row(previous)
                    if on_indexed is not None:
                        on_indexed(conn, stored)
                    return stored

            version_id = uuid4()
            resolved = resolve_version(source_name, {"pdf_creation_date": creation_date}, version)
            conn.execute(document_versions.insert().values(
                user_id=user_id, id=version_id, document_id=document_id, index_id=index_id,
                source_name=source_name, version_info=resolved.to_record(),
            ))
            stored = StoredVersion(version_id, document_id, source_name, page_count, resolved)
            if on_indexed is not None:
                on_indexed(conn, stored)
            return stored

    def _index_pdf(
        self, conn: Connection, user_id: UUID, data: bytes, source_name: str, digest: str,
    ) -> tuple[UUID, int, str | None]:
        document = self.parser.parse_bytes(data, source_name)
        indexed = self.chunker.chunk(document)
        if not indexed:
            raise RetrievalError("The document produced no retrievable text.")
        vectors = self.embedder.embed_documents([chunk.text for chunk in indexed])
        if vectors.ndim != 2 or vectors.shape[0] != len(indexed) or not 1 <= vectors.shape[1] <= 16000:
            raise RetrievalError("The embedder must return one valid pgvector embedding per chunk.")
        index_id = uuid4()
        dimension = vectors.shape[1]
        raw_date = document.metadata.get("pdf_creation_date")
        creation = str(raw_date) if raw_date else None
        conn.execute(insert(pdf_files).values(
            user_id=user_id, sha256=digest, data=data, size_bytes=len(data),
        ).on_conflict_do_nothing(index_elements=[pdf_files.c.user_id, pdf_files.c.sha256]))
        conn.execute(document_indexes.insert().values(
            user_id=user_id, id=index_id, file_sha256=digest, config_fingerprint=self.fingerprint,
            code_version=self.code_version, dimension=dimension, page_count=document.page_count,
            pdf_creation_date=creation,
        ))
        keys = [f"{index_id}:{chunk.index:05d}" for chunk in indexed]
        conn.execute(chunks.insert(), [
            {"user_id": user_id, "index_id": index_id, "chunk_id": key, "position": chunk.index,
             "text": chunk.text, "pages": list(chunk.pages), "section": chunk.section,
             "details": {name: value for name, value in chunk.metadata.items() if name != "source_name"}}
            for key, chunk in zip(keys, indexed)
        ])
        PgVectorStore(conn, user_id, (index_id,), dimension).add(keys, vectors)
        return index_id, document.page_count, creation

    @request_scope("list document versions")
    def list_versions(self, user_id: UUID, document_id: UUID | None = None) -> list[StoredVersion]:
        v = document_versions.c
        query = self._version_query(user_id).order_by(v.created_at, v.id)
        if document_id is not None:
            query = query.where(v.document_id == document_id)
        with tenant_transaction(self.engine, user_id) as conn:
            return [StoredVersion.from_row(row) for row in conn.execute(query).mappings()]

    def list_documents(self, user_id: UUID) -> list[dict]:
        grouped: dict[UUID, list[StoredVersion]] = {}
        for version in self.list_versions(user_id):
            grouped.setdefault(version.document_id, []).append(version)
        return [{"id": key, "name": versions[-1].source_name, "versions": versions}
                for key, versions in grouped.items()]

    @request_scope("read cited page")
    def citation_source(self, user_id: UUID, version_id: UUID, chunk_id: str, page: int) -> CitationSource:
        v, i, f, c = document_versions.c, document_indexes.c, pdf_files.c, chunks.c
        source = document_versions.join(document_indexes, and_(v.user_id == i.user_id, v.index_id == i.id))
        source = source.join(pdf_files, and_(i.user_id == f.user_id, i.file_sha256 == f.sha256))
        source = source.join(chunks, and_(i.user_id == c.user_id, i.id == c.index_id))
        query = select(v.document_id, v.source_name, c.pages, c.text, f.data, c.details).select_from(source).where(
            v.user_id == user_id, v.id == version_id, c.chunk_id == chunk_id, c.pages.any(page),
        )
        with tenant_transaction(self.engine, user_id) as conn:
            row = conn.execute(query).one_or_none()
        if row is None:
            raise LibraryItemNotFound("The cited passage does not exist.")
        return CitationSource(row.document_id, row.source_name, tuple(row.pages), row.text, row.data, bool(row.details.get("ocr")))

    @request_scope("read PDF")
    def read_pdf(self, user_id: UUID, version_id: UUID) -> bytes:
        v, i, f = document_versions.c, document_indexes.c, pdf_files.c
        source = document_versions.join(document_indexes, and_(v.user_id == i.user_id, v.index_id == i.id))
        source = source.join(pdf_files, and_(i.user_id == f.user_id, i.file_sha256 == f.sha256))
        with tenant_transaction(self.engine, user_id) as conn:
            data = conn.execute(select(f.data).select_from(source).where(
                v.user_id == user_id, v.id == version_id,
            )).scalar_one_or_none()
        if data is None:
            raise LibraryItemNotFound("The document version does not exist.")
        return data

    def require_versions(self, user_id: UUID, version_ids: Sequence[UUID]) -> None:
        selected = set(version_ids)
        if not selected:
            raise LibraryItemNotFound("Select at least one document version.")
        with tenant_transaction(self.engine, user_id) as conn:
            actual = set(conn.scalars(select(document_versions.c.id).where(
                document_versions.c.user_id == user_id, document_versions.c.id.in_(selected),
            )))
        if actual != selected:
            raise LibraryItemNotFound("One or more document versions do not exist.")

    @request_scope("delete document")
    def delete_document(self, user_id: UUID, document_id: UUID) -> None:
        v, i, f = document_versions.c, document_indexes.c, pdf_files.c
        with tenant_transaction(self.engine, user_id) as conn:
            self._lock_owner(conn, user_id)
            index_ids = conn.execute(select(v.index_id).where(
                v.user_id == user_id, v.document_id == document_id,
            )).scalars().all()
            deleted = conn.execute(documents.delete().where(
                documents.c.user_id == user_id, documents.c.id == document_id,
            ).returning(documents.c.id)).scalar_one_or_none()
            if deleted is None:
                raise LibraryItemNotFound("The document does not exist.")
            unused_files = conn.execute(document_indexes.delete().where(
                i.user_id == user_id, i.id.in_(index_ids),
                ~exists(select(1).where(v.user_id == user_id, v.index_id == i.id)),
            ).returning(i.file_sha256)).scalars().all()
            conn.execute(pdf_files.delete().where(
                f.user_id == user_id, f.sha256.in_(unused_files),
                ~exists(select(1).where(i.user_id == user_id, i.file_sha256 == f.sha256)),
            ))
            conn.execute(audit_log.insert().values(user_id=user_id, action="document_deleted", resource_id=document_id))

    @request_scope("ask document library")
    def ask(
        self, user_id: UUID, question: str, *, version_ids: Sequence[UUID], turns: Sequence[Turn] = (),
    ) -> RAGResult:
        return run_question(
            self.config.generation.question_timeout_s, lambda: self._ask(user_id, question, version_ids, turns),
        )

    def _ask(self, user_id: UUID, question: str, version_ids: Sequence[UUID], turns: Sequence[Turn]) -> RAGResult:
        v, i, c = document_versions.c, document_indexes.c, chunks.c
        selected = set(version_ids)
        if not selected:
            raise LibraryItemNotFound("Select at least one document version.")
        source = document_versions.join(document_indexes, and_(v.user_id == i.user_id, v.index_id == i.id))
        source = source.join(chunks, and_(i.user_id == c.user_id, i.id == c.index_id))
        query = select(
            v.id, v.index_id, v.source_name, v.version_info, i.dimension, i.config_fingerprint,
            i.code_version, c.chunk_id, c.position, c.text, c.pages, c.section, c.details,
        ).select_from(source).where(
            v.user_id == user_id, v.id.in_(selected),
        ).order_by(v.created_at, v.id, c.position)
        with tenant_transaction(self.engine, user_id) as conn:
            rows = conn.execute(query).mappings().all()
        if {row["id"] for row in rows} != selected:
            raise LibraryItemNotFound("One or more document versions do not exist.")
        if any(
            row["config_fingerprint"] != self.fingerprint or row["code_version"] != self.code_version
            for row in rows
        ):
            raise RetrievalError("Re-index these documents with the current configuration before searching.")

        versions, names, index_versions, corpus = {}, {}, {}, []
        dimension = rows[0]["dimension"]
        for row in rows:
            if row["dimension"] != dimension:
                raise RetrievalError("Selected document versions have different embedding dimensions.")
            version_id = str(row["id"])
            # When two versions reference the same content, search that content once.
            if index_versions.setdefault(row["index_id"], version_id) != version_id:
                continue
            if version_id not in versions:
                versions[version_id] = DocumentVersion(**row["version_info"])
                names[version_id] = row["source_name"]
            corpus.append(Chunk(
                row["chunk_id"], version_id, row["text"], tuple(row["pages"]), row["section"], row["position"],
                {**row["details"], "source_name": names[version_id]},
            ))
        store = PgVectorStore(self.engine, user_id, tuple(index_versions), dimension)
        pipeline = build_pipeline(
            self.config, embedder=self.embedder, backend=self.backend, reranker=self.reranker,
        )
        pipeline.load_index(corpus, store, versions, names)
        if turns:
            session = ChatSession(pipeline)
            session.history = list(turns[-session.max_turns:])
            return session.ask(question)
        return pipeline.ask(question)
