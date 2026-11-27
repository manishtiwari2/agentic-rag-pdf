"""SQLAlchemy tables for the Postgres document library."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, CheckConstraint, Column, DateTime, ForeignKey, ForeignKeyConstraint,
    Identity, Index, Integer, LargeBinary, MetaData, String, Table, Text,
    UniqueConstraint, Uuid, cast, func, literal_column, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

metadata = MetaData(naming_convention={
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
})


def timestamp(name: str = "created_at") -> Column:
    return Column(name, DateTime(timezone=True), nullable=False, server_default=func.now())


def owner() -> Column:
    return Column("user_id", Uuid, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)


def scoped_fk(column: str, table: str, *, target: str = "id", cascade: bool = False) -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        ["user_id", column], [f"{table}.user_id", f"{table}.{target}"],
        ondelete="CASCADE" if cascade else None,
    )


users = Table(
    "users", metadata,
    Column("id", Uuid, primary_key=True),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", String(5), nullable=False, server_default="user"),
    timestamp(),
    CheckConstraint("role IN ('user', 'admin')", name="role"),
    CheckConstraint("email = lower(email)", name="email_lower"),
)
sessions = Table(
    "sessions", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("token_hash", String(64), nullable=False),
    Column("csrf_hash", String(64), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    timestamp(),
)
documents = Table(
    "documents", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("name", Text, nullable=False),
    timestamp(),
)
pdf_files = Table(
    "pdf_files", metadata, owner(),
    Column("sha256", String(64), primary_key=True),
    Column("data", LargeBinary, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    CheckConstraint("size_bytes = octet_length(data) AND size_bytes > 0", name="size"),
)
document_indexes = Table(
    "document_indexes", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("file_sha256", String(64), nullable=False),
    Column("config_fingerprint", String(16), nullable=False),
    Column("code_version", Text, nullable=False),
    Column("dimension", Integer, nullable=False),
    Column("page_count", Integer, nullable=False),
    Column("pdf_creation_date", Text),
    timestamp(),
    scoped_fk("file_sha256", "pdf_files", target="sha256"),
    UniqueConstraint("user_id", "file_sha256", "config_fingerprint", "code_version", name="uq_document_index_key"),
    CheckConstraint("dimension BETWEEN 1 AND 16000", name="dimension"),
    CheckConstraint("page_count > 0", name="pages"),
)
document_versions = Table(
    "document_versions", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("document_id", Uuid, nullable=False),
    Column("index_id", Uuid, nullable=False),
    Column("source_name", Text, nullable=False),
    Column("version_info", JSONB, nullable=False),
    timestamp(),
    scoped_fk("document_id", "documents", cascade=True),
    scoped_fk("index_id", "document_indexes"),
    UniqueConstraint("user_id", "document_id", "index_id", name="uq_document_version_index"),
    Index("ix_document_versions_owner_index", "user_id", "index_id"),
)
chunks = Table(
    "chunks", metadata, owner(),
    Column("index_id", Uuid, primary_key=True),
    Column("chunk_id", Text, primary_key=True),
    Column("position", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("pages", ARRAY(Integer), nullable=False),
    Column("section", Text),
    Column("details", JSONB, nullable=False),
    scoped_fk("index_id", "document_indexes", cascade=True),
    UniqueConstraint("user_id", "index_id", "position", name="uq_chunk_position"),
    CheckConstraint("position >= 0 AND cardinality(pages) > 0 AND pages[1] > 0", name="position_pages"),
)
chunk_embeddings = Table(
    "chunk_embeddings", metadata,
    Column("user_id", Uuid, primary_key=True),
    Column("index_id", Uuid, primary_key=True),
    Column("chunk_id", Text, primary_key=True),
    Column("embedding", Vector(), nullable=False),
    ForeignKeyConstraint(
        ["user_id", "index_id", "chunk_id"], ["chunks.user_id", "chunks.index_id", "chunks.chunk_id"],
        ondelete="CASCADE",
    ),
)
HNSW_DIMENSION = 1024
hnsw_embedding = cast(chunk_embeddings.c.embedding, Vector(HNSW_DIMENSION)).label("embedding")
hnsw_filter = func.vector_dims(chunk_embeddings.c.embedding) == literal_column(str(HNSW_DIMENSION))
Index(
    "ix_chunk_embeddings_hnsw", hnsw_embedding, postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_ip_ops"}, postgresql_where=hnsw_filter,
)
chats = Table(
    "chats", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("title", Text, nullable=False),
    timestamp(),
)
messages = Table(
    "messages", metadata, owner(),
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("chat_id", Uuid, nullable=False),
    Column("role", String(9), nullable=False),
    Column("content", Text, nullable=False),
    Column("citations", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    timestamp(),
    scoped_fk("chat_id", "chats", cascade=True),
    CheckConstraint("role IN ('user', 'assistant')", name="role"),
    Index("ix_messages_owner_chat", "user_id", "chat_id", "id"),
)
jobs = Table(
    "jobs", metadata, owner(),
    Column("id", Uuid, primary_key=True),
    Column("document_id", Uuid),
    Column("result_version_id", Uuid),
    Column("filename", Text),
    Column("status", String(9), nullable=False, server_default="queued"),
    Column("error_code", Text),
    timestamp(),
    timestamp("updated_at"),
    scoped_fk("document_id", "documents", cascade=True),
    scoped_fk("result_version_id", "document_versions", cascade=True),
    CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="status"),
    Index("ix_jobs_owner_document", "user_id", "document_id"),
)
job_uploads = Table(
    "job_uploads", metadata,
    Column("user_id", Uuid, primary_key=True),
    Column("id", Uuid, primary_key=True),
    Column("file_sha256", String(64), nullable=False),
    Column("data", LargeBinary, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("filename", Text, nullable=False),
    Column("version_label", Text),
    scoped_fk("id", "jobs", cascade=True),
    CheckConstraint("size_bytes = octet_length(data) AND size_bytes > 0", name="size"),
)
audit_log = Table(
    "audit_log", metadata, owner(),
    Column("id", BigInteger, Identity(), primary_key=True),
    Column("action", Text, nullable=False),
    Column("resource_id", Uuid),
    timestamp(),
)
