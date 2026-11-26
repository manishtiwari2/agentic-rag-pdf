"""Postgres integration tests. Run docker info before setting RAG_TEST_DATABASE_URL."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

DATABASE_URL = os.environ.get("RAG_TEST_DATABASE_URL")
if not DATABASE_URL:
    pytest.skip("set RAG_TEST_DATABASE_URL to a disposable pgvector database", allow_module_level=True)

from alembic import command
from alembic.config import Config
from sqlalchemy import event, func, select, text

from src.config import RAGConfig
from src.errors import LibraryItemNotFound, RetrievalError
from src.retrieval.embeddings import HashingEmbedder
from src.storage.database import build_engine, tenant_transaction
from src.storage.library import DocumentLibrary
from src.storage.schema import chunk_embeddings, chunks, document_indexes, documents, pdf_files, users
from src.storage.vector_store import PgVectorStore

from . import pdf_fixtures

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What recall did the hybrid system achieve?"


@pytest.fixture(scope="module")
def engine():
    admin = build_engine(DATABASE_URL)
    suffix = uuid4().hex
    schema, role = f"test_rag_{suffix}", f"test_role_{suffix}"
    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
            conn.execute(text(f"CREATE ROLE {role} NOLOGIN NOBYPASSRLS"))
            conn.execute(text(f"SET LOCAL search_path TO {schema}, public"))
            config = Config(str(ROOT / "alembic.ini"))
            config.attributes["connection"] = conn
            command.upgrade(config, "head")
            command.downgrade(config, "base")
            command.upgrade(config, "head")
            command.check(config)
            conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {role}"))
            conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role}"))

        app = build_engine(DATABASE_URL)

        @event.listens_for(app, "connect")
        def _configure(dbapi_connection, _record):
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}, public")
                cursor.execute(f"SET ROLE {role}")
            dbapi_connection.commit()

        try:
            yield app
        finally:
            app.dispose()
    finally:
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
        admin.dispose()


@pytest.fixture
def owners(engine):
    ids = uuid4(), uuid4()
    with engine.begin() as conn:
        conn.execute(users.insert(), [
            {"id": user_id, "email": f"{user_id}@example.test", "password_hash": "unused"}
            for user_id in ids
        ])
    return ids


def count(engine, owner, table):
    with tenant_transaction(engine, owner) as conn:
        return conn.scalar(select(func.count()).select_from(table).where(table.c.user_id == owner))


def test_duplicate_upload_reuses_index_after_reconnect(engine, owners, monkeypatch):
    owner, _ = owners
    data = pdf_fixtures.structured_pdf()
    config = RAGConfig.offline()
    library = DocumentLibrary(engine, config, "test-v1")
    embed = library.embedder.embed_documents
    calls = []

    def tracked(texts):
        calls.append(texts)
        return embed(texts)

    monkeypatch.setattr(library.embedder, "embed_documents", tracked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(library.ingest, owner, data, "report.pdf", version="2024-08 final") for _ in range(2)]
        first, second = [future.result(timeout=30) for future in futures]
    assert first == second and len(calls) == 1
    assert first.version.date == "2024-08"
    assert count(engine, owner, document_indexes) == count(engine, owner, pdf_files) == 1

    engine.dispose()
    restored = DocumentLibrary(engine, config, "test-v1")

    def reject_embed(*args):
        pytest.fail("a duplicate upload or question must not embed documents")

    monkeypatch.setattr(restored.embedder, "embed_documents", reject_embed)
    assert restored.ingest(owner, data, "renamed.pdf") == first
    assert restored.list_versions(owner) == [first]
    assert restored.read_pdf(owner, first.id) == data
    result = restored.ask(owner, QUESTION, version_ids=[first.id])
    assert not result.abstained and result.citations
    assert {citation.document_id for citation in result.citations} == {str(first.id)}
    assert all(1 <= page <= first.page_count for page in result.cited_pages)


def test_owner_isolation_covers_storage_dense_and_lexical(engine, owners):
    alice, bob = owners
    config = RAGConfig.offline()
    config = replace(config, retrieval=replace(config.retrieval, strategy="hybrid"))
    library = DocumentLibrary(engine, config, "test-v1")
    pdf = pdf_fixtures.structured_pdf()
    private = library.ingest(alice, pdf, "alice.pdf")
    own = library.ingest(bob, pdf, "bob.pdf")
    assert private.id != own.id and private.document_id != own.document_id
    assert library.list_versions(bob, private.document_id) == []
    for action in (
        lambda: library.read_pdf(bob, private.id),
        lambda: library.delete_document(bob, private.document_id),
        lambda: library.ingest(bob, pdf, "other.pdf", document_id=private.document_id),
        lambda: library.ask(bob, QUESTION, version_ids=[own.id, private.id]),
    ):
        with pytest.raises(LibraryItemNotFound):
            action()
    assert library.read_pdf(alice, private.id) == pdf
    result = library.ask(bob, QUESTION, version_ids=[own.id])
    assert result.retrieved and {item.chunk.document_id for item in result.retrieved} == {str(own.id)}

    with tenant_transaction(engine, alice) as conn:
        private_index = conn.scalar(select(document_indexes.c.id))
    with tenant_transaction(engine, bob) as conn:
        # Check the database policy independently of the library's owner predicates.
        assert conn.scalar(select(func.count()).select_from(chunks).where(chunks.c.user_id == alice)) == 0
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(chunks)) == 0
    private_store = PgVectorStore(engine, bob, [private_index], library.embedder.dimension)
    assert private_store.search(library.embedder.embed_query(QUESTION), 5) == []
    assert len(private_store) == 0


def test_versions_change_with_config_and_code_and_share_pdf_bytes(engine, owners):
    owner, _ = owners
    config = RAGConfig.offline()
    pdf = pdf_fixtures.structured_pdf()
    first_library = DocumentLibrary(engine, config, "test-v1")
    first = first_library.ingest(owner, pdf, "report.pdf")
    second_config = replace(config, embedding=replace(config.embedding, hashing_dim=64))
    second_library = DocumentLibrary(engine, second_config, "test-v1")
    second = second_library.ingest(owner, pdf, "report.pdf")
    third_library = DocumentLibrary(engine, second_config, "test-v2")
    third = third_library.ingest(owner, pdf, "report.pdf")
    assert first.document_id == second.document_id == third.document_id
    assert len({first.id, second.id, third.id}) == 3
    assert count(engine, owner, document_indexes) == 3 and count(engine, owner, pdf_files) == 1
    assert third_library.ask(owner, QUESTION, version_ids=[third.id]).retrieved
    with pytest.raises(RetrievalError, match="Re-index"):
        third_library.ask(owner, QUESTION, version_ids=[third.id, first.id])

    alternate = third_library.ingest(owner, pdf_fixtures.multipage_pdf(2), "other.pdf")
    alias = third_library.ingest(owner, pdf, "report.pdf", document_id=alternate.document_id)
    assert count(engine, owner, document_indexes) == 4
    third_library.delete_document(owner, third.document_id)
    assert third_library.read_pdf(owner, alias.id) == pdf
    third_library.delete_document(owner, alternate.document_id)
    for table in (document_indexes, pdf_files, chunks, chunk_embeddings):
        assert count(engine, owner, table) == 0


def test_failed_index_is_rolled_back_and_can_retry(engine, owners, monkeypatch):
    owner, _ = owners
    config = RAGConfig.offline()
    library = DocumentLibrary(engine, config, "test-v1", embedder=HashingEmbedder(config.embedding, fit_idf=False))
    data = pdf_fixtures.structured_pdf()
    with monkeypatch.context() as patch:
        patch.setattr(library.embedder, "embed_documents", lambda texts: np.full((len(texts), 8), np.nan))
        with pytest.raises(RetrievalError, match="finite"):
            library.ingest(owner, data, "report.pdf")
    for table in (documents, pdf_files, document_indexes, chunks, chunk_embeddings):
        assert count(engine, owner, table) == 0
    assert library.ingest(owner, data, "report.pdf").page_count > 0
