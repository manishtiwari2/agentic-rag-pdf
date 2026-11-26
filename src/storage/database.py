"""Connections and tenant scoping for persistent data."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from ..errors import ConfigurationError, StorageError


def postgres_url(value: str):
    try:
        url = make_url(value)
    except ArgumentError as exc:
        raise ConfigurationError("DATABASE_URL must be a PostgreSQL connection URL.") from exc
    if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
        raise ConfigurationError("DATABASE_URL must use PostgreSQL with psycopg.")
    return url.set(drivername="postgresql+psycopg")


def build_engine(database_url: str) -> Engine:
    return create_engine(
        postgres_url(database_url), pool_pre_ping=True, pool_timeout=5, hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=15000"},
    )


@contextmanager
def tenant_transaction(engine: Engine, user_id: UUID) -> Iterator[Connection]:
    if not isinstance(user_id, UUID):
        raise ConfigurationError("A UUID user_id is required to access the document library.")
    try:
        with engine.begin() as conn:
            conn.execute(select(func.set_config("rag.user_id", str(user_id), True)))
            yield conn
    except SQLAlchemyError as exc:
        raise StorageError("The document library is temporarily unavailable.") from exc
