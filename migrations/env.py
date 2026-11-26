"""Run migrations with the PostgreSQL URL supplied by the environment."""

import os

from alembic import context
from sqlalchemy import create_engine, pool

from src.errors import ConfigurationError
from src.storage.database import postgres_url
from src.storage.schema import metadata


def migrate(connection) -> None:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()


def run() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        migrate(connection)
        return
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise ConfigurationError("Set DATABASE_URL before running Alembic.")
    url = postgres_url(value)
    if context.is_offline_mode():
        context.configure(dialect_name="postgresql", target_metadata=metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(url, poolclass=pool.NullPool, hide_parameters=True)
    with engine.connect() as connection:
        migrate(connection)


run()
