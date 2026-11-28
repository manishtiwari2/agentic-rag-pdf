"""Apply Compose migrations and grant the app login its runtime permissions."""

import os

from alembic import command
from alembic.config import Config
from psycopg import sql

from ..errors import ConfigurationError
from .database import build_engine
from .schema import metadata


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    password = os.environ.get("RAG_DB_PASSWORD")
    if not database_url or not password:
        raise ConfigurationError("Set DATABASE_URL and RAG_DB_PASSWORD for database setup.")

    engine = build_engine(database_url)
    try:
        with engine.begin() as db:
            db.exec_driver_sql("SELECT pg_advisory_xact_lock(841762502)")
            db.exec_driver_sql("SET LOCAL search_path TO public")
            config = Config("alembic.ini")
            config.attributes["connection"] = db
            command.upgrade(config, "head")

            connection = db.connection.driver_connection
            verifier = connection.pgconn.encrypt_password(password.encode(), b"rag_app", b"scram-sha-256").decode()
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = 'rag_app'")
                if cursor.fetchone() is None:
                    cursor.execute("CREATE ROLE rag_app")
                cursor.execute(sql.SQL(
                    "ALTER ROLE rag_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE "
                    "NOREPLICATION PASSWORD {}"
                ).format(sql.Literal(verifier)))
                cursor.execute("GRANT USAGE ON SCHEMA public TO rag_app")
                tables = sql.SQL(", ").join(sql.Identifier("public", table.name) for table in metadata.sorted_tables)
                cursor.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO rag_app").format(tables))
                cursor.execute("GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO rag_app")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
