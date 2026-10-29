"""Alembic environment: points migrations at Scout's models and database."""

from __future__ import annotations

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import Connection

from scout import models  # noqa: F401  (registers every table on Base.metadata)
from scout.db import Base, make_engine

target_metadata = Base.metadata

# Created by raw SQL in the baseline migration (Postgres-only index types), so they are
# not on the models' metadata; without this, autogenerate would offer to drop them.
SQL_ONLY_INDEXES = {"ix_chunks_embedding_hnsw", "ix_chunks_text_fts"}


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "index" and name in SQL_ONLY_INDEXES)


def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    # SQLite has no vector type: it reads a Vector column back as NUMERIC. Skip that
    # one comparison there; on Postgres the real types are compared as usual.
    if context.dialect.name == "sqlite" and isinstance(metadata_type, Vector):
        return False
    return None  # Alembic's default comparison


def _url() -> str:
    # An explicit URL (tests pass one) wins; otherwise the app's own setting.
    url = context.config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from scout.config import get_secrets

    return get_secrets().database_url


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=compare_type,
        render_as_batch=connection.dialect.name == "sqlite",  # SQLite can't ALTER much
    )


def run_migrations_online() -> None:
    engine = make_engine(_url())
    with engine.connect() as connection:
        _configure(connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations_online()
