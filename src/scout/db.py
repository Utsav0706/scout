"""Database engine, sessions and table creation (sync SQLAlchemy 2.0).

Sync, not async: the RQ worker (M6) is sync anyway, FastAPI runs sync routes in a
thread pool, and Scout's load is a few requests a minute. Async would add a second
driver and more complexity for no measurable gain here.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        # Tests only: an in-memory DB exists per connection, so every thread must share
        # one connection (StaticPool). Not safe for concurrent use; file DBs below are.
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if database_url.startswith("sqlite"):
        # Local file DB: a connection per thread, and wait up to 15s for SQLite's write
        # lock instead of failing at once when parallel tool calls write together.
        return create_engine(database_url, connect_args={"check_same_thread": False, "timeout": 15})
    # pool_pre_ping drops dead connections (e.g. after Postgres restarts) instead of erroring.
    return create_engine(database_url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create tables that don't exist yet.

    create_all is enough while the schema is young. Once there is data worth keeping
    in a deployed DB, schema changes need real migrations (Alembic) instead.
    """
    from scout import models  # noqa: F401  (registers tables on Base.metadata)

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def db_is_up(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session
