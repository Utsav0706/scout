"""Alembic migrations build the same schema as the models.

If someone changes src/scout/models.py without adding a migration, `alembic check`
finds the difference and this test fails (on SQLite locally, on Postgres in CI).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from scout.db import Base, make_engine
from tests.conftest import ON_POSTGRES, REPO_ROOT, TEST_DATABASE_URL


@pytest.fixture
def db_url(tmp_path: Path) -> Iterator[str]:
    url = TEST_DATABASE_URL if ON_POSTGRES else f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    engine = make_engine(url)

    def clean() -> None:  # CI shares one Postgres database between tests
        Base.metadata.drop_all(engine)
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    clean()
    yield url
    clean()
    engine.dispose()


def alembic_config(url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_migrations_match_the_models_and_downgrade_cleanly(db_url: str) -> None:
    cfg = alembic_config(db_url)
    command.upgrade(cfg, "head")
    command.check(cfg)  # raises if the models and the migrated schema differ

    engine = make_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) <= tables
    if ON_POSTGRES:
        indexes = {i["name"] for i in inspect(engine).get_indexes("chunks")}
        assert {"ix_chunks_embedding_hnsw", "ix_chunks_text_fts"} <= indexes

    command.downgrade(cfg, "base")
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    engine.dispose()
