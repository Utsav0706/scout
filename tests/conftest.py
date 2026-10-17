import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from scout.api.main import create_app
from scout.config import Secrets, load_config
from scout.db import Base

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_JWT_SECRET = "test-secret-" + "x" * 40


@pytest.fixture
def secrets() -> Secrets:
    # CI sets TEST_DATABASE_URL to a real Postgres; locally we fall back to in-memory SQLite.
    return Secrets(
        _env_file=None,
        database_url=os.environ.get("TEST_DATABASE_URL", "sqlite://"),
        jwt_secret=SecretStr(TEST_JWT_SECRET),
        log_level="WARNING",
    )


@pytest.fixture
def client(secrets: Secrets) -> Iterator[TestClient]:
    app = create_app(secrets=secrets, config=load_config(REPO_ROOT / "config.yaml"))
    with TestClient(app) as c:  # "with" runs startup, which creates the tables
        yield c
    # Fresh tables for every test, so tests can't leak data into each other.
    Base.metadata.drop_all(app.state.engine)
