"""Protections for a public deployment: an invite code to sign up, and a daily cap on
research spending across all users."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scout.db import Base
from scout.jobs import worker
from scout.models import ResearchJob
from scout.ui.client import ScoutError
from tests.conftest import auth_headers
from tests.test_research_api import QUESTION, make_client
from tests.test_ui import ScoutClient

CREDS = {"email": "new@example.com", "password": "correct-horse-battery"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with make_client(tmp_path) as c:
        yield c
    worker.configure(None)
    Base.metadata.drop_all(c.app.state.engine)


# --- invite code ---


def test_sign_up_is_open_without_an_invite_code_configured(client: TestClient) -> None:
    assert client.post("/auth/register", json=CREDS).status_code == 201


def test_sign_up_needs_the_right_invite_code_when_configured(client: TestClient) -> None:
    client.app.state.registration_code = "lark-7"
    for body in (CREDS, {**CREDS, "invite_code": "wrong"}, {**CREDS, "invite_code": ""}):
        r = client.post("/auth/register", json=body)
        assert r.status_code == 403 and "invite code" in r.json()["detail"]
    assert client.post("/auth/register", json={**CREDS, "invite_code": "lark-7"}).status_code == 201
    # Logging in never needs the code.
    assert client.post("/auth/login", json=CREDS).status_code == 200


def test_ui_client_sends_the_invite_code(client: TestClient) -> None:
    client.app.state.registration_code = "lark-7"
    ui = ScoutClient(http=client)
    with pytest.raises(ScoutError) as err:
        ui.register(CREDS["email"], CREDS["password"])
    assert err.value.status == 403
    ui.register(CREDS["email"], CREDS["password"], invite_code="lark-7")
    ui.login(CREDS["email"], CREDS["password"])
    assert ui.token


# --- daily budget ---


def _add_job(client: TestClient, status: str, cost: float, age=timedelta(0)) -> None:
    me = client.get("/auth/me", headers=auth_headers(client, "other@example.com")).json()
    with client.app.state.session_factory() as db:
        db.add(
            ResearchJob(
                user_id=uuid.UUID(me["id"]),
                question="q?",
                status=status,
                cost_gbp=cost,
                created_at=datetime.now(UTC) - age,
            )
        )
        db.commit()


def test_no_cap_by_default(client: TestClient) -> None:
    _add_job(client, "complete", 100.0)
    assert client.post("/research", json=QUESTION, headers=auth_headers(client)).status_code == 202


def test_new_jobs_refused_once_todays_spend_could_pass_the_cap(client: TestClient) -> None:
    client.app.state.daily_budget_gbp = 1.0  # max_cost_gbp per job is 0.20
    headers = auth_headers(client)
    _add_job(client, "complete", 0.70)
    assert client.post("/research", json=QUESTION, headers=headers).status_code == 202

    _add_job(client, "complete", 0.15)  # spent today: 0.70 + 0.15 + the job above
    r = client.post("/research", json=QUESTION, headers=headers)
    assert r.status_code == 503 and "budget" in r.json()["detail"]
    assert 0 < int(r.headers["Retry-After"]) <= 24 * 3600  # until UTC midnight


def test_running_jobs_count_at_their_maximum_cost(client: TestClient) -> None:
    client.app.state.daily_budget_gbp = 0.5
    for _ in range(2):
        _add_job(client, "running", 0.0)  # 2 x 0.20 committed, + 0.20 for a new one
    assert client.post("/research", json=QUESTION, headers=auth_headers(client)).status_code == 503


def test_yesterdays_spend_does_not_count(client: TestClient) -> None:
    client.app.state.daily_budget_gbp = 1.0
    _add_job(client, "complete", 5.0, age=timedelta(days=1, hours=1))
    assert client.post("/research", json=QUESTION, headers=auth_headers(client)).status_code == 202
