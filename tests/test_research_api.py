"""POST/GET /research end to end, offline: fake LLM + fake web, through both job paths
(RQ on fake Redis in synchronous mode, and the API's inline mode)."""

import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from scout.api.main import create_app
from scout.config import Secrets, load_config
from scout.db import Base
from scout.jobs import worker
from scout.jobs.queue import QueueUnavailableError, RQQueue
from scout.jobs.runner import run_job
from scout.models import ResearchJob, User
from tests.conftest import REPO_ROOT, TEST_DATABASE_URL, TEST_JWT_SECRET, auth_headers
from tests.fakes import FakeEmbedder, FakeReranker
from tests.test_agent import FakeLLM, default_script, web

CONFIG = load_config(REPO_ROOT / "config.yaml")
QUESTION = {"question": "Why did UK inflation rise in 2026?", "use_docs": False}


def make_client(
    tmp_path: Path, *, mode="rq", script=default_script, config=CONFIG, queue_factory=None
) -> TestClient:
    # A file DB (not in-memory): jobs write from several threads at once.
    url = (
        TEST_DATABASE_URL
        if TEST_DATABASE_URL.startswith("postgresql")
        else (f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    )
    secrets = Secrets(
        _env_file=None,
        database_url=url,
        jwt_secret=SecretStr(TEST_JWT_SECRET),
        tavily_api_key=SecretStr("tvly-test"),
        log_level="WARNING",
        jobs_mode=mode,
    )

    def rq_sync(services, loop):
        worker.configure(services)  # the "worker" runs jobs against the test's services
        return RQQueue(fakeredis.FakeStrictRedis(), job_timeout_s=300, is_async=False)

    app = create_app(
        secrets=secrets,
        config=config,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(script),
        http_transport=httpx.MockTransport(web),
        queue_factory=queue_factory or (rq_sync if mode == "rq" else None),
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with make_client(tmp_path) as c:
        yield c
    worker.configure(None)
    Base.metadata.drop_all(c.app.state.engine)


def wait_done(client: TestClient, job_id: str, headers: dict, timeout_s: float = 20) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/research/{job_id}", headers=headers).json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job still {body['status']} after {timeout_s}s")


# --- the happy path, both ways of running jobs ---


def test_research_job_via_rq_worker(client: TestClient) -> None:
    headers = auth_headers(client)
    resp = client.post("/research", json=QUESTION, headers=headers)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.headers["location"] == f"/research/{job_id}" == resp.json()["status_url"]

    job = wait_done(client, job_id, headers)
    assert job["status"] == "complete"
    assert job["report"]["question"] == QUESTION["question"]
    assert job["report"]["verification"]["supported"] == 2
    assert job["cost_gbp"] == pytest.approx(job["report"]["cost_gbp"])
    messages = [e["message"] for e in job["events"]]
    assert messages[0] == "Research started"
    assert "Planned 2 sub-questions" in messages
    assert "Checked claims: 2/2 supported" in messages
    assert messages[-1].startswith("Finished: complete")


def test_research_job_inline_mode(tmp_path: Path) -> None:
    with make_client(tmp_path, mode="inline") as client:
        headers = auth_headers(client)
        job_id = client.post("/research", json=QUESTION, headers=headers).json()["job_id"]
        job = wait_done(client, job_id, headers)
        assert job["status"] == "complete"
        assert len(job["report"]["sections"][0]["claims"]) == 2


def test_report_is_hidden_until_the_job_finishes(client: TestClient) -> None:
    headers = auth_headers(client)
    user_id = client.get("/auth/me", headers=headers).json()["id"]
    with client.app.state.session_factory() as db:
        job = ResearchJob(
            user_id=uuid.UUID(user_id),
            question="q?",
            status="running",
            started_at=datetime.now(UTC),
            report={"leak": True},
        )
        db.add(job)
        db.commit()
        job_id = job.id
    body = client.get(f"/research/{job_id}", headers=headers).json()
    assert (body["status"], body["report"]) == ("running", None)


# --- validation and access ---


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Why?"},  # too short
        {"question": "     a  b      "},  # too short once whitespace is squeezed
        {"question": "x" * 1001},
        {"question": "A fine question here", "use_web": False, "use_docs": False},
        {"question": "A fine question here", "priority": "high"},  # unknown field
    ],
)
def test_bad_requests_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/research", json=payload, headers=auth_headers(client)).status_code == 422


def test_login_required(client: TestClient) -> None:
    assert client.post("/research", json=QUESTION).status_code == 401
    assert client.get(f"/research/{uuid.uuid4()}").status_code == 401


def test_other_users_jobs_are_not_found(client: TestClient) -> None:
    job_id = client.post("/research", json=QUESTION, headers=auth_headers(client)).json()["job_id"]
    other = auth_headers(client, "other@example.com")
    assert client.get(f"/research/{job_id}", headers=other).status_code == 404
    assert client.get(f"/research/{uuid.uuid4()}", headers=other).status_code == 404


# --- limits ---


def _add_jobs(client: TestClient, headers: dict, statuses: list[str], age_min: int = 0) -> None:
    user_id = uuid.UUID(client.get("/auth/me", headers=headers).json()["id"])
    with client.app.state.session_factory() as db:
        for s in statuses:
            db.add(
                ResearchJob(
                    user_id=user_id,
                    question="q?",
                    status=s,
                    created_at=datetime.now(UTC) - timedelta(minutes=age_min),
                )
            )
        db.commit()


def test_too_many_running_jobs_is_429(client: TestClient) -> None:
    headers = auth_headers(client)
    _add_jobs(client, headers, ["queued", "running"])  # the limit is 2
    resp = client.post("/research", json=QUESTION, headers=headers)
    assert resp.status_code == 429
    assert "running" in resp.json()["detail"]
    assert resp.headers["retry-after"] == "30"


def test_hourly_limit_is_429_with_retry_after(client: TestClient) -> None:
    headers = auth_headers(client)
    _add_jobs(client, headers, ["complete"] * 10, age_min=50)  # 10 per hour, oldest 50 min ago
    resp = client.post("/research", json=QUESTION, headers=headers)
    assert resp.status_code == 429
    assert 9 * 60 <= int(resp.headers["retry-after"]) <= 10 * 60  # frees up in ~10 minutes


def test_old_jobs_dont_count_toward_the_hourly_limit(client: TestClient) -> None:
    headers = auth_headers(client)
    _add_jobs(client, headers, ["complete"] * 10, age_min=61)
    assert client.post("/research", json=QUESTION, headers=headers).status_code == 202


# --- failures ---


def test_queue_down_is_503_and_job_marked_failed(tmp_path: Path) -> None:
    class DownQueue:
        def enqueue(self, job_id):
            raise QueueUnavailableError("connection refused")

    with make_client(tmp_path, queue_factory=lambda s, loop: DownQueue()) as client:
        headers = auth_headers(client)
        resp = client.post("/research", json=QUESTION, headers=headers)
        assert resp.status_code == 503
        with client.app.state.session_factory() as db:
            job = db.query(ResearchJob).one()
        assert (job.status, job.error) == ("failed", "The job queue is unavailable.")


def test_crash_inside_research_fails_the_job_without_leaking_details(tmp_path: Path) -> None:
    def script(node, kw):
        if node == "planner":
            raise RuntimeError("secret internal detail")
        return default_script(node, kw)

    with make_client(tmp_path, script=script) as client:
        headers = auth_headers(client)
        job_id = client.post("/research", json=QUESTION, headers=headers).json()["job_id"]
        job = wait_done(client, job_id, headers)
        assert job["status"] == "failed"
        assert job["error"] == "Research failed (RuntimeError)."
        assert "secret" not in str(job)


def test_job_abandoned_by_its_worker_is_marked_failed(client: TestClient) -> None:
    headers = auth_headers(client)
    user_id = uuid.UUID(client.get("/auth/me", headers=headers).json()["id"])
    with client.app.state.session_factory() as db:
        job = ResearchJob(
            user_id=user_id,
            question="q?",
            status="running",
            started_at=datetime.now(UTC) - timedelta(minutes=11),
        )
        db.add(job)
        db.commit()
        job_id = job.id
    body = client.get(f"/research/{job_id}", headers=headers).json()
    assert body["status"] == "failed"
    assert "stopped responding" in body["error"]


@pytest.mark.anyio
async def test_run_job_only_starts_queued_jobs(client: TestClient) -> None:
    services = client.app.state.services
    with services.session_factory() as db:
        user = User(email="x@example.com", password_hash="!")
        db.add(user)
        db.commit()
        job = ResearchJob(user_id=user.id, question="q?", status="complete")
        db.add(job)
        db.commit()
        job_id = job.id
    await run_job(services, job_id)  # already finished: must be a no-op
    await run_job(services, uuid.uuid4())  # unknown id: also a no-op
    with services.session_factory() as db:
        assert db.get(ResearchJob, job_id).status == "complete"
