"""GET /research (history), GET /research/stats and POST /research/{id}/feedback."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scout.db import Base
from scout.models import ResearchJob
from tests.conftest import auth_headers
from tests.test_research_api import QUESTION, make_client, wait_done


def _user_id(client: TestClient, headers: dict) -> uuid.UUID:
    return uuid.UUID(client.get("/auth/me", headers=headers).json()["id"])


def _job(client, user_id, status="complete", age=timedelta(0), cost=0.03, latency=40.0):
    with client.app.state.session_factory() as db:
        job = ResearchJob(
            user_id=user_id,
            question=f"q {uuid.uuid4().hex[:6]}?",
            status=status,
            created_at=datetime.now(UTC) - age,
            cost_gbp=cost,
            report={"latency_s": latency, "sections": [], "sources": []}
            if status != "running"
            else None,
        )
        db.add(job)
        db.commit()
        return job.id


# --- history ---


def test_history_lists_my_jobs_newest_first_with_paging(client: TestClient) -> None:
    headers = auth_headers(client)
    me = _user_id(client, headers)
    ids = [_job(client, me, age=timedelta(minutes=m)) for m in (30, 20, 10)]
    _job(client, _user_id(client, other := auth_headers(client, "o@example.com")))

    page1 = client.get("/research?limit=2", headers=headers).json()
    assert [j["id"] for j in page1] == [str(ids[2]), str(ids[1])]
    page2 = client.get(f"/research?limit=2&after={page1[-1]['id']}", headers=headers)
    assert [j["id"] for j in page2.json()] == [str(ids[0])]
    assert len(client.get("/research", headers=other).json()) == 1  # only their own
    # Someone else's job id as the cursor gives nothing, not their place in your list.
    assert client.get(f"/research?after={ids[0]}", headers=other).json() == []


def test_history_pages_through_jobs_created_in_the_same_instant(client: TestClient) -> None:
    headers = auth_headers(client)
    me = _user_id(client, headers)
    same = datetime.now(UTC) - timedelta(minutes=5)
    with client.app.state.session_factory() as db:
        for _ in range(3):
            db.add(ResearchJob(user_id=me, question="q?", status="complete", created_at=same))
        db.commit()

    seen, after = [], None
    for _ in range(4):
        url = "/research?limit=1" + (f"&after={after}" if after else "")
        page = client.get(url, headers=headers).json()
        if not page:
            break
        seen.append(after := page[0]["id"])
    assert len(seen) == len(set(seen)) == 3  # each job once, none skipped


def test_stats_route_is_not_mistaken_for_a_job_id(client: TestClient) -> None:
    assert client.get("/research/stats", headers=auth_headers(client)).status_code == 200


# --- stats ---


def test_stats_summarise_my_recent_jobs(client: TestClient) -> None:
    headers = auth_headers(client)
    me = _user_id(client, headers)
    _job(client, me, cost=0.02, latency=30)
    _job(client, me, cost=0.04, latency=50)
    _job(client, me, status="failed", cost=0.0)
    _job(client, me, status="running")
    _job(client, me, age=timedelta(days=40))  # outside the window
    s = client.get("/research/stats?days=30", headers=headers).json()
    assert (s["jobs"], s["finished"]) == (4, 3)
    assert s["by_status"] == {"complete": 2, "failed": 1, "running": 1}
    assert s["failure_rate"] == round(1 / 3, 3)
    assert s["avg_cost_gbp"] == round((0.02 + 0.04 + 0.0) / 3, 4)
    assert s["avg_latency_s"] == 40.0
    assert sum(d["reports"] for d in s["reports_per_day"]) == 3


def test_stats_with_no_jobs(client: TestClient) -> None:
    s = client.get("/research/stats", headers=auth_headers(client)).json()
    assert (s["jobs"], s["avg_cost_gbp"], s["failure_rate"]) == (0, None, None)


# --- feedback ---


def test_feedback_is_stored_overwritten_and_sent_to_langfuse(
    tmp_path: Path, spans, request: pytest.FixtureRequest
) -> None:
    # spans: tracing on, so the run has a trace id to attach the score to.
    sent = []
    with make_client(tmp_path) as client:
        request.addfinalizer(lambda: Base.metadata.drop_all(client.app.state.engine))
        client.app.state.send_score = lambda *args: sent.append(args)
        headers = auth_headers(client)
        job_id = client.post("/research", json=QUESTION, headers=headers).json()["job_id"]
        wait_done(client, job_id, headers)

        r = client.post(
            f"/research/{job_id}/feedback",
            headers=headers,
            json={"rating": "down", "comment": "  missed the​ 2025 data "},
        )
        assert r.status_code == 200 and r.json()["comment"] == "missed the 2025 data"
        client.post(f"/research/{job_id}/feedback", headers=headers, json={"rating": "up"})

        assert client.get("/research", headers=headers).json()[0]["rating"] == "up"
        stats = client.get("/research/stats", headers=headers).json()
        assert (stats["thumbs_up"], stats["thumbs_down"]) == (1, 0)  # overwritten, not added
        # Sent to Langfuse each time, keyed by this run's trace and job.
        assert [(a[1], a[2]) for a in sent] == [(job_id, "down"), (job_id, "up")]


def test_feedback_rules(client: TestClient) -> None:
    headers = auth_headers(client)
    me = _user_id(client, headers)
    running = _job(client, me, status="running")
    done = _job(client, me)
    assert (
        client.post(
            f"/research/{running}/feedback", headers=headers, json={"rating": "up"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/research/{done}/feedback", headers=headers, json={"rating": "meh"}
        ).status_code
        == 422
    )
    other = auth_headers(client, "o@example.com")
    assert (
        client.post(f"/research/{done}/feedback", headers=other, json={"rating": "up"}).status_code
        == 404
    )
