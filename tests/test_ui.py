"""The UI: ScoutClient against the real API (in-process, fake model and web), and the
Streamlit pages with AppTest."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from scout.db import Base
from scout.jobs import worker
from scout.ui.client import ScoutClient, ScoutError
from scout.ui.views import md
from tests.conftest import REPO_ROOT
from tests.fakes import make_pdf
from tests.test_research_api import QUESTION, make_client

APP = str(REPO_ROOT / "ui" / "app.py")


@pytest.fixture
def api(tmp_path: Path) -> Iterator[TestClient]:
    with make_client(tmp_path) as client:
        yield client
    worker.configure(None)
    Base.metadata.drop_all(client.app.state.engine)  # CI shares one Postgres DB


@pytest.fixture
def scout(api: TestClient) -> ScoutClient:
    c = ScoutClient(http=api)  # TestClient is an httpx.Client: the real API, in-process
    c.register("uv@example.com", "correct-horse-battery")
    c.login("uv@example.com", "correct-horse-battery")
    return c


# --- ScoutClient ---


def test_full_flow_through_the_client(scout: ScoutClient) -> None:
    job_id = scout.start_research(QUESTION["question"], use_docs=False)
    events = list(scout.stream(job_id))
    assert events[-1] == ("done", {"status": "complete"})
    assert any(e == "progress" and d["message"].startswith("Planned") for e, d in events)

    job = scout.job(job_id)
    assert job["report"]["verification"]["supported"] == 2
    assert scout.history()[0]["id"] == job_id
    assert scout.feedback(job_id, "up")["rating"] == "up"
    assert scout.job(job_id)["feedback"] == {"job_id": job_id, "rating": "up", "comment": None}
    assert scout.history()[0]["rating"] == "up"
    assert scout.stats()["thumbs_up"] == 1


def test_documents_through_the_client(scout: ScoutClient) -> None:
    doc = scout.upload("notes.pdf", make_pdf(["Some notes about inflation."]))
    assert doc["status"] == "ready"  # the test queue processes it straight away
    assert [d["filename"] for d in scout.documents()] == ["notes.pdf"]


def test_api_errors_become_readable_messages(scout: ScoutClient) -> None:
    with pytest.raises(ScoutError) as err:
        scout.start_research("Why?")  # too short
    assert err.value.status == 422 and "at least 5 characters" in err.value.message


def test_wrong_password_and_expired_session(api: TestClient) -> None:
    c = ScoutClient(http=api)
    c.register("a@example.com", "correct-horse-battery")
    with pytest.raises(ScoutError) as err:
        c.login("a@example.com", "wrong-password")
    assert err.value.status == 401 and c.token is None
    c.token = "not-a-real-token"
    with pytest.raises(ScoutError):
        c.history()
    assert c.token is None  # dropped, so the UI shows the login page again


# --- markdown safety ---


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("between $5 and $10", r"between \$5 and \$10"),  # not maths
        ("**bold** <b>x</b>", r"\*\*bold\*\* \<b\>x\</b\>"),  # no formatting or HTML
        ("[click](https://evil.example)", r"\[click\](https://evil.example)"),  # no links
    ],
)
def test_model_text_is_escaped(text: str, shown: str) -> None:
    assert md(text) == shown


# --- pages (AppTest) ---


def test_logged_out_shows_login(api: TestClient) -> None:
    at = AppTest.from_file(APP)
    at.session_state["client"] = ScoutClient(http=api)
    at.run()
    assert not at.exception
    assert at.title[0].value == "🔎 Scout"
    assert {b.label for b in at.button} >= {"Log in", "Create account"}


def test_report_page_shows_claims_badges_sources_and_cost(scout: ScoutClient) -> None:
    job_id = scout.start_research(QUESTION["question"], use_docs=False)
    list(scout.stream(job_id))  # wait for it to finish

    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["client"] = scout
    at.session_state["job_id"] = job_id
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "UK inflation rose." in text  # the summary
    assert "Claim backed by S" in text and "✅" in text  # claims with verdict badges
    assert "example.com" in text  # source links inside the citation pop-ups
    assert "Official statistics page about" in text  # the verifier's quote
    assert any("£" in c.value and "claims supported" in c.value for c in at.caption)


def finished_job(scout: ScoutClient) -> str:
    job_id = scout.start_research(QUESTION["question"], use_docs=False)
    list(scout.stream(job_id))  # wait for it to finish
    return job_id


def app(scout: ScoutClient, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["client"] = scout
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def test_feedback_form_saves_rating_and_comment(scout: ScoutClient) -> None:
    job_id = finished_job(scout)
    at = app(scout, job_id=job_id)
    assert at.feedback[0].value is None

    at.feedback[0].set_value(0)
    at.text_input(key=f"comment-{job_id}").set_value("Missed the 2025 figures")
    at.button(key=f"FormSubmitter:feedback-{job_id}-Send feedback").click().run()
    assert not at.exception
    assert at.success[0].value == "Thanks, feedback saved."
    assert scout.job(job_id)["feedback"]["rating"] == "down"
    assert scout.job(job_id)["feedback"]["comment"] == "Missed the 2025 figures"

    again = app(scout, job_id=job_id)  # reopening the report shows the saved rating
    assert again.feedback[0].value == 0


def test_feedback_needs_a_rating(scout: ScoutClient) -> None:
    job_id = finished_job(scout)
    at = app(scout, job_id=job_id)
    at.button(key=f"FormSubmitter:feedback-{job_id}-Send feedback").click().run()
    assert at.warning[0].value == "Pick 👍 or 👎 first."
    assert scout.job(job_id)["feedback"] is None


def test_history_lists_jobs_and_opens_one(scout: ScoutClient) -> None:
    job_id = finished_job(scout)
    scout.feedback(job_id, "up")
    at = app(scout, page="History")
    assert not at.exception
    assert any(QUESTION["question"][:20] in m.value for m in at.markdown)
    assert any("complete" in c.value and "👍" in c.value for c in at.caption)

    at.button(key=f"open-{job_id}").click().run()
    assert at.session_state["page"] == "Ask"
    assert any("UK inflation rose." in m.value for m in at.markdown)  # the report


def test_history_pages_through_older_jobs(scout: ScoutClient, monkeypatch) -> None:
    from scout.ui import views

    monkeypatch.setattr(views, "PAGE_SIZE", 1)
    jobs = {finished_job(scout), finished_job(scout)}

    def shown(at: AppTest) -> set[str]:
        return {b.key.removeprefix("open-") for b in at.button if (b.key or "").startswith("open-")}

    at = app(scout, page="History")
    page1 = shown(at)
    at.button[-2].click().run()  # Older → (the last button is Log out)
    page2 = shown(at)
    assert len(page1) == len(page2) == 1 and page1 | page2 == jobs
    assert "← Newest" in [b.label for b in at.button]


def test_documents_page_lists_uploads(scout: ScoutClient) -> None:
    from tests.fakes import make_pdf

    at = app(scout, page="Documents")
    assert at.info[0].value == "No documents yet."
    scout.upload("notes.pdf", make_pdf(["Some notes about inflation."]))
    at.run()
    assert any(
        "notes.pdf" in m.value and "ready" in m.value and "1 page" in m.value for m in at.markdown
    )


def test_usage_page_shows_totals(scout: ScoutClient) -> None:
    at = app(scout, page="Usage")
    assert at.info[0].value == "No research in this period."
    scout.feedback(finished_job(scout), "up")
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Reports"] == "1" and metrics["👍"] == "1" and metrics["Failed"] == "0%"
