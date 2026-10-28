"""Tracing (what would be sent to Langfuse) and the exported trajectory, offline: spans
are captured in memory instead of being sent anywhere."""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from scout.config import Secrets
from scout.telemetry import langfuse_exporter, langfuse_settings
from tests.conftest import auth_headers
from tests.test_agent import run
from tests.test_tools import CONFIG


def by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    out: dict[str, list] = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


# --- the trace of a run ---


@pytest.mark.anyio
async def test_a_run_is_one_trace_with_steps_and_calls(db_factory, spans) -> None:
    report, _, ctx = await run(db_factory)
    names = by_name(spans)
    [root] = names["research"]
    assert {s.context.trace_id for s in spans.get_finished_spans()} == {root.context.trace_id}
    assert report.trace_id == trace.format_trace_id(root.context.trace_id)
    for step in ("planner", "researcher", "reader", "writer", "verifier", "finaliser"):
        assert step in names, step
    assert len(names["researcher"]) == 2  # one per parallel sub-question
    # Calls nest under their step, steps under the run.
    [planner] = names["planner"]
    [llm_planner] = names["llm.planner"]
    assert llm_planner.parent.span_id == planner.context.span_id
    assert planner.parent.span_id == root.context.span_id


@pytest.mark.anyio
async def test_every_span_carries_user_and_job(db_factory, spans) -> None:
    _, _, ctx = await run(db_factory)
    for s in spans.get_finished_spans():
        assert s.attributes["langfuse.user.id"] == str(ctx.user_id), s.name
        assert s.attributes["langfuse.session.id"] == str(ctx.job_id), s.name


@pytest.mark.anyio
async def test_model_calls_are_generations_with_tokens_and_usd_cost(db_factory, spans) -> None:
    await run(db_factory)
    [gen] = by_name(spans)["llm.writer"]
    a = gen.attributes
    assert a["langfuse.observation.type"] == "generation"
    assert a["gen_ai.request.model"] == CONFIG.llm.nodes["writer"].model
    assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == (100, 50)
    # The fake call costs £0.001; Langfuse gets USD.
    usd = json.loads(a["langfuse.observation.cost_details"])["total"]
    assert usd == pytest.approx(0.001 / CONFIG.usd_to_gbp)
    assert "Fact from S" in a["langfuse.observation.input"]  # the notes it wrote from


@pytest.mark.anyio
async def test_tool_calls_are_tool_observations_with_input_and_output(db_factory, spans) -> None:
    await run(db_factory)
    fetches = by_name(spans)["tool.fetch_url"]
    assert fetches and all(s.attributes["langfuse.observation.type"] == "tool" for s in fetches)
    assert "https://example.com/" in fetches[0].attributes["langfuse.observation.input"]
    assert "Official statistics page" in fetches[0].attributes["langfuse.observation.output"]


@pytest.mark.anyio
async def test_personal_data_is_masked_before_leaving(db_factory, spans) -> None:
    from scout.agent.graph import run_research
    from tests.test_agent import FakeLLM, default_script, web
    from tests.test_tools import make_ctx

    ctx = make_ctx(db_factory, web)
    await run_research(
        "Why did UK inflation rise? Reply to jane.doe@example.com",
        config=CONFIG,
        llm=FakeLLM(default_script),
        tool_ctx=ctx,
        use_docs=False,
    )
    blob = json.dumps([dict(s.attributes) for s in spans.get_finished_spans()])
    assert "jane.doe@example.com" not in blob and "[email]" in blob


@pytest.mark.anyio
async def test_content_capture_can_be_switched_off(db_factory, spans) -> None:
    cfg = CONFIG.model_copy(deep=True)
    cfg.telemetry.capture_content = False
    report, _, _ = await run(db_factory, config=cfg)
    for s in spans.get_finished_spans():
        assert "langfuse.observation.input" not in s.attributes, s.name
        assert "langfuse.observation.output" not in s.attributes, s.name
    # Tokens and cost are still traced; and the trajectory (our own DB) is still kept.
    assert by_name(spans)["llm.writer"][0].attributes["gen_ai.usage.input_tokens"] == 100
    assert report.trajectory


@pytest.mark.anyio
async def test_long_content_is_cut(db_factory, spans) -> None:
    cfg = CONFIG.model_copy(deep=True)
    cfg.telemetry.max_field_chars = 200
    await run(db_factory, config=cfg)
    for s in spans.get_finished_spans():
        for key in ("langfuse.observation.input", "langfuse.observation.output"):
            assert len(s.attributes.get(key, "")) <= 200 + len(" …[truncated]")


def test_langfuse_settings_use_basic_auth_and_the_otel_endpoint() -> None:
    secrets = Secrets(
        _env_file=None,
        langfuse_public_key=SecretStr("pk-lf-1"),
        langfuse_secret_key=SecretStr("sk-lf-2"),
    )
    endpoint, headers = langfuse_settings(secrets)
    assert endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode()
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert langfuse_exporter(secrets) is not None
    assert langfuse_settings(Secrets(_env_file=None)) is None  # no keys: tracing off


# --- the trajectory ---


@pytest.mark.anyio
async def test_trajectory_lists_every_call_in_order(db_factory) -> None:
    report, llm, _ = await run(db_factory)
    steps = report.trajectory
    assert [s.i for s in steps] == list(range(len(steps)))
    assert sum(s.kind == "llm_call" for s in steps) == len(llm.calls)
    assert {s.name for s in steps if s.kind == "tool_call"} >= {"web_search", "fetch_url"}
    writer = next(s for s in steps if s.node == "writer")
    assert writer.tokens_in == 100 and writer.cost_gbp == 0.001 and "parsed" in writer.output
    assert "trajectory" not in report.model_dump()  # stored separately, not in the report


def test_trajectory_endpoint(tmp_path: Path) -> None:
    from tests.test_research_api import QUESTION, make_client, wait_done

    with make_client(tmp_path) as client:
        headers = auth_headers(client)
        job_id = client.post("/research", json=QUESTION, headers=headers).json()["job_id"]
        wait_done(client, job_id, headers)
        body = client.get(f"/research/{job_id}/trajectory", headers=headers).json()
        assert body["status"] == "complete" and body["question"] == QUESTION["question"]
        kinds = {s["kind"] for s in body["steps"]}
        assert kinds == {"llm_call", "tool_call"}
        assert body["sources"] and body["claims"][0]["verdict"] == "supported"
        other = auth_headers(client, "other@example.com")
        assert client.get(f"/research/{job_id}/trajectory", headers=other).status_code == 404


def test_trajectory_of_an_unfinished_job_is_409(client) -> None:
    from datetime import UTC, datetime

    from scout.models import ResearchJob

    headers = auth_headers(client)
    user_id = uuid.UUID(client.get("/auth/me", headers=headers).json()["id"])
    with client.app.state.session_factory() as db:
        job = ResearchJob(
            user_id=user_id, question="q?", status="running", started_at=datetime.now(UTC)
        )
        db.add(job)
        db.commit()
        job_id = job.id
    assert client.get(f"/research/{job_id}/trajectory", headers=headers).status_code == 409


def test_only_scouts_own_spans_are_exported(spans) -> None:
    # Regression: the Anthropic and MCP SDKs trace themselves; exporting their spans
    # duplicated every call and doubled the cost Langfuse showed.
    with (
        trace.get_tracer("com.anthropic.sdk.python").start_as_current_span("sdk.call"),
        trace.get_tracer("scout").start_as_current_span("ours"),  # nested inside sdk.call
    ):
        pass
    assert [s.name for s in spans.get_finished_spans()] == ["ours"]


@pytest.mark.anyio
async def test_a_run_exports_exactly_one_span_per_call(db_factory, spans) -> None:
    report, llm, _ = await run(db_factory)
    names = [s.name for s in spans.get_finished_spans()]
    assert sum(n.startswith("llm.") for n in names) == len(llm.calls)
    assert sum(n.startswith("tool.") for n in names) == sum(
        1 for s in report.steps if s.kind == "tool" and s.detail != "refused: not a research tool"
    )
