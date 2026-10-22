"""The full research graph, offline: a scripted fake LLM plays each node, while the
real MCP tool server runs against fake websites (no network, no API cost)."""

import re
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select

from scout.agent.budget import BudgetExceeded, RunBudget
from scout.agent.graph import run_research
from scout.agent.nodes import _pick_sources
from scout.agent.report import to_markdown
from scout.agent.state import DraftClaim, DraftOut, DraftSection, Evidence, PlanOut, ReaderOut
from scout.llm.client import LLMError, LLMResult, ToolCall, Usage
from scout.models import Note
from scout.models import ToolCall as ToolCallRow
from tests.test_tools import CONFIG, html, make_ctx

SUBS = ["What is the UK CPI rate?", "What drove UK inflation?"]


def result(
    node: str,
    *,
    text: str = "",
    parsed: Any = None,
    calls: list[ToolCall] | None = None,
    cost: float = 0.001,
) -> LLMResult:
    return LLMResult(
        node=node,
        model="fake",
        text=text,
        stop_reason="end_turn",
        usage=Usage(100, 50),
        cost_gbp=cost,
        latency_ms=5,
        parsed=parsed,
        tool_calls=calls or [],
        content=[{"type": "text", "text": text or "ok"}],
    )


def slug(text: str) -> str:
    return re.sub(r"[^a-z]+", "-", text.lower()).strip("-")


def default_script(node: str, kw: dict[str, Any]) -> LLMResult | Exception:
    msgs = kw["messages"]
    if node == "planner":
        return result(node, parsed=PlanOut(sub_questions=SUBS, search_plan="Use ONS."))
    if node == "researcher":
        sub = msgs[0]["content"]
        if len(msgs) == 1:
            return result(node, calls=[ToolCall("t1", "web_search", {"query": sub})])
        if len(msgs) == 3:
            url = f"https://example.com/{slug(sub)}"
            return result(node, calls=[ToolCall("t2", "fetch_url", {"url": url})])
        return result(node, text="Read enough.")
    if node == "reader":
        sid = re.search(r'<source id="(S\d+)"', msgs[0]["content"]).group(1)
        return result(node, parsed=ReaderOut(facts=[f"Fact from {sid}."]))
    if node == "writer":
        ids = re.findall(r"^\[(S\d+)\]", msgs[0]["content"], re.M)
        claims = [DraftClaim(text=f"Claim backed by {i}.", source_ids=[i]) for i in ids]
        return result(
            node,
            parsed=DraftOut(
                summary="UK inflation rose.",
                sections=[DraftSection(heading="Findings", claims=claims)],
                limitations=[],
            ),
        )
    raise AssertionError(node)


class FakeLLM:
    def __init__(self, script: Callable[[str, dict[str, Any]], LLMResult | Exception]) -> None:
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def complete(self, node: str, **kw: Any) -> LLMResult:
        self.calls.append((node, kw))
        out = self.script(node, kw)
        if isinstance(out, Exception):
            raise out
        return out

    def nodes(self) -> list[str]:
        return [n for n, _ in self.calls]


def web(req: httpx.Request) -> httpx.Response:
    if req.url.host == "api.tavily.com":
        query = slug(__import__("json").loads(req.content)["query"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"Result for {query}",
                        "url": f"https://example.com/{query}",
                        "content": "preview",
                    }
                ]
            },
        )
    return html(f"Official statistics page about {req.url.path}.", title=f"ONS {req.url.path}")


async def run(db_factory, script=default_script, handler=web, config=CONFIG, **kw):
    llm = FakeLLM(script)
    ctx = make_ctx(db_factory, handler)
    report = await run_research(
        "Why did UK inflation rise?", config=config, llm=llm, tool_ctx=ctx, use_docs=False, **kw
    )
    return report, llm, ctx


# --- happy path ---


@pytest.mark.anyio
async def test_full_run_produces_cited_report(db_factory) -> None:
    report, llm, ctx = await run(db_factory)

    assert report.status == "complete", report.limitations
    assert llm.nodes().count("planner") == 1
    assert llm.nodes().count("researcher") == 2 * 3  # 2 sub-questions x 3 turns
    assert llm.nodes().count("reader") == 2  # one fetched page per sub-question
    assert llm.nodes()[-1] == "writer"

    claims = [c for s in report.sections for c in s.claims]
    assert len(claims) == 2
    cited = {sid for c in claims for sid in c.source_ids}
    assert cited == {s.id for s in report.sources}  # every cited id is listed, nothing extra
    assert all(s.url_or_doc.startswith("https://example.com/") for s in report.sources)
    assert all(c.verdict == "unverified" for c in claims)  # verifier arrives in M5
    assert report.cost_gbp == pytest.approx(0.001 * len(llm.calls))
    assert set(report.prompt_versions) == {"planner", "researcher", "reader", "writer"}


@pytest.mark.anyio
async def test_notes_and_tool_calls_are_persisted(db_factory) -> None:
    report, _, ctx = await run(db_factory)
    with db_factory() as db:
        notes = db.scalars(select(Note).where(Note.job_id == ctx.job_id)).all()
        audited = db.scalar(select(func.count()).select_from(ToolCallRow))
    assert {n.source_id for n in notes} == {s.id for s in report.sources}
    tool_steps = [s for s in report.steps if s.kind == "tool"]
    assert audited == len(tool_steps)  # every tool call has its audit row


@pytest.mark.anyio
async def test_researchers_only_get_research_tools(db_factory) -> None:
    _, llm, _ = await run(db_factory)
    tools = next(kw["tools"] for n, kw in llm.calls if n == "researcher")
    assert {t["name"] for t in tools} == {"web_search", "fetch_url"}  # no save_note, no docs


@pytest.mark.anyio
async def test_source_text_is_wrapped_as_data_for_the_reader(db_factory) -> None:
    _, llm, _ = await run(db_factory)
    content = next(kw["messages"][0]["content"] for n, kw in llm.calls if n == "reader")
    assert content.startswith('<source id="S') and content.rstrip().endswith("</source>")


def test_markdown_rendering(db_factory) -> None:
    import asyncio

    report, _, _ = asyncio.run(run(db_factory))
    md = to_markdown(report)
    assert md.startswith("# Why did UK inflation rise?")
    assert "## Sources" in md and "[S" in md and "£" in md


# --- things going wrong ---


@pytest.mark.anyio
async def test_tool_errors_are_returned_to_the_model(db_factory) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.tavily.com":
            return web(req)
        return httpx.Response(404)

    report, llm, _ = await run(db_factory, handler=handler)
    third = [kw for n, kw in llm.calls if n == "researcher"][2]
    tool_result = third["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "HTTP 404" in tool_result["content"]
    assert report.status == "no_evidence"  # nothing could be read
    assert "writer" not in llm.nodes()  # no notes -> no paid write call
    assert report.summary.startswith("Not enough evidence")


@pytest.mark.anyio
async def test_invented_source_ids_are_dropped(db_factory) -> None:
    def script(node, kw):
        if node == "writer":
            return result(
                node,
                parsed=DraftOut(
                    summary="s",
                    limitations=[],
                    sections=[
                        DraftSection(
                            heading="h",
                            claims=[
                                DraftClaim(text="Real.", source_ids=["S2"]),
                                DraftClaim(text="Made up.", source_ids=["S99"]),
                            ],
                        )
                    ],
                ),
            )
        return default_script(node, kw)

    report, _, _ = await run(db_factory, script=script)
    texts = [c.text for s in report.sections for c in s.claims]
    assert texts == ["Real."]
    assert any("1 claim(s) removed" in item for item in report.limitations)


@pytest.mark.anyio
async def test_planner_failure_gives_failed_report_without_research(db_factory) -> None:
    def script(node, kw):
        return LLMError("boom", cost_gbp=0.002) if node == "planner" else default_script(node, kw)

    report, llm, _ = await run(db_factory, script=script)
    assert llm.nodes() == ["planner"]
    assert report.status == "failed"
    assert report.cost_gbp == pytest.approx(0.002)  # a failed call still costs money
    assert any("Planning failed" in item for item in report.limitations)


@pytest.mark.anyio
async def test_step_limit_returns_partial_report(db_factory) -> None:
    tight = CONFIG.model_copy(deep=True)
    tight.limits.max_steps = 4  # planner + 3 researcher turns, then nothing
    report, llm, _ = await run(db_factory, config=tight)
    assert len(llm.calls) == 4
    assert report.status == "partial"
    assert "step limit" in report.stop_reason
    md = to_markdown(report)
    assert "Status: partial" in md


@pytest.mark.anyio
async def test_cost_limit_keeps_money_for_the_writer(db_factory) -> None:
    def script(node, kw):
        out = default_script(node, kw)
        out.cost_gbp = 0.03  # each call costs 3p
        return out

    capped = CONFIG.model_copy(deep=True)
    capped.limits.max_cost_gbp = 0.20
    capped.agent.writer_reserve_gbp = 0.06
    report, llm, _ = await run(db_factory, script=script, config=capped)
    assert report.status == "partial"
    assert "cost limit" in report.stop_reason
    # Research stopped once spend + reserve reached the cap (with parallel researchers
    # each allowed to finish one in-flight call), and nothing ran past the cap.
    research = sum(0.03 for n in llm.nodes() if n != "writer")
    assert research <= 0.20 - 0.06 + 2 * 0.03
    assert report.cost_gbp <= 0.20 + 0.03


@pytest.mark.anyio
async def test_needs_at_least_one_source_type(db_factory) -> None:
    with pytest.raises(ValueError):
        await run_research(
            "q?",
            config=CONFIG,
            llm=FakeLLM(default_script),
            tool_ctx=make_ctx(db_factory, web),
            use_web=False,
            use_docs=False,
        )


# --- units ---


def test_budget_checks_steps_cost_and_reserve() -> None:
    b = RunBudget(max_steps=2, max_cost_gbp=0.10, timeout_s=60)
    b.check()
    b.charge(0.05)
    with pytest.raises(BudgetExceeded, match="cost"):
        b.check(reserve_gbp=0.06)  # 0.05 + 0.06 > 0.10
    b.check()  # without the reserve there's still room
    b.charge(0.01)
    with pytest.raises(BudgetExceeded, match="step"):
        b.check()
    assert "cost" in b.stop_reason  # the first reason is kept


def test_sources_picked_round_robin_and_unique() -> None:
    def ev(sid: str, q: str) -> Evidence:
        return Evidence(source_id=sid, title=sid, text="t", sub_question=q)

    evidence = [ev("S1", "a"), ev("S2", "a"), ev("S3", "a"), ev("S1", "b"), ev("S4", "b")]
    assert [e.source_id for e in _pick_sources(evidence, 3)] == ["S1", "S4", "S2"]
