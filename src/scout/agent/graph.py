"""Wire the nodes into a LangGraph state graph and run it.

    START -> planner -> researcher x N (parallel) -> reader -> writer -> verifier -> reviser
               |              ^                                                       |
               |              +------ retry round for gaps (at most once) ------------+
               |                                                                      v
               +---------------- no sub-questions ----------------------------> finaliser -> END

Why a graph library rather than a plain async function: the flow is explicit and
inspectable (nodes + edges), parallel fan-out and the merging of results are handled
by the framework, and the retry loop is one conditional edge instead of nested control
flow.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from mcp import Client

from scout.agent.budget import RunBudget
from scout.agent.nodes import fan_out, finaliser, planner, reader, researcher, writer
from scout.agent.prompts import load_prompt
from scout.agent.report import Report
from scout.agent.reviser import after_reviser, reviser
from scout.agent.state import AgentDeps, ResearchState
from scout.agent.verifier import verifier
from scout.config import AppConfig
from scout.llm.client import LLMClient
from scout.telemetry import run_scope, set_content, trace_id_of, tracer
from scout.tools.context import ToolContext
from scout.tools.server import build_server

log = logging.getLogger(__name__)

# Prompt versions in use. Older versions stay in prompts/ so past reports (which record
# the version + hash they used) can be traced.
#   verifier v2: stricter "contradicted" rule (v1 flagged a correct claim, 1 of 3 runs).
#   verifier v3: up to 3 quotes per claim, so a claim combining facts from different
#                sentences can be backed without stitching them into one quote.
#   writer v2:   one fact per claim; compound claims caused both verifier mistakes in M5a.
PROMPT_VERSIONS = {
    "planner": 1,
    "researcher": 1,
    "reader": 1,
    "writer": 2,
    "verifier": 3,
    "reviser": 1,
}


def traced(name: str, node):
    """Run a node inside its own span, so a trace shows the graph's steps with each
    step's model and tool calls nested underneath. Same (state, runtime) signature as
    the node, which LangGraph inspects to know what to pass."""

    async def run(state, runtime):
        with tracer().start_as_current_span(name) as span:
            if name == "researcher":
                span.set_attribute("scout.sub_question", state.get("sub_question", ""))
            return await node(state, runtime)

    run.__name__ = name
    run.__annotations__ = node.__annotations__  # LangGraph reads the Runtime annotation
    return run


def build_graph():  # -> CompiledStateGraph
    g = StateGraph(ResearchState, context_schema=AgentDeps)
    g.add_node("planner", traced("planner", planner))
    g.add_node("researcher", traced("researcher", researcher))
    g.add_node("reader", traced("reader", reader))
    g.add_node("writer", traced("writer", writer))
    g.add_node("verifier", traced("verifier", verifier))
    g.add_node("reviser", traced("reviser", reviser))
    g.add_node("finaliser", traced("finaliser", finaliser))

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher", "finaliser"])
    g.add_edge("researcher", "reader")  # waits for every parallel researcher
    g.add_edge("reader", "writer")
    g.add_edge("writer", "verifier")
    g.add_edge("verifier", "reviser")
    g.add_conditional_edges("reviser", after_reviser, ["researcher", "finaliser"])
    g.add_edge("finaliser", END)
    return g.compile()


def _research_tools(listed, use_web: bool, use_docs: bool) -> list[dict]:
    wanted = set()
    if use_web:
        wanted |= {"web_search", "fetch_url"}
    if use_docs:
        wanted.add("rag_search")
    # save_note isn't offered: the reader saves notes, researchers only gather sources.
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in listed
        if t.name in wanted
    ]


# Progress callback: (node, message). Awaited, so it can write to the database.
OnEvent = Callable[[str, str], Awaitable[None]]


def describe(node: str, update: dict[str, Any] | None) -> str | None:
    """One human-readable progress line for a finished node (None = nothing to say)."""
    u = update or {}
    if node == "planner":
        subs = u.get("sub_questions") or []
        return f"Planned {len(subs)} sub-questions" if subs else "Planning failed"
    if node == "researcher":
        evidence = u.get("evidence") or []
        if not evidence:
            return "A researcher found no readable sources"
        return f"Read {len(evidence)} source(s) for: {evidence[0].sub_question}"
    if node == "reader":
        notes, read = u.get("notes") or [], u.get("read_source_ids") or []
        return f"Took {len(notes)} notes from {len(read)} source(s)" if read else None
    if node == "writer":
        sections = u.get("sections")
        if sections is None:
            return None
        return f"Drafted {sum(len(s.claims) for s in sections)} claims"
    if node == "verifier":
        v = u.get("verification")
        return f"Checked claims: {v.supported}/{v.checked} supported" if v else None
    if node == "reviser":
        if u.get("retry_questions"):
            return f"Retrying research for {len(u['retry_questions'])} sub-question(s)"
        v = u.get("verification")
        if v is None:
            return "All claims passed the check"
        return f"Revised: {v.rewritten} rewritten, {v.removed} removed"
    if node == "finaliser":
        return f"Report ready ({u['report'].status})" if u.get("report") else None
    return None


async def run_research(
    question: str,
    *,
    config: AppConfig,
    llm: LLMClient,
    tool_ctx: ToolContext,
    use_web: bool = True,
    use_docs: bool = True,
    on_event: OnEvent | None = None,
) -> Report:
    if not (use_web or use_docs):
        raise ValueError("Enable at least one of use_web / use_docs")
    limits = config.limits
    budget = RunBudget(limits.max_steps, limits.max_cost_gbp, limits.timeout_s)
    prompts = {name: load_prompt(name, v) for name, v in PROMPT_VERSIONS.items()}
    report: Report | None = None

    # One trace per run; every span in it carries the user and job ids (run_scope).
    with (
        run_scope(tool_ctx.user_id, tool_ctx.job_id),
        tracer().start_as_current_span("research") as root,
    ):
        root.set_attribute("langfuse.trace.name", "research")
        set_content(root, config.telemetry, input=question)
        async with Client(build_server(tool_ctx)) as mcp:
            listed = (await mcp.list_tools()).tools
            deps = AgentDeps(
                config=config,
                llm=llm,
                mcp=mcp,
                tool_ctx=tool_ctx,
                budget=budget,
                prompts=prompts,
                research_tools=_research_tools(listed, use_web, use_docs),
            )
            # stream_mode="updates" yields {node: update} as each node finishes: the same run
            # as ainvoke(), plus a progress line per step for the job's live event stream.
            async for chunk in build_graph().astream(
                {"question": question, "use_web": use_web, "use_docs": use_docs},
                context=deps,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if node == "finaliser" and update:
                        report = update["report"]
                    if on_event and (message := describe(node, update)):
                        await on_event(node, message)
        if report is not None:
            set_content(
                root, config.telemetry, output={"status": report.status, "summary": report.summary}
            )
            root.set_attribute("scout.status", report.status)
            report.trace_id = trace_id_of(root)

    if report is None:  # the graph always ends in the finaliser; this is a bug guard
        raise RuntimeError("research graph finished without a report")
    log.info(
        "research finished",
        extra={"status": report.status, "cost_gbp": report.cost_gbp, "latency_s": report.latency_s},
    )
    return report
