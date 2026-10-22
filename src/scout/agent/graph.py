"""Wire the nodes into a LangGraph state graph and run it.

    START -> planner -> researcher x N (parallel) -> reader -> writer -> verifier
                                                          -> finaliser -> END
                 \\------------- no sub-questions ------------------------^

Why a graph library rather than a plain async function: the flow is explicit and
inspectable (nodes + edges), parallel fan-out and the merging of results are handled
by the framework, and M5 adds a conditional loop (reviser -> researcher, at most once)
as one more edge instead of nested control flow.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from mcp import Client

from scout.agent.budget import RunBudget
from scout.agent.nodes import fan_out, finaliser, planner, reader, researcher, writer
from scout.agent.prompts import load_prompt
from scout.agent.report import Report
from scout.agent.state import AgentDeps, ResearchState
from scout.agent.verifier import verifier
from scout.config import AppConfig
from scout.llm.client import LLMClient
from scout.tools.context import ToolContext
from scout.tools.server import build_server

log = logging.getLogger(__name__)

# Prompt versions in use. Older versions stay in prompts/ so past reports (which record
# the version + hash they used) can be traced.
#   verifier v2: stricter "contradicted" rule (v1 flagged a correct claim, 1 of 3 runs).
#   verifier v3: up to 3 quotes per claim, so a claim combining facts from different
#                sentences can be backed without stitching them into one quote.
PROMPT_VERSIONS = {"planner": 1, "researcher": 1, "reader": 1, "writer": 1, "verifier": 3}


def build_graph():  # -> CompiledStateGraph
    g = StateGraph(ResearchState, context_schema=AgentDeps)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("reader", reader)
    g.add_node("writer", writer)
    g.add_node("verifier", verifier)
    g.add_node("finaliser", finaliser)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher", "finaliser"])
    g.add_edge("researcher", "reader")  # waits for every parallel researcher
    g.add_edge("reader", "writer")
    g.add_edge("writer", "verifier")
    g.add_edge("verifier", "finaliser")
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


async def run_research(
    question: str,
    *,
    config: AppConfig,
    llm: LLMClient,
    tool_ctx: ToolContext,
    use_web: bool = True,
    use_docs: bool = True,
) -> Report:
    if not (use_web or use_docs):
        raise ValueError("Enable at least one of use_web / use_docs")
    limits = config.limits
    budget = RunBudget(limits.max_steps, limits.max_cost_gbp, limits.timeout_s)
    prompts = {name: load_prompt(name, v) for name, v in PROMPT_VERSIONS.items()}

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
        final = await build_graph().ainvoke(
            {"question": question, "use_web": use_web, "use_docs": use_docs}, context=deps
        )

    report: Report = final["report"]
    log.info(
        "research finished",
        extra={"status": report.status, "cost_gbp": report.cost_gbp, "latency_s": report.latency_s},
    )
    return report
