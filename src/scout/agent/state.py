"""Graph state and the per-run dependencies every node can reach.

State (ResearchState) is data that flows through the graph and ends up in the report.
Dependencies (AgentDeps) are live objects - the LLM client, the MCP tool session, the
shared budget - passed as LangGraph's run context, so they never get copied between
parallel branches or serialised with the state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from mcp import Client
from pydantic import BaseModel, Field

from scout.agent.budget import RunBudget
from scout.agent.prompts import Prompt
from scout.agent.report import StepRecord
from scout.config import AppConfig
from scout.llm.client import LLMClient
from scout.tools.context import ToolContext


class Evidence(BaseModel):
    """Text a researcher actually read: a fetched page or a document passage."""

    source_id: str
    title: str
    text: str
    sub_question: str


class NoteItem(BaseModel):
    source_id: str
    text: str


class ResearchState(TypedDict, total=False):
    question: str
    use_web: bool
    use_docs: bool
    search_plan: str
    sub_questions: list[str]
    # operator.add: parallel researchers each return a list, LangGraph concatenates them.
    evidence: Annotated[list[Evidence], operator.add]
    notes: list[NoteItem]
    draft: Any  # writer.DraftOut | None
    problems: Annotated[list[str], operator.add]  # non-fatal issues for "limitations"
    report: Any  # report.Report, set by the finaliser


class ResearcherInput(TypedDict):
    """What each parallel researcher receives (via Send), instead of the whole state."""

    question: str
    sub_question: str
    search_plan: str


@dataclass
class AgentDeps:
    config: AppConfig
    llm: LLMClient
    mcp: Client
    tool_ctx: ToolContext
    budget: RunBudget
    prompts: dict[str, Prompt]
    research_tools: list[dict[str, Any]]  # Anthropic tool definitions, from MCP list_tools
    steps: list[StepRecord] = field(default_factory=list)


# --- structured outputs the nodes ask the models for ---


class PlanOut(BaseModel):
    sub_questions: list[str] = Field(description="2-5 focused sub-questions")
    search_plan: str = Field(description="Which kinds of sources to prefer, 1-2 sentences")


class ReaderOut(BaseModel):
    facts: list[str] = Field(description="Facts from the source relevant to the question")


class DraftClaim(BaseModel):
    text: str
    source_ids: list[str]


class DraftSection(BaseModel):
    heading: str
    claims: list[DraftClaim]


class DraftOut(BaseModel):
    summary: str
    sections: list[DraftSection]
    limitations: list[str]
