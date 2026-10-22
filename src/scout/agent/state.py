"""Graph state and the per-run dependencies every node can reach.

State (ResearchState) is data that flows through the graph and ends up in the report.
Dependencies (AgentDeps) are live objects - the LLM client, the MCP tool session, the
shared budget - passed as LangGraph's run context, so they never get copied between
parallel branches or serialised with the state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

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
    # Accumulate across the (optional) second research round instead of being replaced.
    notes: Annotated[list[NoteItem], operator.add]
    read_source_ids: Annotated[list[str], operator.add]  # sources the reader has read
    research_round: int  # 1, or 2 after the reviser sends gaps back to research
    retry_questions: list[str]  # set by the reviser when it wants another round
    # Written by the writer (source ids already cleaned), then given verdicts by the verifier.
    summary: str
    sections: list[Any]  # list[report.Section]
    writer_limitations: list[str]
    notes_at_write: int  # len(notes) when the writer last wrote
    verification: Any  # report.Verification | None
    problems: Annotated[list[str], operator.add]  # non-fatal issues for "limitations"
    report: Any  # report.Report, set by the finaliser


class ResearcherInput(TypedDict):
    """What each parallel researcher receives (via Send), instead of the whole state."""

    question: str
    sub_question: str
    search_plan: str
    hint: str  # extra instruction for a retry round; "" the first time


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


class ClaimVerdict(BaseModel):
    claim_id: int
    verdict: Literal["supported", "unsupported", "contradicted"]
    quotes: list[str] = Field(
        description="1-3 word-for-word passages from the source, one per place the "
        "claim's facts come from"
    )
    reason: str


class VerifyOut(BaseModel):
    verdicts: list[ClaimVerdict]


class ClaimRevision(BaseModel):
    claim_id: int
    action: Literal["rewrite", "delete"]
    text: str = Field(default="", description="The rewritten claim (empty when deleting)")


class ReviseOut(BaseModel):
    revisions: list[ClaimRevision]
    summary: str
