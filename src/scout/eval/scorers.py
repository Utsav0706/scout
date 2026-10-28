"""Scoring one report: free checks done in code, plus two model judges.

- tool selection: did the run call the tools the case requires (and none it forbids)?
- claim support: the report's own verifier results (first check, and after revision).
- task success (judge): does the report cover every key point of the answer key, or,
  for unanswerable questions, say the evidence is lacking without inventing an answer?
- citation accuracy (judge): does each sampled claim's verified quote really support it?
  The judge is a different model from the agent's verifier, so this is an independent
  check rather than the verifier marking its own work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel

from scout.agent.prompts import load_prompt
from scout.agent.report import Report, to_markdown
from scout.eval.cases import Case
from scout.llm.client import LLMClient

TASK_PROMPT = load_prompt("judge_task", 1)
CITATION_PROMPT = load_prompt("judge_citation", 1)
JUDGE_PROMPTS = {"judge_task": TASK_PROMPT.tag, "judge_citation": CITATION_PROMPT.tag}
MAX_CITATIONS_CHECKED = 8  # per report: enough for a rate, cheap enough for 50 reports


# --- free checks ---


def tools_called(report: Report) -> list[str]:
    """Each tool the run tried to call, once, in first-use order."""
    return list(dict.fromkeys(s.name for s in report.steps if s.kind == "tool"))


@dataclass
class ToolCheck:
    ok: bool
    missing: list[str]
    forbidden: list[str]


def check_tools(case: Case, called: list[str]) -> ToolCheck:
    missing = [t for t in case.must_call if t not in called]
    forbidden = [t for t in case.must_not_call if t in called]
    return ToolCheck(ok=not missing and not forbidden, missing=missing, forbidden=forbidden)


@dataclass
class ClaimCounts:
    total: int  # claims in the final report
    supported: int  # of those, marked supported
    first_checked: int  # claims the verifier checked first time round
    first_supported: int


def count_claims(report: Report) -> ClaimCounts:
    claims = [c for s in report.sections for c in s.claims]
    v = report.verification
    return ClaimCounts(
        total=len(claims),
        supported=sum(c.verdict == "supported" for c in claims),
        first_checked=(v.checked - v.unverified) if v else 0,
        first_supported=v.supported if v else 0,
    )


# --- task success (judge) ---


class KeyPointGrade(BaseModel):
    point: int
    covered: bool


class TaskGrade(BaseModel):
    key_points: list[KeyPointGrade]
    contradicts_key: bool
    states_insufficient_evidence: bool
    gives_specific_answer: bool
    reason: str


@dataclass
class TaskResult:
    passed: bool
    coverage: float | None  # share of key points covered (answerable cases only)
    reason: str
    cost_gbp: float


def task_passed(case: Case, grade: TaskGrade) -> tuple[bool, float | None]:
    """The pass rule lives in code, not in the judge prompt, so it is the same for every
    case and can be changed without re-wording the prompt."""
    if not case.answerable:
        return grade.states_insufficient_evidence and not grade.gives_specific_answer, None
    covered = {g.point for g in grade.key_points if g.covered}
    coverage = len(covered & set(range(1, len(case.expected) + 1))) / len(case.expected)
    return coverage == 1.0 and not grade.contradicts_key, coverage


def _task_message(case: Case, report: Report) -> str:
    if case.answerable:
        key = "\n".join(f"{i}. {p}" for i, p in enumerate(case.expected, start=1))
    else:
        key = "UNANSWERABLE: the evidence needed does not exist or is not available."
    return (
        f"<question>\n{case.input}\n</question>\n\n<answer_key>\n{key}\n</answer_key>\n\n"
        f"<report>\n{to_markdown(report)}\n</report>"
    )


async def judge_task(llm: LLMClient, case: Case, report: Report) -> TaskResult:
    result = await llm.complete(
        "judge",
        system=TASK_PROMPT.text,
        messages=[{"role": "user", "content": _task_message(case, report)}],
        output_format=TaskGrade,
    )
    grade = result.parsed
    assert isinstance(grade, TaskGrade)
    passed, coverage = task_passed(case, grade)
    return TaskResult(passed, coverage, grade.reason, result.cost_gbp)


# --- citation accuracy (judge) ---


class CitationGrade(BaseModel):
    claim: int
    supports: bool
    reason: str


class CitationGrades(BaseModel):
    claims: list[CitationGrade]


@dataclass
class CitationResult:
    checked: int
    accurate: int
    failures: list[str]  # the claims judged not supported, for reading the results
    cost_gbp: float


async def judge_citations(llm: LLMClient, case: Case, report: Report) -> CitationResult:
    """Check up to MAX_CITATIONS_CHECKED claims that carry a verified quote. A claim
    citing a source id the report doesn't list fails without a model call."""
    source_ids = {s.id for s in report.sources}
    claims = [c for s in report.sections for c in s.claims if c.quote]
    # Same sample every run for the same case, so reruns compare like with like.
    sample = random.Random(case.id).sample(claims, min(len(claims), MAX_CITATIONS_CHECKED))
    dangling = [c for c in sample if not set(c.source_ids) <= source_ids]
    to_judge = [c for c in sample if c not in dangling]
    failures = [f"{c.text} (cites a source not in the report)" for c in dangling]
    if not to_judge:
        return CitationResult(len(sample), 0, failures, 0.0)

    body = "\n\n".join(
        f"{i}. Claim: {c.text}\n   Passage: {c.quote}" for i, c in enumerate(to_judge, start=1)
    )
    result = await llm.complete(
        "judge",
        system=CITATION_PROMPT.text,
        messages=[{"role": "user", "content": f"<claims>\n{body}\n</claims>"}],
        output_format=CitationGrades,
    )
    grades = result.parsed
    assert isinstance(grades, CitationGrades)
    ok = {g.claim for g in grades.claims if g.supports}
    failures += [c.text for i, c in enumerate(to_judge, start=1) if i not in ok]
    accurate = sum(1 for i in range(1, len(to_judge) + 1) if i in ok)
    return CitationResult(len(sample), accurate, failures, result.cost_gbp)
