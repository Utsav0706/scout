"""The report Scout returns: JSON (this model) and a markdown rendering of it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

# M4 marks every claim "unverified"; the M5 verifier fills in the other verdicts.
Verdict = Literal["unverified", "supported", "unsupported", "contradicted"]
# complete: finished normally. partial: a limit (steps/cost/time) stopped it early.
# no_evidence: no usable sources found. failed: an error stopped it before any report.
Status = Literal["complete", "partial", "no_evidence", "failed"]


class Claim(BaseModel):
    text: str
    source_ids: list[str]
    verdict: Verdict = "unverified"
    # Verbatim passage from the source that the verdict rests on (checked in code to
    # really appear in the source). Lets a UI show the evidence behind a citation.
    quote: str | None = None


class Verification(BaseModel):
    """How the draft's claims fared when checked against their sources."""

    checked: int
    supported: int
    unsupported: int
    contradicted: int
    unverified: int  # not checked, e.g. a limit was hit first
    rewritten: int = 0  # set by the reviser (M5b)
    removed: int = 0

    @computed_field  # included in the JSON report
    @property
    def support_rate(self) -> float | None:
        """Share of checked claims the sources supported on the first check:
        how faithful the writer was. The headline quality number."""
        done = self.checked - self.unverified
        return round(self.supported / done, 3) if done else None


class Section(BaseModel):
    heading: str
    claims: list[Claim]


class ReportSource(BaseModel):
    id: str
    title: str
    url_or_doc: str


class StepRecord(BaseModel):
    """One model or tool call, for cost/latency breakdowns and (later) Dyno's adapter."""

    node: str
    kind: Literal["llm", "tool"]
    name: str
    ok: bool
    ms: int
    cost_gbp: float = 0.0
    detail: str | None = None


class Report(BaseModel):
    question: str
    status: Status
    summary: str
    sections: list[Section] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    verification: Verification | None = None
    cost_gbp: float
    latency_s: float
    steps: list[StepRecord] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(default_factory=dict)


# Supported claims are shown plain; anything else is flagged so a reader can't miss it.
_FLAGS = {
    "unsupported": "**[not supported by source]** ",
    "contradicted": "**[contradicted by source]** ",
    "unverified": "**[not checked]** ",
}


def to_markdown(report: Report) -> str:
    lines = [f"# {report.question}", ""]
    if report.status != "complete":
        lines += [f"> **Status: {report.status}.** {report.stop_reason or ''}".rstrip(), ""]
    lines += [report.summary, ""]
    for section in report.sections:
        lines.append(f"## {section.heading}")
        for claim in section.claims:
            refs = "".join(f"[{sid}]" for sid in claim.source_ids)
            flag = _FLAGS.get(claim.verdict, "")
            lines.append(f"- {flag}{claim.text} {refs}".rstrip())
        lines.append("")
    if report.limitations:
        lines.append("## Limitations")
        lines += [f"- {item}" for item in report.limitations]
        lines.append("")
    if report.sources:
        lines.append("## Sources")
        lines += [f"- **[{s.id}]** {s.title} — {s.url_or_doc}" for s in report.sources]
        lines.append("")
    if (v := report.verification) and v.support_rate is not None:
        lines.append(
            f"_Verification: {v.supported}/{v.checked - v.unverified} claims supported by their "
            f"sources ({v.support_rate:.0%}); {v.unsupported} unsupported, "
            f"{v.contradicted} contradicted, {v.unverified} not checked._"
        )
    llm_calls = sum(1 for s in report.steps if s.kind == "llm")
    tool_calls = sum(1 for s in report.steps if s.kind == "tool")
    lines.append(
        f"_Cost £{report.cost_gbp:.4f} · {report.latency_s:.1f}s · "
        f"{llm_calls} model calls · {tool_calls} tool calls_"
    )
    return "\n".join(lines)
