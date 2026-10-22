"""The reviser node: act on the verifier's verdicts.

Two jobs, in this order:
1. Retry research, at most once. If a sub-question ended with no sources read, or every
   claim drawn from its sources failed verification, send it back to the researchers
   with a hint to search differently. The new notes then go through writer -> verifier
   -> reviser again; the second time round there is no retry.
2. Revise. If every claim passed: no model call, the report passes straight through.
   Otherwise one model call rewrites or deletes each failed claim and rewrites the
   summary from the claims that passed. Every rewrite is verified again (same rules as
   the verifier) and deleted if it still fails, so the reviser can't add unchecked text.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Send

from scout.agent.budget import BudgetExceeded
from scout.agent.calls import llm_call
from scout.agent.report import Claim, Section, Verification
from scout.agent.state import AgentDeps, ResearcherInput, ResearchState, ReviseOut
from scout.agent.verifier import check_claims
from scout.llm.client import LLMError

FAILED = ("unsupported", "contradicted")

RETRY_HINT = (
    "A first search for this found no usable sources, or the sources it found did not "
    "support what was written from them. Try different search terms or different kinds "
    "of sources, and read at least one page."
)


def gap_questions(state: ResearchState, limit: int) -> list[str]:
    """Sub-questions worth a second research round, most clearly failed first."""
    evidence = state.get("evidence", [])
    researched = {ev.sub_question for ev in evidence}
    gaps = [q for q in state.get("sub_questions", []) if q not in researched]

    # Sub-questions whose every claim failed verification.
    question_of = {}
    for ev in evidence:
        question_of.setdefault(ev.source_id, ev.sub_question)
    verdicts: dict[str, list[str]] = {}
    for sec in state.get("sections") or []:
        for claim in sec.claims:
            for q in {question_of[s] for s in claim.source_ids if s in question_of}:
                verdicts.setdefault(q, []).append(claim.verdict)
    gaps += [q for q, vs in verdicts.items() if q not in gaps and all(v in FAILED for v in vs)]
    return gaps[:limit]


def _revision_request(state: ResearchState, claims: list[Claim], failed: list[int]) -> str:
    notes_by_source: dict[str, list[str]] = {}
    for n in state.get("notes", []):
        notes_by_source.setdefault(n.source_id, []).append(n.text)
    passed = [c.text for c in claims if c.verdict not in FAILED]
    parts = ["<passed_claims>", *(f"- {t}" for t in passed), "</passed_claims>", ""]
    for i in failed:
        c = claims[i]
        notes = [f"  [{s}] {t}" for s in c.source_ids for t in notes_by_source.get(s, [])]
        parts += [
            f'<failed_claim id="{i}" verdict="{c.verdict}">',
            f"text: {c.text}",
            f"checker quote: {c.quote or '(none)'}",
            "notes from its sources:",
            *(notes or ["  (none)"]),
            "</failed_claim>",
        ]
    return "\n".join(parts)


async def reviser(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    cfg = deps.config.agent

    # 1. One retry round for sub-questions the first round couldn't answer.
    if state.get("research_round", 1) == 1 and cfg.max_retry_questions:
        gaps = gap_questions(state, cfg.max_retry_questions)
        if gaps and deps.budget.has_room(cfg.retry_min_budget_gbp):
            return {"retry_questions": gaps, "research_round": 2}

    # 2. Revise failed claims.
    sections: list[Section] = state.get("sections") or []
    claims = [c for sec in sections for c in sec.claims]
    failed = [i for i, c in enumerate(claims) if c.verdict in FAILED]
    if not failed:
        return {"retry_questions": []}  # everything passed: no model call

    summary = state.get("summary", "")
    rewrites: dict[int, Claim] = {}
    problems: list[str] = []
    try:
        res = await llm_call(
            deps,
            "reviser",
            system=deps.prompts["reviser"].render(question=state["question"]),
            output_format=ReviseOut,
            messages=[{"role": "user", "content": _revision_request(state, claims, failed)}],
        )
        out: ReviseOut = res.parsed  # type: ignore[assignment]
        summary = out.summary.strip() or summary
        for rev in out.revisions:
            if rev.claim_id in failed and rev.action == "rewrite" and rev.text.strip():
                rewrites[rev.claim_id] = claims[rev.claim_id].model_copy(
                    update={"text": rev.text.strip(), "verdict": "unverified", "quote": None}
                )
    except (BudgetExceeded, LLMError) as exc:
        # Can't revise: fall back to the safe choice, removing every failed claim.
        problems.append(f"Failed claims were removed without rewriting ({exc}).")

    # Re-verify rewrites with the same rules; only "supported" rewrites survive.
    ids = list(rewrites)
    rechecked = dict(
        zip(
            ids,
            await check_claims(deps, [rewrites[i] for i in ids], state.get("evidence", [])),
            strict=True,
        )
    )
    kept = {i: c for i, c in rechecked.items() if c.verdict == "supported"}

    it = iter(range(len(claims)))
    new_sections = []
    for sec in sections:
        new_claims = []
        for claim in sec.claims:
            i = next(it)
            if i not in failed:
                new_claims.append(claim)
            elif i in kept:
                new_claims.append(kept[i])
        if new_claims:
            new_sections.append(Section(heading=sec.heading, claims=new_claims))

    first_check: Verification | None = state.get("verification")
    verification = (
        first_check
        or Verification(
            checked=len(claims), supported=0, unsupported=0, contradicted=0, unverified=0
        )
    ).model_copy(update={"rewritten": len(kept), "removed": len(failed) - len(kept)})
    return {
        "sections": new_sections,
        "summary": summary,
        "verification": verification,
        "retry_questions": [],
        "problems": problems,
    }


def after_reviser(state: ResearchState) -> list[Send] | str:
    """Route: a second research round for the gaps, or on to the finaliser."""
    questions = state.get("retry_questions") or []
    if not questions:
        return "finaliser"
    return [
        Send(
            "researcher",
            ResearcherInput(
                question=state["question"],
                sub_question=q,
                search_plan=state.get("search_plan", ""),
                hint=RETRY_HINT,
            ),
        )
        for q in questions
    ]
