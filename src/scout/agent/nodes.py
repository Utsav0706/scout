"""The graph's nodes: planner -> researcher (one per sub-question, in parallel) ->
reader -> writer -> verifier -> finaliser.

Model and tool calls go through calls.llm_call / calls.tool_call (budget + step
accounting). The verifier lives in its own module (verifier.py). When a limit is hit, nodes
stop doing new work but the graph still runs to the finaliser, so a partial report is
always returned instead of an exception.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Send

from scout.agent.budget import BudgetExceeded
from scout.agent.calls import llm_call, tool_call
from scout.agent.report import Claim, Report, ReportSource, Section, StepRecord
from scout.agent.state import (
    AgentDeps,
    DraftOut,
    Evidence,
    NoteItem,
    PlanOut,
    ReaderOut,
    ResearcherInput,
    ResearchState,
)
from scout.llm.client import LLMError, ToolCall
from scout.safety.injection import neutralise_markup

log = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 12_000  # what the researcher model sees of one tool result
_READ_CONCURRENCY = 4


# --- planner ---


async def planner(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    cfg = deps.config.agent
    available = [
        name
        for name, on in (
            ("the public web", state["use_web"]),
            ("the user's uploaded documents", state["use_docs"]),
        )
        if on
    ]
    system = deps.prompts["planner"].render(
        max_sub_questions=cfg.max_sub_questions, available_sources=" and ".join(available)
    )
    try:
        res = await llm_call(
            deps,
            "planner",
            system=system,
            output_format=PlanOut,
            messages=[{"role": "user", "content": state["question"]}],
        )
    except (BudgetExceeded, LLMError) as exc:
        return {"sub_questions": [], "problems": [f"Planning failed: {exc}"]}
    plan: PlanOut = res.parsed  # type: ignore[assignment]
    subs = [q.strip() for q in plan.sub_questions if q.strip()][: cfg.max_sub_questions]
    return {"sub_questions": subs or [state["question"]], "search_plan": plan.search_plan}


def fan_out(state: ResearchState) -> list[Send] | str:
    """One researcher per sub-question, run in parallel. No plan -> straight to the end."""
    if not state.get("sub_questions"):
        return "finaliser"
    return [
        Send(
            "researcher",
            ResearcherInput(
                question=state["question"],
                sub_question=q,
                search_plan=state.get("search_plan", ""),
                hint="",
            ),
        )
        for q in state["sub_questions"]
    ]


# --- researcher ---


def _evidence_from(name: str, structured: Any, sub_question: str) -> list[Evidence]:
    if name == "fetch_url" and isinstance(structured, dict):
        return [
            Evidence(
                source_id=structured["source_id"],
                title=structured["title"],
                text=structured["text"],
                sub_question=sub_question,
            )
        ]
    if name == "rag_search" and isinstance(structured, dict):
        return [
            Evidence(
                source_id=r["source_id"],
                title=r["title"],
                text=r["text"],
                sub_question=sub_question,
            )
            for r in structured.get("result", [])
        ]
    return []  # web_search snippets are previews, not evidence


async def researcher(state: ResearcherInput, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    cfg = deps.config.agent
    system = deps.prompts["researcher"].render(
        question=state["question"],
        sub_question=state["sub_question"],
        search_plan=state["search_plan"] or "Use reliable sources.",
        max_tool_turns=cfg.max_tool_turns,
    )
    ask = state["sub_question"]
    if state.get("hint"):
        ask += "\n\n" + state["hint"]
    messages: list[dict[str, Any]] = [{"role": "user", "content": ask}]
    evidence: list[Evidence] = []
    problems: list[str] = []

    for _ in range(cfg.max_tool_turns):
        try:
            res = await llm_call(
                deps,
                "researcher",
                reserve_gbp=cfg.finish_reserve_gbp,
                system=system,
                messages=messages,
                tools=deps.research_tools,
            )
        except BudgetExceeded:
            break
        except LLMError as exc:
            problems.append(f"Research on '{state['sub_question']}' stopped early: {exc}")
            break
        if not res.tool_calls:
            break  # the model decided it has read enough

        # Run this turn's tool calls in parallel, then return all results in ONE user
        # message (splitting them teaches the model to stop calling tools in parallel).
        outcomes = await asyncio.gather(*(_allowed_tool_call(deps, c) for c in res.tool_calls))
        results = []
        for call, (ok, structured, text) in zip(res.tool_calls, outcomes, strict=True):
            results.append(_tool_result(call, ok, text))
            if ok:
                evidence += _evidence_from(call.name, structured, state["sub_question"])
        messages.append({"role": "assistant", "content": res.content})
        messages.append({"role": "user", "content": results})

    # Sub-questions left without sources are reported by the finaliser, after any retry.
    return {"evidence": evidence, "problems": problems}


async def _allowed_tool_call(deps: AgentDeps, call: ToolCall) -> tuple[bool, Any, str]:
    """Only tools the researcher was offered can run. The MCP server would execute any
    tool it has (save_note included) if a fooled model named it; refuse that here."""
    if call.name not in {t["name"] for t in deps.research_tools}:
        deps.steps.append(
            StepRecord(
                node="researcher",
                kind="tool",
                name=call.name,
                ok=False,
                ms=0,
                detail="refused: not a research tool",
            )
        )
        log.warning("model asked for a tool it wasn't given", extra={"tool": call.name[:64]})
        return False, None, f"Tool {call.name!r} is not available. Use the tools listed."
    return await tool_call(deps, "researcher", call.name, call.input)


def _tool_result(call: ToolCall, ok: bool, text: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": text[:_MAX_TOOL_RESULT_CHARS],
    }
    if not ok:
        block["is_error"] = True
    return block


# --- reader ---


def _pick_sources(evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Unique sources, taken round-robin across sub-questions so that one
    sub-question's ten passages can't crowd out the others."""
    by_question: dict[str, list[Evidence]] = {}
    seen: set[str] = set()
    for ev in evidence:
        if ev.source_id not in seen:
            seen.add(ev.source_id)
            by_question.setdefault(ev.sub_question, []).append(ev)
    picked: list[Evidence] = []
    queues = list(by_question.values())
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


async def reader(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    cfg = deps.config.agent
    # Only sources not read in an earlier round; the retry round reads fewer.
    already = set(state.get("read_source_ids", []))
    unread = [ev for ev in state.get("evidence", []) if ev.source_id not in already]
    retry = state.get("research_round", 1) > 1
    if retry:  # read what the retry found, not leftovers from the first round
        unread = [ev for ev in unread if ev.sub_question in state.get("retry_questions", [])]
    limit = cfg.max_retry_sources if retry else cfg.max_sources_to_read
    sources = _pick_sources(unread, limit)
    system = deps.prompts["reader"].render(
        question=state["question"], max_notes=cfg.max_notes_per_source
    )
    gate = asyncio.Semaphore(_READ_CONCURRENCY)

    async def read_one(ev: Evidence) -> list[NoteItem]:
        async with gate:
            # Source text goes inside tags so the model can tell data from instructions.
            content = (
                f'<source id="{ev.source_id}" title="{ev.title}">\n'
                f"{ev.text[: cfg.max_source_chars]}\n</source>"
            )
            try:
                res = await llm_call(
                    deps,
                    "reader",
                    reserve_gbp=cfg.finish_reserve_gbp,
                    system=system,
                    output_format=ReaderOut,
                    messages=[{"role": "user", "content": content}],
                )
            except (BudgetExceeded, LLMError):
                return []
            notes = []
            for fact in res.parsed.facts[: cfg.max_notes_per_source]:  # type: ignore[union-attr]
                fact = fact.strip()[:2000]
                if len(fact) < 3:
                    continue
                # Saving through the tool checks the source id and writes the audit row.
                ok, _, _ = await tool_call(
                    deps, "reader", "save_note", {"text": fact, "source_id": ev.source_id}
                )
                if ok:
                    notes.append(NoteItem(source_id=ev.source_id, text=fact))
            return notes

    batches = await asyncio.gather(*(read_one(ev) for ev in sources))
    return {
        "notes": [n for batch in batches for n in batch],
        "read_source_ids": [ev.source_id for ev in sources],
    }


# --- writer ---


def _notes_block(state: ResearchState, deps: AgentDeps) -> str:
    grouped: dict[str, list[str]] = {}
    for note in state.get("notes", []):
        grouped.setdefault(note.source_id, []).append(note.text)
    parts = []
    for sid, facts in grouped.items():
        src = deps.tool_ctx.sources.get(sid)
        title = src.title if src else sid
        parts.append(f"[{sid}] {title}\n" + "\n".join(f"- {f}" for f in facts))
    return "\n\n".join(parts)


async def writer(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    notes = state.get("notes", [])
    if not notes:
        return {}  # nothing to write from: no model call, no cost
    if state.get("sections") and len(notes) == state.get("notes_at_write"):
        return {}  # retry round found nothing new: keep the existing report, don't pay twice
    system = deps.prompts["writer"].render(question=state["question"])
    try:
        res = await llm_call(
            deps,
            "writer",
            system=system,
            output_format=DraftOut,
            messages=[{"role": "user", "content": _notes_block(state, deps)}],
        )
    except BudgetExceeded:
        return {}
    except LLMError as exc:
        return {"problems": [f"Writing the report failed: {exc}"]}
    draft: DraftOut = res.parsed  # type: ignore[assignment]
    sections, dropped = _clean_draft(draft, {n.source_id for n in notes})
    limitations = list(draft.limitations)
    if dropped:
        limitations.append(f"{dropped} claim(s) removed because they cited no real source.")
    return {
        "summary": draft.summary,
        "sections": sections,
        "writer_limitations": limitations,
        "notes_at_write": len(notes),
    }


def _clean_draft(draft: DraftOut, known: set[str]) -> tuple[list[Section], int]:
    """Keep only source ids that came from notes; drop claims left with none.
    Whether a source really supports a claim is the verifier's job."""
    sections, dropped = [], 0
    for sec in draft.sections:
        claims = []
        for c in sec.claims:
            ids = [i for i in dict.fromkeys(c.source_ids) if i in known]
            if ids:
                claims.append(Claim(text=c.text, source_ids=ids))
            else:
                dropped += 1
        if claims:
            sections.append(Section(heading=sec.heading, claims=claims))
    return sections, dropped


# --- finaliser (no model call) ---


def _injection_notes(deps: AgentDeps, limit: int = 5) -> list[str]:
    """One line per source that contained text aimed at the AI (see safety.injection)."""
    notes = []
    for sid, findings in deps.tool_ctx.injection_findings.items():
        src = deps.tool_ctx.sources.get(sid)
        kinds = ", ".join(sorted({f.kind for f in findings}))
        notes.append(
            f"Source {sid} ({src.title if src else 'unknown'}) contained text that looked like "
            f"instructions to an AI ({kinds}); Scout treated it only as data."
        )
    if len(notes) > limit:
        notes = notes[:limit] + [f"...and {len(notes) - limit} more such sources."]
    return notes


async def finaliser(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    budget = deps.budget
    notes = state.get("notes", [])
    limitations = list(state.get("writer_limitations", [])) + list(state.get("problems", []))
    limitations += _injection_notes(deps)
    researched = {ev.sub_question for ev in state.get("evidence", [])}
    limitations += [
        f"No sources were read for: {q}"
        for q in state.get("sub_questions", [])
        if q not in researched
    ]
    sections: list[Section] = state.get("sections") or []

    if sections:
        summary = state.get("summary", "")
    elif notes:
        # Stopped before (or while) writing: return the facts we have, each cited.
        summary = "The report could not be written in full. These are the facts collected so far."
        sections = [
            Section(
                heading="Collected facts",
                claims=[Claim(text=n.text, source_ids=[n.source_id]) for n in notes],
            )
        ]
    else:
        summary = "Not enough evidence was found to answer this question."

    if budget.stop_reason:
        status = "partial"
    elif not notes:
        status = "no_evidence" if state.get("sub_questions") else "failed"
    else:
        status = "complete" if state.get("sections") else "partial"

    cited = {sid for sec in sections for c in sec.claims for sid in c.source_ids}
    sources = [
        ReportSource(id=s.id, title=s.title, url_or_doc=s.url_or_doc)
        for s in deps.tool_ctx.sources.all()
        if s.id in cited
    ]

    # Model-written text may carry markup copied from a page: neutralise it before it can
    # be rendered (an image link would leak data to whoever's server it points at).
    sections = [
        Section(
            heading=neutralise_markup(sec.heading),
            claims=[c.model_copy(update={"text": neutralise_markup(c.text)}) for c in sec.claims],
        )
        for sec in sections
    ]
    report = Report(
        question=state["question"],
        status=status,
        summary=neutralise_markup(summary),
        sections=sections,
        sources=sources,
        limitations=[neutralise_markup(item) for item in limitations],
        stop_reason=budget.stop_reason,
        verification=state.get("verification"),
        cost_gbp=round(budget.cost_gbp, 6),
        latency_s=round(budget.elapsed_s, 2),
        steps=deps.steps,
        prompt_versions={name: p.tag for name, p in deps.prompts.items()},
    )
    return {"report": report}
