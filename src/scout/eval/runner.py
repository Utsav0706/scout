"""Run eval cases through the real agent and score each report.

One fresh user per eval run owns the corpus documents, so runs never see each other's
files. Cases run a few at a time; each finished case is handed to `on_result` straight
away (the CLI appends it to results.jsonl), so a crash or the spend cap loses nothing.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import anyio
from pydantic import BaseModel

from scout.agent.report import Report
from scout.agent.runtime import Services, research
from scout.eval.cases import Case
from scout.eval.corpus import load_corpus
from scout.eval.scorers import (
    check_tools,
    count_claims,
    judge_citations,
    judge_task,
    tools_called,
)
from scout.models import User
from scout.rag.store import ingest_pdf

log = logging.getLogger(__name__)


class CaseResult(BaseModel):
    id: str
    category: str
    tags: list[str]
    outcome: str  # "scored", "skipped" or "error"
    note: str | None = None  # why it was skipped / what failed
    report_status: str | None = None
    # task success (judge)
    passed: bool | None = None
    coverage: float | None = None
    judge_reason: str | None = None
    # tool selection
    tools_called: list[str] = []
    tools_ok: bool | None = None
    tools_missing: list[str] = []
    tools_forbidden: list[str] = []
    # claim support (the agent's own verifier)
    claims: int = 0
    claims_supported: int = 0
    first_checked: int = 0
    first_supported: int = 0
    # citation accuracy (judge)
    citations_checked: int = 0
    citations_accurate: int = 0
    citation_failures: list[str] = []
    # efficiency
    steps: int = 0
    llm_calls: int = 0
    max_steps_ok: bool | None = None
    cost_gbp: float = 0.0  # the research run
    judge_cost_gbp: float = 0.0
    latency_s: float | None = None
    trace_id: str | None = None


def create_eval_user(services: Services) -> uuid.UUID:
    with services.session_factory() as db:
        # "!" is not a valid Argon2 hash, so nobody can log in as this account.
        user = User(email=f"eval-{uuid.uuid4().hex[:12]}@scout.invalid", password_hash="!")
        db.add(user)
        db.commit()
        return user.id


async def ingest_corpus(services: Services, user_id: uuid.UUID, corpus_dir: Path) -> int:
    """Upload every corpus document for this user through the normal pipeline."""

    def work() -> int:
        n = 0
        with services.session_factory() as db:
            for name, data in load_corpus(corpus_dir):
                doc = ingest_pdf(db, user_id, name, data, services.embedder, services.config.rag)
                if doc.status != "ready":
                    raise RuntimeError(f"corpus document {name} failed: {doc.error}")
                n += 1
        return n

    return await anyio.to_thread.run_sync(work)


async def score_case(services: Services, case: Case, report: Report) -> CaseResult:
    llm = services.llm_client()
    called = tools_called(report)
    tools = check_tools(case, called)
    counts = count_claims(report)
    result = CaseResult(
        id=case.id,
        category=case.category,
        tags=case.tags,
        outcome="scored",
        report_status=report.status,
        tools_called=called,
        tools_ok=tools.ok,
        tools_missing=tools.missing,
        tools_forbidden=tools.forbidden,
        claims=counts.total,
        claims_supported=counts.supported,
        first_checked=counts.first_checked,
        first_supported=counts.first_supported,
        steps=len(report.steps),
        llm_calls=sum(1 for s in report.steps if s.kind == "llm"),
        max_steps_ok=None if case.max_steps is None else len(report.steps) <= case.max_steps,
        cost_gbp=report.cost_gbp,
        latency_s=report.latency_s,
        trace_id=report.trace_id,
    )
    # The two judges are independent, so run them together.
    try:
        task, cites = await asyncio.gather(
            judge_task(llm, case, report), judge_citations(llm, case, report)
        )
    except Exception as exc:  # keep the research cost and checks; flag the case
        log.exception("eval judge failed", extra={"case": case.id})
        result.outcome, result.note = "error", f"judge failed: {type(exc).__name__}: {exc}"[:300]
        return result
    result.passed, result.coverage, result.judge_reason = task.passed, task.coverage, task.reason
    result.citations_checked, result.citations_accurate = cites.checked, cites.accurate
    result.citation_failures = cites.failures
    result.judge_cost_gbp = task.cost_gbp + cites.cost_gbp
    return result


async def run_case(
    services: Services, case: Case, user_id: uuid.UUID, use_docs: bool
) -> CaseResult:
    if case.needs_docs and not use_docs:
        return CaseResult(
            id=case.id,
            category=case.category,
            tags=case.tags,
            outcome="skipped",
            note="needs document search (Postgres + pgvector)",
        )
    start = time.perf_counter()
    try:
        report = await asyncio.wait_for(
            research(
                services,
                user_id=user_id,
                job_id=uuid.uuid4(),
                question=case.input,
                use_web=True,  # every case gets every tool: choosing them is being tested
                use_docs=use_docs,
            ),
            timeout=services.config.limits.timeout_s + 60,
        )
        return await score_case(services, case, report)
    except Exception as exc:  # one broken case must not stop the run
        log.exception("eval case failed", extra={"case": case.id})
        return CaseResult(
            id=case.id,
            category=case.category,
            tags=case.tags,
            outcome="error",
            note=f"{type(exc).__name__}: {exc}"[:300],
            latency_s=round(time.perf_counter() - start, 1),
        )


async def run_suite(
    services: Services,
    cases: list[Case],
    *,
    corpus_dir: Path,
    use_docs: bool,
    concurrency: int = 2,
    max_cost_gbp: float = 5.0,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run and score every case. Stops starting new cases once spending (research plus
    judges) reaches max_cost_gbp; cases already running still finish, so the total can
    go over by up to `concurrency` cases."""
    user_id = create_eval_user(services)
    if use_docs and any(c.needs_docs for c in cases):
        n = await ingest_corpus(services, user_id, corpus_dir)
        log.info("eval corpus ingested", extra={"documents": n})

    results: list[CaseResult] = []
    spent = 0.0
    gate = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> None:
        nonlocal spent
        async with gate:
            if spent >= max_cost_gbp:
                r = CaseResult(
                    id=case.id,
                    category=case.category,
                    tags=case.tags,
                    outcome="skipped",
                    note=f"spend cap £{max_cost_gbp:.2f} reached",
                )
            else:
                r = await run_case(services, case, user_id, use_docs)
            spent += r.cost_gbp + r.judge_cost_gbp
            results.append(r)
            if on_result:
                on_result(r)

    await asyncio.gather(*(one(c) for c in cases))
    order = {c.id: i for i, c in enumerate(cases)}
    return sorted(results, key=lambda r: order[r.id])
