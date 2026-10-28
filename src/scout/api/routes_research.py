"""Start a research job and check on it."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import anyio
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scout.api.deps import CurrentUser, DbSession
from scout.jobs.queue import QueueUnavailableError
from scout.jobs.runner import TERMINAL
from scout.models import JobEvent, JobTrajectory, ResearchJob
from scout.safety.injection import scan
from scout.schemas import JobCreatedOut, JobEventOut, JobOut, ResearchIn, TrajectoryOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


def _aware(dt: datetime | None) -> datetime | None:
    # SQLite hands back naive datetimes even for timezone-aware columns; they're UTC.
    return dt.replace(tzinfo=UTC) if dt is not None and dt.tzinfo is None else dt


@router.post("", response_model=JobCreatedOut, status_code=status.HTTP_202_ACCEPTED)
def start_research(
    body: ResearchIn, request: Request, response: Response, user: CurrentUser, db: DbSession
) -> JobCreatedOut:
    """Queue a research job. Returns at once with the job id; poll the status URL."""
    limits = request.app.state.config.jobs

    active = db.scalar(
        select(func.count())
        .select_from(ResearchJob)
        .where(ResearchJob.user_id == user.id, ResearchJob.status.in_(("queued", "running")))
    )
    if active >= limits.max_running_per_user:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"You already have {active} research jobs running; wait for one to finish.",
            headers={"Retry-After": "30"},
        )

    hour_ago = datetime.now(UTC) - timedelta(hours=1)
    recent = db.scalars(
        select(ResearchJob.created_at)
        .where(ResearchJob.user_id == user.id, ResearchJob.created_at >= hour_ago)
        .order_by(ResearchJob.created_at)
    ).all()
    if len(recent) >= limits.max_per_hour_per_user:
        # The window frees up when the oldest job in it turns an hour old.
        wait = (_aware(recent[0]) + timedelta(hours=1) - datetime.now(UTC)).total_seconds()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Limit of {limits.max_per_hour_per_user} research jobs per hour reached.",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )

    if findings := scan(body.question):
        # Logged, not refused: it's the user's own agent, and its tools are read-only.
        log.warning(
            "question looks like a prompt-injection attempt",
            extra={"user_id": str(user.id), "kinds": sorted({f.kind for f in findings})},
        )
    job = ResearchJob(
        user_id=user.id, question=body.question, use_web=body.use_web, use_docs=body.use_docs
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        request.app.state.job_queue.enqueue(job.id)
    except QueueUnavailableError:
        log.exception("could not queue research job", extra={"job_id": str(job.id)})
        job.status, job.error = "failed", "The job queue is unavailable."
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The job queue is unavailable; try again soon."
        ) from None

    log.info("research job queued", extra={"job_id": str(job.id), "user_id": str(user.id)})
    url = f"/research/{job.id}"
    response.headers["Location"] = url
    db.refresh(job)  # inline/sync queues may already have moved it on
    return JobCreatedOut(job_id=job.id, status=job.status, status_url=url)


def _own_job(db: Session, job_id: uuid.UUID, user_id: uuid.UUID) -> ResearchJob:
    job = db.get(ResearchJob, job_id)
    # Someone else's job is "not found", not "forbidden": don't confirm it exists.
    if job is None or job.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Research job not found")
    return job


def _fail_if_stale(db: Session, job: ResearchJob, stale_after_s: int) -> None:
    """A job "running" far longer than any run can take has lost its worker (crash,
    restart): mark it failed instead of leaving it "running" forever."""
    started = _aware(job.started_at)
    too_old = started and datetime.now(UTC) - started > timedelta(seconds=stale_after_s)
    if job.status == "running" and too_old:
        job.status, job.error = "failed", "The worker running this job stopped responding."
        job.finished_at = datetime.now(UTC)
        db.add(JobEvent(job_id=job.id, node="job", message=job.error))
        db.commit()


@router.get("/{job_id}", response_model=JobOut)
def get_research(job_id: uuid.UUID, request: Request, user: CurrentUser, db: DbSession) -> JobOut:
    job = _own_job(db, job_id, user.id)
    _fail_if_stale(db, job, request.app.state.config.jobs.stale_after_s)
    events = db.scalars(
        select(JobEvent).where(JobEvent.job_id == job.id).order_by(JobEvent.id)
    ).all()
    return JobOut(
        id=job.id,
        question=job.question,
        status=job.status,
        created_at=_aware(job.created_at),
        started_at=_aware(job.started_at),
        finished_at=_aware(job.finished_at),
        cost_gbp=job.cost_gbp,
        error=job.error,
        events=[JobEventOut.model_validate(e) for e in events],
        report=job.report if job.status in TERMINAL else None,
    )


# --- live progress (server-sent events) ---

POLL_S = 1.0  # how often the stream checks for new events
HEARTBEAT_S = 15.0  # comment line on quiet streams, so proxies don't close them


def _sse(event: str, data: dict, event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/{job_id}/stream")
async def stream_research(
    job_id: uuid.UUID, request: Request, user: CurrentUser, db: DbSession
) -> StreamingResponse:
    """Progress as server-sent events: `progress` per step, then one `done` with the final
    status (fetch the report from GET /research/{id}).

    Polls the job_events table rather than using Redis pub/sub: the API and the worker
    already share the database, and a 1 s delay is fine for jobs that take 30-120 s.
    Each event carries an id, so a reconnecting client (EventSource does this itself)
    sends Last-Event-ID and resumes without gaps or repeats.
    """
    _own_job(db, job_id, user.id)
    cfg = request.app.state.config
    factory = request.app.state.session_factory
    try:
        last_id = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        last_id = 0

    def poll(after: int) -> tuple[list[tuple[int, str, str, datetime]], str]:
        with factory() as s:
            job = s.get(ResearchJob, job_id)
            _fail_if_stale(s, job, cfg.jobs.stale_after_s)
            rows = s.execute(
                select(JobEvent.id, JobEvent.node, JobEvent.message, JobEvent.created_at)
                .where(JobEvent.job_id == job_id, JobEvent.id > after)
                .order_by(JobEvent.id)
            ).all()
            return [tuple(r) for r in rows], job.status

    async def events() -> AsyncIterator[str]:
        nonlocal last_id
        # Never stream forever: a job can't outlive its timeout plus the stale window.
        deadline = time.monotonic() + cfg.limits.timeout_s + cfg.jobs.stale_after_s
        last_sent = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            rows, job_status = await anyio.to_thread.run_sync(poll, last_id)
            for event_id, node, message, created in rows:
                yield _sse(
                    "progress",
                    {"node": node, "message": message, "at": _aware(created).isoformat()},
                    event_id,
                )
                last_id, last_sent = event_id, time.monotonic()
            if job_status in TERMINAL:
                yield _sse("done", {"status": job_status})
                return
            if time.monotonic() > deadline:
                yield _sse("timeout", {"status": job_status})
                return
            if time.monotonic() - last_sent >= HEARTBEAT_S:
                yield ": keep-alive\n\n"
                last_sent = time.monotonic()
            await asyncio.sleep(POLL_S)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-cache: never serve a stored copy; X-Accel-Buffering: stop nginx buffering it.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/trajectory", response_model=TrajectoryOut)
def get_trajectory(
    job_id: uuid.UUID, request: Request, user: CurrentUser, db: DbSession
) -> TrajectoryOut:
    """Every model and tool call of a finished run, with inputs and outputs (personal data
    masked, long text cut), plus its sources and final claims. Read by Dyno's adapter."""
    job = _own_job(db, job_id, user.id)
    _fail_if_stale(db, job, request.app.state.config.jobs.stale_after_s)
    if job.status not in TERMINAL:
        raise HTTPException(status.HTTP_409_CONFLICT, "The job hasn't finished yet")
    row = db.get(JobTrajectory, job.id)  # absent if the run failed before a report
    report = job.report or {}
    claims = [
        {"section": sec["heading"], **claim}
        for sec in report.get("sections", [])
        for claim in sec["claims"]
    ]
    return TrajectoryOut(
        job_id=job.id,
        question=job.question,
        status=job.status,
        trace_id=row.trace_id if row else None,
        steps=row.steps if row else [],
        sources=report.get("sources", []),
        claims=claims,
    )
