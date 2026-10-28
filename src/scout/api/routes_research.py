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
from typing import Annotated

import anyio
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from scout.api.deps import CurrentUser, DbSession
from scout.jobs.queue import QueueUnavailableError
from scout.jobs.runner import TERMINAL
from scout.models import Feedback, JobEvent, JobTrajectory, ResearchJob
from scout.safety.injection import scan
from scout.schemas import (
    DayCount,
    FeedbackIn,
    FeedbackOut,
    JobCreatedOut,
    JobEventOut,
    JobOut,
    JobSummaryOut,
    ResearchIn,
    StatsOut,
    TrajectoryOut,
)

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


@router.get("", response_model=list[JobSummaryOut])
def list_research(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    after: Annotated[
        uuid.UUID | None, Query(description="id of the last job on the previous page")
    ] = None,
) -> list[JobSummaryOut]:
    """Your research jobs, newest first. Page with ?after=<id of the last job you got>.

    Keyset paging on (created_at, id): jobs created in the same instant still page
    correctly, and the database compares its own stored times (SQLite stores whole
    seconds in a different text format from a bound datetime parameter)."""
    query = select(ResearchJob).where(ResearchJob.user_id == user.id)
    if after is not None:
        last = (
            select(ResearchJob.created_at)
            .where(ResearchJob.id == after, ResearchJob.user_id == user.id)
            .scalar_subquery()
        )
        query = query.where(
            or_(
                ResearchJob.created_at < last,
                and_(ResearchJob.created_at == last, ResearchJob.id < after),
            )
        )
    order = (ResearchJob.created_at.desc(), ResearchJob.id.desc())
    jobs = db.scalars(query.order_by(*order).limit(limit)).all()
    ratings = dict(
        db.execute(
            select(Feedback.job_id, Feedback.rating).where(
                Feedback.user_id == user.id, Feedback.job_id.in_([j.id for j in jobs])
            )
        ).all()
    )
    return [
        JobSummaryOut(
            id=j.id,
            question=j.question,
            status=j.status,
            created_at=_aware(j.created_at),
            finished_at=_aware(j.finished_at),
            cost_gbp=j.cost_gbp,
            rating=ratings.get(j.id),
        )
        for j in jobs
    ]


@router.get("/stats", response_model=StatsOut)
def research_stats(
    user: CurrentUser, db: DbSession, days: Annotated[int, Query(ge=1, le=90)] = 30
) -> StatsOut:
    """Your own numbers over the last `days`: reports per day, average cost and time,
    failure rate, feedback. (Scout has no admin role, so stats are always per user.)"""
    since = datetime.now(UTC) - timedelta(days=days)
    jobs = db.scalars(
        select(ResearchJob).where(ResearchJob.user_id == user.id, ResearchJob.created_at >= since)
    ).all()
    finished = [j for j in jobs if j.status in TERMINAL]
    reported = [j for j in finished if j.report]
    latencies = [j.report.get("latency_s") for j in reported if j.report.get("latency_s")]
    per_day: dict[str, int] = {}
    for j in reported:
        day = _aware(j.created_at).date().isoformat()
        per_day[day] = per_day.get(day, 0) + 1
    by_status: dict[str, int] = {}
    for j in jobs:
        by_status[j.status] = by_status.get(j.status, 0) + 1
    ratings = db.scalars(
        select(Feedback.rating).where(
            Feedback.user_id == user.id, Feedback.job_id.in_([j.id for j in jobs])
        )
    ).all()
    return StatsOut(
        days=days,
        jobs=len(jobs),
        finished=len(finished),
        by_status=by_status,
        reports_per_day=[DayCount(day=d, reports=n) for d, n in sorted(per_day.items())],
        avg_cost_gbp=round(sum(j.cost_gbp for j in reported) / len(reported), 4)
        if reported
        else None,
        avg_latency_s=round(sum(latencies) / len(latencies), 1) if latencies else None,
        failure_rate=round(by_status.get("failed", 0) / len(finished), 3) if finished else None,
        thumbs_up=ratings.count("up"),
        thumbs_down=ratings.count("down"),
    )


@router.get("/{job_id}", response_model=JobOut)
def get_research(job_id: uuid.UUID, request: Request, user: CurrentUser, db: DbSession) -> JobOut:
    job = _own_job(db, job_id, user.id)
    _fail_if_stale(db, job, request.app.state.config.jobs.stale_after_s)
    events = db.scalars(
        select(JobEvent).where(JobEvent.job_id == job.id).order_by(JobEvent.id)
    ).all()
    fb = db.scalar(select(Feedback).where(Feedback.job_id == job.id, Feedback.user_id == user.id))
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
        feedback=FeedbackOut(job_id=job.id, rating=fb.rating, comment=fb.comment) if fb else None,
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


@router.post("/{job_id}/feedback", response_model=FeedbackOut)
def give_feedback(
    job_id: uuid.UUID, body: FeedbackIn, request: Request, user: CurrentUser, db: DbSession
) -> FeedbackOut:
    """Thumbs up/down on a finished report. Re-rating overwrites. Also sent to Langfuse as
    a score on the run's trace (best effort; Scout's own table is the record)."""
    job = _own_job(db, job_id, user.id)
    if job.status not in TERMINAL:
        raise HTTPException(status.HTTP_409_CONFLICT, "Rate a report once it has finished")
    row = db.scalar(select(Feedback).where(Feedback.job_id == job.id, Feedback.user_id == user.id))
    if row is None:
        row = Feedback(job_id=job.id, user_id=user.id, rating=body.rating, comment=body.comment)
        db.add(row)
    else:
        row.rating, row.comment = body.rating, body.comment
    db.commit()

    send = request.app.state.send_score
    trajectory = db.get(JobTrajectory, job.id)
    if send is not None and trajectory is not None and trajectory.trace_id:
        send(trajectory.trace_id, str(job.id), body.rating, body.comment)
    return FeedbackOut(job_id=job.id, rating=body.rating, comment=body.comment)
