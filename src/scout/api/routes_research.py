"""Start a research job and check on it."""

from __future__ import annotations

import logging
import math
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import func, select

from scout.api.deps import CurrentUser, DbSession
from scout.jobs.queue import QueueUnavailableError
from scout.jobs.runner import TERMINAL
from scout.models import JobEvent, ResearchJob
from scout.schemas import JobCreatedOut, JobEventOut, JobOut, ResearchIn

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


@router.get("/{job_id}", response_model=JobOut)
def get_research(job_id: uuid.UUID, request: Request, user: CurrentUser, db: DbSession) -> JobOut:
    job = db.get(ResearchJob, job_id)
    # Someone else's job is "not found", not "forbidden": don't confirm it exists.
    if job is None or job.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Research job not found")

    stale_after = timedelta(seconds=request.app.state.config.jobs.stale_after_s)
    started = _aware(job.started_at)
    if job.status == "running" and started and datetime.now(UTC) - started > stale_after:
        # Its worker died (crash, restart) mid-run: say so instead of "running" forever.
        job.status, job.error = "failed", "The worker running this job stopped responding."
        job.finished_at = datetime.now(UTC)
        db.add(JobEvent(job_id=job.id, node="job", message=job.error))
        db.commit()

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
