"""Run one research job: the same function whether an RQ worker or the API (inline mode)
runs it.

It records progress lines as JobEvent rows while the agent works, then stores the report
(or the error) on the job. It never raises: whatever goes wrong ends up on the job row,
because nobody is waiting on the call to see an exception.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

import anyio

from scout.agent.runtime import Services, research
from scout.models import Document, JobEvent, ResearchJob
from scout.rag.store import process_document

log = logging.getLogger(__name__)

TERMINAL = {"complete", "partial", "no_evidence", "failed"}


def _now() -> datetime:
    return datetime.now(UTC)


def _root_cause(exc: BaseException) -> BaseException:
    """The MCP client's task group wraps errors in (nested) ExceptionGroups, which would
    make every failure read "ExceptionGroup". Unwrap single-error groups to the real one."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


async def run_job(services: Services, job_id: uuid.UUID) -> None:
    factory = services.session_factory

    def start() -> ResearchJob | None:
        with factory() as db:
            job = db.get(ResearchJob, job_id)
            # Only a queued job starts: a job picked up twice (e.g. re-delivered by the
            # queue) or already finished is left alone.
            if job is None or job.status != "queued":
                return None
            job.status, job.started_at = "running", _now()
            db.add(JobEvent(job_id=job_id, node="job", message="Research started"))
            db.commit()
            db.expunge(job)
            return job

    job = await anyio.to_thread.run_sync(start)
    if job is None:
        return

    async def on_event(node: str, message: str) -> None:
        def write() -> None:
            with factory() as db:
                db.add(JobEvent(job_id=job_id, node=node, message=message))
                db.commit()

        await anyio.to_thread.run_sync(write)

    report, error = None, None
    try:
        # Last-resort timeout on top of the graph's own time limit, which ends runs
        # gracefully with a partial report.
        report = await asyncio.wait_for(
            research(
                services,
                user_id=job.user_id,
                job_id=job_id,
                question=job.question,
                use_web=job.use_web,
                use_docs=job.use_docs,
                on_event=on_event,
            ),
            timeout=services.config.limits.timeout_s + 60,
        )
    except TimeoutError:
        error = "Research took too long and was stopped."
    except Exception as exc:  # recorded on the job; details go to the server log only
        log.exception("research job failed", extra={"job_id": str(job_id)})
        error = f"Research failed ({type(_root_cause(exc)).__name__})."

    def finish() -> None:
        with factory() as db:
            row = db.get(ResearchJob, job_id)
            if row is None:  # deleted while running (user removed)
                return
            row.finished_at = _now()
            if report is not None:
                row.status = report.status
                row.report = report.model_dump(mode="json")
                row.cost_gbp = report.cost_gbp
                message = f"Finished: {report.status}, £{report.cost_gbp:.4f}"
            else:
                row.status, row.error = "failed", error
                message = error or "Research failed."
            db.add(JobEvent(job_id=job_id, node="job", message=message))
            db.commit()

    await anyio.to_thread.run_sync(finish)
    log.info("research job finished", extra={"job_id": str(job_id)})


async def run_ingest(services: Services, document_id: uuid.UUID) -> None:
    """Process an uploaded PDF (extract, chunk, embed). CPU-bound, so it runs in a thread.
    Like run_job it never raises: failures end up on the document row."""

    def work() -> None:
        with services.session_factory() as db:
            try:
                process_document(db, document_id, services.embedder, services.config.rag)
            except Exception as exc:
                log.exception("document ingest failed", extra={"document_id": str(document_id)})
                db.rollback()
                doc = db.get(Document, document_id)
                if doc is not None:
                    doc.status = "failed"
                    doc.error = f"Processing failed ({type(_root_cause(exc)).__name__})."
                    doc.pdf_data = None
                    db.commit()

    await anyio.to_thread.run_sync(work)
