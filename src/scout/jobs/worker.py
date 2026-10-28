"""The RQ worker's entry point.

Start a worker (Linux / Docker; RQ workers don't run on Windows):

    rq worker scout --url $REDIS_URL --worker-class rq.worker.SimpleWorker

SimpleWorker runs jobs in its own process instead of forking one per job, so the
embedding models load once and stay loaded (forking would reload ~400 MB per job).
Trade-off: a job that crashes the process takes the worker down; Docker restarts it,
and the API marks the abandoned job failed after jobs.stale_after_s.
"""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache

from sqlalchemy.orm import sessionmaker

from scout.agent.runtime import Services
from scout.config import get_config, get_secrets
from scout.db import init_db, make_engine
from scout.jobs.runner import run_ingest, run_job
from scout.log import setup_logging
from scout.telemetry import setup_tracing

_override: Services | None = None


def configure(services: Services | None) -> None:
    """Tests: run jobs against these services instead of ones built from the env."""
    global _override
    _override = services


@lru_cache(maxsize=1)
def _services_from_env() -> Services:
    secrets, config = get_secrets(), get_config()
    setup_logging(secrets.log_level)
    setup_tracing(secrets)
    engine = make_engine(secrets.database_url)
    init_db(engine)
    return Services.create(config, secrets, sessionmaker(engine, expire_on_commit=False))


def run_job_sync(job_id: str) -> None:
    """Called by RQ (synchronously) for each job."""
    asyncio.run(run_job(_override or _services_from_env(), uuid.UUID(job_id)))


def ingest_document_sync(document_id: str) -> None:
    """Called by RQ for each uploaded PDF."""
    asyncio.run(run_ingest(_override or _services_from_env(), uuid.UUID(document_id)))
