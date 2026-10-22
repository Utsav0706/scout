"""Hand a job to whatever runs it.

RQQueue: Redis + a separate RQ worker process. Research takes 30-120 s and needs the ML
models in memory; a worker keeps that work (and its RAM) out of the API process, jobs
survive an API restart, and more workers can be added. RQ over Celery: one Redis, one
small library, no broker configuration; enough for one job type.

InlineQueue: the API runs the job itself as a background task on its own event loop.
For local development on Windows, where RQ workers can't run (they rely on fork) and
there is no Redis. Same run_job() code, so behaviour matches apart from who runs it.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import Future
from typing import Protocol

from redis import Redis
from rq import Queue

from scout.agent.runtime import Services
from scout.jobs.runner import run_ingest, run_job

QUEUE_NAME = "scout"


class JobQueue(Protocol):
    def enqueue(self, job_id: uuid.UUID) -> None: ...  # a research job
    def enqueue_ingest(self, document_id: uuid.UUID) -> None: ...  # an uploaded PDF


class QueueUnavailableError(Exception):
    pass


class RQQueue:
    # A big PDF (300 pages) can take minutes to embed on CPU.
    INGEST_TIMEOUT_S = 15 * 60

    def __init__(self, connection: Redis, job_timeout_s: int, is_async: bool = True) -> None:
        # is_async=False runs the job immediately in the caller (tests, with fakeredis).
        self._queue = Queue(QUEUE_NAME, connection=connection, is_async=is_async)
        self._timeout = job_timeout_s

    def _put(self, func: str, item_id: uuid.UUID, timeout_s: int) -> None:
        try:
            self._queue.enqueue(
                func,
                str(item_id),
                job_id=str(item_id),  # same id in RQ and in our tables: easy to trace
                job_timeout=timeout_s,
                result_ttl=3600,  # results live in our DB; RQ only needs them briefly
            )
        except Exception as exc:  # redis down, wrong URL...
            raise QueueUnavailableError(str(exc)) from exc

    def enqueue(self, job_id: uuid.UUID) -> None:
        self._put("scout.jobs.worker.run_job_sync", job_id, self._timeout)

    def enqueue_ingest(self, document_id: uuid.UUID) -> None:
        self._put("scout.jobs.worker.ingest_document_sync", document_id, self.INGEST_TIMEOUT_S)


class InlineQueue:
    def __init__(self, services: Services, loop: asyncio.AbstractEventLoop) -> None:
        self._services = services
        self._loop = loop
        self._running: set[Future] = set()  # keep references so tasks aren't collected

    def _start(self, coro) -> None:
        # Routes run in a worker thread; hand the job to the app's event loop.
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._running.add(future)
        future.add_done_callback(self._running.discard)

    def enqueue(self, job_id: uuid.UUID) -> None:
        self._start(run_job(self._services, job_id))

    def enqueue_ingest(self, document_id: uuid.UUID) -> None:
        self._start(run_ingest(self._services, document_id))
