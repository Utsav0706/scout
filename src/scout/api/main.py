"""FastAPI app factory.

A factory (create_app) instead of a module-level app, so tests can build an app
against their own database and config without touching env vars or globals.
Run it with: uvicorn scout.api.main:create_app --factory
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from redis import Redis
from sqlalchemy.orm import sessionmaker

from scout import __version__
from scout.agent.runtime import Services
from scout.api import routes_auth, routes_documents, routes_research
from scout.auth import require_jwt_secret
from scout.config import AppConfig, Secrets, get_config, get_secrets
from scout.db import db_is_up, init_db, make_engine
from scout.feedback_scores import ScoreSender, langfuse_score_sender
from scout.jobs.queue import InlineQueue, JobQueue, RQQueue
from scout.llm.client import LLMClient
from scout.log import setup_logging
from scout.rag.models_ml import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from scout.schemas import HealthOut
from scout.telemetry import setup_tracing

log = logging.getLogger(__name__)


QueueFactory = Callable[[Services, asyncio.AbstractEventLoop], JobQueue]


def default_queue(secrets: Secrets, config: AppConfig) -> QueueFactory:
    if secrets.jobs_mode == "inline":
        return lambda services, loop: InlineQueue(services, loop)
    # Redis.from_url doesn't connect yet: the API starts without Redis, and a queueing
    # attempt while Redis is down answers 503 instead of crashing.
    timeout = config.limits.timeout_s + 120
    return lambda services, loop: RQQueue(Redis.from_url(secrets.redis_url), timeout)


def create_app(
    secrets: Secrets | None = None,
    config: AppConfig | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    queue_factory: QueueFactory | None = None,
    http_transport: httpx.AsyncBaseTransport | None = None,
    score_sender: ScoreSender | None = None,
) -> FastAPI:
    """Tests pass fakes (models, LLM, queue, web); production builds the real ones."""
    secrets = secrets or get_secrets()
    config = config or get_config()
    setup_logging(secrets.log_level)
    setup_tracing(secrets)  # no-op without Langfuse keys
    jwt_secret = require_jwt_secret(secrets.jwt_secret)
    engine = make_engine(secrets.database_url)
    make_queue = queue_factory or default_queue(secrets, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(engine)
        app.state.job_queue = make_queue(app.state.services, asyncio.get_running_loop())
        log.info("scout api started", extra={"version": __version__, "jobs": secrets.jobs_mode})
        yield
        engine.dispose()

    app = FastAPI(title="Scout", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.jwt_secret = jwt_secret
    # Feedback -> Langfuse score (None without Langfuse keys); tests inject a fake.
    app.state.send_score = score_sender or langfuse_score_sender(secrets)
    app.state.engine = engine
    app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
    # Real models load lazily on first use; tests pass small fakes instead.
    app.state.embedder = embedder or SentenceTransformerEmbedder(config.rag.embedding_model)
    app.state.reranker = reranker or CrossEncoderReranker(config.rag.reranker_model)
    # Used by inline-mode jobs; the RQ worker builds its own from the environment.
    app.state.services = Services(
        config=config,
        secrets=secrets,
        session_factory=app.state.session_factory,
        embedder=app.state.embedder,
        reranker=app.state.reranker,
        llm=llm,
        http_transport=http_transport,
    )

    @app.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # One structured line per request, with an id the caller can quote in bug reports.
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    @app.get("/health", response_model=HealthOut, tags=["ops"])
    def health() -> JSONResponse:
        db_ok = db_is_up(engine)
        body = HealthOut(
            status="ok" if db_ok else "degraded", db="ok" if db_ok else "down", version=__version__
        )
        # 503 when the DB is down, so Docker/AWS health checks restart or stop routing to us.
        return JSONResponse(body.model_dump(), status_code=200 if db_ok else 503)

    app.include_router(routes_auth.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_research.router)
    return app
