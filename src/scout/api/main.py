"""FastAPI app factory.

A factory (create_app) instead of a module-level app, so tests can build an app
against their own database and config without touching env vars or globals.
Run it with: uvicorn scout.api.main:create_app --factory
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from scout import __version__
from scout.api import routes_auth
from scout.auth import require_jwt_secret
from scout.config import AppConfig, Secrets, get_config, get_secrets
from scout.db import db_is_up, init_db, make_engine
from scout.log import setup_logging
from scout.schemas import HealthOut

log = logging.getLogger(__name__)


def create_app(secrets: Secrets | None = None, config: AppConfig | None = None) -> FastAPI:
    secrets = secrets or get_secrets()
    config = config or get_config()
    setup_logging(secrets.log_level)
    jwt_secret = require_jwt_secret(secrets.jwt_secret)
    engine = make_engine(secrets.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(engine)
        log.info("scout api started", extra={"version": __version__})
        yield
        engine.dispose()

    app = FastAPI(title="Scout", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.jwt_secret = jwt_secret
    app.state.engine = engine
    app.state.session_factory = sessionmaker(engine, expire_on_commit=False)

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
    return app
