"""Everything a research run needs, built in one place for the CLI, the RQ worker and the
API's inline job mode, so the three can't drift apart.

Services are the long-lived parts (database, ML models, config): built once per process.
Per run we add a ToolContext (bound to one user and one job) and an HTTP client.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session, sessionmaker

from scout.agent.graph import OnEvent, run_research
from scout.agent.report import Report
from scout.config import AppConfig, Secrets
from scout.llm.client import LLMClient
from scout.rag.models_ml import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from scout.tools.context import Resolver, ToolContext
from scout.tools.impl import normalise_url
from scout.tools.netguard import system_resolver

USER_AGENT = "ScoutResearchBot/0.1"
# Links written in the question may be opened by fetch_url (see the link allowlist).
URL_IN_TEXT = re.compile(r"https?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]")


@dataclass
class Services:
    config: AppConfig
    secrets: Secrets
    session_factory: sessionmaker[Session]
    embedder: Embedder
    reranker: Reranker
    resolver: Resolver = system_resolver
    llm: LLMClient | None = None  # tests inject a fake; otherwise built from the API key
    http_transport: httpx.AsyncBaseTransport | None = None  # tests: a fake web

    @classmethod
    def create(
        cls, config: AppConfig, secrets: Secrets, session_factory: sessionmaker[Session]
    ) -> Services:
        return cls(
            config=config,
            secrets=secrets,
            session_factory=session_factory,
            # Loaded lazily on first use, then kept for the life of the process.
            embedder=SentenceTransformerEmbedder(config.rag.embedding_model),
            reranker=CrossEncoderReranker(config.rag.reranker_model),
        )

    def llm_client(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if self.secrets.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self.llm = LLMClient(self.config, self.secrets.anthropic_api_key.get_secret_value())
        return self.llm


@asynccontextmanager
async def tool_context(
    services: Services,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    http: httpx.AsyncClient | None = None,
    question: str = "",
) -> AsyncIterator[ToolContext]:
    """A ToolContext bound to one user and one job, with its own HTTP client."""
    own_client = http is None
    client = http or httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, transport=services.http_transport
    )
    try:
        tavily = services.secrets.tavily_api_key
        yield ToolContext(
            user_id=user_id,
            job_id=job_id,
            session_factory=services.session_factory,
            embedder=services.embedder,
            reranker=services.reranker,
            rag_cfg=services.config.rag,
            cfg=services.config.tools,
            http=client,
            resolver=services.resolver,
            tavily_api_key=tavily.get_secret_value() if tavily else None,
            known_urls=(
                {normalise_url(u) for u in URL_IN_TEXT.findall(question)}
                if services.config.tools.fetch_only_known_urls
                else None
            ),
        )
    finally:
        if own_client:
            await client.aclose()


async def research(
    services: Services,
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    question: str,
    use_web: bool,
    use_docs: bool,
    on_event: OnEvent | None = None,
    http: httpx.AsyncClient | None = None,
) -> Report:
    async with tool_context(services, user_id, job_id, http, question) as ctx:
        return await run_research(
            question,
            config=services.config,
            llm=services.llm_client(),
            tool_ctx=ctx,
            use_web=use_web,
            use_docs=use_docs,
            on_event=on_event,
        )
