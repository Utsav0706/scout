"""Per-job state shared by the tools.

A ToolContext is built for ONE research job and ONE user. The user id is fixed here,
outside anything the LLM controls, so no tool argument can make the agent search
another user's documents.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from scout.config import RagConfig, ToolsConfig
from scout.rag.models_ml import Embedder, Reranker

# host, port -> list of IP strings. Injectable so tests never hit real DNS.
Resolver = Callable[[str, int], Awaitable[list[str]]]


class Source(BaseModel):
    """Something the agent can cite. Ids are short (S1, S2...) so LLMs copy them reliably."""

    id: str
    title: str
    url_or_doc: str  # https://... for web pages, doc:<document_id>#page=<n> for PDFs


class SourceRegistry:
    """Hands out S1, S2... ids, reusing the same id when a source is seen again
    (e.g. a page found by web_search and later read with fetch_url)."""

    def __init__(self) -> None:
        self._by_key: dict[str, Source] = {}
        self._lock = threading.Lock()

    def register(self, key: str, title: str, url_or_doc: str) -> Source:
        with self._lock:
            if key not in self._by_key:
                sid = f"S{len(self._by_key) + 1}"
                self._by_key[key] = Source(id=sid, title=title[:300], url_or_doc=url_or_doc)
            return self._by_key[key]

    def get(self, source_id: str) -> Source | None:
        return next((s for s in self._by_key.values() if s.id == source_id), None)

    def all(self) -> list[Source]:
        return list(self._by_key.values())


@dataclass
class ToolContext:
    user_id: uuid.UUID
    job_id: uuid.UUID
    session_factory: sessionmaker[Session]
    embedder: Embedder
    reranker: Reranker
    rag_cfg: RagConfig
    cfg: ToolsConfig
    http: httpx.AsyncClient
    resolver: Resolver
    tavily_api_key: str | None
    sources: SourceRegistry = field(default_factory=SourceRegistry)
    # url -> the asyncio Task fetching it (done or in flight); see impl.fetch_url.
    fetch_cache: dict[str, Any] = field(default_factory=dict)
