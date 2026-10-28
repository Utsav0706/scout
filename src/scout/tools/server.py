"""Scout's MCP server.

build_server(ctx) returns a server whose tools are bound to one job and one user.
The agent (M4) talks to it in-process with `mcp.Client(server)`; no subprocess, and
the embedding models stay loaded. The same server also runs standalone over stdio
(see main()) so any MCP client, e.g. MCP Inspector or Claude Desktop, can use it.

Why MCP rather than plain Python functions: the tools get one standard, typed interface
(JSON schema per tool) that any agent framework or MCP client can call, and the
agent code never imports tool internals.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import anyio
import httpx
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from sqlalchemy import select

from scout.models import ToolCall, User
from scout.safety.redact import redact, redact_value
from scout.tools import impl
from scout.tools.context import ToolContext

log = logging.getLogger(__name__)

_MAX_AUDIT_ARG_CHARS = 2000


def _result_info(result: Any) -> tuple[bool, str]:
    """(is_error, text) from a tools/call result, whichever shape the SDK hands back."""
    if isinstance(result, dict):
        is_error = bool(result.get("isError") or result.get("is_error"))
        content = result.get("content") or []
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
    else:
        is_error = bool(getattr(result, "is_error", False))
        texts = [getattr(c, "text", "") for c in getattr(result, "content", []) or []]
    return is_error, "".join(texts)


class AuditMiddleware(ServerMiddleware[Any]):
    """Writes one tool_calls row for every tools/call, including calls rejected for
    bad arguments (those never reach the tool function, so they'd be invisible to a
    decorator on the function)."""

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    async def __call__(
        self, rctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        if rctx.method != "tools/call":
            return await call_next(rctx)

        params = rctx.params or {}
        tool = str(params.get("name", "?"))[:64]
        args = params.get("arguments") or {}
        start = time.perf_counter()
        status, error, chars = "error", None, 0
        try:
            result = await call_next(rctx)
            is_error, text = _result_info(result)
            status = "error" if is_error else "ok"
            error = redact(text)[:1000] if is_error else None
            chars = len(text)
            return result
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}")[:1000]
            raise
        finally:
            await self._record(tool, args, status, error, chars, start)

    async def _record(
        self, tool: str, args: Any, status: str, error: str | None, chars: int, start: float
    ) -> None:
        ms = int((time.perf_counter() - start) * 1000)
        # The audit log keeps WHAT was called, with personal data masked (safety.redact).
        args = redact_value(args)
        dumped = json.dumps(args, default=str)
        safe_args = args if len(dumped) <= _MAX_AUDIT_ARG_CHARS else {"truncated": dumped[:2000]}
        row = ToolCall(
            job_id=self._ctx.job_id,
            user_id=self._ctx.user_id,
            tool=tool,
            args=safe_args,
            status=status,
            error=error,
            result_chars=chars,
            duration_ms=ms,
        )

        def write() -> None:
            with self._ctx.session_factory() as db:
                db.add(row)
                db.commit()

        try:
            await anyio.to_thread.run_sync(write)
        except Exception:
            # A failed audit write must not break the research run; it's logged instead.
            log.exception("audit write failed", extra={"tool": tool})
        log.info("tool call", extra={"tool": tool, "status": status, "ms": ms})


async def _with_timeout[T](ctx: ToolContext, name: str, fn: Callable[[], Awaitable[T]]) -> T:
    try:
        with anyio.fail_after(ctx.cfg.tool_timeout_s):
            return await fn()
    except TimeoutError:
        raise ToolError(
            f"{name} took longer than {ctx.cfg.tool_timeout_s:.0f}s and was stopped. "
            "Try a different query or source."
        ) from None


# MCP tool annotations: tell any MCP client what each tool can do. Nothing here can send,
# post, buy or delete; save_note only writes to Scout's own notes table for this job.
WEB_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
DOCS_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
NOTES_WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)


def build_server(ctx: ToolContext) -> MCPServer:
    server = MCPServer(
        "scout-tools",
        log_level="WARNING",  # its INFO lines duplicate our own tool-call logs
        instructions=(
            "Read-only research tools. Every result has a source_id (S1, S2...). "
            "Cite facts with these ids; save_note only accepts ids returned by the tools."
        ),
        middleware=[AuditMiddleware(ctx)],
    )

    @server.tool(
        annotations=WEB_READ,
        description=(
            "Search the public web. Returns up to 5 results, each with a source_id, title, "
            "url and a short snippet. Snippets are previews: use fetch_url to read a page "
            "before relying on it."
        ),
    )
    async def web_search(
        query: Annotated[
            str,
            Field(
                min_length=2,
                max_length=400,
                description="Search query, like you'd type into a search engine",
            ),
        ],
    ) -> list[impl.WebResult]:
        return await _with_timeout(ctx, "web_search", lambda: impl.web_search(ctx, query))

    @server.tool(
        annotations=WEB_READ,
        description=(
            "Read a web page or PDF and return its main text (up to 20,000 characters). "
            "Only public http/https addresses; internal addresses are refused."
        ),
    )
    async def fetch_url(
        url: Annotated[
            str,
            Field(
                min_length=8,
                max_length=2000,
                description="Full URL starting with http:// or https://",
            ),
        ],
    ) -> impl.PageResult:
        return await _with_timeout(ctx, "fetch_url", lambda: impl.fetch_url(ctx, url))

    @server.tool(
        annotations=DOCS_READ,
        description=(
            "Search the user's uploaded documents (PDFs). Returns the most relevant "
            "passages with source_id, document title and page number."
        ),
    )
    async def rag_search(
        query: Annotated[
            str,
            Field(min_length=2, max_length=1000, description="What to look for, in plain words"),
        ],
        document_ids: Annotated[
            list[uuid.UUID] | None,
            Field(default=None, max_length=50, description="Optional: only search these documents"),
        ] = None,
        top_k: Annotated[
            int | None, Field(default=None, ge=1, le=20, description="Optional: number of passages")
        ] = None,
    ) -> list[impl.DocResult]:
        return await _with_timeout(
            ctx, "rag_search", lambda: impl.rag_search(ctx, query, document_ids, top_k)
        )

    @server.tool(
        annotations=NOTES_WRITE,
        description=(
            "Save one key fact for the report, tied to the source it came from. Keep each "
            "note to one or two sentences, in your own words."
        ),
    )
    async def save_note(
        text: Annotated[str, Field(min_length=3, max_length=2000, description="The fact to save")],
        source_id: Annotated[
            str, Field(pattern=r"^S\d{1,4}$", description="Id of the source, e.g. S3")
        ],
    ) -> impl.NoteResult:
        return await _with_timeout(ctx, "save_note", lambda: impl.save_note(ctx, text, source_id))

    return server


def main() -> None:
    """Run the tools over stdio for an MCP client, as the user in SCOUT_MCP_USER_EMAIL.

    Example (MCP Inspector):  npx @modelcontextprotocol/inspector py -m scout.tools.server
    """
    from sqlalchemy.orm import sessionmaker

    from scout.config import get_config, get_secrets
    from scout.db import init_db, make_engine
    from scout.log import setup_logging
    from scout.rag.models_ml import CrossEncoderReranker, SentenceTransformerEmbedder
    from scout.tools.netguard import system_resolver

    secrets, config = get_secrets(), get_config()
    setup_logging(secrets.log_level, stream=sys.stderr)

    email = os.environ.get("SCOUT_MCP_USER_EMAIL", "").strip().lower()
    if not email:
        raise SystemExit("Set SCOUT_MCP_USER_EMAIL to the Scout account the tools act for.")
    engine = make_engine(secrets.database_url)
    init_db(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        user_id = db.scalar(select(User.id).where(User.email == email))
    if user_id is None:
        raise SystemExit(f"No Scout user with email {email}. Register via POST /auth/register.")

    ctx = ToolContext(
        user_id=user_id,
        job_id=uuid.uuid4(),  # one "job" per stdio session, for the audit log
        session_factory=factory,
        embedder=SentenceTransformerEmbedder(config.rag.embedding_model),
        reranker=CrossEncoderReranker(config.rag.reranker_model),
        rag_cfg=config.rag,
        cfg=config.tools,
        http=httpx.AsyncClient(headers={"User-Agent": "ScoutResearchBot/0.1"}),
        resolver=system_resolver,
        tavily_api_key=(
            secrets.tavily_api_key.get_secret_value() if secrets.tavily_api_key else None
        ),
    )
    build_server(ctx).run("stdio")


if __name__ == "__main__":
    main()
