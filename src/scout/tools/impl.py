"""The four tool implementations. Plain async functions taking a ToolContext, so they
can be unit-tested without MCP; server.py exposes them as MCP tools.

Errors are raised as ToolError with a message that tells the agent what to do next
("try a different source"), because the LLM reads these messages and decides its
next step from them.
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import urljoin

import anyio
import httpx
from bs4 import BeautifulSoup
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from scout.models import Note
from scout.rag import store
from scout.tools.context import ToolContext
from scout.tools.netguard import BlockedURLError, check_url

TAVILY_URL = "https://api.tavily.com/search"
_TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_BLANK_LINES = re.compile(r"\n\s*\n+")


class WebResult(BaseModel):
    source_id: str
    title: str
    url: str
    snippet: str


class PageResult(BaseModel):
    source_id: str
    title: str
    url: str
    text: str
    truncated: bool


class DocResult(BaseModel):
    source_id: str
    title: str
    page: int
    text: str
    score: float


class NoteResult(BaseModel):
    note_id: str
    source_id: str


async def web_search(ctx: ToolContext, query: str) -> list[WebResult]:
    if not ctx.tavily_api_key:
        raise ToolError("web_search is not configured (no TAVILY_API_KEY). Use rag_search instead.")
    try:
        resp = await ctx.http.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {ctx.tavily_api_key}"},
            json={
                "query": query,
                "max_results": ctx.cfg.search_max_results,
                "search_depth": "basic",  # 1 credit; "advanced" costs 2
                "include_answer": False,  # we want sources, not someone else's summary
            },
            timeout=ctx.cfg.http_timeout_s,
        )
    except httpx.TimeoutException:
        raise ToolError("web_search timed out. Try again with a shorter query.") from None
    except httpx.HTTPError as exc:
        raise ToolError(f"web_search failed to connect ({type(exc).__name__}).") from None

    if resp.status_code == 401:
        raise ToolError("web_search is misconfigured (API key rejected). Use rag_search instead.")
    if resp.status_code == 429:
        raise ToolError("web_search is rate-limited right now. Use the sources you already have.")
    if resp.status_code >= 400:
        raise ToolError(f"web_search failed (HTTP {resp.status_code}). Try a different query.")

    results = []
    for item in resp.json().get("results", []):
        url = str(item.get("url", ""))
        if not url:
            continue
        title = str(item.get("title") or url)
        src = ctx.sources.register(url, title, url)
        results.append(
            WebResult(
                source_id=src.id, title=title, url=url, snippet=str(item.get("content", ""))[:500]
            )
        )
    return results


def _decode(data: bytes, declared: str) -> str:
    """Decode with the page's declared charset, falling back to Windows-1252.

    Plenty of sites say "utf-8" but send Windows-1252 curly quotes (seen live:
    "UK�s"). Windows-1252 is the usual real encoding in that case.
    """
    try:
        return data.decode(declared)
    except UnicodeDecodeError as exc:
        if exc.start >= len(data) - 3:
            # Only the last few bytes failed: the size cap cut a character in half.
            return data[: exc.start].decode(declared)
    except LookupError:
        pass  # unknown charset name from the server
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    # Drop code, styling and page chrome; keep the readable body. Not <form>: ASP.NET
    # sites wrap the whole page in one form, so removing it deletes all the content.
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return title, _BLANK_LINES.sub("\n\n", text)


async def fetch_url(ctx: ToolContext, url: str) -> PageResult:
    current = url
    # Follow redirects by hand so every hop goes through the SSRF check. With
    # automatic redirects, a public URL could bounce us to http://localhost.
    for _ in range(ctx.cfg.max_redirects + 1):
        try:
            await check_url(current, ctx.resolver)
        except BlockedURLError as exc:
            raise ToolError(
                f"fetch_url refused {current}: {exc}. Try a different source."
            ) from None
        try:
            async with ctx.http.stream(
                "GET", current, timeout=ctx.cfg.http_timeout_s, follow_redirects=False
            ) as resp:
                if resp.is_redirect and "location" in resp.headers:
                    current = urljoin(current, resp.headers["location"])
                    continue
                if resp.status_code >= 400:
                    raise ToolError(
                        f"fetch_url got HTTP {resp.status_code} from {current}. "
                        "Try a different source."
                    )
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in _TEXT_TYPES:
                    raise ToolError(
                        f"fetch_url can only read web pages, not '{ctype or 'unknown'}'. "
                        "Try a different source."
                    )
                body = bytearray()
                async for part in resp.aiter_bytes():
                    body += part
                    if len(body) > ctx.cfg.fetch_max_bytes:
                        break  # read only the first ~2 MB of huge pages
                encoding = resp.encoding or "utf-8"
        except httpx.TimeoutException:
            raise ToolError(f"fetch_url timed out on {current}. Try a different source.") from None
        except httpx.HTTPError as exc:
            raise ToolError(
                f"fetch_url could not reach {current} ({type(exc).__name__}). "
                "Try a different source."
            ) from None

        raw = _decode(bytes(body[: ctx.cfg.fetch_max_bytes]), encoding)
        title, text = _html_to_text(raw) if ctype != "text/plain" else ("", raw.strip())
        if not text:
            raise ToolError(f"fetch_url found no readable text on {current}.")
        limit = ctx.cfg.fetch_max_chars
        src = ctx.sources.register(url, title or current, current)
        return PageResult(
            source_id=src.id,
            title=title or current,
            url=current,
            text=text[:limit],
            truncated=len(text) > limit,
        )
    raise ToolError(f"fetch_url gave up after {ctx.cfg.max_redirects} redirects on {url}.")


async def rag_search(
    ctx: ToolContext, query: str, document_ids: list[uuid.UUID] | None, top_k: int | None
) -> list[DocResult]:
    def run() -> list[store.SearchHit]:
        with ctx.session_factory() as db:
            return store.search(
                db,
                ctx.user_id,  # from the job, never from the LLM
                query,
                ctx.embedder,
                ctx.reranker,
                ctx.rag_cfg,
                document_ids=document_ids,
                top_k=top_k,
            )

    try:
        # Embedding + reranking is CPU work: run it in a thread so the event loop
        # (and other tool calls running in parallel) isn't blocked.
        hits = await anyio.to_thread.run_sync(run)
    except NotImplementedError:
        raise ToolError("rag_search is unavailable (document store not configured).") from None

    results = []
    for h in hits:
        title = f"{h.filename}, p. {h.page}"
        src = ctx.sources.register(
            f"chunk:{h.chunk_id}", title, f"doc:{h.document_id}#page={h.page}"
        )
        results.append(
            DocResult(source_id=src.id, title=title, page=h.page, text=h.text, score=h.score)
        )
    return results


async def save_note(ctx: ToolContext, text: str, source_id: str) -> NoteResult:
    # Only ids the tools actually returned in this job are accepted, so the agent
    # can't attach a note to a source it invented.
    if ctx.sources.get(source_id) is None:
        known = ", ".join(s.id for s in ctx.sources.all()) or "none yet"
        raise ToolError(
            f"Unknown source_id '{source_id}'. Use an id returned by web_search, fetch_url "
            f"or rag_search (known: {known})."
        )

    def run() -> uuid.UUID:
        with ctx.session_factory() as db:
            note = Note(job_id=ctx.job_id, source_id=source_id, text=text)
            db.add(note)
            db.commit()
            return note.id

    note_id = await anyio.to_thread.run_sync(run)
    return NoteResult(note_id=str(note_id), source_id=source_id)
