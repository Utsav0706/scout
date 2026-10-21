import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from scout.config import load_config
from scout.db import Base, init_db, make_engine
from scout.models import Note, ToolCall, User
from scout.rag.store import ingest_pdf
from scout.tools import impl
from scout.tools.context import ToolContext
from scout.tools.netguard import BlockedURLError, check_url
from scout.tools.server import build_server
from tests.conftest import TEST_DATABASE_URL, requires_postgres
from tests.fakes import FakeEmbedder, FakeReranker, make_pdf

CONFIG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")

# Fake DNS: hostnames used in tests and what they "resolve" to.
DNS = {
    "example.com": ["93.184.216.34"],
    "news.example.com": ["93.184.216.35"],
    "evil.example.com": ["127.0.0.1"],
    "mixed.example.com": ["93.184.216.34", "10.0.0.5"],
    "metadata.example.com": ["169.254.169.254"],
    "v6local.example.com": ["::1"],
    "mapped.example.com": ["::ffff:127.0.0.1"],
}


async def fake_resolver(host: str, port: int) -> list[str]:
    if host in DNS:
        return DNS[host]
    if host.replace(".", "").isdigit() or ":" in host:
        return [host]  # literal IP
    raise OSError("unknown host")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def db_factory() -> Iterator[sessionmaker[Session]]:
    engine = make_engine(TEST_DATABASE_URL)
    init_db(engine)
    yield sessionmaker(engine, expire_on_commit=False)
    Base.metadata.drop_all(engine)
    engine.dispose()


def _user(factory: sessionmaker[Session], email: str = "uv@example.com") -> uuid.UUID:
    with factory() as db:
        user = User(email=email, password_hash="x")
        db.add(user)
        db.commit()
        return user.id


Handler = Callable[[httpx.Request], httpx.Response]


def make_ctx(
    factory: sessionmaker[Session], handler: Handler, *, key: str | None = "tvly-test", **cfg
) -> ToolContext:
    return ToolContext(
        user_id=_user(factory),
        job_id=uuid.uuid4(),
        session_factory=factory,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        rag_cfg=CONFIG.rag,
        cfg=CONFIG.tools.model_copy(update=cfg),
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=fake_resolver,
        tavily_api_key=key,
    )


def html(body: str, title: str = "Page") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=f"<html><head><title>{title}</title><script>steal()</script></head>"
        f"<body><nav>Menu</nav><p>{body}</p><footer>Cookies</footer></body></html>",
    )


def tavily(*urls: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"results": [{"title": f"T {u}", "url": u, "content": f"snippet {u}"} for u in urls]},
    )


# --- netguard ---


@pytest.mark.anyio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.1.2.3/",
        "http://evil.example.com/",
        "http://mixed.example.com/",
        "http://metadata.example.com/",
        "http://v6local.example.com/",
        "http://mapped.example.com/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "https://example.com:8443/",
        "https://user:pw@example.com/",
        "https://no-such-host.invalid/",
    ],
)
async def test_netguard_blocks(url: str) -> None:
    with pytest.raises(BlockedURLError):
        await check_url(url, fake_resolver)


@pytest.mark.anyio
async def test_netguard_allows_public_https() -> None:
    await check_url("https://example.com/article?id=1", fake_resolver)


# --- web_search ---


@pytest.mark.anyio
async def test_web_search_needs_key(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily(), key=None)
    with pytest.raises(ToolError, match="not configured"):
        await impl.web_search(ctx, "anything")


@pytest.mark.anyio
async def test_web_search_maps_results_and_numbers_sources(db_factory) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return tavily("https://example.com/a", "https://example.com/b")

    ctx = make_ctx(db_factory, handler)
    results = await impl.web_search(ctx, "uk inflation 2026")
    assert [r.source_id for r in results] == ["S1", "S2"]
    assert results[0].url == "https://example.com/a"
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["max_results"] == CONFIG.tools.search_max_results
    # Searching again returns the same pages: same ids, no new ones.
    again = await impl.web_search(ctx, "uk inflation 2026")
    assert [r.source_id for r in again] == ["S1", "S2"]


@pytest.mark.anyio
@pytest.mark.parametrize(("status", "msg"), [(401, "API key"), (429, "rate-limited"), (500, "500")])
async def test_web_search_errors_are_readable(db_factory, status: int, msg: str) -> None:
    ctx = make_ctx(db_factory, lambda r: httpx.Response(status, json={}))
    with pytest.raises(ToolError, match=msg):
        await impl.web_search(ctx, "q")


# --- fetch_url ---


@pytest.mark.anyio
async def test_fetch_url_returns_clean_text(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: html("Inflation fell to 2.1%.", title="ONS"))
    page = await impl.fetch_url(ctx, "https://example.com/ons")
    assert page.title == "ONS"
    assert "Inflation fell to 2.1%." in page.text
    for junk in ("steal()", "Menu", "Cookies"):
        assert junk not in page.text
    assert not page.truncated


@pytest.mark.anyio
async def test_page_wrapped_in_a_form_keeps_its_content(db_factory) -> None:
    # ASP.NET sites (found live on tradingeconomics.com) put the whole body in <form>.
    page_html = (
        "<html><head><title>UK CPI</title></head><body><form id='aspnetForm'>"
        "<h1>UK inflation</h1><p>CPI rose 3.8% in the 12 months to September.</p>"
        "</form></body></html>"
    )
    ctx = make_ctx(
        db_factory,
        lambda r: httpx.Response(200, headers={"content-type": "text/html"}, text=page_html),
    )
    page = await impl.fetch_url(ctx, "https://example.com/cpi")
    assert "CPI rose 3.8%" in page.text


@pytest.mark.parametrize(
    ("data", "declared", "expected"),
    [
        ("UK’s rate".encode(), "utf-8", "UK’s rate"),
        ("UK’s rate".encode("cp1252"), "utf-8", "UK’s rate"),  # mislabelled page
        ("café".encode()[:-1], "utf-8", "caf"),  # size cap split the last character
        (b"plain", "no-such-charset", "plain"),
    ],
)
def test_decode_handles_real_world_charsets(data: bytes, declared: str, expected: str) -> None:
    assert impl._decode(data, declared) == expected


@pytest.mark.anyio
async def test_fetched_page_keeps_its_search_source_id(db_factory) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.tavily.com":
            return tavily("https://example.com/x", "https://example.com/y")
        return html("body")

    ctx = make_ctx(db_factory, handler)
    await impl.web_search(ctx, "q")
    page = await impl.fetch_url(ctx, "https://example.com/y")
    assert page.source_id == "S2"


@pytest.mark.anyio
async def test_redirect_to_internal_address_is_blocked(db_factory) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    ctx = make_ctx(db_factory, handler)
    with pytest.raises(ToolError, match="private or internal"):
        await impl.fetch_url(ctx, "https://example.com/go")


@pytest.mark.anyio
async def test_public_redirect_is_followed(db_factory) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "example.com":
            return httpx.Response(301, headers={"location": "https://news.example.com/story"})
        return html("The real story.")

    ctx = make_ctx(db_factory, handler)
    page = await impl.fetch_url(ctx, "https://example.com/old")
    assert page.url == "https://news.example.com/story"
    assert "The real story." in page.text


@pytest.mark.anyio
async def test_redirect_loop_gives_up(db_factory) -> None:
    ctx = make_ctx(
        db_factory, lambda r: httpx.Response(302, headers={"location": "https://example.com/again"})
    )
    with pytest.raises(ToolError, match="redirects"):
        await impl.fetch_url(ctx, "https://example.com/start")


@pytest.mark.anyio
async def test_non_html_refused(db_factory) -> None:
    ctx = make_ctx(
        db_factory,
        lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"),
    )
    with pytest.raises(ToolError, match="application/pdf"):
        await impl.fetch_url(ctx, "https://example.com/file.pdf")


@pytest.mark.anyio
async def test_long_page_truncated(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: html("word " * 5000), fetch_max_chars=1000)
    page = await impl.fetch_url(ctx, "https://example.com/long")
    assert len(page.text) == 1000
    assert page.truncated


@pytest.mark.anyio
async def test_http_error_and_timeout_are_readable(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: httpx.Response(404))
    with pytest.raises(ToolError, match="HTTP 404"):
        await impl.fetch_url(ctx, "https://example.com/missing")

    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    ctx.http = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    with pytest.raises(ToolError, match="timed out"):
        await impl.fetch_url(ctx, "https://example.com/slow")


# --- save_note ---


@pytest.mark.anyio
async def test_save_note_rejects_unknown_source(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily("https://example.com/a"))
    with pytest.raises(ToolError, match="none yet"):
        await impl.save_note(ctx, "A fact.", "S1")
    await impl.web_search(ctx, "q")
    with pytest.raises(ToolError, match=r"known: S1"):
        await impl.save_note(ctx, "A fact.", "S7")


@pytest.mark.anyio
async def test_save_note_stores_note_for_job(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily("https://example.com/a"))
    await impl.web_search(ctx, "q")
    result = await impl.save_note(ctx, "Refunds take 30 days.", "S1")
    with db_factory() as db:
        note = db.get(Note, uuid.UUID(result.note_id))
    assert note is not None
    assert (note.job_id, note.source_id, note.text) == (ctx.job_id, "S1", "Refunds take 30 days.")


# --- the MCP server, through a real in-process MCP client ---


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def _audit(factory: sessionmaker[Session]) -> list[ToolCall]:
    with factory() as db:
        return list(db.scalars(select(ToolCall).order_by(ToolCall.created_at)))


@pytest.mark.anyio
async def test_server_lists_four_tools_with_schemas(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily())
    async with Client(build_server(ctx)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"web_search", "fetch_url", "rag_search", "save_note"}
    note_schema = tools["save_note"].input_schema
    assert note_schema["properties"]["source_id"]["pattern"] == r"^S\d{1,4}$"
    assert set(note_schema["required"]) == {"text", "source_id"}


@pytest.mark.anyio
async def test_successful_call_is_audited(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily("https://example.com/a"))
    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("web_search", {"query": "uk inflation"})
    assert not result.is_error
    assert "S1" in _text(result)
    [row] = _audit(db_factory)
    assert (row.tool, row.status, row.args) == ("web_search", "ok", {"query": "uk inflation"})
    assert row.job_id == ctx.job_id and row.user_id == ctx.user_id
    assert row.result_chars > 0


@pytest.mark.anyio
async def test_bad_arguments_rejected_and_audited(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily())
    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("save_note", {"text": "x", "source_id": "DROP TABLE"})
    assert result.is_error
    [row] = _audit(db_factory)
    assert (row.tool, row.status) == ("save_note", "error")


@pytest.mark.anyio
async def test_tool_errors_reach_the_agent_as_readable_text(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily())
    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("fetch_url", {"url": "http://169.254.169.254/latest/"})
    assert result.is_error
    assert "private or internal" in _text(result)
    assert "Try a different source" in _text(result)


@pytest.mark.anyio
async def test_slow_tool_is_stopped_by_timeout(db_factory) -> None:
    async def never_answers(req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return html("too late")

    ctx = make_ctx(db_factory, lambda r: html("x"), tool_timeout_s=0.2)
    ctx.http = httpx.AsyncClient(transport=httpx.MockTransport(never_answers))
    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("fetch_url", {"url": "https://example.com/slow"})
    assert result.is_error
    assert "was stopped" in _text(result)


@pytest.mark.anyio
async def test_rag_search_without_postgres_is_a_readable_error(db_factory) -> None:
    if TEST_DATABASE_URL.startswith("postgresql"):
        pytest.skip("covered by the Postgres test below")
    ctx = make_ctx(db_factory, lambda r: tavily())
    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("rag_search", {"query": "refund policy"})
    assert result.is_error
    assert "unavailable" in _text(result)


@requires_postgres
@pytest.mark.anyio
async def test_rag_search_returns_cited_passages_for_this_user_only(db_factory) -> None:
    ctx = make_ctx(db_factory, lambda r: tavily())
    other_user = _user(db_factory, "other@example.com")
    with db_factory() as db:
        ingest_pdf(
            db,
            ctx.user_id,
            "policy.pdf",
            make_pdf(["Refund policy. Refunds within 30 days."]),
            FakeEmbedder(),
            CONFIG.rag,
        )
        ingest_pdf(
            db,
            other_user,
            "secret.pdf",
            make_pdf(["Refund secret. Other user's refund data."]),
            FakeEmbedder(),
            CONFIG.rag,
        )

    async with Client(build_server(ctx)) as client:
        result = await client.call_tool("rag_search", {"query": "refund"})
    assert not result.is_error, _text(result)
    text = _text(result)
    assert "policy.pdf, p. 1" in text and "S1" in text
    assert "secret" not in text
