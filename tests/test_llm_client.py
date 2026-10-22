import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from scout.config import AppConfig, load_config
from scout.llm.client import (
    FALLBACK_BETA,
    LLMClient,
    LLMError,
    LLMRefusalError,
    LLMTruncatedError,
    LLMUnavailableError,
    Usage,
    cost_gbp,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(REPO_CONFIG)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def message(
    content: list[dict[str, Any]],
    *,
    model: str = "claude-haiku-5-5",
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    **extra: Any,
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            **extra,
        },
        headers={"request-id": "req_test"},
    )


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


class Recorder:
    """Fake Anthropic API: records each request and replies with a canned response."""

    def __init__(self, reply: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.reply(request)

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)

    @property
    def headers(self) -> httpx2.Headers:
        return self.requests[-1].headers


def make_client(
    reply: Callable[[httpx2.Request], httpx2.Response], config: AppConfig = CONFIG
) -> tuple[LLMClient, Recorder]:
    rec = Recorder(reply)
    sdk = AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,  # tests check our error mapping, not the SDK's retry backoff
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(rec)),
    )
    return LLMClient(config, api_key=None, client=sdk), rec


async def call(client: LLMClient, node: str = "reader", **kw: Any):
    return await client.complete(
        node, system="You are a test.", messages=[{"role": "user", "content": "hi"}], **kw
    )


# --- request shape ---


@pytest.mark.anyio
async def test_request_uses_node_config_and_tags_task_type() -> None:
    client, rec = make_client(lambda r: message([text("ok")]))
    await call(client, "reader")
    body = rec.body
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == CONFIG.llm.nodes["reader"].max_tokens
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body  # rejected by Sonnet 5.5 / Haiku 5.5
    assert rec.headers["x-scout-task-type"] == "extract"
    assert rec.headers["x-scout-node"] == "reader"


@pytest.mark.anyio
async def test_fallback_only_on_nodes_that_enable_it() -> None:
    client, rec = make_client(lambda r: message([text("ok")], model="claude-sonnet-5-5"))
    await call(client, "planner")
    assert rec.body["fallbacks"] == "default"
    assert FALLBACK_BETA in rec.headers["anthropic-beta"]

    await call(client, "reader")
    assert "fallbacks" not in rec.body
    assert FALLBACK_BETA not in rec.headers.get("anthropic-beta", "")


# --- results ---


@pytest.mark.anyio
async def test_text_usage_cost_and_latency() -> None:
    client, _ = make_client(
        lambda r: message([text("Hello "), text("there")], input_tokens=2000, output_tokens=500)
    )
    result = await call(client)
    assert result.text == "Hello there"
    assert result.usage == Usage(input_tokens=2000, output_tokens=500)
    # Haiku 5.5: 2000 * $0.10/M + 500 * $0.50/M = $0.00045, then to GBP
    assert result.cost_gbp == pytest.approx(0.00045 * CONFIG.usd_to_gbp)
    assert result.latency_ms >= 0


def test_cost_formula_counts_every_token_type() -> None:
    price = CONFIG.prices_usd_per_mtok["claude-sonnet-5-5"]
    usage = Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert cost_gbp(usage, price, 1.0) == pytest.approx(2.0 + 10.0 + 0.2 + 2.5)


@pytest.mark.anyio
async def test_fallback_model_is_priced_as_itself() -> None:
    client, _ = make_client(
        lambda r: message([text("ok")], model="claude-sonnet-5", output_tokens=1_000_000)
    )
    result = await call(client, "writer")
    assert result.model == "claude-sonnet-5"
    assert result.cost_gbp == pytest.approx((1000 * 2.0 / 1e6 + 10.0) * CONFIG.usd_to_gbp)


@pytest.mark.anyio
async def test_unknown_served_model_never_priced_at_zero() -> None:
    client, _ = make_client(lambda r: message([text("ok")], model="claude-mystery-9"))
    result = await call(client, "reader")
    assert result.cost_gbp > 0


@pytest.mark.anyio
async def test_tool_calls_are_extracted_and_raw_content_kept() -> None:
    blocks = [
        text("Let me search."),
        {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "uk cpi"}},
    ]
    client, rec = make_client(lambda r: message(blocks, stop_reason="tool_use"))
    tools = [{"name": "web_search", "description": "d", "input_schema": {"type": "object"}}]
    result = await call(client, "researcher", tools=tools)
    assert rec.body["tools"][0]["name"] == "web_search"
    assert result.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input) for c in result.tool_calls] == [
        ("toolu_1", "web_search", {"query": "uk cpi"})
    ]
    assert len(result.content) == 2


class Plan(BaseModel):
    sub_questions: list[str]


@pytest.mark.anyio
async def test_structured_output_is_parsed_and_validated() -> None:
    reply = json.dumps({"sub_questions": ["What is CPI?", "Latest UK rate?"]})
    client, rec = make_client(lambda r: message([text(reply)], model="claude-sonnet-5-5"))
    result = await call(client, "planner", output_format=Plan)
    assert isinstance(result.parsed, Plan)
    assert result.parsed.sub_questions == ["What is CPI?", "Latest UK rate?"]
    # The JSON schema and the effort level travel together in output_config.
    assert rec.body["output_config"]["effort"] == "medium"
    assert rec.body["output_config"]["format"]["type"] == "json_schema"


# --- failures ---


@pytest.mark.anyio
async def test_refusal_raises_with_category_and_cost() -> None:
    client, _ = make_client(
        lambda r: message(
            [],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "x"},
        )
    )
    with pytest.raises(LLMRefusalError) as err:
        await call(client)
    assert err.value.category == "cyber"
    assert err.value.cost_gbp > 0


@pytest.mark.anyio
async def test_truncated_answer_raises() -> None:
    client, _ = make_client(lambda r: message([text("half an ans")], stop_reason="max_tokens"))
    with pytest.raises(LLMTruncatedError, match="max_tokens"):
        await call(client)


def _api_error(status: int, kind: str) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda r: httpx2.Response(
        status, json={"type": "error", "error": {"type": kind, "message": "nope"}}
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "kind", "match"),
    [
        (401, "authentication_error", "API key rejected"),
        (429, "rate_limit_error", "rate limit"),
        (529, "overloaded_error", "529"),
    ],
)
async def test_service_errors_are_unavailable(status: int, kind: str, match: str) -> None:
    client, _ = make_client(_api_error(status, kind))
    with pytest.raises(LLMUnavailableError, match=match):
        await call(client)


@pytest.mark.anyio
async def test_bad_request_is_a_bug_not_unavailable() -> None:
    client, _ = make_client(_api_error(400, "invalid_request_error"))
    with pytest.raises(LLMError) as err:
        await call(client)
    assert not isinstance(err.value, LLMUnavailableError)


# --- config ---


def _raw() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))


def test_temperature_is_no_longer_accepted() -> None:
    raw = _raw()
    raw["llm"]["nodes"]["writer"]["temperature"] = 0.3
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_every_configured_model_needs_a_price() -> None:
    raw = _raw()
    del raw["prices_usd_per_mtok"]["claude-haiku-5-5"]
    with pytest.raises(ValidationError, match="claude-haiku-5-5"):
        AppConfig.model_validate(raw)
