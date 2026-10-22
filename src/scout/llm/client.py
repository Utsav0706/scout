"""LLMClient: the single place Scout calls a language model.

Every node calls complete(node, ...). Model, effort, token cap and fallback come from
config.yaml for that node, so nodes never name a model in code. One client also means
one place to point at Gearbox later (llm.base_url) and one place that prices every call.

Why the official Anthropic SDK and not LiteLLM: the current Claude models' request
options (effort, structured outputs, server-side refusal fallback) arrive in the
official SDK first; a multi-provider wrapper lags behind them. If Gearbox ends up
speaking only the OpenAI format, a second implementation of this class plugs in here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from scout.config import AppConfig, ModelPrice, NodeName

log = logging.getLogger(__name__)

# Server-side refusal fallback, "default" form: Anthropic picks the fallback model by
# refusal category. This header goes with fallbacks="default" (the array form uses a
# different one), so the two must not be mixed.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """A call that produced no usable answer. cost_gbp is what it still cost."""

    def __init__(self, message: str, cost_gbp: float = 0.0) -> None:
        super().__init__(message)
        self.cost_gbp = cost_gbp


class LLMRefusalError(LLMError):
    """The model's safety classifier declined (after any fallback also declined)."""

    def __init__(self, category: str | None, cost_gbp: float) -> None:
        super().__init__(f"model declined the request (category: {category})", cost_gbp)
        self.category = category


class LLMTruncatedError(LLMError):
    """Hit max_tokens before finishing. Usually means the node's cap is too low."""


class LLMUnavailableError(LLMError):
    """Auth, rate-limit, network or server errors (after the SDK's own retries)."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens, which are billed as output
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResult:
    node: str
    model: str  # the model that actually answered (differs if a fallback ran)
    text: str
    stop_reason: str
    usage: Usage
    cost_gbp: float
    latency_ms: int
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Raw content blocks, to append as the assistant turn in a tool loop. Sent back
    # unchanged: thinking blocks must not be edited (the API checks them).
    content: list[Any] = field(default_factory=list)


def cost_gbp(usage: Usage, price: ModelPrice, usd_to_gbp: float) -> float:
    usd = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / 1_000_000
    return usd * usd_to_gbp


def _usage(resp: Any) -> Usage:
    u = resp.usage
    return Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
    )


class LLMClient:
    def __init__(
        self, config: AppConfig, api_key: str | None, client: AsyncAnthropic | None = None
    ) -> None:
        self._config = config
        self._client = client or AsyncAnthropic(
            api_key=api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.request_timeout_s,
        )

    def _price(self, served: str, requested: str) -> ModelPrice:
        prices = self._config.prices_usd_per_mtok
        if served in prices:
            return prices[served]
        # A fallback model we have no price for: bill at the requested model's rate
        # rather than 0, and say so, so the cost guard never under-counts silently.
        log.warning("no price for model, using requested model's price", extra={"model": served})
        return prices[requested]

    async def complete(
        self,
        node: NodeName,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_format: type[BaseModel] | None = None,
    ) -> LLMResult:
        """One model call for `node`. Raises LLMError subclasses; never returns a
        refused or truncated answer as if it were fine."""
        n = self._config.llm.nodes[node]
        kwargs: dict[str, Any] = {
            "model": n.model,
            "max_tokens": n.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": n.effort},
            # Anthropic rejects unknown body fields but ignores unknown headers, so the
            # routing tag for Gearbox travels as a header.
            "extra_headers": {"x-scout-task-type": n.task_type, "x-scout-node": node},
        }
        if tools:
            kwargs["tools"] = tools
        if n.fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        start = time.perf_counter()
        try:
            if output_format is not None:
                resp = await self._client.beta.messages.parse(output_format=output_format, **kwargs)
            else:
                resp = await self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            raise LLMUnavailableError(
                "Anthropic API key rejected (check ANTHROPIC_API_KEY)"
            ) from None
        except anthropic.RateLimitError:
            raise LLMUnavailableError("Anthropic rate limit hit; try again shortly") from None
        except anthropic.BadRequestError as exc:
            # Our request was wrong (bad parameter, model id...): a bug, not bad luck.
            raise LLMError(f"Anthropic rejected the request: {exc.message}") from None
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError(f"Anthropic API error {exc.status_code}") from None
        except anthropic.APIConnectionError:
            raise LLMUnavailableError("could not reach the Anthropic API") from None
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = _usage(resp)
        cost = cost_gbp(usage, self._price(resp.model, n.model), self._config.usd_to_gbp)
        log.info(
            "llm call",
            extra={
                "node": node,
                "model": resp.model,
                "stop_reason": resp.stop_reason,
                "in": usage.input_tokens,
                "out": usage.output_tokens,
                "cost_gbp": round(cost, 6),
                "ms": latency_ms,
                "request_id": getattr(resp, "_request_id", None),
            },
        )

        # Check stop_reason before reading content: a refusal's content isn't an answer.
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusalError(getattr(details, "category", None), cost)
        if resp.stop_reason == "max_tokens":
            raise LLMTruncatedError(
                f"{node} hit its max_tokens ({n.max_tokens}) before finishing", cost
            )

        blocks = list(resp.content)
        text = "".join(b.text for b in blocks if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in blocks if b.type == "tool_use"]
        parsed = getattr(resp, "parsed_output", None) if output_format is not None else None
        if output_format is not None and parsed is None:
            raise LLMError(f"{node} returned no valid {output_format.__name__}", cost)

        return LLMResult(
            node=node,
            model=resp.model,
            text=text,
            stop_reason=resp.stop_reason,
            usage=usage,
            cost_gbp=cost,
            latency_ms=latency_ms,
            parsed=parsed,
            tool_calls=calls,
            content=blocks,
        )
