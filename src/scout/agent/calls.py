"""The two ways a node touches the outside world: a model call and a tool call.

llm_call() checks the shared budget first, charges what the call cost (even when it
fails) and records a step. tool_call() calls an MCP tool and records a step. Every node,
including the verifier, goes through these, so cost, step accounting, tracing and the
exported trajectory all live in one place.
"""

from __future__ import annotations

import json
import time
from typing import Any

from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from scout.agent.report import StepRecord, TrajectoryStep
from scout.agent.state import AgentDeps
from scout.config import NodeName
from scout.llm.client import LLMError, LLMResult
from scout.telemetry import clip, set_content, tracer


def _record(deps: AgentDeps, **fields: Any) -> None:
    """Append one step to the run's trajectory (stored for Dyno; see GET .../trajectory)."""
    limit = deps.config.telemetry.max_field_chars
    for key in ("input", "output"):
        if fields.get(key) is not None:
            fields[key] = clip(fields[key], limit)
    deps.trajectory.append(TrajectoryStep(i=len(deps.trajectory), **fields))


def _llm_output(result: LLMResult) -> Any:
    out: dict[str, Any] = {}
    if result.parsed is not None:
        out["parsed"] = result.parsed.model_dump()
    elif result.text:
        out["text"] = result.text
    if result.tool_calls:
        out["tool_calls"] = [{"name": c.name, "input": c.input} for c in result.tool_calls]
    return out


async def llm_call(
    deps: AgentDeps,
    node: NodeName,
    *,
    reserve_gbp: float = 0.0,
    output_format: type[BaseModel] | None = None,
    **kwargs: Any,
) -> LLMResult:
    deps.budget.check(reserve_gbp)
    cfg = deps.config
    model = cfg.llm.nodes[node].model
    prompt = {"system": kwargs.get("system"), "messages": kwargs.get("messages")}
    start_ms = int(deps.budget.elapsed_s * 1000)
    # One "generation" per model call: Langfuse shows model, tokens and cost.
    with tracer().start_as_current_span(f"llm.{node}") as span:
        span.set_attribute("langfuse.observation.type", "generation")
        span.set_attribute("gen_ai.request.model", model)
        set_content(span, cfg.telemetry, input=prompt)
        try:
            result = await deps.llm.complete(node, output_format=output_format, **kwargs)
        except LLMError as exc:
            deps.budget.charge(exc.cost_gbp)
            deps.steps.append(
                StepRecord(
                    node=node,
                    kind="llm",
                    name=node,
                    ok=False,
                    ms=0,
                    cost_gbp=exc.cost_gbp,
                    detail=str(exc),
                )
            )
            _record(
                deps,
                node=node,
                kind="llm_call",
                name=model,
                ok=False,
                start_ms=start_ms,
                ms=0,
                input=prompt,
                cost_gbp=exc.cost_gbp,
                error=str(exc),
            )
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        deps.budget.charge(result.cost_gbp)
        span.set_attribute("gen_ai.response.model", result.model)
        span.set_attribute("gen_ai.usage.input_tokens", result.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.usage.output_tokens)
        # Langfuse prices in USD; Scout's own reports stay in GBP.
        usd = result.cost_gbp / cfg.usd_to_gbp
        span.set_attribute("langfuse.observation.cost_details", json.dumps({"total": usd}))
        output = _llm_output(result)
        set_content(span, cfg.telemetry, output=output)

    deps.steps.append(
        StepRecord(
            node=node,
            kind="llm",
            name=result.model,
            ok=True,
            ms=result.latency_ms,
            cost_gbp=result.cost_gbp,
        )
    )
    _record(
        deps,
        node=node,
        kind="llm_call",
        name=result.model,
        ok=True,
        start_ms=start_ms,
        ms=result.latency_ms,
        input=prompt,
        output=output,
        tokens_in=result.usage.input_tokens,
        tokens_out=result.usage.output_tokens,
        cost_gbp=result.cost_gbp,
    )
    return result


async def tool_call(
    deps: AgentDeps, node: str, name: str, args: dict[str, Any]
) -> tuple[bool, Any, str]:
    """Call an MCP tool. Returns (ok, structured result, text for the model)."""
    start = time.perf_counter()
    start_ms = int(deps.budget.elapsed_s * 1000)
    with tracer().start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("langfuse.observation.type", "tool")
        span.set_attribute("gen_ai.tool.name", name)
        set_content(span, deps.config.telemetry, input=args)
        try:
            result = await deps.mcp.call_tool(name, args)
            ok = not result.is_error
            structured = result.structured_content
            if ok and structured is not None:
                text = json.dumps(structured, ensure_ascii=False)
            else:
                text = "".join(getattr(c, "text", "") for c in result.content)
        except Exception as exc:  # transport-level failure: report it to the model, keep going
            ok, structured, text = False, None, f"Tool {name} failed: {type(exc).__name__}"
        set_content(span, deps.config.telemetry, output=text)
        if not ok:
            span.set_status(Status(StatusCode.ERROR, text[:200]))
    ms = int((time.perf_counter() - start) * 1000)
    deps.steps.append(
        StepRecord(
            node=node, kind="tool", name=name, ok=ok, ms=ms, detail=_detail(ok, structured, text)
        )
    )
    _record(
        deps,
        node=node,
        kind="tool_call",
        name=name,
        ok=ok,
        start_ms=start_ms,
        ms=ms,
        input=args,
        output=text,
        error=None if ok else text[:300],
    )
    return ok, structured, text


EXTRACT_DETAIL = "via tavily_extract"


def _detail(ok: bool, structured: Any, text: str) -> str | None:
    if not ok:
        return text[:300]
    # Pages read through Tavily Extract cost credits: record them so a run's credit use
    # can be counted from its steps.
    if isinstance(structured, dict) and structured.get("via") == "tavily_extract":
        return EXTRACT_DETAIL
    return None
