"""The two ways a node touches the outside world: a model call and a tool call.

llm_call() checks the shared budget first, charges what the call cost (even when it
fails) and records a step. tool_call() calls an MCP tool and records a step. Every node,
including the verifier, goes through these, so cost and step accounting live in one place.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel

from scout.agent.report import StepRecord
from scout.agent.state import AgentDeps
from scout.config import NodeName
from scout.llm.client import LLMError, LLMResult


async def llm_call(
    deps: AgentDeps,
    node: NodeName,
    *,
    reserve_gbp: float = 0.0,
    output_format: type[BaseModel] | None = None,
    **kwargs: Any,
) -> LLMResult:
    deps.budget.check(reserve_gbp)
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
        raise
    deps.budget.charge(result.cost_gbp)
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
    return result


async def tool_call(
    deps: AgentDeps, node: str, name: str, args: dict[str, Any]
) -> tuple[bool, Any, str]:
    """Call an MCP tool. Returns (ok, structured result, text for the model)."""
    start = time.perf_counter()
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
    deps.steps.append(
        StepRecord(
            node=node,
            kind="tool",
            name=name,
            ok=ok,
            ms=int((time.perf_counter() - start) * 1000),
            detail=None if ok else text[:300],
        )
    )
    return ok, structured, text
