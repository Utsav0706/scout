"""Tracing: every research run as an OpenTelemetry trace, exported to Langfuse.

OpenTelemetry rather than Langfuse's own SDK: it's the vendor-neutral standard (Gearbox and
Dyno will use it too, and the MCP SDK already emits OTel spans for tool calls), and
Langfuse ingests OTLP directly, so switching backends later means changing one URL.

Off unless LANGFUSE_PUBLIC_KEY/SECRET_KEY are set: with no provider configured, every span
is a free no-op. What leaves the machine: inputs/outputs with personal data masked
(safety.redact) and cut to telemetry.max_field_chars; or none at all with
telemetry.capture_content: false.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from scout.config import Secrets, TelemetryConfig
from scout.safety.redact import redact

log = logging.getLogger(__name__)

# Attributes copied onto every span of the current run (see RunAttributes).
_run_attributes: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
    "scout_run_attributes", default=()
)


class RunAttributes(SpanProcessor):
    """Langfuse filters by user and session only if EVERY span carries them; copying them
    here means no call site can forget."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        for key, value in _run_attributes.get():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass


class OnlyScoutSpans(SpanProcessor):
    """Export only spans Scout creates (tracer "scout").

    Libraries trace themselves too: the Anthropic SDK emits a span per API call and the
    MCP SDK one per tool call. Exported alongside ours they duplicated every model and tool
    call, and Langfuse priced the SDK's copies as well, doubling the cost it showed
    (found on the first live trace: 137 observations for 62 calls, cost counted twice).
    Ours already carry model, tokens, cost and inputs/outputs, with step names."""

    def __init__(self, delegate: SpanProcessor, scopes: frozenset[str] = frozenset({"scout"})):
        self._delegate = delegate
        self._scopes = scopes

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self._delegate.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        scope = span.instrumentation_scope
        if scope is not None and scope.name in self._scopes:
            self._delegate.on_end(span)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def make_provider(exporter: SpanExporter, processor_cls=BatchSpanProcessor) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "scout"}))
    provider.add_span_processor(RunAttributes())
    provider.add_span_processor(OnlyScoutSpans(processor_cls(exporter)))
    return provider


def langfuse_settings(secrets: Secrets) -> tuple[str, dict[str, str]] | None:
    """(OTLP endpoint, headers) for Langfuse Cloud, or None when its keys aren't set."""
    if not (secrets.langfuse_public_key and secrets.langfuse_secret_key):
        return None
    pair = (
        f"{secrets.langfuse_public_key.get_secret_value()}:"
        f"{secrets.langfuse_secret_key.get_secret_value()}"
    )
    endpoint = f"{secrets.langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
    headers = {
        "Authorization": "Basic " + base64.b64encode(pair.encode()).decode(),
        # Without this header, Langfuse can take up to 10 minutes to show OTel data.
        "x-langfuse-ingestion-version": "4",
    }
    return endpoint, headers


def langfuse_exporter(secrets: Secrets) -> SpanExporter | None:
    settings = langfuse_settings(secrets)
    if settings is None:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint, headers = settings
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def setup_tracing(secrets: Secrets) -> bool:
    """Install the Langfuse exporter once per process. Returns whether tracing is on."""
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return True  # already set up (or a test installed an in-memory one)
    exporter = langfuse_exporter(secrets)
    if exporter is None:
        return False
    trace.set_tracer_provider(make_provider(exporter))  # flushed on process exit by the SDK
    log.info("tracing to Langfuse", extra={"host": secrets.langfuse_host})
    return True


def tracer() -> trace.Tracer:
    return trace.get_tracer("scout")


@contextmanager
def run_scope(user_id: object, job_id: object) -> Iterator[None]:
    token = _run_attributes.set(
        (("langfuse.user.id", str(user_id)), ("langfuse.session.id", str(job_id)))
    )
    try:
        yield
    finally:
        _run_attributes.reset(token)


def clip(value: Any, max_chars: int) -> str:
    """Text for a trace or trajectory: JSON-ish, personal data masked, length capped."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = redact(text)
    return text if len(text) <= max_chars else text[:max_chars] + " …[truncated]"


def set_content(
    span: trace.Span, cfg: TelemetryConfig, *, input: Any = None, output: Any = None
) -> None:
    """Attach input/output to a span, unless content capture is switched off."""
    if not cfg.capture_content:
        return
    if input is not None:
        span.set_attribute("langfuse.observation.input", clip(input, cfg.max_field_chars))
    if output is not None:
        span.set_attribute("langfuse.observation.output", clip(output, cfg.max_field_chars))


def trace_id_of(span: trace.Span) -> str | None:
    ctx = span.get_span_context()
    return trace.format_trace_id(ctx.trace_id) if ctx.is_valid else None
