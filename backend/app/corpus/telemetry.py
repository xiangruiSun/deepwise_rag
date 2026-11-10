"""Tracing and metrics for the corpus platform.

``stage`` times a pipeline stage, feeds the shared Prometheus histogram and opens an
OpenTelemetry span when tracing is configured (a no-op tracer otherwise).  Span
attributes are bounded, privacy-safe values only — counts, durations, versions,
status — never query text or document content.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from app.observability.metrics import REGISTRY, STAGE_LATENCY, StageTimer

try:  # the API package is optional; without it spans are skipped entirely
    from opentelemetry import trace as _trace

    _tracer: Any = _trace.get_tracer("inspect_rag.corpus")
except ImportError:  # pragma: no cover
    _trace = None  # type: ignore[assignment]
    _tracer = None

QUERY_LATENCY = Histogram("corpus_query_latency_seconds", "End-to-end RAG query latency",
                          ["strategy", "mode"], registry=REGISTRY,
                          buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 30, 60))
FIRST_TOKEN_LATENCY = Histogram("corpus_first_token_seconds", "Time to the first streamed token",
                                ["backend"], registry=REGISTRY,
                                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30))
QUERIES = Counter("corpus_queries_total", "RAG queries by outcome", ["outcome"], registry=REGISTRY)
CACHE_EVENTS = Counter("corpus_cache_events_total", "Cache lookups", ["cache", "result"], registry=REGISTRY)
LLM_REQUESTS = Counter("corpus_llm_requests_total", "LLM requests by outcome", ["outcome"], registry=REGISTRY)
LLM_IN_FLIGHT = Gauge("corpus_llm_in_flight", "LLM requests currently executing", registry=REGISTRY)
LLM_QUEUE = Gauge("corpus_llm_queue_depth", "LLM requests waiting for a slot", registry=REGISTRY)
INDEX_PENDING = Gauge("corpus_index_pending_documents", "Documents the active index is behind the registry",
                      ["kind"], registry=REGISTRY)
INDEX_LAG = Gauge("corpus_index_freshness_lag_seconds", "Age of the oldest unprocessed change event",
                  registry=REGISTRY)
EVENTS = Counter("corpus_events_total", "Change events by outcome", ["outcome"], registry=REGISTRY)
BATCH_ITEMS = Counter("corpus_batch_items_total", "Batch job items by outcome", ["outcome"], registry=REGISTRY)
ACCESS_FILTERED = Counter("corpus_access_denied_total", "Requests for documents outside the caller's scope",
                          ["reason"], registry=REGISTRY)


def new_trace_id() -> str:
    """Trace id of the current span when tracing is on, a random id otherwise."""
    if _trace is not None:
        context = _trace.get_current_span().get_span_context()
        if context.is_valid:
            return f"{context.trace_id:032x}"
    return uuid.uuid4().hex


@contextmanager
def stage(timer: StageTimer, name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Time ``name`` and record a span.  Values added to the yielded dict become span attributes."""
    extra: dict[str, Any] = {}
    if _tracer is None:
        with timer.stage(name):
            yield extra
        return
    with _tracer.start_as_current_span(f"corpus.{name}") as span, timer.stage(name):
        for key, value in attributes.items():
            span.set_attribute(f"corpus.{key}", value)
        try:
            yield extra
        except Exception as exc:
            span.set_attribute("corpus.error", type(exc).__name__)
            raise
        finally:
            for key, value in extra.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"corpus.{key}", value)


def record_stage(timer: StageTimer, name: str, started: float, **attributes: Any) -> None:
    """Record a stage that was timed by hand (``started`` from ``time.perf_counter()``).

    Used where a stage spans ``yield`` points of an async generator: a *current* span must not
    be held across a suspension, so the span is created after the fact with explicit times."""
    import time

    elapsed = time.perf_counter() - started
    timer.stages[name] = timer.stages.get(name, 0.0) + elapsed * 1000.0
    STAGE_LATENCY.labels(stage=name).observe(elapsed)
    if _tracer is not None:
        end_ns = time.time_ns()
        span = _tracer.start_span(f"corpus.{name}", start_time=end_ns - int(elapsed * 1e9))
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"corpus.{key}", value)
        span.end(end_time=end_ns)
