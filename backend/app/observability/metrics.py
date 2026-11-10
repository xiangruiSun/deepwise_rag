"""Latency instrumentation, Prometheus metrics, logging and optional tracing."""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()
STAGE_LATENCY = Histogram(
    "inspect_rag_stage_latency_seconds", "Latency of a pipeline stage", ["stage"], registry=REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
REQUESTS = Counter("inspect_rag_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY)
ERRORS = Counter("inspect_rag_errors_total", "Errors by code", ["code"], registry=REGISTRY)
INFERENCE_TOKENS = Counter("inspect_rag_llm_tokens_total", "LLM tokens", ["direction"], registry=REGISTRY)
ACCESS_DENIED = Counter("inspect_rag_access_denied_total", "Denied access attempts", ["resource"],
                        registry=REGISTRY)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


class StageTimer:
    """Collects per-stage wall-clock latency (milliseconds) for one request."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.stages[name] = self.stages.get(name, 0.0) + elapsed * 1000.0
            STAGE_LATENCY.labels(stage=name).observe(elapsed)

    def finish(self) -> dict[str, float]:
        total = (time.perf_counter() - self._start) * 1000.0
        out = {k: round(v, 3) for k, v in self.stages.items()}
        out["total"] = round(total, 3)
        return out


class _RedactingFilter(logging.Filter):
    """Never log clinical text: only whitelisted ``extra`` keys are rendered."""

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "clinical_text"):
            record.clinical_text = "[redacted]"
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(_RedactingFilter())
    root = logging.getLogger("inspect_rag")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"inspect_rag.{name}")


def setup_tracing(app: Any, enabled: bool) -> bool:
    """Enable OpenTelemetry tracing when requested and the SDK is installed."""
    if not enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        get_logger("observability").warning("OpenTelemetry requested but not installed")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": "inspect-rag-backend"}))
    try:  # OTLP exporter is optional
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    except ImportError:
        pass
    trace.set_tracer_provider(provider)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass
    return True


def hardware_description() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def process_memory_mb() -> float | None:
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)
    except Exception:  # pragma: no cover
        return None


def gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except ImportError:
        pass
    return None
