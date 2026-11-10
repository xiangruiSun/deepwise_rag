"""``/v1`` — the document-corpus RAG API (online query, streaming, retrieval, feedback).

The tenant and the permission set of a request come from the authenticated
credential.  No endpoint accepts a tenant identifier from the client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import Ctx, CurrentPrincipal, Researcher
from app.core.errors import AppError, ConfigurationError, NotFoundError
from app.corpus.retrieval import AuthContext
from app.corpus.runtime import CorpusRuntime, build_corpus_runtime
from app.corpus.service import QueryRequest
from app.db.models import AuditEvent, CorpusFeedback
from app.observability.metrics import StageTimer, get_logger

router = APIRouter(tags=["corpus"])
log = get_logger("api.v1")


def get_corpus(request: Request, ctx: Ctx) -> CorpusRuntime:
    """The corpus runtime, built on first use from ``configs/corpus.yaml``."""
    runtime = getattr(request.app.state, "corpus", None)
    if runtime is None:
        try:
            runtime = build_corpus_runtime(ctx.settings, ctx.session_factory, ctx.backend, rerankers=ctx.rerankers)
        except FileNotFoundError as exc:
            raise ConfigurationError("the corpus platform is not configured (corpus config file missing)") from exc
        request.app.state.corpus = runtime
    return runtime  # type: ignore[no-any-return]


Corpus = Annotated[CorpusRuntime, Depends(get_corpus)]


def _auth(principal: CurrentPrincipal) -> AuthContext:
    return AuthContext.from_principal(principal)


Auth = Annotated[AuthContext, Depends(_auth)]


def _strategy_or_404(corpus: CorpusRuntime, name: str | None) -> None:
    try:
        corpus.config.strategy(name)
    except KeyError as exc:
        raise NotFoundError(f"unknown retrieval strategy '{name}'",
                            {"available": ["default", *sorted(corpus.config.strategies)]}) from exc


def _audit(corpus: CorpusRuntime, auth: AuthContext, action: str, details: dict[str, Any]) -> None:
    """Audit record: identifiers and counts only — never the question or any document text."""
    try:
        with corpus.session_factory() as session:
            session.add(AuditEvent(actor=auth.user_id, action=action, resource_type="corpus",
                                   resource_id=auth.tenant_id, outcome="ok", details=details))
            session.commit()
    except Exception as exc:  # auditing must not break serving; the failure itself is logged
        log.error("audit write failed: %s", type(exc).__name__)


# ------------------------------------------------------------------- query
@router.post("/query")
async def query(body: QueryRequest, corpus: Corpus, auth: Auth, ctx: Ctx) -> dict[str, Any]:
    """Synchronous grounded answer with structured citations."""
    _strategy_or_404(corpus, body.strategy)
    result = await corpus.service.answer(auth, body, timeout=ctx.settings.request_timeout_seconds)
    _audit(corpus, auth, "corpus.query", {"trace_id": result["trace_id"], "strategy": result["strategy"],
                                         "abstained": result["abstained"], "citations": len(result["citations"]),
                                         "index_version": result["index_version"]})
    return result


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request, corpus: Corpus, auth: Auth) -> StreamingResponse:
    """Server-sent events: ``retrieval``, ``token`` (one verified sentence each), ``citations``,
    ``done`` — or ``error``.  Closing the connection cancels generation."""
    _strategy_or_404(corpus, body.strategy)

    async def stream() -> AsyncIterator[str]:
        events = corpus.service.events(auth, body)
        try:
            async for event in events:
                if await request.is_disconnected():
                    break  # the client went away: stop retrieval/generation work now
                yield _sse(event["event"], event["data"])
                if event["event"] == "done":
                    data = event["data"]
                    _audit(corpus, auth, "corpus.query_stream", {
                        "trace_id": data["trace_id"], "strategy": data["strategy"],
                        "abstained": data["abstained"], "citations": len(data["citations"]),
                        "index_version": data["index_version"]})
        except AppError as exc:
            yield _sse("error", {"code": exc.code, "message": exc.message})
        except Exception as exc:
            log.error("stream failed: %s", type(exc).__name__)
            yield _sse("error", {"code": "internal_error", "message": "internal server error"})
        finally:
            await events.aclose()  # type: ignore[attr-defined]

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/retrieve")
async def retrieve(body: QueryRequest, corpus: Corpus, auth: Auth, _role: Researcher, ctx: Ctx) -> dict[str, Any]:
    """Authorised retrieval without generation (debugging and evaluation)."""
    from app.api.deps import run_bounded

    _strategy_or_404(corpus, body.strategy)
    timer = StageTimer()
    result = await run_bounded(ctx, corpus.service.retrieve, auth, body, timer)
    return {"index_version": result.index_version, "strategy": result.strategy, "plan": result.plan.as_dict(),
            "candidates": result.candidates, "cache_hit": result.cache_hit, "warnings": result.warnings,
            "timings_ms": {k: round(v, 3) for k, v in timer.stages.items()},
            "results": [c.public() for c in result.chunks[: body.top_k or result.params.final_k]]}


# ---------------------------------------------------------------- feedback
class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=8, max_length=64)
    rating: Literal["up", "down"]
    categories: list[Literal["incorrect_answer", "missing_citation", "irrelevant_documents", "slow",
                             "empty_result", "outdated", "other"]] = Field(default_factory=list, max_length=7)
    chunk_ids: list[str] = Field(default_factory=list, max_length=50)
    query_hash: str | None = Field(default=None, max_length=64)
    strategy: str | None = Field(default=None, max_length=48)
    index_version: str | None = Field(default=None, max_length=128)
    latency_ms: float | None = Field(default=None, ge=0)


@router.post("/feedback", status_code=201)
async def feedback(body: FeedbackRequest, corpus: Corpus, auth: Auth) -> dict[str, Any]:
    """Store privacy-safe feedback: ratings, categories and identifiers — no free text."""
    with corpus.session_factory() as session:
        row = CorpusFeedback(tenant_id=auth.tenant_id, user_id=auth.user_id, trace_id=body.trace_id,
                             query_hash=body.query_hash, rating=body.rating, categories=list(body.categories),
                             chunk_ids=list(body.chunk_ids), strategy=body.strategy,
                             index_version=body.index_version, latency_ms=body.latency_ms)
        session.add(row)
        session.commit()
        return {"id": row.id, "status": "recorded"}


# ------------------------------------------------------------------ health
health_router = APIRouter(tags=["health"])


@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is running."""
    return {"status": "ok"}


@health_router.get("/readyz")
async def readyz(request: Request, ctx: Ctx) -> Any:
    """Readiness: required dependencies answer.  Optional ones (cache, LLM) only degrade."""
    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    checks: dict[str, Any] = {}
    try:
        with ctx.session_factory() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    try:
        corpus = get_corpus(request, ctx)
        report = await corpus.service.readiness()
        checks.update(report["checks"])
        ready = report["status"] == "ready" and checks["database"] == "ok"
        body = {"status": "ready" if ready else "not_ready", "checks": checks,
                "active_index": report["active_index"], "model_version": report["model_version"],
                "generation_backend": report["generation_backend"]}
    except AppError as exc:
        checks["corpus"] = f"error: {exc.code}"
        ready, body = False, {"status": "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if ready else 503, content=body)
