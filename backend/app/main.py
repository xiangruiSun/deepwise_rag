"""FastAPI application startup."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import cases, routes, v1
from app.core.config import get_settings
from app.core.container import AppContext, build_context
from app.core.errors import AppError
from app.observability.metrics import ERRORS, REQUESTS, configure_logging, get_logger, setup_tracing
from app.schemas.search import RESEARCH_NOTICE

log = get_logger("api")


def create_app(ctx: AppContext | None = None) -> FastAPI:
    settings = ctx.settings if ctx else get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = ctx or build_context(settings)
        log.info("started: dataset=%s backend=%s", app.state.ctx.config.dataset.name, app.state.ctx.backend.name)
        yield
        if ctx is None:  # an injected context is owned (and disposed) by its creator
            app.state.ctx.engine.dispose()

    app = FastAPI(
        title="INSPECT-RAG",
        version="0.1.0",
        description="Evaluation-driven multimodal clinical case retrieval. " + RESEARCH_NOTICE,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    @app.middleware("http")
    async def _count_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        response.headers["X-Research-Use-Only"] = "true"
        return response

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        ERRORS.labels(code=exc.code).inc()
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        ERRORS.labels(code="validation_error").inc()
        # only locations and messages are returned: request bodies may contain clinical text
        details = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "error": {"code": "validation_error", "message": "request validation failed", "details": {"errors": details}}
        })

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        ERRORS.labels(code="internal_error").inc()
        log.error("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={
            "error": {"code": "internal_error", "message": "internal server error", "details": {}}
        })

    app.include_router(routes.router, prefix="/api/v1")
    app.include_router(cases.router, prefix="/api/v1")
    # document-corpus platform: tenant-aware RAG over versioned hybrid indexes
    app.include_router(v1.router, prefix="/v1")
    app.include_router(v1.health_router)
    setup_tracing(app, settings.otel_enabled)
    return app


app = create_app()
