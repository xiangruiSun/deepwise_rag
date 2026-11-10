"""Search, comparison, RAG, experiment, admin and health endpoints."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response
from sqlalchemy import select, text

from app.api.deps import Admin, Ctx, Pagination, Policy, Researcher, run_bounded
from app.core.container import AppContext, build_indexes, run_ingestion
from app.core.errors import AuthorizationError, NotFoundError, ValidationFailure
from app.core.security import AccessPolicy
from app.db.base import session_scope
from app.db.models import AuditEvent, ExperimentRun, IndexManifest
from app.observability.metrics import ACCESS_DENIED, render_metrics
from app.schemas.cases import (
    ExperimentDetail,
    ExperimentResults,
    ExperimentRunOut,
    IndexStatus,
    IngestionJob,
    IngestionRequest,
)
from app.schemas.rag import CompareRequest, CompareResponse, RagAnswer, RagRequest
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter()


def _audit(ctx: AppContext, actor: str, action: str, resource_type: str, resource_id: str | None,
           details: dict[str, Any]) -> None:
    """Audit entries hold identifiers and counts only — never query or clinical text."""
    with session_scope(ctx.session_factory) as session:
        session.add(AuditEvent(actor=actor, action=action, resource_type=resource_type,
                               resource_id=resource_id, details=details))


# ------------------------------------------------------------------- health
@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", tags=["health"])
async def ready(ctx: Ctx, response: Response) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        with ctx.session_factory() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    checks["search"] = "ok" if ctx.backend.ping() else "error: unreachable"
    cases_index = ctx.config.index.cases_index
    checks["case_index"] = "ok" if ctx.backend.index_exists(cases_index) else "missing (run indexing)"
    recipe = ctx.config.recipes[ctx.config.default_recipe]
    if recipe.embedding and "dense" in recipe.branches:
        checks["embedding_model"] = "loaded" if ctx.embeddings.is_loaded(recipe.embedding) else "lazy (not loaded yet)"
    checks["generation_backend"] = ctx.config.generation.backend
    ok = all(str(v).startswith(("ok", "loaded", "lazy")) for k, v in checks.items() if k != "generation_backend")
    if not ok:
        response.status_code = 503
    return {"status": "ready" if ok else "not_ready", "checks": checks}


@router.get("/metrics", tags=["health"], include_in_schema=False)
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


# ------------------------------------------------------------------- search
@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchRequest, ctx: Ctx, policy: Policy, principal: Researcher) -> SearchResponse:
    response = await run_bounded(ctx, ctx.retrieval.search, request, policy)
    _audit(ctx, principal.user_id, "search", "query", response.query_id,
           {"mode": request.query_mode, "recipe": response.recipe_id, "results": len(response.results),
            "eligible": response.total_eligible_cases})
    return response


@router.post("/compare", response_model=CompareResponse, tags=["search"])
async def compare(request: CompareRequest, ctx: Ctx, policy: Policy, principal: Researcher) -> CompareResponse:
    response = await run_bounded(ctx, ctx.comparison.compare, request, policy)
    _audit(ctx, principal.user_id, "compare", "cases", None, {"n_cases": len(request.case_ids)})
    return response


@router.post("/rag/answer", response_model=RagAnswer, tags=["rag"])
async def rag_answer(request: RagRequest, ctx: Ctx, policy: Policy, principal: Researcher) -> RagAnswer:
    answer = await run_bounded(ctx, ctx.rag.answer, request, policy)
    _audit(ctx, principal.user_id, "rag.answer", "answer", None,
           {"status": answer.status, "n_claims": len(answer.claims()), "n_citations": len(answer.citations),
            "generation_key": answer.generation_metadata.get("generation_key")})
    return answer


# -------------------------------------------------------------- experiments
def _run_out(run: ExperimentRun) -> ExperimentRunOut:
    metrics = {m.name: m.value for m in run.metrics if m.group in ("overall", "performance")}
    summary = {"recipe": (run.config or {}).get("recipe", {}), "generation": (run.config or {}).get("generation")}
    return ExperimentRunOut(
        run_id=run.run_id, suite_id=run.suite_id, experiment_id=run.experiment_id, status=run.status,
        stage=run.stage, data_kind=run.data_kind, config_hash=run.config_hash,
        dataset_fingerprint=run.dataset_fingerprint, started_at=run.started_at, finished_at=run.finished_at,
        metrics=metrics, summary=summary,
    )


def _list_experiments(ctx: AppContext, limit: int, offset: int, suite_id: str | None) -> list[ExperimentRunOut]:
    with ctx.session_factory() as session:
        stmt = select(ExperimentRun).order_by(ExperimentRun.started_at.desc(), ExperimentRun.run_id)
        if suite_id:
            stmt = stmt.where(ExperimentRun.suite_id == suite_id)
        return [_run_out(r) for r in session.scalars(stmt.limit(limit).offset(offset))]


@router.get("/experiments", response_model=list[ExperimentRunOut], tags=["experiments"])
async def list_experiments(ctx: Ctx, _: Researcher, page: Pagination,
                           suite_id: Annotated[str | None, Query(max_length=64)] = None) -> list[ExperimentRunOut]:
    return await run_bounded(ctx, _list_experiments, ctx, page.limit, page.offset, suite_id)


def _experiment_detail(ctx: AppContext, run_id: str) -> ExperimentDetail:
    with ctx.session_factory() as session:
        run = session.get(ExperimentRun, run_id)
        if run is None:
            raise NotFoundError("experiment run not found", {"run_id": run_id})
        base = _run_out(run).model_dump()
        groups: dict[str, dict[str, float | None]] = {}
        for metric in run.metrics:
            groups.setdefault(metric.group, {})[metric.name] = metric.value
        metrics_path = next((Path(a.path) for a in run.artifacts if a.kind == "metrics"), None)
        if metrics_path and metrics_path.exists():
            base["summary"] = json.loads(metrics_path.read_text(encoding="utf-8")).get("summary", base["summary"])
        return ExperimentDetail(
            **base, config=run.config, model_revisions=run.model_revisions, index_versions=run.index_versions,
            hardware=run.hardware, git_commit=run.git_commit, query_set_version=run.query_set_version,
            qrels_version=run.qrels_version, metrics_by_group=groups, artifacts=[a.kind for a in run.artifacts],
        )


@router.get("/experiments/{run_id}", response_model=ExperimentDetail, tags=["experiments"])
async def get_experiment(run_id: str, ctx: Ctx, _: Researcher) -> ExperimentDetail:
    return await run_bounded(ctx, _experiment_detail, ctx, run_id)


def _experiment_results(ctx: AppContext, policy: AccessPolicy, run_id: str, limit: int, offset: int) -> ExperimentResults:
    with ctx.session_factory() as session:
        run = session.get(ExperimentRun, run_id)
        if run is None:
            raise NotFoundError("experiment run not found", {"run_id": run_id})
        artifact = next((a for a in run.artifacts if a.kind == "per_query"), None)
        if artifact is None:
            raise NotFoundError("this run has no per-query results")
        # per-query artifacts contain case identifiers: restricted to the scope they were produced under
        if not policy.allows(artifact.access_scope):
            ACCESS_DENIED.labels(resource="experiment_artifact").inc()
            raise AuthorizationError("access to this experiment artifact is not permitted")
        path = Path(artifact.path)
    if not path.exists():
        raise NotFoundError("per-query result file is no longer available")
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if offset <= total < offset + limit:
                rows.append(json.loads(line))
            total += 1
    return ExperimentResults(run_id=run_id, total=total, limit=limit, offset=offset, rows=rows)


@router.get("/experiments/{run_id}/results", response_model=ExperimentResults, tags=["experiments"])
async def get_experiment_results(run_id: str, ctx: Ctx, policy: Policy, _: Researcher,
                                 page: Pagination) -> ExperimentResults:
    return await run_bounded(ctx, _experiment_results, ctx, policy, run_id, page.limit, page.offset)


# -------------------------------------------------------------------- admin
_JOBS: dict[str, IngestionJob] = {}
_JOB_LOCK = threading.Lock()


def _ingestion_job(ctx: AppContext, job_id: str, request: IngestionRequest, actor: str) -> None:
    job = _JOBS[job_id]
    try:
        job.status = "running"
        report = run_ingestion(ctx, force=request.force, actor=actor)
        summary: dict[str, Any] = {"ingestion": report.summary()}
        if request.rebuild_indexes:
            summary["indexes"] = build_indexes(ctx, actor=actor)
        job.summary, job.status = json.loads(json.dumps(summary, default=str)), "completed"
    except Exception as exc:  # the job must always reach a terminal state
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        _JOB_LOCK.release()


@router.post("/admin/ingestion", response_model=IngestionJob, status_code=202, tags=["admin"])
async def start_ingestion(request: IngestionRequest, ctx: Ctx, principal: Admin) -> IngestionJob:
    if not _JOB_LOCK.acquire(blocking=False):
        raise ValidationFailure("an ingestion job is already running")
    job = IngestionJob(job_id=str(uuid.uuid4()), status="queued", submitted_by=principal.user_id)
    _JOBS[job.job_id] = job
    _audit(ctx, principal.user_id, "ingestion.requested", "dataset", ctx.config.dataset.name,
           {"force": request.force, "rebuild_indexes": request.rebuild_indexes})
    threading.Thread(target=_ingestion_job, args=(ctx, job.job_id, request, principal.user_id), daemon=True).start()
    return job


@router.get("/admin/ingestion/{job_id}", response_model=IngestionJob, tags=["admin"])
async def ingestion_status(job_id: str, _: Admin) -> IngestionJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise NotFoundError("ingestion job not found", {"job_id": job_id})
    return job


def _index_status(ctx: AppContext) -> list[IndexStatus]:
    with ctx.session_factory() as session:
        manifests = {m.index_name: m for m in session.scalars(select(IndexManifest))}
    out: list[IndexStatus] = []
    for name in sorted({*manifests, *ctx.backend.list_indexes()}):
        manifest = manifests.get(name)
        exists = ctx.backend.index_exists(name)
        stats = ctx.backend.stats(name) if exists else {}
        out.append(IndexStatus(
            index_name=name, kind=manifest.kind if manifest else "unknown",
            alias=manifest.alias if manifest else None, exists=exists,
            embedding_key=manifest.embedding_key if manifest else None,
            embedding_model_id=manifest.embedding_model_id if manifest else None,
            embedding_revision=manifest.embedding_revision if manifest else None,
            dimension=stats.get("dimension") if exists else (manifest.dimension if manifest else None),
            chunking_key=manifest.chunking_key if manifest else None,
            mapping_version=manifest.mapping_version if manifest else None,
            document_count=stats.get("documents"), size_bytes=stats.get("size_bytes"),
            health=stats.get("health"), built_at=manifest.built_at if manifest else None,
        ))
    return out


@router.get("/admin/indexes", response_model=list[IndexStatus], tags=["admin"])
async def index_status(ctx: Ctx, _: Admin) -> list[IndexStatus]:
    return await run_bounded(ctx, _index_status, ctx)
