"""Case endpoints.  Every lookup is authorized server-side before any content is returned."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import Ctx, Pagination, Policy, Researcher, run_bounded
from app.core.container import AppContext
from app.core.errors import NotFoundError
from app.core.security import AccessPolicy
from app.db.models import ClinicalCase, ClinicalEvent, ImagingStudy, Patient, RadiologyReport
from app.imaging.dicom import ImagingError, load_volume
from app.observability.metrics import ACCESS_DENIED
from app.retrieval.features import CaseFeatureService
from app.schemas.cases import (
    CaseListResponse,
    CaseOverview,
    CaseSummary,
    ImageOut,
    MetaResponse,
    ReportOut,
    TimelineEvent,
    TimelineResponse,
)

router = APIRouter(tags=["cases"])


def load_authorized_case(session: Session, case_id: str, policy: AccessPolicy, resource: str) -> ClinicalCase:
    case = session.get(ClinicalCase, case_id)
    if case is None:
        raise NotFoundError("case not found", {"case_id": case_id})
    if not policy.allows(case.access_scope):
        ACCESS_DENIED.labels(resource=resource).inc()
    policy.assert_case_allowed(case.access_scope, case_id)
    return case


def _list_cases(ctx: AppContext, policy: AccessPolicy, limit: int, offset: int, split: str | None) -> CaseListResponse:
    with ctx.session_factory() as session:
        where: list[Any] = [ClinicalCase.access_scope.in_(policy.allowed_scopes)]
        if split:
            where.append(ClinicalCase.dataset_split == split)
        total = session.scalar(select(func.count()).select_from(ClinicalCase).where(*where)) or 0
        rows = session.execute(
            select(ClinicalCase, Patient).join(Patient, Patient.patient_id == ClinicalCase.patient_id)
            .where(*where).order_by(ClinicalCase.case_id).limit(limit).offset(offset)
        ).all()
        case_ids = [c.case_id for c, _ in rows]
        with_report = set(session.scalars(select(RadiologyReport.case_id).where(RadiologyReport.case_id.in_(case_ids))))
        with_image = set(session.scalars(
            select(ImagingStudy.case_id).where(ImagingStudy.case_id.in_(case_ids),
                                               ImagingStudy.feature_status != "missing_source")))
        items = [
            CaseSummary(case_id=c.case_id, age=c.age_at_procedure if c.age_at_procedure is not None else p.age,
                        gender=p.gender, dataset_split=c.dataset_split, has_report=c.case_id in with_report,
                        has_image=c.case_id in with_image)
            for c, p in rows
        ]
    return CaseListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get("/cases", response_model=CaseListResponse)
async def list_cases(ctx: Ctx, policy: Policy, _: Researcher, page: Pagination,
                     split: Annotated[str | None, Query(pattern="^(train|valid|test)$")] = None) -> CaseListResponse:
    return await run_bounded(ctx, _list_cases, ctx, policy, page.limit, page.offset, split)


def _overview(ctx: AppContext, policy: AccessPolicy, case_id: str) -> CaseOverview:
    with ctx.session_factory() as session:
        case = load_authorized_case(session, case_id, policy, "case")
        features = CaseFeatureService(session)
        context = features.contexts([case_id], ctx.config.dataset.default_lookback_days)[case_id]
        before = session.scalar(select(func.count()).select_from(ClinicalEvent).where(
            ClinicalEvent.patient_id == case.patient_id, ClinicalEvent.event_time <= case.procedure_time)) or 0
        after = session.scalar(select(func.count()).select_from(ClinicalEvent).where(
            ClinicalEvent.patient_id == case.patient_id, ClinicalEvent.event_time > case.procedure_time)) or 0
        labs = context.labs
        return CaseOverview(
            case_id=case_id, age=context.age,
            gender=context.patient.gender if context.patient else None,
            race=context.patient.race if context.patient else None,
            ethnicity=context.patient.ethnicity if context.patient else None,
            dataset_split=case.dataset_split, source_version=case.source_version,
            report_ids=[r.report_id for r in context.reports],
            study_ids=[s.study_id for s in context.studies],
            n_events_before_procedure=before, n_events_after_procedure=after,
            laboratory=[f.as_dict() for _, f in sorted(labs.features.items())] if labs else [],
            laboratory_unavailable=dict(labs.unavailable) if labs else {},
            diagnosis_codes=sorted(context.diagnosis_codes),
            modality_coverage={
                "text": bool(context.reports), "laboratory": bool(labs and labs.features),
                "image": any(s.feature_status == "ready" for s in context.studies),
                "diagnosis": bool(context.diagnosis_codes),
            },
        )


@router.get("/cases/{case_id}", response_model=CaseOverview)
async def get_case(case_id: str, ctx: Ctx, policy: Policy, _: Researcher) -> CaseOverview:
    return await run_bounded(ctx, _overview, ctx, policy, case_id)


def _timeline(ctx: AppContext, policy: AccessPolicy, case_id: str, limit: int, offset: int,
              event_type: str | None) -> TimelineResponse:
    with ctx.session_factory() as session:
        case = load_authorized_case(session, case_id, policy, "timeline")
        where = [ClinicalEvent.patient_id == case.patient_id]
        if event_type:
            where.append(ClinicalEvent.event_type == event_type)
        total = session.scalar(select(func.count()).select_from(ClinicalEvent).where(*where)) or 0
        events = session.scalars(
            select(ClinicalEvent).where(*where).order_by(ClinicalEvent.event_time, ClinicalEvent.event_id)
            .limit(limit).offset(offset)
        )
        out = []
        for e in events:
            hours = (e.event_time - case.procedure_time).total_seconds() / 3600.0
            out.append(TimelineEvent(
                event_id=e.event_id, event_type=e.event_type, concept_code=e.concept_code,
                code_system=e.code_system, display_name=e.display_name, numeric_value=e.numeric_value,
                text_value=e.text_value, unit=e.unit, hours_from_procedure=round(hours, 2),
                phase="before_procedure" if hours <= 0 else "follow_up",
            ))
    return TimelineResponse(case_id=case_id, total=total, limit=limit, offset=offset, events=out)


@router.get("/cases/{case_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    case_id: str, ctx: Ctx, policy: Policy, _: Researcher, page: Pagination,
    event_type: Annotated[str | None, Query(pattern="^(diagnosis|laboratory|vital|medication|procedure|visit|other)$")] = None,
) -> TimelineResponse:
    return await run_bounded(ctx, _timeline, ctx, policy, case_id, page.limit, page.offset, event_type)


def _reports(ctx: AppContext, policy: AccessPolicy, case_id: str) -> list[ReportOut]:
    with ctx.session_factory() as session:
        load_authorized_case(session, case_id, policy, "reports")
        reports = session.scalars(
            select(RadiologyReport).where(RadiologyReport.case_id == case_id).order_by(RadiologyReport.report_id)
        )
        return [ReportOut(report_id=r.report_id, case_id=r.case_id, study_id=r.study_id,
                          impression_text=r.impression_text, content_hash=r.content_hash) for r in reports]


@router.get("/cases/{case_id}/reports", response_model=list[ReportOut])
async def get_reports(case_id: str, ctx: Ctx, policy: Policy, _: Researcher) -> list[ReportOut]:
    return await run_bounded(ctx, _reports, ctx, policy, case_id)


def _images(ctx: AppContext, policy: AccessPolicy, case_id: str) -> list[ImageOut]:
    with ctx.session_factory() as session:
        load_authorized_case(session, case_id, policy, "images")
        studies = session.scalars(
            select(ImagingStudy).where(ImagingStudy.case_id == case_id).order_by(ImagingStudy.study_id)
        )
        # storage paths are never exposed; clients receive opaque study ids only
        return [ImageOut(study_id=s.study_id, case_id=s.case_id, modality=s.modality, slice_count=s.slice_count,
                         spacing=s.spacing, feature_status=s.feature_status,
                         preview_available=s.feature_status in ("ready", "pending") and bool(s.source_path))
                for s in studies]


@router.get("/cases/{case_id}/images", response_model=list[ImageOut])
async def get_images(case_id: str, ctx: Ctx, policy: Policy, _: Researcher) -> list[ImageOut]:
    return await run_bounded(ctx, _images, ctx, policy, case_id)


def encode_png_gray(image: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG encoder (no imaging dependency needed)."""
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _preview(ctx: AppContext, policy: AccessPolicy, case_id: str, study_id: str, slice_index: int | None) -> bytes:
    with ctx.session_factory() as session:
        load_authorized_case(session, case_id, policy, "image_preview")
        study = session.get(ImagingStudy, study_id)
        if study is None or study.case_id != case_id:
            raise NotFoundError("imaging study not found for this case", {"study_id": study_id})
        if not study.source_path or study.feature_status in ("missing_source", "rejected"):
            raise NotFoundError("no preview is available for this study", {"study_id": study_id})
        path, spacing = Path(study.source_path), study.spacing
    try:
        volume = load_volume(path, spacing)
    except ImagingError as exc:
        raise NotFoundError(f"no preview is available: {exc}") from exc
    index = volume.slice_count // 2 if slice_index is None else max(0, min(slice_index, volume.slice_count - 1))
    lo, hi = ctx.config.image.hu_window
    window = np.clip((volume.hu[index] - lo) / (hi - lo), 0.0, 1.0)
    return encode_png_gray((window * 255).astype(np.uint8))


@router.get("/cases/{case_id}/images/{study_id}/preview", responses={200: {"content": {"image/png": {}}}})
async def get_image_preview(case_id: str, study_id: str, ctx: Ctx, policy: Policy, _: Researcher,
                            slice_index: Annotated[int | None, Query(ge=0, le=5000)] = None) -> Response:
    data = await run_bounded(ctx, _preview, ctx, policy, case_id, study_id, slice_index)
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})


def _meta(ctx: AppContext, policy: AccessPolicy) -> MetaResponse:
    from app.ingestion.normalization import LabConceptRegistry

    scope = policy.allowed_scopes
    with ctx.session_factory() as session:
        genders = sorted(g for g in session.scalars(
            select(Patient.gender).join(ClinicalCase, ClinicalCase.patient_id == Patient.patient_id)
            .where(ClinicalCase.access_scope.in_(scope)).distinct()) if g)
        codes = sorted(session.scalars(
            select(ClinicalEvent.concept_code).where(ClinicalEvent.event_type == "diagnosis").distinct().limit(500)))
        modalities = sorted(m for m in session.scalars(select(ImagingStudy.modality).distinct()) if m)
        ages = session.execute(
            select(func.min(ClinicalCase.age_at_procedure), func.max(ClinicalCase.age_at_procedure))
            .where(ClinicalCase.access_scope.in_(scope))).one()
        lab_codes = set(session.scalars(
            select(ClinicalEvent.concept_code).where(ClinicalEvent.event_type.in_(("laboratory", "vital"))).distinct()))
    registry = LabConceptRegistry()
    labs = [{"key": c.key, "display_name": c.display_name, "unit": c.canonical_unit}
            for c in registry.concepts() if lab_codes.intersection(c.codes)]
    return MetaResponse(
        dataset=ctx.config.dataset.name, data_kind=ctx.data_kind,
        text_label=ctx.config.dataset.text_label, genders=genders, laboratory_concepts=labs,
        diagnosis_codes=codes, study_modalities=modalities,
        age_range=[float(ages[0]), float(ages[1])] if ages[0] is not None and ages[1] is not None else None,
        recipes=[{"recipe_id": r.recipe_id, "description": r.description,
                  **ctx.retrieval._recipe_summary(r)} for r in ctx.config.recipes.values()],
        default_recipe=ctx.config.default_recipe, generation_backend=ctx.config.generation.backend,
        image_modality_enabled=ctx.image_provider.enabled,
    )


@router.get("/meta", response_model=MetaResponse)
async def get_meta(ctx: Ctx, policy: Policy, _: Researcher) -> MetaResponse:
    return await run_bounded(ctx, _meta, ctx, policy)
