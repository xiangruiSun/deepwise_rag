"""Case-level feature access with temporal validity.

``CaseFeatureService`` is the single place where clinical events are turned
into case features.  Every method takes an explicit cutoff: for a query case it
is ``min(procedure_time, as_of)``; for a historical candidate it is that
candidate's own procedure time.  Records after the cutoff are returned only by
``follow_up_events`` and are always labelled as follow-up information.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    ClinicalCase,
    ClinicalEvent,
    ImageFeature,
    ImagingStudy,
    LabNormalizationStat,
    Patient,
    RadiologyReport,
)
from app.ingestion.normalization import LabConceptRegistry
from app.retrieval.laboratory import (
    CaseLabFeatures,
    LaboratoryFeatureExtractor,
    RobustStat,
    fit_robust_statistics,
)


@dataclass
class CaseContext:
    case: ClinicalCase
    patient: Patient | None
    reports: list[RadiologyReport] = field(default_factory=list)
    studies: list[ImagingStudy] = field(default_factory=list)
    diagnosis_codes: set[str] = field(default_factory=set)  # recorded at or before the cutoff
    diagnosis_events: list[ClinicalEvent] = field(default_factory=list)
    labs: CaseLabFeatures | None = None
    cutoff: datetime | None = None

    @property
    def age(self) -> float | None:
        if self.case.age_at_procedure is not None:
            return self.case.age_at_procedure
        return self.patient.age if self.patient else None


class CaseFeatureService:
    def __init__(self, session: Session, registry: LabConceptRegistry | None = None) -> None:
        self.session = session
        self.registry = registry or LabConceptRegistry()
        self.extractor = LaboratoryFeatureExtractor(self.registry)

    # ------------------------------------------------------------------ loading
    def get_cases(self, case_ids: list[str]) -> dict[str, ClinicalCase]:
        if not case_ids:
            return {}
        rows = self.session.scalars(select(ClinicalCase).where(ClinicalCase.case_id.in_(case_ids)))
        return {c.case_id: c for c in rows}

    def _events_by_patient(self, patient_ids: set[str], types: tuple[str, ...]) -> dict[str, list[ClinicalEvent]]:
        out: dict[str, list[ClinicalEvent]] = defaultdict(list)
        ids = sorted(patient_ids)
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            stmt = (
                select(ClinicalEvent)
                .where(ClinicalEvent.patient_id.in_(chunk), ClinicalEvent.event_type.in_(types))
                .order_by(ClinicalEvent.event_time, ClinicalEvent.event_id)
            )
            for ev in self.session.scalars(stmt):
                out[ev.patient_id].append(ev)
        return out

    def contexts(
        self,
        case_ids: list[str],
        lookback_days: int,
        aggregate: str = "last",
        as_of: datetime | None = None,
        with_reports: bool = True,
    ) -> dict[str, CaseContext]:
        cases = self.get_cases(case_ids)
        if not cases:
            return {}
        patient_ids = {c.patient_id for c in cases.values()}
        patients = {
            p.patient_id: p
            for p in self.session.scalars(select(Patient).where(Patient.patient_id.in_(patient_ids)))
        }
        events = self._events_by_patient(patient_ids, ("laboratory", "vital", "diagnosis"))
        reports: dict[str, list[RadiologyReport]] = defaultdict(list)
        studies: dict[str, list[ImagingStudy]] = defaultdict(list)
        if with_reports:
            for r in self.session.scalars(
                select(RadiologyReport).where(RadiologyReport.case_id.in_(list(cases)))
            ):
                reports[r.case_id].append(r)
            for s in self.session.scalars(
                select(ImagingStudy).where(ImagingStudy.case_id.in_(list(cases)))
            ):
                studies[s.case_id].append(s)
        out: dict[str, CaseContext] = {}
        for case_id, case in cases.items():
            cutoff = case.procedure_time if as_of is None else min(case.procedure_time, as_of)
            patient_events = events.get(case.patient_id, [])
            dx = [e for e in patient_events if e.event_type == "diagnosis" and e.event_time <= cutoff]
            out[case_id] = CaseContext(
                case=case,
                patient=patients.get(case.patient_id),
                reports=sorted(reports.get(case_id, []), key=lambda r: r.report_id),
                studies=sorted(studies.get(case_id, []), key=lambda s: s.study_id),
                diagnosis_codes={e.concept_code for e in dx},
                diagnosis_events=dx,
                labs=self.extractor.extract(patient_events, cutoff, lookback_days, aggregate),
                cutoff=cutoff,
            )
        return out

    def follow_up_events(self, case: ClinicalCase, limit: int = 200) -> list[ClinicalEvent]:
        """Events recorded AFTER the procedure (retrospective follow-up only)."""
        stmt = (
            select(ClinicalEvent)
            .where(ClinicalEvent.patient_id == case.patient_id,
                   ClinicalEvent.event_time > case.procedure_time)
            .order_by(ClinicalEvent.event_time)
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    # ------------------------------------------------------------- lab statistics
    def fit_lab_statistics(self, split: str, lookback_days: int, aggregate: str = "last") -> str:
        """Fit robust scaling on one split (the training corpus) and persist it."""
        version = lab_stats_version(split, lookback_days, aggregate)
        case_ids = list(
            self.session.scalars(select(ClinicalCase.case_id).where(ClinicalCase.dataset_split == split))
        )
        features: list[CaseLabFeatures] = []
        for start in range(0, len(case_ids), 1000):
            ctx = self.contexts(case_ids[start : start + 1000], lookback_days, aggregate, with_reports=False)
            features.extend(c.labs for c in ctx.values() if c.labs is not None)
        stats = fit_robust_statistics(features)
        self.session.query(LabNormalizationStat).filter(
            LabNormalizationStat.stats_version == version
        ).delete()
        for key, stat in stats.items():
            concept = self.registry.by_key(key)
            self.session.add(
                LabNormalizationStat(
                    stats_version=version, concept_key=key,
                    canonical_unit=concept.canonical_unit if concept else "",
                    median=stat.median, iqr=stat.iqr, n=stat.n, fitted_split=split,
                )
            )
        self.session.flush()
        return version

    def load_lab_statistics(self, split: str, lookback_days: int, aggregate: str = "last") -> dict[str, RobustStat]:
        version = lab_stats_version(split, lookback_days, aggregate)
        rows = self.session.scalars(
            select(LabNormalizationStat).where(LabNormalizationStat.stats_version == version)
        )
        return {r.concept_key: RobustStat(r.median, r.iqr, r.n) for r in rows}

    # -------------------------------------------------------------------- images
    def image_vectors(self, case_ids: list[str], provider_id: str, version: str) -> dict[str, np.ndarray]:
        if not case_ids:
            return {}
        stmt = select(ImageFeature).where(
            ImageFeature.case_id.in_(case_ids),
            ImageFeature.provider_id == provider_id,
            ImageFeature.provider_version == version,
        )
        return {f.case_id: np.asarray(f.vector, dtype=np.float32) for f in self.session.scalars(stmt)}


def lab_stats_version(split: str, lookback_days: int, aggregate: str) -> str:
    return f"{split}-lb{lookback_days}-{aggregate}"


def case_metadata_fields(ctx: CaseContext) -> dict[str, Any]:
    """Denormalised, filterable case metadata stored on every search document so that
    structured filters are applied *before/during* lexical and vector retrieval."""
    fields: dict[str, Any] = {
        "case_id": ctx.case.case_id,
        "patient_id": ctx.case.patient_id,
        "dataset_split": ctx.case.dataset_split,
        "access_scope": ctx.case.access_scope,
        "age": ctx.age,
        "gender": ctx.patient.gender if ctx.patient else None,
        "race": ctx.patient.race if ctx.patient else None,
        "ethnicity": ctx.patient.ethnicity if ctx.patient else None,
        "diagnosis_codes": sorted(ctx.diagnosis_codes),
        "study_modality": sorted({s.modality for s in ctx.studies if s.modality}),
        "has_report": bool(ctx.reports),
        "has_image": any(s.feature_status != "missing_source" for s in ctx.studies),
    }
    if ctx.labs is not None:
        for key, feature in ctx.labs.features.items():
            fields[f"lab_{key}"] = feature.value
    return fields
