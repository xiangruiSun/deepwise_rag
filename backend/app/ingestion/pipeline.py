"""Ingestion pipeline: extract → validate & link → normalise → persist (spec section 7).

Design rules:

* streaming, bounded batches — EHR events are never loaded into one DataFrame;
* joins use the dataset's own identifiers only (no fuzzy timestamp joins);
* invalid rows are counted and reported, never silently merged or dropped;
* deterministic source hashes make re-runs incremental and propagate deletions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DatasetConfig
from app.core.hashing import file_sha256, make_case_id, make_id, parse_timestamp, sha256_text, stable_hash
from app.db.base import session_scope
from app.db.models import (
    AuditEvent,
    CaseLabel,
    ClinicalCase,
    ClinicalEvent,
    ImageFeature,
    ImagingStudy,
    Patient,
    RadiologyReport,
    SourceFileState,
)
from app.ingestion.adapters import DatasetAdapter, build_adapter
from app.ingestion.canonical import SPLIT_ALIASES, CohortRow, FailedRow
from app.ingestion.inspector import DatasetInspector, dataset_fingerprint, write_manifest
from app.ingestion.normalization import (
    classify_event,
    normalize_diagnosis_code,
    parse_code,
    parse_numeric,
)
from app.ingestion.text import clean_impression, meaning_preserved

MAX_FAILED_ROWS_PER_REASON = 500
OUTCOME_LABEL_HINTS = ("mortality", "readmission", "month", "ph")


@dataclass
class IngestionReport:
    dataset: str
    adapter: str
    fingerprint: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    failed_rows: list[FailedRow] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    changed_case_ids: set[str] = field(default_factory=set)
    removed_case_ids: set[str] = field(default_factory=set)
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "adapter": self.adapter,
            "fingerprint": self.fingerprint,
            "counts": dict(sorted(self.counts.items())),
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "warnings": self.warnings,
            "changed_cases": len(self.changed_case_ids),
            "removed_cases": len(self.removed_case_ids),
            "manifest_path": self.manifest_path,
        }


class IngestionPipeline:
    def __init__(
        self,
        config: DatasetConfig,
        session_factory: sessionmaker[Session],
        artifact_dir: Path,
        actor: str = "system",
        adapter: DatasetAdapter | None = None,
    ) -> None:
        self.config = config
        self.session_factory = session_factory
        self.artifact_dir = Path(artifact_dir)
        self.actor = actor
        self.adapter = adapter or build_adapter(config)
        self.report = IngestionReport(dataset=config.name, adapter=self.adapter.name)
        self._failure_counter: Counter[str] = Counter()

    # ------------------------------------------------------------------ helpers
    def _fail(self, stage: str, source_file: str, row: int | None, reason: str, **ids: str | None) -> None:
        key = f"{stage}:{reason}"
        self._failure_counter[key] += 1
        if self._failure_counter[key] <= MAX_FAILED_ROWS_PER_REASON:
            self.report.failed_rows.append(FailedRow(stage, source_file, row, reason, dict(ids)))

    def _count(self, key: str, amount: int = 1) -> None:
        self.report.counts[key] = self.report.counts.get(key, 0) + amount

    # ---------------------------------------------------------------------- run
    def run(self, force: bool = False) -> IngestionReport:
        manifest = DatasetInspector(self.adapter.root).inspect()
        layout = self.adapter.validate_layout()
        if manifest["detected_layout"] not in (self.adapter.name, "unknown"):
            self.report.warnings.append(
                f"inspector detected layout '{manifest['detected_layout']}' but adapter "
                f"'{self.adapter.name}' is configured"
            )
        self.report.warnings.extend(layout.warnings)
        self.report.fingerprint = dataset_fingerprint(manifest)

        with session_scope(self.session_factory) as session:
            cases = self._ingest_cohort(session)
        with session_scope(self.session_factory) as session:
            self._ingest_reports(session, cases)
        with session_scope(self.session_factory) as session:
            self._ingest_events(session, cases, force)
        with session_scope(self.session_factory) as session:
            self._ingest_imaging(session, cases)

        self.report.failure_counts = dict(self._failure_counter)
        manifest.update(
            {
                "dataset_name": self.config.name,
                "dataset_version": self.config.release,
                "adapter": self.adapter.name,
                "detected_schema_version": layout.schema_version,
                "resolved_columns": layout.resolved_columns,
                "source_files": layout.files,
                "record_counts": dict(sorted(self.report.counts.items())),
                "unmatched_joins": {
                    k: v for k, v in self.report.counts.items() if k.startswith("unmatched_")
                },
                "failure_counts": self.report.failure_counts,
                "dataset_fingerprint": self.report.fingerprint,
                "ingestion_timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        manifest["validation_warnings"] = sorted({*manifest["validation_warnings"], *self.report.warnings})
        self.report.manifest = manifest
        out_dir = self.artifact_dir / "ingestion" / self.config.name
        manifest_path = out_dir / "data_manifest.json"
        write_manifest(manifest, manifest_path)
        self.report.manifest_path = str(manifest_path)
        with (out_dir / "failed_rows.jsonl").open("w", encoding="utf-8") as fh:
            for row in self.report.failed_rows:
                fh.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
        (out_dir / "validation_report.json").write_text(
            json.dumps(self.report.summary(), indent=2, sort_keys=True), encoding="utf-8"
        )
        with session_scope(self.session_factory) as session:
            session.add(
                AuditEvent(
                    actor=self.actor,
                    action="ingestion.completed",
                    resource_type="dataset",
                    resource_id=self.config.name,
                    details={
                        "fingerprint": self.report.fingerprint,
                        "counts": self.report.counts,
                        "failure_counts": self.report.failure_counts,
                        "changed_cases": len(self.report.changed_case_ids),
                        "removed_cases": len(self.report.removed_case_ids),
                    },
                )
            )
        return self.report

    # ------------------------------------------------------------------- cohort
    def _ingest_cohort(self, session: Session) -> dict[str, dict[str, Any]]:
        """Returns ``case_id -> {patient_id, procedure_time, report_ref, study_ref, ...}``."""
        cases: dict[str, dict[str, Any]] = {}
        patients: dict[str, CohortRow] = {}
        patient_split: dict[str, str] = {}
        report_owner: dict[str, str] = {}
        study_owner: dict[str, str] = {}
        for row in self.adapter.iter_cohort():
            self._count("cohort_rows")
            ids = {"patient_id": row.patient_id or None}
            if not row.patient_id:
                self._fail("cohort", row.source_file, row.source_row, "missing patient identifier")
                continue
            try:
                procedure_time = parse_timestamp(row.procedure_time_raw)
            except ValueError:
                self._fail("cohort", row.source_file, row.source_row, "unparseable procedure time", **ids)
                continue
            split = SPLIT_ALIASES.get((row.split_raw or "").strip().lower())
            if split is None:
                self._fail("cohort", row.source_file, row.source_row, "invalid dataset split", **ids)
                continue
            case_id = make_case_id(row.patient_id, procedure_time)
            if case_id in cases:
                self._fail("cohort", row.source_file, row.source_row,
                           "duplicate primary key (patient_id + procedure_time)", **ids, case_id=case_id)
                continue
            previous_split = patient_split.setdefault(row.patient_id, split)
            if previous_split != split:
                self._fail("cohort", row.source_file, row.source_row,
                           "patient appears in more than one dataset split", **ids, case_id=case_id)
                continue
            report_ref, study_ref = row.report_ref, row.study_ref
            if report_ref and report_owner.setdefault(report_ref, case_id) != case_id:
                self._fail("cohort", row.source_file, row.source_row,
                           "report reference already linked to another case", **ids, case_id=case_id)
                report_ref = None
            if study_ref and study_owner.setdefault(study_ref, case_id) != case_id:
                self._fail("cohort", row.source_file, row.source_row,
                           "study reference already linked to another case", **ids, case_id=case_id)
                study_ref = None
            first = patients.setdefault(row.patient_id, row)
            for attr in ("gender", "race", "ethnicity"):
                if getattr(first, attr) and getattr(row, attr) and getattr(first, attr) != getattr(row, attr):
                    self.report.warnings.append(
                        f"patient demographic '{attr}' differs between cohort rows; first value kept"
                    )
            cases[case_id] = {
                "patient_id": row.patient_id,
                "procedure_time": procedure_time,
                "split": split,
                "report_ref": report_ref,
                "study_ref": study_ref,
                "age": row.age,
                "access_scope": row.access_scope or self.config.default_access_scope,
                "labels": row.labels,
                "slice_count": row.extra.get("slice_count"),
                "row_hash": stable_hash(
                    [row.patient_id, row.procedure_time_raw, split, report_ref, study_ref, row.age,
                     row.gender, row.race, row.ethnicity, row.access_scope, row.labels]
                ),
            }

        existing_patients = {p.patient_id: p for p in session.scalars(select(Patient))}
        for pid, row in patients.items():
            ages = [c["age"] for c in cases.values() if c["patient_id"] == pid and c["age"] is not None]
            patient = existing_patients.get(pid)
            if patient is None:
                patient = Patient(patient_id=pid, source_dataset=self.config.name)
                session.add(patient)
            patient.age = min(ages) if ages else None
            patient.gender = row.gender
            patient.race = row.race
            patient.ethnicity = row.ethnicity
            patient.source_dataset = self.config.name
            patient.source_version = self.config.release
        session.flush()

        existing = {c.case_id: c for c in session.scalars(select(ClinicalCase))}
        for case_id, info in cases.items():
            case = existing.get(case_id)
            if case is not None and case.source_row_hash == info["row_hash"]:
                self._count("cases_unchanged")
                continue
            if case is None:
                case = ClinicalCase(case_id=case_id, patient_id=info["patient_id"],
                                    procedure_time=info["procedure_time"])
                session.add(case)
                self._count("cases_created")
            else:
                self._count("cases_updated")
            case.age_at_procedure = info["age"]
            case.dataset_split = info["split"]
            case.study_id = info["study_ref"]
            case.report_id = info["report_ref"]
            case.source_version = self.config.release
            case.source_patient_key = info["patient_id"]
            case.source_row_hash = info["row_hash"]
            case.access_scope = info["access_scope"]
            case.ingestion_status = "ingested"
            session.flush()
            session.execute(delete(CaseLabel).where(CaseLabel.case_id == case_id))
            for name, value in info["labels"].items():
                kind = "outcome" if any(h in name.lower() for h in OUTCOME_LABEL_HINTS) else "diagnostic"
                session.add(CaseLabel(case_id=case_id, label_name=name, label_value=value, label_kind=kind))
            self.report.changed_case_ids.add(case_id)

        removed = set(existing) - set(cases)
        for case_id in removed:
            self._delete_case(session, case_id)
        self.report.removed_case_ids |= removed
        orphan_patients = set(existing_patients) - set(patients)
        if orphan_patients:
            session.execute(delete(ClinicalEvent).where(ClinicalEvent.patient_id.in_(orphan_patients)))
            session.execute(delete(Patient).where(Patient.patient_id.in_(orphan_patients)))
        self._count("patients", len(patients))
        self._count("cases", len(cases))
        self._count("cases_removed", len(removed))
        for split in ("train", "valid", "test"):
            self._count(f"cases_split_{split}", sum(1 for c in cases.values() if c["split"] == split))
        return cases

    def _delete_case(self, session: Session, case_id: str) -> None:
        studies = list(session.scalars(select(ImagingStudy.study_id).where(ImagingStudy.case_id == case_id)))
        if studies:
            session.execute(delete(ImageFeature).where(ImageFeature.study_id.in_(studies)))
        session.execute(delete(ImagingStudy).where(ImagingStudy.case_id == case_id))
        session.execute(delete(RadiologyReport).where(RadiologyReport.case_id == case_id))
        # ``retrieval_documents`` rows are deliberately kept: they are the indexer's record of
        # what is in each search index, and the next index sync uses them to delete the
        # obsolete documents from the index before removing the rows.
        session.execute(delete(CaseLabel).where(CaseLabel.case_id == case_id))
        session.execute(delete(ClinicalCase).where(ClinicalCase.case_id == case_id))

    # ------------------------------------------------------------------ reports
    def _ingest_reports(self, session: Session, cases: dict[str, dict[str, Any]]) -> None:
        ref_to_case = {info["report_ref"]: cid for cid, info in cases.items() if info["report_ref"]}
        existing = {r.report_id: r for r in session.scalars(select(RadiologyReport))}
        seen: set[str] = set()
        text_hashes: Counter[str] = Counter()
        for row in self.adapter.iter_reports():
            self._count("report_rows")
            ids = {"report_ref": row.report_ref or None}
            if not row.report_ref:
                self._fail("reports", row.source_file, row.source_row, "missing report identifier")
                continue
            if row.report_ref in seen:
                self._fail("reports", row.source_file, row.source_row,
                           "duplicate report identifier (first occurrence kept)", **ids)
                continue
            case_id = ref_to_case.get(row.report_ref)
            if case_id is None:
                self._count("unmatched_reports_without_case")
                self._fail("reports", row.source_file, row.source_row,
                           "report does not map to any cohort case", **ids)
                continue
            raw = row.impression_text or ""
            cleaned = clean_impression(raw)
            if not cleaned:
                self._fail("reports", row.source_file, row.source_row, "empty impression text", **ids)
                continue
            if not meaning_preserved(raw, cleaned):  # defensive invariant
                self._fail("reports", row.source_file, row.source_row,
                           "text cleaning changed content tokens", **ids)
                continue
            seen.add(row.report_ref)
            content_hash = sha256_text(cleaned)
            text_hashes[content_hash] += 1
            info = cases[case_id]
            report = existing.get(row.report_ref)
            if report is not None and report.content_hash == content_hash and report.case_id == case_id:
                self._count("reports_unchanged")
                continue
            if report is None:
                report = RadiologyReport(report_id=row.report_ref, case_id=case_id,
                                         patient_id=info["patient_id"], impression_text=cleaned,
                                         content_hash=content_hash)
                session.add(report)
                self._count("reports_created")
            else:
                self._count("reports_updated")
            report.case_id = case_id
            report.patient_id = info["patient_id"]
            report.study_id = info["study_ref"]
            report.report_time = info["procedure_time"]
            report.impression_text = cleaned
            report.source_file = row.source_file
            report.source_row = row.source_row
            report.content_hash = content_hash
            self.report.changed_case_ids.add(case_id)
        for report_id, report in existing.items():
            if report_id not in seen:
                self.report.changed_case_ids.add(report.case_id)
                session.delete(report)
                self._count("reports_removed")
        missing = [ref for ref in ref_to_case if ref not in seen]
        self._count("unmatched_cases_without_report", len(missing))
        self._count("reports", len(seen))
        self._count("duplicate_report_texts", sum(n - 1 for n in text_hashes.values() if n > 1))

    # ------------------------------------------------------------------- events
    def _ingest_events(self, session: Session, cases: dict[str, dict[str, Any]], force: bool) -> None:
        known_patients = {info["patient_id"] for info in cases.values()}
        patient_cases: dict[str, list[str]] = defaultdict(list)
        for cid, info in cases.items():
            patient_cases[info["patient_id"]].append(cid)
        states = {s.source_key: s for s in session.scalars(select(SourceFileState))}
        current_keys: set[str] = set()
        prefixes = self.config.code_prefixes
        for source in self.adapter.event_sources():
            rel = self.adapter.relative(source)
            key = f"events:{rel}"
            current_keys.add(key)
            digest = file_sha256(source)
            state = states.get(key)
            if state is not None and state.content_hash == digest and not force:
                self._count("event_files_unchanged")
                self._count("events", state.record_count)
                continue
            session.execute(delete(ClinicalEvent).where(ClinicalEvent.source_file == rel))
            touched: set[str] = set()
            batch: list[dict[str, Any]] = []
            inserted = 0
            for row in self.adapter.iter_events(source):
                self._count("event_rows")
                ids = {"patient_id": row.patient_id or None}
                if not row.patient_id or row.patient_id not in known_patients:
                    self._count("unmatched_events_without_patient")
                    self._fail("events", rel, row.source_row, "event patient not in cohort", **ids)
                    continue
                if not row.code:
                    self._fail("events", rel, row.source_row, "missing concept code", **ids)
                    continue
                try:
                    event_time = parse_timestamp(row.time_raw)
                except ValueError:
                    self._fail("events", rel, row.source_row, "unparseable event timestamp", **ids)
                    continue
                if row.end_time_raw:
                    try:
                        if parse_timestamp(row.end_time_raw) < event_time:
                            self._fail("events", rel, row.source_row,
                                       "conflicting timestamps (end before start)", **ids)
                            continue
                    except ValueError:
                        self._fail("events", rel, row.source_row, "unparseable event end timestamp", **ids)
                        continue
                if self._demographic_event(session, row.patient_id, row.code, event_time, prefixes):
                    self._count("demographic_events")
                    touched.add(row.patient_id)
                    continue
                event_type = classify_event(row.code, row.table_hint)
                system, _ = parse_code(row.code)
                numeric = parse_numeric(row.value_raw)
                code = normalize_diagnosis_code(row.code) if event_type == "diagnosis" else row.code
                batch.append(
                    {
                        "event_id": make_id("ev", rel, row.source_row),
                        "patient_id": row.patient_id,
                        "case_id": None,
                        "event_type": event_type,
                        "concept_code": code,
                        "code_system": system,
                        "display_name": row.display_name,
                        "event_time": event_time,
                        "numeric_value": numeric,
                        # the raw value is preserved whenever it is not a plain number
                        "text_value": None if numeric is not None else row.value_raw,
                        "unit": row.unit,
                        "source_record_id": str(row.source_row),
                        "source_file": rel,
                    }
                )
                touched.add(row.patient_id)
                if len(batch) >= self.config.batch_size:
                    session.execute(insert(ClinicalEvent), batch)
                    inserted += len(batch)
                    batch = []
            if batch:
                session.execute(insert(ClinicalEvent), batch)
                inserted += len(batch)
            if state is None:
                session.add(SourceFileState(source_key=key, content_hash=digest, record_count=inserted))
            else:
                state.content_hash = digest
                state.record_count = inserted
            self._count("event_files_ingested")
            self._count("events", inserted)
            for pid in touched:
                self.report.changed_case_ids.update(patient_cases[pid])
        for key, state in states.items():
            if key.startswith("events:") and key not in current_keys:
                rel = key.split(":", 1)[1]
                gone = set(
                    session.scalars(
                        select(ClinicalEvent.patient_id).where(ClinicalEvent.source_file == rel).distinct()
                    )
                )
                session.execute(delete(ClinicalEvent).where(ClinicalEvent.source_file == rel))
                session.delete(state)
                for pid in gone:
                    self.report.changed_case_ids.update(patient_cases.get(pid, []))
                self._count("event_files_removed")
        self._apply_event_ages(session, cases)

    def _demographic_event(self, session: Session, patient_id: str, code: str,
                           event_time: datetime, prefixes: dict[str, str]) -> bool:
        """Some releases encode demographics as events.  Only explicitly configured code
        prefixes are interpreted (``dataset.code_prefixes``)."""
        if not prefixes:
            return False
        for attr in ("gender", "race", "ethnicity"):
            prefix = prefixes.get(attr)
            if prefix and code.startswith(prefix):
                patient = session.get(Patient, patient_id)
                if patient is not None and getattr(patient, attr) is None:
                    setattr(patient, attr, code[len(prefix):])
                return True
        birth = prefixes.get("birth")
        if birth and code == birth:
            self._births = getattr(self, "_births", {})
            self._births[patient_id] = event_time
            return True
        return False

    def _apply_event_ages(self, session: Session, cases: dict[str, dict[str, Any]]) -> None:
        births: dict[str, datetime] = getattr(self, "_births", {})
        if not births:
            return
        # Both timestamps carry the same patient-specific shift, so their difference is valid.
        for case_id, info in cases.items():
            birth = births.get(info["patient_id"])
            if birth is None or info["age"] is not None:
                continue
            case = session.get(ClinicalCase, case_id)
            if case is not None:
                case.age_at_procedure = round((info["procedure_time"] - birth).days / 365.25, 1)
        for pid, _birth in births.items():
            ages = [
                c.age_at_procedure
                for c in session.scalars(select(ClinicalCase).where(ClinicalCase.patient_id == pid))
                if c.age_at_procedure is not None
            ]
            patient = session.get(Patient, pid)
            if patient is not None and ages:
                patient.age = min(ages)

    # ------------------------------------------------------------------ imaging
    def _ingest_imaging(self, session: Session, cases: dict[str, dict[str, Any]]) -> None:
        ref_to_case = {info["study_ref"]: cid for cid, info in cases.items() if info["study_ref"]}
        existing = {s.study_id: s for s in session.scalars(select(ImagingStudy))}
        seen: set[str] = set()
        for row in self.adapter.iter_imaging():
            self._count("imaging_rows")
            ids = {"study_ref": row.study_ref or None}
            if not row.study_ref:
                self._fail("imaging", row.source_file, row.source_row, "missing study identifier")
                continue
            if row.study_ref in seen:
                self._fail("imaging", row.source_file, row.source_row, "duplicate study identifier", **ids)
                continue
            case_id = ref_to_case.get(row.study_ref)
            if case_id is None:
                self._count("unmatched_imaging_without_case")
                self._fail("imaging", row.source_file, row.source_row,
                           "imaging study does not map to any cohort case", **ids)
                continue
            seen.add(row.study_ref)
            path = Path(row.source_path) if row.source_path else None
            source_hash: str | None = None
            status = "pending"
            if path is None or not path.exists():
                status = "missing_source"
                self._count("imaging_missing_source")
                self._fail("imaging", row.source_file, row.source_row,
                           "imaging source file not found", **ids)
            elif path.is_dir():
                source_hash = stable_hash(sorted((f.name, f.stat().st_size) for f in path.iterdir()))
            else:
                source_hash = file_sha256(path, max_bytes=64 * 1024 * 1024)
            study = existing.get(row.study_ref)
            if study is None:
                study = ImagingStudy(study_id=row.study_ref, case_id=case_id,
                                     patient_id=cases[case_id]["patient_id"])
                session.add(study)
            elif study.source_hash != source_hash:
                # source changed: cached features are stale
                session.execute(delete(ImageFeature).where(ImageFeature.study_id == row.study_ref))
                self.report.changed_case_ids.add(case_id)
            elif study.feature_status not in ("missing_source",):
                status = study.feature_status
            study.case_id = case_id
            study.patient_id = cases[case_id]["patient_id"]
            study.modality = row.modality
            study.source_path = str(path) if path else None
            study.slice_count = row.slice_count
            study.spacing = row.spacing
            study.orientation_metadata = row.orientation
            study.source_hash = source_hash
            study.feature_status = status
        for study_id, study in existing.items():
            if study_id not in seen:
                session.execute(delete(ImageFeature).where(ImageFeature.study_id == study_id))
                self.report.changed_case_ids.add(study.case_id)
                session.delete(study)
        self._count("imaging_studies", len(seen))
        self._count("unmatched_cases_without_imaging",
                    sum(1 for ref in ref_to_case if ref not in seen))
