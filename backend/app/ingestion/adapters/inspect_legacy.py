"""Adapter for the layout described in the INSPECT paper.

The paper describes: a cohort CSV, an anonymised impression CSV, CTPA scans and
gzipped FEMR-format EHR CSVs.  Publicly visible loader code confirms only a few
cohort columns (``patient_id``, ``procedure_time``, ``split``); the remaining
names below are *accepted aliases*, not verified facts.  Every field can be
overridden through ``dataset.columns`` and nothing is matched fuzzily — if a
required field cannot be resolved the adapter fails with instructions.

VALIDATION STATUS: exercised end-to-end against synthetic data written in this
layout.  Not yet validated against the access-controlled INSPECT release.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from app.core.errors import DatasetSchemaError
from app.ingestion.adapters.base import (
    DatasetAdapter,
    LayoutReport,
    iter_dict_rows,
    nullable,
    nullable_float,
    nullable_int,
    read_header,
    resolve_columns,
)
from app.ingestion.canonical import CohortRow, EventRow, ImagingRow, ReportRow

COHORT_REQUIRED = {
    "patient_id": ("patient_id", "person_id", "PatientID"),
    "procedure_time": ("procedure_time", "procedure_DATETIME", "StudyTime"),
    "split": ("split",),
}
COHORT_OPTIONAL = {
    "report_ref": ("impression_id",),
    "study_ref": ("image_id",),
    "slice_count": ("num_slices",),
    "age": ("age",),
    "gender": ("gender", "sex"),
    "race": ("race",),
    "ethnicity": ("ethnicity",),
    "access_scope": ("access_scope",),
}
IMPRESSION_REQUIRED = {
    "report_ref": ("impression_id",),
    "impression_text": ("impressions", "impression", "impression_text"),
}
EVENT_REQUIRED = {
    "patient_id": ("patient_id", "person_id"),
    "time": ("start", "time"),
    "code": ("code",),
}
EVENT_OPTIONAL = {
    "end": ("end",),
    "value": ("value",),
    "unit": ("unit",),
    "visit_id": ("visit_id",),
    "table_hint": ("omop_table",),
    "display_name": ("display_name", "description"),
}
IMAGING_REQUIRED = {"study_ref": ("image_id", "study_ref")}
IMAGING_OPTIONAL = {
    "source_path": ("path", "source_path"),
    "modality": ("modality",),
    "slice_count": ("num_slices", "slice_count"),
    "spacing": ("spacing",),
    "orientation": ("orientation",),
}
DEFAULT_LABEL_COLUMNS = (
    "pe_positive_nlp",
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
)


class InspectLegacyAdapter(DatasetAdapter):
    name = "inspect_legacy"
    schema_version = "inspect-paper-csv-v1"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        self._cols: dict[str, dict[str, str]] = {}
        self._label_columns: list[str] = []
        self._validated = False

    # ------------------------------------------------------------------ layout
    def _event_dir(self) -> Path | None:
        configured = self.config.files.get("ehr_dir")
        if not configured:
            return None
        path = self.root / configured
        return path if path.is_dir() else None

    def event_sources(self) -> list[Path]:
        directory = self._event_dir()
        if directory is None:
            return []
        files = [
            p
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz"))
        ]
        return files

    def validate_layout(self) -> LayoutReport:
        if not self.root.is_dir():
            raise DatasetSchemaError(f"dataset root does not exist: {self.root}")
        report = LayoutReport(adapter=self.name, schema_version=self.schema_version)
        cohort = self._file("cohort")
        assert cohort is not None
        header = read_header(cohort)
        self._cols["cohort"] = resolve_columns(
            "cohort",
            header,
            COHORT_REQUIRED,
            COHORT_OPTIONAL,
            self.config.columns.get("cohort"),
            cohort.name,
        )
        configured_labels = self.config.columns.get("labels")
        if configured_labels:
            missing = [c for c in configured_labels.values() if c not in header]
            if missing:
                raise DatasetSchemaError(
                    f"cohort: configured label column(s) {missing} not present in {cohort.name}"
                )
            self._label_columns = list(configured_labels.values())
        else:
            self._label_columns = [c for c in DEFAULT_LABEL_COLUMNS if c in header]
            if not self._label_columns:
                report.warnings.append(
                    "no known research label columns found in cohort file; "
                    "set dataset.columns.labels to ingest them"
                )
        report.files["cohort"] = [self.relative(cohort)]
        report.modalities.append("cohort")

        impressions = self._file("impressions", required=False)
        if impressions is not None:
            if "report_ref" not in self._cols["cohort"]:
                raise DatasetSchemaError(
                    "an impressions file is configured but the cohort file has no resolvable "
                    "report reference column (dataset.columns.cohort.report_ref). Reports are "
                    "joined only through the dataset's own mapping, never by timestamps."
                )
            self._cols["impressions"] = resolve_columns(
                "impressions",
                read_header(impressions),
                IMPRESSION_REQUIRED,
                {},
                self.config.columns.get("impressions"),
                impressions.name,
            )
            report.files["impressions"] = [self.relative(impressions)]
            report.modalities.append("radiology_report")
        else:
            report.warnings.append("missing modality: radiology impressions (files.impressions)")

        sources = self.event_sources()
        if sources:
            for source in sources:
                cols = resolve_columns(
                    "events",
                    read_header(source),
                    EVENT_REQUIRED,
                    EVENT_OPTIONAL,
                    self.config.columns.get("events"),
                    source.name,
                )
                previous = self._cols.setdefault("events", cols)
                if previous != cols:
                    raise DatasetSchemaError(
                        f"clinical event schema is inconsistent across files: {source.name} "
                        "resolves to different columns than earlier event files"
                    )
            report.files["events"] = [self.relative(s) for s in sources]
            report.modalities.append("ehr")
        else:
            report.warnings.append("missing modality: EHR events (files.ehr_dir)")

        imaging = self._file("imaging_metadata", required=False)
        if imaging is not None:
            self._cols["imaging"] = resolve_columns(
                "imaging",
                read_header(imaging),
                IMAGING_REQUIRED,
                IMAGING_OPTIONAL,
                self.config.columns.get("imaging"),
                imaging.name,
            )
            report.files["imaging_metadata"] = [self.relative(imaging)]
            report.modalities.append("imaging")
        elif self.config.files.get("imaging_dir"):
            if "study_ref" not in self._cols["cohort"]:
                raise DatasetSchemaError(
                    "files.imaging_dir is configured but the cohort file has no resolvable "
                    "study reference column (dataset.columns.cohort.study_ref)"
                )
            report.files["imaging_dir"] = [self.config.files["imaging_dir"]]
            report.modalities.append("imaging")
        else:
            report.warnings.append("missing modality: imaging (files.imaging_metadata/imaging_dir)")

        report.resolved_columns = {k: dict(v) for k, v in self._cols.items()}
        report.resolved_columns["labels"] = {c: c for c in self._label_columns}
        self._validated = True
        return report

    def _require_validated(self) -> None:
        if not self._validated:
            self.validate_layout()

    # ------------------------------------------------------------------ cohort
    def iter_cohort(self) -> Iterator[CohortRow]:
        self._require_validated()
        cohort = self._file("cohort")
        assert cohort is not None
        cols = self._cols["cohort"]
        rel = self.relative(cohort)
        for row_number, row in iter_dict_rows(cohort):
            yield CohortRow(
                patient_id=(row.get(cols["patient_id"]) or "").strip(),
                procedure_time_raw=(row.get(cols["procedure_time"]) or "").strip(),
                split_raw=nullable(row.get(cols["split"])),
                report_ref=nullable(row.get(cols["report_ref"])) if "report_ref" in cols else None,
                study_ref=nullable(row.get(cols["study_ref"])) if "study_ref" in cols else None,
                source_file=rel,
                source_row=row_number,
                age=nullable_float(row.get(cols["age"])) if "age" in cols else None,
                gender=nullable(row.get(cols["gender"])) if "gender" in cols else None,
                race=nullable(row.get(cols["race"])) if "race" in cols else None,
                ethnicity=nullable(row.get(cols["ethnicity"])) if "ethnicity" in cols else None,
                access_scope=(
                    nullable(row.get(cols["access_scope"])) if "access_scope" in cols else None
                ),
                labels={c: nullable(row.get(c)) for c in self._label_columns},
                extra={
                    "slice_count": nullable_int(row.get(cols["slice_count"]))
                    if "slice_count" in cols
                    else None
                },
            )

    # ----------------------------------------------------------------- reports
    def iter_reports(self) -> Iterator[ReportRow]:
        self._require_validated()
        impressions = self._file("impressions", required=False)
        if impressions is None:
            return
        cols = self._cols["impressions"]
        rel = self.relative(impressions)
        for row_number, row in iter_dict_rows(impressions):
            yield ReportRow(
                report_ref=(row.get(cols["report_ref"]) or "").strip(),
                impression_text=row.get(cols["impression_text"]),
                source_file=rel,
                source_row=row_number,
            )

    # ------------------------------------------------------------------ events
    def iter_events(self, source: Path) -> Iterator[EventRow]:
        self._require_validated()
        cols = self._cols["events"]
        rel = self.relative(source)
        for row_number, row in iter_dict_rows(source):
            yield EventRow(
                patient_id=(row.get(cols["patient_id"]) or "").strip(),
                time_raw=nullable(row.get(cols["time"])),
                code=(row.get(cols["code"]) or "").strip(),
                value_raw=nullable(row.get(cols["value"])) if "value" in cols else None,
                unit=nullable(row.get(cols["unit"])) if "unit" in cols else None,
                source_file=rel,
                source_row=row_number,
                end_time_raw=nullable(row.get(cols["end"])) if "end" in cols else None,
                table_hint=nullable(row.get(cols["table_hint"])) if "table_hint" in cols else None,
                visit_id=nullable(row.get(cols["visit_id"])) if "visit_id" in cols else None,
                display_name=(
                    nullable(row.get(cols["display_name"])) if "display_name" in cols else None
                ),
            )

    # ----------------------------------------------------------------- imaging
    def iter_imaging(self) -> Iterator[ImagingRow]:
        self._require_validated()
        imaging = self._file("imaging_metadata", required=False)
        if imaging is not None:
            cols = self._cols["imaging"]
            rel = self.relative(imaging)
            for row_number, row in iter_dict_rows(imaging):
                spacing = nullable(row.get(cols["spacing"])) if "spacing" in cols else None
                orientation = (
                    nullable(row.get(cols["orientation"])) if "orientation" in cols else None
                )
                path = nullable(row.get(cols["source_path"])) if "source_path" in cols else None
                yield ImagingRow(
                    study_ref=(row.get(cols["study_ref"]) or "").strip(),
                    source_path=str(self.root / path) if path else None,
                    modality=nullable(row.get(cols["modality"])) if "modality" in cols else None,
                    source_file=rel,
                    source_row=row_number,
                    slice_count=(
                        nullable_int(row.get(cols["slice_count"]))
                        if "slice_count" in cols
                        else None
                    ),
                    spacing=_parse_json_list(spacing),
                    orientation=_parse_json_dict(orientation),
                )
            return
        imaging_dir = self.config.files.get("imaging_dir")
        if not imaging_dir:
            return
        # Directory layout: one entry per study, named by the cohort's study reference
        # (a DICOM series directory, ``<ref>.npy`` or ``<ref>.nii[.gz]``).
        base = self.root / imaging_dir
        for cohort_row in self.iter_cohort():
            ref = cohort_row.study_ref
            if not ref:
                continue
            found: Path | None = None
            for candidate in (base / ref, base / f"{ref}.npy", base / f"{ref}.nii.gz", base / f"{ref}.nii"):
                if candidate.exists():
                    found = candidate
                    break
            yield ImagingRow(
                study_ref=ref,
                source_path=str(found) if found else None,
                modality="CT",
                source_file=cohort_row.source_file,
                source_row=cohort_row.source_row,
                slice_count=cohort_row.extra.get("slice_count"),
            )


def _parse_json_list(text: str | None) -> list[float] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return [float(v) for v in value] if isinstance(value, list) else None


def _parse_json_dict(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
