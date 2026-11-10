"""Adapter for a MEDS/Parquet-style release.

Newer public INSPECT versions may ship EHR data as MEDS (Medical Event Data
Standard) Parquet shards plus separate split/mapping files.  This adapter
implements the generic MEDS contract — event shards with ``subject_id``,
``time``, ``code`` and ``numeric_value`` and a ``subject_splits`` table — and
takes cohort / impression / imaging tables from explicitly configured files.

VALIDATION STATUS: covered by unit tests on generated Parquet fixtures.  The
exact file names and column names of the real release have NOT been verified;
set ``dataset.files`` and ``dataset.columns`` after running the dataset
inspector.  Unresolvable required fields raise ``DatasetSchemaError``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.errors import DatasetSchemaError
from app.ingestion.adapters.base import LayoutReport, nullable, resolve_columns
from app.ingestion.adapters.inspect_legacy import (
    COHORT_OPTIONAL,
    COHORT_REQUIRED,
    DEFAULT_LABEL_COLUMNS,
    IMPRESSION_REQUIRED,
    InspectLegacyAdapter,
)
from app.ingestion.canonical import CohortRow, EventRow, ReportRow

MEDS_EVENT_REQUIRED = {
    "patient_id": ("subject_id", "patient_id"),
    "time": ("time",),
    "code": ("code",),
}
MEDS_EVENT_OPTIONAL = {
    "value": ("numeric_value",),
    "text_value": ("text_value",),
    "unit": ("unit",),
    "table_hint": ("table", "omop_table"),
    "visit_id": ("visit_id",),
}
SPLIT_REQUIRED = {"patient_id": ("subject_id", "patient_id"), "split": ("split",)}
# The cohort file may omit the split when a separate subject_splits table provides it.
MEDS_COHORT_REQUIRED = {k: v for k, v in COHORT_REQUIRED.items() if k != "split"}
MEDS_COHORT_OPTIONAL = {**COHORT_OPTIONAL, "split": COHORT_REQUIRED["split"]}


def _pq():  # type: ignore[no-untyped-def]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise DatasetSchemaError("pyarrow is required for the MEDS adapter") from exc
    return pq


def _iter_table_rows(path: Path, batch_size: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream rows of a Parquet or CSV table in bounded batches."""
    if path.suffix == ".parquet":
        row_number = 0
        parquet_file = _pq().ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, row
    else:
        from app.ingestion.adapters.base import iter_dict_rows

        yield from iter_dict_rows(path)


def _header(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return list(_pq().ParquetFile(path).schema_arrow.names)
    from app.ingestion.adapters.base import read_header

    return read_header(path)


def _text(value: Any) -> str | None:
    return None if value is None else nullable(str(value))


class InspectMedsAdapter(InspectLegacyAdapter):
    name = "inspect_meds"
    schema_version = "meds-parquet-v1"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        self._splits: dict[str, str] | None = None

    def event_sources(self) -> list[Path]:
        configured = self.config.files.get("ehr_dir")
        if not configured:
            return []
        directory = self.root / configured
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.rglob("*.parquet") if p.is_file())

    def validate_layout(self) -> LayoutReport:
        if not self.root.is_dir():
            raise DatasetSchemaError(f"dataset root does not exist: {self.root}")
        report = LayoutReport(adapter=self.name, schema_version=self.schema_version)
        cohort = self._file("cohort")
        assert cohort is not None
        header = _header(cohort)
        self._cols["cohort"] = resolve_columns(
            "cohort",
            header,
            MEDS_COHORT_REQUIRED,
            MEDS_COHORT_OPTIONAL,
            self.config.columns.get("cohort"),
            cohort.name,
        )
        self._label_columns = [c for c in DEFAULT_LABEL_COLUMNS if c in header]
        configured_labels = self.config.columns.get("labels")
        if configured_labels:
            missing = [c for c in configured_labels.values() if c not in header]
            if missing:
                raise DatasetSchemaError(f"cohort: configured label column(s) {missing} missing")
            self._label_columns = list(configured_labels.values())
        report.files["cohort"] = [self.relative(cohort)]
        report.modalities.append("cohort")

        splits = self._file("subject_splits", required=False)
        if splits is not None:
            self._cols["splits"] = resolve_columns(
                "splits",
                _header(splits),
                SPLIT_REQUIRED,
                {},
                self.config.columns.get("splits"),
                splits.name,
            )
            report.files["subject_splits"] = [self.relative(splits)]
        elif "split" not in self._cols["cohort"]:
            raise DatasetSchemaError(
                "no dataset split information: neither a split column in the cohort file nor "
                "files.subject_splits is available. Splits are never assigned implicitly."
            )

        impressions = self._file("impressions", required=False)
        if impressions is not None:
            if "report_ref" not in self._cols["cohort"]:
                raise DatasetSchemaError(
                    "impressions configured but cohort has no report reference column"
                )
            self._cols["impressions"] = resolve_columns(
                "impressions",
                _header(impressions),
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
                    _header(source),
                    MEDS_EVENT_REQUIRED,
                    MEDS_EVENT_OPTIONAL,
                    self.config.columns.get("events"),
                    source.name,
                )
                previous = self._cols.setdefault("events", cols)
                if previous != cols:
                    raise DatasetSchemaError(
                        f"MEDS event schema is inconsistent across shards ({source.name})"
                    )
            report.files["events"] = [self.relative(s) for s in sources]
            report.modalities.append("ehr")
        else:
            report.warnings.append("missing modality: EHR events (files.ehr_dir)")

        if self.config.files.get("imaging_metadata") or self.config.files.get("imaging_dir"):
            # imaging handling is shared with the legacy adapter
            from app.ingestion.adapters.base import read_header
            from app.ingestion.adapters.inspect_legacy import IMAGING_OPTIONAL, IMAGING_REQUIRED

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
            report.modalities.append("imaging")
        else:
            report.warnings.append("missing modality: imaging")

        report.resolved_columns = {k: dict(v) for k, v in self._cols.items()}
        self._validated = True
        return report

    def _load_splits(self) -> dict[str, str]:
        if self._splits is None:
            self._splits = {}
            splits = self._file("subject_splits", required=False)
            if splits is not None:
                cols = self._cols["splits"]
                for _, row in _iter_table_rows(splits, self.config.batch_size):
                    pid = _text(row.get(cols["patient_id"]))
                    split = _text(row.get(cols["split"]))
                    if pid and split:
                        self._splits[pid] = split
        return self._splits

    def iter_cohort(self) -> Iterator[CohortRow]:
        self._require_validated()
        cohort = self._file("cohort")
        assert cohort is not None
        cols = self._cols["cohort"]
        splits = self._load_splits()
        rel = self.relative(cohort)

        def get(row: dict[str, Any], key: str) -> str | None:
            return _text(row.get(cols[key])) if key in cols else None

        for row_number, row in _iter_table_rows(cohort, self.config.batch_size):
            patient_id = get(row, "patient_id") or ""
            age = get(row, "age")
            yield CohortRow(
                patient_id=patient_id,
                procedure_time_raw=get(row, "procedure_time") or "",
                split_raw=get(row, "split") or splits.get(patient_id),
                report_ref=get(row, "report_ref"),
                study_ref=get(row, "study_ref"),
                source_file=rel,
                source_row=row_number,
                age=float(age) if age is not None else None,
                gender=get(row, "gender"),
                race=get(row, "race"),
                ethnicity=get(row, "ethnicity"),
                access_scope=get(row, "access_scope"),
                labels={c: _text(row.get(c)) for c in self._label_columns},
            )

    def iter_reports(self) -> Iterator[ReportRow]:
        self._require_validated()
        impressions = self._file("impressions", required=False)
        if impressions is None:
            return
        cols = self._cols["impressions"]
        rel = self.relative(impressions)
        for row_number, row in _iter_table_rows(impressions, self.config.batch_size):
            text = row.get(cols["impression_text"])
            yield ReportRow(
                report_ref=_text(row.get(cols["report_ref"])) or "",
                impression_text=None if text is None else str(text),
                source_file=rel,
                source_row=row_number,
            )

    def iter_events(self, source: Path) -> Iterator[EventRow]:
        self._require_validated()
        cols = self._cols["events"]
        rel = self.relative(source)
        for row_number, row in _iter_table_rows(source, self.config.batch_size):
            value = row.get(cols["value"]) if "value" in cols else None
            if value is None and "text_value" in cols:
                value = row.get(cols["text_value"])
            time_value = row.get(cols["time"])
            yield EventRow(
                patient_id=_text(row.get(cols["patient_id"])) or "",
                time_raw=None if time_value is None else str(time_value),
                code=_text(row.get(cols["code"])) or "",
                value_raw=_text(value),
                unit=_text(row.get(cols["unit"])) if "unit" in cols else None,
                source_file=rel,
                source_row=row_number,
                table_hint=_text(row.get(cols["table_hint"])) if "table_hint" in cols else None,
                visit_id=_text(row.get(cols["visit_id"])) if "visit_id" in cols else None,
            )
