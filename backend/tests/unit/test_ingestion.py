"""Joins, duplicate handling, validation reports, incremental updates and deletion propagation."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.core.container import AppContext, build_indexes, run_ingestion
from app.db.models import (
    AuditEvent,
    CaseLabel,
    ClinicalCase,
    ClinicalEvent,
    ImagingStudy,
    RadiologyReport,
    RetrievalDocument,
)
from app.ingestion.text import clean_impression, meaning_preserved, sentence_spans
from tests.conftest import make_context


def test_validation_counts_and_failed_rows(ctx: AppContext) -> None:
    import json

    out = ctx.settings.artifact_dir / "ingestion" / "synthetic"
    summary = json.loads((out / "validation_report.json").read_text())
    failures = summary["failure_counts"]
    assert failures["cohort:duplicate primary key (patient_id + procedure_time)"] == 1
    assert failures["cohort:unparseable procedure time"] == 1
    assert failures["cohort:invalid dataset split"] == 1
    assert failures["cohort:missing patient identifier"] == 1
    assert failures["reports:duplicate report identifier (first occurrence kept)"] == 1
    assert failures["reports:report does not map to any cohort case"] == 2
    assert failures["events:event patient not in cohort"] == 1
    assert failures["events:conflicting timestamps (end before start)"] == 1
    assert failures["events:unparseable event timestamp"] == 1
    assert failures["imaging:imaging study does not map to any cohort case"] == 1
    assert summary["counts"]["unmatched_cases_without_report"] == 1
    rows = [json.loads(line) for line in (out / "failed_rows.jsonl").read_text().splitlines()]
    assert len(rows) == sum(failures.values())
    # failed-row reports carry identifiers and reasons only — never clinical text
    assert all(set(r) == {"stage", "source_file", "source_row", "reason", "identifiers"} for r in rows)
    manifest = json.loads((out / "data_manifest.json").read_text())
    for key in ("dataset_version", "source_files", "detected_schema_version", "available_modalities",
                "record_counts", "unmatched_joins", "validation_warnings", "ingestion_timestamp",
                "dataset_fingerprint"):
        assert key in manifest
    assert all(f["sha256"] for f in manifest["files"])


def test_joins_use_validated_identifiers(ctx: AppContext) -> None:
    with ctx.session_factory() as session:
        n_cases = session.scalar(select(func.count()).select_from(ClinicalCase))
        assert n_cases == 189
        # every report and study points at an existing case of the same patient
        for report in session.scalars(select(RadiologyReport)):
            case = session.get(ClinicalCase, report.case_id)
            assert case is not None and case.patient_id == report.patient_id and case.report_id == report.report_id
        for study in session.scalars(select(ImagingStudy)):
            case = session.get(ClinicalCase, study.case_id)
            assert case is not None and case.study_id == study.study_id
        # patients with several CT studies keep distinct cases
        multi = session.execute(
            select(ClinicalCase.patient_id, func.count()).group_by(ClinicalCase.patient_id).having(func.count() > 1)
        ).all()
        assert len(multi) >= 10
        # a patient never crosses a split boundary
        crossing = session.execute(
            select(ClinicalCase.patient_id).group_by(ClinicalCase.patient_id)
            .having(func.count(func.distinct(ClinicalCase.dataset_split)) > 1)
        ).all()
        assert crossing == []
        assert session.scalar(select(func.count()).select_from(CaseLabel)) == 189 * 3
        # raw non-numeric measurement text is preserved, not coerced
        censored = session.scalars(select(ClinicalEvent).where(ClinicalEvent.text_value == "<0.5")).all()
        assert len(censored) == 1 and censored[0].numeric_value is None


def test_audit_events_contain_no_clinical_text(ctx: AppContext) -> None:
    import json

    with ctx.session_factory() as session:
        events = list(session.scalars(select(AuditEvent)))
        impressions = [r.impression_text for r in session.scalars(select(RadiologyReport).limit(50))]
    assert {e.action for e in events} >= {"ingestion.completed", "index.updated"}
    blob = json.dumps([e.details for e in events])
    assert not any(text in blob for text in impressions)


def test_cleaning_preserves_negation_laterality_and_measurements() -> None:
    raw = "1.  No  evidence of\tpulmonary embolism.\r\n\r\n2. 6 mm nodule, left upper lobe;  right effusion.  "
    cleaned = clean_impression(raw)
    assert cleaned == "1. No evidence of pulmonary embolism.\n2. 6 mm nodule, left upper lobe; right effusion."
    assert meaning_preserved(raw, cleaned)
    for term in ("No", "6 mm", "left", "right"):
        assert term in cleaned
    assert not meaning_preserved("No acute embolus", "acute embolus")
    # numbered items stay attached to their sentence; a clause after ';' is not a new sentence
    assert [cleaned[a:b] for a, b in sentence_spans(cleaned)] == cleaned.split("\n")


@pytest.fixture()
def mutable_ctx(synthetic_root: Path, tmp_path: Path) -> tuple[AppContext, Path]:
    root = tmp_path / "data"
    shutil.copytree(synthetic_root, root)
    return make_context(root, tmp_path / "artifacts"), root


def _rewrite(path: Path, transform) -> None:  # type: ignore[no-untyped-def]
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    rows = transform(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_incremental_ingestion_skips_unchanged_sources(mutable_ctx: tuple[AppContext, Path]) -> None:
    ctx, _ = mutable_ctx
    report = run_ingestion(ctx)
    assert report.counts["cases_unchanged"] == 189 and report.counts["reports_unchanged"] == 188
    assert report.counts["event_files_unchanged"] == 3 and not report.changed_case_ids
    before = ctx.embeddings.cache.misses
    summary = build_indexes(ctx)
    index = next(s for s in summary["report_indexes"] if s["index"].endswith("_hashing"))
    assert index["indexed_or_updated"] == 0 and index["unchanged"] == 188
    assert ctx.embeddings.cache.misses == before  # nothing was re-embedded


def test_changed_and_deleted_sources_propagate_to_indexes(mutable_ctx: tuple[AppContext, Path]) -> None:
    ctx, root = mutable_ctx
    index = "report_chunks_v1_hashing"
    with ctx.session_factory() as session:
        docs = list(session.scalars(select(RetrievalDocument).where(RetrievalDocument.index_version == index)
                                    .order_by(RetrievalDocument.source_id)))
        changed_doc, deleted_doc = docs[0], docs[1]
        deleted_case = deleted_doc.case_id
    marker = "Unique revised impression mentioning zebrafish artifact."

    def edit_impressions(rows):  # type: ignore[no-untyped-def]
        out = []
        for row in rows:
            if row["impression_id"] == changed_doc.source_id:
                row = {**row, "impressions": marker}
            out.append(row)
        return out

    _rewrite(root / "impressions.csv", edit_impressions)
    _rewrite(root / "cohort.csv", lambda rows: [r for r in rows if r["impression_id"] != deleted_doc.source_id])

    cache_before = ctx.embeddings.cache.count()
    report = run_ingestion(ctx)
    assert deleted_case in report.removed_case_ids and changed_doc.case_id in report.changed_case_ids
    summary = build_indexes(ctx)
    stats = next(s for s in summary["report_indexes"] if s["index"] == index)
    assert stats["deleted"] == 1 and stats["indexed_or_updated"] >= 1
    assert stats["cache_entries_invalidated"] >= 1
    assert ctx.embeddings.cache.count() <= cache_before + 1

    assert ctx.backend.get_document(index, deleted_doc.document_id) is None
    assert ctx.backend.get_document(index, changed_doc.document_id)["content"] == marker
    hits = ctx.backend.bm25_search(index, "zebrafish", None, 5)
    assert [h.id for h in hits] == [changed_doc.document_id]
    assert ctx.backend.get_document("cases_v1", deleted_case) is None
    with ctx.session_factory() as session:
        assert session.get(ClinicalCase, deleted_case) is None
        assert session.get(RetrievalDocument, deleted_doc.document_id) is None
        assert session.scalar(select(func.count()).select_from(RadiologyReport)
                              .where(RadiologyReport.case_id == deleted_case)) == 0
        actions = [a.action for a in session.scalars(select(AuditEvent))]
    assert actions.count("index.updated") >= 4
