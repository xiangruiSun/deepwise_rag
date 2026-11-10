"""Stable identities, schema detection and adapter column resolution."""

from __future__ import annotations

import csv
import gzip
from datetime import datetime
from pathlib import Path

import pytest

from app.core.config import DatasetConfig
from app.core.errors import DatasetSchemaError
from app.core.hashing import make_case_id, normalize_procedure_time, parse_timestamp, stable_hash
from app.db.base import create_all, make_engine, make_session_factory
from app.db.models import ClinicalCase, Patient
from app.ingestion.adapters import InspectLegacyAdapter, InspectMedsAdapter, build_adapter
from app.ingestion.inspector import DatasetInspector, dataset_fingerprint
from app.ingestion.pipeline import IngestionPipeline


def test_case_id_is_deterministic_and_format_independent() -> None:
    a = make_case_id("12345", "2019-03-04 10:15:00")
    b = make_case_id("12345", datetime(2019, 3, 4, 10, 15))
    c = make_case_id(" 12345 ", "2019-03-04T10:15:00.000")
    assert a == b == c
    assert a.startswith("case_") and len(a) == 25
    assert make_case_id("12345", "2019-03-04 10:15:01") != a
    assert make_case_id("12346", "2019-03-04 10:15:00") != a


def test_timestamp_parsing_is_strict() -> None:
    assert normalize_procedure_time("03/04/2019 10:15") == "2019-03-04T10:15:00"
    for bad in ("", "not-a-date", "13/45/2019", None, "nan"):
        with pytest.raises(ValueError):
            parse_timestamp(bad)


def test_stable_hash_ignores_key_order() -> None:
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_inspector_detects_synthetic_layout_and_builds_manifest(synthetic_root: Path) -> None:
    manifest = DatasetInspector(synthetic_root).inspect()
    assert manifest["detected_layout"] == "synthetic"
    kinds = manifest["file_kinds"]
    assert kinds["csv"] >= 5 and kinds["dicom_series"] == 6 and kinds["numpy"] > 100
    cohort = next(f for f in manifest["files"] if f["path"] == "cohort.csv")
    assert {"patient_id", "procedure_time", "split"} <= set(cohort["columns"])
    assert cohort["sha256"] and cohort["dtypes"]["age"] in {"float", "integer"}
    assert "cohort.csv" in manifest["split_metadata_files"]
    assert manifest["candidate_identifiers"]["cohort.csv"]["patient_id"] == ["patient_id"]
    assert dataset_fingerprint(manifest) == dataset_fingerprint(DatasetInspector(synthetic_root).inspect())


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as fh:  # type: ignore[operator]
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_inspector_recognises_legacy_and_unknown_layouts(tmp_path: Path) -> None:
    _write_csv(tmp_path / "legacy" / "cohort.csv", ["patient_id", "procedure_time", "split"], [[1, "2020-01-01", "train"]])
    _write_csv(tmp_path / "legacy" / "ehr" / "a.csv.gz", ["patient_id", "start", "code", "value"], [[1, "2020-01-01", "LOINC/718-7", 12]])
    inspector = DatasetInspector(tmp_path / "legacy")
    layout, _ = inspector.detect_layout(inspector.discover())
    assert layout == "inspect_legacy"
    _write_csv(tmp_path / "other" / "table.csv", ["foo", "bar"], [[1, 2]])
    manifest = DatasetInspector(tmp_path / "other").inspect()
    assert manifest["detected_layout"] == "unknown"
    assert any("layout not recognised" in w for w in manifest["validation_warnings"])
    assert any("missing modality" in w for w in manifest["validation_warnings"])


def test_missing_required_field_fails_descriptively(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cohort.csv", ["pid", "when", "split"], [[1, "2020-01-01", "train"]])
    config = DatasetConfig(name="x", adapter="inspect_legacy", root=str(tmp_path), files={"cohort": "cohort.csv"})
    with pytest.raises(DatasetSchemaError) as err:
        InspectLegacyAdapter(config).validate_layout()
    assert "patient_id" in err.value.message and "never guessed" in err.value.message
    assert err.value.details["available_columns"] == ["pid", "when", "split"]
    # an explicit mapping resolves it; a wrong explicit mapping is rejected
    ok = config.model_copy(update={"columns": {"cohort": {"patient_id": "pid", "procedure_time": "when"}}})
    assert InspectLegacyAdapter(ok).validate_layout().resolved_columns["cohort"]["patient_id"] == "pid"
    bad = config.model_copy(update={"columns": {"cohort": {"patient_id": "nope"}}})
    with pytest.raises(DatasetSchemaError, match="not present"):
        InspectLegacyAdapter(bad).validate_layout()


def test_reports_require_an_explicit_report_reference(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cohort.csv", ["patient_id", "procedure_time", "split"], [[1, "2020-01-01", "train"]])
    _write_csv(tmp_path / "imp.csv", ["impression_id", "impressions"], [["a", "No PE."]])
    config = DatasetConfig(name="x", adapter="inspect_legacy", root=str(tmp_path),
                           files={"cohort": "cohort.csv", "impressions": "imp.csv"})
    with pytest.raises(DatasetSchemaError, match="never by timestamps"):
        InspectLegacyAdapter(config).validate_layout()


def test_event_derived_demographics_and_age(tmp_path: Path) -> None:
    """Releases that encode demographics as events are supported through explicit code prefixes."""
    _write_csv(tmp_path / "cohort.csv", ["patient_id", "procedure_time", "split"],
               [[7, "2020-06-01 00:00:00", "train"]])
    _write_csv(tmp_path / "ehr" / "e.csv", ["patient_id", "start", "code", "value", "unit"], [
        [7, "1960-06-01 00:00:00", "SNOMED/3950001", "", ""],
        [7, "1960-06-01 00:00:00", "Gender/F", "", ""],
        [7, "1960-06-01 00:00:00", "Race/Asian", "", ""],
        [7, "2020-05-31 12:00:00", "LOINC/718-7", "12.5", "g/dL"],
    ])
    config = DatasetConfig(name="demo", adapter="inspect_legacy", root=str(tmp_path),
                           files={"cohort": "cohort.csv", "ehr_dir": "ehr"},
                           code_prefixes={"gender": "Gender/", "race": "Race/", "birth": "SNOMED/3950001"})
    engine = make_engine("sqlite://")
    create_all(engine)
    factory = make_session_factory(engine)
    report = IngestionPipeline(config, factory, tmp_path / "art").run()
    assert report.counts["demographic_events"] == 3 and report.counts["events"] == 1
    with factory() as session:
        patient = session.get(Patient, "7")
        case = session.query(ClinicalCase).one()
        assert patient.gender == "F" and patient.race == "Asian"
        assert case.age_at_procedure == pytest.approx(60.0, abs=0.1)


def test_meds_adapter_reads_parquet_and_split_table(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data").mkdir()
    pq.write_table(pa.table({
        "subject_id": [1, 1, 2], "time": ["2020-01-01 08:00:00", "2020-01-02 09:00:00", "2020-02-01 10:00:00"],
        "code": ["LOINC/718-7", "ICD10CM/I26.99", "LOINC/2160-0"], "numeric_value": [13.1, None, 88.4],
        "unit": ["g/dL", None, "umol/L"],
    }), tmp_path / "data" / "shard0.parquet")
    pq.write_table(pa.table({"subject_id": [1, 2], "split": ["train", "held_out"]}), tmp_path / "splits.parquet")
    pq.write_table(pa.table({"patient_id": [1, 2], "procedure_time": ["2020-01-01 12:00:00", "2020-02-01 12:00:00"],
                             "impression_id": ["a", "b"]}), tmp_path / "cohort.parquet")
    pq.write_table(pa.table({"impression_id": ["a", "b"], "impressions": ["No PE.", "Acute PE."]}),
                   tmp_path / "impressions.parquet")
    config = DatasetConfig(name="meds", adapter="inspect_meds", root=str(tmp_path), files={
        "cohort": "cohort.parquet", "subject_splits": "splits.parquet", "ehr_dir": "data",
        "impressions": "impressions.parquet"})
    adapter = build_adapter(config)
    assert isinstance(adapter, InspectMedsAdapter)
    layout = adapter.validate_layout()
    assert set(layout.modalities) >= {"cohort", "ehr", "radiology_report"}
    cohort = list(adapter.iter_cohort())
    assert [c.split_raw for c in cohort] == ["train", "held_out"]
    engine = make_engine("sqlite://")
    create_all(engine)
    report = IngestionPipeline(config, make_session_factory(engine), tmp_path / "art", adapter=adapter).run()
    assert report.counts["cases_split_train"] == 1 and report.counts["cases_split_test"] == 1
    assert report.counts["events"] == 3 and report.counts["reports"] == 2


def test_meds_adapter_requires_split_information(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"patient_id": [1], "procedure_time": ["2020-01-01"]}), tmp_path / "cohort.parquet")
    config = DatasetConfig(name="meds", adapter="inspect_meds", root=str(tmp_path), files={"cohort": "cohort.parquet"})
    with pytest.raises(DatasetSchemaError, match="never assigned implicitly"):
        build_adapter(config).validate_layout()


def test_synthetic_generator_never_overwrites_foreign_data(tmp_path: Path) -> None:
    from app.ingestion.synthetic_generator import generate_synthetic_dataset

    target = tmp_path / "synthetic"
    target.mkdir()
    (target / ".gitkeep").touch()  # the repository placeholder must not block generation
    info = generate_synthetic_dataset(target, n_patients=20)
    assert info["synthetic"] and (target / ".gitkeep").exists() and (target / "cohort.csv").exists()
    first = (target / "cohort.csv").read_bytes()
    generate_synthetic_dataset(target, n_patients=20)  # regenerating a synthetic dataset is allowed
    assert (target / "cohort.csv").read_bytes() == first  # and deterministic
    foreign = tmp_path / "real"
    foreign.mkdir()
    (foreign / "patients.csv").write_text("do not delete")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate_synthetic_dataset(foreign)
    assert (foreign / "patients.csv").read_text() == "do not delete"
