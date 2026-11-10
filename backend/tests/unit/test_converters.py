"""Converters for openly licensed datasets, exercised on tiny generated fixtures (no real data)."""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.core.config import DatasetConfig
from app.core.errors import DatasetSchemaError
from app.core.hashing import make_case_id
from app.ingestion.adapters import build_adapter
from app.ingestion.converters.mimic_demo import prepare_mimic_demo
from app.ingestion.converters.pmc_patients import PLACEHOLDER_TIME, parse_demographics, prepare_pmc_patients
from app.ingestion.inspector import DatasetInspector
from app.ingestion.normalization import LabConceptRegistry


def _jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def pmc_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    corpus = [{"_id": f"100{i}-1", "title": "", "text": f"A {40 + i}-year-old man presented with finding {i}.\nSecond, \"quoted\" line."}
              for i in range(8)]
    _jsonl(raw / "PPR" / "corpus.jsonl", corpus)
    _jsonl(raw / "queries" / "dev_queries.jsonl", [
        {"_id": "2001-1", "text": "A 67-year-old female with dyspnea."},
        {"_id": "2002-1", "text": "An 8-month-old boy with fever."},
        {"_id": "2003-1", "text": "Unlabelled dev patient."}])
    _jsonl(raw / "queries" / "test_queries.jsonl", [{"_id": "3001-1", "text": "A patient of unstated age."}])
    (raw / "PPR" / "qrels_dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\n2001-1\t1000-1\t1\n2001-1\t1003-1\t1\n2002-1\t1005-1\t1\n", encoding="utf-8")
    (raw / "PPR" / "qrels_test.tsv").write_text("query-id\tcorpus-id\tscore\n3001-1\t1007-1\t1\n", encoding="utf-8")
    return raw


def test_parse_demographics_is_conservative() -> None:
    assert parse_demographics("A 67-year-old female with COPD") == (67.0, "F")
    assert parse_demographics("An 8-month-old boy") == (0.67, "M")
    assert parse_demographics("A 45 year old man and his wife, a woman") == (45.0, "M")
    assert parse_demographics("The patient had 300 mg of drug") == (None, None)
    assert parse_demographics("x" * 400 + " 50-year-old female") == (None, None)  # only the opening is read


def test_pmc_patients_subset_keeps_every_judged_relevant_patient(pmc_raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    info = prepare_pmc_patients(pmc_raw, out, dev_queries=2, test_queries=1, corpus_size=6, seed=1)
    assert info["is_subset"] and info["corpus_patients"] == 6 and info["judged_relevant_patients_in_corpus"] == 4
    assert info["dev_query_patients"] == 2 and info["test_query_patients"] == 1 and info["qrel_rows"] == 4
    cohort = list(csv.DictReader((out / "cohort.csv").open(encoding="utf-8")))
    by_id = {r["patient_id"]: r for r in cohort}
    assert {"1000-1", "1003-1", "1005-1", "1007-1"} <= set(by_id)          # relevant patients are never dropped
    assert "2003-1" not in by_id                                           # unlabelled queries are not sampled
    assert by_id["2001-1"]["split"] == "valid" and by_id["3001-1"]["split"] == "test"
    assert by_id["1000-1"]["split"] == "train" and by_id["2001-1"]["age"] == "67.0" and by_id["3001-1"]["age"] == ""
    texts = {r["impression_id"]: r["impressions"] for r in csv.DictReader((out / "impressions.csv").open(encoding="utf-8"))}
    assert texts["pmc_1000-1"].endswith('Second, "quoted" line.')           # multi-line text survives CSV
    qrels = [json.loads(line) for line in (out / "eval" / "qrels.jsonl").read_text().splitlines()]
    assert {q["candidate_case_id"] for q in qrels if q["query_id"] == "dev_2001-1"} == {
        make_case_id("1000-1", PLACEHOLDER_TIME), make_case_id("1003-1", PLACEHOLDER_TIME)}
    assert all(q["label_source"] == "pmc_patients_ppr" and q["relevance_grade"] == 1 for q in qrels)
    queries = [json.loads(line) for line in (out / "eval" / "queries.jsonl").read_text().splitlines()]
    assert {q["stage"] for q in queries} == {"dev", "test"} and all(q["query_mode"] == "case" for q in queries)
    # the same seed reproduces the same subset
    again = prepare_pmc_patients(pmc_raw, tmp_path / "again", dev_queries=2, test_queries=1, corpus_size=6, seed=1)
    assert (tmp_path / "again" / "cohort.csv").read_text() == (out / "cohort.csv").read_text() and again == info

    with pytest.raises(ValueError, match="smaller than"):
        prepare_pmc_patients(pmc_raw, tmp_path / "bad", dev_queries=2, test_queries=1, corpus_size=2)
    with pytest.raises(FileNotFoundError):
        prepare_pmc_patients(tmp_path / "nowhere", tmp_path / "bad2")


def test_prepared_adapter_requires_matching_marker(pmc_raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    prepare_pmc_patients(pmc_raw, out, dev_queries=None, test_queries=None, corpus_size=None)
    assert DatasetInspector(out, hash_files=False).inspect()["detected_layout"] == "prepared"
    adapter = build_adapter(DatasetConfig(name="pmc_patients", adapter="prepared", root=str(out)))
    report = adapter.validate_layout()
    assert report.schema_version == "prepared-pmc-patients-1" and any("not INSPECT" in w for w in report.warnings)
    assert sum(1 for _ in adapter.iter_cohort()) == 11 and sum(1 for _ in adapter.iter_reports()) == 11
    with pytest.raises(DatasetSchemaError, match="describes dataset"):
        build_adapter(DatasetConfig(name="something_else", adapter="prepared", root=str(out))).validate_layout()
    with pytest.raises(DatasetSchemaError, match="not a prepared dataset"):
        build_adapter(DatasetConfig(name="pmc_patients", adapter="prepared", root=str(tmp_path))).validate_layout()


def _meds_fixture(root: Path) -> None:
    t = datetime
    rows = [
        # subject 1 (train): one stay
        (1, None, "GENDER//F", None, None, None),
        (1, t(2100, 1, 1), "MEDS_BIRTH", None, None, None),
        (1, t(2150, 3, 1, 8), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None, 11, "WHITE"),
        (1, t(2150, 3, 2, 6), "LAB//50912//mg/dL", 1.2000000476837158, None, None),
        (1, t(2150, 3, 2, 6), "LAB//50971//mEq/L", 4.1, None, None),
        (1, t(2150, 3, 2, 7), "LAB//220045//bpm", 88.0, None, None),
        (1, t(2150, 3, 2, 7), "LAB//999999//UNK", 3.0, None, None),          # unmapped chart item
        (1, t(2150, 3, 2, 7), "LAB//51237//UNK", 1.1, None, None),            # mapped, unit unknown
        (1, t(2150, 3, 2, 7), "LAB//50912//mg/dL", None, None, None),         # no numeric value
        (1, t(2150, 3, 2, 9), "MEDICATION//START//Heparin", None, None, None),
        (1, t(2150, 3, 4, 12), "HOSPITAL_DISCHARGE//HOME", None, 11, None),
        (1, t(2150, 3, 4, 12), "DIAGNOSIS//ICD//10//I2699", None, 11, None),
        (1, t(2150, 3, 4, 12), "DIAGNOSIS//ICD//9//0000", None, 11, None),    # no parent code
    ]
    schema = pa.schema([("subject_id", pa.int64()), ("time", pa.timestamp("us")), ("code", pa.string()),
                        ("numeric_value", pa.float32()), ("hadm_id", pa.int64()), ("race", pa.large_string())])
    cols = list(zip(*rows, strict=True))
    for split, subject in (("train", 1), ("tuning", 2), ("held_out", 3)):
        (root / "data" / split).mkdir(parents=True)
        data = [[subject if i == 0 else v for v in col] if i == 0 else list(col) for i, col in enumerate(cols)]
        pq.write_table(pa.table(data, schema=schema), root / "data" / split / "0.parquet")
    (root / "metadata").mkdir()
    pq.write_table(pa.table({
        "code": ["LAB//50912//mg/dL", "LAB//50971//mEq/L", "LAB//51237//UNK", "DIAGNOSIS//ICD//10//I2699"],
        "parent_codes": [["LOINC/2160-0"], ["LOINC/2823-3"], ["LOINC/6301-6"], ["ICD10CM/I26.99"]],
    }), root / "metadata" / "codes.parquet")
    pq.write_table(pa.table({"subject_id": [1, 2, 3], "split": ["train", "tuning", "held_out"]}),
                   root / "metadata" / "subject_splits.parquet")


def test_mimic_demo_conversion_maps_codes_units_and_counts_what_it_skips(tmp_path: Path) -> None:
    meds, out = tmp_path / "meds", tmp_path / "prepared"
    _meds_fixture(meds)
    info = prepare_mimic_demo(meds, out)
    assert info["cases"] == 3 and info["counts"]["hospital_stays"] == 3
    assert info["counts"]["laboratory_and_vital_events"] == 12 and info["counts"]["diagnosis_events"] == 3
    assert info["skipped"]["chart/lab item without LOINC mapping"] == 3
    assert info["skipped"]["laboratory result without numeric value"] == 3
    assert info["skipped"]["diagnosis without ICD parent code"] == 3 and info["skipped"]["MEDICATION events"] == 3
    cohort = list(csv.DictReader((out / "cohort.csv").open(encoding="utf-8")))
    assert [(r["patient_id"], r["split"]) for r in cohort] == [("1", "train"), ("2", "tuning"), ("3", "held_out")]
    assert cohort[0] == {"patient_id": "1", "procedure_time": "2150-03-04 12:00:00", "split": "train",
                         "age": "50", "gender": "F", "race": "WHITE"}
    with gzip.open(out / "ehr" / "events_train.csv.gz", "rt", encoding="utf-8") as fh:
        events = list(csv.DictReader(fh))
    by_code = {e["code"]: e for e in events}
    assert by_code["LOINC/2160-0"]["value"] == "1.2" and by_code["LOINC/2160-0"]["unit"] == "mg/dL"   # float32 noise removed
    assert by_code["LOINC/8867-4"]["unit"] == "bpm" and by_code["LOINC/6301-6"]["unit"] == ""           # UNK -> no unit
    assert by_code["ICD10CM/I26.99"]["start"] == "2150-03-04 12:00:00"
    # the registry turns the converted units into comparable values
    registry = LabConceptRegistry()
    potassium = registry.by_code("LOINC/2823-3")
    assert potassium is not None and potassium.to_canonical(4.1, "mEq/L") == (4.1, None)
    heart_rate = registry.by_code("LOINC/8867-4")
    assert heart_rate is not None and heart_rate.to_canonical(88.0, "bpm") == (88.0, None)
    adapter = build_adapter(DatasetConfig(name="mimic_iv_demo", adapter="prepared", root=str(out)))
    report = adapter.validate_layout()
    assert "ehr" in report.modalities and "radiology_report" not in report.modalities
