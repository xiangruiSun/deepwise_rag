"""MIMIC-IV Clinical Database Demo (MEDS format) -> canonical CSV layout.

Source: https://physionet.org/content/mimic-iv-demo-meds/0.0.1/ (ODbL 1.0), 100 patients.
Johnson et al., MIMIC-IV Clinical Database Demo; van de Water et al., MIMIC-IV demo data
in the Medical Event Data Standard (MEDS), PhysioNet (2025).

Mapping decisions:

* **One case per hospital stay**, indexed at the discharge time.  The case therefore
  describes a completed stay: laboratory values inside the look-back window before
  discharge and the diagnosis codes recorded at discharge are "at or before" the index
  time; anything from later stays is after it and never used.
* The MEDS subject splits are kept (train / tuning / held_out).
* Laboratory events are kept when the MEDS code table links them to a LOINC code; the
  unit is the one carried in the MEDS code (``LAB//<itemid>//<unit>``).  Four bedside
  vital signs without a LOINC link are mapped through an explicit item table below.
* Diagnoses use the ICD-9-CM / ICD-10-CM parent code of the MEDS code table.
* Everything else (medications, infusions, transfers, procedures, free-text results,
  unmapped chart items) is counted and skipped.
* The demo contains no clinical notes and no images, so cases have no text record.
* Age is the discharge year minus the (year-only) birth year of the MEDS file.
* Nothing is downloaded here.  The extracted MEDS folder must already be on disk.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

CONVERTER_VERSION = "mimic-demo-meds-1"
SPLITS = ("train", "tuning", "held_out")
# MIMIC-IV chart items (itemid -> LOINC) for vital signs that have no LOINC parent in the
# MEDS code table.  Only non-invasive readings, so different methods are not mixed.
VITAL_ITEMS = {
    "220045": "LOINC/8867-4",   # Heart Rate
    "220210": "LOINC/9279-1",   # Respiratory Rate
    "220277": "LOINC/59408-5",  # O2 saturation pulseoxymetry
    "220179": "LOINC/8480-6",   # Non Invasive Blood Pressure systolic
    "220180": "LOINC/8462-4",   # Non Invasive Blood Pressure diastolic
}


def _fmt(ts: Any) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def prepare_mimic_demo(meds_dir: Path, output_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    meds_dir, output_dir = Path(meds_dir), Path(output_dir)
    needed = [meds_dir / "data" / s / "0.parquet" for s in SPLITS] + [
        meds_dir / "metadata" / "codes.parquet", meds_dir / "metadata" / "subject_splits.parquet"]
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(f"MIMIC-IV demo (MEDS) file is missing: {path}")

    codes = pq.read_table(meds_dir / "metadata" / "codes.parquet", columns=["code", "parent_codes"]).to_pylist()
    parents: dict[str, list[str]] = {
        row["code"]: [p for p in (row["parent_codes"] or []) if p] for row in codes
    }
    splits = {
        str(row["subject_id"]): row["split"]
        for row in pq.read_table(meds_dir / "metadata" / "subject_splits.parquet").to_pylist()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ehr").mkdir(exist_ok=True)
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    cohort_rows: list[list[Any]] = []
    columns = ["subject_id", "time", "code", "numeric_value", "hadm_id", "race"]
    for split in SPLITS:
        table = pq.read_table(meds_dir / "data" / split / "0.parquet", columns=columns)
        rows = table.to_pylist()
        birth_year: dict[str, int] = {}
        gender: dict[str, str] = {}
        race_of_stay: dict[Any, str] = {}
        for row in rows:
            subject, code = str(row["subject_id"]), row["code"]
            if code == "MEDS_BIRTH" and row["time"] is not None:
                birth_year[subject] = row["time"].year
            elif code.startswith("GENDER//"):
                gender[subject] = code.split("//", 1)[1]
            elif code.startswith("HOSPITAL_ADMISSION") and row["race"]:
                race_of_stay[row["hadm_id"]] = row["race"]
        with gzip.open(output_dir / "ehr" / f"events_{split}.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            events = csv.writer(fh)
            events.writerow(["patient_id", "start", "code", "value", "unit"])
            for row in rows:
                subject, code, time = str(row["subject_id"]), row["code"], row["time"]
                prefix = code.split("//", 1)[0]
                if time is None:
                    continue  # static rows (gender) were read above
                if prefix == "HOSPITAL_DISCHARGE":
                    age = time.year - birth_year[subject] if subject in birth_year else ""
                    cohort_rows.append([subject, _fmt(time), splits.get(subject, split), age,
                                        gender.get(subject, ""), race_of_stay.get(row["hadm_id"], "")])
                    counts["hospital_stays"] += 1
                elif prefix == "DIAGNOSIS":
                    icd = next((p for p in parents.get(code, []) if p.startswith(("ICD9CM/", "ICD10CM/"))), None)
                    if icd is None:
                        skipped["diagnosis without ICD parent code"] += 1
                        continue
                    events.writerow([subject, _fmt(time), icd, "", ""])
                    counts["diagnosis_events"] += 1
                elif prefix == "LAB":
                    parts = code.split("//")
                    item, unit = parts[1], parts[2] if len(parts) > 2 else ""
                    loinc = next((p for p in parents.get(code, []) if p.startswith("LOINC/")), None)
                    loinc = loinc or VITAL_ITEMS.get(item)
                    if loinc is None:
                        skipped["chart/lab item without LOINC mapping"] += 1
                        continue
                    value = row["numeric_value"]
                    if value is None or value != value:  # None or NaN
                        skipped["laboratory result without numeric value"] += 1
                        continue
                    events.writerow([subject, _fmt(time), loinc, format(float(value), ".6g"),
                                     "" if unit == "UNK" else unit])
                    counts["laboratory_and_vital_events"] += 1
                else:
                    skipped[f"{prefix} events"] += 1

    seen: set[tuple[str, str]] = set()
    with (output_dir / "cohort.csv").open("w", encoding="utf-8", newline="") as fh:
        cohort = csv.writer(fh)
        cohort.writerow(["patient_id", "procedure_time", "split", "age", "gender", "race"])
        for row in sorted(cohort_rows, key=lambda r: (r[0], r[1])):
            if (row[0], row[1]) in seen:
                skipped["duplicate discharge rows"] += 1
                continue
            seen.add((row[0], row[1]))
            cohort.writerow(row)

    info = {
        "dataset": "mimic_iv_demo",
        "prepared": True,
        "converter_version": CONVERTER_VERSION,
        "source": "MIMIC-IV Clinical Database Demo in the Medical Event Data Standard (MEDS) 0.0.1",
        "source_url": "https://physionet.org/content/mimic-iv-demo-meds/0.0.1/",
        "license": "Open Data Commons Open Database License v1.0",
        "citation": "van de Water RP, Steinberg E, Wornow M, Rockenschaub P, McDermott M. MIMIC-IV demo data in "
                    "the Medical Event Data Standard (MEDS), version 0.0.1. PhysioNet (2025); "
                    "Johnson A et al. MIMIC-IV Clinical Database Demo, version 2.2. PhysioNet (2023).",
        "case_definition": "one case per hospital stay, indexed at the discharge time",
        "subjects": len(splits),
        "cases": len(seen),
        "counts": dict(counts),
        "skipped": dict(sorted(skipped.items())),
        "notice": (
            "MIMIC-IV demo (100 patients), not INSPECT. Structured data only: no clinical notes, no images "
            "and no relevance labels, so text retrieval and retrieval-quality metrics are not available."
        ),
    }
    (output_dir / "DATASET.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
