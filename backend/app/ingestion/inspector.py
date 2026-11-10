"""Schema inspection before ingestion (spec section 5.2).

``DatasetInspector`` discovers files, reads only headers / Parquet metadata /
DICOM headers (never whole tables), proposes candidate identifier columns and
recognises known layouts.  It produces the first half of ``data_manifest.json``;
the ingestion pipeline adds measured record and join counts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.hashing import file_sha256, stable_hash
from app.ingestion.adapters.base import open_text, sniff_delimiter
from app.ingestion.adapters.synthetic import SYNTHETIC_MARKER

PATIENT_ID_CANDIDATES = ("patient_id", "person_id", "subject_id", "patientid", "mrn", "anon_id")
STUDY_ID_CANDIDATES = (
    "image_id", "study_id", "impression_id", "accession", "accession_number",
    "studyinstanceuid", "series_id", "procedure_time", "studytime",
)
SPLIT_CANDIDATES = ("split", "dataset_split", "fold")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")
HASH_LIMIT_BYTES = 256 * 1024 * 1024


@dataclass
class FileInfo:
    path: str
    kind: str
    size_bytes: int
    sha256: str | None = None
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    rows: int | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "kind": self.kind, "size_bytes": self.size_bytes,
            "sha256": self.sha256, "columns": self.columns, "dtypes": self.dtypes,
            "rows": self.rows, "notes": self.notes,
        }


def _kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(ARCHIVE_SUFFIXES):
        return "archive"
    if name.endswith((".csv", ".csv.gz")):
        return "csv"
    if name.endswith((".tsv", ".tsv.gz")):
        return "tsv"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith((".dcm", ".dicom")):
        return "dicom"
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if name.endswith(".npy"):
        return "numpy"
    if name.endswith((".json", ".jsonl", ".md", ".txt", ".yaml", ".yml")):
        return "metadata"
    return "dicom" if _looks_like_dicom(path) else "other"


def _looks_like_dicom(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(132)
        return len(head) == 132 and head[128:132] == b"DICM"
    except OSError:
        return False


def _infer_type(values: list[str]) -> str:
    present = [v for v in values if v not in ("", None)]
    if not present:
        return "empty"
    for caster, label in ((int, "integer"), (float, "float")):
        try:
            for v in present:
                caster(v)
            return label
        except ValueError:
            continue
    return "string"


class DatasetInspector:
    def __init__(self, root: Path, sample_rows: int = 200, hash_files: bool = True) -> None:
        self.root = Path(root)
        self.sample_rows = sample_rows
        self.hash_files = hash_files

    def discover(self) -> list[FileInfo]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        infos: list[FileInfo] = []
        dicom_dirs: dict[Path, list[Path]] = {}
        for path in sorted(p for p in self.root.rglob("*") if p.is_file()):
            if path.name.startswith(".") or path.name == "data_manifest.json":
                continue
            kind = _kind(path)
            if kind == "dicom":
                dicom_dirs.setdefault(path.parent, []).append(path)
                continue
            infos.append(self._inspect_file(path, kind))
        for directory, files in sorted(dicom_dirs.items()):
            infos.append(self._inspect_dicom_series(directory, files))
        return infos

    def _inspect_file(self, path: Path, kind: str) -> FileInfo:
        info = FileInfo(path=str(path.relative_to(self.root)), kind=kind, size_bytes=path.stat().st_size)
        if self.hash_files:
            info.sha256 = file_sha256(path, max_bytes=HASH_LIMIT_BYTES)
        try:
            if kind in ("csv", "tsv"):
                self._inspect_delimited(path, info)
            elif kind == "parquet":
                self._inspect_parquet(path, info)
            elif kind == "archive":
                info.notes.append("compressed archive: extract before ingestion")
        except Exception as exc:  # inspection must report, not crash
            info.notes.append(f"could not inspect: {type(exc).__name__}: {exc}")
        return info

    def _inspect_delimited(self, path: Path, info: FileInfo) -> None:
        with open_text(path) as fh:
            reader = csv.reader(fh, delimiter=sniff_delimiter(path))
            header = next(reader, None)
            if header is None:
                info.notes.append("empty file")
                return
            info.columns = [h.strip() for h in header]
            samples: list[list[str]] = []
            for i, row in enumerate(reader):
                if i >= self.sample_rows:
                    break
                samples.append(row)
        for idx, column in enumerate(info.columns):
            info.dtypes[column] = _infer_type([r[idx] for r in samples if idx < len(r)])
        info.notes.append(f"dtypes inferred from first {len(samples)} rows")

    def _inspect_parquet(self, path: Path, info: FileInfo) -> None:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        info.columns = list(schema.names)
        info.dtypes = {f.name: str(f.type) for f in schema}
        info.rows = parquet_file.metadata.num_rows

    def _inspect_dicom_series(self, directory: Path, files: list[Path]) -> FileInfo:
        info = FileInfo(
            path=str(directory.relative_to(self.root)),
            kind="dicom_series",
            size_bytes=sum(f.stat().st_size for f in files),
            rows=len(files),
        )
        if self.hash_files:
            info.sha256 = stable_hash(
                [(f.name, f.stat().st_size) for f in files]
            )
            info.notes.append("series hash = hash of slice names and sizes")
        try:
            import pydicom

            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            info.dtypes = {
                "Modality": str(getattr(ds, "Modality", "")),
                "Rows": str(getattr(ds, "Rows", "")),
                "Columns": str(getattr(ds, "Columns", "")),
            }
        except Exception as exc:
            info.notes.append(f"could not read DICOM header: {type(exc).__name__}")
        return info

    # ------------------------------------------------------------------ layout
    def detect_layout(self, infos: list[FileInfo]) -> tuple[str, list[str]]:
        warnings: list[str] = []
        if (self.root / SYNTHETIC_MARKER).exists():
            return "synthetic", warnings
        if (self.root / "DATASET.json").exists():
            return "prepared", warnings
        tables = [i for i in infos if i.kind in ("csv", "tsv", "parquet")]
        lower_cols = [{c.lower() for c in i.columns} for i in tables]
        has_meds = any(
            i.kind == "parquet" and {"subject_id", "time", "code"} <= cols
            for i, cols in zip(tables, lower_cols, strict=True)
        )
        has_femr = any(
            i.kind in ("csv", "tsv") and {"patient_id", "start", "code"} <= cols
            for i, cols in zip(tables, lower_cols, strict=True)
        )
        has_cohort = any(
            ({"patient_id", "procedure_time"} <= cols) or ({"person_id", "procedure_datetime"} <= cols)
            for cols in lower_cols
        )
        if has_meds:
            if not has_cohort:
                warnings.append("MEDS event shards found but no cohort table was recognised")
            return "inspect_meds", warnings
        if has_femr or has_cohort:
            if not has_cohort:
                warnings.append("FEMR-style event files found but no cohort table was recognised")
            return "inspect_legacy", warnings
        warnings.append(
            "layout not recognised: configure dataset.files and dataset.columns explicitly"
        )
        return "unknown", warnings

    def inspect(self) -> dict[str, Any]:
        infos = self.discover()
        layout, warnings = self.detect_layout(infos)
        kinds = {i.kind for i in infos}
        id_candidates: dict[str, dict[str, list[str]]] = {}
        split_files: list[str] = []
        for info in infos:
            lowered = {c.lower(): c for c in info.columns}
            patient = [lowered[c] for c in PATIENT_ID_CANDIDATES if c in lowered]
            study = [lowered[c] for c in STUDY_ID_CANDIDATES if c in lowered]
            if patient or study:
                id_candidates[info.path] = {"patient_id": patient, "study_id": study}
            if any(c in lowered for c in SPLIT_CANDIDATES):
                split_files.append(info.path)
        modalities = []
        if any(i.kind in ("csv", "tsv", "parquet") for i in infos):
            modalities.append("tabular")
        if kinds & {"dicom_series", "nifti", "numpy"}:
            modalities.append("imaging")
        if "imaging" not in modalities:
            warnings.append("missing modality: no DICOM, NIfTI or NumPy imaging files discovered")
        if not split_files:
            warnings.append("no cohort/split metadata detected in any table")
        if "archive" in kinds:
            warnings.append("compressed archives present: contents were not inspected")
        return {
            "manifest_version": 1,
            "root": str(self.root),
            "inspected_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "detected_layout": layout,
            "files": [i.as_dict() for i in infos],
            "file_kinds": {k: sum(1 for i in infos if i.kind == k) for k in sorted(kinds)},
            "candidate_identifiers": id_candidates,
            "split_metadata_files": split_files,
            "available_modalities": modalities,
            "validation_warnings": warnings,
        }


def dataset_fingerprint(manifest: dict[str, Any]) -> str:
    """Fingerprint of the *content* of a dataset (paths + hashes), independent of time."""
    return stable_hash(
        sorted((f["path"], f.get("sha256"), f["size_bytes"]) for f in manifest.get("files", []))
    )


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
