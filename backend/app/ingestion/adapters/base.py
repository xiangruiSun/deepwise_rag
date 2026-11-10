"""Dataset adapter interface and shared column-resolution helpers."""

from __future__ import annotations

import csv
import gzip
import io
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from app.core.config import REPO_ROOT, DatasetConfig
from app.core.errors import DatasetSchemaError
from app.ingestion.canonical import CohortRow, EventRow, ImagingRow, ReportRow


@dataclass
class LayoutReport:
    adapter: str
    schema_version: str
    files: dict[str, list[str]] = field(default_factory=dict)
    resolved_columns: dict[str, dict[str, str]] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def open_text(path: Path) -> IO[str]:
    """Open plain or gzip-compressed text as a stream (never loads the whole file)."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def sniff_delimiter(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tsv", ".tsv.gz")):
        return "\t"
    return ","


def read_header(path: Path) -> list[str]:
    with open_text(path) as fh:
        reader = csv.reader(fh, delimiter=sniff_delimiter(path))
        try:
            return [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise DatasetSchemaError(f"{path.name}: file is empty (no header row)") from exc


def iter_dict_rows(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield ``(row_number, row)`` with 1-based data row numbers, streaming."""
    with open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter=sniff_delimiter(path))
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        yield from enumerate(reader, start=1)


def resolve_columns(
    table: str,
    header: list[str],
    required: Mapping[str, tuple[str, ...]],
    optional: Mapping[str, tuple[str, ...]],
    explicit: dict[str, str] | None,
    file_name: str,
) -> dict[str, str]:
    """Map logical field names to actual columns.

    Resolution order: (1) explicit mapping from the dataset config, which must
    exist in the file; (2) an exact (case-insensitive) match against the listed
    known aliases.  There is no fuzzy matching: an unresolved required field is
    a hard error that tells the user exactly which mapping to add.
    """
    explicit = explicit or {}
    lower = {h.lower(): h for h in header}
    resolved: dict[str, str] = {}
    for logical, column in explicit.items():
        if logical not in required and logical not in optional:
            raise DatasetSchemaError(
                f"{table}: config maps unknown logical field '{logical}'",
                {"known_fields": sorted({*required, *optional})},
            )
        if column not in header:
            raise DatasetSchemaError(
                f"{table}: configured column '{column}' for field '{logical}' "
                f"is not present in {file_name}",
                {"available_columns": header},
            )
        resolved[logical] = column
    for logical, aliases in {**required, **optional}.items():
        if logical in resolved:
            continue
        for alias in aliases:
            if alias.lower() in lower:
                resolved[logical] = lower[alias.lower()]
                break
    missing = [name for name in required if name not in resolved]
    if missing:
        raise DatasetSchemaError(
            f"{table}: required field(s) {missing} could not be resolved in {file_name}. "
            f"Add an explicit mapping under dataset.columns.{table} in the dataset config; "
            "column meanings are never guessed.",
            {"available_columns": header, "accepted_aliases": {m: required[m] for m in missing}},
        )
    return resolved


class DatasetAdapter(ABC):
    """Transforms one dataset layout into canonical rows."""

    name: str = "base"
    schema_version: str = "unknown"

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        root = Path(config.root)
        self.root = root if root.is_absolute() else REPO_ROOT / root

    @abstractmethod
    def validate_layout(self) -> LayoutReport:
        """Check the layout and resolve columns; raise ``DatasetSchemaError`` if invalid."""

    @abstractmethod
    def iter_cohort(self) -> Iterator[CohortRow]: ...

    @abstractmethod
    def iter_reports(self) -> Iterator[ReportRow]: ...

    @abstractmethod
    def event_sources(self) -> list[Path]:
        """Files holding clinical events (each is hashed for incremental ingestion)."""

    @abstractmethod
    def iter_events(self, source: Path) -> Iterator[EventRow]: ...

    @abstractmethod
    def iter_imaging(self) -> Iterator[ImagingRow]: ...

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return path.name

    def _file(self, key: str, required: bool = True) -> Path | None:
        configured = self.config.files.get(key)
        if not configured:
            if required:
                raise DatasetSchemaError(
                    f"dataset config is missing files.{key}; run `inspect-rag inspect-dataset` "
                    "to list the discovered files and set the path explicitly"
                )
            return None
        path = self.root / configured
        if not path.exists():
            if required:
                raise DatasetSchemaError(f"files.{key} does not exist: {configured}")
            return None
        return path


def nullable(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" or text.lower() in {"nan", "none", "null", "na"} else text


def nullable_float(value: str | None) -> float | None:
    text = nullable(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def nullable_int(value: str | None) -> int | None:
    number = nullable_float(value)
    return None if number is None else int(number)
