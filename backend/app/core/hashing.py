"""Deterministic hashing utilities used for identities, caches and fingerprints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any, length: int = 64) -> str:
    """Hash any JSON-serialisable object independent of dict key order."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)[:length]


def file_sha256(path: Path, chunk_size: int = 1 << 20, max_bytes: int | None = None) -> str:
    """Streamed file hash.  ``max_bytes`` bounds work for very large image files; the
    resulting digest is then prefixed so it cannot be confused with a full hash."""
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            if max_bytes is not None and read + len(block) > max_bytes:
                digest.update(block[: max_bytes - read])
                digest.update(str(path.stat().st_size).encode())
                return f"partial{max_bytes}:{digest.hexdigest()}"
            digest.update(block)
            read += len(block)
    return digest.hexdigest()


def normalize_procedure_time(value: datetime | str) -> str:
    """Canonical, timezone-naive ISO representation with second precision.

    INSPECT timestamps are patient-specifically date shifted and carry no timezone;
    we therefore refuse to guess one and only canonicalise the textual form.
    """
    if isinstance(value, str):
        value = parse_timestamp(value)
    if value.tzinfo is not None:
        value = value.replace(tzinfo=None)
    return value.replace(microsecond=0).isoformat(timespec="seconds")


_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)


def parse_timestamp(value: Any) -> datetime:
    """Strict timestamp parser.  Raises ``ValueError`` on anything ambiguous."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if value is None:
        raise ValueError("timestamp is missing")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        raise ValueError("timestamp is missing")
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp: {text!r}") from exc


def make_case_id(patient_id: str, procedure_time: datetime | str) -> str:
    """Deterministic internal case identity from ``patient_id`` + ``procedure_time``."""
    canonical = f"{str(patient_id).strip()}|{normalize_procedure_time(procedure_time)}"
    return "case_" + sha256_text(canonical)[:20]


def make_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_" + sha256_text("|".join(str(p) for p in parts))[:20]
