"""Corpus ETL: source rows -> validated canonical documents -> document registry.

Two stages, each checkpointed per partition so an interrupted run resumes and a
repeated run changes nothing:

1. **normalise** — read a source in bounded batches, validate and normalise each
   row with ``normalize_row`` and write partitioned Parquet plus dead letters.
   The same pure function runs inside the PySpark job (``spark_job.py``), so the
   local and the distributed engine cannot drift apart.
2. **load** — apply normalised documents to the registry with forward-only
   version semantics (``apply_documents``).

Nothing here loads a whole dataset into memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import StrictModel
from app.core.hashing import sha256_text
from app.corpus.schema import PUBLIC_GROUP, SCHEMA_VERSION, CanonicalDocument, make_document_id
from app.db.models import CorpusDeadLetter, CorpusDocument, CorpusIngestPartition

# reason codes of rejected rows (stable identifiers used in reports and tests)
MISSING_IDENTIFIER = "missing_identifier"
MISSING_TENANT = "missing_tenant"
INVALID_TIMESTAMP = "invalid_timestamp"
INVALID_VERSION = "invalid_version"
EMPTY_CONTENT = "empty_content"
INVALID_METADATA = "invalid_metadata"
INVALID_JSON = "invalid_json"
INVALID_VALIDITY = "invalid_validity_range"
VERSION_CONFLICT = "version_conflict"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TS_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d")

NORMALIZED_SCHEMA = pa.schema([
    ("tenant_id", pa.string()), ("document_id", pa.string()), ("source_system", pa.string()),
    ("source_record_id", pa.string()), ("document_type", pa.string()), ("title", pa.string()),
    ("content", pa.string()), ("content_hash", pa.string()),
    ("source_updated_at", pa.timestamp("us")), ("ingested_at", pa.timestamp("us")),
    ("event_time", pa.timestamp("us")), ("valid_from", pa.timestamp("us")), ("valid_to", pa.timestamp("us")),
    ("document_version", pa.int64()), ("schema_version", pa.string()),
    ("access_policy", pa.list_(pa.string())), ("source_uri", pa.string()), ("is_deleted", pa.bool_()),
    ("extension_json", pa.string()),
])


class SourceSpec(StrictModel):
    """Where the rows come from and how their columns map to canonical fields."""

    uri: str
    format: Literal["parquet", "jsonl", "csv", "text", "sql"] = "parquet"
    # canonical field -> source column; unspecified fields use the canonical generator names
    mapping: dict[str, str] = Field(default_factory=dict)
    # constants for fields the source does not carry (e.g. tenant, source_system, doc_type)
    defaults: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=5_000, ge=1, le=200_000)
    # format == "sql": a SELECT statement; ``uri`` is the SQLAlchemy URL
    query: str | None = None


_DEFAULT_MAPPING = {
    "tenant_id": "tenant", "source_system": "source_system", "source_record_id": "record_id",
    "document_type": "doc_type", "title": "title", "content": "body", "source_updated_at": "updated_at",
    "event_time": "event_time", "valid_from": "valid_from", "valid_to": "valid_to",
    "document_version": "version", "access_policy": "acl", "is_deleted": "deleted", "extension": "extra_json",
}


def clean_text(value: Any) -> str:
    """NFC-normalised UTF-8 text without control characters; whitespace runs are kept as-is."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub(" ", text).strip()


def parse_ts(value: Any) -> datetime | None:
    """Strict timestamp parsing; raises ``ValueError`` instead of guessing."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value
    text = str(value).strip().replace("Z", "")
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp of length {len(text)}")


def normalize_row(row: dict[str, Any], spec: SourceSpec, source_uri: str,
                  ingested_at: datetime) -> tuple[CanonicalDocument | None, str | None, dict[str, Any]]:
    """Validate and normalise one source row.

    Returns ``(document, None, {})`` or ``(None, reason_code, safe_details)``.  The details
    never contain document content."""
    mapping = {**_DEFAULT_MAPPING, **spec.mapping}

    def get(name: str) -> Any:
        value = row.get(mapping[name])
        return spec.defaults.get(name) if value is None or value == "" else value

    record_id = clean_text(get("source_record_id"))
    tenant = clean_text(get("tenant_id"))
    source_system = clean_text(get("source_system")) or "unknown"
    safe = {"source_system": source_system, "tenant_id": tenant or None}
    if not tenant:
        return None, MISSING_TENANT, safe
    if not record_id:
        return None, MISSING_IDENTIFIER, safe
    safe["source_record_id"] = record_id
    try:
        version = int(str(get("document_version") or "1").strip())
        if version < 1:
            raise ValueError
    except ValueError:
        return None, INVALID_VERSION, safe
    try:
        updated = parse_ts(get("source_updated_at"))
        event_time = parse_ts(get("event_time"))
        valid_from, valid_to = parse_ts(get("valid_from")), parse_ts(get("valid_to"))
    except ValueError:
        return None, INVALID_TIMESTAMP, safe
    if valid_from and valid_to and valid_to < valid_from:
        return None, INVALID_VALIDITY, safe
    deleted = bool(get("is_deleted")) if not isinstance(get("is_deleted"), str) else \
        str(get("is_deleted")).strip().lower() in {"1", "true", "yes", "y"}
    content = clean_text(get("content"))
    if not content and not deleted:
        return None, EMPTY_CONTENT, safe
    extension = get("extension") or {}
    if isinstance(extension, str):
        try:
            extension = json.loads(extension)
        except json.JSONDecodeError:
            return None, INVALID_METADATA, safe
    if not isinstance(extension, dict):
        return None, INVALID_METADATA, safe
    acl = get("access_policy")
    if isinstance(acl, str):
        acl = [part.strip() for part in acl.split(",")]
    groups = sorted({clean_text(g) for g in (acl or []) if clean_text(g)}) or [PUBLIC_GROUP]
    title = clean_text(get("title"))
    return CanonicalDocument(
        tenant_id=tenant, document_id=make_document_id(tenant, source_system, record_id),
        source_system=source_system, source_record_id=record_id,
        document_type=clean_text(get("document_type")) or "document", title=title, content=content,
        content_hash=sha256_text(f"{title}\n{content}\n{int(deleted)}"),
        source_updated_at=updated, ingested_at=ingested_at, event_time=event_time or updated,
        valid_from=valid_from, valid_to=valid_to, document_version=version, schema_version=SCHEMA_VERSION,
        access_policy=groups, source_uri=source_uri, is_deleted=deleted, extension=extension,
    ), None, {}


# --------------------------------------------------------------------------- #
# Streaming readers
# --------------------------------------------------------------------------- #
def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_partitions(spec: SourceSpec) -> list[tuple[str, Path | None]]:
    """``(partition_key, path)`` per unit of work.  A SQL source is one partition."""
    if spec.format == "sql":
        return [("query", None)]
    root = Path(spec.uri)
    if root.is_file():
        return [(root.name, root)]
    pattern = {"parquet": "*.parquet", "jsonl": "*.jsonl", "csv": "*.csv", "text": "*.txt"}[spec.format]
    return [(str(p.relative_to(root)), p) for p in sorted(root.rglob(pattern))]


def iter_rows(spec: SourceSpec, path: Path | None) -> Iterator[tuple[int, dict[str, Any] | None]]:
    """Yield ``(row_number, row)``; ``row`` is ``None`` for a line that is not valid JSON."""
    if spec.format == "parquet":
        assert path is not None
        number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=spec.batch_size):
            for row in batch.to_pylist():
                number += 1
                yield number, row
    elif spec.format == "jsonl":
        assert path is not None
        with path.open(encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    yield number, value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    yield number, None
    elif spec.format == "csv":
        assert path is not None
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            yield from enumerate(csv.DictReader(fh), start=1)
    elif spec.format == "text":
        assert path is not None
        root = Path(spec.uri)
        relative = str(path.relative_to(root)) if root.is_dir() else path.name
        yield 1, {"record_id": relative, "title": path.stem, "body": path.read_text(encoding="utf-8", errors="replace")}
    else:  # sql: server-side cursor, bounded fetches
        from sqlalchemy import create_engine, text

        if not spec.query:
            raise ValueError("a sql source requires 'query'")
        engine = create_engine(spec.uri)
        try:
            with engine.connect().execution_options(stream_results=True) as conn:
                result = conn.execute(text(spec.query))
                for number, row in enumerate(result.mappings().yield_per(spec.batch_size), start=1):
                    yield number, dict(row)
        finally:
            engine.dispose()


def _document_to_row(doc: CanonicalDocument) -> dict[str, Any]:
    data = doc.model_dump()
    data["extension_json"] = json.dumps(data.pop("extension"), sort_keys=True, default=str)
    return data


def row_to_document(row: dict[str, Any]) -> CanonicalDocument:
    data = dict(row)
    data["extension"] = json.loads(data.pop("extension_json") or "{}")
    return CanonicalDocument.model_validate(data)


# --------------------------------------------------------------------------- #
# Reports and manifest
# --------------------------------------------------------------------------- #
@dataclass
class EtlReport:
    run_id: str
    stage: str
    partitions_total: int = 0
    partitions_processed: int = 0
    partitions_skipped: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)
    seconds: float = 0.0
    partition_checksums: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        rows = self.counts.get("rows_read", 0)
        return {
            "run_id": self.run_id, "stage": self.stage, "partitions_total": self.partitions_total,
            "partitions_processed": self.partitions_processed, "partitions_skipped": self.partitions_skipped,
            "counts": dict(sorted(self.counts.items())), "rejected_by_reason": dict(sorted(self.rejected.items())),
            "seconds": round(self.seconds, 3),
            "rows_per_second": round(rows / self.seconds, 1) if self.seconds > 0 and rows else None,
            "partition_checksums": self.partition_checksums,
        }


class _Manifest:
    """Per-partition checkpoint: a partition that completed with the same checksum is skipped."""

    def __init__(self, session_factory: sessionmaker[Session], run_id: str) -> None:
        self.session_factory, self.run_id = session_factory, run_id

    def completed(self, uri: str, key: str, checksum: str) -> bool:
        with self.session_factory() as session:
            row = session.scalars(select(CorpusIngestPartition).where(
                CorpusIngestPartition.source_uri == uri, CorpusIngestPartition.partition_key == key,
                CorpusIngestPartition.checksum == checksum)).first()
            return row is not None and row.status == "completed"

    def start(self, uri: str, key: str, checksum: str) -> None:
        with self.session_factory() as session:
            row = session.scalars(select(CorpusIngestPartition).where(
                CorpusIngestPartition.source_uri == uri, CorpusIngestPartition.partition_key == key,
                CorpusIngestPartition.checksum == checksum)).first()
            if row is None:
                session.add(CorpusIngestPartition(source_uri=uri, partition_key=key, checksum=checksum,
                                                  run_id=self.run_id, status="running"))
            else:
                row.status, row.run_id, row.error = "running", self.run_id, None
            session.commit()

    def finish(self, uri: str, key: str, checksum: str, counts: dict[str, Any], error: str | None = None) -> None:
        with self.session_factory() as session:
            row = session.scalars(select(CorpusIngestPartition).where(
                CorpusIngestPartition.source_uri == uri, CorpusIngestPartition.partition_key == key,
                CorpusIngestPartition.checksum == checksum)).one()
            row.status = "failed" if error else "completed"
            row.counts, row.error = counts, error
            row.finished_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()


class DeadLetters:
    """Bounded dead-letter writer: reason codes and safe identifiers only."""

    def __init__(self, session_factory: sessionmaker[Session], run_id: str, limit: int) -> None:
        self.session_factory, self.run_id, self.limit = session_factory, run_id, limit
        self.stored = 0
        self._pending: list[CorpusDeadLetter] = []

    def add(self, uri: str, key: str, row_number: int | None, reason: str, details: dict[str, Any]) -> None:
        if self.stored + len(self._pending) >= self.limit:
            return  # the counters in the report still include every rejected row
        self._pending.append(CorpusDeadLetter(run_id=self.run_id, source_uri=uri, partition_key=key,
                                              row_number=row_number, reason_code=reason, details=details))

    def flush(self) -> None:
        if not self._pending:
            return
        with self.session_factory() as session:
            session.add_all(self._pending)
            session.commit()
        self.stored += len(self._pending)
        self._pending = []


# --------------------------------------------------------------------------- #
# Stage 1: normalise (local streaming engine)
# --------------------------------------------------------------------------- #
def normalize_source(spec: SourceSpec, output_dir: Path, session_factory: sessionmaker[Session],
                     dead_letter_limit: int = 10_000, run_id: str | None = None,
                     now: datetime | None = None) -> EtlReport:
    """Normalise every partition of ``spec`` into ``output_dir`` (one Parquet file per partition)."""
    run_id = run_id or uuid.uuid4().hex[:16]
    report = EtlReport(run_id=run_id, stage="normalize")
    manifest = _Manifest(session_factory, run_id)
    dead = DeadLetters(session_factory, run_id, dead_letter_limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ingested_at = now or datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    started = time.perf_counter()
    partitions = list_partitions(spec)
    report.partitions_total = len(partitions)
    manifest_uri = f"normalize:{spec.uri}"
    for key, path in partitions:
        checksum = file_checksum(path) if path is not None else sha256_text(spec.query or "")
        target = output_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".normalized.parquet")
        report.partition_checksums[key] = checksum
        if path is not None and target.exists() and manifest.completed(manifest_uri, key, checksum):
            report.partitions_skipped += 1
            continue
        manifest.start(manifest_uri, key, checksum)
        counts: Counter[str] = Counter()
        tmp = target.with_suffix(".tmp")
        writer = pq.ParquetWriter(tmp, NORMALIZED_SCHEMA, compression="zstd")
        buffer: list[dict[str, Any]] = []
        try:
            for number, row in iter_rows(spec, path):
                counts["rows_read"] += 1
                doc: CanonicalDocument | None
                reason: str | None
                details: dict[str, Any]
                if row is None:
                    doc, reason, details = None, INVALID_JSON, {}
                else:
                    doc, reason, details = normalize_row(row, spec, f"{spec.uri}#{key}", ingested_at)
                if doc is None:
                    assert reason is not None
                    counts["rows_rejected"] += 1
                    report.rejected[reason] += 1
                    dead.add(manifest_uri, key, number, reason, details)
                    continue
                counts["rows_valid"] += 1
                buffer.append(_document_to_row(doc))
                if len(buffer) >= spec.batch_size:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=NORMALIZED_SCHEMA))
                    buffer = []
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=NORMALIZED_SCHEMA))
            writer.close()
            tmp.replace(target)  # a partition file appears only when it is complete
            dead.flush()
            manifest.finish(manifest_uri, key, checksum, dict(counts))
        except Exception as exc:
            writer.close()
            tmp.unlink(missing_ok=True)
            manifest.finish(manifest_uri, key, checksum, dict(counts), error=type(exc).__name__)
            raise
        report.counts.update(counts)
        report.partitions_processed += 1
    report.seconds = time.perf_counter() - started
    return report


# --------------------------------------------------------------------------- #
# Stage 2: load into the registry (forward-only versions)
# --------------------------------------------------------------------------- #
def apply_documents(session: Session, documents: list[CanonicalDocument]) -> tuple[Counter[str], list[CanonicalDocument]]:
    """Apply one batch to the registry inside the caller's transaction.

    Outcome per document: ``inserted``, ``updated``, ``deleted``, ``unchanged`` (same version and
    content: an idempotent replay), ``stale`` (older than the stored version: an out-of-order
    arrival) or ``conflict`` (same version, different content: rejected, the stored copy wins).
    Returns the counters and the conflicting documents."""
    counts: Counter[str] = Counter()
    conflicts: list[CanonicalDocument] = []
    # the newest version of each document inside the batch decides; equal versions must agree
    best: dict[str, CanonicalDocument] = {}
    for doc in documents:
        current = best.get(doc.document_id)
        if current is None or doc.document_version > current.document_version:
            if current is not None:
                counts["superseded"] += 1  # an older version in the same batch
            best[doc.document_id] = doc
        elif doc.document_version == current.document_version:
            if doc.content_hash == current.content_hash:
                counts["duplicate"] += 1
            else:
                counts["conflict"] += 1
                conflicts.append(doc)
        else:
            counts["superseded"] += 1
    ids = list(best)
    existing: dict[str, CorpusDocument] = {}
    for start in range(0, len(ids), 500):
        rows = session.scalars(select(CorpusDocument).where(CorpusDocument.document_id.in_(ids[start:start + 500])))
        existing.update({row.document_id: row for row in rows})
    for doc in best.values():
        row = existing.get(doc.document_id)
        if row is None:
            session.add(_to_row(doc))
            counts["deleted" if doc.is_deleted else "inserted"] += 1
        elif doc.document_version > row.document_version:
            _update_row(row, doc)
            counts["deleted" if doc.is_deleted else "updated"] += 1
        elif doc.document_version == row.document_version:
            if doc.content_hash == row.content_hash:
                counts["unchanged"] += 1
            else:
                counts["conflict"] += 1
                conflicts.append(doc)
        else:
            counts["stale"] += 1
    return counts, conflicts


def _to_row(doc: CanonicalDocument) -> CorpusDocument:
    row = CorpusDocument(document_id=doc.document_id)
    _update_row(row, doc)
    return row


def _update_row(row: CorpusDocument, doc: CanonicalDocument) -> None:
    row.tenant_id, row.source_system, row.source_record_id = doc.tenant_id, doc.source_system, doc.source_record_id
    row.document_type, row.title = doc.document_type, doc.title
    row.content = "" if doc.is_deleted else doc.content  # deleted content is not retained
    row.content_hash, row.document_version, row.schema_version = doc.content_hash, doc.document_version, doc.schema_version
    row.source_updated_at, row.event_time = doc.source_updated_at, doc.event_time
    row.valid_from, row.valid_to, row.ingested_at = doc.valid_from, doc.valid_to, doc.ingested_at
    row.access_policy, row.source_uri, row.is_deleted = list(doc.access_policy), doc.source_uri, doc.is_deleted
    row.extension = dict(doc.extension)


def load_normalized(normalized_dir: Path, session_factory: sessionmaker[Session], batch_size: int = 2_000,
                    dead_letter_limit: int = 10_000, run_id: str | None = None) -> EtlReport:
    """Apply every normalised partition to the registry; completed partitions are skipped."""
    run_id = run_id or uuid.uuid4().hex[:16]
    report = EtlReport(run_id=run_id, stage="load")
    manifest = _Manifest(session_factory, run_id)
    dead = DeadLetters(session_factory, run_id, dead_letter_limit)
    files = sorted(Path(normalized_dir).rglob("*.parquet"))
    report.partitions_total = len(files)
    manifest_uri = f"load:{normalized_dir}"
    started = time.perf_counter()
    for path in files:
        key, checksum = str(path.relative_to(normalized_dir)), file_checksum(path)
        report.partition_checksums[key] = checksum
        if manifest.completed(manifest_uri, key, checksum):
            report.partitions_skipped += 1
            continue
        manifest.start(manifest_uri, key, checksum)
        counts: Counter[str] = Counter()
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                documents = [row_to_document(row) for row in batch.to_pylist()]
                counts["rows_read"] += len(documents)
                with session_factory() as session:
                    try:
                        outcome, conflicts = apply_documents(session, documents)
                        session.commit()
                    except IntegrityError:
                        # another worker inserted some of these documents concurrently: retry once
                        session.rollback()
                        outcome, conflicts = apply_documents(session, documents)
                        session.commit()
                counts.update(outcome)
                for doc in conflicts:
                    report.rejected[VERSION_CONFLICT] += 1
                    dead.add(manifest_uri, key, None, VERSION_CONFLICT,
                             {"document_id": doc.document_id, "document_version": doc.document_version,
                              "tenant_id": doc.tenant_id})
            dead.flush()
            manifest.finish(manifest_uri, key, checksum, dict(counts))
        except Exception as exc:
            manifest.finish(manifest_uri, key, checksum, dict(counts), error=type(exc).__name__)
            raise
        report.counts.update(counts)
        report.partitions_processed += 1
    report.seconds = time.perf_counter() - started
    return report


def reconcile(session_factory: sessionmaker[Session]) -> dict[str, Any]:
    """Registry totals for reconciliation against source and index counts."""
    from sqlalchemy import func

    with session_factory() as session:
        total = session.scalar(select(func.count()).select_from(CorpusDocument)) or 0
        deleted = session.scalar(select(func.count()).select_from(CorpusDocument)
                                 .where(CorpusDocument.is_deleted.is_(True))) or 0
        by_tenant = dict(session.execute(select(CorpusDocument.tenant_id, func.count())
                                         .where(CorpusDocument.is_deleted.is_(False))
                                         .group_by(CorpusDocument.tenant_id)).all())
    return {"documents": total, "deleted": deleted, "live": total - deleted,
            "live_by_tenant": dict(sorted(by_tenant.items()))}
