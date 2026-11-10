"""Synthetic generation, normalisation, idempotent loading, versions and resumability."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from sqlalchemy import func, select

from app.corpus import etl
from app.corpus.etl import SourceSpec, apply_documents, load_normalized, normalize_row, normalize_source, reconcile
from app.corpus.generator import INJECTION_CANARY, build_queries, golden_signature, iter_source_rows, write_dataset
from app.corpus.schema import CanonicalDocument, make_chunk_id, make_document_id
from app.db.models import CorpusDeadLetter, CorpusDocument, CorpusIngestPartition
from tests.corpus.conftest import Corpus, build_runtime, new_session_factory

NOW = datetime(2026, 1, 1)


def _doc(record: str = "R1", version: int = 1, content: str = "Documented finding in the left lung.",
         tenant: str = "t1", deleted: bool = False) -> CanonicalDocument:
    document, reason, _ = normalize_row({"tenant": tenant, "source_system": "s", "record_id": record,
                                         "body": content, "version": str(version), "deleted": deleted},
                                        SourceSpec(uri="mem"), "mem", NOW)
    assert document is not None, reason
    return document


def test_generator_is_deterministic_streams_files_and_keeps_golden_signatures_unique(tmp_path: Path) -> None:
    a = list(iter_source_rows(400, seed=5, n_tenants=3, n_golden=60))
    b = list(iter_source_rows(400, seed=5, n_tenants=3, n_golden=60))
    assert a == b and a != list(iter_source_rows(400, seed=6, n_tenants=3, n_golden=60))
    # a superseded/current pair shares a signature; every other golden document is unique
    signatures = [golden_signature(g) for g in range(2000) if g % 10 != 8]
    assert len(set(signatures)) == len(signatures)
    info = write_dataset(tmp_path / "d", 400, seed=5, n_tenants=3, n_golden=60, rows_per_file=150)
    files = sorted((tmp_path / "d" / "raw").glob("part-*.parquet"))
    assert info["files"] == len(files) >= 3 and info["source_rows"] == len(a)
    assert sum(pq.ParquetFile(f).metadata.num_rows for f in files) == len(a)
    assert max(pq.ParquetFile(f).metadata.num_rows for f in files) <= 150      # bounded batches
    assert write_dataset(tmp_path / "d", 400, seed=5, n_tenants=3, n_golden=60, rows_per_file=150) == info
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not a synthetic corpus"):
        write_dataset(tmp_path / "other", 10)
    queries = build_queries(60, n_tenants=3)
    assert {q["category"] for q in queries} >= {"exact_identifier", "paraphrase", "abbreviation", "multi_document",
                                               "temporal_update", "freshness_superseded", "historical",
                                               "prompt_injection", "cross_tenant", "no_answer", "unauthorized_group"}
    assert all(q["provenance"] == "synthetic" for q in queries)
    assert any(INJECTION_CANARY in r["body"] for r in a)


@pytest.mark.parametrize(("row", "reason"), [
    ({"tenant": "t", "body": "text"}, etl.MISSING_IDENTIFIER),
    ({"record_id": "r", "body": "text"}, etl.MISSING_TENANT),
    ({"tenant": "t", "record_id": "r", "body": "text", "updated_at": "yesterday"}, etl.INVALID_TIMESTAMP),
    ({"tenant": "t", "record_id": "r", "body": "text", "version": "v2"}, etl.INVALID_VERSION),
    ({"tenant": "t", "record_id": "r", "body": "text", "version": "0"}, etl.INVALID_VERSION),
    ({"tenant": "t", "record_id": "r", "body": " \x00 "}, etl.EMPTY_CONTENT),
    ({"tenant": "t", "record_id": "r", "body": "text", "extra_json": "{broken"}, etl.INVALID_METADATA),
    ({"tenant": "t", "record_id": "r", "body": "text", "valid_from": "2024-02-01", "valid_to": "2024-01-01"},
     etl.INVALID_VALIDITY),
])
def test_invalid_rows_are_rejected_with_reason_codes_and_no_content(row: dict, reason: str) -> None:
    document, code, details = normalize_row(row, SourceSpec(uri="mem"), "mem", NOW)
    assert document is None and code == reason
    assert "text" not in json.dumps(details)                                  # diagnostics never carry content


def test_normalisation_cleans_text_and_builds_deterministic_identity() -> None:
    document, reason, _ = normalize_row(
        {"tenant": " t1 ", "source_system": "ehr", "record_id": "R-9", "title": "Tést",
         "body": "Line one\r\nLine\x07two", "version": "3", "acl": "clinical, research", "deleted": "false",
         "updated_at": "2024-03-05T10:00:00Z", "extra_json": '{"department": "cardiology"}'},
        SourceSpec(uri="mem"), "mem#p0", NOW)
    assert reason is None and document is not None
    assert document.title == "Tést" and document.content == "Line one\nLine two"       # NFC, control char removed
    assert document.document_id == make_document_id("t1", "ehr", "R-9") and document.document_version == 3
    assert document.access_policy == ["clinical", "research"] and not document.is_deleted
    assert document.source_updated_at == datetime(2024, 3, 5, 10) and document.extension == {"department": "cardiology"}
    assert make_chunk_id(document.document_id, 3) == f"{document.document_id}-00003"
    # a mapping adapts a differently shaped source without code changes
    mapped, reason, _ = normalize_row({"id": "7", "text": "Body text."},
                                      SourceSpec(uri="mem", mapping={"source_record_id": "id", "content": "text"},
                                                 defaults={"tenant_id": "t9", "source_system": "wiki",
                                                           "document_type": "policy"}), "mem", NOW)
    assert mapped is not None and (mapped.tenant_id, mapped.document_type, mapped.access_policy) == ("t9", "policy", ["public"])


def test_versions_only_move_forward_and_replays_change_nothing() -> None:
    factory = new_session_factory()
    with factory() as session:
        counts, _ = apply_documents(session, [_doc("A", 1), _doc("B", 1), _doc("B", 1)])   # in-batch duplicate
        session.commit()
    assert counts["inserted"] == 2 and counts["duplicate"] == 1
    with factory() as session:
        replay, conflicts = apply_documents(session, [_doc("A", 1), _doc("B", 1)])
        session.commit()
    assert replay["unchanged"] == 2 and not conflicts                                    # idempotent replay
    with factory() as session:                                                           # newer, then the older arrives late
        newer, _ = apply_documents(session, [_doc("A", 3, "Revised finding.")])
        late, _ = apply_documents(session, [_doc("A", 2, "Intermediate finding.")])
        clash, conflicts = apply_documents(session, [_doc("A", 3, "Different text, same version.")])
        gone, _ = apply_documents(session, [_doc("B", 2, "", deleted=True)])
        session.commit()
    assert (newer["updated"], late["stale"], clash["conflict"], gone["deleted"]) == (1, 1, 1, 1)
    assert [c.document_version for c in conflicts] == [3]
    with factory() as session:
        a = session.get(CorpusDocument, _doc("A").document_id)
        b = session.get(CorpusDocument, _doc("B").document_id)
        assert a is not None and (a.document_version, a.content) == (3, "Revised finding.")
        assert b is not None and b.is_deleted and b.content == ""                         # deleted content is not kept
        assert session.scalar(select(func.count()).select_from(CorpusDocument)) == 2


def test_reingestion_is_idempotent_and_partitions_reconcile(corpus: Corpus, tmp_path: Path) -> None:
    runtime, _ = build_runtime(tmp_path)
    raw, normalized = corpus.data_dir / "raw", tmp_path / "normalized"
    first = normalize_source(SourceSpec(uri=str(raw)), normalized, runtime.session_factory)
    loaded = load_normalized(normalized, runtime.session_factory)
    before = reconcile(runtime.session_factory)
    assert first.partitions_processed == corpus.info["files"] and first.counts["rows_read"] == corpus.info["source_rows"]
    assert first.counts["rows_valid"] + first.counts["rows_rejected"] == first.counts["rows_read"]
    assert loaded.counts["rows_read"] == first.counts["rows_valid"]
    assert before["documents"] == loaded.counts["inserted"] + loaded.counts["deleted"]
    # the same source again: every partition is skipped, nothing changes
    again = normalize_source(SourceSpec(uri=str(raw)), normalized, runtime.session_factory)
    reloaded = load_normalized(normalized, runtime.session_factory)
    assert again.partitions_skipped == first.partitions_total and reloaded.partitions_skipped == loaded.partitions_total
    assert again.partition_checksums == first.partition_checksums                       # checksums reconcile
    # forcing a replay of the same data (manifest cleared) still creates no duplicate documents
    with runtime.session_factory() as session:
        session.query(CorpusIngestPartition).delete()
        session.commit()
    replay = load_normalized(normalized, runtime.session_factory)
    assert replay.counts["inserted"] == 0 and replay.counts["unchanged"] > 0
    assert reconcile(runtime.session_factory) == before
    with runtime.session_factory() as session:
        reasons = dict(session.execute(select(CorpusDeadLetter.reason_code, func.count())
                                       .group_by(CorpusDeadLetter.reason_code)).all())
        assert sum(reasons.values()) >= first.counts["rows_rejected"] and etl.VERSION_CONFLICT in reasons
        sample = session.scalars(select(CorpusDeadLetter)).first()
        assert sample is not None and "body" not in sample.details and "content" not in sample.details


def test_an_interrupted_load_resumes_without_duplicates(corpus: Corpus, tmp_path: Path, monkeypatch) -> None:
    runtime, _ = build_runtime(tmp_path)
    normalized = tmp_path / "normalized"
    normalize_source(SourceSpec(uri=str(corpus.data_dir / "raw")), normalized, runtime.session_factory)
    real, calls = etl.apply_documents, {"n": 0}

    def flaky(session, documents):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("worker killed")
        return real(session, documents)

    monkeypatch.setattr(etl, "apply_documents", flaky)
    with pytest.raises(RuntimeError):
        load_normalized(normalized, runtime.session_factory, batch_size=200)
    with runtime.session_factory() as session:
        statuses = [p.status for p in session.scalars(select(CorpusIngestPartition).where(
            CorpusIngestPartition.source_uri.like("load:%")))]
    assert "failed" in statuses
    monkeypatch.setattr(etl, "apply_documents", real)
    resumed = load_normalized(normalized, runtime.session_factory, batch_size=200)
    assert resumed.partitions_processed >= 1
    assert reconcile(runtime.session_factory) == reconcile(corpus.runtime.session_factory)   # same end state


def test_jsonl_csv_text_and_sql_sources_stream_through_the_same_normaliser(tmp_path: Path) -> None:
    import sqlite3

    factory = new_session_factory()
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "a.jsonl").write_text(
        '{"tenant": "t", "record_id": "1", "body": "First record."}\nnot json\n[1, 2]\n'
        '{"tenant": "t", "record_id": "2", "body": "Second record."}\n')
    report = normalize_source(SourceSpec(uri=str(tmp_path / "j"), format="jsonl"), tmp_path / "nj", factory)
    assert report.counts["rows_valid"] == 2 and report.rejected[etl.INVALID_JSON] == 2
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "a.csv").write_text('tenant,record_id,body\nt,10,"Has, comma"\nt,,missing id\n')
    report = normalize_source(SourceSpec(uri=str(tmp_path / "c"), format="csv"), tmp_path / "nc", factory)
    assert report.counts["rows_valid"] == 1 and report.rejected[etl.MISSING_IDENTIFIER] == 1
    (tmp_path / "t" / "sub").mkdir(parents=True)
    (tmp_path / "t" / "sub" / "note.txt").write_text("Free text report.")
    report = normalize_source(SourceSpec(uri=str(tmp_path / "t"), format="text",
                                         defaults={"tenant_id": "t", "source_system": "files"}),
                              tmp_path / "nt", factory)
    assert report.counts["rows_valid"] == 1
    db = tmp_path / "src.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE notes (id TEXT, org TEXT, txt TEXT, rev INTEGER)")
        conn.executemany("INSERT INTO notes VALUES (?, ?, ?, ?)", [(str(i), "t", f"Note {i}.", 1) for i in range(25)])
    spec = SourceSpec(uri=f"sqlite:///{db}", format="sql", query="SELECT id, org, txt, rev FROM notes", batch_size=10,
                      mapping={"source_record_id": "id", "tenant_id": "org", "content": "txt",
                               "document_version": "rev"}, defaults={"source_system": "notes_db"})
    report = normalize_source(spec, tmp_path / "ns", factory)
    assert report.counts["rows_valid"] == 25
    assert load_normalized(tmp_path / "ns", factory).counts["inserted"] == 25
