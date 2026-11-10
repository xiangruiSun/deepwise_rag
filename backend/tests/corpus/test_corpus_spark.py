"""The PySpark normalisation job must agree with the local streaming engine.

Skipped when PySpark or a JVM is not available (the default CI job installs neither)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from app.corpus import etl
from app.corpus.etl import SourceSpec, load_normalized, normalize_source, reconcile
from app.corpus.generator import write_dataset
from app.db.models import CorpusDocument
from tests.corpus.conftest import new_session_factory

pyspark = pytest.importorskip("pyspark")
pytestmark = [pytest.mark.spark, pytest.mark.skipif(shutil.which("java") is None, reason="no JVM")]


def _registry(normalized: Path) -> tuple[dict, dict]:
    factory = new_session_factory()
    load_normalized(normalized, factory)
    with factory() as session:
        docs = {d.document_id: (d.document_version, d.content_hash, d.is_deleted, d.tenant_id, tuple(d.access_policy),
                                d.event_time, d.valid_to) for d in session.query(CorpusDocument)}
    return reconcile(factory), docs


def test_spark_and_local_engines_produce_the_same_registry(tmp_path: Path) -> None:
    from app.corpus.spark_job import TENANT_BUCKETS, run_spark_normalize

    rates = {"conflict": 0.0}                       # conflicts are compared separately below
    info = write_dataset(tmp_path / "data", 1500, seed=3, n_tenants=5, n_golden=50, rows_per_file=400, rates=rates)
    spec = SourceSpec(uri=str(tmp_path / "data" / "raw"))
    local = normalize_source(spec, tmp_path / "local", new_session_factory())
    report = run_spark_normalize(spec, tmp_path / "spark", master="local[2]", shuffle_partitions=4)
    assert report["engine"] == "spark" and report["counts"]["rows_read"] == info["source_rows"] == local.counts["rows_read"]
    assert report["counts"]["rows_valid"] == local.counts["rows_valid"]
    assert report["rejected_by_reason"] == dict(local.rejected)                 # same reason codes, same counts
    buckets = sorted(p.name for p in (tmp_path / "spark").iterdir() if p.name.startswith("tenant_bucket="))
    assert buckets and all(0 <= int(b.split("=")[1]) < TENANT_BUCKETS for b in buckets)
    local_summary, local_docs = _registry(tmp_path / "local")
    spark_summary, spark_docs = _registry(tmp_path / "spark")
    assert spark_summary == local_summary and spark_docs == local_docs            # identical end state
    assert report["counts"]["documents_written"] == local_summary["documents"]   # only the newest version is written
    rejected = pq.read_table(tmp_path / "spark_rejected").to_pylist()
    assert len(rejected) == report["counts"]["rows_rejected"] and all("body" not in r["reject_details"] for r in rejected)


def test_spark_quarantines_conflicting_versions_and_reads_jsonl(tmp_path: Path) -> None:
    from app.corpus.spark_job import run_spark_normalize

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.jsonl").write_text(
        '{"tenant": "t", "record_id": "1", "body": "Stable record.", "version": "1"}\n'
        '{"tenant": "t", "record_id": "1", "body": "Stable record.", "version": "1"}\n'           # exact duplicate
        '{"tenant": "t", "record_id": "2", "body": "Copy A.", "version": "1"}\n'
        '{"tenant": "t", "record_id": "2", "body": "Copy B.", "version": "1"}\n'                  # conflict
        '{"tenant": "t", "record_id": "3", "body": "Old.", "version": "1"}\n'
        '{"tenant": "t", "record_id": "3", "body": "New.", "version": "2"}\n'
        'this line is not json\n')
    report = run_spark_normalize(SourceSpec(uri=str(source), format="jsonl"), tmp_path / "out", master="local[1]",
                                 shuffle_partitions=2)
    assert report["counts"] == {"rows_read": 7, "rows_valid": 6, "rows_rejected": 1, "exact_duplicates": 1,
                                "version_conflict_rows": 2, "documents_written": 2}
    assert report["rejected_by_reason"] == {etl.INVALID_JSON: 1, etl.VERSION_CONFLICT: 2}
    _summary, docs = _registry(tmp_path / "out")
    assert sorted(v[0] for v in docs.values()) == [1, 2]                           # record 3 at version 2, record 1
