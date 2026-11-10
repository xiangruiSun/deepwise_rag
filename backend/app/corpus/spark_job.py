"""Distributed normalisation with PySpark (stage 1 of the corpus ETL).

The job applies the *same* ``normalize_row`` function as the local engine, through
``mapInArrow``, so both engines produce identical documents.  On top of that it does
what only makes sense across the whole dataset:

* exact duplicates are dropped;
* a document version that arrives with two different contents is a conflict — the
  rows go to the rejected dataset and never reach the registry;
* only the newest version of each document is kept (the registry would discard the
  older ones anyway);
* output is hash-bucketed by tenant, so loaders can run per bucket in parallel.

Stage 2 (``etl.load_normalized``) is unchanged: it is idempotent and partition-
checkpointed, so any number of loader processes may consume the output.

Run it with ``python -m app.cli corpus ingest --engine spark`` (local mode) or submit
``app/corpus/spark_job.py`` to a cluster with the ``app`` package shipped via
``--py-files``.  Requires the ``spark`` extra and a JVM.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.corpus.etl import INVALID_JSON, NORMALIZED_SCHEMA, VERSION_CONFLICT, SourceSpec, normalize_row

TENANT_BUCKETS = 64


def _session(master: str | None, app_name: str, shuffle_partitions: int | None,
             driver_memory: str | None = None) -> Any:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    # JDK 22+ removed the default security manager that Hadoop's login code still calls
    java_options = "-Djava.security.manager=allow" if _java_major() >= 22 else ""
    if java_options:
        builder = builder.config("spark.driver.extraJavaOptions", java_options) \
                         .config("spark.executor.extraJavaOptions", java_options)
    builder = (builder.config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
               .config("spark.ui.showConsoleProgress", "false")
               .config("spark.pyspark.python", sys.executable)
               .config("spark.pyspark.driver.python", sys.executable))
    if shuffle_partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if driver_memory:
        # local mode runs everything in the driver JVM; the 1g default fails around a million rows
        builder = builder.config("spark.driver.memory", driver_memory)
    return builder.getOrCreate()


def _java_major() -> int:
    import re
    import subprocess

    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=20).stderr
        match = re.search(r'version "(\d+)', out)
        return int(match.group(1)) if match else 0
    except (OSError, subprocess.SubprocessError):
        return 0


def _normalizer(spec_json: str, source_uri: str, ingested_at_iso: str, raw_text: bool) -> Any:
    """Build the ``mapInArrow`` function.  Everything it closes over is a plain string."""
    import pyarrow as pa

    out_schema = pa.schema([*NORMALIZED_SCHEMA, ("reject_reason", pa.string()), ("reject_details", pa.string())])
    names = [f.name for f in NORMALIZED_SCHEMA]

    def run(batches: Iterator[Any]) -> Iterator[Any]:
        spec = SourceSpec.model_validate_json(spec_json)
        ingested_at = datetime.fromisoformat(ingested_at_iso)
        for batch in batches:
            rows: list[dict[str, Any]] = []
            for raw in batch.to_pylist():
                row: dict[str, Any] | None = raw
                if raw_text:  # one JSON document per line; a line that is not an object is rejected
                    try:
                        parsed = json.loads(raw["value"])
                        row = parsed if isinstance(parsed, dict) else None
                    except (json.JSONDecodeError, TypeError):
                        row = None
                reason: str | None
                details: dict[str, Any]
                if row is None:
                    doc, reason, details = None, INVALID_JSON, {}
                else:
                    doc, reason, details = normalize_row(row, spec, source_uri, ingested_at)
                if doc is None:
                    rows.append({**dict.fromkeys(names), "reject_reason": reason,
                                 "reject_details": json.dumps(details, sort_keys=True)})
                else:
                    data = doc.model_dump()
                    data["extension_json"] = json.dumps(data.pop("extension"), sort_keys=True, default=str)
                    rows.append({**data, "reject_reason": None, "reject_details": None})
            yield pa.RecordBatch.from_pylist(rows, schema=out_schema)

    return run


def _spark_schema() -> Any:
    from pyspark.sql import types as T

    mapping = {"string": T.StringType(), "int64": T.LongType(), "bool": T.BooleanType(),
               "timestamp[us]": T.TimestampNTZType(), "list<item: string>": T.ArrayType(T.StringType())}
    fields = [T.StructField(f.name, mapping[str(f.type)], True) for f in NORMALIZED_SCHEMA]
    fields += [T.StructField("reject_reason", T.StringType(), True), T.StructField("reject_details", T.StringType(), True)]
    return T.StructType(fields)


def run_spark_normalize(spec: SourceSpec, output_dir: Path, master: str | None = "local[*]",
                        shuffle_partitions: int | None = None, dead_letter_limit: int = 10_000,
                        now: datetime | None = None, driver_memory: str | None = "4g") -> dict[str, Any]:
    """Normalise ``spec`` with Spark into ``output_dir`` and return a report (same keys as the local engine)."""
    try:
        from pyspark import StorageLevel
        from pyspark.sql import Window
        from pyspark.sql import functions as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PySpark is not installed; install the 'spark' extra or use --engine local") from exc
    if spec.format not in ("parquet", "jsonl", "csv"):
        raise ValueError("the Spark engine reads parquet, jsonl and csv sources")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spark = _session(master, "inspect-rag-corpus-normalize", shuffle_partitions,
                     driver_memory if (master or "").startswith("local") else None)
    try:
        if spec.format == "parquet":
            source = spark.read.parquet(spec.uri)
        elif spec.format == "csv":
            source = spark.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(spec.uri)
        else:
            source = spark.read.text(spec.uri)
        ingested_at = (now or datetime.now(UTC).replace(tzinfo=None, microsecond=0)).isoformat()
        mapped = source.mapInArrow(_normalizer(spec.model_dump_json(), spec.uri, ingested_at, spec.format == "jsonl"),
                                   _spark_schema()).persist(StorageLevel.DISK_ONLY)
        rows_read = mapped.count()
        rejected = mapped.where(F.col("reject_reason").isNotNull()).select("reject_reason", "reject_details")
        valid = mapped.where(F.col("reject_reason").isNull()).drop("reject_reason", "reject_details")
        rejected_by_reason = {r["reject_reason"]: r["count"] for r in rejected.groupBy("reject_reason").count().collect()}
        rows_valid = rows_read - sum(rejected_by_reason.values())

        distinct = valid.dropDuplicates(["document_id", "document_version", "content_hash"])
        versions = Window.partitionBy("document_id", "document_version")
        flagged = distinct.withColumn("_variants", F.count(F.lit(1)).over(versions))
        conflicts = flagged.where(F.col("_variants") > 1)
        n_conflict_rows = conflicts.count()
        clean = flagged.where(F.col("_variants") == 1).drop("_variants")
        newest = Window.partitionBy("document_id")
        latest = (clean.withColumn("_max", F.max("document_version").over(newest))
                  .where(F.col("document_version") == F.col("_max")).drop("_max"))
        bucketed = latest.withColumn("tenant_bucket", F.pmod(F.xxhash64("tenant_id"), F.lit(TENANT_BUCKETS)))
        (bucketed.repartition("tenant_bucket").write.mode("overwrite").partitionBy("tenant_bucket")
         .option("compression", "zstd").parquet(str(output_dir)))
        documents = spark.read.parquet(str(output_dir)).count()

        conflict_rows = conflicts.select(
            F.lit(VERSION_CONFLICT).alias("reject_reason"),
            F.to_json(F.struct("document_id", "document_version", "tenant_id")).alias("reject_details"))
        rejected_dir = output_dir.parent / (output_dir.name + "_rejected")
        rejected.unionByName(conflict_rows).limit(dead_letter_limit).coalesce(1) \
            .write.mode("overwrite").parquet(str(rejected_dir))
        if n_conflict_rows:
            rejected_by_reason[VERSION_CONFLICT] = n_conflict_rows
        mapped.unpersist()
        seconds = time.perf_counter() - started
        report = {
            "stage": "normalize", "engine": "spark", "spark_version": spark.version,
            "master": spark.sparkContext.master, "default_parallelism": spark.sparkContext.defaultParallelism,
            "counts": {"rows_read": rows_read, "rows_valid": rows_valid, "rows_rejected": rows_read - rows_valid,
                       "exact_duplicates": rows_valid - distinct.count(), "version_conflict_rows": n_conflict_rows,
                       "documents_written": documents},
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "seconds": round(seconds, 3), "rows_per_second": round(rows_read / seconds, 1) if seconds else None,
            "output": str(output_dir), "rejected_output": str(rejected_dir), "tenant_buckets": TENANT_BUCKETS,
        }
        (output_dir / "_normalize_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        spark.stop()


if __name__ == "__main__":  # spark-submit entry point: spark_job.py <source> <format> <output>
    _source, _format, _output = sys.argv[1:4]
    print(json.dumps(run_spark_normalize(SourceSpec(uri=_source, format=_format), Path(_output), master=None),  # type: ignore[arg-type]
                     indent=2))
