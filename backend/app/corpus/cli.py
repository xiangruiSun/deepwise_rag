"""``python -m app.cli corpus ...`` — tools of the document-corpus platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from app.core.config import REPO_ROOT, get_settings

corpus_app = typer.Typer(add_completion=False, no_args_is_help=True,
                         help="Document-corpus platform: synthetic data, ETL, indexes, queries.")
index_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Versioned index lifecycle.")
corpus_app.add_typer(index_app, name="index")

CorpusConfigOpt = Annotated[Optional[str], typer.Option("--corpus-config", help="Corpus config in configs/.")]
DEFAULT_DATA = REPO_ROOT / "data" / "corpus_synthetic"


def _echo(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


def _runtime(config: str | None):  # type: ignore[no-untyped-def]
    from app.corpus.runtime import standalone_runtime

    return standalone_runtime(get_settings(), config)


@corpus_app.command("generate")
def generate(
    records: Annotated[int, typer.Option(help="Logical source records to generate.")] = 2_000,
    output: Annotated[Path, typer.Option(help="Target directory.")] = DEFAULT_DATA,
    seed: int = 13,
    tenants: int = 8,
    golden: Annotated[int, typer.Option(help="Golden documents with evaluation queries.")] = 200,
    rows_per_file: int = 50_000,
    confirm_large: Annotated[bool, typer.Option("--confirm-large", help="Required above 5,000,000 records.")] = False,
) -> None:
    """Write a deterministic SYNTHETIC source dataset as partitioned Parquet (streaming)."""
    from app.corpus.generator import write_dataset

    if records > 5_000_000 and not confirm_large:
        typer.secho("more than 5,000,000 records needs --confirm-large and enough free disk space "
                    "(roughly 0.3-0.5 GB per million records)", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    _echo(write_dataset(output, records, seed, tenants, golden, rows_per_file))


@corpus_app.command("ingest")
def ingest(
    source: Annotated[Path, typer.Option(help="Source directory or file.")] = DEFAULT_DATA / "raw",
    format: Annotated[str, typer.Option(help="parquet | jsonl | csv | text")] = "parquet",
    normalized: Annotated[Optional[Path], typer.Option(help="Where normalised Parquet is written.")] = None,
    engine: Annotated[str, typer.Option(help="local | spark (normalisation stage)")] = "local",
    batch_size: int = 5_000,
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Normalise a source and load it into the document registry (idempotent, resumable)."""
    from app.corpus.etl import SourceSpec, load_normalized, normalize_source, reconcile

    runtime = _runtime(corpus_config)
    normalized = normalized or source.parent / "normalized"
    spec = SourceSpec(uri=str(source), format=format, batch_size=batch_size)  # type: ignore[arg-type]
    if engine == "spark":
        from app.corpus.spark_job import run_spark_normalize

        first: dict[str, Any] = run_spark_normalize(spec, normalized)
    else:
        first = normalize_source(spec, normalized, runtime.session_factory,
                                 runtime.config.dead_letter_limit).as_dict()
    second = load_normalized(normalized, runtime.session_factory,
                             dead_letter_limit=runtime.config.dead_letter_limit).as_dict()
    _echo({"normalize": first, "load": second, "registry": reconcile(runtime.session_factory)})


@index_app.command("build")
def index_build(
    activate: Annotated[bool, typer.Option(help="Activate the new version when it validates.")] = True,
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Build a new index version next to the active one, validate it, then switch the alias."""
    runtime = _runtime(corpus_config)
    name = runtime.indexes.create_version()
    built = runtime.indexes.build(name)
    validation = runtime.indexes.validate(name)
    out: dict[str, Any] = {"index": name, "build": built, "validation": validation}
    if activate and validation["valid"]:
        out["activation"] = runtime.indexes.activate(name)
        out["sync"] = runtime.indexes.sync()
    elif activate:
        out["activation"] = "skipped: validation failed"
    _echo(out)
    if not validation["valid"]:
        raise typer.Exit(code=3)


@index_app.command("sync")
def index_sync(limit: Optional[int] = None, corpus_config: CorpusConfigOpt = None) -> None:
    """Apply changed and deleted documents to the active index."""
    _echo(_runtime(corpus_config).indexes.sync(limit))


@index_app.command("rollback")
def index_rollback(corpus_config: CorpusConfigOpt = None) -> None:
    """Switch the alias back to the previous index version and repair what it missed."""
    _echo(_runtime(corpus_config).indexes.rollback())


@index_app.command("health")
def index_health(corpus_config: CorpusConfigOpt = None) -> None:
    _echo(_runtime(corpus_config).indexes.health())


@index_app.command("check")
def index_check(repair: bool = False, limit: Optional[int] = None, corpus_config: CorpusConfigOpt = None) -> None:
    """Compare registry and active index document by document."""
    runtime = _runtime(corpus_config)
    result = runtime.indexes.consistency_check(repair, limit)
    if repair and result["repaired"]:
        result["sync"] = runtime.indexes.sync()
    _echo(result)
    if not result["consistent"] and not repair:
        raise typer.Exit(code=3)


@corpus_app.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question or search text.")],
    tenant: Annotated[str, typer.Option(help="Tenant to act as (offline tool only).")] = "tenant-000",
    groups: Annotated[str, typer.Option(help="Comma-separated permission groups.")] = "public",
    strategy: Optional[str] = None,
    retrieve_only: bool = False,
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Ask the corpus from the command line (same code path as the API)."""
    from app.corpus.retrieval import AuthContext
    from app.corpus.service import QueryRequest

    runtime = _runtime(corpus_config)
    auth = AuthContext(tenant, "cli", frozenset(g.strip() for g in groups.split(",") if g.strip()) | {"public"})
    request = QueryRequest(query=question, strategy=strategy)
    if retrieve_only:
        result = runtime.service.retrieve(auth, request, use_cache=False)
        _echo({"plan": result.plan.as_dict(), "results": [c.public() for c in result.chunks[:8]]})
    else:
        _echo(runtime.service.answer_sync(auth, request))


@corpus_app.command("capacity")
def capacity(
    records: Annotated[int, typer.Option(help="Source records to plan for.")] = 100_000_000,
    dimension: int = 1024,
    chunks_per_document: float = 1.3,
    avg_document_bytes: float = 1200.0,
    avg_chunk_bytes: float = 900.0,
    vector_encoding: Annotated[str, typer.Option(help="float32 | float16 | int8")] = "float32",
    quantized_index: bool = False,
    replicas: int = 1,
    shard_target_gb: float = 40.0,
    embedding_rate: Annotated[Optional[float], typer.Option(help="Measured chunks/second per GPU.")] = None,
    gpus: int = 1,
    calibrate_from_active: Annotated[bool, typer.Option("--calibrate", help="Use bytes/chunk of the active index.")] = False,
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Estimate storage and memory for a target scale (a model; calibrate before trusting it)."""
    from app.corpus.capacity import CapacityInputs, calibrate, estimate

    inputs = CapacityInputs(source_records=records, dimension=dimension, chunks_per_document=chunks_per_document,
                            avg_document_bytes=avg_document_bytes, avg_chunk_bytes=avg_chunk_bytes,
                            vector_encoding=vector_encoding, quantized_index=quantized_index,  # type: ignore[arg-type]
                            replicas=replicas, shard_target_gb=shard_target_gb,
                            embedding_chunks_per_second_per_gpu=embedding_rate, gpus=gpus)
    measured = None
    if calibrate_from_active:
        runtime = _runtime(corpus_config)
        measured = calibrate(runtime.backend.stats(runtime.indexes.require_active()))
    _echo(estimate(inputs, measured))


batch_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Offline batch RAG jobs.")
eval_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Evaluation runs.")
corpus_app.add_typer(batch_app, name="batch")
corpus_app.add_typer(eval_app, name="eval")


@batch_app.command("submit")
def batch_submit(
    input_file: Annotated[Path, typer.Argument(help="JSONL or Parquet file with a 'query' column.")],
    tenant: str = "tenant-000",
    groups: str = "public",
    strategy: Optional[str] = None,
    run: Annotated[bool, typer.Option(help="Run the job now instead of leaving it queued.")] = False,
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Queue a batch job from a file (copied into the batch input directory)."""
    import shutil

    from app.corpus.retrieval import AuthContext

    runtime = _runtime(corpus_config)
    runtime.batch.input_dir.mkdir(parents=True, exist_ok=True)
    target = runtime.batch.input_dir / input_file.name
    if input_file.resolve() != target.resolve():
        shutil.copyfile(input_file, target)
    auth = AuthContext(tenant, "cli", frozenset(g.strip() for g in groups.split(",") if g.strip()) | {"public"})
    job = runtime.batch.create(auth, input_uri=input_file.name, strategy=strategy)
    _echo(runtime.batch.run(job["job_id"]) if run else job)


@batch_app.command("work")
def batch_work(max_jobs: Optional[int] = None, requeue_stale_after: float = 900.0,
               corpus_config: CorpusConfigOpt = None) -> None:
    """Batch worker: resume jobs of dead workers, then run queued jobs until the queue is empty."""
    runtime = _runtime(corpus_config)
    _echo({"requeued": runtime.batch.requeue_stale(requeue_stale_after), "finished": runtime.batch.work(max_jobs)})


@batch_app.command("status")
def batch_status(job_id: str, corpus_config: CorpusConfigOpt = None) -> None:
    _echo(_runtime(corpus_config).batch.get(job_id))


@eval_app.command("run")
def eval_run(
    dataset: Annotated[Path, typer.Option(help="Evaluation dataset (JSONL).")] = DEFAULT_DATA / "eval" / "queries.jsonl",
    strategy: Annotated[Optional[list[str]], typer.Option(help="Strategy name; repeat to compare several.")] = None,
    baseline: Optional[str] = None,
    limit: Optional[int] = None,
    output: Annotated[Optional[Path], typer.Option(help="Directory for run artifacts.")] = None,
    notes: str = "",
    corpus_config: CorpusConfigOpt = None,
) -> None:
    """Compare retrieval strategies on a frozen dataset; writes run.json, report.md and per-query files."""
    from app.corpus.evaluation import EvalDataset, ExperimentTracker

    runtime = _runtime(corpus_config)
    tracker = ExperimentTracker(runtime.service, output) if output else runtime.tracker
    names = strategy or ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_routed", "adaptive"]
    run = tracker.run(EvalDataset.load(dataset), names, baseline, limit=limit, notes=notes)
    _echo({"run_id": run["run_id"], "artifact_dir": run["artifact_dir"], "gates_passed": run["gates_passed"],
           "seconds": run["seconds"],
           "ndcg@10": {s["strategy"]: s["aggregate"].get("ndcg@10") for s in run["strategies"]},
           "verdicts": {c["candidate"]: c["metrics"]["ndcg@10"]["verdict"] for c in run["comparisons"]}})
    if not run["gates_passed"]:
        raise typer.Exit(code=3)
