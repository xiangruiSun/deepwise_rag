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
