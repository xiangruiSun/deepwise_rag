"""Command line interface: inspection, ingestion, indexing, retrieval and evaluation.

Run ``python -m app.cli --help`` (or ``inspect-rag --help`` after ``pip install -e backend``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import yaml

from app.core.config import (
    REPO_ROOT,
    Settings,
    get_settings,
    load_app_config,
    load_experiment_suite,
    resolve_suite,
)
from app.core.container import AppContext, build_context, build_indexes, run_ingestion
from app.core.errors import AppError
from app.core.security import AccessPolicy, system_principal

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="INSPECT-RAG research toolkit. Research use only — not for clinical decisions.")

ConfigOpt = Annotated[Optional[str], typer.Option("--config", "-c", help="Application config (in configs/).")]


def _echo(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


def _context(config: str | None) -> AppContext:
    settings: Settings = get_settings()
    return build_context(settings, load_app_config(config or settings.app_config, settings.config_dir))


def _guard(func):  # type: ignore[no-untyped-def]
    """Render application errors as a clean message + non-zero exit code."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return func(*args, **kwargs)
        except AppError as exc:
            typer.secho(f"error [{exc.code}]: {exc.message}", fg=typer.colors.RED, err=True)
            if exc.details:
                typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
            raise typer.Exit(code=2) from exc
        except (FileNotFoundError, ValueError) as exc:
            typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    return wrapper


@app.command("generate-synthetic")
@_guard
def generate_synthetic(
    output: Annotated[Path, typer.Option(help="Target directory (must be empty or a synthetic dataset).")] = REPO_ROOT / "data" / "synthetic",
    patients: Annotated[int, typer.Option(min=20, max=20000)] = 170,
    seed: int = 20240607,
) -> None:
    """Generate the deterministic SYNTHETIC cohort (fabricated data for pipeline verification)."""
    from app.ingestion.synthetic_generator import generate_synthetic_dataset

    _echo(generate_synthetic_dataset(output, n_patients=patients, seed=seed))


@app.command("inspect-dataset")
@_guard
def inspect_dataset(
    root: Annotated[Path, typer.Argument(help="Dataset root directory.")],
    output: Annotated[Optional[Path], typer.Option(help="Write data_manifest.json here.")] = None,
    no_hash: Annotated[bool, typer.Option("--no-hash", help="Skip file hashing (faster).")] = False,
) -> None:
    """Discover files, inspect schemas without loading tables and detect the dataset layout."""
    from app.ingestion.inspector import DatasetInspector, write_manifest

    manifest = DatasetInspector(root, hash_files=not no_hash).inspect()
    if output:
        write_manifest(manifest, output)
    _echo({k: v for k, v in manifest.items() if k != "files"} | {"n_files": len(manifest["files"])})


@app.command("validate-dataset")
@_guard
def validate_dataset(config: ConfigOpt = None) -> None:
    """Validate the configured dataset layout and show the resolved column mapping."""
    from app.ingestion.adapters import build_adapter

    settings = get_settings()
    cfg = load_app_config(config or settings.app_config, settings.config_dir)
    report = build_adapter(cfg.dataset).validate_layout()
    _echo(report.__dict__)


@app.command("ingest")
@_guard
def ingest(config: ConfigOpt = None,
           force: Annotated[bool, typer.Option(help="Re-ingest event files even if unchanged.")] = False) -> None:
    """Ingest the configured dataset (INSPECT or synthetic) into the canonical store."""
    ctx = _context(config)
    _echo(run_ingestion(ctx, force=force, actor="cli").summary())


@app.command("reindex")
@_guard
def reindex(
    config: ConfigOpt = None,
    embedding: Annotated[Optional[list[str]], typer.Option(help="Embedding key(s); 'lexical' for BM25-only.")] = None,
    chunking: Annotated[Optional[list[str]], typer.Option(help="Chunking key(s).")] = None,
    recreate: Annotated[bool, typer.Option(help="Drop and rebuild the selected report indexes.")] = False,
    promote: Annotated[Optional[str], typer.Option(help="Report index name to promote to the 'current' alias.")] = None,
    no_images: Annotated[bool, typer.Option("--no-images", help="Skip image feature extraction/indexing.")] = False,
) -> None:
    """Incrementally (re)build search indexes; unchanged content is not re-embedded."""
    ctx = _context(config)
    pairs: list[tuple[str | None, str]] | None = None
    if embedding or chunking:
        embeddings = embedding or ["lexical"]
        chunkings = chunking or ["whole_impression"]
        pairs = [(None if e == "lexical" else e, c) for e in embeddings for c in chunkings]
    summary = build_indexes(ctx, pairs, recreate=recreate, with_images=not no_images, actor="cli")
    if promote:
        from app.indexing.indexer import Indexer

        summary["promoted_alias"] = Indexer(ctx.session_factory, ctx.backend, ctx.config, ctx.embeddings,
                                            "cli").promote("reports", promote)
    _echo(summary)


@app.command("embed")
@_guard
def embed(embedding: Annotated[str, typer.Argument(help="Embedding key from the config.")],
          config: ConfigOpt = None) -> None:
    """Generate (and cache) document embeddings for all report impressions with one model."""
    import time

    from sqlalchemy import select

    from app.db.models import RadiologyReport

    ctx = _context(config)
    with ctx.session_factory() as session:
        texts = list(session.scalars(select(RadiologyReport.impression_text)))
    embedder = ctx.embeddings.embedder(embedding)
    started = time.perf_counter()
    vectors = embedder.encode_documents(texts)
    elapsed = time.perf_counter() - started
    provider = ctx.embeddings.provider(embedding)
    _echo({"embedding": provider.model_metadata(), "documents": len(texts), "shape": list(vectors.shape),
           "seconds": round(elapsed, 3), "documents_per_second": round(len(texts) / max(elapsed, 1e-9), 1),
           "cache_hits": ctx.embeddings.cache.hits, "cache_misses": ctx.embeddings.cache.misses,
           "truncation": provider.truncation})


@app.command("search")
@_guard
def search(
    config: ConfigOpt = None,
    recipe: Annotated[Optional[str], typer.Option(help="Retrieval recipe id.")] = None,
    case_id: Annotated[Optional[str], typer.Option(help="Query case id (case-based retrieval).")] = None,
    text: Annotated[Optional[str], typer.Option(help="Free-text clinical query.")] = None,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    top_k: int = 10,
    scope: Annotated[Optional[list[str]], typer.Option(help="Access scope(s) of the CLI principal.")] = None,
) -> None:
    """Run one retrieval request and print ranked cases with evidence references."""
    from app.schemas.search import SearchRequest

    ctx = _context(config)
    mode = "case" if case_id else "text" if text else "filter"
    request = SearchRequest(query_mode=mode, case_id=case_id, query_text=text, age_min=age_min,  # type: ignore[arg-type]
                            age_max=age_max, top_k=top_k, retrieval_recipe_id=recipe)
    policy = AccessPolicy(system_principal(scope or [ctx.config.dataset.default_access_scope]))
    response = ctx.retrieval.search(request, policy)
    _echo(response.model_dump(mode="json"))


@app.command("run-experiment")
@_guard
def run_experiment(
    suite: Annotated[str, typer.Argument(help="Suite file, e.g. experiments/smoke.yaml")],
    only: Annotated[Optional[list[str]], typer.Option(help="Run only these experiment ids.")] = None,
    force: Annotated[bool, typer.Option(help="Discard previous results for these runs.")] = False,
    no_resume: Annotated[bool, typer.Option("--no-resume")] = False,
    final_test: Annotated[bool, typer.Option("--final-test", help="Required for held-out TEST suites.")] = False,
    output: Annotated[Optional[Path], typer.Option(help="Experiment output directory.")] = None,
) -> None:
    """Execute one experiment suite (or selected experiments of it)."""
    from app.evaluation.runner import ExperimentRunner

    settings = get_settings()
    suite_cfg = load_experiment_suite(suite, settings.config_dir)
    ctx = build_context(settings, load_app_config(suite_cfg.base_config, settings.config_dir))
    summary = ExperimentRunner(ctx, output).run_suite(suite_cfg, resume=not no_resume, force=force,
                                                      allow_test=final_test, only=only)
    _echo({"suite_id": summary["suite_id"], "data_kind": summary["data_kind"],
           "leaderboard": summary["leaderboard"]})


@app.command("evaluate")
@_guard
def evaluate(
    suites: Annotated[list[str], typer.Argument(help="One or more suite files.")],
    force: bool = False,
    output: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Batch evaluation: run several validation suites in sequence."""
    from app.evaluation.runner import ExperimentRunner

    settings = get_settings()
    out: dict[str, Any] = {}
    contexts: dict[str, AppContext] = {}
    for name in suites:
        suite_cfg = load_experiment_suite(name, settings.config_dir)
        if suite_cfg.base_config not in contexts:
            contexts[suite_cfg.base_config] = build_context(
                settings, load_app_config(suite_cfg.base_config, settings.config_dir))
        summary = ExperimentRunner(contexts[suite_cfg.base_config], output).run_suite(suite_cfg, force=force)
        out[summary["suite_id"]] = [
            {k: row[k] for k in ("experiment_id", "evaluation_status", "ndcg@10", "recall@10", "error")}
            for row in summary["leaderboard"]
        ]
    _echo(out)


@app.command("select")
@_guard
def select(
    suite_id: Annotated[str, typer.Argument(help="Completed VALIDATION suite id.")],
    max_latency_ratio: float = 5.0,
    min_faithfulness: Optional[float] = None,
    output: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Apply the predeclared model-selection policy to validation results."""
    from app.evaluation.report import run_selection

    settings = get_settings()
    _echo(run_selection(output or settings.experiment_dir, suite_id, max_latency_ratio, min_faithfulness))


@app.command("freeze-final")
@_guard
def freeze_final(
    suite: Annotated[str, typer.Argument(help="Suite file the selection was made from.")],
    output: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Write the frozen held-out TEST suite (baseline + selected configuration)."""
    settings = get_settings()
    exp_dir = output or settings.experiment_dir
    selection_file = exp_dir / "selection.json"
    if not selection_file.exists():
        raise FileNotFoundError("no selection.json found: run `select` first")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    suite_cfg = load_experiment_suite(suite, settings.config_dir)
    if selection["suite_id"] != suite_cfg.suite_id:
        raise ValueError("the selection was made from a different suite")
    app_config, eval_cfg = resolve_suite(suite_cfg, settings.config_dir)
    by_id = {r.experiment_id: r for r in suite_cfg.runs}
    chosen = [selection["baseline"]] + ([selection["selected"]] if selection["selected"] != selection["baseline"] else [])
    frozen = {
        "suite_id": "final_test",
        "description": f"Frozen configuration selected from '{suite_cfg.suite_id}' on validation data. Run ONCE.",
        "base_config": suite_cfg.base_config,
        "stage": "test",
        "baseline_experiment": selection["baseline"],
        "recipes": {by_id[e].recipe: app_config.recipes[by_id[e].recipe].model_dump(mode="json") for e in chosen},
        "generation_variants": {k: v.model_dump(mode="json") for k, v in suite_cfg.generation_variants.items()},
        "runs": [by_id[e].model_dump(mode="json") for e in chosen],
        "evaluation": eval_cfg.model_dump(mode="json"),
    }
    target = exp_dir / "final_test.yaml"
    target.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    _echo({"frozen_suite": str(target), "experiments": chosen,
           "next": f"python -m app.cli run-experiment {target} --final-test"})


@app.command("aggregate-metrics")
@_guard
def aggregate_metrics(
    config: ConfigOpt = None,
    output: Annotated[Optional[Path], typer.Option(help="Report directory.")] = None,
    experiments: Annotated[Optional[Path], typer.Option(help="Experiment directory.")] = None,
) -> None:
    """Aggregate all completed suites into the Markdown + JSON evaluation report."""
    from app.evaluation.report import ReportBuilder

    settings = get_settings()
    cfg = load_app_config(config or settings.app_config, settings.config_dir)
    exp_dir = experiments or settings.experiment_dir
    builder = ReportBuilder(exp_dir, settings.artifact_dir, cfg.dataset.name)
    index_stats: dict[str, dict[str, Any]] = {}
    try:  # index build statistics (chunk counts, truncation, embedding time) come from the database
        from sqlalchemy import select as sql_select

        from app.db.base import make_engine, make_session_factory
        from app.db.models import IndexManifest

        with make_session_factory(make_engine(settings.database_url))() as session:
            for manifest in session.scalars(sql_select(IndexManifest)):
                index_stats[manifest.index_name] = dict(manifest.stats or {})
    except Exception as exc:  # the report is still useful without them
        typer.secho(f"warning: index statistics unavailable ({type(exc).__name__})", err=True)
    _echo(builder.build(output or exp_dir / "report", cfg.evaluation.qrels, cfg.evaluation.label_sources,
                        cfg.evaluation.bootstrap_samples, cfg.evaluation.seed, index_stats))


@app.command("build-annotation-pool")
@_guard
def build_annotation_pool_cmd(
    suite_id: Annotated[str, typer.Argument(help="Completed suite whose runs are pooled.")],
    depth: int = 20,
    seed: int = 13,
    output: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Create blinded, shuffled annotation tasks by pooling the top results of several systems."""
    from app.evaluation.datasets import build_annotation_pool, read_jsonl, write_jsonl

    settings = get_settings()
    suite_dir = settings.experiment_dir / suite_id
    summary = json.loads((suite_dir / "suite_summary.json").read_text(encoding="utf-8"))
    runs: dict[str, dict[str, list[str]]] = {}
    for run in summary["runs"]:
        if run["status"] == "completed":
            rows = read_jsonl(Path(run["run_dir"]) / "per_query.jsonl")
            runs[run["experiment_id"]] = {r["query_id"]: r["ranked_case_ids"] for r in rows}
    tasks, provenance, procedure = build_annotation_pool(runs, depth, seed)
    out = output or suite_dir / "annotation_pool"
    write_jsonl(out / "annotation_tasks.jsonl", tasks)
    write_jsonl(out / "provenance_DO_NOT_SHARE_WITH_ANNOTATORS.jsonl", provenance)
    (out / "pooling_procedure.json").write_text(json.dumps(procedure, indent=2), encoding="utf-8")
    _echo({**procedure, "output": str(out)})


@app.command("index-health")
@_guard
def index_health(config: ConfigOpt = None) -> None:
    """Show index status, versions, sizes and health."""
    from app.api.routes import _index_status

    ctx = _context(config)
    _echo([status.model_dump(mode="json") for status in _index_status(ctx)])


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
