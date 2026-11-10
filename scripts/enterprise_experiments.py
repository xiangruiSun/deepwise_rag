"""Before/after experiments of the document-corpus platform on a generated corpus.

    cd backend && python ../scripts/enterprise_experiments.py --config corpus.yaml \
        --records 20000 --golden 400 --work-dir /tmp/corpus-exp --out ../experiments/enterprise/hashing

Every number written by this script was measured by this script on the machine that ran
it.  The corpus and its labels are synthetic (see ``SYNTHETIC_CORPUS.json``): the results
show how the configurations behave relative to each other, not clinical performance.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import Settings  # noqa: E402
from app.corpus.config import CorpusConfig, load_corpus_config  # noqa: E402
from app.corpus.etl import SourceSpec, load_normalized, normalize_source  # noqa: E402
from app.corpus.evaluation import (  # noqa: E402
    EvalDataset,
    ExperimentTracker,
    compare,
    evaluate_strategy,
    render_report,
)
from app.corpus.generator import write_dataset  # noqa: E402
from app.corpus.redteam import InstructionFollowingLLM  # noqa: E402
from app.corpus.runtime import CorpusRuntime, build_corpus_runtime  # noqa: E402
from app.corpus.service import CorpusService  # noqa: E402
from app.db.base import create_all, make_engine, make_session_factory  # noqa: E402
from app.indexing.backend import LocalSearchBackend  # noqa: E402

CONFIG_DIR = BACKEND.parent / "configs"
ABLATION = ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_routed", "adaptive", "no_freshness"]
CHUNKING = ["fixed", "sliding_window", "sentence", "structure", "recursive", "semantic"]
REPORTED = ("ndcg@10", "recall@10", "mrr@10", "context_precision", "context_recall", "citation_coverage",
            "fact_coverage", "abstention_correct", "hallucination", "freshness_error", "leakage",
            "duplicate_context_rate", "context_tokens")


def make_runtime(work: Path, name: str, config: CorpusConfig, normalized: Path) -> tuple[CorpusRuntime, dict[str, Any]]:
    """A fresh registry and index for ``config``, loaded from the shared normalised documents."""
    settings = Settings(database_url="sqlite://", artifact_dir=work / "artifacts", config_dir=CONFIG_DIR,
                        environment="test")
    engine = make_engine("sqlite://")
    create_all(engine)
    runtime = build_corpus_runtime(settings, make_session_factory(engine), LocalSearchBackend(), config)
    t0 = time.perf_counter()
    load_normalized(normalized, runtime.session_factory)
    load_seconds = time.perf_counter() - t0
    index = runtime.indexes.create_version()
    build = runtime.indexes.build(index)
    validation = runtime.indexes.validate(index)
    if not validation["valid"]:
        raise SystemExit(f"index validation failed for {name}: {validation}")
    runtime.indexes.activate(index)
    stats = {"index": index, "load_seconds": round(load_seconds, 2), **{k: build[k] for k in (
        "documents", "chunks", "seconds", "embedding_seconds", "indexing_seconds", "documents_per_second")}}
    return runtime, stats


def with_extra_strategies(config: CorpusConfig) -> CorpusConfig:
    hybrid = config.strategies["hybrid"]
    extra = {"ctx_top_k": hybrid.model_copy(update={"context_selection": "top_k"}),
             "ctx_diverse": hybrid.model_copy(update={"context_selection": "diverse"})}
    return config.model_copy(update={"strategies": {**config.strategies, **extra}})


def row(result: Any) -> dict[str, Any]:
    return {"metrics": {m: result.aggregate.get(m) for m in REPORTED},
            "latency_ms": {k: result.latency_ms[k] for k in ("end_to_end", "retrieval") if k in result.latency_ms},
            "stage_p50_ms": {k: v["p50"] for k, v in result.latency_ms.items()
                             if k not in ("end_to_end", "retrieval", "first_token")},
            "tokens": result.tokens, "gates": result.gates, "errors": result.errors,
            "by_category_ndcg@10": {c: v.get("ndcg@10") for c, v in result.by_category.items()}}


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="corpus.yaml")
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--golden", type=int, default=400)
    parser.add_argument("--tenants", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--chunk-tokens", type=int, default=48, help="max_tokens of the chunking comparison")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip", nargs="*", default=[], choices=["ablation", "context", "chunking", "injection"])
    args = parser.parse_args()
    work, out = args.work_dir, args.out
    out.mkdir(parents=True, exist_ok=True)
    data = work / "data"
    if not (data / "SYNTHETIC_CORPUS.json").exists():
        write_dataset(data, args.records, seed=args.seed, n_tenants=args.tenants, n_golden=args.golden)
    normalized = work / "normalized"
    if not normalized.exists():
        engine = make_engine("sqlite://")
        create_all(engine)
        report = normalize_source(SourceSpec(uri=str(data / "raw")), normalized, make_session_factory(engine))
        (work / "normalize_report.json").write_text(json.dumps(report.as_dict(), indent=2))
    dataset = EvalDataset.load(data / "eval" / "queries.jsonl")
    base_config = with_extra_strategies(load_corpus_config(args.config, CONFIG_DIR))
    meta = {"config": args.config, "records": args.records, "golden_documents": args.golden, "tenants": args.tenants,
            "seed": args.seed, "eval_dataset": dataset.version, "queries": len(dataset.queries),
            "categories": dataset.categories(), "provenance": dataset.provenance, "search_backend": "local",
            "hardware": {"machine": platform.machine(), "platform": platform.platform()},
            "note": "Synthetic corpus and labels; measured on one workstation with the in-process search backend."}
    summary: dict[str, Any] = {"meta": meta}

    runtime, build = make_runtime(work, "base", base_config, normalized)
    summary["index_build"] = build
    summary["embedding"] = runtime.indexes.provider.model_metadata()
    print(json.dumps({"index_build": build}), flush=True)

    if "ablation" not in args.skip:
        tracker = ExperimentTracker(runtime.service, work / "runs")
        run = tracker.run(dataset, ABLATION, baseline="bm25", run_id="retrieval_ablation",
                          notes="retrieval ablation; baseline bm25")
        run_dir = Path(run["artifact_dir"])
        target = out / "retrieval_ablation"
        target.mkdir(exist_ok=True)
        published = json.loads((run_dir / "run.json").read_text())
        published["config"]["dataset_path"] = "eval/queries.jsonl"  # no machine-specific paths in the repository
        (target / "run.json").write_text(json.dumps(published, indent=2, default=str) + "\n", encoding="utf-8")
        (target / "report.md").write_text(render_report(published), encoding="utf-8")
        summary["retrieval_ablation"] = {
            "gates_passed": run["gates_passed"], "seconds": run["seconds"],
            "strategies": {s["strategy"]: {"ndcg@10": s["aggregate"].get("ndcg@10"),
                                           "recall@10": s["aggregate"].get("recall@10"),
                                           "fact_coverage": s["aggregate"].get("fact_coverage"),
                                           "freshness_error": s["aggregate"].get("freshness_error"),
                                           "p50_ms": s["latency_ms"]["end_to_end"]["p50"],
                                           "p95_ms": s["latency_ms"]["end_to_end"]["p95"]} for s in run["strategies"]},
            "verdicts_vs_bm25": {c["candidate"]: {m: c["metrics"][m]["verdict"] for m in ("ndcg@10", "recall@10")}
                                 for c in run["comparisons"]}}
        print(json.dumps(summary["retrieval_ablation"]["strategies"]), flush=True)

    if "context" not in args.skip:
        top_k = evaluate_strategy(runtime.service, dataset, "ctx_top_k")
        diverse = evaluate_strategy(runtime.service, dataset, "ctx_diverse")
        comparison = compare(top_k, diverse, metrics=("context_precision", "context_recall", "citation_coverage",
                                                      "fact_coverage", "duplicate_context_rate"))
        summary["context_selection"] = {"baseline_top_k": row(top_k), "diverse": row(diverse),
                                        "paired_bootstrap": comparison["metrics"]}
        print(json.dumps({k: summary["context_selection"][k]["metrics"] for k in ("baseline_top_k", "diverse")}),
              flush=True)

    if "injection" not in args.skip:
        probes = EvalDataset.from_rows([q for q in dataset.queries if q["category"] == "prompt_injection"])
        results = {}
        for verify in (False, True):
            for quarantine in (False, True):
                llm = InstructionFollowingLLM(runtime.config.generation)
                config = runtime.config.model_copy(update={
                    "generation": runtime.config.generation.model_copy(update={
                        "backend": "mock", "model": "instruction-following-simulator", "verify_citations": verify}),
                    "context": runtime.config.context.model_copy(update={"quarantine_instructions": quarantine})})
                service = CorpusService(config, runtime.indexes, runtime.retriever, runtime.cache, llm)
                result = evaluate_strategy(service, probes, "hybrid")
                results[f"verify_citations={verify},quarantine={quarantine}"] = {
                    "queries": len(probes.queries), "model_obeyed": llm.obeyed,
                    "injection_success_rate": result.aggregate.get("injection_success")}
        extractive = evaluate_strategy(runtime.service, probes, "hybrid")
        results["extractive_generator(default)"] = {
            "queries": len(probes.queries), "model_obeyed": None,
            "injection_success_rate": extractive.aggregate.get("injection_success")}
        summary["prompt_injection"] = {
            "adversary": "SIMULATED model that obeys any 'reply only with X' found in a source and cites it "
                         "(worst case; not a measurement of a real model's compliance rate)", "results": results}
        print(json.dumps(results), flush=True)

    if "chunking" not in args.skip:
        rows = {}
        for strategy in CHUNKING:
            chunking = base_config.chunking.model_copy(update={
                "strategy": strategy, "max_tokens": args.chunk_tokens,
                "overlap": args.chunk_tokens // 4 if strategy == "sliding_window" else 0})
            config = base_config.model_copy(update={"chunking": chunking,
                                                    "index_prefix": f"{base_config.index_prefix}_{strategy}"})
            chunk_runtime, chunk_build = make_runtime(work, strategy, config, normalized)
            result = evaluate_strategy(chunk_runtime.service, dataset, "hybrid")
            rows[strategy] = {"chunking": chunking.identity, "index": chunk_build, **row(result)}
            print(json.dumps({strategy: {"chunks": chunk_build["chunks"], **rows[strategy]["metrics"]}}), flush=True)
        summary["chunking"] = {"max_tokens": args.chunk_tokens, "retrieval_strategy": "hybrid", "results": rows}

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [f"# Corpus platform experiments — `{args.config}`", "",
             f"> **Synthetic corpus and labels** ({args.records:,} source records, {len(dataset.queries)} queries, "
             f"dataset `{dataset.version}`), in-process search backend, {platform.machine()}. "
             "Relative behaviour of configurations only.", ""]
    if "retrieval_ablation" in summary:
        lines += ["## Retrieval ablation (baseline: bm25)", "",
                  "| strategy | nDCG@10 | Recall@10 | fact coverage | freshness error | P50 ms | P95 ms | nDCG vs bm25 |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, s in summary["retrieval_ablation"]["strategies"].items():
            verdict = summary["retrieval_ablation"]["verdicts_vs_bm25"].get(name, {}).get("ndcg@10", "baseline")
            lines.append(f"| {name} | {fmt(s['ndcg@10'])} | {fmt(s['recall@10'])} | {fmt(s['fact_coverage'])} | "
                         f"{fmt(s['freshness_error'])} | {fmt(s['p50_ms'])} | {fmt(s['p95_ms'])} | {verdict} |")
        lines.append("")
    if "context_selection" in summary:
        lines += ["## Context selection (hybrid retrieval)", "",
                  "| selection | context precision | context recall | citation coverage | fact coverage | "
                  "duplicate rate | context tokens |", "|---|---|---|---|---|---|---|"]
        for label, key in (("top_k (baseline)", "baseline_top_k"), ("diverse", "diverse")):
            m = summary["context_selection"][key]["metrics"]
            lines.append(f"| {label} | {fmt(m['context_precision'])} | {fmt(m['context_recall'])} | "
                         f"{fmt(m['citation_coverage'])} | {fmt(m['fact_coverage'])} | "
                         f"{fmt(m['duplicate_context_rate'])} | {fmt(m['context_tokens'])} |")
        lines.append("")
    if "prompt_injection" in summary:
        lines += ["## Indirect prompt injection (simulated instruction-following model)", "",
                  "| configuration | queries | model obeyed | injection success rate |", "|---|---|---|---|"]
        for label, r in summary["prompt_injection"]["results"].items():
            lines.append(f"| {label} | {r['queries']} | {fmt(r['model_obeyed'])} | {fmt(r['injection_success_rate'])} |")
        lines.append("")
    if "chunking" in summary:
        lines += [f"## Chunking (max {args.chunk_tokens} tokens, hybrid retrieval)", "",
                  "| chunking | chunks | build s | nDCG@10 | Recall@10 | context recall | fact coverage | context tokens |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in summary["chunking"]["results"].items():
            m = r["metrics"]
            lines.append(f"| {name} | {r['index']['chunks']} | {fmt(r['index']['seconds'])} | {fmt(m['ndcg@10'])} | "
                         f"{fmt(m['recall@10'])} | {fmt(m['context_recall'])} | {fmt(m['fact_coverage'])} | "
                         f"{fmt(m['context_tokens'])} |")
        lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
