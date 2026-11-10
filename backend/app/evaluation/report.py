"""Evaluation report generator (spec section 31): Markdown + machine-readable JSON.

Every number comes from files written by completed experiment runs.  Sections
whose experiments have not been run say so explicitly; nothing is estimated,
interpolated or invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.charts import bar_chart, grouped_bar_chart, scatter_chart
from app.evaluation.datasets import QUERY_CATEGORIES, Qrels, read_jsonl
from app.evaluation.error_analysis import categorize_failures, compare_runs
from app.evaluation.statistics import paired_bootstrap, select_configuration

NOT_RUN = "_Not yet evaluated: no completed experiments for this section._"
SUITE_SECTIONS = {
    "embedding": ("embedding_comparison", "5. Embedding comparison"),
    "chunking": ("chunking_comparison", "6. Chunking comparison"),
    "retrieval": ("retrieval_comparison", "7. Retrieval-method comparison"),
    "reranking": ("reranking_depth", "8. Reranking comparison"),
    "multimodal": ("multimodal_ablation", "9. Multimodal ablation"),
    "generation": ("generation_comparison", "10. RAG generation evaluation"),
}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return NOT_RUN
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _completed(suite: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in suite.get("runs", []) if r["status"] == "completed"]


def _per_query(run: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(run["run_dir"]) / "per_query.jsonl"
    return read_jsonl(path) if path.exists() else []


def _metric_map(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    return {r["query_id"]: r["metrics"].get(metric) for r in rows
            if r.get("applicable", True) and r.get("has_judgments")}


def _main_rows(suite: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for entry in suite.get("leaderboard", []):
        rows.append([
            entry["experiment_id"], entry["embedding_model"], entry["retrieval_method"],
            entry["fusion_strategy"], entry["reranker"], entry["modality_combination"],
            entry["recall@10"], entry["precision@10"], entry["ndcg@10"],
            entry["p95_retrieval_latency_ms"], entry["faithfulness"], entry["index_size_bytes"],
            entry["evaluation_status"] + (f" ({entry['error']})" if entry.get("error") else ""),
        ])
    return rows


MAIN_HEADERS = ["Experiment", "Embedding", "Retrieval", "Fusion", "Reranker", "Modalities", "Recall@10",
                "Precision@10", "nDCG@10", "p95 retrieval ms", "Faithfulness", "Index bytes", "Status"]


class ReportBuilder:
    def __init__(self, experiment_dir: Path, artifact_dir: Path, dataset_name: str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.artifact_dir = Path(artifact_dir)
        self.dataset_name = dataset_name
        self.suites: dict[str, dict[str, Any]] = {}
        for path in sorted(self.experiment_dir.glob("*/suite_summary.json")):
            suite = json.loads(path.read_text(encoding="utf-8"))
            self.suites[suite["suite_id"]] = suite

    def _load(self, relative: str) -> dict[str, Any] | None:
        path = self.artifact_dir / "ingestion" / self.dataset_name / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ------------------------------------------------------------------ build
    def build(self, output_dir: Path, qrels_path: str | None = None, label_sources: list[str] | None = None,
              bootstrap_samples: int = 2000, seed: int = 13,
              index_stats: dict[str, dict[str, Any]] | None = None, label_note: str | None = None,
              dataset_info: dict[str, Any] | None = None,
              text_label: str = "Radiology impression") -> dict[str, Any]:
        """``index_stats``: build statistics per index name (from ``index_manifests``).
        ``dataset_info``: the ``DATASET.json`` provenance marker of a prepared open dataset."""
        self.index_stats = index_stats or {}
        output_dir = Path(output_dir)
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load("data_manifest.json")
        validation = self._load("validation_report.json")
        qrels: Qrels | None = None
        if qrels_path:
            try:
                qrels = Qrels.load(qrels_path, label_sources or ["clinician"])
            except FileNotFoundError:
                qrels = None
        any_suite = next(iter(self.suites.values()), None)
        data_kind = any_suite["data_kind"] if any_suite else "unknown"
        report: dict[str, Any] = {"data_kind": data_kind, "suites": {}, "statistics": {}, "errors": {},
                                  "charts": [], "selection": None}
        md: list[str] = ["# INSPECT-RAG evaluation report", ""]
        if data_kind == "synthetic":
            md += ["> **SYNTHETIC DATA.** Every number below was measured on fabricated fixtures and "
                   "verifies that the pipeline works. It is **not** evidence of clinical retrieval "
                   "performance and no INSPECT experiment is reported here.", ""]
        elif dataset_info:
            md += [f"> **Dataset: {dataset_info.get('source', data_kind)}** "
                   f"(licence: {dataset_info.get('license', 'see source')}). {dataset_info.get('notice', '')} "
                   "**No INSPECT experiment is reported here.**", ""]
            if dataset_info.get("citation"):
                md += [f"> Source: {dataset_info['citation']} <{dataset_info.get('source_url', '')}>", ""]
        md += ["> Research prototype. Not for diagnosis, treatment decisions or clinical use.", ""]

        # 1 ------------------------------------------------------------------
        md += ["## 1. Dataset version and cohort characteristics", ""]
        if manifest:
            counts = manifest.get("record_counts", {})
            md.append(_table(["Item", "Value"], [
                ["Dataset", manifest.get("dataset_name")], ["Release", manifest.get("dataset_version")],
                ["Adapter / schema", f"{manifest.get('adapter')} / {manifest.get('detected_schema_version')}"],
                ["Fingerprint", manifest.get("dataset_fingerprint")],
                ["Patients", counts.get("patients")], ["Cases", counts.get("cases")],
                ["Cases train / valid / test", f"{counts.get('cases_split_train')} / "
                 f"{counts.get('cases_split_valid')} / {counts.get('cases_split_test')}"],
                [f"{text_label} records", counts.get("reports")], ["Clinical events", counts.get("events")],
                ["Imaging studies", counts.get("imaging_studies")],
                ["Modalities", ", ".join(manifest.get("available_modalities", []))],
            ]))
        else:
            md.append("_No data manifest found: the dataset has not been ingested._")
        # 2 ------------------------------------------------------------------
        md += ["", "## 2. Data cleaning and join validation", ""]
        if validation:
            unmatched = {k: v for k, v in validation.get("counts", {}).items() if k.startswith("unmatched_")}
            md.append("Joins use the dataset's own identifiers only; rejected rows are counted, never merged.")
            md += ["", _table(["Unmatched join", "Rows"], [[k, v] for k, v in sorted(unmatched.items())]), "",
                   _table(["Rejected rows (stage: reason)", "Rows"],
                          [[k, v] for k, v in sorted(validation.get("failure_counts", {}).items())])]
            if validation.get("warnings"):
                md += ["", "Warnings: " + "; ".join(validation["warnings"])]
        else:
            md.append("_No validation report found._")
        # 3, 4 ---------------------------------------------------------------
        md += ["", "## 3. Query-set composition", ""]
        if any_suite:
            rows = []
            for stage, comp in sorted(any_suite.get("query_composition", {}).items()):
                for category, n in comp.items():
                    rows.append([stage, f"{category} — {QUERY_CATEGORIES.get(category, '')}", n])
            md.append(_table(["Stage", "Category", "Queries"], rows))
            md += ["", f"Query-set version `{any_suite['query_set_version']}`. Candidate corpus = training "
                       "split; development queries = validation patients; final queries = held-out test "
                       "patients. Query patients are excluded from their own results."]
        else:
            md.append(NOT_RUN)
        md += ["", "## 4. Relevance annotation methodology", ""]
        if any_suite:
            md += [f"Label sources used: `{', '.join(any_suite['label_sources'])}` "
                   f"(version `{any_suite['qrels_version']}`); "
                   f"{any_suite['weak_or_other_label_rows_excluded']} rows from other label sources "
                   "(e.g. weak labels) were excluded.",
                   "", "Graded case-level relevance: 3 highly, 2 moderately, 1 weakly, 0 not relevant; "
                   "unjudged pairs are tracked separately (judged coverage is reported with every metric) "
                   "and recall is judged-pool recall."]
            if data_kind == "synthetic":
                md.append("")
                md.append("Synthetic labels come from a hand-specified rubric over latent attributes of the "
                          "generator — they are NOT clinician judgments.")
            elif label_note:
                md += ["", label_note]
            else:
                md.append("")
                md.append("Authoritative labels must come from qualified clinical reviewers using the blinded "
                          "pooling workflow (`inspect-rag build-annotation-pool`).")
        else:
            md.append(NOT_RUN)

        # 5-10 ---------------------------------------------------------------
        for key, (suite_id, title) in SUITE_SECTIONS.items():
            md += ["", f"## {title}", ""]
            suite = self.suites.get(suite_id)
            if suite is None or not _completed(suite):
                md.append(NOT_RUN if suite is None else
                          NOT_RUN + "\n\n" + _table(["Experiment", "Status"],
                                                    [[r["experiment_id"], f"{r['status']}: {r.get('error')}"]
                                                     for r in suite["runs"]]))
                continue
            report["suites"][suite_id] = suite["leaderboard"]
            md.append(_table(MAIN_HEADERS, _main_rows(suite)))
            md += ["", self._suite_details(key, suite, charts_dir, report)]

        # main comparison table ----------------------------------------------
        md += ["", "## Main comparison table (all completed and attempted experiments)", ""]
        all_rows = [row for suite in self.suites.values() for row in _main_rows(suite)]
        md.append(_table(MAIN_HEADERS, all_rows))

        # 11 -----------------------------------------------------------------
        md += ["", "## 11. Latency and resource evaluation", ""]
        perf_rows, scatter = [], []
        hardware: dict[str, Any] = {}
        for suite in self.suites.values():
            for run in _completed(suite):
                s = run["summary"]
                lat = s["latency_ms"].get("total", {})
                perf_rows.append([s["experiment_id"], lat.get("p50"), lat.get("p90"), lat.get("p95"),
                                  s["latency_ms"].get("reranking", {}).get("p95"),
                                  s["latency_ms"].get("query_embedding", {}).get("p95"),
                                  s["index_size_bytes"], s["resources"].get("process_peak_memory_mb"),
                                  s["resources"].get("gpu_peak_memory_mb")])
                recall = s["metrics"].get("recall@10")
                if recall is not None and lat.get("p95") is not None:
                    scatter.append((s["experiment_id"], float(lat["p95"]), float(recall)))
                metrics_file = Path(run["run_dir"]) / "metrics.json"
                if not hardware and metrics_file.exists():
                    hardware = json.loads(metrics_file.read_text(encoding="utf-8")).get("hardware", {})
        md.append(_table(["Experiment", "p50 ms", "p90 ms", "p95 ms", "rerank p95 ms", "query-embed p95 ms",
                          "Index bytes", "Peak RSS MB", "Peak GPU MB"], perf_rows))
        if hardware:
            md += ["", "Measured on: " + ", ".join(f"{k}={v}" for k, v in sorted(hardware.items()))
                   + ". Latencies are single-process, concurrency 1, after warm-up queries; they are only "
                     "comparable between runs on the same hardware."]
        self._chart(report, md, charts_dir, "recall_vs_latency.svg",
                    scatter_chart("Recall@10 versus p95 latency", scatter, "p95 end-to-end latency (ms)", "Recall@10"))

        # 12 -----------------------------------------------------------------
        md += ["", "## 12. Statistical uncertainty", "",
               "Paired bootstrap over evaluation queries (same queries for both systems), 95% interval for "
               "the difference in nDCG@10 versus each suite's baseline.", ""]
        stat_rows = []
        selection_inputs: dict[str, Any] = {}
        for suite_id, suite in self.suites.items():
            baseline_id = suite.get("baseline_experiment")
            runs = {r["experiment_id"]: r for r in _completed(suite)}
            if not baseline_id or baseline_id not in runs:
                continue
            per_query = {eid: _metric_map(_per_query(run), "ndcg@10") for eid, run in runs.items()}
            selection_inputs[suite_id] = (suite, runs, per_query, baseline_id)
            for eid in sorted(runs):
                if eid == baseline_id:
                    continue
                cmp_ = paired_bootstrap(per_query[baseline_id], per_query[eid], "ndcg@10", bootstrap_samples, seed)
                report["statistics"][f"{suite_id}:{eid}"] = cmp_.as_dict()
                stat_rows.append([suite_id, f"{eid} vs {baseline_id}", cmp_.n_queries, cmp_.mean_difference,
                                  cmp_.ci_low, cmp_.ci_high, f"{cmp_.wins}/{cmp_.losses}/{cmp_.ties}",
                                  cmp_.top_decile_share, cmp_.credible_improvement])
        md.append(_table(["Suite", "Comparison", "Queries", "Δ nDCG@10", "CI low", "CI high",
                          "Wins/Losses/Ties", "Gain share of top 10% queries", "CI excludes 0 (improvement)"],
                         stat_rows))
        md += ["", "Relevance judgments are incomplete (see judged coverage); intervals reflect query "
                   "sampling variability only, not annotation uncertainty."]

        # 13 -----------------------------------------------------------------
        md += ["", "## 13. Error analysis", ""]
        selection_summary: dict[str, Any] | None = None
        error_rows = []
        for suite_id, (_, runs, _metric_maps, baseline_id) in selection_inputs.items():
            best_id = max((e for e in runs if e != baseline_id),
                          key=lambda e: runs[e]["summary"]["metrics"].get("ndcg@10") or -1.0, default=None)
            if best_id is None or qrels is None:
                continue
            base_rows, cand_rows = _per_query(runs[baseline_id]), _per_query(runs[best_id])
            comparison = compare_runs(base_rows, cand_rows, qrels.by_query)
            analysis = categorize_failures(cand_rows, runs[best_id]["summary"], qrels.by_query, comparison,
                                           runs[baseline_id]["summary"], validation_report=validation)
            report["errors"][suite_id] = {"candidate": best_id, "baseline": baseline_id, **analysis,
                                          "per_query_comparison": comparison}
            for category, count in analysis["category_counts"].items():
                if count:
                    error_rows.append([suite_id, best_id, category, count])
            diffs = [c for c in comparison if c["difference"] is not None]
            self._chart(report, md, charts_dir, f"per_query_diff_{suite_id}.svg",
                        bar_chart(f"Per-query nDCG@10 difference: {best_id} − {baseline_id} ({suite_id})",
                                  [c["query_id"] for c in diffs], [c["difference"] for c in diffs],
                                  "Δ nDCG@10", signed=True))
        md.append(_table(["Suite", "Run", "Failure category (heuristic)", "Queries"], error_rows))
        md += ["", "Categories are automatic triage labels for manual review. Per-query lists of cases "
                   "retrieved by one method and missed by the other are in `report.json` (identifiers only)."]

        # 14 -----------------------------------------------------------------
        md += ["", "## 14. Selected final architecture", ""]
        selection_file = self.experiment_dir / "selection.json"
        if selection_file.exists():
            selection_summary = json.loads(selection_file.read_text(encoding="utf-8"))
            report["selection"] = selection_summary
            md += [f"Selected on validation data: **{selection_summary.get('selected')}** "
                   f"(baseline `{selection_summary.get('baseline')}`, suite `{selection_summary.get('suite_id')}`).",
                   "", f"Rationale: {selection_summary.get('rationale')}.", "",
                   _table(["Experiment", "Decision", "Reason", "nDCG@10"],
                          [[d["experiment_id"], d["status"], d.get("reason"), d.get("primary_metric")]
                           for d in selection_summary.get("decisions", [])])]
            final = self.suites.get("final_test")
            if final and _completed(final):
                md += ["", "Held-out test evaluation of the frozen configuration (run once):", "",
                       _table(MAIN_HEADERS, _main_rows(final))]
            else:
                md += ["", "_The frozen configuration has not yet been evaluated on the held-out test set._"]
        else:
            md.append("_No selection has been made yet (`inspect-rag select --suite <suite>`)._")

        # 15 -----------------------------------------------------------------
        md += ["", "## 15. Limitations and future work", ""]
        limits = [
            "INSPECT is a research dataset; this system is not validated for clinical care.",
            "INSPECT labels (PE status, outcomes) do not define clinical similarity; conclusions depend on "
            "the separate, incomplete relevance annotations.",
            "Dates are patient-specifically shifted: only within-patient temporal order is meaningful.",
            "Image similarity uses an experimental representation and is not a validated lesion comparison.",
            "Automated generation metrics are heuristics that require calibration against human review.",
        ]
        if data_kind == "synthetic":
            limits.insert(0, "All results in this report are from synthetic fixtures; no INSPECT experiment has "
                             "been run. The INSPECT adapters are not yet validated against the real release.")
        elif dataset_info:
            # an open stand-in dataset: the INSPECT-specific caveats above do not describe it
            limits = [
                f"All results in this report are from {dataset_info.get('source', data_kind)}, not INSPECT; "
                "they do not transfer to INSPECT, whose adapters are still not validated against the real "
                "release.",
                "This system is a research prototype and is not validated for clinical care.",
                *([label_note] if label_note else []),
                "Automated generation metrics are heuristics that require calibration against human review.",
            ]
        md += [f"- {item}" for item in limits]

        (output_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return {"markdown": str(output_dir / "report.md"), "json": str(output_dir / "report.json"),
                "charts": report["charts"], "suites": sorted(self.suites)}

    # ----------------------------------------------------------------- helpers
    def _chart(self, report: dict[str, Any], md: list[str], charts_dir: Path, name: str, svg: str) -> None:
        (charts_dir / name).write_text(svg, encoding="utf-8")
        report["charts"].append(f"charts/{name}")
        md += ["", f"![{name}](charts/{name})"]

    def _dense_index(self, run: dict[str, Any]) -> dict[str, Any]:
        """Build statistics of the dense (or, failing that, lexical) index a run used."""
        names = sorted(run["summary"]["index_versions"])
        dense = [n for n in names if "_lexical" not in n] or names
        return self.index_stats.get(dense[0], {}) if dense else {}

    @staticmethod
    def _truncated_queries(run: dict[str, Any]) -> int:
        return sum(1 for row in _per_query(run)
                   if any("truncated for dense retrieval" in w for w in row.get("warnings") or []))

    def _suite_details(self, key: str, suite: dict[str, Any], charts_dir: Path, report: dict[str, Any]) -> str:
        md: list[str] = []
        runs = _completed(suite)
        labels = [r["experiment_id"] for r in runs]
        ndcg = [r["summary"]["metrics"].get("ndcg@10") for r in runs]
        if key == "embedding":
            names = [f"{r['experiment_id']} {r['summary']['embedding'] or 'bm25'}" for r in runs]
            self._chart(report, md, charts_dir, "ndcg_by_embedding.svg",
                        bar_chart("nDCG@10 by embedding model", names, ndcg, "nDCG@10"))
            md += ["", _table(["Experiment", "Model", "Revision", "Dim", "Query-embed p95 ms", "Vector-search p95 ms",
                               "Index build s (embedding)", "Index bytes", "Truncated documents",
                               "Queries truncated by the encoder"],
                              [[r["experiment_id"], (m := r["summary"]["model_revisions"].get("embedding", {})).get("model_id"),
                                (m.get("revision") or {}).get("document"), m.get("dimension"),
                                r["summary"]["latency_ms"].get("query_embedding", {}).get("p95"),
                                r["summary"]["latency_ms"].get("vector_search", {}).get("p95"),
                                (ix := self._dense_index(r)).get("embedding_seconds") if m else None,
                                r["summary"]["index_size_bytes"],
                                (ix.get("encoder_truncation") or {}).get("documents_truncated") if m else None,
                                self._truncated_queries(r) if m else None] for r in runs]),
                   "", "kNN is exact on the local backend and approximate (HNSW) on Elasticsearch; compare "
                       "latency and recall only between runs on the same backend."]
        elif key == "chunking":
            self._chart(report, md, charts_dir, "ndcg_by_chunking.svg",
                        bar_chart("nDCG@10 by chunking strategy",
                                  [f"{r['experiment_id']} {r['summary']['chunking']}" for r in runs], ndcg, "nDCG@10"))
            md += ["", _table(["Experiment", "Chunking", "Chunks", "Avg tokens", "Max tokens", "% documents truncated",
                               "Embedding s", "Index bytes", "Recall@10", "nDCG@10", "p95 ms"],
                              [[r["experiment_id"], r["summary"]["chunking"], (ix := self._dense_index(r)).get("chunks"),
                                ix.get("avg_tokens"), ix.get("max_tokens"), ix.get("pct_documents_truncated"),
                                ix.get("embedding_seconds"), r["summary"]["index_size_bytes"],
                                r["summary"]["metrics"].get("recall@10"), r["summary"]["metrics"].get("ndcg@10"),
                                r["summary"]["p95_latency_ms"]] for r in runs]),
                   "", "Texts shorter than the chunk limit are never split, so strategies whose limits exceed "
                       "the text length produce identical chunk sets and therefore identical results."]
        elif key == "reranking":
            points = [(r["experiment_id"], float(r["summary"]["rerank_depth"] or 0),
                       float(r["summary"]["latency_ms"].get("reranking", {}).get("p95") or 0.0)) for r in runs]
            self._chart(report, md, charts_dir, "rerank_depth_vs_latency.svg",
                        scatter_chart("Reranking depth versus reranking latency", points,
                                      "Reranking depth (documents)", "Rerank p95 latency (ms)"))
        elif key == "multimodal":
            self._chart(report, md, charts_dir, "multimodal_ablation.svg",
                        bar_chart("Multimodal ablation: nDCG@10",
                                  [f"{r['experiment_id']} {'+'.join(r['summary']['modalities'])}" for r in runs],
                                  ndcg, "nDCG@10"))
            md += ["", _table(["Experiment", "Modalities", "Applicable queries", "Coverage: laboratory",
                               "Coverage: image", "nDCG@10"],
                              [[r["experiment_id"], "+".join(r["summary"]["modalities"]),
                                r["summary"].get("query_applicability"),
                                r["summary"]["modality_coverage"].get("laboratory"),
                                r["summary"]["modality_coverage"].get("image"),
                                r["summary"]["metrics"].get("ndcg@10")] for r in runs]),
                   "", "Runs without a text retriever cannot represent free-text queries; their means cover "
                       "applicable queries only (see the applicability column)."]
        elif key == "generation":
            gen = [(r["experiment_id"], r["summary"].get("generation") or {}) for r in runs]
            md += [_table(["Experiment", "Backend / model", "Prompt", "Verified", "Relevance", "Faithfulness",
                           "Citation correctness", "Citation coverage", "Citation relevance", "Completeness",
                           "Abstention", "Contradiction", "Numeric accuracy", "Parse failures",
                           "Latency ms", "Output tokens"],
                          [[eid, f"{g.get('backend')} / {g.get('model')}", g.get("prompt_variant"),
                            g.get("post_verification"), g.get("answer_relevance"), g.get("faithfulness"),
                            g.get("citation_correctness"), g.get("citation_coverage"), g.get("citation_relevance"),
                            g.get("completeness"), g.get("abstention_correctness"),
                            g.get("contradiction_rate"), g.get("numeric_accuracy"), g.get("parse_failure"),
                            g.get("generation_latency_ms"), g.get("output_tokens")]
                           for eid, g in gen])]
            points = [(eid, float(g["output_tokens"]), float(g["faithfulness"])) for eid, g in gen
                      if g.get("output_tokens") is not None and g.get("faithfulness") is not None]
            self._chart(report, md, charts_dir, "faithfulness_vs_cost.svg",
                        scatter_chart("Faithfulness versus generation cost", points,
                                      "Mean output tokens per answer", "Faithfulness (heuristic)"))
        if key in ("retrieval", "embedding") and runs:
            baseline_id = suite.get("baseline_experiment")
            by_id = {r["experiment_id"]: r for r in runs}
            best = max(runs, key=lambda r: r["summary"]["metrics"].get("ndcg@10") or -1.0)
            base = by_id.get(baseline_id or "", best)
            categories = sorted({*base["summary"]["metrics_by_category"], *best["summary"]["metrics_by_category"]})
            series = {
                f"{base['experiment_id']} (baseline)": [
                    base["summary"]["metrics_by_category"].get(c, {}).get("ndcg@10") for c in categories],
                f"{best['experiment_id']} (best nDCG@10)": [
                    best["summary"]["metrics_by_category"].get(c, {}).get("ndcg@10") for c in categories],
            }
            self._chart(report, md, charts_dir, f"category_{suite['suite_id']}.svg",
                        grouped_bar_chart(f"nDCG@10 by query category ({suite['suite_id']})",
                                          [f"{c} {QUERY_CATEGORIES.get(c, '')}" for c in categories], series, "nDCG@10"))
        del labels
        return "\n".join(md)


def run_selection(experiment_dir: Path, suite_id: str, max_latency_ratio: float = 5.0,
                  min_faithfulness: float | None = None, samples: int = 2000, seed: int = 13) -> dict[str, Any]:
    """Apply the selection policy to a completed VALIDATION suite and persist the decision."""
    summary_file = Path(experiment_dir) / suite_id / "suite_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"suite '{suite_id}' has no results in {experiment_dir}")
    suite = json.loads(summary_file.read_text(encoding="utf-8"))
    if suite["stage"] != "validation":
        raise ValueError("model selection must use validation results, never the held-out test set")
    baseline_id = suite.get("baseline_experiment")
    if not baseline_id:
        raise ValueError("suite has no baseline_experiment")
    runs, per_query = [], {}
    for run in _completed(suite):
        s = run["summary"]
        runs.append({"experiment_id": s["experiment_id"], "ndcg@10": s["metrics"].get("ndcg@10"),
                     "p95_latency_ms": s.get("p95_latency_ms"), "gates": s.get("gates", {}),
                     "faithfulness": s.get("faithfulness"), "recipe_id": s.get("recipe_id")})
        per_query[s["experiment_id"]] = _metric_map(_per_query(run), "ndcg@10")
    decision = select_configuration(runs, baseline_id, per_query, "ndcg@10", max_latency_ratio,
                                    min_faithfulness, samples, seed)
    decision.update(suite_id=suite_id, data_kind=suite["data_kind"],
                    dataset_fingerprint=suite["dataset_fingerprint"],
                    selected_recipe=next((r["recipe_id"] for r in runs
                                          if r["experiment_id"] == decision["selected"]), None))
    (Path(experiment_dir) / "selection.json").write_text(json.dumps(decision, indent=2, default=str),
                                                         encoding="utf-8")
    return decision
