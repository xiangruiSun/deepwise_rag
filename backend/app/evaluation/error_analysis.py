"""Automatic failure categorisation and per-query comparisons (spec section 32).

Only identifiers, grades and metric values are stored — never clinical text —
so a failure can be reproduced from authorized sources without copying
restricted records into ordinary artifacts.  Categories are heuristic triage
labels to direct manual inspection; they are not verified root causes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

FAILURE_CATEGORIES = (
    "missing_ehr_modality", "incorrect_source_join", "incompatible_laboratory_units",
    "wrong_temporal_cutoff", "semantic_mismatch", "negation_mismatch", "bm25_vocabulary_mismatch",
    "embedding_model_failure", "duplicate_chunk_ranking_bias", "incorrect_case_aggregation",
    "image_representation_mismatch", "reranker_degradation", "unsupported_generation",
    "wrong_citation", "excessive_latency",
)


def _relevant(qrels: dict[str, int], threshold: int) -> set[str]:
    return {c for c, g in qrels.items() if g >= threshold}


def compare_runs(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    threshold: int = 2,
    metric: str = "ndcg@10",
) -> list[dict[str, Any]]:
    """Per query: which relevant cases one method retrieved and the other missed."""
    base = {r["query_id"]: r for r in baseline_rows if r.get("applicable", True)}
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        other = base.get(row["query_id"])
        if other is None or not row.get("applicable", True):
            continue
        relevant = _relevant(qrels.get(row["query_id"], {}), threshold)
        cand_top, base_top = set(row["ranked_case_ids"][:k]), set(other["ranked_case_ids"][:k])
        m_c, m_b = row["metrics"].get(metric), other["metrics"].get(metric)
        out.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "baseline_metric": m_b,
                "candidate_metric": m_c,
                "difference": (m_c - m_b) if (m_c is not None and m_b is not None) else None,
                "relevant_only_in_candidate": sorted((cand_top - base_top) & relevant),
                "relevant_only_in_baseline": sorted((base_top - cand_top) & relevant),
                "relevant_missed_by_both": sorted(relevant - cand_top - base_top)[:20],
            }
        )
    out.sort(key=lambda r: (r["difference"] is None, r["difference"] or 0.0))
    return out


def categorize_failures(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    qrels: dict[str, dict[str, int]],
    comparison: list[dict[str, Any]] | None = None,
    baseline_summary: dict[str, Any] | None = None,
    k: int = 10,
    threshold: int = 2,
    low_metric: float = 0.3,
    latency_budget_ms: float | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign heuristic failure categories to low-scoring queries of one run."""
    by_query_cmp = {c["query_id"]: c for c in comparison or []}
    modalities = set(summary.get("modalities") or ["text"])
    has_reranker = bool(summary.get("reranker"))
    base_has_reranker = bool((baseline_summary or {}).get("reranker"))
    method = summary.get("retrieval_method") or ""
    base_method = (baseline_summary or {}).get("retrieval_method") or ""
    failures: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        if not row.get("applicable", True) or not row.get("has_judgments"):
            continue
        categories: list[str] = []
        ndcg = row["metrics"].get(f"ndcg@{k}")
        warnings = " ".join(row.get("warnings") or []).lower()
        coverage = row.get("modality_coverage") or {}
        relevant = _relevant(qrels.get(row["query_id"], {}), threshold)
        cmp_row = by_query_cmp.get(row["query_id"])
        low = ndcg is not None and ndcg < low_metric
        if latency_budget_ms and float(row["latency_ms"].get("total", 0.0)) > latency_budget_ms:
            categories.append("excessive_latency")
        if "laboratory" in modalities and ("no comparable laboratory" in warnings
                                           or coverage.get("laboratory", 1.0) < 0.5):
            categories.append("missing_ehr_modality")
        if "not comparable" in warnings:
            categories.append("incompatible_laboratory_units")
        if "image" in modalities and ("no usable image" in warnings or "image modality is disabled" in warnings):
            categories.append("image_representation_mismatch")
        if low and relevant:
            if cmp_row and cmp_row["relevant_only_in_baseline"]:
                if has_reranker and not base_has_reranker:
                    categories.append("reranker_degradation")
                elif method == "dense" and "bm25" in base_method:
                    categories.append("embedding_model_failure")
                elif method == "bm25" and "dense" in base_method:
                    categories.append("bm25_vocabulary_mismatch")
                elif modalities - {"text"} and "image" in modalities:
                    categories.append("image_representation_mismatch")
                else:
                    categories.append("semantic_mismatch")
            elif method == "bm25":
                categories.append("bm25_vocabulary_mismatch")
            elif method == "dense":
                categories.append("embedding_model_failure")
            else:
                categories.append("semantic_mismatch")
            if row.get("category") == "E":
                categories.append("negation_mismatch")  # difficult/negation query set: inspect manually
        if categories:
            for category in categories:
                counts[category] += 1
            failures.append({"query_id": row["query_id"], "category": row["category"],
                             f"ndcg@{k}": ndcg, "failure_categories": categories,
                             "top_case_ids": row["ranked_case_ids"][:k]})
    gates = summary.get("gates") or {}
    if gates.get("temporal_cutoff") is False:
        counts["wrong_temporal_cutoff"] += 1
    generation = summary.get("generation") or {}
    if generation:
        if (generation.get("faithfulness") or 1.0) < 1.0:
            counts["unsupported_generation"] += 1
        if (generation.get("citation_correctness") or 1.0) < 1.0 or generation.get("invalid_citations"):
            counts["wrong_citation"] += 1
    if validation_report:
        unmatched = sum(v for key, v in (validation_report.get("counts") or {}).items()
                        if key.startswith("unmatched_"))
        if unmatched:
            counts["incorrect_source_join"] += 0  # joins are validated; unmatched rows are reported, not mis-joined
    return {
        "experiment_id": summary.get("experiment_id"),
        "category_counts": {c: counts.get(c, 0) for c in FAILURE_CATEGORIES},
        "failures": failures,
        "note": "Heuristic triage labels for manual inspection; identifiers only, no clinical text.",
    }
