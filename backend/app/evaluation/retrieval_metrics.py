"""Patient-level retrieval metrics (spec section 22).

All metrics are computed over ranked lists of DISTINCT case ids.

Definitions
-----------
* gain(g) = 2**g − 1 for graded relevance g ∈ {0, 1, 2, 3}
* DCG@K   = Σ_{i=1..K} gain(g_i) / log2(i + 1)
* nDCG@K  = DCG@K / IDCG@K, where IDCG@K is the DCG of the best possible
  ordering of the query's JUDGED candidates.  Undefined (``None``) when the
  query has no judged candidate with grade > 0.
* Precision@K = (# retrieved cases in the top K with grade ≥ threshold) / K.
  The denominator is always K, also when fewer than K cases were returned.
* Recall@K = (# relevant in the top K) / (# relevant among the judged
  candidates).  This is JUDGED-POOL recall, not exhaustive corpus recall:
  pooled judgments are incomplete.  Undefined when the query has no relevant
  judged candidate.
* MRR@K = 1 / rank of the first relevant case within the top K, else 0.
* Judged coverage@K = share of the returned top-K cases that carry a judgment.
* Success@K = 1 if at least one relevant case is in the top K.
* Zero-result = 1 if the ranking is empty.

Unjudged results
----------------
``unjudged_policy="non_relevant"`` scores unjudged cases with gain 0 (a
conservative lower bound).  ``"condensed"`` removes unjudged cases from the
ranking before scoring (condensed-list evaluation).  Either way the judged
coverage is always reported next to the metric; unjudged is never silently
equated with irrelevant in the reports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

UNJUDGED_POLICIES = ("non_relevant", "condensed")


def gain(grade: int) -> float:
    return float(2**grade - 1)


def _apply_policy(ranked: Sequence[str], qrels: dict[str, int], policy: str) -> list[str]:
    if policy not in UNJUDGED_POLICIES:
        raise ValueError(f"unknown unjudged policy '{policy}'")
    seen: set[str] = set()
    unique: list[str] = []
    for case_id in ranked:  # metrics are over distinct cases
        if case_id not in seen:
            seen.add(case_id)
            unique.append(case_id)
    if policy == "condensed":
        return [c for c in unique if c in qrels]
    return unique


def dcg(grades: Iterable[int]) -> float:
    return sum(gain(g) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int,
              unjudged_policy: str = "non_relevant") -> float | None:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal <= 0:
        return None
    ranking = _apply_policy(ranked, qrels, unjudged_policy)[:k]
    return dcg(qrels.get(c, 0) for c in ranking) / ideal


def precision_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int, threshold: int = 2,
                   unjudged_policy: str = "non_relevant") -> float:
    ranking = _apply_policy(ranked, qrels, unjudged_policy)[:k]
    return sum(1 for c in ranking if qrels.get(c, 0) >= threshold) / k


def recall_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int, threshold: int = 2,
                unjudged_policy: str = "non_relevant") -> float | None:
    relevant = {c for c, g in qrels.items() if g >= threshold}
    if not relevant:
        return None
    ranking = _apply_policy(ranked, qrels, unjudged_policy)[:k]
    return len(relevant.intersection(ranking)) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int, threshold: int = 2,
             unjudged_policy: str = "non_relevant") -> float | None:
    if not any(g >= threshold for g in qrels.values()):
        return None
    ranking = _apply_policy(ranked, qrels, unjudged_policy)[:k]
    for rank, case_id in enumerate(ranking, start=1):
        if qrels.get(case_id, 0) >= threshold:
            return 1.0 / rank
    return 0.0


def judged_coverage_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float | None:
    ranking = _apply_policy(ranked, qrels, "non_relevant")[:k]
    if not ranking:
        return None
    return sum(1 for c in ranking if c in qrels) / len(ranking)


def success_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int, threshold: int = 2,
                 unjudged_policy: str = "non_relevant") -> float | None:
    if not any(g >= threshold for g in qrels.values()):
        return None
    ranking = _apply_policy(ranked, qrels, unjudged_policy)[:k]
    return 1.0 if any(qrels.get(c, 0) >= threshold for c in ranking) else 0.0


def evaluate_query(ranked: Sequence[str], qrels: dict[str, int], k_values: Sequence[int] = (5, 10, 20),
                   threshold: int = 2, unjudged_policy: str = "non_relevant") -> dict[str, float | None]:
    """All per-query metrics.  ``None`` means "undefined for this query"."""
    metrics: dict[str, float | None] = {"zero_result": 1.0 if not ranked else 0.0}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels, k, unjudged_policy)
        metrics[f"recall@{k}"] = recall_at_k(ranked, qrels, k, threshold, unjudged_policy)
        metrics[f"precision@{k}"] = precision_at_k(ranked, qrels, k, threshold, unjudged_policy)
        metrics[f"judged_coverage@{k}"] = judged_coverage_at_k(ranked, qrels, k)
        metrics[f"success@{k}"] = success_at_k(ranked, qrels, k, threshold, unjudged_policy)
    metrics["mrr@10"] = mrr_at_k(ranked, qrels, 10, threshold, unjudged_policy)
    return metrics


def aggregate(per_query: Sequence[dict[str, float | None]]) -> dict[str, Any]:
    """Macro average over queries; undefined values are excluded and counted."""
    out: dict[str, Any] = {"n_queries": len(per_query)}
    names = sorted({name for row in per_query for name in row})
    for name in names:
        values = [float(v) for row in per_query if (v := row.get(name)) is not None]
        out[name] = float(np.mean(values)) if values else None
        out[f"{name}__n"] = len(values)
    return out


def percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "mean": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": len(values),
    }


METRIC_DEFINITIONS = {
    "ndcg@k": "DCG@k / IDCG@k with gain 2^grade-1 over distinct cases; ideal from judged candidates.",
    "recall@k": "Relevant (grade >= threshold) cases in top k / relevant cases among JUDGED candidates "
                "(judged-pool recall, not exhaustive corpus recall).",
    "precision@k": "Relevant cases in top k / k (denominator is always k).",
    "mrr@10": "Reciprocal rank of the first relevant case within the top 10 (0 if none).",
    "judged_coverage@k": "Share of returned top-k cases that have a relevance judgment.",
    "success@k": "1 if at least one relevant case appears in the top k.",
    "zero_result": "1 if the system returned no case.",
}
