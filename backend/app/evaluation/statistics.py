"""Paired bootstrap confidence intervals and the model-selection policy (spec section 30)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PairedComparison:
    metric: str
    n_queries: int
    mean_baseline: float | None
    mean_candidate: float | None
    mean_difference: float | None
    ci_low: float | None
    ci_high: float | None
    p_not_better: float | None  # share of bootstrap resamples where candidate <= baseline
    wins: int
    losses: int
    ties: int
    top_decile_share: float | None  # share of the total gain owed to the top 10% of queries

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @property
    def credible_improvement(self) -> bool:
        return self.ci_low is not None and self.ci_low > 0


def paired_bootstrap(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    metric: str = "ndcg@10",
    samples: int = 2000,
    seed: int = 13,
    confidence: float = 0.95,
) -> PairedComparison:
    """Paired bootstrap over evaluation queries (same queries for both systems).

    Queries where the metric is undefined for either system are excluded.
    """
    query_ids = sorted(q for q in baseline if q in candidate
                       and baseline[q] is not None and candidate[q] is not None)
    n = len(query_ids)
    if n == 0:
        return PairedComparison(metric, 0, None, None, None, None, None, None, 0, 0, 0, None)
    base = np.asarray([baseline[q] for q in query_ids], dtype=np.float64)
    cand = np.asarray([candidate[q] for q in query_ids], dtype=np.float64)
    diff = cand - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(samples, n))
    boot = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    gains = np.sort(np.clip(diff, 0, None))[::-1]
    total_gain = float(gains.sum())
    top = max(1, int(np.ceil(0.1 * n)))
    return PairedComparison(
        metric=metric,
        n_queries=n,
        mean_baseline=float(base.mean()),
        mean_candidate=float(cand.mean()),
        mean_difference=float(diff.mean()),
        ci_low=float(np.quantile(boot, alpha)),
        ci_high=float(np.quantile(boot, 1.0 - alpha)),
        p_not_better=float((boot <= 0).mean()),
        wins=int((diff > 1e-12).sum()),
        losses=int((diff < -1e-12).sum()),
        ties=int((np.abs(diff) <= 1e-12).sum()),
        top_decile_share=float(gains[:top].sum() / total_gain) if total_gain > 0 else None,
    )


SAFETY_GATES = (
    "authorization_isolation",
    "split_isolation",
    "temporal_cutoff",
    "citation_validity",
    "reproducible_execution",
)


def select_configuration(
    runs: list[dict[str, Any]],
    baseline_id: str,
    per_query: dict[str, dict[str, float | None]],
    metric: str = "ndcg@10",
    max_latency_ratio: float = 5.0,
    min_faithfulness: float | None = None,
    samples: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    """Apply the predeclared selection policy on VALIDATION results.

    1. Reject any run that fails a safety gate (or the faithfulness threshold).
    2. Among the rest, a candidate replaces the baseline only if its paired-bootstrap
       95% CI for the primary metric excludes zero AND its p95 latency is within
       ``max_latency_ratio`` of the baseline's.
    3. If several candidates qualify, the highest primary metric wins; ties go to
       the lower latency.  If none qualifies, the simpler baseline is selected.
    """
    by_id = {r["experiment_id"]: r for r in runs}
    if baseline_id not in by_id:
        raise ValueError(f"baseline experiment '{baseline_id}' not found among runs")
    baseline = by_id[baseline_id]
    base_latency = baseline.get("p95_latency_ms")
    decisions: list[dict[str, Any]] = []
    qualified: list[tuple[float, float, str]] = []
    for run in runs:
        exp_id = run["experiment_id"]
        decision: dict[str, Any] = {"experiment_id": exp_id, "primary_metric": run.get(metric)}
        failed = [g for g in SAFETY_GATES if run.get("gates", {}).get(g) is False]
        if failed:
            decision.update(status="rejected", reason=f"failed safety gate(s): {failed}")
        elif (min_faithfulness is not None and run.get("faithfulness") is not None
              and run["faithfulness"] < min_faithfulness):
            decision.update(status="rejected", reason="faithfulness below the predefined threshold")
        elif exp_id == baseline_id:
            decision.update(status="baseline")
        else:
            comparison = paired_bootstrap(per_query[baseline_id], per_query[exp_id], metric, samples, seed)
            decision["comparison"] = comparison.as_dict()
            latency = run.get("p95_latency_ms")
            too_slow = bool(base_latency and latency and latency > max_latency_ratio * base_latency)
            if not comparison.credible_improvement:
                decision.update(status="not_selected",
                                reason="no credible improvement over the baseline (95% CI includes 0)")
            elif too_slow:
                decision.update(status="not_selected",
                                reason=f"p95 latency exceeds {max_latency_ratio}x the baseline")
            else:
                decision.update(status="qualified")
                qualified.append((-(run.get(metric) or 0.0), latency or 0.0, exp_id))
        decisions.append(decision)
    baseline_rejected = any(d["experiment_id"] == baseline_id and d["status"] == "rejected" for d in decisions)
    if qualified:
        selected = sorted(qualified)[0][2]
        rationale = "highest validation primary metric among configurations with a credible improvement"
    elif baseline_rejected:
        selected, rationale = None, "baseline failed a safety gate and no candidate qualified"
    else:
        selected, rationale = baseline_id, "no configuration showed a credible improvement; simpler baseline kept"
    return {
        "policy": {
            "primary_metric": metric, "selection_data": "validation", "safety_gates": list(SAFETY_GATES),
            "max_latency_ratio": max_latency_ratio, "min_faithfulness": min_faithfulness,
            "bootstrap_samples": samples, "seed": seed,
        },
        "baseline": baseline_id,
        "selected": selected,
        "rationale": rationale,
        "decisions": decisions,
    }
