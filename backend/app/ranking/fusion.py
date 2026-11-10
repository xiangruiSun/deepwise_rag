"""Rank- and score-based fusion of ranked lists (spec section 11.5).

These functions are level-agnostic: the *caller* decides whether the fused
items are documents (document-level fusion) or cases (case-level fusion).  The
two are different operations; ``RetrievalService`` performs document-level
fusion and ``CaseAggregator`` (method ``case_rrf``) performs case-level fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusedItem:
    item_id: str
    score: float
    contributions: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedItem]:
    """``RRF(c) = Σ_m w_m / (k + rank_m(c))`` with ranks starting at 1.

    A candidate missing from a list contributes 0 for that list.  Ties are broken
    by item id so the output is deterministic.
    """
    if k < 1:
        raise ValueError("RRF rank constant k must be >= 1")
    weights = weights or {}
    items: dict[str, FusedItem] = {}
    for name in sorted(rankings):
        weight = weights.get(name, 1.0)
        seen: set[str] = set()
        rank = 0
        for item_id in rankings[name]:
            if item_id in seen:  # a list must not vote twice for the same item
                continue
            seen.add(item_id)
            rank += 1
            item = items.setdefault(item_id, FusedItem(item_id, 0.0))
            contribution = weight / (k + rank)
            item.score += contribution
            item.contributions[name] = contribution
            item.ranks[name] = rank
    return sorted(items.values(), key=lambda i: (-i.score, i.item_id))


def normalize_scores(scores: dict[str, float], method: str = "min_max") -> dict[str, float]:
    """Normalise one list's scores to a comparable range.

    ``min_max`` maps to [0, 1] (a constant list maps to 1.0); ``z_score`` standardises
    and squashes through a logistic so missing-item contribution 0 stays the minimum.
    """
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if method == "min_max":
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return dict.fromkeys(scores, 1.0)
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    if method == "z_score":
        mean, std = float(values.mean()), float(values.std())
        if std < 1e-12:
            return dict.fromkeys(scores, 0.5)
        return {k: float(1.0 / (1.0 + np.exp(-(v - mean) / std))) for k, v in scores.items()}
    raise ValueError(f"unknown score normalisation '{method}'")


def weighted_score_fusion(
    scored_lists: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    normalization: str = "min_max",
) -> list[FusedItem]:
    """Normalised weighted score fusion.  Weights are normalised to sum to 1."""
    weights = weights or {}
    names = sorted(scored_lists)
    raw_weights = {n: weights.get(n, 1.0) for n in names}
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("at least one fusion weight must be positive")
    items: dict[str, FusedItem] = {}
    for name in names:
        normalized = normalize_scores(scored_lists[name], normalization)
        ordered = sorted(scored_lists[name], key=lambda i: (-scored_lists[name][i], i))
        for rank, item_id in enumerate(ordered, start=1):
            item = items.setdefault(item_id, FusedItem(item_id, 0.0))
            contribution = raw_weights[name] / total * normalized[item_id]
            item.score += contribution
            item.contributions[name] = contribution
            item.ranks[name] = rank
    return sorted(items.values(), key=lambda i: (-i.score, i.item_id))
