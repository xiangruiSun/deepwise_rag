"""Case-level multimodal fusion (spec section 15).

Inputs are per-case modality scores (``None`` = modality unavailable for that
case).  Two fusion families are implemented — weighted RRF over per-modality
rankings and weighted normalised-score fusion — with an explicit
missing-modality policy.  Effective weights and coverage are returned for every
case so that a result can always be explained.

The default weights in the configuration are hypotheses to be tuned on
validation data; they are not medically established coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import MultimodalConfig
from app.ranking.fusion import normalize_scores


@dataclass
class MultimodalResult:
    case_id: str
    score: float
    rank: int
    component_scores: dict[str, float | None] = field(default_factory=dict)
    component_ranks: dict[str, int | None] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, bool] = field(default_factory=dict)


class MultimodalCaseRanker:
    def __init__(self, config: MultimodalConfig) -> None:
        self.config = config
        self.modalities = list(config.modalities)
        missing = [m for m in self.modalities if config.weights.get(m, 0.0) <= 0]
        if missing:
            raise ValueError(f"enabled modalities need a positive fusion weight: {missing}")

    def rank(self, scores: dict[str, dict[str, float | None]]) -> list[MultimodalResult]:
        """``scores[case_id][modality] -> score or None``.  Higher is more similar."""
        modalities: list[str] = [str(m) for m in self.modalities]
        base: dict[str, float] = {m: self.config.weights[m] for m in modalities}
        total_weight = sum(base.values())
        case_ids = sorted(scores)
        if self.config.missing_policy == "exclude":
            case_ids = [c for c in case_ids if all(scores[c].get(m) is not None for m in modalities)]

        ranks: dict[str, dict[str, int]] = {}
        normalized: dict[str, dict[str, float]] = {}
        for modality in modalities:
            available = {c: float(scores[c][modality]) for c in case_ids  # type: ignore[arg-type]
                         if scores[c].get(modality) is not None}
            ordered = sorted(available, key=lambda c: (-available[c], c))
            ranks[modality] = {c: r for r, c in enumerate(ordered, start=1)}
            normalized[modality] = normalize_scores(available, "min_max")

        results: list[MultimodalResult] = []
        for case_id in case_ids:
            coverage = {m: scores[case_id].get(m) is not None for m in modalities}
            available_weight = sum(base[m] for m in modalities if coverage[m])
            if available_weight == 0:
                continue
            if self.config.missing_policy == "renormalize":
                # redistribute the weight of unavailable modalities over the available ones
                effective = {m: (base[m] / available_weight if coverage[m] else 0.0) for m in modalities}
            else:  # "zero" and "exclude": fixed weights, a missing modality contributes 0
                effective = {m: (base[m] / total_weight if coverage[m] else 0.0) for m in modalities}
            contributions: dict[str, float] = {}
            for modality in modalities:
                if not coverage[modality]:
                    continue
                if self.config.family == "rank":
                    value = 1.0 / (self.config.rrf_k + ranks[modality][case_id])
                else:
                    value = normalized[modality][case_id]
                contributions[modality] = effective[modality] * value
            results.append(
                MultimodalResult(
                    case_id=case_id,
                    score=sum(contributions.values()),
                    rank=0,
                    component_scores={m: scores[case_id].get(m) for m in modalities},
                    component_ranks={m: ranks[m].get(case_id) for m in modalities},
                    contributions=contributions,
                    effective_weights=effective,
                    coverage=coverage,
                )
            )
        results.sort(key=lambda r: (-r.score, r.case_id))
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results
