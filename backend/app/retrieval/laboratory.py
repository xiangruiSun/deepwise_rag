"""Structured laboratory features and similarity (spec section 13).

Numeric laboratory similarity is computed from typed values — never from text
embeddings.  Key guarantees:

* only observations at or before the cutoff (procedure time / ``as_of``) and
  inside the lookback window are used (no future leakage);
* only explicitly convertible units are compared;
* missing values are *unavailable*, never "normal";
* with too few overlapping measurements the similarity is ``None``.

The similarity is a monotone transform of a robust-scaled distance.  It is not
a probability and says nothing about diagnostic equivalence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np

from app.core.config import LaboratoryConfig
from app.ingestion.normalization import LabConceptRegistry


class EventLike(Protocol):
    event_id: str
    event_type: str
    concept_code: str
    event_time: datetime
    numeric_value: float | None
    text_value: str | None
    unit: str | None


@dataclass
class LabFeature:
    concept_key: str
    display_name: str
    value: float  # canonical unit
    unit: str
    aggregate: str
    n_observations: int
    event_id: str  # authoritative source record of the reported value
    event_time: datetime
    original_value: float
    original_unit: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_key": self.concept_key,
            "display_name": self.display_name,
            "value": self.value,
            "unit": self.unit,
            "aggregate": self.aggregate,
            "n_observations": self.n_observations,
            "event_id": self.event_id,
            "event_time": self.event_time.isoformat(),
            "original_value": self.original_value,
            "original_unit": self.original_unit,
        }


@dataclass
class CaseLabFeatures:
    cutoff: datetime
    lookback_days: int
    aggregate: str
    features: dict[str, LabFeature] = field(default_factory=dict)
    # concept -> reason the concept has observations but no comparable value
    unavailable: dict[str, str] = field(default_factory=dict)


class LaboratoryFeatureExtractor:
    def __init__(self, registry: LabConceptRegistry | None = None) -> None:
        self.registry = registry or LabConceptRegistry()

    def extract(
        self,
        events: Iterable[EventLike],
        cutoff: datetime,
        lookback_days: int,
        aggregate: str = "last",
    ) -> CaseLabFeatures:
        if aggregate not in ("last", "mean"):
            raise ValueError(f"unsupported laboratory aggregate '{aggregate}'")
        window_start = cutoff - timedelta(days=lookback_days)
        result = CaseLabFeatures(cutoff=cutoff, lookback_days=lookback_days, aggregate=aggregate)
        comparable: dict[str, list[tuple[datetime, float, EventLike]]] = defaultdict(list)
        reasons: dict[str, str] = {}
        for ev in events:
            if ev.event_type not in ("laboratory", "vital"):
                continue
            # temporal validity: strictly nothing after the cutoff, nothing before the window
            if ev.event_time > cutoff or ev.event_time < window_start:
                continue
            concept = self.registry.by_code(ev.concept_code)
            if concept is None:
                continue
            if ev.numeric_value is None:
                reasons.setdefault(concept.key, "non-numeric value")
                continue
            value, reason = concept.to_canonical(ev.numeric_value, ev.unit)
            if value is None:
                reasons.setdefault(concept.key, reason or "not comparable")
                continue
            comparable[concept.key].append((ev.event_time, value, ev))
        for key, observations in comparable.items():
            concept = self.registry.by_key(key)
            assert concept is not None
            observations.sort(key=lambda o: (o[0], o[2].event_id))
            latest_time = observations[-1][0]
            latest = [o for o in observations if o[0] == latest_time]
            if aggregate == "last":
                distinct = {round(o[1], 9) for o in latest}
                if len(distinct) > 1:
                    result.unavailable[key] = "conflicting values at the same timestamp"
                    continue
                value = latest[-1][1]
            else:
                value = float(np.mean([o[1] for o in observations]))
            source = latest[-1][2]
            assert source.numeric_value is not None
            result.features[key] = LabFeature(
                concept_key=key,
                display_name=concept.display_name,
                value=value,
                unit=concept.canonical_unit,
                aggregate=aggregate,
                n_observations=len(observations),
                event_id=source.event_id,
                event_time=latest_time,
                original_value=source.numeric_value,
                original_unit=source.unit,
            )
        for key, reason in reasons.items():
            if key not in result.features and key not in result.unavailable:
                result.unavailable[key] = reason
        return result


@dataclass(frozen=True)
class RobustStat:
    median: float
    iqr: float
    n: int

    def scale(self, value: float) -> float:
        return (value - self.median) / self.iqr


def fit_robust_statistics(
    training_features: Iterable[CaseLabFeatures], min_samples: int = 5
) -> dict[str, RobustStat]:
    """Median / IQR per concept.  Must be called with TRAINING-split cases only."""
    values: dict[str, list[float]] = defaultdict(list)
    for case in training_features:
        for key, feature in case.features.items():
            values[key].append(feature.value)
    stats: dict[str, RobustStat] = {}
    for key, vals in values.items():
        if len(vals) < min_samples:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        iqr = float(q75 - q25)
        if iqr <= 0:
            spread = float(np.std(vals))
            iqr = spread if spread > 0 else 1.0
        stats[key] = RobustStat(median=float(q50), iqr=iqr, n=len(vals))
    return stats


@dataclass
class LabSimilarity:
    available: bool
    similarity: float | None
    distance: float | None
    n_comparable: int
    n_query_measurements: int
    coverage: float  # fraction of the query's measurements that were comparable
    per_concept: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "similarity": self.similarity,
            "distance": self.distance,
            "n_comparable": self.n_comparable,
            "n_query_measurements": self.n_query_measurements,
            "coverage": self.coverage,
            "per_concept": self.per_concept,
            "reason": self.reason,
        }


class LaboratorySimilarityScorer:
    def __init__(self, stats: dict[str, RobustStat], config: LaboratoryConfig) -> None:
        self.stats = stats
        self.config = config

    def score(self, query: CaseLabFeatures, candidate: CaseLabFeatures) -> LabSimilarity:
        n_query = len(query.features)
        if n_query == 0:
            return LabSimilarity(False, None, None, 0, 0, 0.0, reason="query has no laboratory measurements")
        per_concept: list[dict[str, Any]] = []
        weighted, total_weight = 0.0, 0.0
        for key in sorted(query.features):
            q = query.features[key]
            c = candidate.features.get(key)
            stat = self.stats.get(key)
            if c is None or stat is None:
                continue
            weight = self.config.concept_weights.get(key, 1.0)
            if weight <= 0:
                continue
            diff = abs(stat.scale(q.value) - stat.scale(c.value))
            weighted += weight * diff
            total_weight += weight
            per_concept.append(
                {
                    "concept_key": key,
                    "display_name": q.display_name,
                    "unit": q.unit,
                    "query_value": q.value,
                    "candidate_value": c.value,
                    "standardized_difference": round(diff, 4),
                    "query_event_id": q.event_id,
                    "candidate_event_id": c.event_id,
                }
            )
        n = len(per_concept)
        coverage = n / n_query
        if n < self.config.min_overlap or total_weight == 0:
            return LabSimilarity(
                False, None, None, n, n_query, coverage, per_concept,
                reason=f"only {n} overlapping valid measurement(s); {self.config.min_overlap} required",
            )
        distance = weighted / total_weight
        similarity = math.exp(-distance / self.config.distance_scale)
        return LabSimilarity(True, similarity, distance, n, n_query, coverage, per_concept)


def diagnosis_similarity(query_codes: set[str], candidate_codes: set[str]) -> float | None:
    """Jaccard overlap of pre-procedure diagnosis code sets; ``None`` when either is empty
    (an empty history is 'unknown', not 'identical')."""
    if not query_codes or not candidate_codes:
        return None
    return len(query_codes & candidate_codes) / len(query_codes | candidate_codes)
