"""Unit conversion, missing-value behaviour, temporal cutoff and robust normalisation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from app.core.config import LaboratoryConfig
from app.ingestion.normalization import (
    LabConceptRegistry,
    classify_event,
    normalize_diagnosis_code,
    normalize_unit,
    parse_numeric,
)
from app.retrieval.laboratory import (
    LaboratoryFeatureExtractor,
    LaboratorySimilarityScorer,
    RobustStat,
    diagnosis_similarity,
    fit_robust_statistics,
)

T0 = datetime(2020, 5, 1, 12, 0)


@dataclass
class Ev:
    event_id: str
    concept_code: str
    event_time: datetime
    numeric_value: float | None
    unit: str | None
    event_type: str = "laboratory"
    text_value: str | None = None


def ev(i: int, code: str, hours: float, value: float | None, unit: str | None, **kw) -> Ev:  # type: ignore[no-untyped-def]
    return Ev(f"e{i}", code, T0 + timedelta(hours=hours), value, unit, **kw)


def test_explicit_unit_conversions_only() -> None:
    reg = LabConceptRegistry()
    creat = reg.by_key("creatinine")
    assert creat.to_canonical(88.4, "umol/L")[0] == pytest.approx(1.0)
    assert creat.to_canonical(88.4, "µmol/L")[0] == pytest.approx(1.0)
    assert creat.to_canonical(1.2, "mg/dL") == (1.2, None)
    assert reg.by_key("hemoglobin").to_canonical(135, "g/L")[0] == pytest.approx(13.5)
    assert reg.by_key("troponin_i").to_canonical(40, "ng/L")[0] == pytest.approx(0.04)
    assert reg.by_key("d_dimer").to_canonical(1.2, "mg/L FEU")[0] == pytest.approx(1200)
    value, reason = reg.by_key("d_dimer").to_canonical(12.0, "arb'U/mL")
    assert value is None and "unsupported unit" in reason
    assert creat.to_canonical(1.0, None) == (None, "missing unit")
    assert normalize_unit(" 10*3/uL ") == "k/ul" and normalize_unit("") is None


def test_code_parsing_and_event_typing() -> None:
    assert classify_event("LOINC/718-7", "measurement") == "laboratory"
    assert classify_event("LOINC/8867-4", "measurement") == "vital"
    assert classify_event("ICD10CM/I26.99", None) == "diagnosis"
    assert classify_event("RxNorm/5224", "drug_exposure") == "medication"
    assert classify_event("Weird/1", None) == "other"
    assert normalize_diagnosis_code("icd10cm/i26.99") == "ICD10CM/I26.99"
    assert parse_numeric("<0.5") is None and parse_numeric("3.20") == 3.2 and parse_numeric(float("nan")) is None


def test_temporal_cutoff_excludes_future_and_out_of_window_values() -> None:
    events = [
        ev(1, "LOINC/48065-7", -3, 2500.0, "ng/mL FEU"),
        ev(2, "LOINC/48065-7", +48, 99999.0, "ng/mL FEU"),  # AFTER the procedure: must never be used
        ev(3, "LOINC/2160-0", -24 * 20, 1.4, "mg/dL"),  # 20 days before
        ev(4, "LOINC/2160-0", -24 * 60, 3.0, "mg/dL"),  # 60 days before
    ]
    extractor = LaboratoryFeatureExtractor()
    f7 = extractor.extract(events, T0, 7)
    assert f7.features["d_dimer"].value == 2500.0 and "creatinine" not in f7.features
    f30 = extractor.extract(events, T0, 30)
    assert f30.features["creatinine"].value == 1.4 and f30.features["creatinine"].n_observations == 1
    f90 = extractor.extract(events, T0, 90, aggregate="mean")
    assert f90.features["creatinine"].value == pytest.approx(2.2)
    for feats in (f7, f30, f90):
        assert all(f.event_time <= T0 for f in feats.features.values())
        assert 99999.0 not in {f.value for f in feats.features.values()}
    # an earlier as-of cutoff hides later (still pre-procedure) measurements
    earlier = extractor.extract(events, T0 - timedelta(hours=5), 7)
    assert "d_dimer" not in earlier.features


def test_unconvertible_missing_and_conflicting_values_are_unavailable() -> None:
    events = [
        ev(1, "LOINC/48065-7", -2, 12.0, "arb'U/mL"),
        ev(2, "LOINC/2524-7", -2, None, "mmol/L", text_value="<0.5"),
        ev(3, "LOINC/718-7", -1, 9.1, "g/dL"),
        ev(4, "LOINC/718-7", -1, 14.6, "g/dL"),  # same timestamp, different value
        ev(5, "LOINC/6690-2", -1, 8.0, "K/uL"),
    ]
    feats = LaboratoryFeatureExtractor().extract(events, T0, 30)
    assert set(feats.features) == {"wbc"}
    assert "unsupported unit" in feats.unavailable["d_dimer"]
    assert feats.unavailable["lactate"] == "non-numeric value"
    assert feats.unavailable["hemoglobin"] == "conflicting values at the same timestamp"
    # an older comparable value is used when the newest one has an unsupported unit
    events.append(ev(6, "LOINC/48065-7", -30, 700.0, "ng/mL FEU"))
    feats = LaboratoryFeatureExtractor().extract(events, T0, 30)
    assert feats.features["d_dimer"].value == 700.0 and "d_dimer" not in feats.unavailable
    # the reported value always keeps its original unit and source record
    conv = LaboratoryFeatureExtractor().extract([ev(7, "LOINC/2160-0", -1, 176.8, "umol/L")], T0, 7)
    feature = conv.features["creatinine"]
    assert (feature.original_value, feature.original_unit, feature.event_id) == (176.8, "umol/L", "e7")
    assert feature.value == pytest.approx(2.0) and feature.unit == "mg/dl"


def _features(values: dict[str, float]):  # type: ignore[no-untyped-def]
    codes = {"d_dimer": ("LOINC/48065-7", "ng/mL FEU"), "creatinine": ("LOINC/2160-0", "mg/dL"),
             "wbc": ("LOINC/6690-2", "K/uL"), "hemoglobin": ("LOINC/718-7", "g/dL")}
    events = [ev(i, codes[k][0], -1, v, codes[k][1]) for i, (k, v) in enumerate(values.items())]
    return LaboratoryFeatureExtractor().extract(events, T0, 7)


def test_robust_statistics_and_similarity() -> None:
    train = [_features({"creatinine": v, "wbc": w}) for v, w in
             zip([0.8, 0.9, 1.0, 1.1, 1.2, 5.0], [6, 7, 8, 9, 10, 11], strict=True)]
    stats = fit_robust_statistics(train)
    assert stats["creatinine"].median == pytest.approx(1.05)
    assert stats["creatinine"].iqr == pytest.approx(0.25)  # robust to the 5.0 outlier
    assert fit_robust_statistics(train[:3]) == {}  # too few samples: no statistic is invented

    scorer = LaboratorySimilarityScorer({"creatinine": RobustStat(1.0, 0.5, 10), "wbc": RobustStat(8.0, 2.0, 10)},
                                        LaboratoryConfig(min_overlap=2))
    query = _features({"creatinine": 1.0, "wbc": 8.0, "d_dimer": 3000.0})
    same = scorer.score(query, _features({"creatinine": 1.0, "wbc": 8.0}))
    assert same.available and same.similarity == pytest.approx(1.0) and same.n_comparable == 2
    assert same.coverage == pytest.approx(2 / 3)  # d-dimer has no statistic/overlap: reported, not hidden
    far = scorer.score(query, _features({"creatinine": 2.0, "wbc": 12.0}))
    assert far.distance == pytest.approx(2.0) and far.similarity == pytest.approx(math.exp(-2.0))
    assert far.similarity < same.similarity


def test_insufficient_overlap_is_unavailable_not_favourable() -> None:
    scorer = LaboratorySimilarityScorer({"creatinine": RobustStat(1.0, 0.5, 10)}, LaboratoryConfig(min_overlap=2))
    query = _features({"creatinine": 1.0, "wbc": 8.0})
    one = scorer.score(query, _features({"creatinine": 1.0}))
    assert not one.available and one.similarity is None and "overlapping" in one.reason
    none = scorer.score(_features({}), _features({"creatinine": 1.0}))
    assert not none.available and none.similarity is None
    # missing measurements are never treated as "normal": no value is imputed
    assert scorer.score(query, _features({})).per_concept == []


def test_diagnosis_similarity_handles_empty_sets() -> None:
    assert diagnosis_similarity({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)
    assert diagnosis_similarity(set(), {"A"}) is None and diagnosis_similarity({"A"}, set()) is None
