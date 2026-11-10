"""RRF, score fusion, patient-level aggregation, rerankers and multimodal fusion."""

from __future__ import annotations

import pytest

from app.core.config import AggregationConfig, MultimodalConfig, RerankerConfig
from app.ranking.aggregation import CaseAggregator, RetrievedDocument
from app.ranking.fusion import normalize_scores, reciprocal_rank_fusion, weighted_score_fusion
from app.ranking.multimodal import MultimodalCaseRanker
from app.ranking.rerankers import LexicalOverlapReranker, RerankerRegistry, build_reranker


def test_rrf_formula_ranks_start_at_one_and_missing_contributes_zero() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["b", "d"]}, k=60)
    scores = {f.item_id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [f.item_id for f in fused] == ["b", "a", "d", "c"]
    b = next(f for f in fused if f.item_id == "b")
    assert b.ranks == {"bm25": 2, "dense": 1} and set(b.contributions) == {"bm25", "dense"}


def test_rrf_weights_ties_and_duplicates() -> None:
    fused = reciprocal_rank_fusion({"x": ["a", "b"], "y": ["b", "a"]}, k=60)
    assert [f.item_id for f in fused] == ["a", "b"]  # equal scores → deterministic id order
    weighted = reciprocal_rank_fusion({"x": ["a", "b"], "y": ["b", "a"]}, k=60, weights={"y": 2.0})
    assert weighted[0].item_id == "b" and weighted[0].score == pytest.approx(1 / 62 + 2 / 61)
    dup = reciprocal_rank_fusion({"x": ["a", "a", "b"]}, k=1)
    assert {f.item_id: f.ranks["x"] for f in dup} == {"a": 1, "b": 2}  # a list cannot vote twice
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"x": ["a"]}, k=0)


def test_weighted_score_fusion_normalises_each_list() -> None:
    fused = weighted_score_fusion({"bm25": {"a": 10.0, "b": 5.0, "c": 0.0}, "dense": {"b": 0.9, "c": 0.8}},
                                  weights={"bm25": 0.5, "dense": 0.5})
    scores = {f.item_id: f.score for f in fused}
    assert scores["a"] == pytest.approx(0.5) and scores["b"] == pytest.approx(0.25 + 0.5)
    assert scores["c"] == pytest.approx(0.0) and fused[0].item_id == "b"
    assert normalize_scores({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}
    with pytest.raises(ValueError):
        weighted_score_fusion({"x": {"a": 1.0}}, weights={"x": 0.0})


def doc(doc_id: str, case: str, patient: str, score: float, source: str | None = None, chunk: int = 0,
        content: str | None = None) -> RetrievedDocument:
    return RetrievedDocument(document_id=doc_id, case_id=case, patient_id=patient, source_id=source or f"r_{case}",
                             content=content or f"text {doc_id}", chunk_index=chunk, score=score,
                             branch_scores={"bm25": score})


def test_aggregation_returns_distinct_cases_with_best_score() -> None:
    docs = [doc("d1", "A", "p1", 0.9), doc("d2", "B", "p2", 0.8), doc("d3", "A", "p1", 0.7, chunk=1),
            doc("d4", "C", "p3", 0.6)]
    cases = CaseAggregator(AggregationConfig(method="best_score")).aggregate(docs)
    assert [c.case_id for c in cases] == ["A", "B", "C"]
    assert [c.text_rank for c in cases] == [1, 2, 3] and cases[0].text_score == 0.9
    assert cases[0].n_documents == 2 and [d.document_id for d in cases[0].evidence] == ["d1", "d3"]


def test_many_chunks_do_not_dominate() -> None:
    many = [doc(f"m{i}", "MANY", "p1", 0.50 - i * 0.01, chunk=i) for i in range(8)]
    single = [doc("s1", "ONE", "p2", 0.55)]
    ordered = sorted(many + single, key=lambda d: -d.score)
    for method in ("best_score", "best_rank", "top_n_mean"):
        cases = CaseAggregator(AggregationConfig(method=method, top_n=3)).aggregate(list(ordered))  # type: ignore[arg-type]
        assert cases[0].case_id == "ONE", method
    # top_n_mean cannot be raised by adding weaker chunks
    top = CaseAggregator(AggregationConfig(method="top_n_mean", top_n=3)).aggregate(list(ordered))
    assert next(c for c in top if c.case_id == "MANY").text_score == pytest.approx((0.50 + 0.49 + 0.48) / 3)


def test_overlapping_chunks_of_one_report_are_deduplicated() -> None:
    docs = [doc("d1", "A", "p1", 0.9, source="rep", chunk=0, content="same text"),
            doc("d2", "A", "p1", 0.8, source="rep", chunk=1, content="same text"),
            doc("d3", "A", "p1", 0.7, source="rep", chunk=2, content="different")]
    case = CaseAggregator(AggregationConfig()).aggregate(docs)[0]
    assert [d.document_id for d in case.evidence] == ["d1", "d3"]


def test_same_patient_and_query_case_are_excluded() -> None:
    docs = [doc("d1", "Q2", "query_patient", 0.99), doc("d2", "B", "p2", 0.8), doc("d3", "SELF", "other", 0.7)]
    cases = CaseAggregator(AggregationConfig()).aggregate(docs, exclude_patient_ids={"query_patient"},
                                                          exclude_case_ids={"SELF"})
    assert [c.case_id for c in cases] == ["B"]


def test_case_level_rrf_differs_from_document_level_fusion() -> None:
    bm25 = [doc("b1", "A", "p1", 9.0), doc("b2", "A", "p1", 8.0, chunk=1), doc("b3", "B", "p2", 7.0)]
    dense = [doc("x1", "B", "p2", 0.9), doc("x2", "C", "p3", 0.8)]
    agg = CaseAggregator(AggregationConfig(method="case_rrf", rrf_k=60))
    cases = agg.aggregate(bm25 + dense, {"bm25": bm25, "dense": dense})
    scores = {c.case_id: c.text_score for c in cases}
    # case ranks: bm25 A=1, B=2 (A's second chunk does not occupy a rank); dense B=1, C=2
    assert scores["A"] == pytest.approx(1 / 61) and scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 62) and cases[0].case_id == "B"
    with pytest.raises(ValueError):
        agg.aggregate(bm25)


def test_lexical_reranker_and_registry() -> None:
    config = RerankerConfig(enabled=True, key="lex", provider="lexical_overlap", depth=10)
    reranker = build_reranker(config)
    assert isinstance(reranker, LexicalOverlapReranker)
    scores = reranker.score("no pulmonary embolism", [
        "No evidence of pulmonary embolism.", "Acute pulmonary embolism in the right lower lobe.", "Pleural effusion."])
    assert scores[0] > scores[1] > scores[2]
    assert build_reranker(RerankerConfig()) is None and RerankerRegistry().get(RerankerConfig()) is None
    registry = RerankerRegistry()
    assert registry.get(config) is registry.get(config.model_copy(update={"depth": 20}))


def mm(**kw) -> MultimodalConfig:  # type: ignore[no-untyped-def]
    base = {"enabled": True, "modalities": ["text", "laboratory", "image"],
                "weights": {"text": 0.5, "laboratory": 0.3, "image": 0.2}, "family": "rank", "rrf_k": 60}
    return MultimodalConfig(**{**base, **kw})


SCORES = {
    "A": {"text": 0.9, "laboratory": 0.2, "image": 0.5},
    "B": {"text": 0.8, "laboratory": 0.9, "image": None},
    "C": {"text": 0.1, "laboratory": None, "image": None},
}


def test_multimodal_rank_fusion_with_renormalised_missing_modalities() -> None:
    results = {r.case_id: r for r in MultimodalCaseRanker(mm(missing_policy="renormalize")).rank(SCORES)}
    a, b, c = results["A"], results["B"], results["C"]
    assert a.effective_weights == {"text": 0.5, "laboratory": 0.3, "image": 0.2}
    assert b.effective_weights == pytest.approx({"text": 0.625, "laboratory": 0.375, "image": 0.0})
    assert c.effective_weights == {"text": 1.0, "laboratory": 0.0, "image": 0.0}
    assert a.score == pytest.approx(0.5 / 61 + 0.3 / 62 + 0.2 / 61)
    assert b.score == pytest.approx(0.625 / 62 + 0.375 / 61)
    assert b.coverage == {"text": True, "laboratory": True, "image": False}
    assert b.component_ranks == {"text": 2, "laboratory": 1, "image": None}
    assert sorted(results, key=lambda k: results[k].rank) == ["A", "B", "C"]


def test_multimodal_zero_and_exclude_policies() -> None:
    zero = {r.case_id: r for r in MultimodalCaseRanker(mm(missing_policy="zero")).rank(SCORES)}
    assert zero["B"].effective_weights == {"text": 0.5, "laboratory": 0.3, "image": 0.0}
    assert zero["B"].score == pytest.approx(0.5 / 62 + 0.3 / 61)
    exclude = MultimodalCaseRanker(mm(missing_policy="exclude")).rank(SCORES)
    assert [r.case_id for r in exclude] == ["A"]


def test_multimodal_score_fusion_and_weight_validation() -> None:
    results = {r.case_id: r for r in MultimodalCaseRanker(mm(family="score", missing_policy="zero")).rank(SCORES)}
    # min-max per modality: text A=1, B=.875, C=0; laboratory A=0, B=1; image A=1 (single value)
    assert results["A"].score == pytest.approx(0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 1.0)
    assert results["B"].score == pytest.approx(0.5 * 0.875 + 0.3 * 1.0)
    with pytest.raises(ValueError):
        MultimodalCaseRanker(mm(weights={"text": 0.5, "laboratory": 0.0, "image": 0.2}))
    with pytest.raises(ValueError):
        MultimodalConfig(weights={"text": -1.0})


def test_equal_scores_are_ordered_by_case_not_by_document_id() -> None:
    from app.indexing.backend import Hit
    from app.retrieval.retrievers import _stable_order

    hits = [Hit("doc_zzz", 0.9, {"case_id": "case_a", "chunk_index": 0}),
            Hit("doc_aaa", 0.9, {"case_id": "case_b", "chunk_index": 0}),
            Hit("doc_mmm", 0.95, {"case_id": "case_c", "chunk_index": 0})]
    assert [h.source["case_id"] for h in _stable_order(hits)] == ["case_c", "case_a", "case_b"]
    assert [h.id for h in _stable_order(list(reversed(hits)))] == ["doc_mmm", "doc_zzz", "doc_aaa"]
