"""Contradiction screening, feedback failure analysis and the query cost model."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.corpus.analysis import CostInputs, detect_conflicts, estimate_query_cost, feedback_failure_analysis
from app.corpus.context import ContextBundle, ContextItem
from app.corpus.retrieval import RetrievedChunk
from app.corpus.service import QueryRequest
from app.db.models import CorpusFeedback
from tests.corpus.conftest import Corpus, new_session_factory


def _item(source: str, doc: str, text: str, when: str | None = None, patient: str | None = "P1") -> ContextItem:
    chunk = RetrievedChunk(chunk_id=f"{doc}-00000", document_id=doc, document_version=1, chunk_index=0, chunk_count=1,
                           text=text, indexed_text=text, score=1.0, title="t", document_type="clinical_report", source_system="s",
                           source_record_id=doc, event_time=when, source_updated_at=None, token_count=10,
                           section_path=None, metadata={"x_patient_pseudo_id": patient} if patient else {})
    return ContextItem(source, chunk, text, 10)


def _bundle(*items: ContextItem) -> ContextBundle:
    return ContextBundle(list(items), 10 * len(items), 1000)


def test_contradictions_between_documents_are_flagged_with_the_newer_source() -> None:
    old = _item("S1", "doc_a", "IMPRESSION: Follow-up status: unresolved, reassessment planned.", "2025-01-01T00:00:00")
    new = _item("S2", "doc_b", "IMPRESSION: Follow-up status: resolved after anticoagulation.", "2025-06-01T00:00:00")
    conflicts = detect_conflicts(_bundle(old, new))
    assert len(conflicts) == 1 and conflicts[0]["sources"] == ["S1", "S2"]
    assert conflicts[0]["kind"] == "resolved/unresolved" and conflicts[0]["newer_source"] == "S2"
    negated = detect_conflicts(_bundle(
        _item("S1", "doc_a", "CT shows segmental pulmonary embolism in the right lower lobe."),
        _item("S2", "doc_b", "CT shows no segmental pulmonary embolism in the right lower lobe.")))
    assert [c["kind"] for c in negated] == ["negation"] and negated[0]["newer_source"] is None
    # not conflicts: the same document, another patient, unrelated statements, agreeing statements
    assert not detect_conflicts(_bundle(old, _item("S2", "doc_a", new.text)))
    assert not detect_conflicts(_bundle(old, _item("S2", "doc_b", new.text, patient="P2")))
    assert not detect_conflicts(_bundle(old, _item("S2", "doc_b", "Hemoglobin was stable at 13.2 g/dL today.")))
    assert not detect_conflicts(_bundle(old, _item("S2", "doc_b", old.text)))


def test_superseded_and_current_reports_conflict_only_when_history_is_included(corpus: Corpus) -> None:
    flagged = 0
    for query in corpus.by_category("freshness_superseded"):
        auth = corpus.auth(query["tenant_id"])
        current = corpus.runtime.service.answer_sync(auth, QueryRequest(query=query["question"]), use_cache=False)
        assert current["conflicts"] == []                    # the superseded report is filtered out by default
        history = corpus.runtime.service.answer_sync(auth, QueryRequest(query=query["question"],
                                                                       strategy="no_freshness"), use_cache=False)
        if history["conflicts"]:
            flagged += 1
            assert any("disagree" in w for w in history["warnings"])
            conflict = history["conflicts"][0]
            assert set(conflict["documents"]) & set(query["stale_documents"])
    assert flagged >= 1


def test_feedback_failure_analysis_builds_regression_candidates_without_text() -> None:
    factory, now = new_session_factory(), datetime(2026, 6, 1)

    def add(rating: str, query_hash: str, user: str, categories: list[str], age_days: int = 1,
            strategy: str = "hybrid", tenant: str = "tenant-000") -> None:
        with factory() as session:
            session.add(CorpusFeedback(tenant_id=tenant, user_id=user, trace_id=f"t-{user}-{query_hash}-{age_days}",
                                       query_hash=query_hash, rating=rating, categories=categories,
                                       chunk_ids=["doc_1-00000"], strategy=strategy, index_version="idx_v1",
                                       created_at=now - timedelta(days=age_days)))
            session.commit()

    add("down", "h1", "u1", ["wrong_citation"])
    add("down", "h1", "u2", ["wrong_citation", "outdated"])
    add("down", "h2", "u1", ["incomplete"], strategy="bm25")
    add("up", "h3", "u3", [])
    add("down", "h1", "u9", ["wrong_citation"], age_days=90)          # outside the window
    add("down", "h1", "u4", ["wrong_citation"], tenant="tenant-001")   # another tenant is another cluster
    report = feedback_failure_analysis(factory, days=30, min_reports=2, now=now)
    assert (report["feedback"], report["negative"], report["negative_rate"]) == (5, 4, 0.8)
    assert report["by_category"]["wrong_citation"] == 3 and report["by_strategy"]["bm25"]["negative_rate"] == 1.0
    assert len(report["regression_candidates"]) == 1
    candidate = report["regression_candidates"][0]
    assert (candidate["tenant_id"], candidate["query_hash"], candidate["reports"], candidate["users"]) == \
        ("tenant-000", "h1", 2, 2)
    assert candidate["needs_human_review"] and candidate["flagged_chunks"] == {"doc_1-00000": 2}
    assert "question" not in candidate and "query" not in candidate


def test_cost_model_separates_measured_inputs_from_assumed_prices() -> None:
    base = estimate_query_cost(CostInputs(prompt_tokens=2000, completion_tokens=150, rerank_pairs=20,
                                          gpu_hour_usd=3.6, utilisation=1.0))
    assert base["gpu_seconds_per_query"] == {"generation": 0.2, "reranking": 0.01}
    assert base["usd_per_1k_queries"] == {"generation": 0.2, "reranking": 0.01, "fixed": 0.0, "total": 0.21}
    half = estimate_query_cost(CostInputs(prompt_tokens=1000, completion_tokens=75, rerank_pairs=20,
                                          gpu_hour_usd=3.6, utilisation=1.0, fixed_monthly_usd=1000))
    assert half["usd_per_1k_queries"]["generation"] == pytest.approx(0.1)
    assert half["usd_per_1k_queries"]["fixed"] == 1.0 and "assumptions" in half["note"]
    with pytest.raises(ValueError):
        CostInputs(prompt_tokens=-1, completion_tokens=0)
