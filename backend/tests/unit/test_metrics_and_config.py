"""Retrieval metric formulas (hand-computed), bootstrap, selection policy and config hashing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import (
    AppConfig,
    GenerationConfig,
    RetrievalRecipe,
    load_app_config,
    load_experiment_suite,
    resolve_suite,
)
from app.core.hashing import stable_hash
from app.evaluation.datasets import Qrels, adjudicate, build_annotation_pool, write_jsonl
from app.evaluation.retrieval_metrics import (
    aggregate,
    dcg,
    evaluate_query,
    judged_coverage_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile_summary,
    precision_at_k,
    recall_at_k,
    success_at_k,
)
from app.evaluation.statistics import paired_bootstrap, select_configuration
from tests.conftest import CONFIG_DIR

QRELS = {"a": 3, "b": 2, "c": 0, "d": 1, "e": 2}


def test_dcg_and_ndcg_match_hand_computation() -> None:
    # gains 2^g-1: 3→7, 2→3, 1→1, 0→0 ; discounts 1/log2(i+1)
    assert dcg([3, 2, 0]) == pytest.approx(7 + 3 / math.log2(3))
    ranked = ["b", "x", "a", "c"]  # x is unjudged
    expected_dcg = 3 / 1 + 0 + 7 / math.log2(4) + 0
    ideal = 7 + 3 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
    assert ndcg_at_k(ranked, QRELS, 4) == pytest.approx(expected_dcg / ideal)
    assert ndcg_at_k(["a", "b", "e", "d", "c"], QRELS, 5) == pytest.approx(1.0)
    assert ndcg_at_k([], QRELS, 10) == 0.0
    assert ndcg_at_k(["a"], {"a": 0, "b": 0}, 10) is None  # undefined without any positive judgment
    # condensed-list evaluation removes the unjudged case before scoring
    condensed = (3 + 7 / math.log2(3)) / ideal
    assert ndcg_at_k(ranked, QRELS, 4, "condensed") == pytest.approx(condensed)
    with pytest.raises(ValueError):
        ndcg_at_k(ranked, QRELS, 4, "ignore")


def test_precision_recall_mrr_definitions() -> None:
    ranked = ["x", "b", "a", "c"]
    assert precision_at_k(ranked, QRELS, 5) == pytest.approx(2 / 5)  # denominator is always K
    assert precision_at_k(ranked, QRELS, 2) == pytest.approx(1 / 2)
    assert recall_at_k(ranked, QRELS, 5) == pytest.approx(2 / 3)  # relevant judged: a, b, e
    assert recall_at_k(ranked, QRELS, 5, threshold=3) == pytest.approx(1.0)
    assert recall_at_k(ranked, {"c": 0}, 5) is None
    assert mrr_at_k(ranked, QRELS, 10) == pytest.approx(1 / 2)
    assert mrr_at_k(["c", "x"], QRELS, 10) == 0.0
    assert success_at_k(ranked, QRELS, 1) == 0.0 and success_at_k(ranked, QRELS, 2) == 1.0
    assert judged_coverage_at_k(ranked, QRELS, 4) == pytest.approx(3 / 4)
    assert judged_coverage_at_k([], QRELS, 4) is None


def test_metrics_are_over_distinct_cases() -> None:
    assert precision_at_k(["a", "a", "a", "b"], QRELS, 4) == pytest.approx(2 / 4)
    assert ndcg_at_k(["a", "a", "b"], QRELS, 3) == ndcg_at_k(["a", "b"], QRELS, 3)


def test_evaluate_query_and_macro_aggregation() -> None:
    first = evaluate_query(["a", "b"], QRELS, (5, 10))
    second = evaluate_query([], {"z": 0}, (5, 10))
    assert first["zero_result"] == 0.0 and second["zero_result"] == 1.0
    assert second["recall@10"] is None and second["ndcg@10"] is None
    agg = aggregate([first, second])
    assert agg["n_queries"] == 2 and agg["zero_result"] == 0.5
    assert agg["recall@10"] == first["recall@10"] and agg["recall@10__n"] == 1  # undefined excluded
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["p50"] == 3.0 and summary["p95"] > summary["p90"] > 4.0 and summary["n"] == 5
    assert percentile_summary([])["p95"] is None


def test_paired_bootstrap_detects_consistent_and_spurious_gains() -> None:
    base = {f"q{i}": 0.5 for i in range(40)}
    better = {f"q{i}": 0.6 for i in range(40)}
    cmp_ = paired_bootstrap(base, better, samples=500, seed=1)
    assert cmp_.mean_difference == pytest.approx(0.1) and cmp_.ci_low == pytest.approx(0.1)
    assert cmp_.credible_improvement and (cmp_.wins, cmp_.losses, cmp_.ties) == (40, 0, 0)
    assert paired_bootstrap(base, better, samples=500, seed=1).ci_low == cmp_.ci_low  # seeded
    lucky = dict(base)
    lucky["q0"] = 1.0  # the whole gain comes from one query
    spurious = paired_bootstrap(base, lucky, samples=500, seed=1)
    assert not spurious.credible_improvement and spurious.top_decile_share == pytest.approx(1.0)
    partial = paired_bootstrap({"a": 0.1, "b": None}, {"a": 0.2, "b": 0.9, "c": 0.3})
    assert partial.n_queries == 1
    assert paired_bootstrap({}, {}).mean_difference is None


def _run(exp: str, ndcg: float, latency: float = 10.0, **gates: bool) -> dict:
    return {"experiment_id": exp, "ndcg@10": ndcg, "p95_latency_ms": latency,
            "gates": {"authorization_isolation": True, "split_isolation": True, **gates}}


def test_selection_policy_prefers_simpler_baseline_without_credible_gain() -> None:
    queries = [f"q{i}" for i in range(30)]
    per_query = {
        "base": dict.fromkeys(queries, 0.5),
        "noise": {q: 0.50 + (0.3 if i == 0 else 0.0) for i, q in enumerate(queries)},
        "good": dict.fromkeys(queries, 0.6),
        "leaky": dict.fromkeys(queries, 0.9),
        "slow": dict.fromkeys(queries, 0.7),
    }
    runs = [_run("base", 0.50), _run("noise", 0.51), _run("good", 0.60),
            _run("leaky", 0.90, split_isolation=False), _run("slow", 0.70, latency=500.0)]
    decision = select_configuration(runs, "base", per_query, samples=300)
    status = {d["experiment_id"]: d["status"] for d in decision["decisions"]}
    assert status == {"base": "baseline", "noise": "not_selected", "good": "qualified",
                      "leaky": "rejected", "slow": "not_selected"}
    assert decision["selected"] == "good"
    only_noise = select_configuration(runs[:2], "base", per_query, samples=300)
    assert only_noise["selected"] == "base" and "simpler baseline" in only_noise["rationale"]
    with pytest.raises(ValueError):
        select_configuration(runs, "missing", per_query)


def test_qrels_loading_separates_weak_labels_and_rejects_conflicts(tmp_path: Path) -> None:
    base = {"query_category": "A", "relevant_modalities": ["text"], "reviewer_id": "r1",
            "annotation_version": "v1", "confidence": 1.0, "evidence_references": [], "adjudication_status": "agreed"}
    rows = [
        {**base, "query_id": "q1", "candidate_case_id": "a", "relevance_grade": 3, "label_source": "clinician"},
        {**base, "query_id": "q1", "candidate_case_id": "b", "relevance_grade": 1, "label_source": "weak"},
        {**base, "query_id": "q1", "candidate_case_id": "c", "relevance_grade": None, "label_source": "clinician"},
    ]
    write_jsonl(tmp_path / "q.jsonl", rows)
    qrels = Qrels.load(tmp_path / "q.jsonl", ["clinician"])
    assert qrels.for_query("q1") == {"a": 3} and qrels.n_excluded == 1  # weak excluded, unjudged stays unjudged
    assert Qrels.load(tmp_path / "q.jsonl", ["weak"]).for_query("q1") == {"b": 1}
    write_jsonl(tmp_path / "bad.jsonl", [*rows, {**rows[0], "relevance_grade": 0}])
    with pytest.raises(ValueError, match="conflicting"):
        Qrels.load(tmp_path / "bad.jsonl", ["clinician"])
    write_jsonl(tmp_path / "invalid.jsonl", [{**rows[0], "relevance_grade": 7}])
    with pytest.raises(ValueError, match="invalid relevance grade"):
        Qrels.load(tmp_path / "invalid.jsonl", ["clinician"])


def test_annotation_pool_is_deduplicated_shuffled_and_blinded() -> None:
    runs = {"bm25": {"q1": ["a", "b", "c"]}, "dense": {"q1": ["b", "d"]}}
    tasks, provenance, procedure = build_annotation_pool(runs, depth=2, seed=5)
    assert sorted(t["candidate_case_id"] for t in tasks) == ["a", "b", "d"]
    assert all("systems" not in t and "bm25" not in str(t) for t in tasks)  # no system attribution
    assert {p["candidate_case_id"]: p["systems"] for p in provenance}["b"] == ["bm25", "dense"]
    assert build_annotation_pool(runs, depth=2, seed=5)[0] == tasks and procedure["n_tasks"] == 3
    merged, stats = adjudicate([
        {"query_id": "q1", "candidate_case_id": "a", "relevance_grade": 3, "reviewer_id": "r1", "confidence": 0.9},
        {"query_id": "q1", "candidate_case_id": "a", "relevance_grade": 3, "reviewer_id": "r2", "confidence": 0.7},
        {"query_id": "q1", "candidate_case_id": "b", "relevance_grade": 2, "reviewer_id": "r1", "confidence": 0.9},
        {"query_id": "q1", "candidate_case_id": "b", "relevance_grade": 0, "reviewer_id": "r2", "confidence": 0.9},
    ])
    by_case = {m["candidate_case_id"]: m for m in merged}
    assert by_case["a"]["relevance_grade"] == 3 and by_case["a"]["adjudication_status"] == "agreed"
    assert by_case["b"]["relevance_grade"] is None and by_case["b"]["adjudication_status"] == "needs_adjudication"
    assert stats["n_disagreements"] == 1 and stats["raw_agreement"] == 0.5


def test_configuration_hash_is_stable_and_sensitive() -> None:
    config = load_app_config("synthetic.yaml", CONFIG_DIR)
    assert config.config_hash() == load_app_config("synthetic.yaml", CONFIG_DIR).config_hash()
    recipe = config.recipes["hybrid_rrf_hashing"]
    changed = recipe.model_copy(update={"fusion": recipe.fusion.model_copy(update={"rrf_k": 61})})
    assert stable_hash(recipe.model_dump(mode="json")) != stable_hash(changed.model_dump(mode="json"))
    assert stable_hash(recipe.model_dump(mode="json")) == stable_hash(
        RetrievalRecipe.model_validate(recipe.model_dump(mode="json")).model_dump(mode="json"))


def test_all_shipped_configurations_validate() -> None:
    for name in ("synthetic.yaml", "baseline.yaml"):
        assert isinstance(load_app_config(name, CONFIG_DIR), AppConfig)
    suites = sorted((CONFIG_DIR / "experiments").glob("*.yaml"))
    assert len(suites) >= 6
    for path in suites:
        suite = load_experiment_suite(path)
        app_config, _ = resolve_suite(suite, CONFIG_DIR)
        assert all(run.recipe in app_config.recipes for run in suite.runs)


def test_configuration_validation_rejects_inconsistent_settings() -> None:
    with pytest.raises(ValidationError, match="dense branch requires an embedding"):
        RetrievalRecipe(recipe_id="x", branches=["dense"])
    with pytest.raises(ValidationError):
        RetrievalRecipe(recipe_id="x", branches=["bm25"], unknown_field=1)  # type: ignore[call-arg]
    raw = load_app_config("synthetic.yaml", CONFIG_DIR).model_dump(mode="json")
    raw["default_recipe"] = "does_not_exist"
    with pytest.raises(ValidationError, match="default_recipe"):
        AppConfig.model_validate(raw)
    # an external LLM API is impossible without the explicit confirmation flags
    with pytest.raises(ValidationError, match="i_confirm_data_is_authorized"):
        GenerationConfig(backend="external", model="m")
    with pytest.raises(ValidationError):
        GenerationConfig(backend="external", model="m", external_api={"enabled": True, "base_url": "https://x"})
    assert GenerationConfig().external_api.enabled is False
