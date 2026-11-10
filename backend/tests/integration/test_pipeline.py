"""End-to-end on the synthetic cohort with the local search backend and SQLite:
ingest → index → BM25 / dense / hybrid / multimodal search → filters → compare → RAG → evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from app.core.config import ExperimentRunSpec, ExperimentSuite, load_experiment_suite
from app.core.container import AppContext
from app.core.errors import ConfigurationError, NotFoundError, ValidationFailure
from app.core.security import AccessPolicy
from app.db.models import ClinicalCase, ExperimentRun, IndexManifest, LabNormalizationStat
from app.evaluation.datasets import Qrels, read_jsonl
from app.evaluation.report import ReportBuilder, run_selection
from app.evaluation.retrieval_metrics import evaluate_query
from app.evaluation.runner import ExperimentRunner
from app.schemas.rag import CompareRequest
from app.schemas.search import LabFilter, SearchRequest
from tests.conftest import CONFIG_DIR, cases_index, image_index, make_config, report_index

RECIPES = ("bm25", "dense_hashing", "hybrid_rrf_hashing", "multimodal_hashing")


def search(ctx: AppContext, policy: AccessPolicy, **kw: Any):  # type: ignore[no-untyped-def]
    return ctx.retrieval.search(SearchRequest(**kw), policy)


def test_cohort_is_retrievable_by_case_id_and_indexes_exist(ctx: AppContext, case_query: dict[str, Any]) -> None:
    with ctx.session_factory() as session:
        case = session.get(ClinicalCase, case_query["case_id"])
        assert case is not None and case.dataset_split == "valid"
        manifests = {m.index_name: m for m in session.scalars(select(IndexManifest))}
        stats = {s.stats_version for s in session.scalars(select(LabNormalizationStat))}
        assert all(s.fitted_split == "train" for s in session.scalars(select(LabNormalizationStat)))
    hashing = report_index(ctx, "hashing")
    assert {cases_index(ctx), report_index(ctx, None), hashing, image_index(ctx)} <= set(manifests)
    assert manifests[hashing].dimension == 384
    assert manifests[hashing].document_count == 188
    assert manifests[cases_index(ctx)].document_count == 189
    assert {"train-lb7-last", "train-lb30-last", "train-lb90-mean"} <= stats


@pytest.mark.parametrize("recipe", RECIPES)
def test_case_search_returns_distinct_cases_with_valid_sources(ctx: AppContext, policy: AccessPolicy,
                                                               case_query: dict[str, Any], recipe: str) -> None:
    response = search(ctx, policy, query_mode="case", case_id=case_query["case_id"], top_k=10,
                      retrieval_recipe_id=recipe, candidate_splits=["train"])
    ids = [r.case_id for r in response.results]
    assert len(ids) == 10 and len(set(ids)) == 10 and case_query["case_id"] not in ids
    assert [r.rank for r in response.results] == list(range(1, 11))
    scores = [r.final_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert response.recipe_id == recipe and response.total_eligible_cases > 100
    assert response.latency_ms["total"] > 0 and response.score_descriptions
    with ctx.session_factory() as session:
        from app.db.models import RadiologyReport

        for result in response.results:
            assert result.dataset_split == "train" and result.why_retrieved
            for evidence in result.evidence:  # every reference resolves to the right case's record
                report = session.get(RadiologyReport, evidence.source_id)
                assert report is not None and report.case_id == result.case_id
                assert evidence.excerpt and evidence.excerpt in report.impression_text
    if recipe == "multimodal_hashing":
        first = response.results[0]
        assert {c.name for c in first.components} == {"text", "laboratory", "image", "diagnosis"}
        assert pytest.approx(sum(first.effective_weights.values()), abs=1e-3) == 1.0
        assert set(response.modality_coverage) >= {"text", "laboratory", "image"}
        assert response.results[0].score_type == "multimodal_rank"
    else:
        assert response.results[0].score_type in {"bm25", "dense", "rrf"}


def test_text_search_finds_the_right_findings(ctx: AppContext, policy: AccessPolicy, truth: dict[str, Any]) -> None:
    response = search(ctx, policy, query_mode="text", query_text="webs and bands chronic thromboembolic disease",
                      top_k=5, retrieval_recipe_id="bm25")
    # these terms only occur in chronic-thromboembolic impressions: lexical matches must rank first
    assert response.results and all(truth[r.case_id]["pe"] == "chronic" for r in response.results)
    assert "webs and bands" in response.results[0].evidence[0].excerpt.lower()
    negated = search(ctx, policy, query_mode="text", query_text="no evidence of pulmonary embolism", top_k=5,
                     retrieval_recipe_id="hybrid_rrf_hashing")
    assert all(truth[r.case_id]["pe"] == "none" for r in negated.results)


def test_structured_filters_are_applied_before_retrieval(ctx: AppContext, policy: AccessPolicy) -> None:
    unfiltered = search(ctx, policy, query_mode="text", query_text="pulmonary embolism", top_k=50)
    filtered = search(ctx, policy, query_mode="text", query_text="pulmonary embolism", top_k=50, age_min=50,
                      age_max=75, gender="F", lab_filters=[LabFilter(concept="d_dimer", min_value=2000.0)])
    assert 0 < filtered.total_eligible_cases < unfiltered.total_eligible_cases
    assert filtered.results and len(filtered.results) <= filtered.total_eligible_cases
    for result in filtered.results:
        assert 50 <= result.age <= 75 and result.gender == "F"
        d_dimer = next(v for v in result.laboratory_values if v["concept_key"] == "d_dimer")
        assert d_dimer["value"] >= 2000.0
    # pre-filtering: a top-1 request still returns the best ELIGIBLE case for every retriever
    for recipe in ("bm25", "dense_hashing"):
        top1 = search(ctx, policy, query_mode="text", query_text="no evidence of pulmonary embolism", top_k=1,
                      retrieval_recipe_id=recipe, age_min=85)
        assert len(top1.results) == 1 and top1.results[0].age >= 85
    dx = search(ctx, policy, query_mode="filter", diagnosis_codes=["icd10cm/i50.9", "ICD10CM/E11.9"], top_k=100)
    assert dx.total_eligible_cases == len(dx.results) > 0
    with ctx.session_factory() as session:
        from app.retrieval.features import CaseFeatureService

        contexts = CaseFeatureService(session).contexts([r.case_id for r in dx.results], 30)
    assert all({"ICD10CM/I50.9", "ICD10CM/E11.9"} <= c.diagnosis_codes for c in contexts.values())


def test_filter_only_mode_needs_no_embedding_and_paginates(ctx: AppContext, policy: AccessPolicy) -> None:
    before = dict(ctx.embeddings.provider("hashing").truncation)
    first = search(ctx, policy, query_mode="filter", gender="M", top_k=5, retrieval_recipe_id="dense_hashing")
    second = search(ctx, policy, query_mode="filter", gender="M", top_k=5, offset=5, retrieval_recipe_id="dense_hashing")
    assert ctx.embeddings.provider("hashing").truncation == before  # no query was embedded
    assert len(first.results) == 5 and not {r.case_id for r in first.results} & {r.case_id for r in second.results}
    assert all(r.gender == "M" and r.score_type == "none" and r.final_score is None for r in first.results)
    assert "query_embedding" not in first.latency_ms and second.results[0].rank == 6


def test_empty_population_and_invalid_requests(ctx: AppContext, policy: AccessPolicy) -> None:
    empty = search(ctx, policy, query_mode="text", query_text="pulmonary embolism", age_min=120, age_max=130)
    assert empty.results == [] and empty.total_eligible_cases == 0 and "No authorized cases" in empty.warnings[0]
    with pytest.raises(ValidationFailure, match="unknown retrieval_recipe_id"):
        search(ctx, policy, query_mode="text", query_text="x", retrieval_recipe_id="nope")
    with pytest.raises(ValidationFailure, match="unknown laboratory concept"):
        search(ctx, policy, query_mode="filter", lab_filters=[LabFilter(concept="unobtainium", min_value=1)])
    with pytest.raises(NotFoundError):
        search(ctx, policy, query_mode="case", case_id="case_missing")
    for bad in ({"query_mode": "case"}, {"query_mode": "text", "query_text": "  "},
                {"query_mode": "filter", "age_min": 80, "age_max": 20}, {"query_mode": "filter", "top_k": 1000},
                {"query_mode": "filter", "user_scopes": ["restricted"]}):
        with pytest.raises(ValueError):
            SearchRequest(**bad)


def test_missing_modalities_are_reported_not_imputed(ctx: AppContext, policy: AccessPolicy,
                                                     queries: list[dict[str, Any]]) -> None:
    seen_missing = False
    for query in [q for q in queries if q["query_mode"] == "case"][:12]:
        response = search(ctx, policy, query_mode="case", case_id=query["case_id"], top_k=20,
                          retrieval_recipe_id="multimodal_hashing", candidate_splits=["train"])
        for result in response.results:
            by_name = {c.name: c for c in result.components}
            for modality in result.missing_modalities:
                seen_missing = True
                assert not by_name[modality].available and by_name[modality].value is None
                assert result.effective_weights[modality] == 0.0
            available = [m for m in ("laboratory", "image", "diagnosis") if by_name[m].available]
            assert all(by_name[m].value is not None for m in available)
            if not by_name["laboratory"].available and result.laboratory_note:
                assert "unavailable" in result.laboratory_note or "comparable" in result.laboratory_note
    assert seen_missing  # the synthetic cohort does contain cases with missing labs / imaging


def test_compare_separates_follow_up_from_query_time_features(ctx: AppContext, policy: AccessPolicy,
                                                              queries: list[dict[str, Any]], truth: dict[str, Any]) -> None:
    acute = [q["case_id"] for q in queries if q["query_mode"] == "case"
             and truth[q["case_id"]]["pe"] in ("central", "lobar", "segmental", "subsegmental")][:2]
    result = ctx.comparison.compare(CompareRequest(case_ids=acute), policy)
    assert [c.case_id for c in result.cases] == acute
    for column in result.cases:
        assert column.impressions and column.follow_up
        assert all(d["hours_from_procedure"] <= 0 for d in column.diagnoses)
        assert all(f["hours_after_procedure"] > 0 for f in column.follow_up)
        # the post-procedure PE diagnosis appears only as follow-up
        assert any(f["concept_code"].startswith("ICD10CM/I26") for f in column.follow_up)
        assert not any(d["concept_code"].startswith("ICD10CM/I26") for d in column.diagnoses)
        assert all(lab["unit"] and lab["original_unit"] for lab in column.laboratory)
    with pytest.raises(NotFoundError):
        ctx.comparison.compare(CompareRequest(case_ids=[acute[0], "case_missing"]), policy)


def test_parent_child_retrieval_returns_full_parent_impression(synthetic_root: Path, tmp_path: Path) -> None:
    from app.core.config import RetrievalRecipe
    from tests.conftest import make_context

    config = make_config(synthetic_root)
    recipes = dict(config.recipes)
    recipes["pc"] = RetrievalRecipe(recipe_id="pc", branches=["bm25"], chunking="parent_child")
    ctx = make_context(synthetic_root, tmp_path, config.model_copy(update={"recipes": recipes}))
    from app.core.security import Principal, Role

    policy = AccessPolicy(Principal("t", Role.RESEARCHER, frozenset({"research"})))
    response = search(ctx, policy, query_mode="text", query_text="right heart strain septum flattening", top_k=5,
                      retrieval_recipe_id="pc")
    assert ctx.backend.count(report_index(ctx, None, "parent_child")) > 188  # impressions were split
    assert len({r.case_id for r in response.results}) == len(response.results) == 5
    evidence = response.results[0].evidence[0]
    assert evidence.matched_text and evidence.matched_text in evidence.excerpt
    assert len(evidence.excerpt) > len(evidence.matched_text)


@pytest.fixture(scope="module")
def suite_summary(ctx: AppContext, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    out = tmp_path_factory.mktemp("experiments")
    base = load_experiment_suite("experiments/smoke.yaml", CONFIG_DIR)
    suite = base.model_copy(update={"evaluation": ctx.config.evaluation})
    return ExperimentRunner(ctx, out).run_suite(suite), out


def test_evaluation_run_persists_metrics_and_passes_gates(ctx: AppContext, suite_summary: tuple[dict[str, Any], Path]) -> None:
    summary, _out = suite_summary
    assert summary["data_kind"] == "synthetic" and summary["weak_or_other_label_rows_excluded"] == 0
    assert [r["status"] for r in summary["runs"]] == ["completed"] * 6
    board = summary["leaderboard"]
    assert [row["ndcg@10"] for row in board] == sorted((row["ndcg@10"] for row in board), reverse=True)
    for run in summary["runs"]:
        s = run["summary"]
        assert all(s["gates"].values()), s["gates"]
        assert 0 < s["metrics"]["ndcg@10"] <= 1 and 0 < s["metrics"]["judged_coverage@10"] <= 1
        assert s["n_queries_without_judgments"] == 1 and s["abstention_queries_with_zero_results"] == 1.0
        assert s["latency_ms"]["total"]["p95"] >= s["latency_ms"]["total"]["p50"] > 0
        run_dir = Path(run["run_dir"])
        assert {p.name for p in run_dir.iterdir()} >= {"config.json", "metrics.json", "per_query.jsonl"}
        saved = json.loads((run_dir / "config.json").read_text())
        for key in ("dataset_fingerprint", "query_set_version", "qrels_version", "model_revisions", "index_versions"):
            assert key in saved
    generation = next(r for r in summary["runs"] if r["experiment_id"] == "S03")["summary"]["generation"]
    assert generation["abstention_correctness"] == 1.0 and generation["citation_correctness"] == 1.0
    assert generation["faithfulness"] == 1.0 and generation["invalid_citations"] == 0
    with ctx.session_factory() as session:
        runs = {r.experiment_id: r for r in session.scalars(select(ExperimentRun))}
        assert set(runs) >= {"S01", "S02", "S03", "S04", "S05", "S06"}
        s01 = runs["S01"]
        assert s01.status == "completed" and s01.config_hash and s01.hardware and s01.seed == 13
        names = {m.name for m in s01.metrics}
        assert {"ndcg@10", "recall@10", "precision@10", "p95_latency_ms", "gate:split_isolation"} <= names
        assert {a.kind for a in s01.artifacts} >= {"config", "metrics", "per_query"}


def test_reported_metrics_match_independent_computation(ctx: AppContext, suite_summary: tuple[dict[str, Any], Path]) -> None:
    summary, _ = suite_summary
    qrels = Qrels.load(ctx.config.evaluation.qrels, ["synthetic_rubric"])
    run = next(r for r in summary["runs"] if r["experiment_id"] == "S01")
    rows = [r for r in read_jsonl(Path(run["run_dir"]) / "per_query.jsonl") if r["has_judgments"]]
    values = {"ndcg@10": [], "recall@10": [], "precision@10": []}
    for row in rows:
        metrics = evaluate_query(row["ranked_case_ids"], qrels.for_query(row["query_id"]))
        assert metrics == pytest.approx(row["metrics"], nan_ok=True)
        for name in values:
            if metrics[name] is not None:
                values[name].append(metrics[name])
    for name, vals in values.items():
        assert run["summary"]["metrics"][name] == pytest.approx(sum(vals) / len(vals))
    # corpus/query split isolation as recorded in the raw per-query output
    with ctx.session_factory() as session:
        split = {c.case_id: (c.dataset_split, c.patient_id) for c in session.scalars(select(ClinicalCase))}
    assert all(split[c][0] == "train" for row in rows for c in row["ranked_case_ids"])


def test_runs_are_reproducible_resumable_and_never_mix_configurations(ctx: AppContext,
                                                                      suite_summary: tuple[dict[str, Any], Path]) -> None:
    summary, out = suite_summary
    base = load_experiment_suite("experiments/smoke.yaml", CONFIG_DIR).model_copy(update={"evaluation": ctx.config.evaluation})
    runner = ExperimentRunner(ctx, out)
    again = runner.run_suite(base, only=["S01"])
    first = next(r for r in summary["runs"] if r["experiment_id"] == "S01")
    assert again["runs"][0]["run_id"] == first["run_id"]  # same config hash → same run, reused
    forced = runner.run_suite(base, only=["S01"], force=True)
    assert forced["runs"][0]["summary"]["metrics"] == first["summary"]["metrics"]  # deterministic
    changed_eval = ctx.config.evaluation.model_copy(update={"top_k": 15})
    other = runner.run_suite(base.model_copy(update={"evaluation": changed_eval}), only=["S01"])
    assert other["runs"][0]["config_hash"] != first["config_hash"]
    assert other["runs"][0]["run_dir"] != first["run_dir"]  # a different configuration never shares a directory


def test_held_out_test_set_is_guarded(ctx: AppContext, tmp_path: Path) -> None:
    suite = ExperimentSuite(suite_id="final_test", base_config="synthetic.yaml", stage="test",
                            runs=[ExperimentRunSpec(experiment_id="F01", recipe="bm25")],
                            evaluation=ctx.config.evaluation)
    runner = ExperimentRunner(ctx, tmp_path)
    with pytest.raises(ConfigurationError, match="held-out TEST"):
        runner.run_suite(suite)
    done = runner.run_suite(suite, allow_test=True)
    assert done["runs"][0]["status"] == "completed" and done["stage"] == "test"
    rows = read_jsonl(Path(done["runs"][0]["run_dir"]) / "per_query.jsonl")
    assert all(r["query_id"].startswith("test_") for r in rows)
    retuned = suite.model_copy(update={"runs": [ExperimentRunSpec(experiment_id="F02", recipe="dense_hashing")]})
    with pytest.raises(ConfigurationError, match="already evaluated with a different configuration"):
        runner.run_suite(retuned, allow_test=True)


def test_selection_and_report_use_only_completed_results(ctx: AppContext, suite_summary: tuple[dict[str, Any], Path],
                                                         tmp_path: Path) -> None:
    summary, out = suite_summary
    decision = run_selection(out, "smoke", samples=200)
    assert decision["baseline"] == "S01" and decision["selected"] in {r["experiment_id"] for r in summary["runs"]}
    assert all(d["status"] in {"baseline", "qualified", "not_selected", "rejected"} for d in decision["decisions"])
    built = ReportBuilder(out, ctx.settings.artifact_dir, "synthetic").build(
        tmp_path / "report", ctx.config.evaluation.qrels, ["synthetic_rubric"], 200)
    text = Path(built["markdown"]).read_text()
    assert "SYNTHETIC DATA" in text and "not** evidence of clinical" in text
    for heading in ("## 1. Dataset version", "## 5. Embedding comparison", "## 9. Multimodal ablation",
                    "## 12. Statistical uncertainty", "## 13. Error analysis", "## 14. Selected final architecture",
                    "## 15. Limitations"):
        assert heading in text
    # suites that were never run are reported as not evaluated — no numbers are invented
    section = text.split("## 5. Embedding comparison")[1].split("## 6.")[0]
    assert "Not yet evaluated" in section and "0." not in section
    data = json.loads(Path(built["json"]).read_text())
    assert data["data_kind"] == "synthetic" and data["selection"]["selected"] == decision["selected"]
    for chart in built["charts"]:
        assert (tmp_path / "report" / chart).read_text().startswith("<svg")
    s01 = next(r for r in summary["runs"] if r["experiment_id"] == "S01")["summary"]["metrics"]["ndcg@10"]
    assert f"{s01:.4f}" in text
