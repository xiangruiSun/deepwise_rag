"""Authorization isolation through every access path, and data-leakage probes."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.container import AppContext
from app.core.errors import AuthorizationError
from app.core.security import AccessPolicy, Principal, Role
from app.db.models import ClinicalCase, ClinicalEvent, ImagingStudy
from app.main import create_app
from app.retrieval.features import CaseFeatureService
from app.schemas.rag import CompareRequest, RagRequest
from app.schemas.search import SearchRequest
from tests.conftest import cases_index

RESEARCH = {"X-API-Key": "researcher-key"}
PRIVILEGED = {"X-API-Key": "restricted-key"}
ADMIN = {"X-API-Key": "test-admin-key"}


@pytest.fixture(scope="module")
def client(ctx: AppContext):  # type: ignore[no-untyped-def]
    with TestClient(create_app(ctx)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def restricted_ids(ctx: AppContext) -> list[str]:
    with ctx.session_factory() as session:
        ids = list(session.scalars(select(ClinicalCase.case_id).where(ClinicalCase.access_scope == "restricted")))
    assert len(ids) >= 5
    return ids


@pytest.fixture(scope="module")
def restricted_with_image(ctx: AppContext, restricted_ids: list[str]) -> tuple[str, str]:
    with ctx.session_factory() as session:
        study = session.scalars(select(ImagingStudy).where(ImagingStudy.case_id.in_(restricted_ids),
                                                           ImagingStudy.feature_status == "ready")).first()
    assert study is not None
    return study.case_id, study.study_id


# ------------------------------------------------------------------ authentication
def test_requests_without_valid_credentials_are_rejected(client: TestClient, case_query: dict[str, Any]) -> None:
    for headers in ({}, {"X-API-Key": "wrong"}):
        for method, url, body in (("get", "/api/v1/cases", None), ("post", "/api/v1/search", {"query_mode": "filter"}),
                                  ("get", f"/api/v1/cases/{case_query['case_id']}", None),
                                  ("get", "/api/v1/experiments", None), ("get", "/api/v1/admin/indexes", None)):
            response = getattr(client, method)(url, headers=headers, **({"json": body} if body else {}))
            assert response.status_code == 401 and response.json()["error"]["code"] == "unauthenticated"
    assert client.get("/api/v1/health").status_code == 200  # liveness needs no credential


def test_roles_are_enforced(client: TestClient) -> None:
    annotator = {"X-API-Key": "annotator-key"}
    assert client.get("/api/v1/cases", headers=annotator).status_code == 403
    assert client.get("/api/v1/admin/indexes", headers=RESEARCH).status_code == 403
    assert client.post("/api/v1/admin/ingestion", headers=RESEARCH, json={}).status_code == 403
    assert client.get("/api/v1/admin/indexes", headers=ADMIN).status_code == 200


# -------------------------------------------------------------- unauthorized cases
def test_case_endpoints_deny_unauthorized_cases(client: TestClient, restricted_with_image: tuple[str, str]) -> None:
    case_id, study_id = restricted_with_image
    for suffix in ("", "/timeline", "/reports", "/images", f"/images/{study_id}/preview"):
        denied = client.get(f"/api/v1/cases/{case_id}{suffix}", headers=RESEARCH)
        assert denied.status_code == 403, suffix
        body = denied.text
        assert "impression" not in body.lower() and "LOINC" not in body
        assert client.get(f"/api/v1/cases/{case_id}{suffix}", headers=PRIVILEGED).status_code == 200, suffix
    listed = client.get("/api/v1/cases?limit=200", headers=RESEARCH).json()
    assert case_id not in {c["case_id"] for c in listed["items"]} and listed["total"] == 189 - 7
    assert client.get("/api/v1/cases?limit=200", headers=PRIVILEGED).json()["total"] == 189
    assert client.get("/api/v1/cases", headers={"X-API-Key": "noscope-key"}).json()["total"] == 0


@pytest.mark.parametrize("recipe", ["bm25", "dense_hashing", "hybrid_rrf_hashing", "multimodal_hashing"])
def test_no_retrieval_path_returns_unauthorized_cases(ctx: AppContext, policy: AccessPolicy,
                                                      restricted_policy: AccessPolicy, restricted_ids: list[str],
                                                      queries: list[dict[str, Any]], recipe: str) -> None:
    restricted = set(restricted_ids)
    seen_privileged = False
    for query in [q for q in queries if q["query_mode"] == "case"][:8]:
        request = SearchRequest(query_mode="case", case_id=query["case_id"], top_k=100, retrieval_recipe_id=recipe)
        research = ctx.retrieval.search(request, policy)
        assert not restricted & {r.case_id for r in research.results}
        privileged = ctx.retrieval.search(request, restricted_policy)
        seen_privileged |= bool(restricted & {r.case_id for r in privileged.results})
        assert privileged.total_eligible_cases > research.total_eligible_cases
    assert seen_privileged  # the restricted cases ARE retrievable — but only with the right scope
    text = SearchRequest(query_mode="text", query_text="pulmonary embolism pleural effusion", top_k=100,
                         retrieval_recipe_id=recipe if recipe != "multimodal_hashing" else "bm25")
    assert not restricted & {r.case_id for r in ctx.retrieval.search(text, policy).results}
    nobody = AccessPolicy(Principal("nobody", Role.RESEARCHER, frozenset()))
    assert ctx.retrieval.search(text, nobody).results == []


def test_image_and_structured_candidate_paths_respect_scope(ctx: AppContext, policy: AccessPolicy,
                                                            restricted_ids: list[str],
                                                            queries: list[dict[str, Any]]) -> None:
    from app.core.config import MultimodalConfig, RetrievalRecipe

    restricted = set(restricted_ids)
    recipes = [
        RetrievalRecipe(recipe_id="img", branches=[], multimodal=MultimodalConfig(
            enabled=True, modalities=["image"], candidate_sources=["image"])),
        RetrievalRecipe(recipe_id="ehr", branches=[], multimodal=MultimodalConfig(
            enabled=True, modalities=["laboratory", "diagnosis"], candidate_sources=["laboratory", "diagnosis"])),
    ]
    for recipe in recipes:
        returned = 0
        for query in [q for q in queries if q["query_mode"] == "case"][:10]:
            response = ctx.retrieval.search(
                SearchRequest(query_mode="case", case_id=query["case_id"], top_k=100), policy, recipe)
            returned += len(response.results)
            assert not restricted & {r.case_id for r in response.results}
        assert returned > 0


def test_request_body_cannot_widen_scope_or_use_restricted_query_case(client: TestClient, restricted_ids: list[str]) -> None:
    forged = client.post("/api/v1/search", headers=RESEARCH, json={
        "query_mode": "filter", "access_scope": ["restricted"], "top_k": 5})
    assert forged.status_code == 422  # unknown fields are rejected: scope is never client-supplied
    as_query = client.post("/api/v1/search", headers=RESEARCH, json={"query_mode": "case", "case_id": restricted_ids[0]})
    assert as_query.status_code == 403
    labels = client.post("/api/v1/search", headers=RESEARCH, json={"query_mode": "filter", "pe_positive_nlp": "True"})
    assert labels.status_code == 422  # research labels are not ordinary query filters


def test_compare_rag_and_citations_deny_unauthorized_cases(ctx: AppContext, client: TestClient, policy: AccessPolicy,
                                                           restricted_ids: list[str], case_query: dict[str, Any]) -> None:
    payload = {"case_ids": [case_query["case_id"], restricted_ids[0]]}
    assert client.post("/api/v1/compare", headers=RESEARCH, json=payload).status_code == 403
    assert client.post("/api/v1/compare", headers=PRIVILEGED, json=payload).status_code == 200
    rag = client.post("/api/v1/rag/answer", headers=RESEARCH, json={"question": "Compare these cases.", **payload})
    assert rag.status_code == 403 and "impression" not in rag.text.lower()
    with pytest.raises(AuthorizationError):
        ctx.comparison.compare(CompareRequest(case_ids=payload["case_ids"]), policy)
    with pytest.raises(AuthorizationError):
        ctx.rag.answer(RagRequest(question="Compare these cases.", query_case_id=restricted_ids[0],
                                  case_ids=[case_query["case_id"]]), policy)
    ok = client.post("/api/v1/rag/answer", headers=RESEARCH, json={
        "question": "How do the documented impressions compare?",
        "search": {"query_mode": "case", "case_id": case_query["case_id"], "top_k": 3}})
    assert ok.status_code == 200
    cited_cases = {c["case_id"] for c in ok.json()["citations"]}
    assert cited_cases and not cited_cases & set(restricted_ids)


def test_experiment_artifacts_are_scope_restricted(ctx: AppContext, client: TestClient, tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
    from app.core.config import load_experiment_suite
    from app.evaluation.runner import ExperimentRunner
    from tests.conftest import CONFIG_DIR

    suite = load_experiment_suite("experiments/smoke.yaml", CONFIG_DIR).model_copy(update={"evaluation": ctx.config.evaluation})
    summary = ExperimentRunner(ctx, tmp_path_factory.mktemp("exp")).run_suite(suite, only=["S01"])
    run_id = summary["runs"][0]["run_id"]
    listing = client.get("/api/v1/experiments", headers=RESEARCH)
    assert listing.status_code == 200 and run_id in {r["run_id"] for r in listing.json()}
    detail = client.get(f"/api/v1/experiments/{run_id}", headers=RESEARCH).json()
    assert detail["metrics_by_group"]["overall"]["ndcg@10"] > 0 and detail["summary"]["gates"]["split_isolation"]
    results = client.get(f"/api/v1/experiments/{run_id}/results?limit=3", headers=RESEARCH)
    assert results.status_code == 200 and len(results.json()["rows"]) == 3
    # per-query artifacts list case identifiers: a principal without the scope cannot download them
    assert client.get(f"/api/v1/experiments/{run_id}/results", headers={"X-API-Key": "noscope-key"}).status_code == 403
    assert client.get("/api/v1/experiments/unknown", headers=RESEARCH).status_code == 404


# --------------------------------------------------------------------- leakage
def test_future_records_never_enter_query_features(ctx: AppContext, queries: list[dict[str, Any]],
                                                   truth: dict[str, Any]) -> None:
    case_ids = [q["case_id"] for q in queries if q["query_mode"] == "case"]
    with ctx.session_factory() as session:
        from sqlalchemy import func

        single_study_patients = set(session.scalars(
            select(ClinicalCase.patient_id).group_by(ClinicalCase.patient_id).having(func.count() == 1)))
        features = CaseFeatureService(session)
        for lookback in (7, 30, 90):
            for aggregate in ("last", "mean"):
                contexts = features.contexts(case_ids, lookback, aggregate)
                for ctx_case in contexts.values():
                    cutoff = ctx_case.case.procedure_time
                    assert all(f.event_time <= cutoff for f in ctx_case.labs.features.values())
                    assert all(f.event_time >= cutoff - timedelta(days=lookback) for f in ctx_case.labs.features.values())
                    assert all(e.event_time <= cutoff for e in ctx_case.diagnosis_events)
                    # the planted post-procedure D-dimer of 99999 must never be a feature
                    assert all(f.value < 90000 for f in ctx_case.labs.features.values())
                    # a PE code recorded after THIS procedure is not history.  (For a patient's
                    # second study, the code recorded after the first study legitimately is.)
                    if ctx_case.case.patient_id in single_study_patients:
                        assert not any(c.startswith(("ICD10CM/I26", "ICD10CM/I27.24"))
                                       for c in ctx_case.diagnosis_codes)
        # the probes really exist in the data — they are excluded, not absent
        acute = [c for c in case_ids if truth[c]["pe"] in ("central", "lobar", "segmental", "subsegmental")]
        sample = session.get(ClinicalCase, acute[0])
        future = session.scalars(select(ClinicalEvent).where(
            ClinicalEvent.patient_id == sample.patient_id, ClinicalEvent.event_time > sample.procedure_time)).all()
        assert any(e.concept_code.startswith("ICD10CM/I26") for e in future)
        planted = session.scalars(select(ClinicalEvent).where(ClinicalEvent.numeric_value == 99999.0)).all()
        assert planted
        # an as-of cutoff before the procedure hides even pre-procedure same-day labs
        early = features.contexts([acute[0]], 30, as_of=sample.procedure_time - timedelta(days=1))[acute[0]]
        late = features.contexts([acute[0]], 30)[acute[0]]
        assert set(early.labs.features) <= set(late.labs.features)
        assert all(f.event_time <= sample.procedure_time - timedelta(days=1) for f in early.labs.features.values())


def test_indexed_filter_metadata_contains_no_future_information(ctx: AppContext) -> None:
    hits, total = ctx.backend.filter_search(cases_index(ctx), None, 1000)
    assert total == 189
    with ctx.session_factory() as session:
        cases = {c.case_id: c for c in session.scalars(select(ClinicalCase))}
        events = list(session.scalars(select(ClinicalEvent).where(ClinicalEvent.event_type == "diagnosis")))
    for hit in hits:
        case = cases[hit.id]
        expected = sorted({e.concept_code for e in events
                           if e.patient_id == case.patient_id and e.event_time <= case.procedure_time})
        assert hit.source["diagnosis_codes"] == expected  # exactly the codes known at procedure time
        assert hit.source.get("lab_d_dimer", 0) < 90000
        assert not any(key.startswith(("pe_positive", "label", "1_month", "12_month")) for key in hit.source)


def test_same_patient_studies_are_excluded_from_results(ctx: AppContext, restricted_policy: AccessPolicy) -> None:
    with ctx.session_factory() as session:
        cases = list(session.scalars(select(ClinicalCase).order_by(ClinicalCase.patient_id, ClinicalCase.procedure_time)))
    by_patient: dict[str, list[ClinicalCase]] = {}
    for case in cases:
        by_patient.setdefault(case.patient_id, []).append(case)
    multi = [group for group in by_patient.values() if len(group) > 1]
    assert len(multi) >= 10
    checked = 0
    for group in multi[:8]:
        query, sibling = group[0], group[1]
        for recipe in ("bm25", "dense_hashing", "hybrid_rrf_hashing", "multimodal_hashing"):
            request = SearchRequest(query_mode="case", case_id=query.case_id, top_k=100, retrieval_recipe_id=recipe)
            excluded = ctx.retrieval.search(request, restricted_policy)
            ids = {r.case_id for r in excluded.results}
            assert query.case_id not in ids and sibling.case_id not in ids
            checked += 1
        allowed = ctx.retrieval.search(SearchRequest(query_mode="case", case_id=query.case_id, top_k=100,
                                                     retrieval_recipe_id="bm25", exclude_same_patient=False),
                                       restricted_policy)
        assert query.case_id not in {r.case_id for r in allowed.results}  # the query case itself never returns
    assert checked == 32


def test_evaluation_splits_are_patient_disjoint(ctx: AppContext, queries: list[dict[str, Any]]) -> None:
    with ctx.session_factory() as session:
        cases = {c.case_id: c for c in session.scalars(select(ClinicalCase))}
    corpus_patients = {c.patient_id for c in cases.values() if c.dataset_split == "train"}
    for query in queries:
        if query["query_mode"] != "case":
            continue
        case = cases[query["case_id"]]
        assert case.dataset_split == ("valid" if query["stage"] == "dev" else "test")
        assert case.patient_id not in corpus_patients
