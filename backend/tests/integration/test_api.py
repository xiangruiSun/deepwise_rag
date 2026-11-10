"""HTTP API contract tests against the synthetic cohort."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.container import AppContext
from app.main import create_app
from tests.conftest import report_index

H = {"X-API-Key": "researcher-key"}
ADMIN = {"X-API-Key": "test-admin-key"}


@pytest.fixture(scope="module")
def client(ctx: AppContext):  # type: ignore[no-untyped-def]
    with TestClient(create_app(ctx)) as test_client:
        yield test_client


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.json()["checks"]["database"] == "ok" and ready.json()["checks"]["case_index"] == "ok"
    assert "inspect_rag_requests_total" in client.get("/api/v1/metrics").text
    assert ready.headers["X-Research-Use-Only"] == "true"


def test_meta_exposes_only_supported_filters(client: TestClient) -> None:
    meta = client.get("/api/v1/meta", headers=H).json()
    assert meta["data_kind"] == "synthetic" and meta["genders"] == ["F", "M"]
    assert {c["key"] for c in meta["laboratory_concepts"]} >= {"d_dimer", "troponin_i", "creatinine"}
    assert "ICD10CM/I50.9" in meta["diagnosis_codes"] and meta["study_modalities"] == ["CT"]
    assert {r["recipe_id"] for r in meta["recipes"]} >= {"bm25", "multimodal_hashing"}
    assert "geograph" not in str(meta).lower() and "zip" not in str(meta).lower()
    assert "Not for diagnosis" in meta["notice"]


def test_case_endpoints(client: TestClient, case_query: dict[str, Any]) -> None:
    case_id = case_query["case_id"]
    page = client.get("/api/v1/cases?limit=5&offset=5&split=valid", headers=H).json()
    assert page["limit"] == 5 and page["offset"] == 5 and len(page["items"]) == 5
    assert all(item["dataset_split"] == "valid" for item in page["items"])
    overview = client.get(f"/api/v1/cases/{case_id}", headers=H).json()
    assert overview["case_id"] == case_id and overview["report_ids"] and "patient_id" not in overview
    timeline = client.get(f"/api/v1/cases/{case_id}/timeline?limit=200", headers=H).json()
    assert timeline["total"] == overview["n_events_before_procedure"] + overview["n_events_after_procedure"]
    phases = {e["phase"] for e in timeline["events"]}
    assert phases <= {"before_procedure", "follow_up"}
    assert all((e["hours_from_procedure"] <= 0) == (e["phase"] == "before_procedure") for e in timeline["events"])
    labs = client.get(f"/api/v1/cases/{case_id}/timeline?event_type=laboratory", headers=H).json()
    assert labs["events"] and all(e["event_type"] == "laboratory" for e in labs["events"])
    reports = client.get(f"/api/v1/cases/{case_id}/reports", headers=H).json()
    assert reports[0]["report_id"] == overview["report_ids"][0] and reports[0]["impression_text"]
    images = client.get(f"/api/v1/cases/{case_id}/images", headers=H).json()
    assert all("source_path" not in image and "path" not in image for image in images)  # no storage paths
    for image in images:
        if image["preview_available"]:
            preview = client.get(f"/api/v1/cases/{case_id}/images/{image['study_id']}/preview", headers=H)
            assert preview.status_code == 200 and preview.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/api/v1/cases/case_unknown", headers=H).status_code == 404
    assert client.get("/api/v1/cases?limit=5000", headers=H).status_code == 422
    assert client.get("/api/v1/cases?split=everything", headers=H).status_code == 422


def test_search_endpoint_contract(client: TestClient, case_query: dict[str, Any]) -> None:
    response = client.post("/api/v1/search", headers=H, json={
        "query_mode": "case", "case_id": case_query["case_id"], "top_k": 5,
        "retrieval_recipe_id": "multimodal_hashing", "candidate_splits": ["train"]})
    assert response.status_code == 200
    body = response.json()
    for key in ("request_id", "query_id", "recipe_id", "recipe_hash", "recipe_summary", "total_eligible_cases",
                "results", "score_descriptions", "modality_coverage", "latency_ms", "warnings", "notice"):
        assert key in body
    result = body["results"][0]
    for key in ("rank", "case_id", "evidence", "components", "modality_coverage", "missing_modalities",
                "why_retrieved", "score_type", "effective_weights"):
        assert key in result
    assert "patient_id" not in result  # only de-identified case identifiers are exposed
    assert all("probab" not in c["score_type"] for c in result["components"])
    assert "not a probability" in body["score_descriptions"]["laboratory"].lower()
    assert {"bm25", "vector_search", "fusion", "aggregation", "multimodal_fusion", "total"} <= set(body["latency_ms"])
    text = client.post("/api/v1/search", headers=H, json={"query_mode": "text", "query_text": "saddle embolus"})
    assert text.status_code == 200 and text.json()["recipe_id"] == "bm25"


def test_structured_errors(client: TestClient) -> None:
    bad = client.post("/api/v1/search", headers=H, json={"query_mode": "text"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    unknown = client.post("/api/v1/search", headers=H, json={"query_mode": "filter", "retrieval_recipe_id": "nope"})
    assert unknown.status_code == 422 and "available" in unknown.json()["error"]["details"]
    too_many = client.post("/api/v1/search", headers=H, json={"query_mode": "filter", "top_k": 101})
    assert too_many.status_code == 422
    compare = client.post("/api/v1/compare", headers=H, json={"case_ids": ["only_one"]})
    assert compare.status_code == 422
    rag = client.post("/api/v1/rag/answer", headers=H, json={"question": "Compare"})
    assert rag.status_code == 422


def test_compare_and_rag_endpoints(client: TestClient, queries: list[dict[str, Any]]) -> None:
    ids = [q["case_id"] for q in queries if q["query_mode"] == "case"][:2]
    compare = client.post("/api/v1/compare", headers=H, json={"case_ids": ids})
    assert compare.status_code == 200 and [c["case_id"] for c in compare.json()["cases"]] == ids
    answer = client.post("/api/v1/rag/answer", headers=H, json={
        "question": "What imaging and laboratory characteristics are shared by these cases?",
        "search": {"query_mode": "case", "case_id": ids[0], "top_k": 3, "candidate_splits": ["train"]}})
    assert answer.status_code == 200
    body = answer.json()
    assert body["status"] == "answered" and body["citations"] and body["generation_metadata"]["backend"] == "extractive"
    known = {c["source_id"] for c in body["citations"]}
    assert all(set(claim["citations"]) <= known for section in ("summary", "similarities", "differences")
               for claim in body[section])
    refused = client.post("/api/v1/rag/answer", headers=H, json={
        "question": "Which dose should be given?", "case_ids": ids})
    assert refused.json()["status"] == "insufficient_evidence" and refused.json()["citations"] == []


def test_admin_indexes_and_audit_trail(ctx: AppContext, client: TestClient) -> None:
    indexes = {i["index_name"]: i for i in client.get("/api/v1/admin/indexes", headers=ADMIN).json()}
    hashing = indexes[report_index(ctx, "hashing")]
    assert hashing["exists"] and hashing["dimension"] == 384 and hashing["health"] == "green"
    assert hashing["embedding_model_id"] == "hashing-ngram-v1" and hashing["document_count"] == 188
    assert client.get("/api/v1/admin/ingestion/unknown", headers=ADMIN).status_code == 404
    client.post("/api/v1/search", headers=H, json={"query_mode": "text", "query_text": "unique audit probe zebra"})
    from sqlalchemy import select

    from app.db.models import AuditEvent

    with ctx.session_factory() as session:
        events = list(session.scalars(select(AuditEvent).where(AuditEvent.action == "search")))
    assert events and events[-1].actor == "rita"
    assert all("zebra" not in str(e.details) for e in events)  # query text is never written to the audit log
