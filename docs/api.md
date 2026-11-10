# HTTP API

Base path `/api/v1`. Authentication: `X-API-Key` header. Interactive OpenAPI docs: `/docs`.
Errors are structured: `{"error": {"code", "message", "details"}}`.

| Endpoint | Method | Role | Behaviour |
|---|---|---|---|
| `/health` | GET | — | process liveness |
| `/health/ready` | GET | — | database, search backend, case index, model readiness (503 if not ready) |
| `/metrics` | GET | — | Prometheus metrics |
| `/meta` | GET | researcher | supported filters, laboratory concepts, recipes — the UI shows only these |
| `/cases` | GET | researcher | authorized, paginated case listing (`limit`, `offset`, `split`) |
| `/cases/{case_id}` | GET | researcher | overview: demographics, comparable laboratory values, coverage |
| `/cases/{case_id}/timeline` | GET | researcher | events as offsets from the procedure, `before_procedure` / `follow_up` |
| `/cases/{case_id}/reports` | GET | researcher | radiology impressions |
| `/cases/{case_id}/images` | GET | researcher | imaging metadata (no storage paths) |
| `/cases/{case_id}/images/{study_id}/preview` | GET | researcher | PNG of one windowed slice when a source is available |
| `/search` | POST | researcher | structured, lexical, dense, hybrid and multimodal case retrieval |
| `/compare` | POST | researcher | side-by-side comparison of 2–8 cases |
| `/rag/answer` | POST | researcher | grounded comparison with citations, or `insufficient_evidence` |
| `/experiments` | GET | researcher | experiment runs and headline metrics |
| `/experiments/{run_id}` | GET | researcher | configuration, model revisions, gates, metrics |
| `/experiments/{run_id}/results` | GET | researcher + artifact scope | per-query rankings |
| `/admin/ingestion` | POST | admin | start an ingestion (+ index) job; `GET /admin/ingestion/{job_id}` for status |
| `/admin/indexes` | GET | admin | index status, versions, sizes, health |

## `POST /search`

```json
{
  "query_mode": "case",
  "case_id": "case_0623178b6ecb526db8d8",
  "age_min": 50, "age_max": 75,
  "gender": "F",
  "diagnosis_codes": ["ICD10CM/I50.9"],
  "lab_filters": [{"concept": "d_dimer", "min_value": 2000}],
  "candidate_splits": ["train"],
  "top_k": 10,
  "retrieval_recipe_id": "multimodal_hashing",
  "exclude_same_patient": true
}
```

`query_mode`: `case` (needs `case_id`), `text` (needs `query_text`) or `filter` (structured criteria
only; no embedding is computed). Unknown fields are rejected — in particular there is no way to send
a user identity, an access scope or a research label.

The response contains `request_id`, `query_id`, the selected recipe and its hash,
`total_eligible_cases`, ranked `results` (case id, demographics, evidence references with branch
ranks and scores, component scores with availability and rank, effective weights, laboratory
comparison, modality coverage, missing modalities, "why retrieved"), `score_descriptions`,
per-stage `latency_ms` and `warnings`. Scores are ranking signals; none is a clinical probability.

## `POST /rag/answer`

```json
{"question": "What imaging and laboratory characteristics are shared by these cases?",
 "search": {"query_mode": "case", "case_id": "case_…", "top_k": 5}}
```

or `{"question": "...", "case_ids": ["case_a", "case_b"], "query_case_id": "case_q"}`.

Response: `status`, `summary`, `similarities`, `differences` (each claim with `citations`),
`missing_evidence`, `citations` (marker, record type, authoritative record id, case id, excerpt),
`limitations`, `generation_metadata` (backend, model, tokens, latency, citation report), `warnings`.

## Limits

`top_k` ≤ 100, list `limit` ≤ 200, 2–8 cases per comparison, query text ≤ 4000 characters, request
timeout `INSPECT_RAG_REQUEST_TIMEOUT_SECONDS` (default 60 s, HTTP 504 on expiry).
