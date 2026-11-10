# Corpus platform — API contracts (`/v1`)

Authentication: `X-API-Key`. The key determines user, role, groups and **tenant**. Roles:
`annotator` < `researcher` < `admin`. Errors use one envelope:

```json
{"error": {"code": "validation_error", "message": "…", "details": {}}}
```

| Status | Code | When |
|---|---|---|
| 401 | `unauthenticated` | missing or unknown key |
| 403 | `forbidden` | role too low |
| 404 | `not_found` | unknown strategy, job or run — also for another tenant's job |
| 422 | `validation_error` | invalid body, unknown field (e.g. `tenant_id`), unsupported filter |
| 429 | `overloaded` | generation queue full |
| 503 | `dependency_unavailable` | search backend or model server unavailable |
| 504 | `timeout` | request exceeded the server timeout |

## `POST /v1/query`

```json
{"query": "What is the latest follow-up status for patient PSN-G000007?",
 "strategy": "hybrid_rerank", "document_types": ["clinical_report"],
 "filters": {"x_department": ["cardiology"]}, "time_from": null, "time_to": null,
 "as_of": null, "top_k": 8, "response_length": "medium"}
```

Only `query` is required. `filters` accepts `document_type`, `source_system`, `x_department`,
`x_patient_pseudo_id`, `x_encounter_id`; anything else is rejected.

Response (abridged):

```json
{"answer": "… [S1]", "sentences": [{"text": "…", "citations": ["S1"], "supported": true}],
 "citations": [{"source_id": "S1", "document_id": "doc_…", "document_version": 2, "chunk_id": "…",
                "title": "…", "event_time": "…", "excerpt": "…"}],
 "abstained": false, "abstention_reason": null,
 "model_version": "extractive:extractive-v1", "index_version": "corpus_chunks_v…",
 "strategy": "hybrid_rerank", "query_kind": "temporal", "trace_id": "…",
 "total_ms": 12.3, "first_token_ms": 9.8, "timings_ms": {"bm25_retrieval": 1.2, "…": 0},
 "usage": {"prompt_tokens": 512, "completion_tokens": 40, "context_tokens": 480},
 "context": {"chunks": 6, "tokens": 480, "dropped": {"duplicate": 1}, "quarantined_sentences": 0},
 "conflicts": [], "removed_sentences": 0, "invalid_citations": [],
 "mock": false, "degraded": false, "cache_hit": false, "warnings": [], "notice": "…"}
```

`mock: true` means no language model is configured and the text is canned demo output.
`degraded: true` means the model server failed and the extractive generator answered.

## `POST /v1/query/stream`

Same body; `text/event-stream` response with events in this order:

| Event | Data |
|---|---|
| `retrieval` | strategy, index version, candidate counts, retrieved documents, cache hit |
| `token` | one **verified** sentence at a time (text and its citations) |
| `citations` | the citation list |
| `done` | the full `/v1/query` response |
| `error` | `{code, message}` — replaces `done` |

Closing the connection cancels generation.

## `POST /v1/retrieve`

Same body; returns ranked chunks with scores per stage and the query plan, no generation.

## `POST /v1/feedback` → 201

`{"trace_id", "rating": "up"|"down", "categories": [...], "chunk_ids": [...], "query_hash",
"strategy", "index_version", "latency_ms"}`. Categories are a closed list; there is no free text.

## Batch jobs (researcher and above)

- `POST /v1/batch-jobs` → 202: `{"items": [{"id", "query"}], "strategy", "response_length",
  "priority"}` or `{"input_uri": "file-in-batch-input-dir.jsonl"}`.
- `GET /v1/batch-jobs/{job_id}`: status (`queued|running|completed|failed|cancelled`), progress
  counts, model and index version, output location.
- `POST /v1/batch-jobs/{job_id}/cancel`.

A job runs with the permissions of the user who created it and is invisible to other tenants.

## Evaluations (admin)

- `POST /v1/evaluations` → 202: `{"dataset": "queries.jsonl", "strategies": ["bm25", "hybrid"],
  "baseline": "bm25", "limit": null}`; runs in the background.
- `GET /v1/evaluations/{run_id}`: status, dataset version, per-strategy metrics, paired-bootstrap
  comparisons, release-gate results, artifact location.

## Health

- `GET /healthz` — liveness.
- `GET /readyz` — 200 only when the database answers, an index is active and the search backend
  is reachable. Cache and model-server problems are reported as `degraded` without failing readiness.
- `GET /api/v1/metrics` — Prometheus metrics.
