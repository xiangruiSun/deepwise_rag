# Corpus platform — runbook

Every command below is `python -m app.cli …` run in `backend/` (or the backend container).

## Daily signals

| Signal | Where | Healthy |
|---|---|---|
| Readiness | `GET /readyz` | `status: ready` |
| Index backlog | `corpus index health`, `corpus_index_pending_documents` | near zero |
| Freshness lag | `corpus_index_freshness_lag_seconds` | below the configured `consistency_interval_seconds` |
| Dead letters | `corpus schedule run-task data_quality` | `dead_events: 0`, no unexpected reason codes |
| Release gates | nightly `corpus eval run` (exit code 3 on failure) | all gates pass |

## Latency {#latency}

1. Look at per-stage timings: every response carries `timings_ms`; the dashboard splits P95 by strategy.
2. `reranking` dominates → lower `rerank_depth` or route fewer queries to a reranking strategy.
3. `dense_retrieval` dominates → check `num_candidates`, shard count and whether the index is merging.
4. `generation` dominates → check `corpus_llm_in_flight` / `corpus_llm_queue_depth`; scale the
   model server or lower `max_output_tokens`.
5. Cache hit rate dropped → a burst of document changes invalidates tenant caches by design; check
   the change-event rate.

## Index lag {#index-lag}

1. `corpus index health` — pending upserts and deletes.
2. Is the indexer running? `corpus events process` processes one batch and prints outcomes.
3. Embedding slow → the embedding service or GPU; the sync is resumable, restart it freely.
4. `corpus index sync` applies everything pending regardless of events.

## Dead letters {#dead-letters}

- Ingestion rejects: table `corpus_dead_letters`, grouped by `reason_code`. Fix at the source and
  re-run the partition; loading is idempotent.
- Change events: table `corpus_outbox`, `status = 'dead'`, with the error class. After fixing the
  cause, set `status = 'pending'` and `attempts = 0`; replays are safe.

## Registry and index disagree

`corpus index check` reports missing, stale and orphaned documents. `corpus index check --repair`
marks them and `corpus index sync` fixes them. The weekly `consistency` task does both.

## Model server down {#model-server}

Expected behaviour: after `circuit_failure_threshold` failures the circuit opens, answers come from
the extractive generator with `degraded: true`, and the alert `CorpusLlmCircuitOpen` fires. The
circuit half-opens after `circuit_reset_seconds`. No action is needed for recovery once the model
server is healthy.

## Bad index or bad release

- `corpus index rollback` switches the alias to the previous version (kept until `retire_old`).
- Roll back the application image; responses state `model_version`, `index_version` and `strategy`,
  so affected answers can be identified from audit records by trace id.

## Suspected access-control problem

1. Run `pytest -m security` against the deployed configuration.
2. `corpus eval run` on the adversarial queries — the `leakage` gate must be zero.
3. Check the caller's key: tenant and groups come only from `INSPECT_RAG_API_KEYS_JSON`.
4. If a document's ACL was changed, confirm the change event was processed (`corpus index health`);
   until then the old ACL is still in the index — this window is the freshness lag.

## Batch job stuck

`corpus batch status <job_id>`. A job left in `running` by a dead worker is re-queued by
`corpus batch work` after 15 minutes and resumes at the first uncommitted batch; completed parts
are never rewritten with different content.

## Full rebuild

`corpus index build` builds a new version next to the active one (resumable; reuses cached
embeddings where content is unchanged), validates it and switches the alias. Plan capacity first:
`corpus capacity --records N --calibrate`.
