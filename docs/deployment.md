# Deployment and operations

## Local, no infrastructure (CPU-only)

```bash
make setup            # .venv, pinned backend deps, frontend deps, .env
make seed-synthetic   # generate + ingest the SYNTHETIC cohort
make index-baseline   # case index, BM25 index, hashing-dense index, image features
make run-baseline     # smoke suite
make report           # artifacts/experiments/report/report.md
make serve            # API on http://127.0.0.1:8000  (docs at /docs)
make frontend         # UI on http://localhost:5173
```

This mode uses SQLite and the local search backend under `artifacts/`.

## Docker Compose (PostgreSQL + Elasticsearch)

```bash
cp .env.example .env    # set POSTGRES_PASSWORD
make up                 # postgres, elasticsearch, backend (runs alembic), frontend on :8080
docker compose exec backend python -m app.cli generate-synthetic
docker compose exec backend python -m app.cli ingest
docker compose exec backend python -m app.cli reindex
docker compose --profile llm up -d        # optional Ollama
docker compose --profile tracking up -d   # optional MLflow
```

**Not verified in this repository's development environment:** the Compose stack, the Docker images
and the live Elasticsearch/PostgreSQL path were written and reviewed but could not be run there (no
Docker daemon). The Elasticsearch query DSL is unit-tested and
`backend/tests/integration/test_elasticsearch_live.py` runs automatically when a node is reachable:

```bash
INSPECT_RAG_ELASTICSEARCH_URL=http://localhost:9200 make test
```

Run it once after `make up` before relying on the Elasticsearch backend.

## Configuration

* Environment (`INSPECT_RAG_*`, see `.env.example`): connections, secrets, paths, API keys.
* YAML (`configs/`): dataset, embeddings, chunking, indexes, retrieval recipes, multimodal fusion,
  generation, evaluation. Validated on load; unknown keys are errors.

## Authentication

`INSPECT_RAG_API_KEYS_JSON` maps keys to `{user, role, scopes}`. Roles: `annotator` < `researcher` <
`admin`. Scopes are the `access_scope` values a principal may read. The development key
(`INSPECT_RAG_DEV_API_KEY`) is ignored when `INSPECT_RAG_ENVIRONMENT=production`. Terminate TLS in
front of the API; do not expose Elasticsearch or PostgreSQL beyond localhost.

## GPU

`device: auto` uses CUDA when available, otherwise CPU. Install `backend/requirements-ml.lock`
(`make setup-ml`) for the neural embedding, reranking and local Hugging Face generation backends.

## Models and revisions

Model weights are downloaded by Hugging Face on first use and are never committed. Each run records
the exact commit of every model. To freeze them, copy the output of
`python scripts/pin_model_revisions.py configs/<config>.yaml` into the `revision` fields.

## Indexes

One report index per (embedding, chunking) pair, e.g. `report_chunks_v1_minilm`,
`report_chunks_v1_e5__sentence`; `report_chunks_v1_lexical` for BM25. Promote a version without
deleting others: `python -m app.cli reindex --promote report_chunks_v1_minilm` (alias
`inspect_rag_reports_current`). `python -m app.cli index-health` shows versions, sizes and health.

## Observability

`/api/v1/metrics` (Prometheus): request counts, error codes, per-stage latency histograms, LLM token
counters, denied-access counters. OpenTelemetry tracing is enabled with `INSPECT_RAG_OTEL_ENABLED=true`
when the optional packages are installed. Logs contain identifiers and counts, not clinical text.

## Object storage

Artifacts are written to the local filesystem (`INSPECT_RAG_ARTIFACT_DIR`). Image sources are
referenced by path in the relational store; an S3-compatible store can replace this by mounting it or
by resolving `ImagingStudy.source_path` references in `imaging/dicom.py:load_volume`.
