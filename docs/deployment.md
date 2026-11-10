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
docker compose --profile llm up -d        # optional Ollama (see "Local LLM generation")
docker compose --profile tracking up -d   # optional MLflow
```

The stack was run end to end on synthetic data (Docker Desktop 4.99, Elasticsearch 8.15.3,
PostgreSQL 16.4): images build, the backend applies the Alembic migration on start, ingestion,
indexing and the baseline suite run inside the container, and the UI and API work through the nginx
frontend on port 8080. The `llm` profile (Ollama 0.3.12) was run with Llama 3.2 1B and Qwen 2.5 1.5B;
the `tracking` profile (MLflow) was not started.

### Local LLM generation (Ollama)

The default generator is extractive and needs no model. To generate answers with a local
open-source model instead:

```bash
docker compose --profile llm up -d                     # adds the Ollama server
docker compose exec ollama ollama pull llama3.2:1b     # ~1.3 GB, once (make pull-llm also pulls qwen2.5:1.5b)
INSPECT_RAG_APP_CONFIG=synthetic_llm.yaml INSPECT_RAG_REQUEST_TIMEOUT_SECONDS=110 docker compose up -d backend
```

`configs/synthetic_llm.yaml` (and `baseline_llm.yaml` for INSPECT) set `generation.backend: ollama`
with Llama 3.2 1B, the smallest widely used open-weight chat model. The model runs on your machine;
no report text leaves it. Its answers go through the same citation verification as every other
generator, and the same model serves the optional `llm_rewrite` / `hyde` query transformations.

* The backend finds the server through `INSPECT_RAG_OLLAMA_BASE_URL` (Compose: `http://ollama:11434`;
  outside Docker: `http://localhost:11434`).
* Docker on macOS has no GPU access, so the container runs the model on CPU (tens of seconds per
  answer). A native Ollama install uses the Apple GPU and is several times faster; point the backend
  at it with `INSPECT_RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434`.
* `generation.context_window` (default 8192) is sent to Ollama as `num_ctx`. Ollama's own default is
  smaller than the evidence budget and silently truncates the prompt, instructions first.
* A larger model is a one-line change (`generation.model`, e.g. `llama3.2:3b`); compare candidates
  with `experiments/generation_comparison.yaml` before switching.

### Testing against live services

The test suite is hermetic by default. To run every test context on real services instead:

```bash
cd backend
INSPECT_RAG_TEST_ELASTICSEARCH_URL=http://127.0.0.1:9200 \
INSPECT_RAG_TEST_DATABASE_URL=postgresql+psycopg://inspect_rag:<password>@127.0.0.1:5432/inspect_rag \
python -m pytest -m "not neural"
```

Each context gets uniquely prefixed indexes and its own database, removed when the session ends.

### Elasticsearch on a nearly full disk

Elasticsearch stops allocating shards when the disk holding its data is more than 90% full (high
watermark); index creation then hangs until the request times out. Free disk space, or — for local
development only — disable the check:

```bash
curl -X PUT localhost:9200/_cluster/settings -H 'Content-Type: application/json' \
  -d '{"persistent": {"cluster.routing.allocation.disk.threshold_enabled": false}}'
```

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
