# Phase 0 — repository audit

Audit of the repository at commit `ff2f59c`, before the enterprise extension. Everything below was
verified by reading the code and running the test suite, not taken from the specification.

## What the project was

INSPECT-RAG: a research application for retrieving similar *clinical cases* (one case = one patient
procedure) with evidence-grounded generation. About 14,400 lines of Python in `backend/app`, a React
frontend, 131 backend tests (all passing), Docker Compose, GitHub Actions CI.

## Verified capabilities (before this work)

| Area | Present | Where |
|---|---|---|
| FastAPI, Pydantic, API-key auth with roles and access scopes | yes | `app/api`, `app/core/security.py` |
| PostgreSQL / SQLite via SQLAlchemy, Alembic | yes | `app/db`, `alembic/` |
| Ingestion adapters for clinical cohort layouts, validation, incremental sync, deletion propagation | yes, case-centric | `app/ingestion` |
| Search backend interface with Elasticsearch (BM25, kNN with pre-filter, aliases) and an in-process backend | yes | `app/indexing` |
| Embedding providers (hashing, SentenceTransformers, MedCPT) with a persistent cache | yes | `app/embeddings` |
| Chunking strategies on tokenizer spans | yes | `app/indexing/chunking.py` |
| RRF and weighted score fusion, cross-encoder rerankers | yes | `app/ranking` |
| Grounded generation with citation verification and abstention; Ollama / OpenAI-compatible clients | yes, synchronous | `app/rag` |
| Evaluation: retrieval metrics, paired bootstrap, selection policy, experiment runner, reports | yes, case-level | `app/evaluation` |
| Prometheus metrics, optional OpenTelemetry bootstrap, audit events | partly | `app/observability` |
| Docker Compose (PostgreSQL, Elasticsearch, backend, frontend, optional Ollama) | yes | `docker-compose.yml` |

## Missing for the enterprise target

| Requirement | State before |
|---|---|
| Tenant model; tenant taken from the credential | absent (access scopes only) |
| Generic document / chunk schema with versions, validity and soft deletion | absent (clinical case schema only) |
| Streaming ETL for SQL, JSONL, CSV, Parquet, text; dead letters; partition manifest; resume | absent |
| Distributed ETL (PySpark) | absent |
| Synthetic data at configurable scale with updates, deletions, tenants and labelled queries | absent (189-case cohort only) |
| Versioned index build → validate → alias switch → rollback for a document corpus; consistency check | partly (alias support existed) |
| Delete-by-filter in the search backends | absent |
| Query routing, rewriting, adaptive depth, freshness handling, diversity-aware context | absent / rule expansion only |
| Async LLM client with streaming, concurrency limits, backpressure, circuit breaker | absent (blocking `httpx.post`) |
| SSE streaming endpoint, feedback endpoint, batch-job and evaluation endpoints, `/healthz`, `/readyz` | absent |
| Cache layer (Redis) with tenant-safe keys | absent |
| Durable batch inference with checkpoint and resume | absent |
| Change events / outbox with idempotent, ordered processing | absent |
| Capacity estimation, load tests, Kubernetes manifests, Grafana dashboards | absent |

## Design decision for the extension

The clinical case-retrieval code keeps working unchanged. The enterprise platform is a new package,
`backend/app/corpus`, that **reuses** the existing building blocks rather than copying them:

* `SearchBackend` / `FilterSpec` (both backends gained `delete_by_filter` and a `chunks` mapping);
* `EmbeddingRegistry` and its content-hash cache; the tokenizer-span `Chunker`;
* `reciprocal_rank_fusion`; the `Reranker` implementations and registry;
* `Principal` (now carrying the tenant of the credential), audit events, error types;
* `assess_claim_support`, `neutralize`, the retrieval metric functions and the paired bootstrap.

New HTTP endpoints live under `/v1` (the existing `/api/v1` is untouched).

## Environment used for this work

macOS (Apple M3 Pro, 12 cores, 18 GB RAM, Apple GPU; no CUDA), Python 3.11, Docker Desktop with
Elasticsearch 8.15.3 and PostgreSQL 16, about 9 GB of free disk. Consequences, stated once here and
repeated wherever they matter: nothing at 100M scale was run, no multi-GPU serving was run, and no
Kubernetes cluster was available.
