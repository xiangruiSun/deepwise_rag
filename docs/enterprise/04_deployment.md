# Corpus platform — deployment

## What was validated, and how

| Target | Status |
|---|---|
| Local, no infrastructure (SQLite, in-process search backend) | run and tested (default test suite) |
| PostgreSQL 16 + Elasticsearch 8.15 (Docker Compose services) | run: corpus test suite in live mode — see [06_experiments.md](06_experiments.md) |
| Redis cache | tested against `fakeredis` only; a real Redis server was not run |
| Kafka / Redpanda change events | contract test with a fake consumer only; no broker was run |
| vLLM / GPU model server | client tested against a mocked HTTP transport only; no model server was run |
| Compose profiles `platform`, `monitoring` | `docker compose config` validates; containers were not started |
| Kubernetes manifests (`infra/k8s`) | YAML validity only; never applied to a cluster |

## Local (no Docker)

```bash
make setup
make ingest-demo            # synthetic corpus -> registry
make reindex                # build, validate, activate an index version
make serve                  # http://127.0.0.1:8000  (docs at /docs)
make eval                   # strategy comparison with release gates
```

Query it:

```bash
curl -s http://127.0.0.1:8000/v1/query -H 'X-API-Key: dev-research-key' \
  -H 'Content-Type: application/json' -d '{"query": "pulmonary embolism follow-up"}'
```

## Docker Compose

```bash
cp .env.example .env        # set POSTGRES_PASSWORD
docker compose up -d                          # postgres, elasticsearch, backend, frontend
docker compose --profile platform up -d       # + redis, redpanda, minio, indexer, batch worker, scheduler
docker compose --profile monitoring up -d     # + prometheus (9090), grafana (3000)
```

All ports bind to 127.0.0.1. Elasticsearch security is disabled and the MinIO / Grafana passwords
have local defaults: this file is for development, not for a shared network.

To use Redis, set in the corpus config: `cache: {backend: redis, redis_url: redis://redis:6379/0}`.
To consume change events from Redpanda, run the indexer with
`corpus events process --loop --kafka redpanda:9092` (needs the `streaming` extra in the image).

## Model serving

`generation.backend` selects the generator:

| Backend | Use |
|---|---|
| `extractive` | default; deterministic, no model, no GPU |
| `openai_compatible` | vLLM or any server with the OpenAI chat-completions API; set `base_url`, `model`, and the token in `INSPECT_RAG_LLM_API_KEY` |
| `mock` | canned stream for demos; every response is flagged `mock: true` |

Online and batch traffic should use separate model-server deployments (or priorities) so a large
batch job cannot exhaust `max_concurrency` for online users. The client enforces a concurrency
limit and a bounded wait queue per process.

## Configuration

| Variable | Purpose |
|---|---|
| `INSPECT_RAG_DATABASE_URL` | registry database (PostgreSQL in production) |
| `INSPECT_RAG_SEARCH_BACKEND`, `INSPECT_RAG_ELASTICSEARCH_URL`, `…_API_KEY` | search cluster |
| `INSPECT_RAG_CORPUS_CONFIG` | corpus config file in `configs/` |
| `INSPECT_RAG_API_KEYS_JSON` | keys → user, role, groups (`scopes`), `tenant` |
| `INSPECT_RAG_LLM_API_KEY` | bearer token for the model server |
| `INSPECT_RAG_ENVIRONMENT=production` | disables the development API key |

Secrets come from the environment (Kubernetes secrets, a vault agent); none are stored in the
repository or in config files.

## Kubernetes

See [`infra/k8s/README.md`](../../infra/k8s/README.md). In short: the API is a stateless
deployment with liveness (`/healthz`), readiness (`/readyz`) and an HPA; the indexer and batch
workers are separate deployments; maintenance runs as CronJobs; vLLM runs on GPU nodes behind its
own service; a default-deny network policy admits only ingress → API and API/workers → model server.

## Release procedure

1. `make lint test test-security`.
2. Apply database migrations (`alembic upgrade head`) — additive migrations first, code second.
3. If the embedding model, chunking or mapping changed: `corpus index build` (new version, old one
   keeps serving), then evaluate with `corpus eval run`; activation is refused if validation fails.
4. Roll out the API (rolling update, `maxUnavailable: 0`).
5. Watch P95 latency, error rate, abstention rate and index lag for one evaluation interval.

## Rollback

| What | How |
|---|---|
| Application | `kubectl rollout undo deployment/corpus-api` (or redeploy the previous image tag) |
| Index version | `corpus index rollback` — switches the alias back; caches are invalidated |
| Prompt / model | revert the config value; `model_version` and `prompt_version` are in every response and cache key |
| Migration | `alembic downgrade -1` (0002 was verified up and down on SQLite and up on PostgreSQL) |
