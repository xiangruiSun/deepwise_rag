# INSPECT-RAG

Evaluation-driven, multimodal, patient-level clinical case retrieval with evidence-grounded RAG, built
for the Stanford INSPECT dataset.

> **Research prototype. Not for diagnosis, treatment decisions or any clinical use.**
> No INSPECT experiment has been run with this code yet: the dataset is access-controlled, so
> everything was developed and verified on fabricated synthetic fixtures (on both search backends,
> including the Docker Compose stack). Synthetic results show that
> the pipeline works — they are not evidence of clinical retrieval performance.
> See [docs/limitations.md](docs/limitations.md).

## What it does

A researcher picks an existing case or describes clinical characteristics, optionally with structured
eligibility filters. The system returns a ranked list of **distinct historical cases** with the
evidence behind each one, lets the user compare cases side by side, and can produce a comparison in
which every statement cites an authoritative source record — or says that the evidence is
insufficient.

The core principle is **evaluation-driven development**: competing embeddings, retrievers, fusion
rules, rerankers, modalities and generators are all configurations; the final pipeline is chosen by a
predeclared protocol on validation data and evaluated once on a held-out test set.

## Components

| Area | Implementation |
|---|---|
| Ingestion | schema inspector, adapters (paper-described CSV layout, MEDS/Parquet, synthetic), validation reports, incremental updates with deletion propagation |
| Canonical store | PostgreSQL / SQLite via SQLAlchemy 2 + Alembic |
| Search | Elasticsearch 8 (BM25, filters, kNN) or a local in-process backend with the same contract |
| Embeddings | MiniLM, E5-base-v2, BGE-base-v1.5, MedCPT (query + article encoders), persistent cache |
| Retrieval | BM25, dense, RRF, weighted score fusion, cross-encoder reranking, patient-level aggregation |
| Multimodal | laboratory similarity on typed values, diagnosis-code overlap, CT study representations, rank/score fusion with a missing-modality policy |
| RAG | deterministic token-budgeted evidence bundles, extractive or local-LLM generation, citation verification, abstention |
| Evaluation | graded relevance, nDCG/Recall/Precision/MRR/judged coverage, safety gates, paired bootstrap, selection policy, Markdown/JSON reports |
| Frontend | React + TypeScript: search, results, case view, comparison, grounded answer, evaluation dashboard |
| Ops | Docker Compose, Makefile, CLI, Prometheus metrics, optional OpenTelemetry and MLflow |

Architecture: [docs/architecture.md](docs/architecture.md) · API: [docs/api.md](docs/api.md) ·
Evaluation protocol: [docs/evaluation.md](docs/evaluation.md) · Dataset: [docs/dataset.md](docs/dataset.md) ·
Deployment: [docs/deployment.md](docs/deployment.md)

## Quick start (synthetic data, CPU-only, no infrastructure)

Requirements: Python 3.11, Node 20+ (frontend only).

```bash
make setup            # virtualenv, pinned dependencies, frontend packages, .env
make seed-synthetic   # generate and ingest the fabricated cohort
make index-baseline   # build the baseline indexes
make run-baseline     # run the baseline evaluation
make report           # artifacts/experiments/report/report.md
make serve            # API: http://127.0.0.1:8000/docs
make frontend         # UI:  http://localhost:5173
```

The development API key is `dev-research-key` (ignored in production mode).

Other commands: `make test`, `make lint`, `make run-experiments`, `make run-neural-experiments`
(after `make setup-ml`), `make select SUITE=<suite>`, `make e2e-prepare && make e2e`, `make up`,
`make down`. `make help` lists everything.

## CLI

```bash
cd backend && source ../.venv/bin/activate

python -m app.cli --help
python -m app.cli inspect-dataset ../data/synthetic                 # files, schemas, layout
python -m app.cli validate-dataset -c synthetic.yaml                # resolved column mapping
python -m app.cli ingest                                            # incremental ingestion
python -m app.cli reindex --embedding minilm --chunking sentence    # one versioned index
python -m app.cli embed minilm                                      # embed + cache, with throughput
python -m app.cli search --text "segmental embolism without right heart strain" --age-min 50 --age-max 75
python -m app.cli search --case-id <case_id> --recipe multimodal_hashing --top-k 5
python -m app.cli run-experiment experiments/embedding_comparison.yaml --only E01 --only E02
python -m app.cli evaluate experiments/smoke.yaml experiments/multimodal_ablation.yaml
python -m app.cli select retrieval_comparison                       # selection policy on validation
python -m app.cli freeze-final experiments/retrieval_comparison.yaml
python -m app.cli run-experiment ../artifacts/experiments/final_test.yaml --final-test
python -m app.cli aggregate-metrics                                 # report.md + report.json + charts
python -m app.cli build-annotation-pool embedding_comparison --depth 20
python -m app.cli index-health
```

## Using authorized INSPECT data

The repository never downloads INSPECT. After obtaining it under its data-use agreement:

```bash
# 1. place the files under data/inspect/ (git-ignored)
cd backend
python -m app.cli inspect-dataset ../data/inspect
# 2. set dataset.files / dataset.columns in configs/baseline.yaml to what the inspector reports
python -m app.cli validate-dataset -c baseline.yaml
python -m app.cli ingest -c baseline.yaml
python -m app.cli reindex -c baseline.yaml
```

The adapters have not been validated against the real release; unresolved fields stop ingestion with
an explicit message rather than being guessed. Then create the clinician-reviewed relevance set
(see [docs/evaluation.md](docs/evaluation.md)) and point the experiment suites at `baseline.yaml`.

## Experiment suites

| Suite | Question |
|---|---|
| `smoke` | does the whole pipeline run (CPU-only)? |
| `embedding_comparison` | E01–E05: BM25 vs MiniLM, E5, BGE, MedCPT |
| `chunking_comparison` | C01–C07: whole impression, fixed, sliding, sentence, parent-child |
| `retrieval_comparison` | R01–R12: fusion, rerankers, laboratory/image, query transformations |
| `reranking_depth` | depth 20/40/80 for both cross-encoders |
| `multimodal_ablation` | text / structured EHR / image and their combinations |
| `generation_comparison` | prompt variants and models with frozen retrieval |

A report generated from a full run of these suites on the **synthetic** cohort is kept in
[experiments/synthetic_verification/](experiments/synthetic_verification/report.md) as a worked
example of the output format. It is pipeline verification only.

## Tests

```bash
make test         # unit, integration, security and data-leakage tests (hermetic, synthetic data)
make test-neural  # real embedding models, when PyTorch and weights are available
make lint         # ruff, mypy, eslint, tsc
make e2e          # Playwright against the live API (after make e2e-prepare)
```

The same suite can run against live Elasticsearch and PostgreSQL
(see [docs/deployment.md](docs/deployment.md#testing-against-live-services)).

Covered: stable case ids, schema detection, joins and duplicate handling, unit conversion, missing
measurements, temporal cutoffs, chunk boundaries, encoding contracts, RRF, case aggregation,
same-patient exclusion, robust normalisation, citation validation, metric formulas (against hand
computation), configuration hashing, unauthorized access through every retrieval path, and
future-record leakage probes.

## Repository layout

```
backend/app/          FastAPI application (api, core, db, ingestion, imaging, embeddings, indexing,
                      retrieval, ranking, rag, evaluation, observability, cli.py, main.py)
backend/tests/        unit and integration tests
backend/alembic/      database migrations
frontend/             React application and Playwright tests
configs/              application configs and experiment suites
experiments/          non-sensitive result summaries
scripts/              setup helpers
data/                 git-ignored; synthetic fixtures are generated, INSPECT is supplied by you
docs/                 architecture, API, evaluation, dataset, deployment, limitations
```

## Data and privacy

No medical data, model weights, credentials or database state are committed. Audit events and logs
hold identifiers and counts, not clinical text. Per-query experiment artifacts contain case
identifiers and are scope-restricted; only aggregate, non-sensitive summaries belong in `experiments/`.
