# INSPECT-RAG

Evaluation-driven, multimodal, patient-level clinical case retrieval with evidence-grounded RAG, built
for the Stanford INSPECT dataset.

> **Research prototype. Not for diagnosis, treatment decisions or any clinical use.**
> No INSPECT experiment has been run with this code yet: the dataset is access-controlled. The code
> was developed on fabricated synthetic fixtures (both search backends, including the Docker Compose
> stack) and then run on two small open datasets, PMC-Patients and the MIMIC-IV demo
> ([docs/open_datasets.md](docs/open_datasets.md)). Synthetic results show that the pipeline works;
> the open-data results are real but come from different data than INSPECT (no CT images, no
> radiology impressions). Neither is evidence of clinical retrieval performance on INSPECT.
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
| Ingestion | schema inspector, adapters (paper-described CSV layout, MEDS/Parquet, synthetic, prepared open datasets), converters for PMC-Patients and the MIMIC-IV demo, validation reports, incremental updates with deletion propagation |
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
Deployment: [docs/deployment.md](docs/deployment.md) · Open datasets: [docs/open_datasets.md](docs/open_datasets.md)

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

## Local LLM generation

The default generator is extractive and needs no model. For generated answers, start the bundled
Ollama server and use a config that names a model, for example Llama 3.2 1B (about 1.3 GB, CPU-capable):

```bash
make up-llm && make pull-llm
INSPECT_RAG_APP_CONFIG=synthetic_llm.yaml INSPECT_RAG_REQUEST_TIMEOUT_SECONDS=110 docker compose up -d backend
```

The model runs locally and its statements pass the same citation verification as every other
generator. Models this small are weak writers; see the measured comparison in
[experiments/open_data/](experiments/open_data/README.md) and
[docs/deployment.md](docs/deployment.md#local-llm-generation-ollama).

## Real data without INSPECT access

Two small, openly licensed, published datasets can stand in for INSPECT while access is pending:
**PMC-Patients** (patient summaries with published similarity labels — tests text retrieval) and the
**MIMIC-IV demo** (100 patients with laboratory values and diagnosis codes — tests the structured
path). Neither has images. You download them yourself; `prepare-dataset` converts them:

```bash
cd backend
python -m app.cli prepare-dataset pmc-patients     # needs data/pmc_patients/raw/
python -m app.cli prepare-dataset mimic-demo       # needs data/mimic_iv_demo_meds/
```

Full steps, mapping decisions and caveats: [docs/open_datasets.md](docs/open_datasets.md). Results:
[experiments/open_data/](experiments/open_data/README.md).

## CLI

```bash
cd backend && source ../.venv/bin/activate

python -m app.cli --help
python -m app.cli prepare-dataset pmc-patients                      # convert a downloaded open dataset
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

`configs/experiments/pmc/` holds the same stages for PMC-Patients (embedding, chunking, retrieval,
generation); their report is in [experiments/open_data/](experiments/open_data/README.md).

## Document-corpus platform (enterprise extension)

`backend/app/corpus` and the `/v1` API extend the case-retrieval application into a tenant-aware RAG
platform over a large document corpus: versioned ingestion, a hybrid index behind an alias,
permission-filtered retrieval, verified streaming answers, durable batch jobs, an evaluation
platform with release gates, change-event indexing, scheduled maintenance and monitoring.

> **What was and was not run.** The design targets 100M+ records. Measured here, on one
> workstation with **synthetic** documents: 1M records through the ETL (local engine and Spark local
> mode), 20k-record corpora through indexing, retrieval experiments and an HTTP load test on
> PostgreSQL + Elasticsearch. **Not run:** anything at 100M, a Spark cluster, a real Redis or Kafka
> broker, a GPU model server (vLLM), a Kubernetes cluster. Generation in the measurements is the
> deterministic extractive generator, not a language model.

```bash
make ingest-demo     # generate a synthetic corpus and load the document registry
make reindex         # build -> validate -> activate an index version
make serve           # /v1/query, /v1/query/stream, /v1/retrieve, /v1/batch-jobs, /v1/evaluations
make eval            # strategy comparison; fails when a release gate fails
make test-security   # tenant isolation, access control, cache isolation, prompt injection
make experiments     # before/after experiments -> experiments/enterprise/
```

| Document | Content |
|---|---|
| [docs/ENGINEERING_CONTRIBUTIONS.md](docs/ENGINEERING_CONTRIBUTIONS.md) | each contribution: problem, change, validation, measured before/after, limits |
| [docs/enterprise/01_architecture.md](docs/enterprise/01_architecture.md) | components, request and write paths, consistency and failure behaviour |
| [docs/enterprise/02_data_contracts.md](docs/enterprise/02_data_contracts.md) | canonical document, chunk, change event, evaluation and batch formats |
| [docs/enterprise/03_api.md](docs/enterprise/03_api.md) | `/v1` request and response contracts, errors, streaming events |
| [docs/enterprise/04_deployment.md](docs/enterprise/04_deployment.md) | local, Docker Compose, Kubernetes, release and rollback |
| [docs/enterprise/05_runbook.md](docs/enterprise/05_runbook.md) | operating procedures for latency, lag, dead letters, rollback |
| [docs/enterprise/06_experiments.md](docs/enterprise/06_experiments.md) | every experiment and benchmark with its numbers and how to reproduce it |

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
backend/app/corpus/   document-corpus platform: ETL, index manager, retrieval, generation, cache,
                      batch jobs, evaluation, change events, scheduler
backend/tests/        unit and integration tests
backend/alembic/      database migrations
frontend/             React application and Playwright tests
configs/              application configs (synthetic, INSPECT, PMC-Patients, MIMIC-IV demo) and experiment suites
experiments/          non-sensitive result summaries
scripts/              setup helpers, experiment and benchmark runners
loadtest/             Locust load test
infra/                Prometheus, Grafana and Kubernetes manifests
data/                 git-ignored; synthetic fixtures are generated, INSPECT and the open datasets are supplied by you
docs/                 architecture, API, evaluation, dataset, deployment, limitations
```

## Data and privacy

No medical data, model weights, credentials or database state are committed. Audit events and logs
hold identifiers and counts, not clinical text. Per-query experiment artifacts contain case
identifiers and are scope-restricted; only aggregate, non-sensitive summaries belong in `experiments/`.
