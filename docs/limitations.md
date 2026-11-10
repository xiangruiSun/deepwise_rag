# Assumptions, blockers and what has not been validated

This file is the honest status of the project. Read it before drawing conclusions from any output.

## No INSPECT experiment has been run

The dataset is access-controlled and was not available. Numbers produced so far come from
**fabricated synthetic fixtures** (pipeline verification) and from two **small open datasets**,
PMC-Patients and the MIMIC-IV demo ([open_datasets.md](open_datasets.md)). The open datasets are real
data but they are not INSPECT: no CT images, no radiology impressions, and labels that are not
clinician judgments. Nothing in this repository is evidence about clinical retrieval quality on
INSPECT.

Still required on authorized data: validating the adapters against the real release, building the
clinician-reviewed relevance set, running the suites, selecting on validation and evaluating the
frozen configuration once on the held-out test set.

## Not validated

| Component | Status |
|---|---|
| `inspect_legacy` / `inspect_meds` adapters | Synthetic or generated fixtures only; real file/column names unknown |
| `inspect_pretrained` image provider | Integration contract only; no weights; usefulness for similarity unproven |
| LLM generation through Ollama | Run with Llama 3.2 1B and Qwen 2.5 1.5B (CPU, Docker) on synthetic data and on PMC-Patients. Models of this size are weak: see the generation results in [open_datasets.md](open_datasets.md). vLLM / Hugging Face backends: fake client only |
| LLM query rewriting / HyDE (R10, R12) | Not run: they need the application config to name an LLM (`synthetic_llm.yaml`) and were not part of this round |
| Imaging on real data | Neither open dataset has images; the imaging path is verified on synthetic volumes only |
| Laboratory similarity as a relevance signal | Runs on real MIMIC-IV demo values, but a diagnosis-overlap proxy showed no advantage over chance (37 query stays; see [open_datasets.md](open_datasets.md)) |
| GPU execution | Apple GPU (MPS) for embeddings and rerankers, checked to match CPU vectors; CUDA not exercised |
| MLflow logging, OpenTelemetry export | Optional paths, not exercised |
| Scale (≈20k studies, millions of events) | Text side measured on 20,400 PMC-Patients cases (Elasticsearch); structured side on 150k MIMIC-IV demo events. Millions of events not measured |
| Compose `tracking` (MLflow) profile | Defined but never started (the `llm` profile is verified) |

## Verified

Backend unit, integration, security and leakage tests — on the local backend + SQLite and again on
live Elasticsearch 8.15.3 + PostgreSQL 16; real MiniLM, E5, BGE, MedCPT (query + article encoders)
and both cross-encoders run end to end on the synthetic cohort on both backends; the Alembic
migration on PostgreSQL; the Docker Compose stack (images, migration on start, workflow inside the
container, UI and API through nginx); frontend build, lint and Playwright end-to-end tests against
the live API on both backends.

Added in the second round: answer generation with local open-source models (Llama 3.2 1B,
Qwen 2.5 1.5B) through the Compose `llm` profile, from the host and from inside the backend container;
ingestion, indexing and the evaluation protocol on PMC-Patients (20,400 cases on Elasticsearch,
including selection on validation queries and one held-out test run); structured ingestion and
laboratory/diagnosis similarity on the MIMIC-IV demo; the web UI on PMC-Patients with an LLM answer.

On the synthetic cohort the two search backends give near-identical quality (nDCG@10 within about
0.01; differences come from Elasticsearch's analyzer and HNSW search). The committed example report
was produced on the local backend.

## Design decisions worth knowing

* **Haystack** is integrated as optional components and a pipeline that reproduces the native
  ranking (`retrieval/haystack_components.py`, parity-tested with `haystack-ai` 3.4). The core service
  does not depend on it, so each stage stays independently testable.
* **Local search backend** exists for CPU-only, infrastructure-free runs and hermetic tests. It uses
  exact kNN, so its latency and recall behaviour differ from Elasticsearch's approximate kNN;
  performance conclusions must come from the Elasticsearch backend.
* **Deep result pages on Elasticsearch** (beyond `index.max_result_window`, 10,000) are read with
  `search_after`; this is used when laboratory or diagnosis similarity ranks every eligible case.
* **Hashing embedding** is a dependency-free smoke baseline, not one of the four compared models.
* **`device: auto`** picks CUDA, then the Apple GPU (MPS), then CPU. MPS gave the same vectors as CPU
  (cosine above 0.999999) about 4.5 times faster on the development machine.
* **Open datasets go through a converter, not a new ingestion path.** `prepare-dataset` writes the
  canonical CSV layout plus a `DATASET.json` provenance marker; the `prepared` adapter refuses a
  folder without a matching marker, and reports name the dataset instead of calling it INSPECT.
* **Intensity-histogram image encoder** is a transparent placeholder for the multimodal code path.
* **Query-case encoding** for dense retrieval defaults to the document encoder
  (`case_query_encoding`), an evaluated hyper-parameter.
* **Diagnosis filters** require all listed codes.
* **Claim-support and failure categorisation** are heuristics for triage, pending human calibration.
* **Safetensors weights are copied into ordinary tensors after loading.** On the development machine
  (PyTorch 2.15, Apple Silicon) memory-mapped weights made one cross-encoder return NaN logits; model
  outputs are additionally checked for non-finite values and fail loudly.

## Research-only

Not a medical device. No diagnosis, treatment recommendation or risk prediction. Image similarity is
an experimental representation score and never a statement about findings.
