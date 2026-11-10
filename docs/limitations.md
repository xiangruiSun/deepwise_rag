# Assumptions, blockers and what has not been validated

This file is the honest status of the project. Read it before drawing conclusions from any output.

## No INSPECT experiment has been run

The dataset is access-controlled and was not available. **Every number produced so far comes from
fabricated synthetic fixtures** and verifies that the code works. Nothing in this repository is
evidence about clinical retrieval quality on INSPECT.

Still required on authorized data: validating the adapters against the real release, building the
clinician-reviewed relevance set, running the suites, selecting on validation and evaluating the
frozen configuration once on the held-out test set.

## Not validated

| Component | Status |
|---|---|
| `inspect_legacy` / `inspect_meds` adapters | Synthetic or generated fixtures only; real file/column names unknown |
| Elasticsearch backend, PostgreSQL, Docker Compose, images | Written, DSL unit-tested; never run against live services here |
| `inspect_pretrained` image provider | Integration contract only; no weights; usefulness for similarity unproven |
| LLM generation (Ollama / vLLM / Hugging Face) and LLM query rewriting / HyDE | Tested with a scripted fake client only; no local LLM was available, so G02–G06, R10 and R12 report "failed"/"skipped" |
| GPU execution | CPU only so far |
| MLflow logging, OpenTelemetry export | Optional paths, not exercised |
| Scale (≈20k studies, millions of events) | Streaming/batched design, but only measured on 189 synthetic cases |

## Verified

Backend unit, integration, security and leakage tests; real MiniLM, E5, BGE, MedCPT (query + article
encoders) and both cross-encoders run end to end on the synthetic cohort; frontend build, lint and
Playwright end-to-end tests against the live API.

## Design decisions worth knowing

* **Haystack** is integrated as optional components and a pipeline that reproduces the native
  ranking (`retrieval/haystack_components.py`, parity-tested with `haystack-ai` 3.4). The core service
  does not depend on it, so each stage stays independently testable.
* **Local search backend** exists for CPU-only, infrastructure-free runs and hermetic tests. It uses
  exact kNN, so its latency and recall behaviour differ from Elasticsearch's approximate kNN;
  performance conclusions must come from the Elasticsearch backend.
* **Hashing embedding** is a dependency-free smoke baseline, not one of the four compared models.
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
