# Architecture

INSPECT-RAG is a research prototype for retrospective, patient-level clinical case retrieval. It is
**not** a diagnostic or decision-support system.

## Principles

1. **Evaluation-driven.** Every model, retriever, fusion rule, reranker and generator is one
   interchangeable configuration among competitors. Nothing is selected in code; selection happens
   through the protocol in [evaluation.md](evaluation.md).
2. **The relational store is authoritative.** Search indexes hold derived, versioned copies. Typed
   values (laboratory numbers, units, timestamps) are never replaced by generated text.
3. **The retrieval unit is a clinical case** (`patient_id` + `procedure_time`), not a text chunk.
4. **Fail loudly.** Unknown columns, incompatible units, mismatched vector dimensions, non-finite
   model outputs and missing indexes raise descriptive errors instead of being guessed around.
5. **Access control is server-side** and applied inside every query, then re-checked on results.

## Data flow

```
dataset files ──► DatasetInspector ──► data_manifest.json
      │
      ▼
 DatasetAdapter (synthetic | inspect_legacy | inspect_meds)  ─► canonical rows
      │
      ▼
 IngestionPipeline: extract → validate & link → normalise → persist      (PostgreSQL / SQLite)
      │                                   └─► failed_rows.jsonl, validation_report.json, audit event
      ▼
 build_indexes: lab statistics (train only) → image features → Indexer
      │              cases_v1 · report_chunks_v1_<embedding>[__<chunking>] · image_features_v1
      ▼
 RetrievalService.search ──► SearchResponse (ranked distinct cases + evidence + component scores)
      │
      ▼
 RagPipeline: scope guard → EvidenceBundle → Generator → CitationVerifier ──► RagAnswer
      │
      ▼
 ExperimentRunner → per-query results, metrics, gates → ReportBuilder / selection
```

## Modules (`backend/app`)

| Package | Responsibility |
|---|---|
| `core` | settings and YAML config models, hashing/identities, errors, security, wiring (`container.py`) |
| `db` | SQLAlchemy models (section 6 of the spec), engine/session helpers; migrations in `backend/alembic` |
| `ingestion` | inspector, adapters, canonical rows, unit/code normalisation, text cleaning, pipeline, synthetic generator |
| `imaging` | DICOM/NIfTI/NumPy loading, HU conversion, resampling, image embedding providers, feature pipeline |
| `embeddings` | provider interface, hashing / SentenceTransformers / MedCPT providers, persistent cache, registry |
| `indexing` | chunkers, `SearchBackend` protocol, local backend, Elasticsearch backend + DSL, incremental indexer |
| `retrieval` | case features with temporal validity, laboratory similarity, retrievers, query enhancement, `RetrievalService`, Haystack components |
| `ranking` | RRF / score fusion, patient-level aggregation, rerankers, multimodal case ranker |
| `rag` | evidence bundle, prompts, generators, citation verifier + claim support, pipeline, comparison service |
| `evaluation` | metrics, datasets/qrels/pooling, statistics + selection, runner, error analysis, charts, report |
| `observability` | stage timers, Prometheus metrics, logging, optional OpenTelemetry |
| `api` | FastAPI routers and dependencies |

## Search backends

`SearchBackend` (`indexing/backend.py`) has two implementations with one contract:

* **Elasticsearch 8.x** — BM25 (`match` + non-scoring `bool.filter`), approximate kNN with the filter
  *inside* the `knn` clause (pre-filtering), aliases for promotion.
* **Local** — in-process Okapi BM25 and exact cosine / inner-product search, persisted under
  `artifacts/local_index`. It lets the baseline, CI and the synthetic workflow run CPU-only with no
  infrastructure. It is a real retriever with the same pre-filtering semantics, not a mock.

Structured filters (authorization scope, age, gender, diagnosis codes, laboratory ranges, split,
same-patient exclusion) are denormalised onto every search document so they are applied before or
during retrieval for lexical and vector search alike — never only after a top-K cut.

## Retrieval stages

`RetrievalService.search` (see the module docstring) runs: query parsing → query representation →
filter construction → eligibility count → BM25 / dense kNN → **document-level** fusion (RRF or
normalised weighted scores) → optional cross-encoder reranking of a bounded pool → aggregation to
distinct cases → optional **case-level** multimodal fusion → result assembly. Each stage is timed.

Document-level and case-level fusion are separate operations:

* document-level: `ranking/fusion.py` over document ids;
* case-level: `CaseAggregator(method="case_rrf")` over per-branch case rankings, and
  `MultimodalCaseRanker` over modality rankings/scores.

### Modalities

| Modality | Signal | Notes |
|---|---|---|
| text | BM25 / embeddings / cross-encoder over impressions | retrospective, report-available task |
| laboratory | `exp(-robust-scaled mean abs difference)` over overlapping measurements | training-split median/IQR; unavailable below `min_overlap`; never a probability |
| diagnosis | Jaccard of codes recorded at or before the procedure | empty history = unknown, not identical |
| image | cosine over study-level vectors | experimental representation only |

Missing modalities follow an explicit policy (`renormalize`, `zero`, `exclude`); effective weights and
coverage are returned with every result.

### Imaging

`ImageEmbeddingProvider` implementations: `disabled`, `precomputed` (vectors computed elsewhere, e.g.
in the isolated legacy INSPECT environment), `inspect_pretrained` (TorchScript export contract; no
weights bundled, **not validated**), and `intensity_histogram` (a transparent non-learned HU-histogram
+ coarse-grid descriptor that makes the multimodal path runnable). None of them is a validated
lesion-similarity model; the ablation study decides whether imaging belongs in a final recipe. No CT
finding is ever generated from vector similarity.

## Temporal validity

`CaseFeatureService` is the only place that turns events into features and always takes a cutoff:
the case's own procedure time (or an earlier `as_of`). Laboratory values must lie in
`[cutoff − lookback, cutoff]`; diagnoses must be recorded at or before the cutoff. Later records are
only reachable through `follow_up_events` and are labelled as follow-up in the API and UI. INSPECT
dates are patient-specifically shifted, so the API exposes offsets from the procedure, never absolute
dates, and never compares dates across patients.

## RAG

The pipeline never asks a model to fill gaps: no authorized cases, no citable evidence or an
out-of-scope question (treatment, diagnosis, risk prediction, unavailable attributes) yields
`insufficient_evidence` before any generation. Evidence items are references to authoritative
records with markers `S1…`; retrieved text is treated as untrusted (delimited, marker-neutralised).
After generation `CitationVerifier` checks that each cited source exists, belongs to an authorized
case of the request, was in the context and still matches the underlying record; a separate heuristic
claim-support check removes unsupported statements when `post_verification` is on.

Generators: `extractive` (deterministic, model-free, default), `ollama`, `openai_compatible` (vLLM),
`huggingface`. `external` is rejected by config validation unless `external_api.enabled` **and**
`external_api.i_confirm_data_is_authorized_for_this_provider` are both true.

## Security model

* API key → `Principal(user, role, scopes)`; `AccessPolicy` supplies the `access_scope` filter merged
  into every query. Request bodies cannot carry scopes (`extra="forbid"`).
* Direct lookups (cases, timeline, reports, images, previews, compare, RAG, citations, experiment
  artifacts) call `assert_case_allowed` / check artifact scope.
* Audit events store identifiers and counts only; logs never contain clinical text by default.
* Storage paths of images are never returned by the API.
* Research labels (PE status, outcomes) live in a separate table, are not filterable and never enter
  query features.

## Dataset adapters

`synthetic`, `inspect_legacy` (cohort CSV + impression CSV + FEMR-style event CSVs), `inspect_meds`
(MEDS Parquet) and `prepared` — open datasets converted to the canonical CSV layout by
`prepare-dataset` (`app/ingestion/converters/`), identified by a `DATASET.json` marker. All adapters
emit the same canonical rows; nothing downstream knows which dataset it is, except that reports and
the API name it (`data_kind`) and the text record carries the dataset's own label
(`dataset.text_label`, e.g. "Patient summary").

## Known deviations and open items

See [limitations.md](limitations.md).
