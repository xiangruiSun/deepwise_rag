# Results on open data and with a local LLM

Aggregate results only. Per-query files stay in `artifacts/` (git-ignored).
**None of this is an INSPECT result.** How the datasets were mapped: [docs/open_datasets.md](../../docs/open_datasets.md).

* `pmc_patients/` — the generated evaluation report (`report.md`, `report.json`, charts), leaderboards,
  the selection decision and `ENVIRONMENT.md`.
* `mimic_demo_check.json` — output of `scripts/mimic_demo_check.py`.
* `synthetic_generation/` — leaderboard of the LLM generation comparison on the synthetic cohort.

## 1. PMC-Patients: similar-patient retrieval

Subset of the published benchmark: 20,000 of 155,151 corpus patients,
200 validation and 200 test query patients (seed 13),
964 published similarity labels. Fewer distractors than the full benchmark make retrieval
easier, so these numbers are **not comparable with published PMC-Patients results**. Labels come from
the PubMed citation graph, not from clinicians; unlabelled pairs count as not relevant. Search backend:
Elasticsearch 8.15.3. All safety gates passed in every run.

### Embedding models (validation queries, whole summaries, no reranking)

| Run | Method | nDCG@10 | Recall@10 | Precision@10 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|
| E01 | BM25 (no embedding) | 0.344 | 0.429 | 0.086 | 0.379 | 76 |
| E04 | Dense BGE-base-v1.5 | 0.269 | 0.375 | 0.077 | 0.282 | 16 |
| E05 | Dense MedCPT (article encoder on both sides) | 0.225 | 0.305 | 0.064 | 0.251 | 17 |
| E03 | Dense E5-base-v2 | 0.219 | 0.301 | 0.066 | 0.240 | 16 |
| E02 | Dense MiniLM | 0.153 | 0.201 | 0.041 | 0.184 | 15 |

BM25 reads the whole summary. The dense models see only the first 256 (MiniLM) or 512 tokens, and
most summaries are longer than that.

### Chunking (MiniLM, dense only)

| Run | Chunking | nDCG@10 | Recall@10 | Precision@10 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|
| C04 | Fixed 256 tokens, 32-token overlap | 0.170 | 0.229 | 0.048 | 0.201 | 20 |
| C03 | Fixed 256 tokens, zero overlap | 0.168 | 0.225 | 0.048 | 0.193 | 84 |
| C01 | Whole summary (encoder sees the first 256 tokens) | 0.153 | 0.201 | 0.041 | 0.184 | 15 |
| C02 | Fixed 128 tokens, zero overlap | 0.142 | 0.184 | 0.042 | 0.166 | 82 |

Latency in the chunking table was measured while other jobs were running and is not reliable; the
quality columns are. The gain of 256-token windows with overlap over whole summaries is small (95%
interval 0.000 to 0.036).

### Fusion and reranking (dense branch: BGE)

| Run | Method | nDCG@10 | Recall@10 | Precision@10 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|
| R03 | BM25 + dense with RRF | 0.346 | 0.462 | 0.094 | 0.370 | 98 |
| R01 | BM25 | 0.344 | 0.429 | 0.086 | 0.379 | 73 |
| R04 | BM25 + dense with normalized weighted fusion | 0.343 | 0.445 | 0.091 | 0.368 | 94 |
| R06 | RRF + MedCPT cross-encoder (depth 20) | 0.309 | 0.419 | 0.086 | 0.324 | 681 |
| R02 | Dense vectors (BGE) | 0.269 | 0.375 | 0.077 | 0.282 | 17 |
| R05 | RRF + general cross-encoder (depth 20) | 0.222 | 0.353 | 0.073 | 0.217 | 273 |

Selection policy on validation data: **R01** — no configuration showed a credible improvement; simpler baseline kept. RRF fusion is nominally
ahead of BM25 but the 95% interval of the difference includes zero. Both cross-encoders made the
ranking worse; they were trained on short question–passage pairs and here see two long summaries
truncated to 512 tokens together.

### Held-out test (run once, frozen configuration R01)

| | nDCG@10 | Recall@10 | Precision@10 | Queries |
|---|---|---|---|---|
| Validation | 0.344 | 0.429 | 0.086 | 200 |
| Test | 0.346 | 0.424 | 0.096 | 200 |

### Generated comparisons on real summaries (frozen BM25 retrieval, 8 validation items)

| Run | Generator | Prompt | Verified | Relevance | Faithfulness | Citation coverage | Citation relevance | Completeness | Abstention | Parse failures | Seconds per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G01 | extractive-evidence-v1 | structured_citations | yes | 0.33 | 1.00 | 1.00 | 0.92 | — | 1.00 | 0.00 | <0.01 |
| G02 | llama3.2:1b | structured_citations | no | 0.67 | 0.64 | 1.00 | 1.00 | — | 1.00 | 0.00 | 61 |
| G03 | llama3.2:1b | structured_citations | yes | 0.67 | 1.00 | 1.00 | 1.00 | — | 1.00 | 0.00 | 52 |

There is no gold answer, so completeness is not measured. Relevance here only asks whether a
comparison question got similarity or difference statements. Faithfulness is the heuristic
claim-support check; with verification on it is 1.00 by construction, because unsupported statements
are removed before the answer is returned. The numbers rest on very few items.

## 2. MIMIC-IV demo: structured retrieval

100 patients, 275 hospital stays, 145,985 laboratory and
vital-sign values and 4,506 diagnosis codes ingested with no rejected rows.

Laboratory similarity alone (37 query stays against 238 corpus stays, top 5):

| | Value |
|---|---|
| Query stays with a laboratory-similarity ranking | 36 of 37 |
| Mean overlapping laboratory concepts per result | 11.23 |
| Results from the query patient (must be 0) | 0 |
| Diagnosis-category overlap with the query, top 5 by laboratory similarity | 0.0396 |
| Diagnosis-category overlap, average corpus stay | 0.0397 |
| Queries where the top 5 beat the corpus average | 18 of 36 |

The mechanics work on real records: units are converted, the query patient is excluded, and each
result shows the measurements it was compared on. But stays retrieved by laboratory similarity share
no more diagnosis categories with the query than an average stay does. With routine chemistry and
vital signs on 37 query stays this is a weak test, yet it gives no support for weighting
laboratory similarity heavily. There are no relevance labels for this dataset, so no retrieval-quality
metric is reported.

## 3. Local open-source LLMs on the synthetic cohort

Llama 3.2 1B and Qwen 2.5 1.5B through Ollama 0.3.12 on CPU (Docker on macOS has no GPU access),
9 validation items, frozen retrieval.

| Run | Generator | Prompt | Verified | Relevance | Faithfulness | Citation coverage | Citation relevance | Completeness | Abstention | Parse failures | Seconds per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G01 | extractive-evidence-v1 | structured_citations | yes | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 0.00 | <0.01 |
| G02 | llama3.2:1b | basic | no | 0.50 | 0.31 | 0.44 | 0.86 | 0.33 | 1.00 | 0.00 | 34 |
| G03 | llama3.2:1b | structured | no | 1.00 | 0.50 | 0.94 | 0.48 | 0.50 | 1.00 | 0.00 | 37 |
| G04 | llama3.2:1b | structured_citations | no | 1.00 | 0.36 | 1.00 | 0.47 | 0.33 | 1.00 | 0.00 | 43 |
| G05 | llama3.2:1b | structured_citations | yes | 0.60 | 1.00 | 1.00 | 0.90 | 0.40 | 0.89 | 0.00 | 34 |
| G06 | qwen2.5:1.5b | structured_citations | yes | 0.67 | 1.00 | 1.00 | 0.49 | 0.61 | 1.00 | 0.00 | 65 |

What the comparison shows:

* Without verification, most statements of the 1B model fail the claim-support check, and it attaches
  unrelated source ids to its claims (low citation relevance).
* With verification the surviving statements are supported, but many are removed, so answers become
  thin and the model sometimes ends with nothing to say.
* The extractive generator remains the default. A larger local model is a one-line configuration
  change and should be compared with this same suite before use.
