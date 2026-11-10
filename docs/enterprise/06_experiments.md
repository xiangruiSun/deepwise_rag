# Corpus platform — experiments and benchmarks

Every number on this page was produced by a command in this repository on one workstation
(Apple silicon, arm64, 12 cores, 18 GB). The corpus and its labels are **synthetic**
(`app/corpus/generator.py`): the results describe relative behaviour of configurations and the
mechanics of the platform. They are not evidence of quality on real clinical documents, and
nothing here was run at 100M records.

| Artifact | Content |
|---|---|
| `experiments/enterprise/hashing/` | experiments with the CPU-only configuration (`configs/corpus.yaml`) |
| `experiments/enterprise/neural/` | the same experiments with MiniLM embeddings and a cross-encoder reranker (`configs/corpus_neural.yaml`) |
| `experiments/enterprise/benchmarks/etl_1m.json` | 1M-record ETL, local engine and Spark |
| `experiments/enterprise/benchmarks/index_build_es_pg.json` | index build on Elasticsearch from a PostgreSQL registry |
| `experiments/enterprise/benchmarks/api_es_pg_cache_*.json` | HTTP concurrency sweeps, cache off and on |
| `experiments/enterprise/benchmarks/freshness_es_pg.json` | update-visibility lag |
| `experiments/enterprise/benchmarks/locust_es_pg_stats.csv` | Locust mixed-traffic run |

## Reproduce

```bash
make experiments                                   # hashing configuration, ~13 minutes
cd backend && python ../scripts/enterprise_experiments.py --config corpus_neural.yaml \
    --work-dir ../artifacts/corpus_experiments_neural --out ../experiments/enterprise/neural   # needs setup-ml
docker compose up -d postgres elasticsearch        # then: migrate, ingest-demo, reindex, serve
make benchmark loadtest
cd backend && python ../scripts/corpus_freshness_benchmark.py --out ../experiments/enterprise/benchmarks/freshness.json
```

## The evaluation set

20,000 logical source records (21,797 rows with injected duplicates, late versions and invalid
rows) across 8 tenants become 19,387 live documents. 400 "golden" documents carry 800 queries in
13 categories: `keyword`, `abbreviation`, `paraphrase` (120 each), `temporal_update` (80), and 40
each of `exact_identifier`, `multi_document`, `authorized_restricted`, `unauthorized_group`,
`cross_tenant`, `freshness_superseded`, `historical`, `prompt_injection`, `no_answer`.

## 1. Retrieval ablation

Baseline `bm25`; differences are paired-bootstrap means over queries with relevant documents.
"hashing" is a character-n-gram hashing vectoriser with a lexical-overlap reranker — a stand-in
that needs no model. "neural" is `all-MiniLM-L6-v2` (384 dimensions) with
`cross-encoder/ms-marco-MiniLM-L6-v2`, both general-domain models.

| Strategy | hashing: nDCG@10 | Recall@10 | vs bm25 | P50 ms | neural: nDCG@10 | Recall@10 | vs bm25 | P50 ms |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.746 | 0.846 | baseline | 25.5 | 0.746 | 0.846 | baseline | 25.9 |
| `dense` | 0.556 | 0.711 | regressed (-0.190) | 16.0 | 0.613 | 0.761 | regressed (-0.133) | 25.2 |
| `hybrid` | 0.637 | 0.793 | regressed (-0.110) | 34.5 | 0.726 | 0.887 | no credible difference (-0.020) | 35.6 |
| `hybrid_rerank` | 0.529 | 0.736 | regressed (-0.217) | 34.9 | 0.839 | 0.956 | improved (+0.092) | 76.6 |
| `hybrid_rerank_routed` | 0.672 | 0.804 | regressed (-0.075) | 35.2 | 0.926 | 0.957 | improved (+0.180) | 62.9 |
| `adaptive` | 0.672 | 0.804 | regressed (-0.075) | 35.5 | 0.926 | 0.957 | improved (+0.180) | 61.2 |
| `no_freshness` | 0.630 | 0.794 | regressed (-0.116) | 34.4 | 0.721 | 0.885 | no credible difference (-0.025) | 35.4 |

Latency is sequential, single-process, in-process search backend, extractive generation; it is
not a load test (see section 6).

**What the neural run shows** (paired bootstrap against `bm25`, 680 queries with relevant documents):

- `dense` alone is worse than BM25 (nDCG@10 −0.133, 95 % CI −0.174 … −0.093): it wins on
  paraphrases and loses identifier, temporal and multi-document questions.
- `hybrid` raises Recall@10 (+0.041, CI +0.013 … +0.069) but not nDCG@10 (−0.020, no credible
  difference): fusion finds more relevant documents and ranks them no better.
- Adding the cross-encoder (`hybrid_rerank`) improves both: nDCG@10 +0.092 (CI +0.063 … +0.121),
  Recall@10 0.846 → 0.956, for about 38 ms of reranking at the median.
- Adding routing (`hybrid_rerank_routed`) gives the best result: nDCG@10 0.926 (+0.180,
  CI +0.157 … +0.203). It is also faster than reranking without routing (P50 63 ms vs 77 ms),
  because entity-scoped queries rerank fewer candidates.
- `adaptive` is indistinguishable from `hybrid_rerank_routed` on this fixture: adaptive depth
  changed no top-10 result.

**What the hashing run shows.** With stand-in models every strategy that uses the vector or the
reranker is *worse* than BM25 (all verdicts "regressed"). That is the expected result for a
character-n-gram vectoriser, and it is the reason the platform treats the strategy as
configuration and compares it per release: the same pipeline is a clear gain with real models
and a loss with weak ones.

### nDCG@10 by query category — neural configuration

| Category | Queries | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_routed |
|---|---|---|---|---|---|---|
| abbreviation | 120 | 0.747 | 0.642 | 0.838 | 0.906 | 0.966 |
| authorized_restricted | 40 | 0.860 | 0.950 | 0.960 | 1.000 | 1.000 |
| exact_identifier | 40 | 1.000 | 0.042 | 0.256 | 0.982 | 1.000 |
| freshness_superseded | 40 | 0.988 | 0.510 | 0.752 | 0.646 | 1.000 |
| historical | 40 | 1.000 | 0.721 | 0.852 | 0.892 | 1.000 |
| keyword | 120 | 0.888 | 0.904 | 0.962 | 1.000 | 1.000 |
| multi_document | 40 | 1.000 | 0.019 | 0.577 | 0.959 | 1.000 |
| paraphrase | 120 | 0.074 | 0.493 | 0.362 | 0.615 | 0.615 |
| prompt_injection | 40 | 0.717 | 0.941 | 0.960 | 1.000 | 1.000 |
| temporal_update | 80 | 0.995 | 0.562 | 0.751 | 0.606 | 1.000 |

### nDCG@10 by query category — hashing configuration

| Category | Queries | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_routed |
|---|---|---|---|---|---|---|
| abbreviation | 120 | 0.749 | 0.719 | 0.744 | 0.652 | 0.678 |
| authorized_restricted | 40 | 0.860 | 0.816 | 0.833 | 0.606 | 0.598 |
| exact_identifier | 40 | 1.000 | 0.378 | 0.760 | 0.925 | 1.000 |
| freshness_superseded | 40 | 0.988 | 0.469 | 0.693 | 0.483 | 1.000 |
| historical | 40 | 1.000 | 0.763 | 0.884 | 0.692 | 1.000 |
| keyword | 120 | 0.888 | 0.897 | 0.896 | 0.725 | 0.725 |
| multi_document | 40 | 1.000 | 0.187 | 0.441 | 0.635 | 1.000 |
| paraphrase | 120 | 0.074 | 0.038 | 0.041 | 0.038 | 0.038 |
| prompt_injection | 40 | 0.717 | 0.760 | 0.717 | 0.498 | 0.498 |
| temporal_update | 80 | 0.995 | 0.558 | 0.725 | 0.453 | 1.000 |

Reading by category (neural):

- **Paraphrase** is where lexical search fails (BM25 0.074) and dense retrieval is needed (0.493);
  reranking lifts it to 0.615. It remains the weakest category.
- **Exact identifier** and **multi-document** questions collapse under pure dense retrieval
  (0.042, 0.019) and are damaged by plain fusion (0.256, 0.577). Routing restores them to 1.000 by
  turning the identifier or the patient into a filter instead of relying on similarity.
- **Temporal** and **superseded-report** questions are hurt by reranking without routing (0.606,
  0.646 vs BM25 0.995, 0.988): a general-domain cross-encoder scores an older report of another
  patient as highly as the right one. Entity scoping fixes this (1.000).
- **Keyword** and **abbreviation** questions gain from fusion and again from reranking.

### Answer-level metrics — neural configuration

| Strategy | Abstention correct | Citation coverage | Fact coverage | Freshness error | Context tokens | Rerank P50 ms | End-to-end P95 ms |
|---|---|---|---|---|---|---|---|
| `bm25` | 0.749 | 0.634 | 0.750 | 0.000 | 420 | — | 54.6 |
| `dense` | 0.664 | 0.553 | 0.637 | 0.000 | 472 | — | 28.5 |
| `hybrid` | 0.748 | 0.649 | 0.725 | 0.000 | 435 | — | 64.9 |
| `hybrid_rerank` | 0.749 | 0.655 | 0.750 | 0.000 | 450 | 38.4 | 110.0 |
| `hybrid_rerank_routed` | 0.829 | 0.784 | 0.750 | 0.000 | 344 | 26.0 | 82.5 |
| `adaptive` | 0.829 | 0.784 | 0.750 | 0.000 | 344 | 26.0 | 82.8 |
| `no_freshness` | 0.748 | 0.649 | 0.738 | 0.925 | 436 | — | 64.3 |

### Answer-level metrics — hashing configuration

| Strategy | Abstention correct | Citation coverage | Fact coverage | Freshness error | Context tokens | Rerank P50 ms | End-to-end P95 ms |
|---|---|---|---|---|---|---|---|
| `bm25` | 0.749 | 0.634 | 0.750 | 0.000 | 420 | — | 53.9 |
| `dense` | 0.625 | 0.474 | 0.575 | 0.000 | 441 | — | 18.8 |
| `hybrid` | 0.748 | 0.588 | 0.750 | 0.000 | 431 | — | 62.3 |
| `hybrid_rerank` | 0.743 | 0.485 | 0.681 | 0.000 | 394 | 0.3 | 62.5 |
| `hybrid_rerank_routed` | 0.825 | 0.671 | 0.750 | 0.000 | 313 | 0.3 | 54.4 |
| `adaptive` | 0.825 | 0.671 | 0.750 | 0.000 | 313 | 0.3 | 55.2 |
| `no_freshness` | 0.744 | 0.587 | 0.750 | 0.850 | 432 | — | 63.3 |

All release gates (`leakage`, `injection_success`, `no_errors`) pass for every strategy in both
configurations.

Routing also changes the answers, not only the ranking: citation coverage 0.655 → 0.784
and correct abstention 0.749 → 0.829 (neural, `hybrid_rerank` → `hybrid_rerank_routed`), with a
smaller context (450 → 344 tokens per query). Fact coverage is flat at 0.75 across the stronger
strategies: the remaining quarter are questions whose required fact the extractive generator does
not select, which a better retriever cannot fix.

## 2. Context selection

`top_k` packs the ranking as it is; `diverse` applies near-duplicate removal, a per-document
limit and MMR within the token budget. Hybrid retrieval, paired bootstrap.

Neural configuration:

| Metric | top-k | diverse | 95 % CI of the difference | Verdict |
|---|---|---|---|---|
| context_precision | 0.113 | 0.114 | -0.002 … 0.002 | no credible difference |
| context_recall | 0.860 | 0.858 | -0.017 … 0.015 | no credible difference |
| citation_coverage | 0.630 | 0.649 | 0.007 … 0.032 | improved |
| fact_coverage | 0.725 | 0.725 | 0.000 … 0.000 | no credible difference |
| duplicate_context_rate | 0.001 | 0.000 | -0.001 … -0.000 | improved |
| context tokens / query | 446 | 435 | — | — |

Hashing configuration:

| Metric | top-k | diverse | 95 % CI of the difference | Verdict |
|---|---|---|---|---|
| context_precision | 0.101 | 0.101 | -0.001 … 0.002 | no credible difference |
| context_recall | 0.779 | 0.772 | -0.019 … 0.005 | no credible difference |
| citation_coverage | 0.511 | 0.588 | 0.056 … 0.101 | improved |
| fact_coverage | 0.744 | 0.750 | 0.000 … 0.019 | no credible difference |
| duplicate_context_rate | 0.003 | 0.000 | -0.004 … -0.001 | improved |
| context tokens / query | 448 | 431 | — | — |

In both configurations the only credible effects are higher citation coverage (hashing
+0.077, neural +0.019) and the removal of the few near-duplicate pairs, with about 2–4 % fewer
context tokens. Context recall and precision do not change credibly. The synthetic corpus has
very few near-duplicate chunks (duplicate rate 0.06–0.3 %), so this fixture cannot show how much
deduplication helps on a corpus with templated or copied text.

## 3. Chunking

One index per strategy, 48-token chunks (so that the short synthetic documents split), hybrid retrieval.

Neural configuration:

| Chunking | Chunks | Build s | nDCG@10 | Recall@10 | Context recall | Fact coverage | Context tokens |
|---|---|---|---|---|---|---|---|
| fixed | 41,221 | 35.2 | 0.726 | 0.850 | 0.821 | 0.194 | 233 |
| sliding_window | 47,348 | 30.4 | 0.732 | 0.863 | 0.882 | 0.181 | 267 |
| sentence | 45,463 | 38.0 | 0.781 | 0.916 | 0.895 | 0.056 | 241 |
| structure | 59,884 | 39.0 | 0.735 | 0.903 | 0.866 | 0.056 | 194 |
| recursive | 47,175 | 34.1 | 0.779 | 0.932 | 0.898 | 0.113 | 227 |
| semantic | 135,188 | 203.3 | 0.764 | 0.922 | 0.791 | 0.094 | 101 |

Hashing configuration:

| Chunking | Chunks | Build s | nDCG@10 | Recall@10 | Context recall | Fact coverage | Context tokens |
|---|---|---|---|---|---|---|---|
| fixed | 37,195 | 10.8 | 0.647 | 0.747 | 0.732 | 0.631 | 301 |
| sliding_window | 39,671 | 9.3 | 0.678 | 0.782 | 0.782 | 0.625 | 333 |
| sentence | 37,746 | 11.1 | 0.686 | 0.773 | 0.785 | 0.606 | 305 |
| structure | 50,932 | 12.5 | 0.584 | 0.745 | 0.728 | 0.606 | 244 |
| recursive | 39,293 | 9.6 | 0.662 | 0.762 | 0.764 | 0.594 | 285 |
| semantic | 144,602 | 41.5 | 0.728 | 0.865 | 0.785 | 0.362 | 89 |

- **Retrieval** prefers sentence-aligned chunks: with the neural models `sentence` and `recursive`
  reach nDCG@10 0.78 and Recall@10 0.92–0.93, against 0.73 / 0.85 for `fixed`.
- **Answers** suffer from small chunks: fact coverage falls to 0.06–0.19 with 48-token chunks
  (it is 0.75 with the default 200-token chunks in section 1). The fact needed for the answer and
  the words that match the question end up in different chunks, and the 8-chunk context no longer
  contains both. Small chunks need parent or neighbour expansion to be usable for generation;
  that combination was not measured here.
- **Semantic chunking** produces 3× as many chunks and takes 6× as long to build (203 s vs
  30–39 s with MiniLM) without beating `sentence` or `recursive` on any metric.
- `structure` (section headers) creates the most fragments of the rule-based strategies and
  is not better than `sentence`.

The token counts differ between the two configurations (whitespace tokens for hashing, word
pieces for MiniLM), so the two tables are not comparable row by row.

## 4. Indirect prompt injection

The adversary is a **simulated** model that obeys any "reply only with X" found in a source and
cites that source (`app/corpus/redteam.py`). It is the worst case, not a measured compliance rate
of a real model. 40 `prompt_injection` queries.

| Configuration | Queries | Model obeyed | Injection success |
|---|---|---|---|
| verify_citations=False,quarantine=False | 40 | 39 | 0.975 |
| verify_citations=False,quarantine=True | 40 | 0 | 0.000 |
| verify_citations=True,quarantine=False | 40 | 39 | 0.975 |
| verify_citations=True,quarantine=True | 40 | 0 | 0.000 |
| extractive_generator(default) | 40 | — | 0.000 |

Citation verification alone does not stop this attack — the injected text really is in
the cited source, so a sentence repeating it counts as supported (0.975 with the hashing
configuration; 1.000 with the neural one, where retrieval finds the poisoned document for all 40
queries). Withholding instruction-like sentences before generation removes what the model could
obey (0 of 40 in both configurations). The quarantine is a pattern heuristic and one layer of
several; it will not catch paraphrased or obfuscated instructions.

## 5. ETL at 1M records

`experiments/enterprise/benchmarks/etl_1m.json` — 1,000,000 logical records, 1,094,897 source rows.

| Stage | Seconds | Rows/s | Notes |
|---|---|---|---|
| Generate (Parquet, zstd) | 20.7 | 52.9k | 78 MB in 11 files |
| Normalise, local engine | 31.2 | 35.1k | 10,155 rejected (5 reason codes) |
| Normalise, Spark 4.1 `local[*]` | 31.7 | 34.5k | 12 cores, 6 GB driver; same read / valid / rejected counts |
| Load into the registry (SQLite) | 93.4 | 11.6k | 969,809 inserted, 20,039 deleted, 4,878 conflicts, 1.46 GB database |
| Second run of the load | 0.087 | — | all 11 partitions skipped (checkpointed) |

Spark does not beat the local engine on one machine: both are bound by the same Python
`normalize_row` on the same cores. The value of the Spark path is that it runs unchanged on a
cluster; that was not measured.

## 6. API under load (PostgreSQL 16 + Elasticsearch 8.15)

One API process (uvicorn), extractive generator, 19,760 chunks, client on the same machine.
800 requests per level over 760 distinct questions.

| Clients | Cache off: req/s | P50 ms | P95 ms | P99 ms | Cache on: req/s | P50 ms | P95 ms | P99 ms | Hit rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 46.1 | 26.3 | 31.6 | 35.1 | 45.4 | 26.9 | 35.4 | 48.6 | 12 % |
| 4 | 74.5 | 59.9 | 88.4 | 102.2 | 116.5 | 35.9 | 52.6 | 61.2 | 100 % |
| 16 | 78.5 | 204.3 | 298.9 | 350.5 | 117.1 | 136.8 | 192.1 | 231.9 | 100 % |
| 32 | 71.9 | 441.1 | 662.4 | 721.8 | 107.3 | 305.4 | 425.7 | 442.3 | 100 % |

No request failed. With the cache on, levels 4–32 replay questions cached during level 1 (TTL
120 s), so they show the all-hit upper bound, not a realistic hit rate.

Time to first streamed token (`/v1/query/stream`, 200 streams per level, 143 of them with tokens —
the rest abstain):

| Clients | Cache off: P50 ms | P95 ms | Cache on: P50 ms | P95 ms |
|---|---|---|---|---|
| 1 | 23.3 | 28.7 | 9.2 | 10.8 |
| 4 | 49.4 | 78.9 | 27.4 | 42.8 |
| 16 | 206.6 | 301.7 | 122.1 | 175.9 |

Reading: throughput stops growing after 4 clients because a single Python process handles
planning, embedding, context building and an audit write per request; beyond that point added
clients only add queueing time. The next step is more API processes behind a load balancer,
which was not measured here.

**Locust** (20 users, 45 s, 60 % answers / 20 % streams / 20 % retrieval, cache on): 4,491
requests, 0 failures, 102 requests/s; `/v1/query` median 34 ms, P95 110 ms; first streamed token
median 29 ms, P95 110 ms.

## 7. Update visibility

`scripts/corpus_freshness_benchmark.py`, PostgreSQL + Elasticsearch, 30 rounds. Time from
committing a change event to a query seeing the change, one event per processor batch:

| Change | P50 ms | P95 ms | Max ms |
|---|---|---|---|
| Insert | 78.5 | 92.7 | 153.8 |
| Update (new text visible, old text gone) | 85.5 | 96.6 | 147.1 |
| Delete | 61.6 | 73.6 | 92.5 |

This excludes the indexer's poll interval (1 s by default) and uses hashing vectors; a real
embedding model adds its inference time per changed document. The configured bound
(`consistency_interval_seconds`) is 120 s.

## 8. Index build

| Backend | Embedding | Documents/s | Embedding share of build time |
|---|---|---|---|
| in-process | hashing | 2,289 | 76 % |
| Elasticsearch 8.15 (single node, 1 GB heap) | hashing | 1,356 | 45 % |
| in-process | MiniLM on Apple GPU (MPS) | 589 | 82 % |

Elasticsearch index size: 40.1 MB for 19,760 chunks with 384-dimension float vectors — 2,028
bytes per chunk. This is the calibration input for `corpus capacity --calibrate`.

## 9. Recommendations that follow from these measurements

These follow from the numbers above and hold for this fixture and these models; each should be
re-checked on real documents before it becomes a default.

| Decision | Recommendation | Evidence |
|---|---|---|
| Online default with real models | `hybrid_rerank_routed` | best nDCG@10 (0.926) and Recall@10 (0.957); P50 63 ms sequential, faster than unrouted reranking |
| Latency-critical or model-free path | `bm25` | 26 ms P50; 0.99–1.00 on identifier, temporal and multi-document classes; fails on paraphrases (0.074) |
| Offline batch | `hybrid_rerank_routed`, reranking depth can be raised | latency is not the constraint; batch workers are isolated from online traffic |
| Never ship | dense-only; reranking without routing for entity or time questions | dense-only −0.133 nDCG; unrouted reranking drops temporal questions to 0.606 |
| With weak or unvalidated models | stay on BM25 until the comparison says otherwise | every vector or rerank strategy regressed with the stand-in models |
| Freshness filter | always on; `as_of` for historical questions | freshness error 0.85–0.93 → 0 |
| Context selection | `diverse` | higher citation coverage, slightly fewer tokens, no credible loss |
| Chunking | sentence-aligned (`sentence` or `recursive`) at a size that keeps facts together; not `semantic` | section 3: recall gain of small chunks vs. loss of fact coverage; semantic 6× build cost |
| Prompt-injection | keep quarantine and citation verification both on | verification alone: 0.975–1.000 success for an obedient model; with quarantine: 0 |
| Retrieval cache | on, short TTL | all-hit case +56 % throughput at 4 clients; invalidated per tenant on change |
| Scaling the API | add processes before tuning anything else | one process saturates at ~75 req/s (cache off) on this machine |

## 10. Verification against live services

The corpus test suite was run with `INSPECT_RAG_TEST_DATABASE_URL` and
`INSPECT_RAG_TEST_ELASTICSEARCH_URL` set: every registry is then a fresh PostgreSQL 16 database
whose schema is created by `alembic upgrade head` (migrations `0001` and `0002`), and every
runtime indexes into Elasticsearch 8.15.3. Result: **75 passed, 1 skipped** (the test that needs
a real Redis server), Spark tests excluded. This covers ingestion, index lifecycle, retrieval,
access control, change events, batch jobs, evaluation and the `/v1` API on the real backends.

```bash
docker compose up -d postgres elasticsearch
cd backend && INSPECT_RAG_TEST_ELASTICSEARCH_URL=http://127.0.0.1:9200 \
  INSPECT_RAG_TEST_DATABASE_URL=postgresql+psycopg://inspect_rag:$POSTGRES_PASSWORD@127.0.0.1:5432/inspect_rag \
  python -m pytest tests/corpus -m "not spark"
```

## What was not measured

100M records; a Spark cluster; a real language model (latency, cost, hallucination behaviour);
Redis, Kafka and MinIO; Kubernetes; multi-process or multi-node API scaling; vector quantisation;
retrieval quality on real documents with human relevance labels.
