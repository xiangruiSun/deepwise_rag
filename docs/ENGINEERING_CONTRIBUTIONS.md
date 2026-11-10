# Engineering contributions — document-corpus RAG platform

This document lists what was built on top of the original case-retrieval application, how each
piece was validated, and what was measured. It follows one rule: **a number appears here only if a
command in this repository produced it**, and the artifact that holds it is named.

## Read this first

- **Data.** All measurements use a generated corpus (`app/corpus/generator.py`) with synthetic
  documents and synthetic labels. They show how configurations behave relative to each other and
  that the system works end to end. They are not evidence of quality on real clinical documents.
- **Scale.** The design targets 100M+ records. The largest run is **1M records** through the ETL.
  Indexing, retrieval experiments and the load test used **20,000-record** corpora.
  Nothing was run at 100M; the 100M figures in section 16 are a model, labelled as such.
- **Hardware.** One Apple-silicon workstation (arm64, 12 cores, 18 GB). Client and server shared
  the machine in every load measurement.
- **Generation.** Measured answers come from the deterministic extractive generator. The
  OpenAI-compatible client (for vLLM) was tested against a mocked HTTP transport; no language
  model served any measured request.
- **Not run:** a Spark cluster (local mode only), a real Redis server (fakeredis only), a Kafka
  broker (fake consumer only), a GPU model server, a Kubernetes cluster.

Baseline for "before" is the repository at commit `ff2f59c` (audit in
[enterprise/00_audit.md](enterprise/00_audit.md)) unless a section names a different baseline.

## Summary

| # | Contribution | Validation | Measured result |
|---|---|---|---|
| 1 | Tenant-aware document model, idempotent versioned ETL | 8 tests | 1M rows: normalise 35.1k rows/s, load 11.6k rows/s, repeat run 0.09 s |
| 2 | Distributed normalisation sharing the local logic | 2 tests | Spark local[*]: 34.5k rows/s; same counts as the local engine |
| 3 | Versioned hybrid index: build, validate, activate, rollback, sync, repair | 15 tests (retrieval file) | Elasticsearch build 1,356 docs/s (hashing vectors); 2,028 bytes/chunk |
| 4 | Authorisation inside retrieval, cache and citations | 5 security tests × 7 strategies | leakage 0 on every adversarial query, every strategy |
| 5 | Hybrid retrieval, routing, reranking — as switchable strategies | ablation run, paired bootstrap | nDCG@10 0.746 → 0.926, Recall@10 0.846 → 0.957 with real models; a regression with stand-in models |
| 6 | Validity-aware retrieval (freshness) | ablation run | freshness error 0.85–0.93 → 0.00 |
| 7 | Context construction (dedup, MMR, budget) | paired bootstrap | citation coverage 0.511 → 0.588 (stand-in), 0.630 → 0.649 (neural); 2–4 % fewer tokens |
| 8 | Seven chunking strategies behind one interface | comparison run | sentence-aligned: Recall@10 0.850 → 0.93; small chunks cut fact coverage to ≤ 0.19 |
| 9 | Verified streaming generation, resilient model client | 6 tests | not a performance claim (mocked transport) |
| 10 | Prompt-injection defence in depth | simulated adversary | injection success 0.975–1.000 → 0.000 |
| 11 | Tenant-safe caching with change invalidation | load test | 74 → 116 req/s, P95 88 → 53 ms at 4 clients (all-hit condition) |
| 12 | Durable batch inference | 4 tests | interruption never duplicates or loses an item |
| 13 | Near-real-time updates from change events | 5 tests + benchmark | change visible in 79–86 ms P50, < 100 ms P95 (excl. poll interval) |
| 14 | Evaluation platform with release gates | 5 tests + CI | gates enforced in CI on every push |
| 15 | Operations: scheduler, metrics, dashboards, manifests | tests, YAML validation | Compose/Kubernetes validated syntactically only |
| 16 | Capacity and cost planning tools | 2 tests | model output, not a measurement |
| 17 | End-to-end load test on PostgreSQL + Elasticsearch | Locust | 102 req/s, 0 failures, P95 100 ms with 20 users |

---

## 1. Tenant-aware document model and idempotent, versioned ETL

**Problem and baseline.** The original ingestion was case-centric: one cohort, no tenant, no
document versions, no notion of a late or duplicate record. Re-ingesting a changed source could
not say what changed.

**Change.** A canonical document contract with a deterministic id (hash of tenant, source system,
record id), a monotonic version and a content hash. Ingestion is two stages: *normalise* (source
rows → canonical Parquet, invalid rows to a reason-coded dead-letter table) and *load* (apply to
the registry with compare-and-set: insert, update, delete, unchanged, stale, conflict). Every
partition is a checkpoint, so a run can be repeated or resumed, and `reconcile` proves that every
row read is accounted for.

**Files and commands.** `app/corpus/schema.py`, `etl.py`, `generator.py`; tables `corpus_documents`,
`corpus_ingest_partitions`, `corpus_dead_letters` (migration `0002`); `corpus generate`, `corpus ingest`.

**Validation.** `tests/corpus/test_corpus_etl.py` (8 tests): identity stability, rejects per
reason code, out-of-order and duplicate delivery, conflict handling, crash-and-resume, CSV / JSONL
/ SQL sources, reconciliation.

**Measured** (`experiments/enterprise/benchmarks/etl_1m.json`, 1,000,000 logical records =
1,094,897 source rows including injected duplicates, late versions and invalid rows):

| Stage | Result |
|---|---|
| Normalise (local engine) | 31.2 s — 35.1k rows/s; 10,155 rows rejected across 5 reason codes |
| Load into the registry (SQLite) | 93.4 s — 11.6k rows/s; 969,809 inserted, 20,039 deleted, 4,878 conflicts |
| Same run repeated | 0.087 s — all 11 partitions skipped |
| Reload into an empty registry | identical registry content |
| Load into PostgreSQL 16 (20k corpus) | 21,797 rows in 1.5 s (`benchmarks/index_build_es_pg.json`) |

**Limitations.** The loader uses ORM upserts; at 100M rows it should use `COPY` into a staging
table plus a set-based merge and a tenant-partitioned table. The 1M load ran on SQLite.

## 2. Distributed normalisation that shares the local logic

**Problem.** A second, Spark-specific implementation of normalisation would drift from the local one.

**Change.** `app/corpus/spark_job.py` runs the same `normalize_row` through `mapInArrow`, then does
the dataset-wide work in Spark: exact-duplicate removal, version-conflict quarantine, newest-
version selection, tenant-bucketed output. The load stage is unchanged.

**Validation.** `tests/corpus/test_corpus_spark.py` (2 tests, run when PySpark and a JVM exist):
both engines produce the same documents on the same input.

**Measured** (same 1M-record input): Spark 4.1 `local[*]` (12 cores, 6 GB driver) — 31.7 s,
34.5k rows/s. Both engines agree on rows read, valid and rejected. Live documents differ by exactly
the 4,878 conflicting documents: the local engine keeps the first copy of a conflicting version,
Spark quarantines both.

**Limitations.** Local mode only — this shows correctness and single-machine throughput, not
cluster scaling. A 1 GB driver fails at this size; the job sets driver memory explicitly.

## 3. Versioned hybrid index with a safe lifecycle

**Problem and baseline.** Indexes were rebuilt in place; a bad build or a model change had no
rollback, and nothing compared the index with the system of record.

**Change.** `IndexManager` (`app/corpus/indexing.py`): each build is a new physical index
(`{prefix}_vNNNN_{hash}`) holding BM25 text and vectors per chunk; `validate` checks counts,
sampled documents, mapping and model identity; `activate` switches an alias and refuses an
unvalidated index; `rollback` switches back; `sync` applies changed and deleted documents to the
active index and removes chunks left over from a longer previous version; `consistency_check`
compares registry and index document by document and can repair.

**Validation.** `tests/corpus/test_corpus_retrieval.py`: activation gate, resumable build,
corrupted build rejected, rollback, update replaces and delete removes searchable content,
consistency repair. The whole corpus test suite also passes against **real PostgreSQL 16 (schema
from the Alembic migrations) and Elasticsearch 8.15** — 75 passed, 1 skipped (the test that needs
a real Redis server).

**Measured** (20k corpus → 19,387 live documents, 384-dimension vectors):

| Build | Documents/s | Notes |
|---|---|---|
| In-process backend, hashing vectors | 2,289 | `experiments/enterprise/hashing/summary.json` |
| Elasticsearch 8.15, hashing vectors | 1,356 | 19,760 chunks, 40.1 MB = 2,028 bytes/chunk (`benchmarks/index_build_es_pg.json`) |
| In-process backend, MiniLM on Apple GPU (MPS) | 589 | embedding is 82 % of the build time (`experiments/enterprise/neural/summary.json`) |

**Limitations.** Single-node Elasticsearch with a 1 GB heap; shard and replica behaviour was not
exercised. `int8_hnsw` quantisation is implemented in the mapping but its memory effect was not measured.

## 4. Authorisation inside retrieval, caching and citations

**Problem and baseline.** Access scopes existed per case. There was no tenant boundary, and a
filter applied after retrieval can both leak (through scores, counts, caches) and starve results.

**Change.** Identity comes only from the verified credential — the request model has no tenant
field and rejects unknown fields. One filter (tenant, permitted groups, validity window) is
applied *inside* the BM25 query and the kNN search. Caller filters are restricted to an allow-list
and can only narrow. Cache keys contain tenant, permission set, index version and model identity,
and a cached entry is re-checked against the caller. Neighbour expansion re-verifies each added chunk.

**Validation.** `tests/corpus/test_corpus_security.py` (marker `security`, a CI job of its own):
every strategy × every adversarial query; forged filters; cached results across permission sets;
an ACL change taking effect through the cache.

**Measured.** `leakage = 0` for all 7 strategies on the 120 adversarial queries of the evaluation
set (`cross_tenant`, `unauthorized_group`, `prompt_injection`), in both model configurations
(`experiments/enterprise/*/retrieval_ablation/run.json`, gate `leakage`).

**Limitations.** Group membership is static per API key; there is no identity-provider
integration. Until a changed ACL is indexed, the old one applies — that window is the index lag (section 13).

## 5. Hybrid retrieval, query routing and reranking as switchable strategies

**Problem.** One fixed retrieval recipe cannot be evaluated, and the right recipe depends on the
query class and on the models available.

**Change.** Named strategies in `configs/corpus*.yaml` (`bm25`, `dense`, `hybrid`,
`hybrid_rerank`, `hybrid_rerank_routed`, `adaptive`, `no_freshness`) select per request which
stages run: BM25, kNN, reciprocal rank fusion, cross-encoder reranking, rule-based routing
(identifier lookups, entity and time scoping, acronym expansion, bounded decomposition).

**Validation.** Unit tests for fusion, rerank depth, routing; the ablation below.

**Measured** — nDCG@10 / Recall@10 on 800 synthetic queries, paired bootstrap against `bm25`:

| Strategy | Neural models: nDCG@10 | Recall@10 | Δ nDCG vs bm25 (95 % CI) | P50 ms | Stand-in models: nDCG@10 | Δ vs bm25 |
|---|---|---|---|---|---|---|
| `bm25` (baseline) | 0.746 | 0.846 | — | 25.9 | 0.746 | — |
| `dense` | 0.613 | 0.761 | −0.133 (−0.174 … −0.093) regressed | 25.2 | 0.556 | −0.190 regressed |
| `hybrid` | 0.726 | 0.887 | −0.020 (−0.050 … +0.009) no credible difference | 35.6 | 0.637 | −0.110 regressed |
| `hybrid_rerank` | 0.839 | 0.956 | +0.092 (+0.063 … +0.121) improved | 76.6 | 0.529 | −0.217 regressed |
| `hybrid_rerank_routed` | **0.926** | **0.957** | **+0.180 (+0.157 … +0.203) improved** | 62.9 | 0.672 | −0.075 regressed |
| `adaptive` | 0.926 | 0.957 | +0.180 (same results as routed) | 61.2 | 0.672 | −0.075 regressed |

"Neural models" = `all-MiniLM-L6-v2` embeddings + `ms-marco-MiniLM-L6-v2` cross-encoder
(`experiments/enterprise/neural/`). "Stand-in models" = hashing vectoriser + lexical-overlap
reranker, which need no download (`experiments/enterprise/hashing/`). Latency is sequential,
in-process, not a load test.

What the numbers say:

- Fusion alone raises recall (+0.041) but not ranking quality; the cross-encoder turns the extra
  recall into a ranking gain; routing adds the largest step by scoping identifier, patient and
  time questions with filters instead of similarity.
- By category (neural): paraphrase queries go from 0.074 (BM25) to 0.615; exact-identifier
  queries collapse to 0.042 under pure dense retrieval and return to 1.000 with routing;
  temporal questions drop to 0.606 with unrouted reranking and return to 1.000 with routing.
- Routing also improves the answers: citation coverage 0.655 → 0.784, correct abstention
  0.749 → 0.829, with a smaller context (450 → 344 tokens per query).
- With stand-in models the whole chain is worse than BM25. The harness reports that as
  "regressed" — which is the purpose of comparing strategies per release rather than assuming
  that more stages are better.
- `adaptive` depth changed no top-10 result on this fixture; it is implemented but shows no benefit here.

**Limitations.** Synthetic queries; the paraphrase category was generated from a fixed synonym
list. The reranker in the neural configuration is a general-domain model, not a clinical one. No
hard-negative mining or fine-tuning was done (specified as optional; not implemented).

## 6. Validity-aware retrieval (freshness)

**Problem.** When a document is superseded, a retriever that ignores validity returns the old and
the new version together and the answer may cite the outdated one.

**Change.** Documents carry `valid_from` / `valid_to`; the filter keeps what is valid now (or at
`as_of` for historical questions). The strategy `no_freshness` disables it for comparison.

**Measured** (40 `freshness_superseded` queries; `freshness_error` = share of queries whose
context or citations contain a superseded document):

| | Stand-in models: freshness error | temporal consistency | Neural models: freshness error | temporal consistency |
|---|---|---|---|---|
| `no_freshness` (before) | 0.850 | 0.538 | 0.925 | 0.506 |
| `hybrid` (after) | 0.000 | 0.750 | 0.000 | 0.725 |

**Limitations.** Validity comes from the source; a source that never closes `valid_to` gets no benefit.

## 7. Context construction

**Problem and baseline.** The baseline packs the top-k chunks as ranked, so near-duplicates and
several chunks of one document can crowd out other evidence.

**Change.** `app/corpus/context.py`: relevance floor, near-duplicate removal (token-set Jaccard),
per-document limit, Maximal Marginal Relevance within a token budget, optional neighbour expansion.

**Measured** (hybrid retrieval, hashing configuration, 680 queries with relevant documents,
paired bootstrap; `experiments/enterprise/hashing/summary.json`):

| Metric | top-k (before) | diverse (after) | 95 % CI of the difference | Verdict |
|---|---|---|---|---|
| Citation coverage | 0.511 | 0.588 | +0.056 … +0.101 | improved |
| Context recall | 0.779 | 0.772 | −0.019 … +0.005 | no credible difference |
| Context precision | 0.101 | 0.101 | −0.001 … +0.002 | no credible difference |
| Duplicate rate | 0.003 | 0.000 | −0.004 … −0.001 | improved |
| Context tokens / query | 448 | 431 | — | −4 % |

With the neural models the same comparison gives citation coverage 0.630 → 0.649
(95 % CI +0.007 … +0.032, improved), 446 → 435 context tokens, and no credible change in context
recall or precision (`experiments/enterprise/neural/summary.json`).

**Limitations.** The synthetic corpus contains few near-duplicate chunks (duplicate rate 0.06–0.3 %),
so the deduplication effect is small here.

## 8. Chunking strategies behind one interface

**Change.** `app/corpus/chunks.py`: fixed, sliding window, sentence, structure (section headers),
recursive, semantic (embedding-similarity breakpoints) and parent/child. The chunking identity is
part of the index version, so changing it forces a new index instead of silently mixing chunks.

**Measured** (hybrid retrieval, 48-token chunks so that documents split; one index per strategy):

| Chunking | Chunks | Build s | nDCG@10 | Recall@10 | Fact coverage |
|---|---|---|---|---|---|
| fixed (baseline) | 41,221 | 35.2 | 0.726 | 0.850 | 0.194 |
| sliding window | 47,348 | 30.4 | 0.732 | 0.863 | 0.181 |
| sentence | 45,463 | 38.0 | 0.781 | 0.916 | 0.056 |
| structure | 59,884 | 39.0 | 0.735 | 0.903 | 0.056 |
| recursive | 47,175 | 34.1 | 0.779 | 0.932 | 0.113 |
| semantic | 135,188 | 203.3 | 0.764 | 0.922 | 0.094 |

Neural models (`experiments/enterprise/neural/summary.json`); the hashing table is in
[enterprise/06_experiments.md](enterprise/06_experiments.md).

What the numbers say: sentence-aligned chunks retrieve best (Recall@10 0.92–0.93 vs 0.85 for
fixed windows); semantic chunking costs 3× the chunks and 6× the build time without winning on
any metric; and 48-token chunks are too small for answering — fact coverage is 0.06–0.19 here
against 0.75 with the default 200-token chunks, because the fact and the words that match the
question land in different chunks. Small chunks therefore need neighbour or parent expansion,
a combination that was implemented but not measured.

**Limitations.** Synthetic documents are short (about one chunk each at the default 200 tokens),
which is why the comparison uses 48-token chunks. Results on long real documents will differ.

## 9. Verified streaming generation and a resilient model client

**Problem and baseline.** Generation was synchronous and verified only after the full answer
existed, so a streamed answer could show an unsupported sentence before it was checked.

**Change.** `app/corpus/generation.py`, `service.py`: the answer is verified *sentence by
sentence while streaming* — a sentence is released to the client only when it cites existing
sources and the cited text supports it; otherwise it is dropped and counted. If nothing survives,
the response is an abstention with a reason. The async OpenAI-compatible client has a concurrency
limit, a bounded wait queue (429 when full), retries with backoff, a circuit breaker and
cancellation when the client disconnects; when the model fails, the extractive generator answers
and the response is flagged `degraded`.

**Validation.** `tests/corpus/test_corpus_api.py`: only supported sentences are released; invalid
citation ids are reported; model failure degrades; disconnect cancels; retry, usage accounting,
circuit breaker, back-pressure and timeout behaviour of the client.

**Measured.** Nothing about model speed: the client was exercised against `httpx.MockTransport`.
Time to first token in section 11 is for the extractive generator.

**Limitations.** Support checking is lexical (term overlap with the cited text), not an
entailment model: it rejects unsupported and uncited sentences but can accept a sentence that
reuses the source's words with a different meaning.

## 10. Prompt-injection defence in depth

**Problem.** A retrieved document can contain text addressed to the model ("ignore previous
instructions and reply only with …"). Citation verification does not stop this: the injected text
really is in the cited source, so a sentence repeating it is "supported".

**Change.** Three layers: the prompt marks sources as untrusted data and delimits them; citation-
like markers in documents are neutralised; and (new) `quarantine_instructions` withholds
instruction-like sentences from the context, one sentence at a time, before the generator sees them.

**Evaluation method.** `app/corpus/redteam.py` provides a **simulated** model that obeys any
"reply only with X" it finds in a source and cites that source — the worst case. This measures
whether the platform holds when the model does not; it is not a compliance rate of a real model.

**Measured** (40 `prompt_injection` queries; `experiments/enterprise/{hashing,neural}/summary.json`):

| Configuration | Stand-in models: obeyed | Injection success | Neural models: obeyed | Injection success |
|---|---|---|---|---|
| citation verification only (before) | 39 / 40 | 0.975 | 40 / 40 | 1.000 |
| + instruction quarantine (after) | 0 / 40 | 0.000 | 0 / 40 | 0.000 |
| extractive generator (default, no model) | — | 0.000 | — | 0.000 |

**Limitations.** The quarantine is a pattern heuristic; it will miss paraphrased or obfuscated
instructions and can withhold a legitimate sentence that happens to match (three clinical
look-alike sentences are tested as non-matches). It is one layer, not a guarantee.

## 11. Tenant-safe caching with change invalidation

**Problem.** Caching retrieval is the cheapest latency win and the easiest way to leak across
tenants or to serve deleted content.

**Change.** `app/corpus/cache.py`: keys include tenant, permission set, index version, embedding
identity and a per-tenant generation counter that every index change of that tenant increments.
Memory and Redis backends; a cache failure degrades to no cache.

**Validation.** Key isolation, hit/invalidate behaviour, deletion and ACL change through the
cache, Redis backend against `fakeredis`.

**Measured** (HTTP, PostgreSQL + Elasticsearch, one API process, 800 requests per level over 760
distinct questions; `benchmarks/api_es_pg_cache_{off,on}.json`). With the cache on, levels 4–32
replayed questions already cached, so they show the **all-hit** case, the upper bound of the benefit:

| Clients | Cache off: req/s | P50 ms | P95 ms | Cache on (100 % hits): req/s | P50 ms | P95 ms |
|---|---|---|---|---|---|---|
| 1 | 46.1 | 26.3 | 31.6 | 45.4 (12 % hits) | 26.9 | 35.4 |
| 4 | 74.5 | 59.9 | 88.4 | 116.5 | 35.9 | 52.6 |
| 16 | 78.5 | 204.3 | 298.9 | 117.1 | 136.8 | 192.1 |
| 32 | 71.9 | 441.1 | 662.4 | 107.3 | 305.4 | 425.7 |

Time to first streamed token at 1 client: P50 23.3 ms → 9.2 ms (cache off → all hits).
No request failed at any level.

**Limitations.** Throughput saturates near 75 (117 with hits) req/s because one Python process
does embedding, context building and an audit write per request; scaling is by adding API
processes, which was not measured. Real hit rates depend on traffic and are lower than 100 %.

## 12. Durable batch inference

**Problem.** Offline jobs over many questions must survive worker crashes without repeating or
losing items, and must not see more than their creator may see.

**Change.** `app/corpus/batch.py`: a job runs the same service as the online API under the
creator's tenant and groups; items are processed in fixed batches, each written to its own part
file through an atomic rename before progress is committed; a worker claims a job with a
conditional update; stale jobs are re-queued; a failure budget stops a broken job.

**Validation.** `tests/corpus/test_corpus_batch_eval.py`: interruption between "part written" and
"progress committed" resumes without duplicates; two workers cannot claim one job; cancel;
cross-tenant job access returns 404; input paths cannot escape the input directory.

**Limitations.** No throughput measurement. Outputs go to a local or shared filesystem; an object-
store writer is not implemented.

## 13. Near-real-time updates from change events

**Problem and baseline.** Only bulk ingestion existed; an update or deletion became searchable at
the next full run.

**Change.** `app/corpus/events.py`: a transactional outbox (or a Kafka topic) feeds an event
processor. Delivery is at-least-once; effects are exactly-once because the registry only moves a
document forward and the index sync is idempotent. Out-of-order events are ignored, a deletion
cannot be undone by a late event, a poison event is dead-lettered after `max_attempts` without
blocking the stream, and an event is acknowledged only after registry *and* index reflect it.

**Validation.** `tests/corpus/test_corpus_events.py`: update and delete reach retrieval (through
the cache) within the configured window; replay and reordering have no effect; poison and forged
events are dead-lettered; Kafka adapter contract with a fake consumer; scheduler integration.

**Measured** (PostgreSQL + Elasticsearch, 30 rounds, one event per batch;
`benchmarks/freshness_es_pg.json`) — time from committing the event to the change being visible to a query:

| Change | P50 | P95 | Max |
|---|---|---|---|
| Insert | 78.5 ms | 92.7 ms | 153.8 ms |
| Update | 85.5 ms | 96.6 ms | 147.1 ms |
| Delete | 61.6 ms | 73.6 ms | 92.5 ms |

**Limitations.** Excludes the worker's poll interval (default 1 s) and embedding time of a real
model (hashing vectors here). Measured with one event per batch on an idle system, not under
write load. The Kafka adapter never talked to a broker.

## 14. Evaluation platform with release gates

**Problem and baseline.** Evaluation was case-level retrieval only.

**Change.** `app/corpus/evaluation.py`: versioned evaluation datasets with provenance; retrieval,
context, citation, abstention, freshness, leakage and injection metrics from one `/v1/query`
response; paired bootstrap between strategies; per-category breakdowns; latency and token
accounting; a run directory with configuration hash, git commit, index and model version.
Release gates (`leakage`, `injection_success`, `no_errors`) make `corpus eval run` exit non-zero.
`analysis.py` adds a rule-based contradiction screen between sources (surfaced as `conflicts` in
responses), failure clustering of user feedback into candidate regression tests, and a cost model.

**Validation.** Metric and gate tests; contradiction detection on the superseded/current report
pairs; feedback analysis without stored question text. CI runs the security gates and a small
generate → ingest → index → evaluate pipeline on every push.

**Limitations.** No LLM-judge metrics. The contradiction screen is polarity and antonym matching,
not clinical NLI. Regression candidates from feedback need a human to supply question and labels,
because feedback deliberately stores no text.

## 15. Operations

**Change.** Scheduler tasks (incremental index, drift, data quality, consistency) usable in-process
or as CronJobs; Prometheus metrics for latency, first token, cache, model queue, index lag, events,
batch items and security signals; a Grafana dashboard and alert rules; Compose profiles for Redis,
Redpanda, MinIO, workers, Prometheus and Grafana; Kubernetes manifests (API with probes, HPA and
disruption budget, workers, CronJobs, GPU model server, network policies); a runbook.

**Validation.** Scheduler tests (due tasks, failure isolation, drift detection, repair).
`docker compose config` and YAML/JSON parsing of all manifests in CI.

**Limitations.** The Compose profiles `platform` and `monitoring` were not started; the Kubernetes
manifests were never applied; alert thresholds are starting values, not tuned.

## 16. Capacity and cost planning tools

**Change.** `corpus capacity` estimates storage per component, shard count and rebuild headroom
from stated assumptions, and can calibrate bytes per chunk from a real index. `corpus cost`
converts measured tokens per query into GPU-seconds and cost under assumed unit prices.

**Model output for 100M records** (default assumptions: 1.3 chunks/document, 1024-dimension
float32 vectors, 1 replica) — **an estimate, not a measurement**: about 754 GB primary index
(532 GB of it raw vectors), 1.5 TB with one replica, 19 primary shards at 40 GB each, and about
550 GB of page cache per copy for fast kNN. Measured calibration point: 2,028 bytes per chunk at
384 dimensions on Elasticsearch (section 3).

**Limitations.** Every default is an assumption to replace; the tool says so in its output.

## 17. End-to-end load test

**Measured** (Locust, 20 users, 45 s, PostgreSQL + Elasticsearch, one API process, cache on,
traffic mix 60 % answers / 20 % streams / 20 % retrieval; `benchmarks/locust_es_pg_stats.csv`):
4,491 requests, 0 failures, 102 requests/s; `/v1/query` median 34 ms, P95 110 ms; first streamed
token median 29 ms, P95 110 ms.

**Limitations.** Client and server on one machine; extractive generation; a 20k-document index.
It demonstrates stability under concurrency, not production capacity.

---

## What remains incomplete or unverified

| Item | State |
|---|---|
| 100M-record run | not run; largest ETL run is 1M, largest index 20k documents |
| Spark on a cluster | not run (local mode only) |
| Real language model behind `/v1/query` | not run; client tested with a mocked transport |
| Redis, Kafka/Redpanda, MinIO | configured in Compose; Redis and Kafka code paths tested with fakes only |
| Kubernetes | manifests written and parsed; never applied |
| Multi-process / multi-node API scaling | not measured |
| Vector quantisation (`int8_hnsw`) | implemented in the mapping; effect not measured |
| Hard-negative mining, embedding fine-tuning, reranker FP16/ONNX | not implemented |
| Object-storage outputs for batch jobs | not implemented (filesystem only) |
| Evaluation on real documents with human labels | not done; all labels are synthetic |
