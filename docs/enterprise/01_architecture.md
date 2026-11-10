# Corpus platform — architecture

The document-corpus platform (`backend/app/corpus`, HTTP API under `/v1`) extends the original
case-retrieval application. It reuses the existing search-backend interface, embedding providers
and cache, rerankers, citation verification, metrics and API-key authentication; it adds tenant-
aware documents, versioned ingestion and indexing, hybrid retrieval, streaming generation, batch
inference, an evaluation platform and operations tooling.

**Scale statement.** The design targets 100M+ source records. What was actually run is listed in
[06_experiments.md](06_experiments.md): 1M records through the ETL (local and Spark local mode) and
20k–1M-record corpora through indexing and retrieval on one workstation. Nothing at 100M was run.

## Components

```mermaid
flowchart LR
  subgraph Sources
    S1[(Parquet / JSONL / CSV / SQL)]
    S2[[Change events<br/>outbox table or Kafka topic]]
  end
  subgraph Ingestion
    N[Normalise<br/>local engine or PySpark]
    L[Load<br/>idempotent, checkpointed]
    DL[(Dead letters<br/>reason-coded)]
    EV[Event processor<br/>at-least-once, ordered by version]
  end
  REG[(Document registry<br/>PostgreSQL — system of record)]
  subgraph Indexing
    IM[Index manager<br/>build · validate · activate · rollback · sync]
    EMB[Embedding provider<br/>+ content-hash cache]
  end
  IDX[(Hybrid chunk index vN<br/>BM25 + kNN, alias → current)]
  subgraph Serving
    API[/v1 API<br/>query · stream · retrieve · feedback/]
    RET[Retriever<br/>ACL filter → BM25 ∥ kNN → RRF → rerank]
    CTX[Context builder<br/>dedup · MMR · budget · quarantine]
    GEN[Generator<br/>extractive | OpenAI-compatible | mock]
    VER[Sentence verifier]
    CACHE[(Cache<br/>memory or Redis)]
  end
  subgraph Offline
    BW[Batch workers]
    EVAL[Evaluation tracker<br/>release gates]
    SCH[Scheduler / CronJobs]
  end
  S1 --> N --> L --> REG
  N --> DL
  S2 --> EV --> REG
  REG --> IM --> IDX
  EMB --> IM
  API --> RET --> IDX
  RET --> CTX --> GEN --> VER --> API
  RET <--> CACHE
  IM -. invalidates tenant .-> CACHE
  BW --> RET
  EVAL --> RET
  SCH --> IM
  SCH --> EVAL
```

| Component | Module | Responsibility |
|---|---|---|
| Contracts | `schema.py` | `CanonicalDocument`, `ChunkRecord`, deterministic document and chunk ids |
| ETL | `etl.py`, `spark_job.py` | normalise → load; rejects with reason codes; partition checkpoints; reconciliation |
| Registry | `db/models.py` (`corpus_*` tables) | one row per document: current version, hash, deletion, what each index has |
| Chunking | `chunks.py` | fixed, sliding window, sentence, structure, recursive, semantic, parent/child |
| Index manager | `indexing.py` | versioned physical indexes behind an alias; incremental sync; consistency check |
| Change events | `events.py` | outbox / Kafka source, idempotent ordered processing, dead letters, lag metric |
| Query planning | `router.py` | query class, acronym expansion, bounded decomposition, entity and time scoping |
| Retrieval | `retrieval.py` | permission pre-filter, BM25 and kNN, RRF, optional rerank, adaptive depth |
| Context | `context.py` | relevance floor, near-duplicate removal, per-document limit, MMR, token budget, instruction quarantine |
| Generation | `generation.py` | extractive answerer, async OpenAI-compatible client, sentence verifier, circuit breaker |
| Service | `service.py` | the one pipeline used by the API, batch jobs and evaluation |
| Cache | `cache.py` | tenant- and permission-scoped keys, per-tenant generation counter |
| Batch | `batch.py` | durable resumable offline jobs under the creator's permissions |
| Evaluation | `evaluation.py`, `analysis.py`, `redteam.py` | metrics, paired bootstrap, release gates, contradiction screen, feedback analysis, cost model |
| Operations | `scheduler.py`, `capacity.py`, `telemetry.py` | maintenance tasks, capacity estimate, metrics and trace spans |

## Request path (online)

1. **Authenticate.** The API key resolves to a principal with a tenant, a role and groups. The
   request body has no tenant or user field (`extra="forbid"`), so identity cannot be supplied by
   the caller.
2. **Plan.** The router classifies the question (identifier lookup, keyword, semantic, temporal,
   multi-document, ambiguous), expands known acronyms, extracts entity and date scopes, and bounds
   the number of sub-queries.
3. **Filter first.** One `FilterSpec` — tenant, permitted groups, validity window, caller filters
   restricted to an allow-list — is applied *inside* both the BM25 query and the kNN search. There
   is no post-filtering step that could leak or starve results.
4. **Retrieve and fuse.** BM25 and kNN run in parallel; rankings are fused by reciprocal rank
   fusion; an optional cross-encoder reranks the top `rerank_depth` candidates.
5. **Build context.** Near-duplicates are dropped, a per-document limit and a token budget are
   applied, MMR spends the budget on distinct evidence. Citation-like markers in document text are
   neutralised and instruction-like sentences are withheld (`[instruction-like text withheld]`).
6. **Generate and verify.** Sources are given to the generator as `[S1] … [Sn]`. Output is split
   into sentences as it streams; a sentence is released only if it cites existing sources and the
   cited text supports it. Unsupported sentences are counted and dropped; if nothing survives, the
   answer is an abstention with a reason.
7. **Respond.** Answer, citations (document id, version, excerpt), model version, index version,
   strategy, trace id, per-stage timings, token usage, detected source conflicts.

The same `CorpusService.events()` generator backs `/v1/query`, `/v1/query/stream`, batch jobs and
evaluation, so offline and online results come from identical logic.

## Write path

- **Bulk:** `corpus ingest` normalises source files (locally or with Spark through the same
  `normalize_row`) into Parquet, then loads them into the registry. Each partition is a checkpoint;
  a repeated run skips completed partitions; a crashed run resumes.
- **Incremental:** writers insert a change event into the outbox in the same transaction as the
  change. The event processor applies events to the registry (which only moves a document forward)
  and then syncs the active index. An event is acknowledged only after both happened.
- **Index:** `corpus index build` creates a new physical index, validates it (counts, sampled
  lookups, mapping and model identity) and switches the alias atomically. `rollback` switches back.
  `sync` applies changed and deleted documents to the active index; a `write_id` per write lets it
  delete chunks left over from a longer previous version.

## Consistency model

| Property | Guarantee | Mechanism |
|---|---|---|
| Document identity | stable across re-ingestion | `doc_` + hash of tenant, source system, record id |
| Versions | monotonic; equal version + different content is a conflict | compare-and-set in `apply_documents` |
| Deletion | tombstone; cannot be resurrected by an older event | `is_deleted` + version check |
| Registry → index | eventual; lag is measured | `indexed_version` per document, `corpus_index_pending_documents`, `corpus_index_freshness_lag_seconds` |
| Cache | never served across tenants or permission sets; invalidated on change | key = tenant + scope + index version + model identity + tenant generation |
| Drift | detected and repaired | `corpus index check [--repair]`, scheduled `consistency` task |

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Model server slow or failing | retries with backoff, then circuit opens; answers degrade to the extractive generator and are flagged `degraded` |
| Generation queue full | `429` with `Overloaded`; nothing is queued unboundedly |
| Cache unreachable | served without cache; `/readyz` reports `degraded` |
| Search backend unreachable | `503 dependency_unavailable`; `/readyz` fails so the pod leaves the load balancer |
| Client disconnects during a stream | generation is cancelled |
| Poison change event | parked as `dead` after `max_attempts`; later events continue |
| Batch worker dies | job is re-queued after a timeout and resumes at the first uncommitted batch |

## Scaling path to 100M+ (design, not measured)

- **Normalisation** scales out with Spark; output is hash-bucketed by tenant so loaders run per
  bucket in parallel. Measured: 1M rows in ~32 s in Spark local mode on 12 cores.
- **Registry** is PostgreSQL; at 100M rows it needs partitioning by tenant hash and batched upserts
  (`COPY` into a staging table + merge) instead of the ORM path measured here (11.6k rows/s on SQLite).
- **Index** uses one alias over versioned indexes; shard count comes from `corpus capacity`
  (target shard size, replicas, vector encoding). `int8_hnsw` quantisation is supported in the mapping.
- **Embedding** is the dominant cost of a full build: it is resumable, cached by content hash, and
  horizontally parallel by document-id range.
- **Serving** is stateless; the API scales horizontally, the model server (vLLM) separately on GPUs.
