# Corpus platform — data contracts

## Canonical document (`app/corpus/schema.py`)

One version of one source document after normalisation. Unknown fields are rejected.

| Field | Type | Notes |
|---|---|---|
| `tenant_id` | string | required; every query is scoped to exactly one tenant |
| `document_id` | string | `doc_` + first 24 hex of sha256(`tenant|source_system|source_record_id`) |
| `source_system`, `source_record_id` | string | identity in the source |
| `document_type` | string | e.g. `clinical_report`, `discharge_summary` |
| `title`, `content` | string | NFC-normalised, control characters removed |
| `content_hash` | string | sha256 of the content; drives change detection and the embedding cache |
| `document_version` | int ≥ 1 | monotonic per document |
| `source_updated_at`, `event_time` | datetime | when the source changed / when the event happened |
| `valid_from`, `valid_to` | datetime | validity window; an open end means "current" |
| `access_policy` | list of string | groups allowed to read the document inside its tenant (default `["public"]`) |
| `is_deleted` | bool | tombstone |
| `extension` | object | domain fields (patient pseudonym, encounter, department, codes) |
| `schema_version` | string | `corpus-doc-1` |

Source rows are mapped with a `SourceSpec` (`mapping`: canonical field → source column,
`defaults`: constants). Default source column names are `tenant`, `source_system`, `record_id`,
`doc_type`, `title`, `body`, `updated_at`, `event_time`, `valid_from`, `valid_to`, `version`, `acl`,
`deleted`, `extra_json`.

### Rejection reason codes

A row that cannot become a canonical document is written to the dead-letter table with one code;
the pipeline continues. Codes: `missing_tenant`, `missing_identifier`, `empty_content`,
`invalid_version`, `invalid_timestamp`, `invalid_validity_range`, `invalid_metadata`, `invalid_json`,
`version_conflict`.
`corpus ingest` prints the counts per code and `reconcile` proves
`read = loaded + rejected + duplicates + superseded`.

### Apply outcomes

`inserted` · `updated` · `deleted` · `unchanged` (idempotent replay) · `stale` (older than the
stored version) · `conflict` (same version, different content — rejected) · `duplicate` /
`superseded` (within one batch).

## Chunk (index document)

| Field | Notes |
|---|---|
| `chunk_id` | `{document_id}-{chunk_index:05d}` — stable across versions, so a new version overwrites |
| `document_id`, `document_version`, `chunk_index`, `chunk_count` | position inside the document |
| `text`, `chunk_text` | BM25-analysed text (standard analyzer) and the stored original chunk text |
| `embedding` | dense vector (`dense_vector`, HNSW; optional `int8_hnsw`) |
| `tenant_id`, `acl_groups` | authorisation filter fields (keyword) |
| `document_type`, `source_system`, `source_record_id`, `title` | metadata and exact lookup |
| `event_ts`, `valid_from_ts`, `valid_to_ts` | epoch seconds; open end = `4102444800` |
| `x_*` | extension fields, e.g. `x_patient_pseudo_id`, `x_encounter_id`, `x_department` |
| `content_hash`, `chunking_version`, `embedding_model_version`, `write_id` | provenance and cleanup |

Index metadata (`_meta`) records the embedding identity, chunking identity and mapping version;
activation is refused when they do not match the running configuration.

## Change event

```json
{"event_id": "evt_…", "tenant_id": "tenant-000", "document_id": "doc_…", "document_version": 7,
 "change_type": "upsert", "occurred_at": "2026-03-01T12:00:00", "payload": { …canonical document… }}
```

The envelope must agree with its payload (tenant and document id); a mismatch is dead-lettered.
`change_type` is `upsert` or `delete`. Delivery is at-least-once; effects are idempotent.

## Evaluation query (`eval/queries.jsonl`)

Required: `query_id`, `tenant_id`, `question`, `category`, `relevant_documents` (document id →
grade), `provenance` (`human` | `synthetic` | `weak`). Optional: `groups`, `forbidden_documents`
(must never appear in ranking, context or citations), `stale_documents`, `forbidden_text`,
`required_facts`, `expects_abstention`, `expected_citation_sources`, `temporal.as_of`.
The dataset version is a hash of its rows; every report states the provenance mix.

## Batch job

Input: JSONL or Parquet with `query` (or `question`) and optional `id`, `strategy`,
`document_types`. Output: `part-NNNNN.jsonl` files (one per batch, written atomically) and a
`manifest.json` with the job configuration, input hash, model version and index version.

## Privacy rules built into the contracts

- Audit events store identifiers, counts and a trace id — never the question or document text.
- Feedback stores ratings, categories, chunk ids and a query hash — no free text.
- Error responses and logs carry error classes and codes, not payloads.
- Dead letters store the reason code, source location (file, partition, row number) and small
  structured details; the rejected document text is not copied.
