# Evaluation protocol

Research question: *which combination of embeddings, lexical retrieval, multimodal features, fusion,
reranking and generation gives the most clinically relevant, faithful and efficient retrieval of
historical INSPECT cases?*

Nothing here presumes an answer. The more sophisticated option must earn its place.

## Task

Retrospective, **report-available** case retrieval. At query time the query patient's impression,
demographics and temporally eligible EHR features (at or before the procedure) may be used. Research
labels and later outcomes may not. A pre-report task would need a separate experiment that also hides
the query impression.

## Splits

| Role | Source |
|---|---|
| Candidate corpus | training patients |
| Development queries (tuning, selection) | validation patients |
| Final queries (run once) | held-out test patients |

The runner restricts candidates to the corpus split, always excludes the query patient (all of the
patient's studies) and records a `split_isolation` gate per run.

## Query categories

A radiology semantics · B demographic/diagnosis constraints · C laboratory characteristics ·
D multimodal case retrieval · E difficult cases (negation, rare terms, missing modalities) ·
F insufficient-evidence queries (must return nothing / make the generator abstain).

## Relevance labels

Graded, case-level: 3 highly, 2 moderately, 1 weakly, 0 not relevant; anything else is *unjudged*.
Two cases are never "highly similar" just because both are PE-positive.

Workflow for real data:

1. Run several diverse baseline suites (BM25, each embedding, image / laboratory retrieval).
2. `python -m app.cli build-annotation-pool <suite_id> --depth 20` pools the top results per query,
   de-duplicates by case, shuffles deterministically and writes **blinded** tasks. System attribution
   goes to a separate provenance file that annotators must not see.
3. Qualified clinical reviewers grade the tasks, ideally in duplicate. `adjudicate()` merges them:
   agreement is accepted, disagreement is flagged `needs_adjudication` (never averaged); raw agreement
   and Cohen's kappa are reported.
4. Store JSONL rows with `query_id`, `query_case_id`/`query_text`, `candidate_case_id`,
   `relevance_grade`, `query_category`, `relevant_modalities`, `reviewer_id`, `annotation_version`,
   `confidence`, `evidence_references`, `adjudication_status`, `label_source`.

Weak labels carry `label_source: weak` and are loaded only when a config asks for them; they never mix
silently with reviewed labels. Start with a pilot of roughly 30 queries × 20 pooled candidates.

## Metrics (`app/evaluation/retrieval_metrics.py`)

Computed over **distinct cases**.

| Metric | Definition |
|---|---|
| nDCG@K (primary: K=10) | DCG/IDCG, gain `2^grade − 1`, discount `1/log2(i+1)`, ideal from the judged candidates |
| Precision@K | relevant (grade ≥ threshold, default 2) in top K, divided by **K** |
| Recall@K | relevant in top K / relevant among **judged** candidates — judged-pool recall, not corpus recall |
| MRR@10 | reciprocal rank of the first relevant case |
| Judged coverage@K | share of returned top-K cases that have a judgment |
| Success@K, zero-result rate | as named |

Unjudged results: `unjudged_policy: non_relevant` (gain 0, conservative) or `condensed` (removed before
scoring). Judged coverage is always reported next to the metric; unjudged is never silently treated
as irrelevant. Queries without judgments are counted separately.

## Experiment suites (`configs/experiments`)

Run sequentially; shortlist at each stage on validation data — never a full Cartesian product.

| Stage | Suite | Experiments |
|---|---|---|
| 0 | `smoke.yaml` | CPU-only pipeline verification |
| 1 | `embedding_comparison.yaml` | E01 BM25, E02 MiniLM, E03 E5-base-v2, E04 BGE-base-v1.5, E05 MedCPT (no reranking) |
| 2 | `chunking_comparison.yaml` | C01 whole impression … C07 parent-child |
| 3 | `retrieval_comparison.yaml` | R01–R12 (fusion, cross-encoders, laboratory/image, query transformations) |
| 3b | `reranking_depth.yaml` | depth 20 / 40 / 80 for both cross-encoders |
| 4 | `multimodal_ablation.yaml` | text / structured EHR / image and combinations |
| 5 | `generation_comparison.yaml` | prompt variants and models with frozen retrieval |

Each run records: git commit, configuration hash, dataset fingerprint, query-set and label versions,
exact model revisions, index identities, seed, hardware, metric definitions, per-query rankings,
stage latencies (p50/p90/p95), index size and peak memory. Runs are keyed by the configuration hash,
so resuming never mixes configurations.

Model-specific notes:

* E5 uses the `query: ` / `passage: ` prefixes; BGE uses its query instruction.
* MedCPT uses its separate query (64 tokens) and article (512 tokens) encoders, CLS pooling and
  inner-product scoring. Impressions are passed to the article encoder as a single segment (it was
  trained on title + abstract pairs) — a documented deviation. Query truncation is recorded.
* MedCPT is a literature-retrieval model; whether it helps on radiology impressions is an empirical
  question the suite answers.

## Generation evaluation

Retrieval is frozen per run. Metrics (`app/evaluation/generation_metrics.py`): answer relevance,
faithfulness (claim support against cited evidence), citation correctness, citation coverage,
completeness, abstention correctness, contradiction rate, numeric accuracy, latency and tokens.
Citation-id verification is deterministic. Claim support is heuristic (numbers, negation polarity,
lexical grounding) and optionally a local-LLM judge; both must be calibrated on human-reviewed samples
and are not a safety guarantee.

## Selection policy (`app/evaluation/statistics.py`)

1. **Safety gates** — reject a run that fails authorization isolation, split isolation, the temporal
   feature cutoff, citation validity or reproducible execution (and an optional faithfulness floor).
2. **Primary objective** — validation nDCG@10 on reviewed labels.
3. **Credible improvement** — a candidate replaces the baseline only if the paired-bootstrap 95%
   interval of the per-query nDCG@10 difference excludes zero. The report also shows wins/losses and
   how much of the gain comes from the top 10% of queries.
4. **Resource constraint** — p95 latency within `max_latency_ratio` of the baseline's.
5. Otherwise the **simpler baseline** is kept.

```bash
python -m app.cli select <validation_suite_id>
python -m app.cli freeze-final experiments/<suite>.yaml      # writes artifacts/experiments/final_test.yaml
python -m app.cli run-experiment ../artifacts/experiments/final_test.yaml --final-test
python -m app.cli aggregate-metrics
```

The test stage needs the explicit `--final-test` flag and writes a lock file: evaluating the held-out
set again with a different configuration is refused.

## Performance measurement

Single process, concurrency 1, warm-up queries excluded, all stages timed inside the service. Compare
latencies only between runs on the same hardware (recorded with each run).

## Reports

`python -m app.cli aggregate-metrics` writes `report.md`, `report.json` and SVG charts with the 15
sections required by the specification. Sections without completed experiments say "Not yet
evaluated"; no number is ever filled in by hand.
