# How this example was produced

**Synthetic data only.** Fabricated fixtures from `app/ingestion/synthetic_generator.py` (170 patients, 189 cases, seed 20240607). Not evidence of clinical retrieval performance.

| Item | Value |
|---|---|
| Dataset fingerprint | `519f8e9069a09160aa92f93107209118c1ee3aed1c0e3d28e8e5e6d6f2bc23bc` |
| Query set / labels | `queries-45074590e1bbb4d9` / `qrels-869bc7b7882f62c8` (rubric labels, not clinician judgments) |
| Code version | the code of this repository's initial commit (`7a0d49c`); the run itself started just before that commit was created |
| Search backend | local (in-process BM25, exact kNN) — not Elasticsearch |
| Database | SQLite |
| Hardware | macOS-15.5-arm64-arm-64bit, 12 CPU cores, no GPU |
| Python / PyTorch | 3.11.1 / 2.15.0 |
| Measurement | single process, concurrency 1, 2 warm-up queries, query vectors never served from cache |
| Suites | chunking_comparison, embedding_comparison, final_test, generation_comparison, multimodal_ablation, reranking_depth, retrieval_comparison, smoke |

## Commands

```bash
make seed-synthetic index-baseline
cd backend
python -m app.cli evaluate experiments/smoke.yaml experiments/multimodal_ablation.yaml \
    experiments/embedding_comparison.yaml experiments/chunking_comparison.yaml \
    experiments/retrieval_comparison.yaml experiments/reranking_depth.yaml \
    experiments/generation_comparison.yaml
python -m app.cli select retrieval_comparison
python -m app.cli freeze-final experiments/retrieval_comparison.yaml
python -m app.cli run-experiment ../artifacts/experiments/final_test.yaml --final-test
python -m app.cli aggregate-metrics
```

## Not run

* G02–G06 (LLM generation variants) — no local LLM server was available; reported as failed.
* R10 and R12 (LLM query rewriting, HyDE) — need a local LLM; reported as skipped.
* Nothing was run on Elasticsearch, PostgreSQL, a GPU or INSPECT data.

Latencies describe this one machine with a 127-case corpus and exact kNN; they do not transfer to INSPECT scale or to Elasticsearch.
