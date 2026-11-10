# Corpus platform experiments — `corpus.yaml`

> **Synthetic corpus and labels** (20,000 source records, 800 queries, dataset `evalset-3815819a02958942`), in-process search backend, arm64. Relative behaviour of configurations only.

## Retrieval ablation (baseline: bm25)

| strategy | nDCG@10 | Recall@10 | fact coverage | freshness error | P50 ms | P95 ms | nDCG vs bm25 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.746 | 0.846 | 0.750 | 0.000 | 25.525 | 53.901 | baseline |
| dense | 0.556 | 0.711 | 0.575 | 0.000 | 16.010 | 18.760 | regressed |
| hybrid | 0.637 | 0.793 | 0.750 | 0.000 | 34.550 | 62.278 | regressed |
| hybrid_rerank | 0.529 | 0.736 | 0.681 | 0.000 | 34.935 | 62.533 | regressed |
| hybrid_rerank_routed | 0.672 | 0.804 | 0.750 | 0.000 | 35.250 | 54.441 | regressed |
| adaptive | 0.672 | 0.804 | 0.750 | 0.000 | 35.535 | 55.223 | regressed |
| no_freshness | 0.630 | 0.794 | 0.750 | 0.850 | 34.415 | 63.262 | regressed |

## Context selection (hybrid retrieval)

| selection | context precision | context recall | citation coverage | fact coverage | duplicate rate | context tokens |
|---|---|---|---|---|---|---|
| top_k (baseline) | 0.101 | 0.779 | 0.511 | 0.744 | 0.003 | 447.915 |
| diverse | 0.101 | 0.772 | 0.588 | 0.750 | 0.000 | 430.752 |

## Indirect prompt injection (simulated instruction-following model)

| configuration | queries | model obeyed | injection success rate |
|---|---|---|---|
| verify_citations=False,quarantine=False | 40 | 39 | 0.975 |
| verify_citations=False,quarantine=True | 40 | 0 | 0.000 |
| verify_citations=True,quarantine=False | 40 | 39 | 0.975 |
| verify_citations=True,quarantine=True | 40 | 0 | 0.000 |
| extractive_generator(default) | 40 | — | 0.000 |

## Chunking (max 48 tokens, hybrid retrieval)

| chunking | chunks | build s | nDCG@10 | Recall@10 | context recall | fact coverage | context tokens |
|---|---|---|---|---|---|---|---|
| fixed | 37195 | 10.799 | 0.647 | 0.747 | 0.732 | 0.631 | 301.168 |
| sliding_window | 39671 | 9.277 | 0.678 | 0.782 | 0.782 | 0.625 | 333.221 |
| sentence | 37746 | 11.134 | 0.686 | 0.773 | 0.785 | 0.606 | 305.249 |
| structure | 50932 | 12.467 | 0.584 | 0.745 | 0.728 | 0.606 | 243.786 |
| recursive | 39293 | 9.637 | 0.662 | 0.762 | 0.764 | 0.594 | 285.231 |
| semantic | 144602 | 41.454 | 0.728 | 0.865 | 0.785 | 0.362 | 89.403 |
