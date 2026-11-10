# Corpus platform experiments — `corpus_neural.yaml`

> **Synthetic corpus and labels** (20,000 source records, 800 queries, dataset `evalset-3815819a02958942`), in-process search backend, arm64. Relative behaviour of configurations only.

## Retrieval ablation (baseline: bm25)

| strategy | nDCG@10 | Recall@10 | fact coverage | freshness error | P50 ms | P95 ms | nDCG vs bm25 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.746 | 0.846 | 0.750 | 0.000 | 25.870 | 54.555 | baseline |
| dense | 0.613 | 0.761 | 0.637 | 0.000 | 25.200 | 28.532 | regressed |
| hybrid | 0.726 | 0.887 | 0.725 | 0.000 | 35.585 | 64.919 | no credible difference |
| hybrid_rerank | 0.839 | 0.956 | 0.750 | 0.000 | 76.575 | 110.011 | improved |
| hybrid_rerank_routed | 0.926 | 0.957 | 0.750 | 0.000 | 62.870 | 82.501 | improved |
| adaptive | 0.926 | 0.957 | 0.750 | 0.000 | 61.180 | 82.762 | improved |
| no_freshness | 0.721 | 0.885 | 0.738 | 0.925 | 35.420 | 64.344 | no credible difference |

## Context selection (hybrid retrieval)

| selection | context precision | context recall | citation coverage | fact coverage | duplicate rate | context tokens |
|---|---|---|---|---|---|---|
| top_k (baseline) | 0.113 | 0.860 | 0.630 | 0.725 | 0.001 | 445.626 |
| diverse | 0.114 | 0.858 | 0.649 | 0.725 | 0.000 | 435.111 |

## Indirect prompt injection (simulated instruction-following model)

| configuration | queries | model obeyed | injection success rate |
|---|---|---|---|
| verify_citations=False,quarantine=False | 40 | 40 | 1.000 |
| verify_citations=False,quarantine=True | 40 | 0 | 0.000 |
| verify_citations=True,quarantine=False | 40 | 40 | 1.000 |
| verify_citations=True,quarantine=True | 40 | 0 | 0.000 |
| extractive_generator(default) | 40 | — | 0.000 |

## Chunking (max 48 tokens, hybrid retrieval)

| chunking | chunks | build s | nDCG@10 | Recall@10 | context recall | fact coverage | context tokens |
|---|---|---|---|---|---|---|---|
| fixed | 41221 | 35.221 | 0.726 | 0.850 | 0.821 | 0.194 | 233.306 |
| sliding_window | 47348 | 30.392 | 0.732 | 0.863 | 0.882 | 0.181 | 266.586 |
| sentence | 45463 | 37.954 | 0.781 | 0.916 | 0.895 | 0.056 | 240.979 |
| structure | 59884 | 38.966 | 0.735 | 0.903 | 0.866 | 0.056 | 194.341 |
| recursive | 47175 | 34.068 | 0.779 | 0.932 | 0.898 | 0.113 | 226.920 |
| semantic | 135188 | 203.273 | 0.764 | 0.922 | 0.791 | 0.094 | 100.692 |
