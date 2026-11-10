# Corpus evaluation run `retrieval_ablation`

Dataset `evalset-3815819a02958942` (800 queries; provenance {'synthetic': 800}). Index `corpus_chunks_v0001_cb0cfc`. Git `2aad6244a5f9cda91e5d8448c596a8bda7e7bf72`.

> **Synthetic data and labels.** These numbers show how the configurations behave on a controlled fixture. They are not evidence of quality on real documents.

## Retrieval

| Strategy | nDCG@10 | Recall@10 | MRR@10 | Hit@1 | Context precision | Context recall | Duplicate rate | Freshness error | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.746 | 0.846 | 0.714 | 0.638 | 0.111 | 0.832 | 0.000 | 0.000 | 0.000 |
| dense | 0.556 | 0.711 | 0.510 | 0.391 | 0.088 | 0.693 | 0.000 | 0.000 | 0.000 |
| hybrid | 0.637 | 0.793 | 0.593 | 0.465 | 0.101 | 0.772 | 0.000 | 0.000 | 0.000 |
| hybrid_rerank | 0.529 | 0.736 | 0.471 | 0.354 | 0.089 | 0.663 | 0.000 | 0.000 | 0.000 |
| hybrid_rerank_routed | 0.672 | 0.804 | 0.629 | 0.546 | 0.353 | 0.768 | 0.000 | 0.000 | 0.000 |
| adaptive | 0.672 | 0.804 | 0.629 | 0.546 | 0.353 | 0.768 | 0.000 | 0.000 | 0.000 |
| no_freshness | 0.630 | 0.794 | 0.584 | 0.456 | 0.101 | 0.771 | 0.000 | 0.850 | 0.000 |

## Generation and safety

| Strategy | Abstention correct | Citation precision | Citation coverage | Fact coverage | Unsupported-claim rate | Hallucination (rubric) | Injection success | Gates |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.749 | 0.349 | 0.634 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| dense | 0.625 | 0.358 | 0.474 | 0.575 | 0.000 | 0.000 | 0.000 | pass |
| hybrid | 0.748 | 0.354 | 0.588 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| hybrid_rerank | 0.743 | 0.329 | 0.485 | 0.681 | 0.000 | 0.000 | 0.000 | pass |
| hybrid_rerank_routed | 0.825 | 0.578 | 0.671 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| adaptive | 0.825 | 0.578 | 0.671 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| no_freshness | 0.744 | 0.342 | 0.587 | 0.750 | 0.000 | 0.000 | 0.000 | pass |

## Latency and cost drivers

| Strategy | End-to-end p50 ms | p95 | p99 | Retrieval p50 | Retrieval p95 | Rerank p50 | Context tokens / query | Errors |
|---|---|---|---|---|---|---|---|---|
| bm25 | 25.5 | 53.9 | 56.7 | 18.9 | 46.5 | — | 420 | 0 |
| dense | 16.0 | 18.8 | 20.8 | 9.7 | 11.2 | — | 441 | 0 |
| hybrid | 34.5 | 62.3 | 66.6 | 27.7 | 56.0 | — | 431 | 0 |
| hybrid_rerank | 34.9 | 62.5 | 67.1 | 28.4 | 56.7 | 0.3 | 394 | 0 |
| hybrid_rerank_routed | 35.2 | 54.4 | 58.7 | 27.3 | 54.8 | 0.3 | 313 | 0 |
| adaptive | 35.5 | 55.2 | 61.4 | 28.0 | 55.4 | 0.3 | 313 | 0 |
| no_freshness | 34.4 | 63.3 | 68.9 | 27.5 | 56.0 | — | 432 | 0 |

Latency: single process, sequential queries, arm64 (12 cores), search backend `local`, generation `extractive:extractive-v1:corpus-p1`. Not a load test.

## nDCG@10 by query category

| Category | Queries | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_routed | adaptive | no_freshness |
|---|---|---|---|---|---|---|---|---|
| abbreviation | 120 | 0.749 | 0.719 | 0.744 | 0.652 | 0.678 | 0.678 | 0.736 |
| authorized_restricted | 40 | 0.860 | 0.816 | 0.833 | 0.606 | 0.598 | 0.598 | 0.833 |
| cross_tenant | 40 | — | — | — | — | — | — | — |
| exact_identifier | 40 | 1.000 | 0.378 | 0.760 | 0.925 | 1.000 | 1.000 | 0.760 |
| freshness_superseded | 40 | 0.988 | 0.469 | 0.693 | 0.483 | 1.000 | 1.000 | 0.621 |
| historical | 40 | 1.000 | 0.763 | 0.884 | 0.692 | 1.000 | 1.000 | 0.884 |
| keyword | 120 | 0.888 | 0.897 | 0.896 | 0.725 | 0.725 | 0.725 | 0.889 |
| multi_document | 40 | 1.000 | 0.187 | 0.441 | 0.635 | 1.000 | 1.000 | 0.447 |
| no_answer | 40 | — | — | — | — | — | — | — |
| paraphrase | 120 | 0.074 | 0.038 | 0.041 | 0.038 | 0.038 | 0.038 | 0.040 |
| prompt_injection | 40 | 0.717 | 0.760 | 0.717 | 0.498 | 0.498 | 0.498 | 0.717 |
| temporal_update | 80 | 0.995 | 0.558 | 0.725 | 0.453 | 1.000 | 1.000 | 0.724 |
| unauthorized_group | 40 | — | — | — | — | — | — | — |

## Paired comparison against `bm25`

95% paired-bootstrap interval of the per-query difference (candidate − baseline). A difference is called an improvement only when the interval excludes zero.

| Candidate | Metric | Baseline | Candidate | Δ | CI low | CI high | Wins/Losses/Ties | Verdict |
|---|---|---|---|---|---|---|---|---|
| dense | ndcg@10 | 0.746 | 0.556 | -0.190 | -0.220 | -0.163 | 75/252/353 | regressed |
| dense | recall@10 | 0.846 | 0.711 | -0.135 | -0.162 | -0.108 | 8/107/565 | regressed |
| dense | mrr@10 | 0.714 | 0.510 | -0.204 | -0.237 | -0.174 | 75/246/359 | regressed |
| dense | context_precision | 0.111 | 0.088 | -0.023 | -0.028 | -0.019 | 14/116/550 | regressed |
| dense | fact_coverage | 0.750 | 0.575 | -0.175 | -0.231 | -0.119 | 0/28/132 | regressed |
| dense | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| hybrid | ndcg@10 | 0.746 | 0.637 | -0.110 | -0.131 | -0.090 | 56/180/444 | regressed |
| hybrid | recall@10 | 0.846 | 0.793 | -0.053 | -0.072 | -0.035 | 5/50/625 | regressed |
| hybrid | mrr@10 | 0.714 | 0.593 | -0.121 | -0.145 | -0.099 | 56/166/458 | regressed |
| hybrid | context_precision | 0.111 | 0.101 | -0.010 | -0.013 | -0.007 | 6/60/614 | regressed |
| hybrid | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| hybrid | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| hybrid_rerank | ndcg@10 | 0.746 | 0.529 | -0.217 | -0.242 | -0.193 | 22/290/368 | regressed |
| hybrid_rerank | recall@10 | 0.846 | 0.736 | -0.110 | -0.133 | -0.085 | 5/85/590 | regressed |
| hybrid_rerank | mrr@10 | 0.714 | 0.471 | -0.243 | -0.270 | -0.216 | 22/273/385 | regressed |
| hybrid_rerank | context_precision | 0.111 | 0.089 | -0.023 | -0.027 | -0.019 | 9/132/539 | regressed |
| hybrid_rerank | fact_coverage | 0.750 | 0.681 | -0.069 | -0.106 | -0.031 | 0/11/149 | regressed |
| hybrid_rerank | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| hybrid_rerank_routed | ndcg@10 | 0.746 | 0.672 | -0.075 | -0.091 | -0.059 | 28/140/512 | regressed |
| hybrid_rerank_routed | recall@10 | 0.846 | 0.804 | -0.041 | -0.057 | -0.024 | 5/33/642 | regressed |
| hybrid_rerank_routed | mrr@10 | 0.714 | 0.629 | -0.084 | -0.103 | -0.067 | 28/140/512 | regressed |
| hybrid_rerank_routed | context_precision | 0.111 | 0.353 | 0.242 | 0.212 | 0.271 | 209/53/418 | improved |
| hybrid_rerank_routed | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| hybrid_rerank_routed | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| adaptive | ndcg@10 | 0.746 | 0.672 | -0.075 | -0.091 | -0.059 | 28/140/512 | regressed |
| adaptive | recall@10 | 0.846 | 0.804 | -0.041 | -0.057 | -0.024 | 5/33/642 | regressed |
| adaptive | mrr@10 | 0.714 | 0.629 | -0.084 | -0.103 | -0.067 | 28/140/512 | regressed |
| adaptive | context_precision | 0.111 | 0.353 | 0.242 | 0.212 | 0.271 | 209/53/418 | improved |
| adaptive | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| adaptive | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| no_freshness | ndcg@10 | 0.746 | 0.630 | -0.116 | -0.138 | -0.097 | 55/191/434 | regressed |
| no_freshness | recall@10 | 0.846 | 0.794 | -0.051 | -0.070 | -0.035 | 5/49/626 | regressed |
| no_freshness | mrr@10 | 0.714 | 0.584 | -0.130 | -0.155 | -0.107 | 55/176/449 | regressed |
| no_freshness | context_precision | 0.111 | 0.101 | -0.010 | -0.013 | -0.008 | 6/61/613 | regressed |
| no_freshness | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| no_freshness | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |

### Where each candidate changed results (nDCG@10)

| Candidate | Improved queries by category | Regressed queries by category |
|---|---|---|
| dense | {'paraphrase': 9, 'keyword': 15, 'abbreviation': 30, 'prompt_injection': 16, 'authorized_restricted': 5} | {'multi_document': 40, 'authorized_restricted': 11, 'temporal_update': 54, 'freshness_superseded': 32, 'prompt_injection': 8, 'abbreviation': 34, 'exact_identifier': 29, 'historical': 17, 'keyword': 13, 'paraphrase': 14} |
| hybrid | {'paraphrase': 7, 'abbreviation': 25, 'keyword': 11, 'prompt_injection': 8, 'authorized_restricted': 5} | {'multi_document': 37, 'authorized_restricted': 7, 'temporal_update': 42, 'freshness_superseded': 24, 'prompt_injection': 5, 'abbreviation': 21, 'historical': 11, 'keyword': 9, 'paraphrase': 12, 'exact_identifier': 12} |
| hybrid_rerank | {'paraphrase': 7, 'prompt_injection': 3, 'abbreviation': 10, 'keyword': 1, 'authorized_restricted': 1} | {'multi_document': 32, 'keyword': 47, 'abbreviation': 37, 'temporal_update': 57, 'prompt_injection': 26, 'freshness_superseded': 30, 'authorized_restricted': 22, 'historical': 24, 'paraphrase': 12, 'exact_identifier': 3} |
| hybrid_rerank_routed | {'abbreviation': 14, 'paraphrase': 7, 'freshness_superseded': 1, 'prompt_injection': 3, 'keyword': 1, 'authorized_restricted': 1, 'temporal_update': 1} | {'keyword': 47, 'abbreviation': 33, 'prompt_injection': 26, 'authorized_restricted': 22, 'paraphrase': 12} |
| adaptive | {'abbreviation': 14, 'paraphrase': 7, 'freshness_superseded': 1, 'prompt_injection': 3, 'keyword': 1, 'authorized_restricted': 1, 'temporal_update': 1} | {'keyword': 47, 'abbreviation': 33, 'prompt_injection': 26, 'authorized_restricted': 22, 'paraphrase': 12} |
| no_freshness | {'paraphrase': 7, 'abbreviation': 23, 'keyword': 12, 'prompt_injection': 8, 'authorized_restricted': 5} | {'multi_document': 38, 'authorized_restricted': 7, 'temporal_update': 42, 'freshness_superseded': 27, 'prompt_injection': 5, 'abbreviation': 24, 'historical': 11, 'keyword': 13, 'paraphrase': 12, 'exact_identifier': 12} |

## Reproduction

```json
{
  "git_commit": "2aad6244a5f9cda91e5d8448c596a8bda7e7bf72",
  "dataset_version": "evalset-3815819a02958942",
  "dataset_path": "eval/queries.jsonl",
  "embedding": {
    "key": "hashing",
    "provider": "hashing",
    "model_id": "hashing-ngram-v1",
    "query_model_id": null,
    "revision": {
      "document": "hashing-v1",
      "query": "hashing-v1"
    },
    "device": "cpu",
    "batch_size": 32,
    "max_document_tokens": 512,
    "max_query_tokens": 512,
    "pooling": "hashing",
    "normalize": true,
    "similarity": "cosine",
    "dimension": 384,
    "query_prefix": "",
    "document_prefix": "",
    "preprocessing_version": "text-v1"
  },
  "embedding_identity": "e5e7c350fd92e573",
  "reranker": {
    "key": "lexical_overlap",
    "provider": "lexical_overlap",
    "model_id": null,
    "revision": null,
    "depth": 20
  },
  "generation": {
    "backend": "extractive",
    "model": "extractive-v1",
    "base_url": null,
    "api_key_env": "INSPECT_RAG_LLM_API_KEY",
    "prompt_version": "corpus-p1",
    "max_output_tokens": 400,
    "temperature": 0.0,
    "timeout_seconds": 60.0,
    "max_concurrency": 8,
    "queue_limit": 32,
    "max_retries": 1,
    "circuit_failure_threshold": 5,
    "circuit_reset_seconds": 30.0,
    "verify_citations": true
  },
  "prompt_version": "corpus-p1",
  "chunking": "sliding_window-200-30-c1",
  "context": {
    "selection": "diverse",
    "token_budget": 1500,
    "max_chunks": 8,
    "duplicate_threshold": 0.85,
    "mmr_lambda": 0.7,
    "per_document_limit": 3,
    "min_relative_score": 0.0,
    "parent_expansion": false,
    "quarantine_instructions": true,
    "detect_conflicts": true
  },
  "index_version": "corpus_chunks_v0001_cb0cfc",
  "strategies": {
    "bm25": {
      "strategy": "bm25",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": false,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": false,
      "query_rewriting": false,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": true,
      "context_selection": null
    },
    "dense": {
      "strategy": "dense",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": false,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": false,
      "query_rewriting": false,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": true,
      "context_selection": null
    },
    "hybrid": {
      "strategy": "hybrid",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": false,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": false,
      "query_rewriting": false,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": true,
      "context_selection": null
    },
    "hybrid_rerank": {
      "strategy": "hybrid",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": true,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": false,
      "query_rewriting": false,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": true,
      "context_selection": null
    },
    "hybrid_rerank_routed": {
      "strategy": "hybrid",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": true,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": true,
      "query_rewriting": true,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": true,
      "context_selection": null
    },
    "adaptive": {
      "strategy": "hybrid",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": true,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": true,
      "query_rewriting": true,
      "max_rewrites": 2,
      "adaptive_depth": true,
      "freshness": true,
      "context_selection": null
    },
    "no_freshness": {
      "strategy": "hybrid",
      "bm25_top_k": 100,
      "dense_top_k": 100,
      "num_candidates": 200,
      "rrf_k": 60,
      "rrf_weights": {},
      "fusion_window": 100,
      "rerank": false,
      "rerank_depth": 20,
      "final_k": 8,
      "query_routing": false,
      "query_rewriting": false,
      "max_rewrites": 2,
      "adaptive_depth": false,
      "freshness": false,
      "context_selection": null
    }
  },
  "k_values": [
    1,
    5,
    10
  ],
  "seed": 13,
  "limit": null,
  "notes": "retrieval ablation; baseline bm25"
}
```
