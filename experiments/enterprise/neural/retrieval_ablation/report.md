# Corpus evaluation run `retrieval_ablation`

Dataset `evalset-3815819a02958942` (800 queries; provenance {'synthetic': 800}). Index `corpus_neural_chunks_v0001_5a1c85`. Git `2aad6244a5f9cda91e5d8448c596a8bda7e7bf72`.

> **Synthetic data and labels.** These numbers show how the configurations behave on a controlled fixture. They are not evidence of quality on real documents.

## Retrieval

| Strategy | nDCG@10 | Recall@10 | MRR@10 | Hit@1 | Context precision | Context recall | Duplicate rate | Freshness error | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.746 | 0.846 | 0.714 | 0.638 | 0.111 | 0.832 | 0.000 | 0.000 | 0.000 |
| dense | 0.613 | 0.761 | 0.566 | 0.469 | 0.092 | 0.732 | 0.000 | 0.000 | 0.000 |
| hybrid | 0.726 | 0.887 | 0.677 | 0.572 | 0.114 | 0.858 | 0.000 | 0.000 | 0.000 |
| hybrid_rerank | 0.839 | 0.956 | 0.802 | 0.709 | 0.111 | 0.827 | 0.000 | 0.000 | 0.000 |
| hybrid_rerank_routed | 0.926 | 0.957 | 0.916 | 0.896 | 0.362 | 0.838 | 0.000 | 0.000 | 0.000 |
| adaptive | 0.926 | 0.957 | 0.916 | 0.896 | 0.362 | 0.838 | 0.000 | 0.000 | 0.000 |
| no_freshness | 0.721 | 0.885 | 0.672 | 0.566 | 0.114 | 0.859 | 0.000 | 0.925 | 0.000 |

## Generation and safety

| Strategy | Abstention correct | Citation precision | Citation coverage | Fact coverage | Unsupported-claim rate | Hallucination (rubric) | Injection success | Gates |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.749 | 0.349 | 0.634 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| dense | 0.664 | 0.414 | 0.553 | 0.637 | 0.000 | 0.001 | 0.000 | pass |
| hybrid | 0.748 | 0.392 | 0.649 | 0.725 | 0.000 | 0.000 | 0.000 | pass |
| hybrid_rerank | 0.749 | 0.366 | 0.655 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| hybrid_rerank_routed | 0.829 | 0.611 | 0.784 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| adaptive | 0.829 | 0.611 | 0.784 | 0.750 | 0.000 | 0.000 | 0.000 | pass |
| no_freshness | 0.748 | 0.377 | 0.649 | 0.738 | 0.000 | 0.000 | 0.000 | pass |

## Latency and cost drivers

| Strategy | End-to-end p50 ms | p95 | p99 | Retrieval p50 | Retrieval p95 | Rerank p50 | Context tokens / query | Errors |
|---|---|---|---|---|---|---|---|---|
| bm25 | 25.9 | 54.6 | 58.4 | 19.0 | 47.6 | — | 420 | 0 |
| dense | 25.2 | 28.5 | 44.8 | 17.5 | 20.4 | — | 472 | 0 |
| hybrid | 35.6 | 64.9 | 69.1 | 28.3 | 57.8 | — | 435 | 0 |
| hybrid_rerank | 76.6 | 110.0 | 123.0 | 69.9 | 102.1 | 38.4 | 450 | 0 |
| hybrid_rerank_routed | 62.9 | 82.5 | 91.7 | 56.4 | 86.5 | 26.0 | 344 | 0 |
| adaptive | 61.2 | 82.8 | 93.8 | 54.5 | 86.4 | 26.0 | 344 | 0 |
| no_freshness | 35.4 | 64.3 | 69.1 | 28.2 | 57.1 | — | 436 | 0 |

Latency: single process, sequential queries, arm64 (12 cores), search backend `local`, generation `extractive:extractive-v1:corpus-p1`. Not a load test.

## nDCG@10 by query category

| Category | Queries | bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_routed | adaptive | no_freshness |
|---|---|---|---|---|---|---|---|---|
| abbreviation | 120 | 0.747 | 0.642 | 0.838 | 0.906 | 0.966 | 0.966 | 0.849 |
| authorized_restricted | 40 | 0.860 | 0.950 | 0.960 | 1.000 | 1.000 | 1.000 | 0.960 |
| cross_tenant | 40 | — | — | — | — | — | — | — |
| exact_identifier | 40 | 1.000 | 0.042 | 0.256 | 0.982 | 1.000 | 1.000 | 0.256 |
| freshness_superseded | 40 | 0.988 | 0.510 | 0.752 | 0.646 | 1.000 | 1.000 | 0.653 |
| historical | 40 | 1.000 | 0.721 | 0.852 | 0.892 | 1.000 | 1.000 | 0.852 |
| keyword | 120 | 0.888 | 0.904 | 0.962 | 1.000 | 1.000 | 1.000 | 0.958 |
| multi_document | 40 | 1.000 | 0.019 | 0.577 | 0.959 | 1.000 | 1.000 | 0.579 |
| no_answer | 40 | — | — | — | — | — | — | — |
| paraphrase | 120 | 0.074 | 0.493 | 0.362 | 0.615 | 0.615 | 0.615 | 0.360 |
| prompt_injection | 40 | 0.717 | 0.941 | 0.960 | 1.000 | 1.000 | 1.000 | 0.960 |
| temporal_update | 80 | 0.995 | 0.562 | 0.751 | 0.606 | 1.000 | 1.000 | 0.751 |
| unauthorized_group | 40 | — | — | — | — | — | — | — |

## Paired comparison against `bm25`

95% paired-bootstrap interval of the per-query difference (candidate − baseline). A difference is called an improvement only when the interval excludes zero.

| Candidate | Metric | Baseline | Candidate | Δ | CI low | CI high | Wins/Losses/Ties | Verdict |
|---|---|---|---|---|---|---|---|---|
| dense | ndcg@10 | 0.746 | 0.613 | -0.133 | -0.174 | -0.093 | 173/276/231 | regressed |
| dense | recall@10 | 0.846 | 0.761 | -0.085 | -0.126 | -0.046 | 68/127/485 | regressed |
| dense | mrr@10 | 0.714 | 0.566 | -0.147 | -0.190 | -0.103 | 173/276/231 | regressed |
| dense | context_precision | 0.111 | 0.092 | -0.020 | -0.026 | -0.013 | 70/139/471 | regressed |
| dense | fact_coverage | 0.750 | 0.637 | -0.113 | -0.163 | -0.062 | 0/18/142 | regressed |
| dense | hallucination | 0.000 | 0.001 | 0.001 | 0.000 | 0.004 | 1/0/799 | no credible difference |
| hybrid | ndcg@10 | 0.746 | 0.726 | -0.020 | -0.050 | 0.009 | 183/170/327 | no credible difference |
| hybrid | recall@10 | 0.846 | 0.887 | 0.041 | 0.013 | 0.069 | 59/31/590 | improved |
| hybrid | mrr@10 | 0.714 | 0.677 | -0.036 | -0.070 | -0.002 | 183/156/341 | regressed |
| hybrid | context_precision | 0.111 | 0.114 | 0.002 | -0.001 | 0.006 | 52/39/589 | no credible difference |
| hybrid | fact_coverage | 0.750 | 0.725 | -0.025 | -0.050 | -0.006 | 0/4/156 | regressed |
| hybrid | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| hybrid_rerank | ndcg@10 | 0.746 | 0.839 | 0.092 | 0.063 | 0.121 | 210/136/334 | improved |
| hybrid_rerank | recall@10 | 0.846 | 0.956 | 0.110 | 0.085 | 0.135 | 76/1/603 | improved |
| hybrid_rerank | mrr@10 | 0.714 | 0.802 | 0.088 | 0.053 | 0.122 | 210/128/342 | improved |
| hybrid_rerank | context_precision | 0.111 | 0.111 | -0.001 | -0.003 | 0.001 | 17/21/642 | no credible difference |
| hybrid_rerank | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| hybrid_rerank | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| hybrid_rerank_routed | ndcg@10 | 0.746 | 0.926 | 0.180 | 0.157 | 0.203 | 214/8/458 | improved |
| hybrid_rerank_routed | recall@10 | 0.846 | 0.957 | 0.112 | 0.088 | 0.137 | 76/0/604 | improved |
| hybrid_rerank_routed | mrr@10 | 0.714 | 0.916 | 0.203 | 0.177 | 0.229 | 214/8/458 | improved |
| hybrid_rerank_routed | context_precision | 0.111 | 0.362 | 0.251 | 0.222 | 0.279 | 217/13/450 | improved |
| hybrid_rerank_routed | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| hybrid_rerank_routed | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| adaptive | ndcg@10 | 0.746 | 0.926 | 0.180 | 0.157 | 0.203 | 214/8/458 | improved |
| adaptive | recall@10 | 0.846 | 0.957 | 0.112 | 0.088 | 0.137 | 76/0/604 | improved |
| adaptive | mrr@10 | 0.714 | 0.916 | 0.203 | 0.177 | 0.229 | 214/8/458 | improved |
| adaptive | context_precision | 0.111 | 0.362 | 0.251 | 0.222 | 0.279 | 217/13/450 | improved |
| adaptive | fact_coverage | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0/0/160 | no credible difference |
| adaptive | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |
| no_freshness | ndcg@10 | 0.746 | 0.721 | -0.025 | -0.054 | 0.005 | 183/175/322 | no credible difference |
| no_freshness | recall@10 | 0.846 | 0.885 | 0.040 | 0.012 | 0.068 | 59/32/589 | improved |
| no_freshness | mrr@10 | 0.714 | 0.672 | -0.042 | -0.077 | -0.007 | 183/161/336 | regressed |
| no_freshness | context_precision | 0.111 | 0.114 | 0.003 | -0.001 | 0.006 | 52/38/590 | no credible difference |
| no_freshness | fact_coverage | 0.750 | 0.738 | -0.013 | -0.031 | 0.000 | 0/2/158 | no credible difference |
| no_freshness | hallucination | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/800 | no credible difference |

### Where each candidate changed results (nDCG@10)

| Candidate | Improved queries by category | Regressed queries by category |
|---|---|---|
| dense | {'abbreviation': 37, 'paraphrase': 72, 'authorized_restricted': 12, 'prompt_injection': 24, 'keyword': 28} | {'exact_identifier': 40, 'keyword': 18, 'multi_document': 40, 'abbreviation': 49, 'temporal_update': 60, 'prompt_injection': 4, 'freshness_superseded': 31, 'authorized_restricted': 4, 'paraphrase': 8, 'historical': 22} |
| hybrid | {'paraphrase': 68, 'abbreviation': 51, 'keyword': 30, 'authorized_restricted': 10, 'prompt_injection': 23, 'temporal_update': 1} | {'multi_document': 38, 'temporal_update': 41, 'exact_identifier': 35, 'keyword': 5, 'freshness_superseded': 20, 'abbreviation': 15, 'historical': 14, 'paraphrase': 1, 'authorized_restricted': 1} |
| hybrid_rerank | {'paraphrase': 84, 'abbreviation': 54, 'keyword': 34, 'authorized_restricted': 13, 'prompt_injection': 24, 'temporal_update': 1} | {'multi_document': 10, 'temporal_update': 64, 'abbreviation': 17, 'freshness_superseded': 30, 'historical': 11, 'paraphrase': 2, 'exact_identifier': 2} |
| hybrid_rerank_routed | {'paraphrase': 84, 'abbreviation': 57, 'keyword': 34, 'authorized_restricted': 13, 'freshness_superseded': 1, 'prompt_injection': 24, 'temporal_update': 1} | {'abbreviation': 6, 'paraphrase': 2} |
| adaptive | {'paraphrase': 84, 'abbreviation': 57, 'keyword': 34, 'authorized_restricted': 13, 'freshness_superseded': 1, 'prompt_injection': 24, 'temporal_update': 1} | {'abbreviation': 6, 'paraphrase': 2} |
| no_freshness | {'paraphrase': 68, 'abbreviation': 51, 'keyword': 30, 'authorized_restricted': 10, 'prompt_injection': 23, 'temporal_update': 1} | {'multi_document': 38, 'temporal_update': 41, 'freshness_superseded': 26, 'exact_identifier': 35, 'keyword': 7, 'abbreviation': 12, 'historical': 14, 'paraphrase': 1, 'authorized_restricted': 1} |

## Reproduction

```json
{
  "git_commit": "2aad6244a5f9cda91e5d8448c596a8bda7e7bf72",
  "dataset_version": "evalset-3815819a02958942",
  "dataset_path": "eval/queries.jsonl",
  "embedding": {
    "key": "minilm",
    "provider": "sentence_transformers",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "query_model_id": null,
    "revision": {
      "document": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
      "query": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    },
    "device": "mps",
    "batch_size": 64,
    "max_document_tokens": 256,
    "max_query_tokens": 256,
    "pooling": "mean",
    "normalize": true,
    "similarity": "cosine",
    "dimension": 384,
    "query_prefix": "",
    "document_prefix": "",
    "preprocessing_version": "text-v1"
  },
  "embedding_identity": "eceedfba72f542ef",
  "reranker": {
    "key": "ms_marco_minilm",
    "provider": "cross_encoder",
    "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
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
  "index_version": "corpus_neural_chunks_v0001_5a1c85",
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
