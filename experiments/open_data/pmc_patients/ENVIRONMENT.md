# How these results were produced

**PMC-Patients, not INSPECT.** PMC-Patients (published case-report summaries), not INSPECT. SUBSET of the benchmark: all judged-relevant patients of the sampled queries plus random distractors; results are not comparable with the full 155k-patient benchmark. Relevance labels come from the dataset's PubMed citation graph, not from clinician review. No dates, laboratory values or images; age and sex are parsed from the summary text.

| Item | Value |
|---|---|
| Source | PMC-Patients-ReCDS, patient-to-patient retrieval (PPR) — <https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS> (CC BY-NC-SA 4.0) |
| Citation | Zhao Z, Jin Q, Chen F, Peng T, Yu S. A large-scale dataset of patient summaries for retrieval-based clinical decision support systems. Scientific Data 10, 909 (2023). |
| Subset | 20,000 of 155,151 corpus patients; 200 of 2,866 labelled dev queries; 200 of 2,812 labelled test queries; seed 13 |
| Labels | 964 published patient-to-patient similarity labels (binary), label source `pmc_patients_ppr` |
| Dataset fingerprint | `4a4c237e06adf06e6441e0a27e67c858b57b794e0e778ed6bd3a86c24d229f43` |
| Query set / labels | `queries-8a6f739347934252` / `qrels-4773d052cdf1d2f8` |
| Search backend | Elasticsearch 8.15.3 (Docker), HNSW kNN |
| Database | SQLite |
| Hardware | Apple M3 Pro, 12 CPU cores, Apple GPU (MPS) for embeddings and rerankers; no CUDA |
| Python / PyTorch | 3.11.1 / 2.15.0 |
| Generation model | Llama 3.2 1B through Ollama 0.3.12 in Docker (CPU only) |
| Measurement | single process, concurrency 1, 2 warm-up queries |

## Notes on the numbers

* Case queries are encoded with the document encoder, and their vectors are already in the embedding
  cache from indexing, so dense-run latency excludes model inference for the query.
* The embedding, retrieval and generation suites were re-run on an otherwise idle machine after the
  first pass; quality metrics were identical in both passes. The chunking suite (C01–C04) was run once
  while other jobs were active, so its latency columns are not reliable; its quality metrics are.
* The held-out test queries were evaluated once, with the configuration frozen on validation data.

## Commands

```bash
cd backend
export INSPECT_RAG_APP_CONFIG=pmc_patients.yaml
export INSPECT_RAG_ARTIFACT_DIR=../artifacts/pmc_patients
export INSPECT_RAG_DATABASE_URL=sqlite:///../artifacts/pmc_patients/inspect_rag.db
export INSPECT_RAG_SEARCH_BACKEND=elasticsearch INSPECT_RAG_ELASTICSEARCH_URL=http://127.0.0.1:9200

python -m app.cli prepare-dataset pmc-patients
python -m app.cli ingest
python -m app.cli reindex --embedding lexical --embedding minilm --embedding bge --embedding e5 --embedding medcpt --no-images
python -m app.cli evaluate experiments/pmc/embedding_comparison.yaml experiments/pmc/chunking_comparison.yaml \
    experiments/pmc/retrieval_comparison.yaml experiments/pmc/generation_comparison.yaml
python -m app.cli select retrieval_comparison
python -m app.cli freeze-final experiments/pmc/retrieval_comparison.yaml
python -m app.cli run-experiment ../artifacts/pmc_patients/experiments/final_test.yaml --final-test
python -m app.cli aggregate-metrics
```
