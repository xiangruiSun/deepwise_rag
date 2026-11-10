# Open datasets used while INSPECT access is pending

INSPECT is access-controlled and large. Until it is available, two small, openly licensed, published
datasets exercise the system on **real** data. Neither is INSPECT, neither has images, and results on
them say nothing about INSPECT.

| | PMC-Patients (PPR) | MIMIC-IV demo (MEDS) |
|---|---|---|
| What | Patient summaries from published case reports | 100 ICU patients, structured hospital records |
| Reference | Zhao et al., *Scientific Data* 10, 909 (2023) | Johnson et al. / van de Water et al., PhysioNet |
| Licence | CC BY-NC-SA 4.0 (non-commercial) | ODbL 1.0 |
| Access | Open download, no agreement | Open download, no credentialing |
| Free text | Yes (median about 450 words) | No |
| Age / sex | Parsed from the summary text | Yes |
| Laboratory values, diagnosis codes | No | Yes |
| Images | No | No |
| Relevance labels | Yes, published patient-to-patient similarity | No |
| What it tests here | Text retrieval, embeddings, chunking, fusion, reranking, generation | Structured ingestion, unit handling, laboratory and diagnosis similarity |

The imaging path therefore remains verified on synthetic data only. Results:
[experiments/open_data/](../experiments/open_data/README.md).

Nothing is downloaded by the code and nothing under `data/` is committed. Both datasets are converted
by `prepare-dataset` into the canonical CSV layout (the same one the INSPECT legacy adapter reads)
with a `DATASET.json` provenance marker, and are read by the `prepared` adapter. Reports, the API and
the UI name the dataset; they never call it INSPECT.

## PMC-Patients

Download from <https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS> into
`data/pmc_patients/raw/`, keeping the folder names: `PPR/corpus.jsonl`, `PPR/qrels_dev.tsv`,
`PPR/qrels_test.tsv`, `queries/dev_queries.jsonl`, `queries/test_queries.jsonl` (about 470 MB).

```bash
cd backend
export INSPECT_RAG_APP_CONFIG=pmc_patients.yaml
export INSPECT_RAG_ARTIFACT_DIR=../artifacts/pmc_patients
export INSPECT_RAG_DATABASE_URL=sqlite:///../artifacts/pmc_patients/inspect_rag.db

python -m app.cli prepare-dataset pmc-patients          # seeded subset (see below)
python -m app.cli ingest
python -m app.cli reindex --embedding lexical --embedding minilm --embedding e5 --embedding bge --embedding medcpt
python -m app.cli evaluate experiments/pmc/embedding_comparison.yaml experiments/pmc/chunking_comparison.yaml \
    experiments/pmc/retrieval_comparison.yaml
python -m app.cli select retrieval_comparison
python -m app.cli freeze-final experiments/pmc/retrieval_comparison.yaml
python -m app.cli run-experiment ../artifacts/pmc_patients/experiments/final_test.yaml --final-test
python -m app.cli aggregate-metrics
```

How the data is mapped:

* Corpus patients are the candidate corpus (split `train`); dev and test query patients are cases of
  the `valid` and `test` splits. A query is "find patients similar to this one" (`query_mode: case`),
  so split isolation and same-patient exclusion apply exactly as they would for INSPECT.
* **Subset.** By default `prepare-dataset` keeps 200 dev and 200 test query patients (seeded sample of
  those that have labels), every corpus patient labelled similar to one of them, and random
  distractors up to 20,000 corpus patients. Fewer distractors make retrieval easier, so **these numbers
  are not comparable with published results on the full 155k-patient benchmark.** Use
  `--corpus-size 0 --dev-queries 0 --test-queries 0` for the full benchmark.
* **Labels** are the published PPR annotations, which the dataset authors derived from the PubMed
  citation graph. They are binary in this split and are not clinician judgments. Unlabelled pairs count
  as not relevant, so similar patients without a citation link lower every score.
* The summaries have **no dates**: all cases share a placeholder index time and nothing temporal is
  available. **Age and sex** are not columns of the benchmark files; they are parsed from the opening
  of each summary by a conservative pattern and left empty otherwise.
* Summaries are longer than the encoders' input limits. Whole-summary dense indexing therefore sees
  only the first 256 (MiniLM) or 512 tokens; BM25 sees everything. The chunking suite measures what
  that costs.

## MIMIC-IV demo

Download the ZIP from <https://physionet.org/content/mimic-iv-demo-meds/0.0.1/> and extract it into
`data/mimic_iv_demo_meds/` (so that `data/mimic_iv_demo_meds/data/train/0.parquet` exists).

```bash
cd backend
export INSPECT_RAG_APP_CONFIG=mimic_demo.yaml
export INSPECT_RAG_ARTIFACT_DIR=../artifacts/mimic_iv_demo
export INSPECT_RAG_DATABASE_URL=sqlite:///../artifacts/mimic_iv_demo/inspect_rag.db

python -m app.cli prepare-dataset mimic-demo
python -m app.cli ingest
python -m app.cli reindex --embedding lexical --no-images
python -m app.cli search --case-id <case_id> --recipe structured_lab_dx --top-k 5
PYTHONPATH=. python ../scripts/mimic_demo_check.py
```

How the data is mapped:

* **One case per hospital stay, indexed at discharge.** Laboratory values in the look-back window
  before discharge and the diagnosis codes recorded at discharge count as "at or before" the index
  time; later stays never do. The MEDS subject splits are kept (80 / 10 / 10 patients).
* Laboratory events are kept when the MEDS code table links them to a LOINC code; four bedside vital
  signs are mapped through an explicit item table. Units come from the MEDS code. Everything else
  (medications, infusions, procedures, unmapped chart items, results without a number) is counted in
  `DATASET.json` and skipped.
* The laboratory registry gained routine chemistry and vital signs for this (sodium, potassium,
  chloride, bicarbonate, urea nitrogen, glucose, haematocrit, respiratory rate, blood pressure), each
  with explicitly listed unit conversions only.
* There are no notes, so cases have no text record and text retrieval is unavailable. There are no
  relevance labels, so no retrieval-quality metric is reported for this dataset.
