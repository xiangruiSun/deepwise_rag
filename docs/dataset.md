# Dataset: Stanford INSPECT

INSPECT is an access-controlled, de-identified research dataset for pulmonary-embolism diagnosis and
prognosis (CTPA studies, radiology impressions, structured EHR). The dataset and the models published
with it are for research, not clinical care.

* Dataset: <https://aimi.stanford.edu/datasets/inspect-Multimodal-Dataset-for-Pulmonary-Embolism-Diagnosis-and-Prognosis>
* Research repository: <https://github.com/som-shahlab/INSPECT_public>
* Project website: <https://som-shahlab.github.io/inspect-website/>
* Paper (NeurIPS 2023 Datasets and Benchmarks): <https://papers.neurips.cc/paper_files/paper/2023/hash/39736af1b9d87a1fddad9f84a6bcf64c-Abstract-Datasets_and_Benchmarks.html>

**This repository never downloads INSPECT, never bypasses its access controls and never commits
medical data.** Obtain the data yourself under its data-use agreement and place it under
`data/inspect/` (git-ignored).

## What is and is not verified

| Item | Status |
|---|---|
| Canonical schema, ingestion, joins, incremental updates | Verified on synthetic data written in the paper-described layout |
| `inspect_legacy` adapter (cohort CSV, impression CSV, gzipped FEMR-style event CSVs, imaging) | Exercised end-to-end on synthetic data; **not validated against the real release** |
| `inspect_meds` adapter (MEDS Parquet shards + split table) | Unit-tested on generated Parquet fixtures; **not validated against the real release** |
| Exact file and column names of the release | **Unknown.** Public loader code confirms only `patient_id`, `procedure_time`, `split` in the cohort table; everything else is an accepted alias |
| Demographics location (cohort columns vs. events) | Unknown; both are supported (`dataset.code_prefixes` for event-encoded demographics) |
| Image volumes (DICOM / NIfTI / NumPy) | Loader tested on generated DICOM series and NumPy volumes only |

Because of that, ingestion starts with schema inspection and never guesses:

```bash
cd backend
python -m app.cli inspect-dataset ../data/inspect --output ../artifacts/inspect_manifest.json
# edit configs/baseline.yaml: dataset.files, dataset.columns, dataset.code_prefixes
python -m app.cli validate-dataset -c baseline.yaml   # raises a descriptive error for any unresolved field
python -m app.cli ingest -c baseline.yaml
```

A required field that cannot be resolved by an explicit mapping or an exact known alias stops
ingestion with the list of available columns and the mapping to add.

## Identity and joins

* `case_id = "case_" + sha256(patient_id | normalised procedure_time)[:20]` — deterministic.
* A patient may have several CT studies; each is its own case. A patient must stay in one split.
* Reports and imaging studies are joined **only** through the dataset's own identifiers
  (the cohort's report / study reference columns). There is no timestamp-proximity join.
* Unmatched or invalid rows are counted in the manifest and listed in `failed_rows.jsonl`
  (identifiers and reasons only).

## Limitations the system respects

* **Impressions, not full notes.** Only de-identified impression sections are assumed; detailed
  findings sections are not expected. Whole-impression indexing is an explicit baseline.
* **Date shifting.** Dates are shifted per patient. Only within-patient order is meaningful; the
  system shows offsets from the procedure and never compares absolute dates across patients.
* **No geography, no lesion-level annotations, no validated image-similarity labels.** These are not
  offered as filters or facts; questions about them are answered with "insufficient evidence".
* **Labels are not similarity.** PE and outcome labels do not define clinical similarity; they are
  stored separately, never filterable, never used as query features. Relevance comes from a separate
  annotation effort (see [evaluation.md](evaluation.md)).
* **Laboratory units.** Only explicitly listed unit pairs are converted
  (`app/ingestion/normalization.py`); anything else is "not comparable", never coerced.

## Legacy INSPECT / FEMR environment

The published INSPECT code depends on older FEMR and CUDA stacks. Keep that environment separate from
this application: run image feature extraction there, export study-level vectors as an `.npz` file
(`ids`, `vectors`) and use `image.provider: precomputed`. The main backend never needs FEMR or CUDA
to run text retrieval.

## Synthetic data

`make seed-synthetic` writes a deterministic, fabricated cohort (170 patients / 189 cases) in the
paper-described layout, including deliberately difficult records: duplicate rows, orphan references,
unparseable and conflicting timestamps, negated findings, convertible and incompatible laboratory
units, post-procedure records (leakage probes), missing imaging and restricted cases. Its relevance
labels come from a hand-specified rubric over latent attributes of the generator. It proves the
pipeline works; it is not evidence about clinical retrieval quality.
