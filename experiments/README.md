# experiments/

Experiment **definitions** live in `configs/experiments/`. Experiment **outputs** are written to
`artifacts/experiments/` (git-ignored) because per-query results contain case identifiers.

Only non-sensitive, aggregate result summaries belong in this directory.

## `synthetic_verification/`

A complete worked example produced by running every suite on the **synthetic** cohort:

* `report.md`, `report.json`, `charts/` — the generated evaluation report
* `leaderboards/` — the machine-readable leaderboard of each suite
* `selection.json` — the outcome of the model-selection policy
* `ENVIRONMENT.md` — how and where it was produced

It demonstrates the output format and that the pipeline runs end to end with the real embedding and
reranking models. **The data are fabricated; the numbers say nothing about clinical retrieval
performance and no INSPECT experiment is reported.** Experiments needing a local LLM are listed as
failed/skipped there because no LLM server was available when it was produced; the LLM generation
comparison was run later and is reported in `open_data/`.

## `open_data/`

Results on two small open datasets and with local open-source LLMs — see
[open_data/README.md](open_data/README.md):

* `pmc_patients/` — full evaluation report for PMC-Patients (similar-patient retrieval with published
  labels): embedding, chunking, fusion and reranking comparisons, the selection decision, one held-out
  test run and LLM generation on real summaries.
* `mimic_demo_check.json` — sanity check of laboratory and diagnosis similarity on the MIMIC-IV demo.
* `synthetic_generation/` — LLM generation comparison (Llama 3.2 1B, Qwen 2.5 1.5B) on the synthetic cohort.

These are real data, but **not INSPECT**: no CT images, no radiology impressions, and labels that are
not clinician judgments.

## Adding real results

After running suites on authorized INSPECT data, copy only the aggregate report (no per-query files,
no case identifiers, no clinical text) into a new subdirectory, together with the dataset
fingerprint and the configuration hashes it reports.
