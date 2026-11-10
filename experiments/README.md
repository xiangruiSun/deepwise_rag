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
failed/skipped there because no LLM server was available when it was produced.

## Adding real results

After running suites on authorized INSPECT data, copy only the aggregate report (no per-query files,
no case identifiers, no clinical text) into a new subdirectory, together with the dataset
fingerprint and the configuration hashes it reports.
