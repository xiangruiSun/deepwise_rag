#!/usr/bin/env bash
# One-shot local setup + CPU-only smoke run on SYNTHETIC data.
set -euo pipefail
cd "$(dirname "$0")/.."
make setup
make seed-synthetic index-baseline run-baseline report
echo
echo "Done. Start the API with 'make serve' and the UI with 'make frontend'."
echo "All numbers produced so far come from fabricated fixtures and verify the pipeline only."
