"""Adapter for the deterministic synthetic cohort.

The synthetic generator writes files in the same layout as the paper-described
INSPECT release (plus demographic / access-scope columns), so this adapter is a
thin specialisation of the legacy adapter and exercises the same code path.
A marker file prevents synthetic data from ever being mistaken for real data.
"""

from __future__ import annotations

import json

from app.core.errors import DatasetSchemaError
from app.ingestion.adapters.base import LayoutReport
from app.ingestion.adapters.inspect_legacy import InspectLegacyAdapter

SYNTHETIC_MARKER = "SYNTHETIC_DATA.json"
SYNTHETIC_FILES = {
    "cohort": "cohort.csv",
    "impressions": "impressions.csv",
    "ehr_dir": "ehr",
    "imaging_metadata": "imaging/metadata.csv",
}


class SyntheticAdapter(InspectLegacyAdapter):
    name = "synthetic"
    schema_version = "synthetic-v1"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        merged = {**SYNTHETIC_FILES, **config.files}
        super().__init__(config.model_copy(update={"files": merged}))

    def validate_layout(self) -> LayoutReport:
        marker = self.root / SYNTHETIC_MARKER
        if not marker.exists():
            raise DatasetSchemaError(
                f"{self.root} is not a synthetic dataset (missing {SYNTHETIC_MARKER}). "
                "Generate one with `make seed-synthetic`, or use an INSPECT adapter for real data."
            )
        info = json.loads(marker.read_text(encoding="utf-8"))
        report = super().validate_layout()
        report.schema_version = f"synthetic-{info.get('generator_version', 'unknown')}"
        report.warnings.append(
            "SYNTHETIC DATA: pipeline verification only; not evidence of clinical performance"
        )
        return report
