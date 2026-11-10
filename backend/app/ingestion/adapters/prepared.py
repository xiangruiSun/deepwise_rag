"""Adapter for open datasets converted to the canonical CSV layout.

``inspect-rag prepare-dataset`` converts an openly licensed dataset (PMC-Patients, the
MIMIC-IV demo) into the same cohort / impressions / event-file layout the INSPECT
legacy adapter reads, and writes a ``DATASET.json`` provenance marker next to it.
The marker names the source, its licence and how the subset was drawn, so prepared
data can never be mistaken for INSPECT or for the synthetic fixtures.
"""

from __future__ import annotations

import json

from app.core.errors import DatasetSchemaError
from app.ingestion.adapters.base import LayoutReport
from app.ingestion.adapters.inspect_legacy import InspectLegacyAdapter

PREPARED_MARKER = "DATASET.json"
PREPARED_FILES = {"cohort": "cohort.csv", "impressions": "impressions.csv", "ehr_dir": "ehr"}


class PreparedDatasetAdapter(InspectLegacyAdapter):
    name = "prepared"
    schema_version = "prepared-v1"

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(config)
        # optional parts are used only when the converter wrote them
        defaults = {k: v for k, v in PREPARED_FILES.items() if (self.root / v).exists()}
        self.config = config.model_copy(update={"files": {**defaults, **config.files}})

    def validate_layout(self) -> LayoutReport:
        marker = self.root / PREPARED_MARKER
        if not marker.exists():
            raise DatasetSchemaError(
                f"{self.root} is not a prepared dataset (missing {PREPARED_MARKER}). "
                "Create it with `python -m app.cli prepare-dataset <name>`."
            )
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("dataset") != self.config.name:
            raise DatasetSchemaError(
                f"{PREPARED_MARKER} describes dataset '{info.get('dataset')}' but the configuration "
                f"names '{self.config.name}'"
            )
        report = super().validate_layout()
        report.schema_version = f"prepared-{info.get('converter_version', 'unknown')}"
        if info.get("notice"):
            report.warnings.append(str(info["notice"]))
        return report
