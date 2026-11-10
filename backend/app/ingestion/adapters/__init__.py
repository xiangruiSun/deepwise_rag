"""Dataset adapters.  All adapters emit the same canonical rows."""

from __future__ import annotations

from app.core.config import DatasetConfig
from app.core.errors import DatasetSchemaError
from app.ingestion.adapters.base import DatasetAdapter, LayoutReport
from app.ingestion.adapters.inspect_legacy import InspectLegacyAdapter
from app.ingestion.adapters.inspect_meds import InspectMedsAdapter
from app.ingestion.adapters.prepared import PreparedDatasetAdapter
from app.ingestion.adapters.synthetic import SyntheticAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "synthetic": SyntheticAdapter,
    "inspect_legacy": InspectLegacyAdapter,
    "inspect_meds": InspectMedsAdapter,
    "prepared": PreparedDatasetAdapter,
}


def build_adapter(config: DatasetConfig) -> DatasetAdapter:
    try:
        return ADAPTERS[config.adapter](config)
    except KeyError as exc:
        raise DatasetSchemaError(f"unknown dataset adapter '{config.adapter}'") from exc


__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "InspectLegacyAdapter",
    "InspectMedsAdapter",
    "LayoutReport",
    "PreparedDatasetAdapter",
    "SyntheticAdapter",
    "build_adapter",
]
