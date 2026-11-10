"""Adapter-independent canonical records.

Every dataset adapter converts its native layout into these records; the
ingestion pipeline only ever sees this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

EVENT_TYPES = ("diagnosis", "laboratory", "vital", "medication", "procedure", "visit", "other")
VALID_SPLITS = ("train", "valid", "test")
# Aliases seen in different releases (MEDS uses tuning / held_out).
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "tuning": "valid",
    "dev": "valid",
    "test": "test",
    "held_out": "test",
    "heldout": "test",
}


@dataclass
class CohortRow:
    """One row of cohort metadata = one CTPA procedure of one patient."""

    patient_id: str
    procedure_time_raw: str
    split_raw: str | None
    report_ref: str | None
    study_ref: str | None
    source_file: str
    source_row: int
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    access_scope: str | None = None
    labels: dict[str, str | None] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportRow:
    report_ref: str
    impression_text: str | None
    source_file: str
    source_row: int


@dataclass
class EventRow:
    patient_id: str
    time_raw: str | None
    code: str
    value_raw: str | None
    unit: str | None
    source_file: str
    source_row: int
    end_time_raw: str | None = None
    table_hint: str | None = None
    visit_id: str | None = None
    display_name: str | None = None


@dataclass
class ImagingRow:
    study_ref: str
    source_path: str | None
    modality: str | None
    source_file: str
    source_row: int
    slice_count: int | None = None
    spacing: list[float] | None = None
    orientation: dict[str, Any] | None = None


@dataclass
class FailedRow:
    """A rejected record.  Contains identifiers and a reason — never clinical text."""

    stage: str
    source_file: str
    source_row: int | None
    reason: str
    identifiers: dict[str, str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "source_file": self.source_file,
            "source_row": self.source_row,
            "reason": self.reason,
            "identifiers": self.identifiers,
        }


@dataclass
class LabObservation:
    """A laboratory measurement with both the original and the comparable form."""

    event_id: str
    concept_key: str
    concept_code: str
    event_time: datetime
    original_value: float
    original_unit: str | None
    canonical_value: float | None  # None => not directly comparable
    canonical_unit: str | None
    comparable: bool
    reason: str | None = None
