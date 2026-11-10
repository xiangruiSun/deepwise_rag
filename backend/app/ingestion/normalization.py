"""Clinical normalisation: code parsing, event typing, and explicit unit conversion.

Rules enforced here (spec section 7, stage C):

* original code, value, unit and timestamp are always preserved;
* conversions happen only for explicitly listed unit pairs of a known concept;
* anything else is marked *not comparable* instead of being guessed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Code parsing and event typing
# --------------------------------------------------------------------------- #
CODE_SYSTEM_EVENT_TYPE = {
    "LOINC": "laboratory",
    "ICD10CM": "diagnosis",
    "ICD9CM": "diagnosis",
    "ICD10": "diagnosis",
    "SNOMED": "diagnosis",
    "RXNORM": "medication",
    "NDC": "medication",
    "CPT4": "procedure",
    "ICD10PCS": "procedure",
    "HCPCS": "procedure",
    "VISIT": "visit",
}

OMOP_TABLE_EVENT_TYPE = {
    "measurement": "laboratory",
    "condition_occurrence": "diagnosis",
    "drug_exposure": "medication",
    "procedure_occurrence": "procedure",
    "visit_occurrence": "visit",
    "visit_detail": "visit",
    "observation": "other",
    "device_exposure": "other",
    "death": "other",
    "person": "other",
}

# LOINC codes that are vital signs rather than laboratory tests.
VITAL_SIGN_CODES = {
    "LOINC/8867-4",  # heart rate
    "LOINC/9279-1",  # respiratory rate
    "LOINC/8480-6",  # systolic blood pressure
    "LOINC/8462-4",  # diastolic blood pressure
    "LOINC/8310-5",  # body temperature
    "LOINC/59408-5",  # SpO2 by pulse oximetry
    "LOINC/2708-6",  # oxygen saturation in arterial blood
}


def parse_code(code: str) -> tuple[str | None, str]:
    """Split ``SYSTEM/value`` into ``(system, value)``; codes without a system keep ``None``."""
    code = code.strip()
    if "/" in code:
        system, value = code.split("/", 1)
        return system.strip() or None, value.strip()
    return None, code


def classify_event(code: str, table_hint: str | None) -> str:
    if code in VITAL_SIGN_CODES:
        return "vital"
    if table_hint:
        mapped = OMOP_TABLE_EVENT_TYPE.get(table_hint.strip().lower())
        if mapped and mapped != "other":
            return mapped
    system, _ = parse_code(code)
    if system:
        return CODE_SYSTEM_EVENT_TYPE.get(system.upper(), "other")
    return "other"


def normalize_diagnosis_code(code: str) -> str:
    """Canonical ``SYSTEM/CODE`` with upper-cased parts (dots preserved)."""
    system, value = parse_code(code)
    value = value.upper()
    return f"{system.upper()}/{value}" if system else value


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
_UNIT_ALIASES = {
    "mcg/l": "ug/l",
    "µg/l": "ug/l",
    "μg/l": "ug/l",
    "mcg/ml": "ug/ml",
    "µg/ml": "ug/ml",
    "μg/ml": "ug/ml",
    "µmol/l": "umol/l",
    "μmol/l": "umol/l",
    "micromol/l": "umol/l",
    "10*3/ul": "k/ul",
    "10^3/ul": "k/ul",
    "x10e3/ul": "k/ul",
    "thou/ul": "k/ul",
    "10*9/l": "k/ul",
    "10^9/l": "k/ul",
    "bpm": "/min",
    "beats/min": "/min",
    "ng/ml feu": "ng/ml feu",
    "mg/l feu": "mg/l feu",
    "ug/ml feu": "ug/ml feu",
    "percent": "%",
}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = unicodedata.normalize("NFKC", str(unit)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text or text in {"nan", "none", "null"}:
        return None
    return _UNIT_ALIASES.get(text, text)


@dataclass(frozen=True)
class LabConcept:
    key: str
    display_name: str
    codes: tuple[str, ...]
    canonical_unit: str
    # explicit (normalised source unit -> multiplicative factor into canonical unit)
    conversions: dict[str, float] = field(default_factory=dict)

    def to_canonical(self, value: float, unit: str | None) -> tuple[float | None, str | None]:
        """Return ``(canonical_value, reason_if_not_comparable)``."""
        norm = normalize_unit(unit)
        if norm is None:
            return None, "missing unit"
        if norm == self.canonical_unit:
            return float(value), None
        factor = self.conversions.get(norm)
        if factor is None:
            return None, f"unsupported unit '{unit}' for {self.key}"
        return float(value) * factor, None


# The registry is intentionally small and explicit.  Extend it through
# ``LabConceptRegistry.from_config`` rather than adding heuristics.
DEFAULT_LAB_CONCEPTS: tuple[LabConcept, ...] = (
    LabConcept(
        "d_dimer",
        "D-dimer (FEU)",
        ("LOINC/48065-7", "LOINC/48066-5"),
        "ng/ml feu",
        {"ug/ml feu": 1000.0, "mg/l feu": 1000.0, "ug/l feu": 1.0},
    ),
    LabConcept(
        "troponin_i",
        "Troponin I",
        ("LOINC/10839-9", "LOINC/89579-7"),
        "ng/ml",
        {"ng/l": 0.001, "pg/ml": 0.001, "ug/l": 1.0},
    ),
    LabConcept(
        "bnp", "B-type natriuretic peptide", ("LOINC/30934-4",), "pg/ml", {"ng/l": 1.0}
    ),
    LabConcept(
        "creatinine",
        "Creatinine (serum)",
        ("LOINC/2160-0",),
        "mg/dl",
        {"umol/l": 1.0 / 88.4},
    ),
    LabConcept("hemoglobin", "Hemoglobin", ("LOINC/718-7",), "g/dl", {"g/l": 0.1}),
    LabConcept("wbc", "Leukocytes", ("LOINC/6690-2",), "k/ul", {"/ul": 0.001}),
    LabConcept("platelets", "Platelets", ("LOINC/777-3",), "k/ul", {"/ul": 0.001}),
    LabConcept("lactate", "Lactate", ("LOINC/2524-7",), "mmol/l", {"mg/dl": 1.0 / 9.01}),
    LabConcept("heart_rate", "Heart rate", ("LOINC/8867-4",), "/min", {}),
    LabConcept("spo2", "Oxygen saturation (pulse oximetry)", ("LOINC/59408-5",), "%", {}),
)


class LabConceptRegistry:
    def __init__(self, concepts: tuple[LabConcept, ...] = DEFAULT_LAB_CONCEPTS) -> None:
        self._by_key = {c.key: c for c in concepts}
        self._by_code: dict[str, LabConcept] = {}
        for concept in concepts:
            for code in concept.codes:
                if code in self._by_code:
                    raise ValueError(f"lab code {code} is mapped to more than one concept")
                self._by_code[code] = concept

    def by_code(self, code: str) -> LabConcept | None:
        return self._by_code.get(code)

    def by_key(self, key: str) -> LabConcept | None:
        return self._by_key.get(key)

    def keys(self) -> list[str]:
        return sorted(self._by_key)

    def concepts(self) -> list[LabConcept]:
        return [self._by_key[k] for k in self.keys()]


def parse_numeric(value: str | float | int | None) -> float | None:
    """Parse a numeric measurement.  Censored values such as ``<0.01`` are *not*
    numeric measurements and return ``None`` (the raw text is preserved elsewhere)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if number != number else number  # NaN check
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
