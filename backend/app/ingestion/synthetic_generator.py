"""Deterministic synthetic cohort generator (spec section 35).

Everything written here is FABRICATED.  It exists so that the complete pipeline
(ingestion → indexing → retrieval → RAG → evaluation) can be exercised without
access-controlled INSPECT data.  Metrics computed on it verify that the code
works; they say nothing about clinical retrieval performance.

The files follow the paper-described INSPECT layout so that the legacy adapter
code path is exercised.  Deliberately difficult / broken records are included:
duplicate rows, orphan references, unparseable and conflicting timestamps,
negated findings, convertible and incompatible laboratory units, records dated
after the procedure (leakage probes), missing imaging and restricted cases.
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from app.core.hashing import make_case_id, stable_hash
from app.ingestion.adapters.synthetic import SYNTHETIC_MARKER

GENERATOR_VERSION = "1.0"
ANNOTATION_VERSION = "synthetic-rubric-v1"

ACUTE = ("central", "lobar", "segmental", "subsegmental")
PE_TYPES = (*ACUTE, "chronic", "none")
PE_WEIGHTS = (0.07, 0.11, 0.17, 0.10, 0.08, 0.47)
SECONDARY = ("none", "effusion", "consolidation", "nodule", "atelectasis", "emphysema")
SECONDARY_WEIGHTS = (0.34, 0.17, 0.15, 0.10, 0.14, 0.10)
NON_LATERAL = {"none", "atelectasis", "emphysema"}

PE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "central": (
        "Saddle pulmonary embolus extending into the {main} pulmonary arteries.",
        "Large central filling defects in the {main} pulmonary arteries consistent with acute pulmonary embolism.",
        "Extensive acute thromboembolic disease involving the {main} pulmonary arteries.",
    ),
    "lobar": (
        "Acute pulmonary emboli within the {lobe} lobar pulmonary artery.",
        "Filling defect in the {lobe} lobar artery, compatible with acute pulmonary thromboembolism.",
        "Lobar pulmonary embolism involving the {lobe}.",
    ),
    "segmental": (
        "Acute pulmonary embolism in segmental branches of the {lobe}.",
        "Segmental filling defects in the {lobe} pulmonary arteries consistent with acute PE.",
        "Thromboembolic filling defects within segmental arteries of the {lobe}.",
    ),
    "subsegmental": (
        "Small subsegmental pulmonary embolus in the {lobe}.",
        "Isolated subsegmental filling defect in the {lobe}, likely a small acute embolus.",
        "Tiny peripheral embolus in a subsegmental branch of the {lobe}.",
    ),
    "chronic": (
        "Eccentric wall-adherent thrombus in the {lobe} pulmonary artery compatible with chronic thromboembolic disease. No acute embolus.",
        "Webs and bands in the {lobe} arteries suggesting chronic pulmonary embolism. No acute filling defect.",
        "Findings of chronic thromboembolic disease in the {lobe} without acute pulmonary embolism.",
    ),
    "none": (
        "No evidence of pulmonary embolism.",
        "Negative for pulmonary thromboembolism to the segmental level.",
        "No filling defect to suggest pulmonary embolus.",
    ),
}
RV_TEMPLATES = {
    True: (
        "Right ventricular enlargement with flattening of the interventricular septum, suggestive of right heart strain.",
        "RV/LV ratio greater than 1, consistent with right ventricular strain.",
    ),
    False: ("No CT evidence of right heart strain.", "Normal RV/LV ratio."),
}
SECONDARY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "effusion": ("Small {side2} pleural effusion.", "Moderate {side2} pleural fluid collection."),
    "consolidation": (
        "{Side2} lower lobe consolidation concerning for pneumonia.",
        "Airspace opacity in the {side2} lower lobe, likely infectious.",
    ),
    "nodule": (
        "{size} mm pulmonary nodule in the {side2} upper lobe; follow-up recommended.",
        "Solid {size} mm nodule, {side2} upper lobe.",
    ),
    "atelectasis": (
        "Bibasilar atelectasis.",
        "Dependent subsegmental atelectasis at the lung bases.",
    ),
    "emphysema": (
        "Moderate centrilobular emphysema.",
        "Emphysematous changes in the upper lobes.",
    ),
}

HISTORY_CODES = {
    "ICD10CM/I50.9": ("Heart failure, unspecified", 0.14),
    "ICD10CM/C34.90": ("Malignant neoplasm of lung", 0.10),
    "ICD10CM/J44.9": ("Chronic obstructive pulmonary disease", 0.16),
    "ICD10CM/N18.3": ("Chronic kidney disease, stage 3", 0.12),
    "ICD10CM/E11.9": ("Type 2 diabetes mellitus", 0.20),
    "ICD10CM/I82.409": ("Acute deep vein thrombosis of lower extremity", 0.08),
}

# concept key -> (LOINC code, canonical unit, alternate unit, factor canonical->alternate)
LABS: dict[str, tuple[str, str, str | None, float]] = {
    "d_dimer": ("LOINC/48065-7", "ng/mL FEU", "mg/L FEU", 0.001),
    "troponin_i": ("LOINC/10839-9", "ng/mL", "ng/L", 1000.0),
    "bnp": ("LOINC/30934-4", "pg/mL", None, 1.0),
    "creatinine": ("LOINC/2160-0", "mg/dL", "umol/L", 88.4),
    "hemoglobin": ("LOINC/718-7", "g/dL", "g/L", 10.0),
    "wbc": ("LOINC/6690-2", "K/uL", "10*3/uL", 1.0),
    "lactate": ("LOINC/2524-7", "mmol/L", "mg/dL", 9.01),
    "heart_rate": ("LOINC/8867-4", "/min", "bpm", 1.0),
    "spo2": ("LOINC/59408-5", "%", None, 1.0),
}
LAB_PRESENCE = {
    "d_dimer": 0.72,
    "troponin_i": 0.70,
    "bnp": 0.55,
    "creatinine": 0.92,
    "hemoglobin": 0.92,
    "wbc": 0.90,
    "lactate": 0.45,
    "heart_rate": 0.95,
    "spo2": 0.95,
}


@dataclass
class Profile:
    pe: str
    side: str
    rv_strain: bool
    secondary: str
    side2: str

    @property
    def broad(self) -> str:
        return "acute" if self.pe in ACUTE else self.pe


@dataclass
class SynCase:
    patient_id: str
    procedure_time: datetime
    split: str
    profile: Profile
    age: float | None
    gender: str | None
    race: str | None
    ethnicity: str | None
    access_scope: str
    impression_id: str | None
    image_id: str | None
    impression: str
    history: list[str] = field(default_factory=list)
    labs_latest: dict[str, float] = field(default_factory=dict)  # canonical, comparable only
    has_image: bool = False

    @property
    def case_id(self) -> str:
        return make_case_id(self.patient_id, self.procedure_time)


# --------------------------------------------------------------------------- #
# Relevance rubric (hand-specified; applies to latent synthetic attributes)
# --------------------------------------------------------------------------- #
def rubric_grade(query: Profile, cand: Profile) -> int:
    """Graded case-level relevance.

    3 = same clot distribution AND matching RV strain, laterality and secondary finding
    2 = same clot distribution with exactly one of those attributes different
    1 = same broad class (acute / chronic / negative) only
    0 = different broad class

    Two PE-positive cases are never "highly similar" solely for being PE positive.
    """
    if query.broad != cand.broad:
        return 0
    if query.pe != cand.pe:
        return 1
    if query.pe == "none":
        if query.secondary != cand.secondary:
            return 1
        if query.secondary in NON_LATERAL or query.side2 == cand.side2:
            return 3
        return 2
    matches = (
        int(query.rv_strain == cand.rv_strain)
        + int(query.secondary == cand.secondary)
        + int(query.side == cand.side)
    )
    return 3 if matches == 3 else 2 if matches == 2 else 1


def target_grade(target: dict[str, Any], cand: Profile) -> int:
    """Relevance of a candidate to a free-text query with a partially specified target."""
    broad = target.get("broad") or ("acute" if target.get("pe") in ACUTE else target.get("pe"))
    if broad and broad != cand.broad:
        return 0
    if target.get("pe") and target["pe"] != cand.pe:
        return 1
    checks = []
    for attr in ("rv_strain", "secondary", "side", "side2"):
        if attr in target:
            checks.append(target[attr] == getattr(cand, attr))
    if not checks or all(checks):
        return 3
    return 2 if sum(not c for c in checks) == 1 else 1


# --------------------------------------------------------------------------- #
# Free-text evaluation queries (hand-written)
# --------------------------------------------------------------------------- #
TEXT_QUERIES: list[dict[str, Any]] = [
    # --- development (validation) queries
    {
        "query_id": "dev_t01", "stage": "dev", "category": "A",
        "query_text": "acute segmental pulmonary embolism without right heart strain",
        "target": {"pe": "segmental", "rv_strain": False},
    },
    {
        "query_id": "dev_t02", "stage": "dev", "category": "A",
        "query_text": "saddle embolus with right ventricular strain",
        "target": {"pe": "central", "rv_strain": True},
    },
    {
        "query_id": "dev_t03", "stage": "dev", "category": "E",
        "query_text": "no pulmonary embolism, pleural effusion present",
        "target": {"pe": "none", "secondary": "effusion"},
        "note": "negation: must not match PE-positive reports",
    },
    {
        "query_id": "dev_t04", "stage": "dev", "category": "E",
        "query_text": "clot in a small peripheral subsegmental branch, heart size normal",
        "target": {"pe": "subsegmental", "rv_strain": False},
        "note": "vocabulary mismatch and 'subsegmental atelectasis' lexical confound",
    },
    {
        "query_id": "dev_t05", "stage": "dev", "category": "B",
        "query_text": "acute pulmonary embolism",
        "target": {"broad": "acute"},
        "filters": {"age_min": 50, "age_max": 75},
    },
    {
        "query_id": "dev_t06", "stage": "dev", "category": "B",
        "query_text": "pulmonary embolism in a patient with heart failure",
        "target": {"broad": "acute"},
        "filters": {"diagnosis_codes": ["ICD10CM/I50.9"]},
    },
    {
        "query_id": "dev_t07", "stage": "dev", "category": "C",
        "query_text": "acute pulmonary embolism with markedly elevated D-dimer",
        "target": {"broad": "acute"},
        "filters": {"lab_filters": [{"concept": "d_dimer", "min_value": 2000.0}]},
    },
    {
        "query_id": "dev_t08", "stage": "dev", "category": "E",
        "query_text": "chronic thromboembolic disease, webs and bands",
        "target": {"pe": "chronic"},
        "note": "rare terminology",
    },
    {
        "query_id": "dev_t09", "stage": "dev", "category": "F",
        "query_text": "pulmonary embolism",
        "filters": {"age_min": 120, "age_max": 130},
        "expects_abstention": True,
        "note": "impossible filter: no eligible case may be returned",
    },
    # --- held-out test queries
    {
        "query_id": "test_t01", "stage": "test", "category": "A",
        "query_text": "lobar pulmonary embolism with right heart strain",
        "target": {"pe": "lobar", "rv_strain": True},
    },
    {
        "query_id": "test_t02", "stage": "test", "category": "E",
        "query_text": "negative for embolus with lower lobe pneumonia",
        "target": {"pe": "none", "secondary": "consolidation"},
    },
    {
        "query_id": "test_t03", "stage": "test", "category": "B",
        "query_text": "segmental pulmonary embolism",
        "target": {"pe": "segmental"},
        "filters": {"age_min": 60, "age_max": 90},
    },
    {
        "query_id": "test_t04", "stage": "test", "category": "C",
        "query_text": "pulmonary embolism with elevated troponin",
        "target": {"broad": "acute", "rv_strain": True},
        "filters": {"lab_filters": [{"concept": "troponin_i", "min_value": 0.05}]},
    },
    {
        "query_id": "test_t05", "stage": "test", "category": "E",
        "query_text": "no acute embolism but old adherent thrombus",
        "target": {"pe": "chronic"},
    },
    {
        "query_id": "test_t06", "stage": "test", "category": "F",
        "query_text": "pulmonary embolism",
        "filters": {"gender": "F", "age_min": 121, "age_max": 125},
        "expects_abstention": True,
    },
]


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class SyntheticGenerator:
    def __init__(self, n_patients: int = 170, seed: int = 20240607, dicom_studies: int = 6) -> None:
        self.n_patients = n_patients
        self.seed = seed
        self.dicom_studies = dicom_studies
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cases: list[SynCase] = []
        self.events: list[dict[str, Any]] = []
        self.imaging_rows: list[dict[str, Any]] = []
        self._img_counter = 0
        self._imp_counter = 0

    # ------------------------------------------------------------- utilities
    def _choice(self, items: tuple, weights: tuple | None = None):  # type: ignore[no-untyped-def]
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _profile(self) -> Profile:
        pe = self._choice(PE_TYPES, PE_WEIGHTS)
        side = "bilateral" if pe == "central" and self.rng.random() < 0.7 else self._choice(
            ("left", "right", "bilateral"), (0.35, 0.45, 0.20)
        )
        rv_prob = {"central": 0.8, "lobar": 0.45, "segmental": 0.2, "subsegmental": 0.04}.get(pe, 0.0)
        return Profile(
            pe=pe,
            side=side,
            rv_strain=self.rng.random() < rv_prob,
            secondary=self._choice(SECONDARY, SECONDARY_WEIGHTS),
            side2=self._choice(("left", "right")),
        )

    def _impression(self, p: Profile) -> str:
        lobe_side = p.side if p.side != "bilateral" else "bilateral"
        lobe = f"{lobe_side} lower lobe" if lobe_side != "bilateral" else "bilateral lower lobes"
        main = {"left": "left main", "right": "right main", "bilateral": "bilateral main"}[p.side]
        sentences = [self._choice(PE_TEMPLATES[p.pe]).format(lobe=lobe, main=main)]
        if p.pe in ACUTE:
            sentences.append(self._choice(RV_TEMPLATES[p.rv_strain]))
        if p.secondary != "none":
            sentences.append(
                self._choice(SECONDARY_TEMPLATES[p.secondary]).format(
                    side2=p.side2, Side2=p.side2.capitalize(), size=self.rng.choice((4, 6, 8, 11))
                )
            )
        if len(sentences) > 1 and self.rng.random() < 0.4:
            return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        # irregular whitespace on purpose: cleaning must normalise it without changing meaning
        return ("  " if self.rng.random() < 0.2 else " ").join(sentences)

    def _event(self, patient_id: str, when: datetime, code: str, table: str,
               value: Any = "", unit: str = "", end: datetime | None = None) -> None:
        self.events.append(
            {
                "patient_id": patient_id,
                "start": when.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end.strftime("%Y-%m-%d %H:%M:%S") if end else "",
                "code": code,
                "value": value,
                "unit": unit,
                "visit_id": f"v{stable_hash([patient_id, when.date().isoformat()], 8)}",
                "omop_table": table,
            }
        )

    def _lab_value(self, key: str, p: Profile, history: list[str]) -> float:
        g = self.rng.gauss
        ln = self.rng.lognormvariate
        if key == "d_dimer":
            base = {"acute": 3200.0, "chronic": 900.0, "none": 420.0}[p.broad]
            if p.pe == "central":
                base = 6500.0
            if p.pe == "subsegmental":
                base = 1500.0
            return round(base * ln(0, 0.35), 0)
        if key == "troponin_i":
            return round((0.22 if p.rv_strain else 0.012) * ln(0, 0.45), 3)
        if key == "bnp":
            base = 620.0 if p.rv_strain else (340.0 if "ICD10CM/I50.9" in history else 70.0)
            return round(base * ln(0, 0.4), 0)
        if key == "creatinine":
            return round(max(0.4, g(2.1 if "ICD10CM/N18.3" in history else 0.95, 0.2)), 2)
        if key == "hemoglobin":
            return round(max(6.0, g(10.4 if "ICD10CM/C34.90" in history else 13.4, 1.1)), 1)
        if key == "wbc":
            return round(max(2.0, g(15.5 if p.secondary == "consolidation" else 8.1, 1.8)), 1)
        if key == "lactate":
            return round(max(0.5, g(3.3 if (p.pe == "central" and p.rv_strain) else 1.3, 0.4)), 1)
        if key == "heart_rate":
            return round(max(45.0, g(116.0 if p.rv_strain else 86.0, 9.0)), 0)
        if key == "spo2":
            return round(min(100.0, max(78.0, g(89.5 if p.pe == "central" else 95.8, 1.8))), 0)
        raise KeyError(key)

    def _emit_lab(self, patient_id: str, when: datetime, key: str, canonical: float,
                  force_unit: str | None = None) -> bool:
        """Write one lab event; returns True when the unit is convertible."""
        code, unit, alt_unit, factor = LABS[key]
        table = "measurement"
        if force_unit is not None:
            self._event(patient_id, when, code, table, canonical, force_unit)
            return False
        if alt_unit and self.rng.random() < 0.3:
            self._event(patient_id, when, code, table, round(canonical * factor, 4), alt_unit)
        else:
            self._event(patient_id, when, code, table, canonical, unit)
        return True

    # ------------------------------------------------------------- cohort
    def _build(self) -> None:
        rng = self.rng
        patient_ids = [str(100000 + i * 37 + rng.randrange(0, 30)) for i in range(self.n_patients)]
        order = patient_ids[:]
        rng.shuffle(order)
        n_train = int(0.70 * len(order))
        n_valid = int(0.15 * len(order))
        split_of = {
            pid: ("train" if i < n_train else "valid" if i < n_train + n_valid else "test")
            for i, pid in enumerate(order)
        }
        for idx, pid in enumerate(patient_ids):
            base_age = rng.randint(22, 90)
            gender = rng.choices(("F", "M", None), weights=(0.50, 0.47, 0.03))[0]
            race = rng.choices(
                ("White", "Asian", "Black or African American", "Other", None),
                weights=(0.50, 0.20, 0.10, 0.12, 0.08),
            )[0]
            ethnicity = rng.choices(
                ("Not Hispanic or Latino", "Hispanic or Latino", None), weights=(0.74, 0.18, 0.08)
            )[0]
            history = [code for code, (_, prob) in HISTORY_CODES.items() if rng.random() < prob]
            first = datetime(2015, 1, 1) + timedelta(
                days=rng.randrange(0, 2200), hours=rng.randrange(0, 24), minutes=rng.randrange(0, 60)
            )
            n_studies = 2 if idx % 9 == 0 else 1
            for hist_code in history:
                self._event(pid, first - timedelta(days=rng.randrange(40, 900)), hist_code,
                            "condition_occurrence")
            self._event(pid, first - timedelta(hours=5), "Visit/ER", "visit_occurrence",
                        end=first + timedelta(days=2))
            for study_idx in range(n_studies):
                when = first + timedelta(days=study_idx * rng.randrange(120, 400))
                profile = self._profile()
                if study_idx == 1 and "ICD10CM/I82.409" not in history and rng.random() < 0.5:
                    profile.pe = "chronic"
                    profile.rv_strain = False
                self._imp_counter += 1
                self._img_counter += 1
                case = SynCase(
                    patient_id=pid,
                    procedure_time=when,
                    split=split_of[pid],
                    profile=profile,
                    age=None if idx % 41 == 7 else float(base_age + study_idx),
                    gender=gender,
                    race=race,
                    ethnicity=ethnicity,
                    access_scope="restricted" if idx % 29 == 3 else "research",
                    impression_id=f"imp_{self._imp_counter:05d}",
                    image_id=f"img_{self._img_counter:05d}",
                    impression=self._impression(profile),
                    history=list(history),
                )
                self._case_events(case)
                self.cases.append(case)

    def _case_events(self, case: SynCase) -> None:
        rng = self.rng
        pid, when, p = case.patient_id, case.procedure_time, case.profile
        for key, presence in LAB_PRESENCE.items():
            if rng.random() > presence:
                continue
            value = self._lab_value(key, p, case.history)
            lab_time = when - timedelta(hours=rng.randrange(1, 9), minutes=rng.randrange(0, 60))
            if key == "d_dimer" and rng.random() < 0.06:
                # incompatible unit: must be excluded from numeric comparison
                self._emit_lab(pid, lab_time, key, round(value / 250.0, 2), force_unit="arb'U/mL")
            else:
                self._emit_lab(pid, lab_time, key, value)
                case.labs_latest[key] = value
            # older repeated measurements so that 7/30/90-day lookbacks differ
            if rng.random() < 0.35:
                self._emit_lab(pid, when - timedelta(days=rng.randrange(10, 28)), key,
                               self._lab_value(key, Profile("none", "left", False, "none", "left"),
                                               case.history))
            if rng.random() < 0.2:
                self._emit_lab(pid, when - timedelta(days=rng.randrange(40, 85)), key,
                               self._lab_value(key, Profile("none", "left", False, "none", "left"),
                                               case.history))
        # --- records dated AFTER the procedure (must never enter query-time features)
        if p.pe in ACUTE:
            code = "ICD10CM/I26.02" if p.pe == "central" else "ICD10CM/I26.99"
            self._event(pid, when + timedelta(hours=rng.randrange(3, 30)), code, "condition_occurrence")
            self._event(pid, when + timedelta(hours=rng.randrange(1, 6)), "RxNorm/5224", "drug_exposure")
            if rng.random() < 0.5:
                self._emit_lab(pid, when + timedelta(days=2), "d_dimer", 99999.0)
        elif p.pe == "chronic":
            self._event(pid, when + timedelta(days=1), "ICD10CM/I27.24", "condition_occurrence")
        if "ICD10CM/I82.409" in case.history and rng.random() < 0.6:
            self._event(pid, when - timedelta(days=rng.randrange(5, 30)), "RxNorm/1364430",
                        "drug_exposure")

    # ------------------------------------------------------------- imaging
    def _volume(self, p: Profile) -> np.ndarray:
        """Tiny fake CT volume in Hounsfield units whose intensity pattern depends on the
        latent profile.  It is NOT anatomically meaningful."""
        r = self.np_rng
        vol = r.normal(-820, 40, size=(12, 24, 24))
        vol[:, :3, :] = 40 + r.normal(0, 8, size=(12, 3, 24))
        vol[:, -3:, :] = 40 + r.normal(0, 8, size=(12, 3, 24))
        vol[:, :, 10:14] = 250 + r.normal(0, 15, size=(12, 24, 4))  # contrast-filled vessels
        if p.pe != "none":
            n_blobs = {"central": 6, "lobar": 4, "segmental": 3, "subsegmental": 1, "chronic": 2}[p.pe]
            for _ in range(n_blobs):
                z, y = int(r.integers(2, 10)), int(r.integers(5, 19))
                vol[z : z + 2, y : y + 2, 10:14] = (70 if p.pe == "chronic" else 45) + r.normal(0, 5)
        if p.rv_strain:
            vol[4:8, 14:20, 14:20] = 300 + r.normal(0, 10, size=(4, 6, 6))
        cols = slice(3, 9) if p.side2 == "left" else slice(15, 21)
        if p.secondary == "effusion":
            vol[:, 18:21, cols] = 8 + r.normal(0, 4, size=(12, 3, 6))
        elif p.secondary == "consolidation":
            vol[6:11, 13:18, cols] = 35 + r.normal(0, 10, size=(5, 5, 6))
        elif p.secondary == "nodule":
            vol[3:5, 5:7, cols.start : cols.start + 2] = 60
        elif p.secondary == "atelectasis":
            vol[:, 17:19, 3:21] = -250 + r.normal(0, 30, size=(12, 2, 18))
        elif p.secondary == "emphysema":
            vol[:5, 4:12, 3:9] = -960 + r.normal(0, 10, size=(5, 8, 6))
        return np.clip(vol, -1024, 1500).astype(np.int16)

    def _write_imaging(self, root: Path) -> None:
        img_dir = root / "imaging"
        (img_dir / "volumes").mkdir(parents=True, exist_ok=True)
        dicom_written = 0
        for i, case in enumerate(self.cases):
            if i % 13 == 5:  # no imaging row at all
                continue
            assert case.image_id is not None
            if i % 31 == 11:  # row present but file missing
                self.imaging_rows.append(
                    {"image_id": case.image_id, "path": f"imaging/volumes/{case.image_id}.npy",
                     "modality": "CT", "num_slices": 12, "spacing": "[2.5, 1.2, 1.2]",
                     "orientation": '{"axes": "ZYX"}'}
                )
                continue
            volume = self._volume(case.profile)
            if dicom_written < self.dicom_studies:
                rel = f"imaging/dicom/{case.image_id}"
                write_dicom_series(root / rel, volume, case.image_id, spacing=(2.5, 1.2, 1.2))
                dicom_written += 1
            else:
                rel = f"imaging/volumes/{case.image_id}.npy"
                np.save(root / rel, volume)
            case.has_image = True
            self.imaging_rows.append(
                {"image_id": case.image_id, "path": rel, "modality": "CT", "num_slices": 12,
                 "spacing": "[2.5, 1.2, 1.2]", "orientation": '{"axes": "ZYX"}'}
            )
        # invalid source reference: imaging row for a study that is not in the cohort
        self.imaging_rows.append(
            {"image_id": "img_99999", "path": "imaging/volumes/img_99999.npy", "modality": "CT",
             "num_slices": 12, "spacing": "[2.5, 1.2, 1.2]", "orientation": "{}"}
        )

    # ------------------------------------------------------------- writing
    def write(self, root: Path) -> dict[str, Any]:
        if root.exists():
            marker = root / SYNTHETIC_MARKER
            existing = [p for p in root.iterdir() if not p.name.startswith(".")]  # e.g. .gitkeep
            if existing and not marker.exists():
                raise ValueError(
                    f"refusing to overwrite {root}: it is not empty and is not a synthetic dataset"
                )
            for path in existing:  # regenerate in place; hidden placeholder files are kept
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        root.mkdir(parents=True, exist_ok=True)
        self._build()
        self._write_imaging(root)
        self._write_cohort(root)
        self._write_impressions(root)
        self._write_events(root)
        with (root / "imaging" / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["image_id", "path", "modality", "num_slices", "spacing", "orientation"]
            )
            writer.writeheader()
            writer.writerows(self.imaging_rows)
        eval_info = self._write_evaluation(root)
        info = {
            "synthetic": True,
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "n_patients": self.n_patients,
            "n_cases": len(self.cases),
            "notice": "Fabricated data for pipeline verification only. Not clinical evidence.",
            **eval_info,
        }
        (root / SYNTHETIC_MARKER).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return info

    def _write_cohort(self, root: Path) -> None:
        fields = ["patient_id", "procedure_time", "split", "impression_id", "image_id", "num_slices",
                  "age", "gender", "race", "ethnicity", "access_scope", "pe_positive_nlp",
                  "1_month_mortality", "12_month_PH"]
        rows = []
        for i, c in enumerate(self.cases):
            rows.append(
                {
                    "patient_id": c.patient_id,
                    "procedure_time": c.procedure_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "split": c.split,
                    # one case points at an impression id that does not exist
                    "impression_id": "imp_missing" if i == 17 else c.impression_id,
                    "image_id": c.image_id,
                    "num_slices": 12,
                    "age": "" if c.age is None else c.age,
                    "gender": c.gender or "",
                    "race": c.race or "",
                    "ethnicity": c.ethnicity or "",
                    "access_scope": c.access_scope,
                    "pe_positive_nlp": str(c.profile.pe in ACUTE),
                    "1_month_mortality": str(c.profile.pe == "central" and c.profile.rv_strain
                                             and self.rng.random() < 0.3),
                    "12_month_PH": str(c.profile.pe == "chronic" and self.rng.random() < 0.5),
                }
            )
        if len(self.cases) > 17:
            self.cases[17].impression_id = None
        broken = [
            dict(rows[3]),  # exact duplicate primary key
            {**rows[5], "patient_id": "999001", "procedure_time": "not-a-date"},
            {**rows[6], "patient_id": "999002", "split": "holdout2"},
            {**rows[7], "patient_id": "", "procedure_time": "2019-01-01 00:00:00"},
        ]
        with (root / "cohort.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows + broken)

    def _write_impressions(self, root: Path) -> None:
        with (root / "impressions.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["impression_id", "impressions"])
            for c in self.cases:
                if c.impression_id:
                    writer.writerow([c.impression_id, c.impression])
            first = next(c for c in self.cases if c.impression_id)
            writer.writerow([first.impression_id, "Conflicting duplicate row for an existing id."])
            writer.writerow(["imp_orphan", "No evidence of pulmonary embolism."])
            writer.writerow(["imp_empty", "   "])

    def _write_events(self, root: Path) -> None:
        ehr = root / "ehr"
        ehr.mkdir()
        sample = self.cases[2]
        when = sample.procedure_time
        # conflicting measurement: two different values at the same timestamp
        conflict_case = self.cases[4]
        t = conflict_case.procedure_time - timedelta(minutes=30)
        conflict_case.labs_latest.pop("hemoglobin", None)
        broken: list[dict[str, Any]] = [
            {"patient_id": "888888", "start": "2018-03-01 10:00:00", "end": "", "code": "LOINC/718-7",
             "value": 12.1, "unit": "g/dL", "visit_id": "vx", "omop_table": "measurement"},
            {"patient_id": sample.patient_id, "start": "13/45/2019", "end": "", "code": "LOINC/718-7",
             "value": 11.0, "unit": "g/dL", "visit_id": "vx", "omop_table": "measurement"},
            {"patient_id": sample.patient_id, "start": when.strftime("%Y-%m-%d %H:%M:%S"),
             "end": (when - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"), "code": "Visit/IP",
             "value": "", "unit": "", "visit_id": "vx", "omop_table": "visit_occurrence"},
            {"patient_id": sample.patient_id, "start": when.strftime("%Y-%m-%d %H:%M:%S"), "end": "",
             "code": "", "value": "", "unit": "", "visit_id": "vx", "omop_table": "measurement"},
            {"patient_id": conflict_case.patient_id, "start": t.strftime("%Y-%m-%d %H:%M:%S"),
             "end": "", "code": "LOINC/718-7", "value": 9.1, "unit": "g/dL", "visit_id": "vc",
             "omop_table": "measurement"},
            {"patient_id": conflict_case.patient_id, "start": t.strftime("%Y-%m-%d %H:%M:%S"),
             "end": "", "code": "LOINC/718-7", "value": 14.6, "unit": "g/dL", "visit_id": "vc",
             "omop_table": "measurement"},
            {"patient_id": sample.patient_id,
             "start": (when - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"), "end": "",
             "code": "LOINC/2524-7", "value": "<0.5", "unit": "mmol/L", "visit_id": "vx",
             "omop_table": "measurement"},
        ]
        fields = ["patient_id", "start", "end", "code", "value", "unit", "visit_id", "omop_table"]
        shards: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
        for ev in [*self.events, *broken]:
            pid = str(ev["patient_id"])
            shards[int(pid) % 3 if pid.isdigit() else 0].append(ev)
        for shard, rows in shards.items():
            path = ehr / f"events_{shard}.csv.gz"
            # mtime=0 keeps the gzip bytes (and therefore source hashes) deterministic
            with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                import io

                text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                text.flush()
                text.detach()

    # ------------------------------------------------------------- evaluation data
    def _write_evaluation(self, root: Path) -> dict[str, Any]:
        eval_dir = root / "eval"
        eval_dir.mkdir()
        truth_dir = root / "_truth"
        truth_dir.mkdir()
        valid_cases = [c for c in self.cases if c.access_scope == "research" and c.impression_id]
        corpus = [c for c in valid_cases if c.split == "train"]
        with (truth_dir / "latent_profiles.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.cases:
                fh.write(json.dumps({"case_id": c.case_id, "split": c.split, **asdict(c.profile)}) + "\n")

        queries: list[dict[str, Any]] = []
        qrels: list[dict[str, Any]] = []

        def judged(query_id: str, cand: SynCase) -> bool:
            # ~12% of pairs stay unjudged on purpose (pooled judgments are incomplete)
            return int(stable_hash([query_id, cand.case_id], 8), 16) % 100 >= 12

        def add_qrel(q: dict[str, Any], cand: SynCase, grade: int, modalities: list[str]) -> None:
            qrels.append(
                {
                    "query_id": q["query_id"],
                    "query_case_id": q.get("case_id"),
                    "query_text": q.get("query_text"),
                    "candidate_case_id": cand.case_id,
                    "relevance_grade": grade,
                    "query_category": q["category"],
                    "relevant_modalities": modalities,
                    "reviewer_id": "synthetic_rubric",
                    "annotation_version": ANNOTATION_VERSION,
                    "confidence": 1.0,
                    "evidence_references": [f"report:{cand.impression_id}"],
                    "adjudication_status": "rubric",
                    "label_source": "synthetic_rubric",
                }
            )

        for stage, split in (("dev", "valid"), ("test", "test")):
            pool = [c for c in valid_cases if c.split == split]
            for n, qc in enumerate(pool[:22], start=1):
                multimodal = bool(qc.labs_latest) and qc.has_image
                q: dict[str, Any] = {
                    "query_id": f"{stage}_c{n:02d}",
                    "stage": stage,
                    "category": "D" if (multimodal and n % 2 == 0) else "A",
                    "query_mode": "case",
                    "case_id": qc.case_id,
                    "filters": {},
                    "expects_abstention": False,
                }
                queries.append(q)
                for cand in corpus:
                    if cand.patient_id == qc.patient_id or not judged(q["query_id"], cand):
                        continue
                    add_qrel(q, cand, rubric_grade(qc.profile, cand.profile),
                             ["text", "laboratory", "image"] if q["category"] == "D" else ["text"])

        for spec in TEXT_QUERIES:
            q = {  # type: ignore[no-redef]
                "query_id": spec["query_id"],
                "stage": spec["stage"],
                "category": spec["category"],
                "query_mode": "text",
                "query_text": spec["query_text"],
                "filters": spec.get("filters", {}),
                "expects_abstention": spec.get("expects_abstention", False),
                "note": spec.get("note"),
            }
            queries.append(q)
            if "target" not in spec:
                continue
            for cand in corpus:
                if not judged(q["query_id"], cand):
                    continue
                grade = target_grade(spec["target"], cand.profile)
                if not self._passes(cand, q["filters"]):
                    grade = 0  # outside the eligible population
                add_qrel(q, cand, grade, ["text"] + (["laboratory"] if spec["category"] == "C" else []))

        # a handful of weak labels, kept separate from rubric/clinician labels
        weak = []
        for q in queries[:3]:
            for cand in corpus[:5]:
                weak.append(
                    {"query_id": q["query_id"], "query_case_id": q.get("case_id"),
                     "candidate_case_id": cand.case_id,
                     "relevance_grade": 1, "query_category": q["category"],
                     "relevant_modalities": ["text"], "reviewer_id": "weak_same_label",
                     "annotation_version": "weak-v1", "confidence": 0.3, "evidence_references": [],
                     "adjudication_status": "none", "label_source": "weak"}
                )

        def dump(name: str, rows: list[dict[str, Any]]) -> None:
            with (eval_dir / name).open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")

        dump("queries.jsonl", queries)
        dump("qrels.jsonl", qrels)
        dump("qrels_weak.jsonl", weak)
        dump("generation_eval.jsonl", self._generation_items(queries))
        (eval_dir / "POOLING.md").write_text(
            "Synthetic judgments are produced by the hand-specified rubric in "
            "`app/ingestion/synthetic_generator.py` over latent attributes; ~12% of query/candidate "
            "pairs are intentionally left unjudged. Real benchmarks must use the pooled, blinded "
            "annotation workflow in `app/evaluation/pooling.py`.\n",
            encoding="utf-8",
        )
        return {
            "query_set_version": "synthetic-queries-" + stable_hash(queries, 12),
            "qrels_version": ANNOTATION_VERSION + "-" + stable_hash(qrels, 12),
            "n_queries": len(queries),
            "n_qrels": len(qrels),
        }

    def _passes(self, cand: SynCase, filters: dict[str, Any]) -> bool:
        if "age_min" in filters and (cand.age is None or cand.age < filters["age_min"]):
            return False
        if "age_max" in filters and (cand.age is None or cand.age > filters["age_max"]):
            return False
        if "gender" in filters and cand.gender != filters["gender"]:
            return False
        for code in filters.get("diagnosis_codes", []):
            if code not in cand.history:
                return False
        for lab in filters.get("lab_filters", []):
            value = cand.labs_latest.get(lab["concept"])
            if value is None:
                return False
            if "min_value" in lab and value < lab["min_value"]:
                return False
            if "max_value" in lab and value > lab["max_value"]:
                return False
        return True

    def _generation_items(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_case = {c.case_id: c for c in self.cases}
        items: list[dict[str, Any]] = []
        case_queries = [q for q in queries if q["query_mode"] == "case"]
        for stage in ("dev", "test"):
            staged = [q for q in case_queries if q["stage"] == stage][:8]
            for n, q in enumerate(staged, start=1):
                qc = by_case[q["case_id"]]
                kind = ("comparison", "labs", "multi_case", "insufficient")[n % 4]
                item: dict[str, Any] = {
                    "item_id": f"gen_{stage}_{n:02d}",
                    "stage": stage,
                    "query_id": q["query_id"],
                    "kind": kind,
                    "expects_abstention": False,
                    "expected_terms": [],
                    "expected_numeric_facts": [],
                }
                if kind == "comparison":
                    item["question"] = (
                        "How does the selected patient's documented imaging impression compare "
                        "with the retrieved cases?"
                    )
                    item["expected_terms"] = _expected_terms(qc.profile)
                elif kind == "labs":
                    item["question"] = "Which laboratory measurements are available for comparison?"
                    item["expected_numeric_facts"] = [
                        {"concept": key, "value": value, "case_id": qc.case_id}
                        for key, value in sorted(qc.labs_latest.items())
                        if key in {"d_dimer", "troponin_i", "bnp", "creatinine", "hemoglobin",
                                   "wbc", "lactate"}
                    ]
                elif kind == "multi_case":
                    item["question"] = (
                        "What imaging and laboratory characteristics are shared by these cases?"
                    )
                    item["expected_terms"] = _expected_terms(qc.profile)[:1]
                else:
                    item["question"] = (
                        "Which anticoagulant dose should be given to this patient, and in which "
                        "zip code does the patient live?"
                    )
                    item["expects_abstention"] = True
                    item["abstention_reason"] = "treatment recommendation / unavailable attribute"
                items.append(item)
        for q in queries:
            if q.get("expects_abstention"):
                items.append(
                    {"item_id": f"gen_{q['query_id']}", "stage": q["stage"], "query_id": q["query_id"],
                     "kind": "insufficient", "question": "Summarise the similar historical cases.",
                     "expects_abstention": True, "abstention_reason": "no eligible cases",
                     "expected_terms": [], "expected_numeric_facts": []}
                )
        return items


def _expected_terms(p: Profile) -> list[str]:
    terms = {
        "central": ["main pulmonary arter"],
        "lobar": ["lobar"],
        "segmental": ["segmental"],
        "subsegmental": ["subsegmental"],
        "chronic": ["chronic"],
        "none": [],
    }[p.pe]
    return terms + ([{"effusion": "pleural", "consolidation": "lower lobe", "nodule": "nodule",
                      "atelectasis": "atelectasis", "emphysema": "emphysem"}[p.secondary]]
                    if p.secondary != "none" else [])


def write_dicom_series(directory: Path, volume: np.ndarray, study_id: str,
                       spacing: tuple[float, float, float] = (2.5, 1.2, 1.2),
                       shuffle_seed: int | None = 7) -> None:
    """Write a minimal CT DICOM series (one file per slice, stored out of order)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid(entropy_srcs=[study_id, "series"])
    study_uid = generate_uid(entropy_srcs=[study_id, "study"])
    order = list(range(volume.shape[0]))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)  # file order must not be trusted
    intercept = -1024.0
    for file_idx, z in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid(entropy_srcs=[study_id, str(z)])
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(directory / f"{file_idx:04d}.dcm"), {}, file_meta=meta,
                         preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientID = "SYNTHETIC"
        ds.PatientName = "SYNTHETIC^DATA"
        ds.Rows, ds.Columns = int(volume.shape[1]), int(volume.shape[2])
        ds.PixelSpacing = [spacing[1], spacing[2]]
        ds.SliceThickness = spacing[0]
        ds.ImagePositionPatient = [0.0, 0.0, float(z) * spacing[0]]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.InstanceNumber = z + 1
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = intercept
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        stored = (volume[z].astype(np.float64) - intercept).clip(0, 65535).astype(np.uint16)
        ds.PixelData = stored.tobytes()
        ds.save_as(str(directory / f"{file_idx:04d}.dcm"), enforce_file_format=True)


def generate_synthetic_dataset(root: Path, n_patients: int = 170, seed: int = 20240607) -> dict[str, Any]:
    return SyntheticGenerator(n_patients=n_patients, seed=seed).write(root)
