"""Deterministic citation verification and claim-support assessment (spec 17.4).

``CitationVerifier`` answers "is this reference real and allowed?":
  1. the cited source id exists;
  2. it belongs to an authorized case of this request (query or retrieved);
  3. it was part of the constructed evidence context;
  4. it maps to the correct underlying record (id, owner case and content hash).

A valid reference does NOT mean the claim is supported.  ``assess_claim_support``
is a separate, heuristic check (numbers, negation polarity, lexical grounding)
used for post-generation verification and as an automated diagnostic metric.
It must be calibrated against human review and is not a safety guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.hashing import sha256_text
from app.core.security import AccessPolicy
from app.db.models import ClinicalCase, ClinicalEvent, ImagingStudy, RadiologyReport
from app.rag.evidence import EvidenceBundle, EvidenceItem
from app.schemas.rag import Citation, Claim, RagAnswer


@dataclass
class CitationReport:
    checked: int = 0
    invalid: list[dict[str, str]] = field(default_factory=list)
    uncited_claims: int = 0
    removed_claims: int = 0
    unsupported_claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.invalid

    def as_dict(self) -> dict[str, Any]:
        return {
            "citations_checked": self.checked,
            "invalid_citations": self.invalid,
            "uncited_claims": self.uncited_claims,
            "removed_claims": self.removed_claims,
            "unsupported_claims": self.unsupported_claims,
            "all_citations_valid": self.valid,
        }


class CitationVerifier:
    def __init__(self, session: Session, policy: AccessPolicy) -> None:
        self.session = session
        self.policy = policy

    def _record_matches(self, item: EvidenceItem) -> str | None:
        """Check 4: the evidence item still maps to the authoritative record."""
        case = self.session.get(ClinicalCase, item.case_id)
        if case is None:
            return "case no longer exists"
        if not self.policy.allows(case.access_scope):
            return "case is not authorized"
        if item.record_type == "report":
            report = self.session.get(RadiologyReport, item.record_id)
            if report is None or report.case_id != item.case_id:
                return "report does not belong to the cited case"
            if sha256_text(report.impression_text) != item.content_hash:
                return "report content differs from the cited evidence"
        elif item.record_type in ("laboratory", "diagnosis"):
            event = self.session.get(ClinicalEvent, item.record_id)
            if event is None or event.patient_id != case.patient_id:
                return "clinical event does not belong to the cited patient"
            if item.record_type == "laboratory" and event.numeric_value != item.data.get("original_value"):
                return "laboratory value differs from the cited evidence"
        elif item.record_type == "imaging":
            study = self.session.get(ImagingStudy, item.record_id)
            if study is None or study.case_id != item.case_id:
                return "imaging study does not belong to the cited case"
        else:
            return f"unknown record type '{item.record_type}'"
        return None

    def verify(self, answer: RagAnswer, bundle: EvidenceBundle, enforce: bool,
               require_citations: bool) -> CitationReport:
        """Validate every citation.  With ``enforce`` invalid references are removed and
        (when citations are required) claims left without any valid reference are dropped."""
        report = CitationReport()
        sources = bundle.by_source_id()
        allowed_cases = set(bundle.all_case_ids)
        record_cache: dict[str, str | None] = {}
        used: dict[str, EvidenceItem] = {}

        def check(source_id: str) -> str | None:
            item = sources.get(source_id)
            if item is None:
                return "source id does not exist in the evidence context"
            if item.case_id not in allowed_cases:
                return "source does not belong to an authorized retrieved case"
            if source_id not in record_cache:
                record_cache[source_id] = self._record_matches(item)
            return record_cache[source_id]

        for section in (answer.summary, answer.similarities, answer.differences):
            kept: list[Claim] = []
            for claim in section:
                valid: list[str] = []
                for source_id in claim.citations:
                    report.checked += 1
                    problem = check(source_id)
                    if problem is None:
                        valid.append(source_id)
                        used[source_id] = sources[source_id]
                    else:
                        report.invalid.append({"source_id": source_id, "reason": problem,
                                               "claim": claim.text[:120]})
                if not valid:
                    report.uncited_claims += 1
                if enforce:
                    claim.citations = valid
                    if require_citations and not valid:
                        report.removed_claims += 1
                        continue
                    support = assess_claim_support(claim, [sources[s] for s in valid])
                    if not support["supported"]:
                        report.unsupported_claims.append({"claim": claim.text[:160], **support})
                        report.removed_claims += 1
                        continue
                kept.append(claim)
            section[:] = kept
        answer.citations = [
            Citation(source_id=sid, record_type=item.record_type, record_id=item.record_id,  # type: ignore[arg-type]
                     case_id=item.case_id, role=item.role, excerpt=item.text[:400])  # type: ignore[arg-type]
            for sid, item in sorted(used.items(), key=lambda kv: int(kv[0][1:]))
        ]
        return report


# --------------------------------------------------------------------------- #
# Claim-level support heuristics
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]{3,}")
_NEG = re.compile(r"\b(no|not|without|negative|absent|negated)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(r"\b(?:case_[0-9a-f]+|S\d+|imp_\w+|img_\w+|ev_[0-9a-f]+)\b")
_STOP = {
    "the", "and", "for", "with", "that", "this", "case", "cases", "query", "retrieved", "both",
    "document", "documented", "documents", "states", "versus", "impression", "are", "was", "were",
    "has", "have", "does", "not", "but", "its", "from", "than", "while", "which", "mention", "shows",
    "show", "patient", "patients", "affirmed", "negated", "converted", "comparison", "between",
    "source", "wording", "each", "these", "there", "also", "other",
}


def _stem(word: str) -> str:
    """Crude prefix stem so that 'emboli', 'embolus' and 'embolism' ground each other."""
    return word[:5]


def _numbers(text: str) -> set[float]:
    return {round(float(n), 6) for n in _NUMBER.findall(_IDENTIFIER.sub(" ", text))}


def citation_grounding(claim: Claim, cited: list[EvidenceItem]) -> list[bool]:
    """Per citation: does the cited record share any content word or number with the claim?

    A reference can be real and authorized and still have nothing to do with the statement it
    is attached to (small models pad claims with extra source ids).  This is a coarse
    diagnostic: a correct "case X does not mention it" reference also shares nothing."""
    words = {_stem(w) for w in _WORD.findall(_IDENTIFIER.sub(" ", claim.text).lower()) if w not in _STOP}
    numbers = _numbers(claim.text)
    grounded: list[bool] = []
    for item in cited:
        item_words = {_stem(w) for w in _WORD.findall(item.text.lower())}
        concept = item.data.get("concept_key")
        if concept:
            item_words.update(_stem(w) for w in _WORD.findall(str(concept).lower()))
        item_numbers = _numbers(item.text)
        for key in ("original_value", "canonical_value"):
            value = item.data.get(key)
            if isinstance(value, (int, float)):
                item_numbers.add(round(float(f"{value:g}"), 6))
        grounded.append(bool(words & item_words) or bool(numbers & item_numbers))
    return grounded


def assess_claim_support(claim: Claim, cited: list[EvidenceItem]) -> dict[str, Any]:
    """Heuristic support check of one claim against its cited evidence.

    * numeric: every number in the claim must appear in the cited evidence
      (original or canonical values);
    * lexical: a minimum share of the claim's content words must be grounded;
    * polarity: a claim that asserts a finding without negation must not rest
      only on evidence sentences that negate it (coarse check).
    """
    if not cited:
        return {"supported": False, "reason": "no valid citation", "numeric_ok": False, "lexical_overlap": 0.0}
    evidence_text = " ".join(i.text for i in cited)
    evidence_numbers = _numbers(evidence_text)
    for item in cited:
        for key in ("original_value", "canonical_value"):
            value = item.data.get(key)
            if isinstance(value, (int, float)):
                evidence_numbers.add(round(float(value), 6))
                evidence_numbers.add(round(float(f"{value:g}"), 6))
    claim_numbers = _numbers(claim.text)
    missing_numbers = sorted(n for n in claim_numbers if n not in evidence_numbers)
    words = {_stem(w) for w in _WORD.findall(_IDENTIFIER.sub(" ", claim.text).lower()) if w not in _STOP}
    evidence_words = {_stem(w) for w in _WORD.findall(evidence_text.lower())}
    for item in cited:
        concept = item.data.get("concept_key")
        if concept:
            evidence_words.update(_stem(w) for w in _WORD.findall(str(concept).lower()))
    if not words and not claim_numbers:
        # nothing that could be checked against a record: never count it as supported
        return {"supported": False, "reason": "claim has no checkable content", "numeric_ok": True,
                "lexical_overlap": 0.0, "polarity_ok": True}
    overlap = len(words & evidence_words) / len(words) if words else 1.0
    polarity_ok = True
    claim_negated = bool(_NEG.search(claim.text))
    if not claim_negated and words:
        relevant = [s for s in re.split(r"(?<=[.;])\s+|\n+", evidence_text)
                    if words & {_stem(w) for w in _WORD.findall(s.lower())}]
        if relevant and all(_NEG.search(s) for s in relevant) and overlap < 1.0:
            polarity_ok = False
    supported = not missing_numbers and overlap >= 0.5 and polarity_ok
    reason = None
    if missing_numbers:
        reason = f"numbers not found in cited evidence: {missing_numbers}"
    elif overlap < 0.5:
        reason = "claim wording is not grounded in the cited evidence"
    elif not polarity_ok:
        reason = "cited evidence only negates the asserted finding"
    return {"supported": supported, "reason": reason, "numeric_ok": not missing_numbers,
            "lexical_overlap": round(overlap, 3), "polarity_ok": polarity_ok}
