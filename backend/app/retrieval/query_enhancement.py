"""Optional query transformations (spec section 11.6).  Disabled in every baseline.

Every transformed query keeps a link to the original.  HyDE output is a
retrieval aid only: it is used to build a query vector and is never stored,
returned as evidence or citable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from app.core.config import QueryEnhancementConfig

# Small, explicit terminology map (lower-case phrase -> expansions).
MEDICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "pe": ("pulmonary embolism",),
    "pulmonary embolism": ("pulmonary embolus", "pulmonary thromboembolism", "filling defect"),
    "embolus": ("embolism", "thromboembolic"),
    "clot": ("thrombus", "embolus"),
    "saddle": ("main pulmonary arteries",),
    "right heart strain": ("right ventricular strain", "rv/lv ratio"),
    "rv strain": ("right ventricular strain", "right heart strain"),
    "pneumonia": ("consolidation", "airspace opacity"),
    "effusion": ("pleural fluid",),
    "chronic thromboembolic": ("webs and bands", "wall-adherent thrombus"),
    "emphysema": ("emphysematous",),
}


class TextCompleter(Protocol):
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str: ...


@dataclass
class EnhancedQuery:
    original: str
    method: str
    lexical_query: str
    dense_query: str
    dense_as_document: bool = False  # HyDE: encode the hypothetical text as a document
    expansions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original,
            "method": self.method,
            "lexical_query": self.lexical_query,
            "dense_query_is_generated": self.dense_as_document,
            "expansions": self.expansions,
            "warnings": self.warnings,
        }


class QueryEnhancer:
    def __init__(self, completer: TextCompleter | None = None) -> None:
        self.completer = completer

    def enhance(self, query: str, config: QueryEnhancementConfig) -> EnhancedQuery:
        method = config.method
        if method == "none":
            return EnhancedQuery(query, "none", query, query)
        if method == "rule_expansion":
            lowered = query.lower()
            expansions: list[str] = []
            for phrase, extra in MEDICAL_EXPANSIONS.items():
                if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered):
                    for term in extra:
                        if term not in lowered and term not in expansions:
                            expansions.append(term)
            expansions = expansions[: config.max_expansion_terms]
            lexical = query if not expansions else f"{query} {' '.join(expansions)}"
            return EnhancedQuery(query, method, lexical, query, expansions=expansions)
        if self.completer is None:
            return EnhancedQuery(
                query, "none", query, query,
                warnings=[f"query enhancement '{method}' needs an LLM backend; original query used"],
            )
        if method == "llm_rewrite":
            rewritten = self.completer.complete(
                "You rewrite clinical search queries for a radiology-report retrieval system. "
                "Keep every negation, laterality and qualifier. Do not add findings. "
                "Return only the rewritten query.",
                query, max_tokens=96, temperature=0.0,
            ).strip()
            if not rewritten:
                return EnhancedQuery(query, "none", query, query,
                                     warnings=["LLM rewrite returned no text; original query used"])
            return EnhancedQuery(query, method, rewritten, rewritten)
        if method == "hyde":
            hypothetical = self.completer.complete(
                "Write a short, plausible CT pulmonary angiography IMPRESSION that would answer the "
                "search request. This text is used only to build a search vector.",
                query, max_tokens=128, temperature=0.0,
            ).strip()
            if not hypothetical:
                return EnhancedQuery(query, "none", query, query,
                                     warnings=["HyDE returned no text; original query used"])
            return EnhancedQuery(query, method, query, hypothetical, dense_as_document=True)
        raise ValueError(f"unknown query enhancement '{method}'")
