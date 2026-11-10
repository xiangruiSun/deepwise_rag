"""Deterministic query understanding: classification, rewriting and retrieval budgets.

Rules only — no model call sits in the request path.  Every decision is recorded in
the returned plan so an evaluation can attribute a gain or a loss to it, and each
part can be switched off independently (``RetrievalParams``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.corpus.config import RetrievalParams

# record-style identifiers such as REP-G00000012, LAB-0000004711 or PSN-0012345
IDENTIFIER = re.compile(r"\b[A-Z]{2,5}-[A-Z]{0,2}\d{4,}\b")
ENTITY_PREFIXES = ("PSN-",)  # identifiers of an entity (patient pseudonym), not of one record
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_LATEST = re.compile(r"\b(latest|current|currently|most recent|newest|up[- ]to[- ]date|now)\b", re.IGNORECASE)
_HISTORICAL = re.compile(r"\b(as of|before|prior to|until|earlier|previously|at the time)\b", re.IGNORECASE)
_MULTI = re.compile(r"\b(compare|both|all|each|every|across|list|versus|vs|difference between|"
                    r"what was documented for)\b", re.IGNORECASE)
_QUESTION = re.compile(r"\b(what|which|who|when|where|why|how|does|did|is|are|was|were|can|should)\b|\?",
                       re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")


@dataclass
class QueryPlan:
    original: str
    kind: str  # exact_identifier | keyword | semantic | temporal | multi_document | ambiguous
    lexical_query: str
    dense_query: str
    identifiers: list[str] = field(default_factory=list)
    entity_filters: dict[str, list[str]] = field(default_factory=dict)
    wants_latest: bool = False
    as_of: datetime | None = None
    expansions: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    params: RetrievalParams | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "identifiers": len(self.identifiers), "entity_filter": bool(self.entity_filters),
                "wants_latest": self.wants_latest, "as_of": self.as_of.isoformat() if self.as_of else None,
                "expansions": len(self.expansions), "sub_queries": len(self.sub_queries), "notes": self.notes}


def classify(query: str) -> str:
    words = _WORD.findall(query)
    identifiers = [i for i in IDENTIFIER.findall(query) if not i.startswith(ENTITY_PREFIXES)]
    if identifiers and len(words) <= 2 * len(identifiers) + 1:
        return "exact_identifier"
    if _LATEST.search(query) or _HISTORICAL.search(query) or _DATE.search(query):
        return "temporal"
    if _MULTI.search(query):
        return "multi_document"
    if not _QUESTION.search(query):
        return "ambiguous" if len(words) <= 1 else "keyword"
    return "semantic"


def expand_acronyms(query: str, acronyms: dict[str, str], limit: int) -> list[str]:
    """Full forms of the known abbreviations that appear in the query as whole words."""
    found: list[str] = []
    for word in _WORD.findall(query):
        full = acronyms.get(word) or acronyms.get(word.upper())
        if full and full.lower() not in query.lower() and full not in found:
            found.append(full)
    return found[:limit]


def plan_query(query: str, params: RetrievalParams, acronyms: dict[str, str] | None = None,
               as_of: datetime | None = None) -> QueryPlan:
    """Build the retrieval plan.  With routing and rewriting off this is the plain strategy."""
    text = " ".join(query.split())
    plan = QueryPlan(original=query, kind="semantic", lexical_query=text, dense_query=text, as_of=as_of)
    effective = params
    if params.query_rewriting and acronyms:
        plan.expansions = expand_acronyms(text, acronyms, limit=3)
        if plan.expansions:
            plan.lexical_query = f"{text} {' '.join(plan.expansions)}"
            plan.dense_query = plan.lexical_query
            plan.notes.append("acronym_expansion")
    if params.query_rewriting and params.max_rewrites > 0:
        parts = [p.strip() for p in re.split(r"\s+(?:and also|as well as|;)\s+", text) if len(p.split()) >= 3]
        if len(parts) > 1:  # bounded decomposition of a compound question
            plan.sub_queries = parts[: params.max_rewrites + 1]
            plan.notes.append("decomposition")
    if not params.query_routing:
        plan.params = effective
        return plan

    plan.kind = classify(text)
    identifiers = IDENTIFIER.findall(text)
    plan.identifiers = [i for i in identifiers if not i.startswith(ENTITY_PREFIXES)]
    entities = [i for i in identifiers if i.startswith(ENTITY_PREFIXES)]
    if entities:
        # a question about one entity must not be answered from another entity's records
        plan.entity_filters = {"x_patient_pseudo_id": entities}
        plan.notes.append("entity_scope")
    if _LATEST.search(text):
        plan.wants_latest = True
    date = _DATE.search(text)
    if plan.as_of is None and date and _HISTORICAL.search(text):
        plan.as_of = datetime.strptime(date.group(1), "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        plan.notes.append("historical_as_of")

    if plan.kind == "exact_identifier":
        effective = effective.model_copy(update={"strategy": "bm25", "rerank": False}
                                         | ({"bm25_top_k": 20} if params.adaptive_depth else {}))
    elif plan.kind == "keyword" and effective.strategy == "hybrid":
        effective = effective.model_copy(update={"rrf_weights": {"bm25": 1.5, "dense": 1.0}})
    elif plan.kind == "multi_document" and params.adaptive_depth:
        effective = effective.model_copy(update={
            "bm25_top_k": min(1000, effective.bm25_top_k * 2), "dense_top_k": min(1000, effective.dense_top_k * 2),
            "rerank_depth": min(200, effective.rerank_depth * 2), "final_k": min(100, effective.final_k + 4)})
    elif plan.kind == "ambiguous":
        plan.notes.append("ambiguous_query_fallback")
    if params.adaptive_depth and plan.kind in ("semantic", "temporal") and plan.entity_filters:
        # an entity-scoped question has a small candidate population: a shallow pass is enough
        effective = effective.model_copy(update={"bm25_top_k": min(effective.bm25_top_k, 40),
                                                 "dense_top_k": min(effective.dense_top_k, 40),
                                                 "rerank_depth": min(effective.rerank_depth, 10)})
    plan.params = effective
    return plan
