"""Answer-quality analysis that needs no model: contradictions between sources,
failure analysis of user feedback, and a per-query cost model.

All three are deliberately simple and inspectable.  The contradiction detector is a
rule-based screen (polarity and antonym mismatches between closely matching sentences of
*different* documents); it flags candidates for the reader and is not a clinical NLI model.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.corpus.context import WITHHELD, ContextBundle
from app.db.models import CorpusFeedback

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATION = {"no", "not", "without", "negative", "absent", "denies", "excluded", "unremarkable", "never"}
_ANTONYMS = [("resolved", "unresolved"), ("present", "absent"), ("positive", "negative"),
             ("increased", "decreased"), ("improved", "worsened"), ("stable", "unstable"),
             ("benign", "malignant"), ("normal", "abnormal"), ("started", "discontinued")]
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "was", "is", "were", "are", "at",
         "by", "after"}
MIN_SHARED = 0.5


def _terms(sentence: str) -> set[str]:
    return {w for w in _WORD.findall(sentence.lower()) if w not in _STOP}


def detect_conflicts(bundle: ContextBundle, max_conflicts: int = 5) -> list[dict[str, Any]]:
    """Pairs of sentences from different documents that describe the same thing with opposite polarity.

    Each conflict names both sources and, when the documents are dated, which one is newer —
    so the caller can show "sources disagree; S2 is more recent" instead of silently picking one."""
    sentences: list[tuple[str, str, set[str], Any]] = []
    for item in bundle.items:
        for sentence in _SENTENCE.split(item.text):
            terms = _terms(sentence)
            if len(terms) >= 3 and WITHHELD not in sentence:
                sentences.append((item.source_id, sentence.strip(), terms, item.chunk))
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, (source_a, text_a, terms_a, chunk_a) in enumerate(sentences):
        for source_b, text_b, terms_b, chunk_b in sentences[i + 1:]:
            if chunk_a.document_id == chunk_b.document_id or (source_a, source_b) in seen:
                continue
            kind = None
            for left, right in _ANTONYMS:
                if (left in terms_a and right in terms_b) or (right in terms_a and left in terms_b):
                    kind = f"{left}/{right}"
                    break
            shared = terms_a & terms_b
            polarity_words = _NEGATION | {w for pair in _ANTONYMS for w in pair}
            overlap = len(shared - polarity_words) / max(1, min(len(terms_a - polarity_words),
                                                                len(terms_b - polarity_words)))
            if kind is None and bool(terms_a & _NEGATION) != bool(terms_b & _NEGATION):
                kind = "negation"
            if kind is None or overlap < MIN_SHARED:
                continue
            # the same subject is required: same patient when both carry one, otherwise the same record
            patient_a = chunk_a.metadata.get("x_patient_pseudo_id")
            patient_b = chunk_b.metadata.get("x_patient_pseudo_id")
            if patient_a and patient_b and patient_a != patient_b:
                continue
            newer = None
            if chunk_a.event_time and chunk_b.event_time and chunk_a.event_time != chunk_b.event_time:
                newer = source_a if chunk_a.event_time > chunk_b.event_time else source_b
            seen.add((source_a, source_b))
            conflicts.append({"sources": [source_a, source_b], "kind": kind, "newer_source": newer,
                              "documents": [chunk_a.document_id, chunk_b.document_id],
                              "statements": [text_a[:200], text_b[:200]]})
            if len(conflicts) >= max_conflicts:
                return conflicts
    return conflicts


# ------------------------------------------------------------------ feedback
def feedback_failure_analysis(session_factory: sessionmaker[Session], days: int = 30,
                              min_reports: int = 2, now: datetime | None = None) -> dict[str, Any]:
    """Aggregate negative feedback into failure clusters and candidate regression tests.

    Feedback stores no question text, only a query hash, so a candidate names the hash, the
    chunks users marked and the failure categories.  Turning a candidate into an evaluation row
    is a human step (``needs_human_review``): the reviewer supplies the question and the labels."""
    now = now or datetime.utcnow()
    with session_factory() as session:
        rows = list(session.scalars(select(CorpusFeedback).where(CorpusFeedback.created_at >= now - timedelta(days=days))))
    down = [r for r in rows if r.rating == "down"]
    by_category = Counter(c for r in down for c in (r.categories or ["unspecified"]))
    by_strategy: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = by_strategy.setdefault(r.strategy or "default", {"up": 0, "down": 0})
        bucket[r.rating] = bucket.get(r.rating, 0) + 1
    clusters: dict[tuple[str, str], list[CorpusFeedback]] = {}
    for r in down:
        if r.query_hash:
            clusters.setdefault((r.tenant_id, r.query_hash), []).append(r)
    candidates = []
    for (tenant_id, query_hash), items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_reports:
            continue
        candidates.append({
            "tenant_id": tenant_id, "query_hash": query_hash, "reports": len(items),
            "users": len({i.user_id for i in items}),
            "categories": dict(Counter(c for i in items for c in i.categories)),
            "flagged_chunks": dict(Counter(c for i in items for c in i.chunk_ids).most_common(10)),
            "index_versions": sorted({i.index_version for i in items if i.index_version}),
            "trace_ids": [i.trace_id for i in items][:5], "needs_human_review": True})
    return {"window_days": days, "feedback": len(rows), "negative": len(down),
            "negative_rate": round(len(down) / len(rows), 4) if rows else None,
            "by_category": dict(by_category.most_common()),
            "by_strategy": {k: {**v, "negative_rate": round(v.get("down", 0) / max(1, sum(v.values())), 4)}
                            for k, v in sorted(by_strategy.items())},
            "regression_candidates": candidates}


# ---------------------------------------------------------------------- cost
class CostInputs(BaseModel):
    """Unit prices and throughputs.  Every default is a placeholder ASSUMPTION to be replaced by
    the operator's own contract prices and measured throughput — none of them is a measurement."""

    prompt_tokens: float = Field(ge=0)
    completion_tokens: float = Field(ge=0)
    rerank_pairs: float = Field(default=0, ge=0)
    queries_per_month: float = Field(default=1_000_000, gt=0)
    gpu_hour_usd: float = Field(default=2.0, gt=0)
    llm_prompt_tokens_per_gpu_second: float = Field(default=20_000, gt=0)
    llm_completion_tokens_per_gpu_second: float = Field(default=1_500, gt=0)
    rerank_pairs_per_gpu_second: float = Field(default=2_000, gt=0)
    utilisation: float = Field(default=0.5, gt=0, le=1)
    fixed_monthly_usd: float = Field(default=0, ge=0)  # search cluster, cache, storage, ...


def estimate_query_cost(inputs: CostInputs) -> dict[str, Any]:
    """GPU-seconds and cost of one query from token counts (measured) and unit prices (assumed)."""
    llm_seconds = (inputs.prompt_tokens / inputs.llm_prompt_tokens_per_gpu_second
                   + inputs.completion_tokens / inputs.llm_completion_tokens_per_gpu_second)
    rerank_seconds = inputs.rerank_pairs / inputs.rerank_pairs_per_gpu_second
    per_second = inputs.gpu_hour_usd / 3600.0 / inputs.utilisation
    variable = (llm_seconds + rerank_seconds) * per_second
    fixed = inputs.fixed_monthly_usd / inputs.queries_per_month
    gpu_seconds_month = (llm_seconds + rerank_seconds) * inputs.queries_per_month
    return {"gpu_seconds_per_query": {"generation": round(llm_seconds, 5), "reranking": round(rerank_seconds, 5)},
            "usd_per_1k_queries": {"generation": round(1000 * llm_seconds * per_second, 4),
                                   "reranking": round(1000 * rerank_seconds * per_second, 4),
                                   "fixed": round(1000 * fixed, 4), "total": round(1000 * (variable + fixed), 4)},
            "gpus_at_average_load": round(gpu_seconds_month / (30 * 86_400) / inputs.utilisation, 3),
            "assumptions": inputs.model_dump(),
            "note": "Token and pair counts are inputs measured by an evaluation run; prices and throughputs "
                    "are assumptions. Average-load GPU count ignores peaks; size for peak with the capacity tool."}
