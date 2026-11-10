"""Patient-level aggregation of retrieved documents into distinct clinical cases
(spec section 12)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import AggregationConfig
from app.ranking.fusion import reciprocal_rank_fusion


@dataclass
class RetrievedDocument:
    document_id: str
    case_id: str
    patient_id: str
    source_id: str  # e.g. report id
    content: str
    chunk_index: int = 0
    source_type: str = "radiology_report"
    score: float = 0.0  # final document-level score used for ordering
    score_type: str = "bm25"
    branch_scores: dict[str, float] = field(default_factory=dict)
    branch_ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None
    rank: int = 0


@dataclass
class CaseCandidate:
    case_id: str
    patient_id: str
    text_score: float
    text_rank: int
    evidence: list[RetrievedDocument] = field(default_factory=list)
    n_documents: int = 0


class CaseAggregator:
    """Groups documents by case, de-duplicates and ranks distinct cases.

    A case never gains score merely by having more retrieved chunks: every method
    is bounded by the case's best documents (``top_n_mean`` averages over the
    documents actually present, so additional weaker chunks cannot raise it).
    """

    def __init__(self, config: AggregationConfig, max_evidence_per_case: int = 3) -> None:
        self.config = config
        self.max_evidence = max_evidence_per_case

    def aggregate(
        self,
        documents: list[RetrievedDocument],
        branch_rankings: dict[str, list[RetrievedDocument]] | None = None,
        exclude_patient_ids: set[str] | None = None,
        exclude_case_ids: set[str] | None = None,
    ) -> list[CaseCandidate]:
        exclude_patient_ids = exclude_patient_ids or set()
        exclude_case_ids = exclude_case_ids or set()
        groups: dict[str, list[RetrievedDocument]] = {}
        for position, doc in enumerate(documents, start=1):
            if doc.patient_id in exclude_patient_ids or doc.case_id in exclude_case_ids:
                continue
            if not doc.rank:
                doc.rank = position
            groups.setdefault(doc.case_id, []).append(doc)

        scores: dict[str, float] = {}
        method = self.config.method
        if method == "case_rrf":
            if not branch_rankings:
                raise ValueError("case_rrf aggregation requires per-branch document rankings")
            case_rankings: dict[str, list[str]] = {}
            for branch, docs in branch_rankings.items():
                seen: list[str] = []
                for doc in docs:  # first (best) document of each case defines its branch rank
                    if doc.case_id in groups and doc.case_id not in seen:
                        seen.append(doc.case_id)
                case_rankings[branch] = seen
            fused = reciprocal_rank_fusion(case_rankings, k=self.config.rrf_k)
            scores = {item.item_id: item.score for item in fused}
            for case_id in groups:
                scores.setdefault(case_id, 0.0)
        else:
            for case_id, docs in groups.items():
                unique = self._deduplicate(docs)
                if method == "best_score":
                    scores[case_id] = max(d.score for d in unique)
                elif method == "best_rank":
                    scores[case_id] = 1.0 / min(d.rank for d in unique)
                elif method == "top_n_mean":
                    top = sorted((d.score for d in unique), reverse=True)[: self.config.top_n]
                    scores[case_id] = sum(top) / len(top)
                else:  # pragma: no cover - guarded by config validation
                    raise ValueError(f"unknown aggregation method '{method}'")

        ordered = sorted(groups, key=lambda c: (-scores[c], min(d.rank for d in groups[c]), c))
        candidates: list[CaseCandidate] = []
        for rank, case_id in enumerate(ordered, start=1):
            docs = self._deduplicate(groups[case_id])
            docs.sort(key=lambda d: (d.rank, d.document_id))
            candidates.append(
                CaseCandidate(
                    case_id=case_id,
                    patient_id=docs[0].patient_id,
                    text_score=scores[case_id],
                    text_rank=rank,
                    evidence=docs[: self.max_evidence],
                    n_documents=len(groups[case_id]),
                )
            )
        return candidates

    @staticmethod
    def _deduplicate(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Keep the best-ranked instance of each (source, chunk) and drop exact-text repeats
        from the same source report (overlapping windows)."""
        best: dict[tuple[str, int], RetrievedDocument] = {}
        for doc in docs:
            key = (doc.source_id, doc.chunk_index)
            if key not in best or doc.rank < best[key].rank:
                best[key] = doc
        seen_text: set[tuple[str, str]] = set()
        unique: list[RetrievedDocument] = []
        for doc in sorted(best.values(), key=lambda d: (d.rank, d.document_id)):
            text_key = (doc.source_id, doc.content)
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            unique.append(doc)
        return unique
