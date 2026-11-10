"""Haystack 2.x integration.

The retrieval stages of this project are plain, independently testable Python
classes.  This module exposes them as Haystack ``@component`` s and assembles a
Haystack ``Pipeline`` (retrievers → document-level fusion → case aggregation),
so they can be combined with other Haystack components.  ``RetrievalService``
does not depend on Haystack; the dependency is optional (``pip install
inspect-rag[haystack]``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.core.config import AggregationConfig, FusionConfig
from app.core.errors import DependencyUnavailable
from app.indexing.backend import FilterSpec
from app.ranking.aggregation import CaseAggregator, RetrievedDocument
from app.ranking.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.retrievers import BM25Retriever, DenseRetriever

try:
    from haystack import Document, Pipeline, component
except ImportError as exc:  # pragma: no cover - optional dependency
    raise DependencyUnavailable(
        "Haystack is not installed; install the optional dependency group 'haystack'"
    ) from exc


def to_haystack(doc: RetrievedDocument) -> Document:
    return Document(
        id=doc.document_id,
        content=doc.content,
        score=doc.score,
        meta={
            "case_id": doc.case_id, "patient_id": doc.patient_id, "source_id": doc.source_id,
            "source_type": doc.source_type, "chunk_index": doc.chunk_index,
            "branch_scores": dict(doc.branch_scores), "branch_ranks": dict(doc.branch_ranks),
        },
    )


def from_haystack(doc: Document) -> RetrievedDocument:
    meta = doc.meta
    return RetrievedDocument(
        document_id=doc.id, case_id=meta["case_id"], patient_id=meta["patient_id"],
        source_id=meta.get("source_id", ""), source_type=meta.get("source_type", "radiology_report"),
        content=doc.content or "", chunk_index=int(meta.get("chunk_index", 0)), score=float(doc.score or 0.0),
        branch_scores=dict(meta.get("branch_scores", {})), branch_ranks=dict(meta.get("branch_ranks", {})),
    )


@component
class CaseBM25Retriever:
    """Pre-filtered BM25 retrieval over report text."""

    def __init__(self, retriever: BM25Retriever, top_k: int = 50) -> None:
        self.retriever, self.top_k = retriever, top_k

    @component.output_types(documents=list[Document])
    def run(self, query: str, filters: FilterSpec | None = None) -> dict[str, Any]:
        docs = self.retriever.retrieve(query, filters or FilterSpec(), self.top_k)
        return {"documents": [to_haystack(d) for d in docs]}


@component
class CaseDenseRetriever:
    """Pre-filtered kNN retrieval with the configured query encoder."""

    def __init__(self, retriever: DenseRetriever, top_k: int = 50, num_candidates: int = 200) -> None:
        self.retriever, self.top_k, self.num_candidates = retriever, top_k, num_candidates

    @component.output_types(documents=list[Document])
    def run(self, query: str, filters: FilterSpec | None = None) -> dict[str, Any]:
        vector = self.retriever.encode(query)
        docs = self.retriever.retrieve(vector, filters or FilterSpec(), self.top_k, self.num_candidates)
        return {"documents": [to_haystack(d) for d in docs]}


@component
class DocumentFusion:
    """Document-level RRF or normalised weighted-score fusion of two ranked lists."""

    def __init__(self, config: FusionConfig, top_k: int = 100) -> None:
        self.config, self.top_k = config, top_k

    @component.output_types(documents=list[Document])
    def run(self, bm25: list[Document], dense: list[Document]) -> dict[str, Any]:
        lists = {"bm25": bm25, "dense": dense}
        by_id: dict[str, Document] = {}
        for docs in lists.values():
            for doc in docs:
                kept = by_id.setdefault(doc.id, doc)
                if kept is not doc:
                    kept.meta["branch_scores"].update(doc.meta["branch_scores"])
                    kept.meta["branch_ranks"].update(doc.meta["branch_ranks"])
        if self.config.method == "rrf":
            fused = reciprocal_rank_fusion({n: [d.id for d in docs] for n, docs in lists.items()},
                                           k=self.config.rrf_k, weights=self.config.weights)
        else:
            fused = weighted_score_fusion({n: {d.id: float(d.score or 0.0) for d in docs} for n, docs in lists.items()},
                                          weights=self.config.weights, normalization=self.config.normalization)
        out = [replace(by_id[item.item_id], score=item.score) for item in fused[: self.top_k]]
        return {"documents": out}


@component
class CaseAggregation:
    """Collapses ranked documents into distinct clinical cases."""

    def __init__(self, config: AggregationConfig) -> None:
        self.aggregator = CaseAggregator(config)

    @component.output_types(cases=list[dict[str, Any]])
    def run(self, documents: list[Document], exclude_patient_ids: list[str] | None = None) -> dict[str, Any]:
        cases = self.aggregator.aggregate([from_haystack(d) for d in documents],
                                          exclude_patient_ids=set(exclude_patient_ids or []))
        return {"cases": [{"case_id": c.case_id, "rank": c.text_rank, "score": c.text_score,
                           "evidence": [d.document_id for d in c.evidence]} for c in cases]}


def build_hybrid_pipeline(bm25: BM25Retriever, dense: DenseRetriever, fusion: FusionConfig,
                          aggregation: AggregationConfig, bm25_top_k: int = 50, dense_top_k: int = 50,
                          num_candidates: int = 200, fused_top_k: int = 100) -> Pipeline:
    """BM25 + dense → document-level fusion → case aggregation as a Haystack pipeline."""
    pipeline = Pipeline()
    pipeline.add_component("bm25", CaseBM25Retriever(bm25, bm25_top_k))
    pipeline.add_component("dense", CaseDenseRetriever(dense, dense_top_k, num_candidates))
    pipeline.add_component("fusion", DocumentFusion(fusion, fused_top_k))
    pipeline.add_component("aggregation", CaseAggregation(aggregation))
    pipeline.connect("bm25.documents", "fusion.bm25")
    pipeline.connect("dense.documents", "fusion.dense")
    pipeline.connect("fusion.documents", "aggregation.documents")
    return pipeline
