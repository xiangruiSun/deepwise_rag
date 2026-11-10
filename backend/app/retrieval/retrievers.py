"""Independently testable first-stage retrievers: BM25, dense, image, laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.errors import DependencyUnavailable
from app.embeddings.base import CachedEmbedder
from app.indexing.backend import FilterSpec, Hit, SearchBackend
from app.ranking.aggregation import RetrievedDocument
from app.retrieval.laboratory import CaseLabFeatures, LaboratorySimilarityScorer, LabSimilarity


def _require_index(backend: SearchBackend, index: str) -> None:
    if not backend.index_exists(index):
        raise DependencyUnavailable(
            f"search index '{index}' has not been built; run the indexing command for this recipe",
            {"index": index},
        )


def _stable_order(hits: list[Hit]) -> list[Hit]:
    """Deterministic order for equal scores.

    Identical impressions (common for short reports) receive identical scores.  Breaking
    those ties by case id and chunk index — not by opaque, index-specific document ids —
    makes rankings reproducible across indexes and search backends."""
    return sorted(hits, key=lambda h: (-h.score, str(h.source.get("case_id", "")),
                                       int(h.source.get("chunk_index", 0)), h.id))


def _to_document(hit: Hit, branch: str, rank: int) -> RetrievedDocument:
    src = hit.source
    return RetrievedDocument(
        document_id=hit.id,
        case_id=src["case_id"],
        patient_id=src["patient_id"],
        source_id=src.get("source_id", ""),
        source_type=src.get("source_type", "radiology_report"),
        content=src.get("content", ""),
        chunk_index=int(src.get("chunk_index", 0)),
        score=hit.score,
        score_type=branch,
        branch_scores={branch: hit.score},
        branch_ranks={branch: rank},
        rank=rank,
    )


class BM25Retriever:
    name = "bm25"

    def __init__(self, backend: SearchBackend, index: str) -> None:
        self.backend, self.index = backend, index

    def retrieve(self, query: str, filters: FilterSpec, top_k: int) -> list[RetrievedDocument]:
        _require_index(self.backend, self.index)
        hits = _stable_order(self.backend.bm25_search(self.index, query, filters, top_k))
        return [_to_document(h, self.name, r) for r, h in enumerate(hits, start=1)]


class DenseRetriever:
    name = "dense"

    def __init__(self, backend: SearchBackend, index: str, embedder: CachedEmbedder) -> None:
        self.backend, self.index, self.embedder = backend, index, embedder

    def encode(self, query: str, as_document: bool = False) -> np.ndarray:
        if as_document:
            return self.embedder.encode_documents([query])[0]
        return self.embedder.encode_queries([query])[0]

    def retrieve(self, vector: np.ndarray, filters: FilterSpec, top_k: int,
                 num_candidates: int) -> list[RetrievedDocument]:
        _require_index(self.backend, self.index)
        hits = _stable_order(self.backend.knn_search(self.index, vector, filters, top_k, num_candidates))
        return [_to_document(h, self.name, r) for r, h in enumerate(hits, start=1)]


@dataclass
class CaseScore:
    case_id: str
    patient_id: str
    score: float
    detail: Any = None


class ImageRetriever:
    """Cosine kNN over study-level image representations, collapsed to cases."""

    name = "image"

    def __init__(self, backend: SearchBackend, index: str) -> None:
        self.backend, self.index = backend, index

    def available(self) -> bool:
        return self.backend.index_exists(self.index)

    def retrieve(self, vector: np.ndarray, filters: FilterSpec, top_k: int,
                 num_candidates: int) -> list[CaseScore]:
        _require_index(self.backend, self.index)
        hits = self.backend.knn_search(self.index, vector, filters, top_k * 2, num_candidates)
        best: dict[str, CaseScore] = {}
        for hit in hits:  # several studies of one case: keep the most similar
            case_id = hit.source["case_id"]
            if case_id not in best or hit.score > best[case_id].score:
                best[case_id] = CaseScore(case_id, hit.source["patient_id"], hit.score,
                                          {"study_id": hit.source.get("study_id")})
        return sorted(best.values(), key=lambda c: (-c.score, c.case_id))[:top_k]


class LaboratoryRetriever:
    """Exhaustive laboratory-similarity ranking over an eligible case population.

    Feasible because laboratory vectors are tiny; candidates are the cases that
    already passed structured pre-filtering."""

    name = "laboratory"

    def __init__(self, scorer: LaboratorySimilarityScorer) -> None:
        self.scorer = scorer

    def score_all(self, query: CaseLabFeatures,
                  candidates: dict[str, CaseLabFeatures]) -> dict[str, LabSimilarity]:
        return {case_id: self.scorer.score(query, feats) for case_id, feats in candidates.items()}

    def retrieve(self, query: CaseLabFeatures, candidates: dict[str, tuple[str, CaseLabFeatures]],
                 top_k: int) -> list[CaseScore]:
        scored: list[CaseScore] = []
        for case_id, (patient_id, feats) in candidates.items():
            sim = self.scorer.score(query, feats)
            if sim.available and sim.similarity is not None:
                scored.append(CaseScore(case_id, patient_id, sim.similarity, sim))
        return sorted(scored, key=lambda c: (-c.score, c.case_id))[:top_k]
