"""Authorised hybrid retrieval over the active chunk index.

Order of operations for every request::

    credential -> AuthContext -> pre-filter (tenant, groups, validity, caller filters)
      -> BM25 and/or kNN over the SAME filtered population -> RRF -> optional rerank

The tenant and permission filter is built from the authenticated context only and is
part of both the lexical and the vector query, so a document outside the caller's
scope is never scored, fused, reranked, cached or cited.  A second check on the hits
is defence in depth and counts any violation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.errors import ValidationFailure
from app.core.hashing import sha256_text
from app.core.security import Principal
from app.corpus.config import CorpusConfig, RetrievalParams
from app.corpus.router import QueryPlan, plan_query
from app.corpus.schema import PUBLIC_GROUP, to_epoch
from app.corpus.telemetry import ACCESS_FILTERED, stage
from app.embeddings.base import CachedEmbedder
from app.indexing.backend import TEXT_FIELD, FilterSpec, Hit, SearchBackend
from app.observability.metrics import StageTimer
from app.ranking.fusion import reciprocal_rank_fusion
from app.ranking.rerankers import Reranker

# caller-supplied metadata filters are limited to these index fields
FILTERABLE_FIELDS = {"document_type", "source_system", "x_department", "x_patient_pseudo_id", "x_encounter_id"}


@dataclass(frozen=True)
class AuthContext:
    """Who is asking.  Built from the verified credential, never from the request body."""

    tenant_id: str
    user_id: str
    groups: frozenset[str]

    @classmethod
    def from_principal(cls, principal: Principal) -> AuthContext:
        return cls(principal.tenant_id, principal.user_id, frozenset({*principal.scopes, PUBLIC_GROUP}))

    @property
    def scope_key(self) -> str:
        """Stable key of the permission set; part of every cache key."""
        return sha256_text(f"{self.tenant_id}|{','.join(sorted(self.groups))}")[:24]

    def allows(self, source: dict[str, Any]) -> bool:
        groups = source.get("acl_groups") or []
        return source.get("tenant_id") == self.tenant_id and any(g in self.groups for g in groups)


@dataclass
class RetrievalRequest:
    query: str
    strategy: str | None = None
    document_types: list[str] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    time_from: datetime | None = None
    time_to: datetime | None = None
    as_of: datetime | None = None
    top_k: int | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    chunk_count: int
    text: str
    indexed_text: str
    title: str
    document_type: str
    source_system: str
    source_record_id: str
    event_time: str | None
    source_updated_at: str | None
    token_count: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    section_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "document_id": self.document_id,
                "document_version": self.document_version, "chunk_index": self.chunk_index,
                "title": self.title, "document_type": self.document_type, "source_system": self.source_system,
                "source_record_id": self.source_record_id, "event_time": self.event_time,
                "source_updated_at": self.source_updated_at, "score": round(self.score, 6), "ranks": self.ranks,
                "section_path": self.section_path, "text": self.text}


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    plan: QueryPlan
    params: RetrievalParams
    strategy: str
    index_version: str
    timings: dict[str, float]
    candidates: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False


def _chunk(hit: Hit) -> RetrievedChunk:
    s = hit.source
    return RetrievedChunk(
        chunk_id=hit.id, document_id=s["document_id"], document_version=int(s.get("document_version", 0)),
        chunk_index=int(s.get("chunk_index", 0)), chunk_count=int(s.get("chunk_count", 1)),
        text=s.get("chunk_text") or s.get(TEXT_FIELD, ""), indexed_text=s.get(TEXT_FIELD, ""),
        title=s.get("title") or "", document_type=s.get("document_type", ""),
        source_system=s.get("source_system", ""), source_record_id=s.get("source_record_id", ""),
        event_time=s.get("event_time"), source_updated_at=s.get("source_updated_at"),
        token_count=int(s.get("token_count", 0)), score=hit.score, section_path=s.get("section_path"),
        metadata={k: v for k, v in s.items() if k.startswith("x_") or k in ("valid_to_ts", "event_ts", "tenant_id",
                                                                           "acl_groups")},
    )


class HybridRetriever:
    def __init__(self, config: CorpusConfig, backend: SearchBackend, embedder: CachedEmbedder,
                 reranker: Reranker | None = None, acronyms: dict[str, str] | None = None) -> None:
        self.config, self.backend, self.embedder, self.reranker = config, backend, embedder, reranker
        self.acronyms = acronyms or {}

    # ------------------------------------------------------------------ filter
    def build_filter(self, auth: AuthContext, request: RetrievalRequest, plan: QueryPlan,
                     params: RetrievalParams, now: datetime | None = None) -> FilterSpec:
        unknown = sorted(set(request.filters) - FILTERABLE_FIELDS)
        if unknown:
            raise ValidationFailure("unsupported filter field", {"fields": unknown,
                                                                 "allowed": sorted(FILTERABLE_FIELDS)})
        # 1. authorisation — from the credential only
        spec = FilterSpec(terms={"tenant_id": [auth.tenant_id], "acl_groups": sorted(auth.groups)})
        # 2. caller filters can only narrow the result
        extra = FilterSpec(terms={k: list(v) for k, v in request.filters.items() if v})
        if request.document_types:
            extra.terms["document_type"] = list(request.document_types)
        for key, values in plan.entity_filters.items():
            extra.terms[key] = values
        spec = spec.merged(extra)
        # 3. time: an explicit historical point, or "currently valid" by default
        as_of = request.as_of or plan.as_of
        if as_of is not None:
            ts = to_epoch(as_of)
            spec = spec.merged(FilterSpec(ranges={"valid_from_ts": (None, ts), "valid_to_ts": (ts, None),
                                                  "event_ts": (None, ts)}))
        elif params.freshness:
            ts = to_epoch(now or datetime.now(UTC))
            spec = spec.merged(FilterSpec(ranges={"valid_to_ts": (ts, None)}))
        if request.time_from or request.time_to:
            spec = spec.merged(FilterSpec(ranges={"event_ts": (to_epoch(request.time_from),
                                                               to_epoch(request.time_to))}))
        return spec

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, auth: AuthContext, request: RetrievalRequest, index: str, timer: StageTimer | None = None,
                 now: datetime | None = None) -> RetrievalResult:
        timer = timer or StageTimer()
        base = self.config.strategy(request.strategy)
        with stage(timer, "query_planning") as span:
            plan = plan_query(request.query, base, self.acronyms, request.as_of)
            params = plan.params or base
            span["query_kind"] = plan.kind
        if request.top_k:
            params = params.model_copy(update={"final_k": request.top_k})
        filters = self.build_filter(auth, request, plan, params, now)
        warnings: list[str] = []
        queries = [(plan.lexical_query, plan.dense_query), *[(q, q) for q in plan.sub_queries]]
        rankings: dict[str, list[str]] = {}
        hits: dict[str, Hit] = {}
        candidates = {"bm25": 0, "dense": 0}
        for n, (lexical, dense) in enumerate(queries):
            suffix = "" if n == 0 else f"_{n}"
            if params.strategy in ("bm25", "hybrid"):
                with stage(timer, "bm25_retrieval", top_k=params.bm25_top_k) as span:
                    found = self.backend.bm25_search(index, lexical, filters, params.bm25_top_k)
                    span["hits"] = len(found)
                rankings[f"bm25{suffix}"] = [h.id for h in found]
                candidates["bm25"] += len(found)
                hits.update({h.id: h for h in found if h.id not in hits})
            if params.strategy in ("dense", "hybrid"):
                with stage(timer, "query_embedding"):
                    vector = self.embedder.encode_queries([dense])[0]
                with stage(timer, "dense_retrieval", top_k=params.dense_top_k) as span:
                    found = self.backend.knn_search(index, vector, filters, params.dense_top_k,
                                                    max(params.num_candidates, params.dense_top_k))
                    span["hits"] = len(found)
                rankings[f"dense{suffix}"] = [h.id for h in found]
                candidates["dense"] += len(found)
                hits.update({h.id: h for h in found if h.id not in hits})
        if plan.entity_filters and not hits:
            # the entity filter matched nothing: fall back to the unscoped strategy rather than fail
            plan.entity_filters = {}
            plan.notes.append("entity_scope_fallback")
            filters = self.build_filter(auth, request, plan, params, now)
            found = self.backend.bm25_search(index, plan.lexical_query, filters, params.bm25_top_k)
            rankings = {"bm25": [h.id for h in found]}
            hits = {h.id: h for h in found}

        with stage(timer, "fusion", lists=len(rankings)) as span:
            weights = {name: params.rrf_weights.get(name.split("_")[0], 1.0) for name in rankings}
            fused = reciprocal_rank_fusion(rankings, params.rrf_k, weights)[: params.fusion_window]
            span["candidates"] = len(fused)
        chunks: list[RetrievedChunk] = []
        denied = 0
        for rank, item in enumerate(fused, start=1):
            hit = hits[item.item_id]
            if not auth.allows(hit.source):  # must be unreachable: the pre-filter already excluded it
                denied += 1
                continue
            chunk = _chunk(hit)
            chunk.score = item.score
            chunk.ranks = {**{k.split("_")[0]: v for k, v in item.ranks.items()}, "fused": rank}
            chunk.scores = {"fused": item.score}
            chunks.append(chunk)
        if denied:
            ACCESS_FILTERED.labels(reason="post_filter").inc(denied)
            warnings.append("authorization post-filter removed results; the pre-filter is misconfigured")

        if params.rerank and self.reranker is not None and chunks:
            depth = min(params.rerank_depth, len(chunks))
            with stage(timer, "reranking", depth=depth):
                t0 = time.perf_counter()
                scores = self.reranker.score(request.query, [c.indexed_text for c in chunks[:depth]])
                order = sorted(range(depth), key=lambda i: (-scores[i], chunks[i].chunk_id))
                head = []
                for new_rank, i in enumerate(order, start=1):
                    chunks[i].ranks["rerank"] = new_rank
                    chunks[i].scores["rerank"] = float(scores[i])
                    chunks[i].score = float(scores[i])
                    head.append(chunks[i])
                chunks = head + chunks[depth:]
                candidates["reranked"] = depth
                candidates["rerank_ms"] = int((time.perf_counter() - t0) * 1000)
        elif params.rerank and self.reranker is None:
            warnings.append("reranking was requested but no reranker is configured")
        if plan.wants_latest and chunks:
            # among near-equal candidates prefer the more recent source (stable sort keeps rank order)
            best = chunks[0].score
            head = [c for c in chunks if c.score >= 0.9 * best]
            head.sort(key=lambda c: -(c.metadata.get("event_ts") or 0.0))
            chunks = head + [c for c in chunks if c.score < 0.9 * best]
        return RetrievalResult(chunks=chunks, plan=plan, params=params,
                               strategy=request.strategy or "default", index_version=index,
                               timings=dict(timer.stages), candidates=candidates, warnings=warnings)
