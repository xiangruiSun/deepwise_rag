"""The RAG core shared by the online API, the streaming API and offline batch jobs.

``CorpusService.events`` is the single implementation of
"retrieve -> build context -> generate -> verify".  ``answer`` collects its events
into one response, ``stream`` forwards them, and batch workers call ``answer`` —
so online and offline results cannot diverge.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import AppError, DependencyUnavailable, RequestTimeout
from app.core.hashing import sha256_text, stable_hash
from app.corpus.cache import CorpusCache
from app.corpus.config import CorpusConfig
from app.corpus.context import ContextBundle, select_context
from app.corpus.generation import (
    ABSTAIN_TOKEN,
    ExtractiveAnswerer,
    GenerationResult,
    MockLLM,
    OpenAICompatibleLLM,
    Overloaded,
    SentenceVerifier,
    build_messages,
    split_sentences,
)
from app.corpus.indexing import IndexManager
from app.corpus.retrieval import AuthContext, HybridRetriever, RetrievalRequest, RetrievalResult, RetrievedChunk
from app.corpus.telemetry import (
    FIRST_TOKEN_LATENCY,
    QUERIES,
    QUERY_LATENCY,
    new_trace_id,
    record_stage,
    stage,
)
from app.observability.metrics import INFERENCE_TOKENS, StageTimer
from app.rag.evidence import estimate_tokens

RESPONSE_LENGTHS = {"short": (2, 160), "medium": (4, 400), "long": (8, 800)}
NOTICE = "Evidence-backed assistance from retrieved records. Not a diagnosis or a treatment recommendation."


class QueryRequest(BaseModel):
    """Body of ``/v1/query`` and ``/v1/retrieve``.  There is deliberately no tenant or user
    field: identity and permissions come from the verified credential only."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    strategy: str | None = Field(default=None, max_length=48)
    document_types: list[str] = Field(default_factory=list, max_length=20)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    time_from: datetime | None = None
    time_to: datetime | None = None
    as_of: datetime | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    response_length: Literal["short", "medium", "long"] = "medium"

    def to_retrieval(self) -> RetrievalRequest:
        return RetrievalRequest(query=self.query, strategy=self.strategy, document_types=self.document_types,
                                filters=self.filters, time_from=self.time_from, time_to=self.time_to,
                                as_of=self.as_of, top_k=self.top_k)

    def cache_shape(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"response_length"})


class CorpusService:
    def __init__(self, config: CorpusConfig, indexes: IndexManager, retriever: HybridRetriever,
                 cache: CorpusCache, llm: OpenAICompatibleLLM | MockLLM | None) -> None:
        self.config, self.indexes, self.retriever, self.cache, self.llm = config, indexes, retriever, cache, llm
        self.extractive = ExtractiveAnswerer(config.generation)

    # ------------------------------------------------------------------ model
    @property
    def model_version(self) -> str:
        g = self.config.generation
        return f"{g.backend}:{g.model}:{g.prompt_version}"

    @property
    def reranker_identity(self) -> str:
        r = self.retriever.reranker
        return stable_hash(r.metadata(), 16) if r is not None else "none"

    # -------------------------------------------------------------- retrieval
    def retrieve(self, auth: AuthContext, request: QueryRequest, timer: StageTimer | None = None,
                 use_cache: bool = True) -> RetrievalResult:
        timer = timer or StageTimer()
        index = self.indexes.require_active()
        key = None
        if use_cache:
            params = self.config.strategy(request.strategy)
            key = self.cache.retrieval_key(auth.tenant_id, auth.scope_key, index, self.indexes.embedding_identity,
                                           self.reranker_identity,
                                           {"q": request.cache_shape(), "p": params.model_dump(mode="json")})
            cached = self.cache.get_retrieval(key)
            if cached is not None:
                chunks = [RetrievedChunk(**c) for c in cached["chunks"]]
                # defence in depth: a cached entry is re-checked against the caller's permissions
                chunks = [c for c in chunks if c.metadata.get("tenant_id") == auth.tenant_id
                          and any(g in auth.groups for g in c.metadata.get("acl_groups") or [])]
                from app.corpus.router import plan_query

                plan = plan_query(request.query, params, self.retriever.acronyms, request.as_of)
                return RetrievalResult(chunks=chunks, plan=plan, params=plan.params or params,
                                       strategy=request.strategy or "default", index_version=index,
                                       timings=dict(timer.stages), candidates=cached["candidates"],
                                       warnings=cached["warnings"], cache_hit=True)
        try:
            result = self.retriever.retrieve(auth, request.to_retrieval(), index, timer)
        except AppError:
            raise
        except Exception as exc:  # connection errors, timeouts, malformed responses of the search service
            raise DependencyUnavailable(f"search backend request failed: {type(exc).__name__}") from exc
        if key is not None and not result.warnings:
            self.cache.put_retrieval(key, {"chunks": [asdict(c) for c in result.chunks],
                                           "candidates": result.candidates, "warnings": result.warnings})
        return result

    def build_context(self, auth: AuthContext, result: RetrievalResult, timer: StageTimer) -> ContextBundle:
        with stage(timer, "context_construction") as span:
            bundle = select_context(result.chunks, self.config.context, result.params.final_k,
                                    self.retriever.backend, result.index_version, auth)
            span["chunks"], span["tokens"] = len(bundle.items), bundle.tokens_used
        return bundle

    # ------------------------------------------------------------------ events
    async def events(self, auth: AuthContext, request: QueryRequest,
                     use_cache: bool = True) -> AsyncIterator[dict[str, Any]]:
        """Typed events of one RAG request: ``retrieval``, ``token``*, ``citations``, ``done``."""
        started = time.perf_counter()
        timer = StageTimer()
        trace_id = new_trace_id()
        max_sentences, max_tokens = RESPONSE_LENGTHS[request.response_length]
        warnings: list[str] = []
        result = await asyncio.to_thread(self.retrieve, auth, request, timer, use_cache)
        bundle = await asyncio.to_thread(self.build_context, auth, result, timer)
        warnings.extend(result.warnings)
        yield {"event": "retrieval", "data": {
            "trace_id": trace_id, "index_version": result.index_version, "strategy": result.strategy,
            "query_kind": result.plan.kind, "candidates": result.candidates, "context": bundle.stats(),
            "cache_hit": result.cache_hit}}

        generation: GenerationResult | None = None
        degraded = False
        first_token_at: float | None = None
        raw = ""
        if not bundle.items:
            generation = GenerationResult("", [], True, "no authorized document matches the question",
                                          estimate_tokens(request.query), 0)
        elif self.llm is None:
            with stage(timer, "generation", backend="extractive"):
                generation = self.extractive.answer(request.query, bundle, max_sentences, result.plan.identifiers)
            for sentence in generation.sentences:
                first_token_at = first_token_at or time.perf_counter()
                yield {"event": "token", "data": {"text": _render(sentence)}}
        else:
            verifier = SentenceVerifier(bundle, self.config.generation.verify_citations)
            messages = build_messages(request.query, bundle, self.config.generation)
            sentences: list[dict[str, Any]] = []
            buffer = ""
            generation_started = time.perf_counter()
            try:
                stream = self.llm.stream(messages)
                try:
                    async for delta in stream:
                        raw += delta
                        buffer += delta
                        complete, buffer = split_sentences(buffer, final=False)
                        for text in complete:
                            accepted = None if text.strip().startswith(ABSTAIN_TOKEN) else verifier.check(text)
                            if accepted is not None and len(sentences) < max_sentences:
                                sentences.append(accepted)
                                first_token_at = first_token_at or time.perf_counter()
                                yield {"event": "token", "data": {"text": _render(accepted)}}
                        if len(sentences) >= max_sentences or estimate_tokens(raw) >= max_tokens:
                            break  # closing the stream cancels generation on the server
                finally:
                    await stream.aclose()  # type: ignore[attr-defined]
                for text in split_sentences(buffer, final=True)[0]:
                    accepted = None if text.strip().startswith(ABSTAIN_TOKEN) else verifier.check(text)
                    if accepted is not None and len(sentences) < max_sentences:
                        sentences.append(accepted)
                        first_token_at = first_token_at or time.perf_counter()
                        yield {"event": "token", "data": {"text": _render(accepted)}}
            except (TimeoutError, DependencyUnavailable, Overloaded, RequestTimeout) as exc:
                degraded = True
                code = exc.code if isinstance(exc, AppError) else "timeout"
                if sentences:
                    # statements already shown stay: they were verified before being released
                    warnings.append(f"generation was interrupted ({code}); the answer may be incomplete")
                else:
                    # the model is optional: degrade to the extractive answer instead of failing
                    warnings.append(f"language model unavailable ({code}); extractive fallback used")
                    generation = self.extractive.answer(request.query, bundle, max_sentences, result.plan.identifiers)
                    for sentence in generation.sentences:
                        first_token_at = first_token_at or time.perf_counter()
                        yield {"event": "token", "data": {"text": _render(sentence)}}
            record_stage(timer, "generation", generation_started, backend=self.llm.name,
                         sentences=len(sentences), degraded=degraded)
            if generation is None:
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
                abstained = not sentences
                reason = None
                if abstained:
                    reason = ("the model found the evidence insufficient" if ABSTAIN_TOKEN in raw
                              else "no generated statement was supported by a cited source")
                generation = GenerationResult(
                    " ".join(_render(s) for s in sentences), sentences, abstained, reason, prompt_tokens,
                    estimate_tokens(raw), verifier.removed, verifier.invalid, mock=self.llm.mock)

        assert generation is not None
        with stage(timer, "citation_validation") as span:
            sources = bundle.by_source()
            citations = []
            for source_id in generation.cited:
                item = sources.get(source_id)
                if item is None:  # cannot happen after verification; never emit an unknown source
                    continue
                c = item.chunk
                citations.append({
                    "source_id": source_id, "chunk_id": c.chunk_id, "document_id": c.document_id,
                    "document_version": c.document_version, "title": c.title, "document_type": c.document_type,
                    "source_system": c.source_system, "source_record_id": c.source_record_id,
                    "event_time": c.event_time, "source_updated_at": c.source_updated_at,
                    "excerpt": item.text[:300]})
            span["citations"] = len(citations)
        yield {"event": "citations", "data": {"citations": citations}}

        total_ms = (time.perf_counter() - started) * 1000.0
        first_ms = (first_token_at - started) * 1000.0 if first_token_at else None
        outcome = "abstained" if generation.abstained else "answered"
        QUERIES.labels(outcome="degraded" if degraded else outcome).inc()
        QUERY_LATENCY.labels(strategy=result.strategy, mode="rag").observe(total_ms / 1000.0)
        if first_ms is not None:
            FIRST_TOKEN_LATENCY.labels(backend=self.llm.name if self.llm and not degraded else "extractive") \
                .observe(first_ms / 1000.0)
        INFERENCE_TOKENS.labels(direction="input").inc(generation.prompt_tokens)
        INFERENCE_TOKENS.labels(direction="output").inc(generation.completion_tokens)
        if generation.mock:
            warnings.append("MOCK GENERATION: no language model is configured; this is not a model answer")
        yield {"event": "done", "data": {
            "answer": generation.text, "sentences": generation.sentences, "citations": citations,
            "abstained": generation.abstained, "abstention_reason": generation.abstention_reason,
            "model_version": self.model_version if generation.sentences and not (degraded and not raw)
            else ("extractive:extractive-v1" if degraded else self.model_version),
            "index_version": result.index_version, "strategy": result.strategy,
            "query_kind": result.plan.kind, "plan": result.plan.as_dict(), "trace_id": trace_id,
            "total_ms": round(total_ms, 2), "first_token_ms": round(first_ms, 2) if first_ms else None,
            "timings_ms": {k: round(v, 3) for k, v in timer.stages.items()},
            "usage": {"prompt_tokens": generation.prompt_tokens, "completion_tokens": generation.completion_tokens,
                      "context_tokens": bundle.tokens_used},
            "context": bundle.stats(), "retrieved_chunks": [c.chunk.chunk_id for c in bundle.items],
            "context_documents": [c.chunk.document_id for c in bundle.items],
            "ranked_documents": list(dict.fromkeys(c.document_id for c in result.chunks)),
            "candidates": result.candidates,
            "removed_sentences": generation.removed_sentences,
            "invalid_citations": sorted(set(generation.invalid_citations)),
            "mock": generation.mock, "degraded": degraded, "cache_hit": result.cache_hit,
            "warnings": warnings, "notice": NOTICE,
            "query_hash": sha256_text(request.query)[:32],
        }}

    async def answer(self, auth: AuthContext, request: QueryRequest, timeout: float | None = None,
                     use_cache: bool = True) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            final: dict[str, Any] = {}
            async for event in self.events(auth, request, use_cache):
                if event["event"] == "done":
                    final = event["data"]
            return final

        try:
            return await asyncio.wait_for(run(), timeout) if timeout else await run()
        except TimeoutError as exc:
            QUERIES.labels(outcome="timeout").inc()
            raise RequestTimeout("the request exceeded the configured time limit") from exc

    def answer_sync(self, auth: AuthContext, request: QueryRequest, timeout: float | None = None,
                    use_cache: bool = True) -> dict[str, Any]:
        """For batch workers and scripts that are not inside an event loop."""
        return asyncio.run(self.answer(auth, request, timeout, use_cache))

    # ------------------------------------------------------------------ health
    async def readiness(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        checks["search"] = "ok" if self.retriever.backend.ping() else "error: unreachable"
        active = self.indexes.active()
        checks["index"] = "ok" if active else "missing (build and activate an index)"
        checks["cache"] = "ok" if self.cache.backend.ping() else "degraded: cache unreachable (served without cache)"
        if self.llm is not None:
            health = await self.llm.health()
            checks["llm"] = "ok" if health.get("ok") else "degraded: language model unreachable (extractive fallback)"
            checks["llm_circuit"] = health.get("circuit")
        else:
            checks["llm"] = "ok (extractive generator, no model)"
        required_ok = checks["search"] == "ok" and checks["index"] == "ok"
        return {"status": "ready" if required_ok else "not_ready", "checks": checks, "active_index": active,
                "model_version": self.model_version, "generation_backend": self.config.generation.backend}


def _render(sentence: dict[str, Any]) -> str:
    return sentence["text"] + " " + "".join(f"[{c}]" for c in sentence["citations"])
