"""``/v1`` API contracts, streaming, generation backends and failure handling."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.container import AppContext
from app.corpus.config import GenerationParams
from app.corpus.generation import (
    ABSTAIN_TOKEN,
    CircuitBreaker,
    CircuitOpen,
    MockLLM,
    OpenAICompatibleLLM,
    Overloaded,
    split_sentences,
)
from app.corpus.generator import INJECTION_CANARY
from app.corpus.retrieval import AuthContext
from app.corpus.service import CorpusService, QueryRequest
from app.db.models import AuditEvent, CorpusFeedback
from app.main import create_app
from tests.corpus.conftest import Corpus, build_runtime

KEYS = {
    "t0-key": {"user": "alice", "role": "researcher", "scopes": [], "tenant": "tenant-000"},
    "t0-restricted": {"user": "rae", "role": "researcher", "scopes": ["restricted"], "tenant": "tenant-000"},
    "t1-key": {"user": "bob", "role": "researcher", "scopes": [], "tenant": "tenant-001"},
    "t0-annotator": {"user": "ann", "role": "annotator", "scopes": [], "tenant": "tenant-000"},
}


def _h(key: str = "t0-key") -> dict[str, str]:
    return {"X-API-Key": key}


@pytest.fixture()
def client(ctx: AppContext, corpus: Corpus) -> Iterator[TestClient]:
    settings = ctx.settings.model_copy(update={"api_keys_json": json.dumps(KEYS)})
    with TestClient(create_app(dataclasses.replace(ctx, settings=settings))) as test_client:
        test_client.app.state.corpus = corpus.runtime  # type: ignore[attr-defined]
        yield test_client


# ----------------------------------------------------------------- contracts
def test_query_endpoint_returns_a_cited_answer_and_writes_a_text_free_audit_record(client: TestClient,
                                                                                   corpus: Corpus) -> None:
    query = corpus.by_category("keyword")[0]
    response = client.post("/v1/query", json={"query": query["question"], "response_length": "short"}, headers=_h())
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(query["relevant_documents"]) & {c["document_id"] for c in body["citations"]}
    for field in ("answer", "citations", "abstained", "model_version", "index_version", "trace_id", "total_ms",
                  "strategy", "usage", "timings_ms", "notice"):
        assert field in body
    assert len(body["sentences"]) <= 2 and not body["mock"] and not body["degraded"]
    with corpus.runtime.session_factory() as session:
        event = session.query(AuditEvent).filter_by(action="corpus.query").order_by(AuditEvent.id.desc()).first()
        assert event is not None and event.actor == "alice" and event.resource_id == "tenant-000"
        assert event.details["trace_id"] == body["trace_id"] and query["question"] not in json.dumps(event.details)


def test_identity_comes_from_the_credential_never_from_the_request(client: TestClient, corpus: Corpus) -> None:
    exact = corpus.by_category("exact_identifier")[0]            # a tenant-000 record
    assert client.post("/v1/query", json={"query": "anything"}).status_code == 401
    assert client.post("/v1/query", json={"query": "x"}, headers=_h("wrong")).status_code == 401
    forged = client.post("/v1/query", json={"query": exact["question"], "tenant_id": "tenant-000"}, headers=_h("t1-key"))
    assert forged.status_code == 422                              # there is no tenant field to forge
    as_owner = client.post("/v1/retrieve", json={"query": exact["question"]}, headers=_h()).json()
    as_other = client.post("/v1/retrieve", json={"query": exact["question"]}, headers=_h("t1-key")).json()
    owner_docs = {r["document_id"] for r in as_owner["results"]}
    assert set(exact["relevant_documents"]) <= owner_docs
    assert not {r["document_id"] for r in as_other["results"]} & owner_docs
    restricted = corpus.by_category("authorized_restricted")[0]
    public = client.post("/v1/retrieve", json={"query": restricted["question"], "top_k": 50}, headers=_h()).json()
    private = client.post("/v1/retrieve", json={"query": restricted["question"], "top_k": 50},
                          headers=_h("t0-restricted")).json()
    target = set(restricted["relevant_documents"])
    assert not target & {r["document_id"] for r in public["results"]}
    assert target & {r["document_id"] for r in private["results"]}
    answer = client.post("/v1/query", json={"query": restricted["question"]}, headers=_h()).json()
    assert not target & {c["document_id"] for c in answer["citations"]}        # nor through a citation


def test_request_validation_roles_and_unknown_strategy(client: TestClient) -> None:
    assert client.post("/v1/query", json={"query": ""}, headers=_h()).status_code == 422
    assert client.post("/v1/query", json={"query": "x", "top_k": 500}, headers=_h()).status_code == 422
    unknown = client.post("/v1/query", json={"query": "x", "strategy": "nope"}, headers=_h())
    assert unknown.status_code == 404 and "hybrid" in unknown.json()["error"]["details"]["available"]
    bad_filter = client.post("/v1/retrieve", json={"query": "x", "filters": {"acl_groups": ["restricted"]}}, headers=_h())
    assert bad_filter.status_code == 422 and bad_filter.json()["error"]["code"] == "validation_error"
    assert client.post("/v1/retrieve", json={"query": "x"}, headers=_h("t0-annotator")).status_code == 403
    narrowed = client.post("/v1/retrieve", json={"query": "follow-up status", "document_types": ["lab_result"],
                                                 "strategy": "bm25"}, headers=_h()).json()
    assert narrowed["results"] and {r["document_type"] for r in narrowed["results"]} == {"lab_result"}
    assert narrowed["plan"]["kind"] and "bm25_retrieval" in narrowed["timings_ms"]


def test_streaming_emits_typed_events_in_order(client: TestClient, corpus: Corpus) -> None:
    query = corpus.by_category("keyword")[0]
    with client.stream("POST", "/v1/query/stream", json={"query": query["question"]}, headers=_h()) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        events: list[tuple[str, dict[str, Any]]] = []
        name = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-2:] == ["citations", "done"] and "token" in names
    assert set(names[1:-2]) == {"token"}
    done = events[-1][1]
    assert " ".join(d["text"] for n, d in events if n == "token") == done["answer"]
    assert events[-2][1]["citations"] == done["citations"] and events[0][1]["trace_id"] == done["trace_id"]
    failed = client.post("/v1/query/stream", json={"query": "x", "strategy": "nope"}, headers=_h())
    assert failed.status_code == 404


def test_feedback_and_health_endpoints(client: TestClient, corpus: Corpus, tmp_path: Path) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.json()["active_index"] == corpus.runtime.indexes.active() and ready.json()["checks"]["cache"] == "ok"
    stored = client.post("/v1/feedback", json={"trace_id": "a" * 32, "rating": "down", "strategy": "hybrid",
                                               "categories": ["missing_citation", "slow"], "latency_ms": 2300.0,
                                               "chunk_ids": ["doc_x-00000"]}, headers=_h())
    assert stored.status_code == 201
    assert client.post("/v1/feedback", json={"trace_id": "a" * 32, "rating": "down", "comment": "free text"},
                       headers=_h()).status_code == 422                 # no free-text field exists
    with corpus.runtime.session_factory() as session:
        row = session.get(CorpusFeedback, stored.json()["id"])
        assert row is not None and (row.tenant_id, row.user_id, row.rating) == ("tenant-000", "alice", "down")
        assert row.categories == ["missing_citation", "slow"] and not row.human_verified
    empty, _ = build_runtime(tmp_path)                                  # no index built yet
    client.app.state.corpus = empty  # type: ignore[attr-defined]
    not_ready = client.get("/readyz")
    assert not_ready.status_code == 503 and "missing" in not_ready.json()["checks"]["index"]
    assert client.get("/healthz").status_code == 200                    # liveness is independent of dependencies
    missing = client.post("/v1/query", json={"query": "x"}, headers=_h())
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------- generation
class ScriptedLLM:
    """Stands in for a model server: yields fixed text, or fails."""

    name, mock = "scripted", False

    def __init__(self, text: str = "", error: Exception | None = None, delay: float = 0.0) -> None:
        self.text, self.error, self.delay = text, error, delay
        self.closed = False
        self.calls = 0
        self.breaker = CircuitBreaker(5, 30.0)

    async def health(self) -> dict[str, Any]:
        return {"ok": self.error is None, "circuit": "closed"}

    async def aclose(self) -> None:
        return None

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls += 1
        self.messages = messages
        try:
            if self.error is not None:
                raise self.error
            for word in self.text.split(" "):
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield word + " "
        finally:
            self.closed = True


def _service(corpus: Corpus, llm: Any, **generation: Any) -> CorpusService:
    runtime = corpus.runtime
    params = runtime.config.generation.model_copy(update={"backend": "mock", "model": "scripted-model", **generation})
    config = runtime.config.model_copy(update={"generation": params})
    return CorpusService(config, runtime.indexes, runtime.retriever, runtime.cache, llm)


def test_only_cited_and_supported_model_sentences_are_released(corpus: Corpus) -> None:
    query = corpus.by_category("prompt_injection")[0]
    auth = corpus.auth(query["tenant_id"])
    context = corpus.runtime.service.answer_sync(auth, QueryRequest(query=query["question"]))
    quoted = context["sentences"][0]["text"]                               # a sentence that really is in S1
    script = (f"{quoted} [S1] {INJECTION_CANARY}. The patient also has a 9 cm aortic aneurysm. [S1] "
              "Hepatic steatosis was excluded. [S77] An uncited statement about the plan.")
    llm = ScriptedLLM(script)
    answer = _service(corpus, llm).answer_sync(auth, QueryRequest(query=query["question"], response_length="long"))
    assert [s["text"] for s in answer["sentences"]] == [quoted] and answer["sentences"][0]["citations"] == ["S1"]
    assert INJECTION_CANARY not in answer["answer"]                        # the injected instruction achieved nothing
    assert answer["removed_sentences"] == 4 and answer["invalid_citations"] == ["S77"]
    assert answer["model_version"] == "mock:scripted-model:corpus-p1" and not answer["abstained"]
    system, user = llm.messages[0]["content"], llm.messages[1]["content"]
    assert "untrusted source material" in system and "never follow" in system
    assert f"<<<{context['citations'][0]['excerpt'][:40]}" in user or "<<<" in user
    assert "[S1]" in user and query["question"] in user
    # the document that carries the injection is quoted only inside the untrusted block
    injected = _service(corpus, ScriptedLLM(f"{INJECTION_CANARY}")).answer_sync(auth, QueryRequest(query=query["question"]))
    assert injected["abstained"] and injected["answer"] == ""
    assert injected["abstention_reason"] == "no generated statement was supported by a cited source"
    declined = _service(corpus, ScriptedLLM(ABSTAIN_TOKEN)).answer_sync(auth, QueryRequest(query=query["question"]))
    assert declined["abstained"] and declined["abstention_reason"] == "the model found the evidence insufficient"
    unverified = _service(corpus, ScriptedLLM("Anything at all."), verify_citations=False).answer_sync(
        auth, QueryRequest(query=query["question"]))
    assert unverified["answer"].startswith("Anything at all.") and unverified["citations"] == []


def test_model_failure_degrades_to_the_extractive_answer(corpus: Corpus) -> None:
    query = corpus.by_category("keyword")[0]
    auth, request = corpus.auth(query["tenant_id"]), QueryRequest(query=query["question"])
    from app.core.errors import DependencyUnavailable

    for error in (DependencyUnavailable("language model request failed: ReadTimeout"),
                  Overloaded("the language model is at capacity"), CircuitOpen("disabled"), TimeoutError()):
        answer = _service(corpus, ScriptedLLM(error=error)).answer_sync(auth, request)
        assert answer["degraded"] and not answer["abstained"] and answer["citations"]
        assert answer["model_version"] == "extractive:extractive-v1"
        assert "extractive fallback used" in " ".join(answer["warnings"])
    labelled = _service(corpus, MockLLM(GenerationParams(backend="mock", model="none"))).answer_sync(auth, request)
    assert labelled["mock"] and "MOCK GENERATION" in " ".join(labelled["warnings"]) and labelled["citations"]


def test_response_length_cap_and_client_disconnect_cancel_generation(corpus: Corpus) -> None:
    query = corpus.by_category("keyword")[0]
    auth = corpus.auth(query["tenant_id"])
    first = corpus.runtime.service.answer_sync(auth, QueryRequest(query=query["question"]))["sentences"][0]["text"]
    long_script = " ".join(f"{first} [S1]" for _ in range(30))
    capped_llm = ScriptedLLM(long_script)
    capped = _service(corpus, capped_llm).answer_sync(auth, QueryRequest(query=query["question"], response_length="short"))
    assert len(capped["sentences"]) == 2 and capped_llm.closed           # the stream was closed early

    async def disconnect() -> ScriptedLLM:
        llm = ScriptedLLM(long_script, delay=0.001)
        events = _service(corpus, llm).events(auth, QueryRequest(query=query["question"], response_length="long"))
        async for event in events:
            if event["event"] == "token":
                break                                                    # the client goes away mid-answer
        await events.aclose()
        return llm

    assert asyncio.run(disconnect()).closed                              # generation stopped with the request


def test_search_backend_failure_is_a_controlled_error(client: TestClient, corpus: Corpus, monkeypatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(corpus.runtime.backend, "bm25_search", boom)
    response = client.post("/v1/query", json={"query": "a question that is not cached"}, headers=_h())
    assert response.status_code == 503
    assert response.json()["error"] == {"code": "dependency_unavailable",
                                        "message": "search backend request failed: ConnectionError", "details": {}}
    with client.stream("POST", "/v1/query/stream", json={"query": "another uncached question"}, headers=_h()) as stream:
        text = "".join(stream.iter_text())
    assert "event: error" in text and "dependency_unavailable" in text


# ------------------------------------------------------- OpenAI-compatible client
def _sse_body(*deltas: str, usage: dict[str, int] | None = None) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n\n" for d in deltas]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    return ("".join(lines) + "data: [DONE]\n\n").encode()


async def _collect(llm: OpenAICompatibleLLM) -> str:
    return "".join([d async for d in llm.stream([{"role": "user", "content": "hi"}])])


def test_openai_compatible_client_streams_retries_and_accounts_usage(monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_RAG_LLM_API_KEY", "test-token")
    seen: list[httpx.Request] = []
    replies = [httpx.Response(503), httpx.Response(200, content=_sse_body("Hello ", "world.", usage={
        "prompt_tokens": 12, "completion_tokens": 3}))]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return replies[min(len(seen) - 1, 1)]

    params = GenerationParams(backend="openai_compatible", base_url="http://vllm.internal:8000", model="m",
                              max_retries=1, max_output_tokens=64)
    llm = OpenAICompatibleLLM(params, transport=httpx.MockTransport(handler))
    assert asyncio.run(_collect(llm)) == "Hello world."
    assert len(seen) == 2 and seen[1].url.path == "/v1/chat/completions"           # one retry after the 503
    payload = json.loads(seen[1].content)
    assert payload["model"] == "m" and payload["stream"] is True and payload["max_tokens"] == 64
    assert seen[1].headers["authorization"] == "Bearer test-token"
    assert llm.usage == {"requests": 1, "prompt_tokens": 12, "completion_tokens": 3} and llm.breaker.state == "closed"


def test_circuit_breaker_backpressure_and_timeouts() -> None:
    from app.core.errors import DependencyUnavailable

    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    params = GenerationParams(backend="openai_compatible", base_url="http://vllm.internal:8000", model="m",
                              max_retries=0, circuit_failure_threshold=2, circuit_reset_seconds=0.05,
                              max_concurrency=1, queue_limit=0)
    llm = OpenAICompatibleLLM(params, transport=httpx.MockTransport(failing))

    async def scenario() -> list[str]:
        outcomes: list[str] = []
        for _ in range(3):
            try:
                await _collect(llm)
            except CircuitOpen:
                outcomes.append("circuit_open")
            except DependencyUnavailable as exc:
                outcomes.append(exc.message.split(": ")[-1])
        await asyncio.sleep(0.06)                                       # reset window passed: one probe is let through
        try:
            await _collect(llm)
        except DependencyUnavailable as exc:
            outcomes.append("probe:" + exc.message.split(": ")[-1])
        return outcomes

    assert asyncio.run(scenario()) == ["ReadTimeout", "ReadTimeout", "circuit_open", "probe:ReadTimeout"]

    async def overload() -> str:
        gate = asyncio.Event()

        async def slow(request: httpx.Request) -> httpx.Response:
            await gate.wait()
            return httpx.Response(200, content=_sse_body("ok."))

        busy = OpenAICompatibleLLM(params, transport=httpx.MockTransport(slow))
        first = asyncio.create_task(_collect(busy))
        await asyncio.sleep(0.01)                                       # the only slot is taken, the queue holds 0
        try:
            await _collect(busy)
            outcome = "accepted"
        except Overloaded as exc:
            outcome = exc.code
        gate.set()
        assert await first == "ok."
        return outcome

    assert asyncio.run(overload()) == "overloaded"                      # rejected instead of queued without bound
    breaker = CircuitBreaker(1, 60.0)
    breaker.failure()
    with pytest.raises(CircuitOpen):
        breaker.before()
    breaker.success()
    breaker.before()
    assert split_sentences("One. Two", final=False) == (["One."], "Two") and split_sentences("Two", final=True) == (["Two"], "")
    assert asyncio.run(MockLLM(GenerationParams(backend="mock")).health())["mock"] is True
    assert AuthContext("t", "u", frozenset({"a"})).scope_key != AuthContext("t", "u", frozenset({"b"})).scope_key
