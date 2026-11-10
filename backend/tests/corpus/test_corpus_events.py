"""Change events (outbox / CDC), near-real-time index updates and scheduled maintenance."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.corpus.etl import SourceSpec, normalize_row
from app.corpus.events import ChangeEvent, EventProcessor, KafkaEventSource, OutboxEventSource, publish
from app.corpus.retrieval import AuthContext
from app.corpus.runtime import CorpusRuntime
from app.corpus.scheduler import TASKS, CorpusScheduler
from app.corpus.schema import CanonicalDocument
from app.corpus.service import QueryRequest
from app.db.models import CorpusDocument, CorpusOutboxEvent
from app.indexing.backend import FilterSpec
from tests.corpus.conftest import Corpus, build_and_activate, build_runtime, ingest

AUTH = AuthContext("tenant-000", "u", frozenset({"public"}))
ROW = {"tenant": "tenant-000", "source_system": "ehr_reports", "record_id": "REP-EVT-1", "title": "Event report"}


def _doc(version: int, body: str, **extra: Any) -> CanonicalDocument:
    doc, reason, _ = normalize_row({**ROW, "version": str(version), "body": body, **extra}, SourceSpec(uri="mem"),
                                   "mem", datetime(2026, 1, 1))
    assert doc is not None, reason
    return doc


def _emit(runtime: CorpusRuntime, *documents: CanonicalDocument, occurred_at: datetime | None = None) -> list[str]:
    with runtime.session_factory() as session:
        ids = [publish(session, d, occurred_at) for d in documents]
        session.commit()
    return ids


def _hits(runtime: CorpusRuntime, text: str, document_id: str) -> list[Any]:
    request = QueryRequest(query=text, strategy="hybrid")           # through the cache, like a real caller
    return [c for c in runtime.service.retrieve(AUTH, request).chunks if c.document_id == document_id]


@pytest.fixture()
def live(corpus: Corpus, tmp_path: Path) -> tuple[CorpusRuntime, EventProcessor]:
    runtime, _ = build_runtime(tmp_path)
    ingest(runtime, corpus.data_dir / "raw", tmp_path / "normalized")
    build_and_activate(runtime)
    return runtime, EventProcessor(OutboxEventSource(runtime.session_factory), runtime.session_factory,
                                   runtime.indexes, max_attempts=2)


def test_update_and_delete_events_reach_retrieval_within_the_consistency_window(
        live: tuple[CorpusRuntime, EventProcessor]) -> None:
    runtime, processor = live
    v1 = _doc(1, "Axolotl regimen gamma was started for the patient.")
    assert not _hits(runtime, "axolotl regimen gamma", v1.document_id)       # and this miss is now cached
    started = time.perf_counter()
    _emit(runtime, v1)
    report = processor.process_batch()
    assert report["outcomes"] == {"inserted": 1} and report["sync"]["documents"] == 1
    found = _hits(runtime, "axolotl regimen gamma", v1.document_id)
    assert found and found[0].document_version == 1                          # the cached miss was invalidated
    # an update replaces the searchable content
    _emit(runtime, _doc(2, "Capybara regimen delta replaced the earlier plan."))
    assert processor.process_batch()["outcomes"] == {"updated": 1}
    stale = _hits(runtime, "axolotl regimen gamma", v1.document_id)          # same query as the cached one
    assert all(c.document_version == 2 and "Axolotl" not in c.text for c in stale)
    assert _hits(runtime, "capybara regimen delta", v1.document_id)[0].document_version == 2
    # a deletion removes it everywhere, including from results cached a moment ago
    _emit(runtime, _doc(3, "", deleted=True))
    assert processor.process_batch()["outcomes"] == {"deleted": 1}
    assert not _hits(runtime, "capybara regimen delta", v1.document_id)
    assert runtime.backend.count(runtime.indexes.active(), FilterSpec(terms={"document_id": [v1.document_id]})) == 0
    answer = runtime.service.answer_sync(AUTH, QueryRequest(query="capybara regimen delta"))
    assert v1.document_id not in {c["document_id"] for c in answer["citations"]}
    # visibility lag of three round trips, measured here, against the configured bound
    assert time.perf_counter() - started < runtime.config.consistency_interval_seconds
    assert runtime.indexes.pending() == {"upserts": 0, "deletes": 0}
    assert runtime.indexes.consistency_check()["consistent"]


def test_replayed_and_out_of_order_events_have_no_effect(live: tuple[CorpusRuntime, EventProcessor]) -> None:
    runtime, processor = live
    v1, v2 = _doc(1, "Narwhal protocol one."), _doc(2, "Narwhal protocol two, the current text.")
    t0 = datetime(2026, 3, 1, 12, 0, 0)
    # version 2 is delivered BEFORE version 1, and version 2 is delivered twice
    _emit(runtime, v2, occurred_at=t0)
    _emit(runtime, v1, occurred_at=t0 + timedelta(seconds=1))
    _emit(runtime, v2, occurred_at=t0 + timedelta(seconds=2))
    report = processor.process_batch()
    assert report["outcomes"] == {"inserted": 1, "stale": 1, "unchanged": 1}
    assert _hits(runtime, "narwhal protocol", v1.document_id)[0].document_version == 2
    chunks_before = runtime.backend.count(runtime.indexes.active(), FilterSpec(terms={"document_id": [v1.document_id]}))
    # at-least-once delivery: the broker hands every event over again
    with runtime.session_factory() as session:
        for row in session.query(CorpusOutboxEvent).all():
            row.status = "pending"
        session.commit()
    again = processor.process_batch()
    assert again["outcomes"] == {"unchanged": 2, "stale": 1} and again["sync"]["documents"] == 0
    assert runtime.backend.count(runtime.indexes.active(),
                                 FilterSpec(terms={"document_id": [v1.document_id]})) == chunks_before
    # a deleted document is not resurrected by a late, older event
    _emit(runtime, _doc(3, "", deleted=True))
    processor.process_batch()
    _emit(runtime, v2)
    assert processor.process_batch()["outcomes"] == {"stale": 1}
    assert not _hits(runtime, "narwhal protocol", v1.document_id)
    with runtime.session_factory() as session:
        assert session.get(CorpusDocument, v1.document_id).is_deleted


def test_a_poison_event_is_dead_lettered_and_never_blocks_the_stream(
        live: tuple[CorpusRuntime, EventProcessor]) -> None:
    runtime, processor = live
    good = _doc(1, "Ocelot follow-up note.")
    with runtime.session_factory() as session:
        session.add(CorpusOutboxEvent(event_id="evt_poison", tenant_id="tenant-000", document_id="doc_x",
                                      document_version=1, change_type="upsert", payload={"not": "a document"},
                                      occurred_at=datetime(2020, 1, 1)))
        forged = good.model_dump(mode="json")
        session.add(CorpusOutboxEvent(event_id="evt_forged", tenant_id="tenant-001", document_id=good.document_id,
                                      document_version=1, change_type="upsert", payload=forged,
                                      occurred_at=datetime(2020, 1, 2)))
        session.commit()
    _emit(runtime, good)
    assert processor.source.lag_seconds() > 365 * 86_400                    # the oldest pending event is from 2020
    first = processor.process_batch()
    assert first["outcomes"] == {"retry": 2, "inserted": 1}                  # the good event went through
    assert _hits(runtime, "ocelot follow-up note", good.document_id)
    second = processor.process_batch()
    assert second["outcomes"] == {"dead_letter": 2} and second["lag_seconds"] == 0.0
    with runtime.session_factory() as session:
        poison, envelope = session.get(CorpusOutboxEvent, "evt_poison"), session.get(CorpusOutboxEvent, "evt_forged")
        assert (poison.status, poison.attempts, poison.error) == ("dead", 2, "ValidationError")
        # an envelope that names another tenant than its payload is refused, not applied
        assert (envelope.status, envelope.error) == ("dead", "ValueError")
        assert "document" not in (poison.error or "")                       # error class only, never payload text
    assert processor.process_batch()["events"] == 0
    quality = TASKS["data_quality"](runtime)
    assert quality["dead_events"] == 2 and quality["ok"] is False


class _Message:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None


class _FakeConsumer:
    def __init__(self, values: list[bytes]) -> None:
        self.queue, self.committed = [_Message(v) for v in values], []

    def consume(self, num_messages: int, timeout: float) -> list[_Message]:
        batch, self.queue = self.queue[:num_messages], self.queue[num_messages:]
        return batch

    def commit(self, message: _Message, asynchronous: bool) -> None:
        self.committed.append(message)


class _FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, dict[str, str]]] = []

    def produce(self, topic: str, value: bytes, headers: dict[str, str]) -> None:
        self.sent.append((topic, value, headers))

    def flush(self, timeout: float) -> None:
        return None


def test_kafka_source_commits_after_effects_and_dead_letters_bad_messages(
        live: tuple[CorpusRuntime, EventProcessor]) -> None:
    """Contract test with a fake consumer; no broker is involved."""
    runtime, _ = live
    doc = _doc(1, "Pangolin discharge summary.")
    event = ChangeEvent("evt_k1", doc.tenant_id, doc.document_id, 1, "upsert", datetime(2026, 1, 1),
                        doc.model_dump(mode="json"))
    assert ChangeEvent.from_json(event.to_json()).document() == doc          # the wire format round-trips
    broken = ChangeEvent("evt_k2", doc.tenant_id, "doc_other", 1, "upsert", datetime(2026, 1, 1), {"bad": True})
    consumer, producer = _FakeConsumer([event.to_json().encode(), b"not json", broken.to_json().encode()]), _FakeProducer()
    source = KafkaEventSource("unused:9092", "corpus.document-changes", consumer=consumer, producer=producer)
    processor = EventProcessor(source, runtime.session_factory, runtime.indexes, max_attempts=3)
    report = processor.process_batch()
    # the broken event is retried inline three times, then dead-lettered: no offset is stepped over
    assert report["outcomes"] == {"inserted": 1, "dead_letter": 1} and report["events"] == 2
    assert source._attempts == {}
    assert _hits(runtime, "pangolin discharge summary", doc.document_id)
    assert len(consumer.committed) == 3                                      # good, malformed and exhausted
    assert [(t, h["error"]) for t, _, h in producer.sent] == [
        ("corpus.document-changes.dlq", "malformed_event"), ("corpus.document-changes.dlq", "ValidationError")]


def test_scheduler_runs_due_tasks_isolates_failures_and_detects_drift(
        live: tuple[CorpusRuntime, EventProcessor], tmp_path: Path) -> None:
    runtime, _ = live
    clock = [0.0]
    scheduler = CorpusScheduler(runtime, {"incremental_index": 60.0, "drift": 600.0, "consistency": 3600.0},
                                clock=lambda: clock[0])
    doc = _doc(1, "Quetzal medication reconciliation.")
    _emit(runtime, doc)
    first = {o["task"]: o for o in scheduler.run_pending()}
    assert set(first) == {"incremental_index", "drift", "consistency"}
    assert all(o["status"] == "ok" for o in first.values())
    assert first["incremental_index"]["result"]["events"]["outcomes"] == {"inserted": 1}
    assert _hits(runtime, "quetzal medication reconciliation", doc.document_id)
    assert scheduler.run_pending() == []                                     # nothing is due yet
    clock[0] = 61.0
    assert [o["task"] for o in scheduler.run_pending()] == ["incremental_index"]
    # the index silently loses a document: the consistency task finds and repairs it
    runtime.backend.delete_by_filter(runtime.indexes.active(), FilterSpec(terms={"document_id": [doc.document_id]}))
    assert not _hits(runtime, "quetzal medication reconciliation but different words", doc.document_id)
    repaired = TASKS["consistency"](runtime)
    assert repaired["missing"] == 1 and repaired["ok"] is False and repaired["sync"]["documents"] == 1
    assert TASKS["consistency"](runtime)["ok"] is True
    # a failing task is recorded and the others still run
    def boom(_: CorpusRuntime) -> dict[str, Any]:
        raise RuntimeError("search cluster unreachable")

    scheduler.tasks[1].function = boom
    clock[0] = 10_000.0
    outcomes = {o["task"]: o["status"] for o in scheduler.run_pending()}
    assert outcomes == {"incremental_index": "ok", "drift": "failed", "consistency": "ok"}
    assert scheduler.tasks[1].failures == 1 and scheduler.tasks[1].last_result == {"error": "RuntimeError"}
    # drift: the configured chunking no longer matches what the active index was built with
    changed = runtime.config.model_copy(update={
        "chunking": runtime.config.chunking.model_copy(update={"max_tokens": 64, "overlap": 8})})
    other, _ = build_runtime(tmp_path / "other", changed, runtime.session_factory)
    other.backend, other.indexes.backend = runtime.backend, runtime.backend
    report = TASKS["drift"](other)
    assert report["ok"] is False and "chunking" in report["problems"][0]
    assert TASKS["drift"](runtime)["ok"] is True
