"""Near-real-time document updates: change events -> registry -> active index.

Events come from a *transactional outbox* (``corpus_outbox``): the writer that changes
a source record inserts the event in the same database transaction, so an event exists
if and only if the change was committed.  A relay may forward the outbox to Kafka or
Redpanda; ``KafkaEventSource`` consumes that topic.  ``OutboxEventSource`` reads the
table directly and is what local deployments and the tests use.

Delivery is at-least-once; the *effects* are exactly-once because every step is
idempotent and version-ordered:

* the registry only moves a document forward (``apply_documents``), so a replayed
  event is a no-op and an event older than the stored version is ignored;
* the index sync writes only documents whose stored version differs from the indexed
  one, and reuses cached embeddings when the text did not change;
* a deletion removes every chunk of the document, and no late event can bring it back.

An event that keeps failing is parked as ``dead`` after ``max_attempts`` with its error
class recorded; it never blocks the events behind it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.corpus.etl import apply_documents
from app.corpus.indexing import IndexManager
from app.corpus.schema import CanonicalDocument
from app.corpus.telemetry import EVENTS, INDEX_LAG, INDEX_PENDING
from app.db.models import CorpusOutboxEvent


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class ChangeEvent:
    event_id: str
    tenant_id: str
    document_id: str
    document_version: int
    change_type: str  # upsert | delete
    occurred_at: datetime
    payload: dict[str, Any]

    def document(self) -> CanonicalDocument:
        return CanonicalDocument.model_validate(self.payload)

    def to_json(self) -> str:
        return json.dumps({"event_id": self.event_id, "tenant_id": self.tenant_id, "document_id": self.document_id,
                           "document_version": self.document_version, "change_type": self.change_type,
                           "occurred_at": self.occurred_at.isoformat(), "payload": self.payload}, default=str)

    @classmethod
    def from_json(cls, raw: str | bytes) -> ChangeEvent:
        data = json.loads(raw)
        return cls(event_id=data["event_id"], tenant_id=data["tenant_id"], document_id=data["document_id"],
                   document_version=int(data["document_version"]), change_type=data["change_type"],
                   occurred_at=datetime.fromisoformat(data["occurred_at"]), payload=data["payload"])


def publish(session: Session, document: CanonicalDocument, occurred_at: datetime | None = None) -> str:
    """Add a change event to the outbox inside the caller's transaction and return its id."""
    event_id = "evt_" + uuid.uuid4().hex[:24]
    session.add(CorpusOutboxEvent(
        event_id=event_id, tenant_id=document.tenant_id, document_id=document.document_id,
        document_version=document.document_version, change_type="delete" if document.is_deleted else "upsert",
        payload=document.model_dump(mode="json"), occurred_at=occurred_at or _now()))
    return event_id


class EventSource(Protocol):
    def poll(self, max_events: int) -> list[ChangeEvent]: ...
    def ack(self, event: ChangeEvent, outcome: str) -> None: ...
    def fail(self, event: ChangeEvent, error: str, max_attempts: int) -> bool: ...
    def lag_seconds(self) -> float: ...


class OutboxEventSource:
    """Reads pending events from the outbox table, oldest first."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def poll(self, max_events: int) -> list[ChangeEvent]:
        with self.session_factory() as session:
            rows = session.scalars(select(CorpusOutboxEvent).where(CorpusOutboxEvent.status == "pending")
                                   .order_by(CorpusOutboxEvent.occurred_at, CorpusOutboxEvent.event_id)
                                   .limit(max_events))
            return [ChangeEvent(r.event_id, r.tenant_id, r.document_id, r.document_version, r.change_type,
                                r.occurred_at, r.payload) for r in rows]

    def ack(self, event: ChangeEvent, outcome: str) -> None:
        with self.session_factory() as session:
            row = session.get(CorpusOutboxEvent, event.event_id)
            if row is not None:
                row.status, row.outcome, row.processed_at, row.error = "done", outcome, _now(), None
                session.commit()

    def fail(self, event: ChangeEvent, error: str, max_attempts: int) -> bool:
        """Record a failed attempt; returns True when the event was parked as dead."""
        with self.session_factory() as session:
            row = session.get(CorpusOutboxEvent, event.event_id)
            if row is None:
                return False
            row.attempts += 1
            row.error = error
            dead = row.attempts >= max_attempts
            if dead:
                row.status, row.outcome, row.processed_at = "dead", "dead_letter", _now()
            session.commit()
            return dead

    def lag_seconds(self) -> float:
        with self.session_factory() as session:
            oldest = session.scalar(select(func.min(CorpusOutboxEvent.occurred_at))
                                    .where(CorpusOutboxEvent.status == "pending"))
        return max(0.0, (_now() - oldest).total_seconds()) if oldest else 0.0


class KafkaEventSource:
    """Consumes change events from a Kafka / Redpanda topic (``confluent-kafka``).

    Offsets are committed only after an event's effects are durable; events that exhaust
    their attempts are produced to ``<topic>.dlq``.  A failed event is retried inline
    (``retry_inline``) until it succeeds or is dead-lettered, so a later commit can never
    step over a message that was not processed.  This adapter mirrors the outbox source and
    is exercised by a contract test with a fake consumer; it has NOT been run against a real
    broker in this repository."""

    retry_inline = True

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "corpus-indexer",
                 consumer: Any = None, producer: Any = None) -> None:
        self.topic = topic
        if consumer is None:
            from confluent_kafka import Consumer, Producer

            consumer = Consumer({"bootstrap.servers": bootstrap_servers, "group.id": group_id,
                                 "enable.auto.commit": False, "auto.offset.reset": "earliest"})
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            consumer.subscribe([topic])
        self._consumer, self._producer = consumer, producer
        self._messages: dict[str, Any] = {}
        self._attempts: dict[str, int] = {}
        self._last_lag = 0.0

    def poll(self, max_events: int) -> list[ChangeEvent]:
        events: list[ChangeEvent] = []
        for message in self._consumer.consume(num_messages=max_events, timeout=1.0):
            if message.error():
                continue
            try:
                event = ChangeEvent.from_json(message.value())
            except (ValueError, KeyError, TypeError):
                self._dead_letter(message.value(), "malformed_event")
                self._consumer.commit(message=message, asynchronous=False)
                continue
            self._messages[event.event_id] = message
            events.append(event)
        if events:
            self._last_lag = max(0.0, (_now() - min(e.occurred_at for e in events)).total_seconds())
        return events

    def ack(self, event: ChangeEvent, outcome: str) -> None:
        message = self._messages.pop(event.event_id, None)
        self._attempts.pop(event.event_id, None)
        if message is not None:
            self._consumer.commit(message=message, asynchronous=False)

    def fail(self, event: ChangeEvent, error: str, max_attempts: int) -> bool:
        attempts = self._attempts.get(event.event_id, 0) + 1
        self._attempts[event.event_id] = attempts
        if attempts < max_attempts:
            return False  # not committed: the broker redelivers it
        self._dead_letter(event.to_json().encode("utf-8"), error)
        self.ack(event, "dead_letter")
        return True

    def _dead_letter(self, value: bytes, error: str) -> None:
        if self._producer is not None:
            self._producer.produce(f"{self.topic}.dlq", value=value, headers={"error": error})
            self._producer.flush(5)

    def lag_seconds(self) -> float:
        return self._last_lag


class EventProcessor:
    def __init__(self, source: EventSource, session_factory: sessionmaker[Session], indexes: IndexManager,
                 max_attempts: int = 5) -> None:
        self.source, self.session_factory, self.indexes, self.max_attempts = source, session_factory, indexes, max_attempts

    def process_batch(self, max_events: int = 200) -> dict[str, Any]:
        """Apply one batch of events and bring the active index up to date."""
        started = time.perf_counter()
        events = self.source.poll(max_events)
        outcomes: dict[str, int] = {}
        applied: list[tuple[ChangeEvent, str]] = []
        retry_inline = bool(getattr(self.source, "retry_inline", False))
        for event in events:
            outcome = "retry"
            while True:
                try:
                    document = event.document()
                    if document.document_id != event.document_id or document.tenant_id != event.tenant_id:
                        raise ValueError("event envelope does not match its payload")
                    with self.session_factory() as session:
                        counts, conflicts = apply_documents(session, [document])
                        session.commit()
                    outcome = "conflict" if conflicts else next(iter(counts))
                    applied.append((event, outcome))
                    break
                except Exception as exc:
                    dead = self.source.fail(event, type(exc).__name__, self.max_attempts)
                    if not dead and retry_inline:
                        continue  # a log-based source must settle this message before moving on
                    outcome = "dead_letter" if dead else "retry"
                    break
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            EVENTS.labels(outcome=outcome).inc()
        sync = self.indexes.sync() if applied and self.indexes.active() else None
        for event, outcome in applied:  # acknowledged only after registry AND index reflect the change
            self.source.ack(event, outcome)
        pending = self.indexes.pending()
        INDEX_PENDING.labels(kind="upserts").set(pending["upserts"])
        INDEX_PENDING.labels(kind="deletes").set(pending["deletes"])
        lag = self.source.lag_seconds()
        INDEX_LAG.set(lag)
        return {"events": len(events), "outcomes": outcomes, "sync": sync, "lag_seconds": round(lag, 3),
                "seconds": round(time.perf_counter() - started, 3)}

    def run(self, poll_interval: float = 1.0, max_iterations: int | None = None) -> dict[str, int]:
        totals: dict[str, int] = {"batches": 0, "events": 0}
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            result = self.process_batch()
            totals["batches"] += 1
            totals["events"] += result["events"]
            if result["events"] == 0:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                time.sleep(poll_interval)
        return totals
