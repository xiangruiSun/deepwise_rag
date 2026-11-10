"""Scheduled maintenance of the corpus platform.

``CorpusScheduler`` is a small in-process scheduler for local and single-node use:
each task has an interval, runs when it is due, and records its outcome.  Production
deployments run the same task functions from Kubernetes CronJobs
(``infra/k8s/cronjobs.yaml``) through ``python -m app.cli corpus schedule run-task``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.corpus.etl import reconcile
from app.corpus.events import EventProcessor, OutboxEventSource
from app.corpus.runtime import CorpusRuntime


def incremental_index(runtime: CorpusRuntime) -> dict[str, Any]:
    """Apply pending change events and any document the active index is behind on."""
    processor = EventProcessor(OutboxEventSource(runtime.session_factory), runtime.session_factory, runtime.indexes)
    events = processor.process_batch(max_events=5_000)
    return {"events": events, "sync": runtime.indexes.sync()}


def data_quality(runtime: CorpusRuntime) -> dict[str, Any]:
    """Registry totals and dead-letter counts by reason."""
    from sqlalchemy import func, select

    from app.db.models import CorpusDeadLetter, CorpusOutboxEvent

    with runtime.session_factory() as session:
        dead = dict(session.execute(select(CorpusDeadLetter.reason_code, func.count())
                                    .group_by(CorpusDeadLetter.reason_code)).all())
        dead_events = session.scalar(select(func.count()).select_from(CorpusOutboxEvent)
                                     .where(CorpusOutboxEvent.status == "dead")) or 0
    return {"registry": reconcile(runtime.session_factory), "dead_letters_by_reason": dead,
            "dead_events": int(dead_events), "ok": dead_events == 0}


def consistency(runtime: CorpusRuntime) -> dict[str, Any]:
    """Full registry/index comparison; differences are repaired by a sync."""
    result = runtime.indexes.consistency_check(repair=True)
    if result["repaired"]:
        result["sync"] = runtime.indexes.sync()
    result["ok"] = result["consistent"]
    return result


def drift(runtime: CorpusRuntime) -> dict[str, Any]:
    """Does the active index still match the configured embedding model and chunking?"""
    active = runtime.indexes.active()
    if active is None:
        return {"ok": False, "reason": "no active index"}
    meta = runtime.backend.index_spec(active).meta
    problems = []
    if meta.get("embedding_identity") != runtime.indexes.embedding_identity:
        problems.append("the active index was built with a different embedding model identity")
    if meta.get("chunking_identity") != runtime.config.chunking.identity:
        problems.append("the active index was built with a different chunking configuration")
    return {"ok": not problems, "active": active, "problems": problems, "pending": runtime.indexes.pending()}


TASKS: dict[str, Callable[[CorpusRuntime], dict[str, Any]]] = {
    "incremental_index": incremental_index, "data_quality": data_quality,
    "consistency": consistency, "drift": drift,
}
DEFAULT_INTERVALS = {"incremental_index": 60.0, "drift": 3600.0, "data_quality": 86_400.0, "consistency": 604_800.0}


@dataclass
class ScheduledTask:
    name: str
    interval_seconds: float
    function: Callable[[CorpusRuntime], dict[str, Any]]
    next_run: float = 0.0
    last_result: dict[str, Any] = field(default_factory=dict)
    runs: int = 0
    failures: int = 0


class CorpusScheduler:
    def __init__(self, runtime: CorpusRuntime, intervals: dict[str, float] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.runtime, self.clock = runtime, clock
        self.tasks = [ScheduledTask(name, seconds, TASKS[name])
                      for name, seconds in (intervals or DEFAULT_INTERVALS).items()]

    def run_pending(self) -> list[dict[str, Any]]:
        """Run every task that is due; a failing task is recorded and does not stop the others."""
        now = self.clock()
        outcomes: list[dict[str, Any]] = []
        for task in self.tasks:
            if now < task.next_run:
                continue
            started = time.perf_counter()
            try:
                task.last_result = task.function(self.runtime)
                status = "ok" if task.last_result.get("ok", True) else "attention"
            except Exception as exc:
                task.failures += 1
                task.last_result = {"error": type(exc).__name__}
                status = "failed"
            task.runs += 1
            task.next_run = now + task.interval_seconds
            outcomes.append({"task": task.name, "status": status,
                             "seconds": round(time.perf_counter() - started, 3), "result": task.last_result})
        return outcomes

    def run_forever(self, tick_seconds: float = 5.0, max_ticks: int | None = None) -> None:
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            self.run_pending()
            ticks += 1
            time.sleep(tick_seconds)
