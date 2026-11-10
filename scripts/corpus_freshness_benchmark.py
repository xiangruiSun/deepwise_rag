"""Update-visibility lag: time from committing a change event to the change being searchable.

    cd backend && python ../scripts/corpus_freshness_benchmark.py --rounds 30 --out ../experiments/enterprise/benchmarks/freshness.json

Uses the configured database and search backend (``INSPECT_RAG_*`` settings) and an active
corpus index.  Each round publishes one event to the outbox, runs the event processor once
(what the indexer worker does on every poll) and queries until the change is visible:
a new document appears, an update replaces the text, a deletion removes it.  The measured
lag therefore EXCLUDES the worker's poll interval, which adds up to one interval on top.
The test documents are deleted again at the end.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.corpus.etl import SourceSpec, normalize_row  # noqa: E402
from app.corpus.events import EventProcessor, OutboxEventSource, publish  # noqa: E402
from app.corpus.retrieval import AuthContext  # noqa: E402
from app.corpus.runtime import standalone_runtime  # noqa: E402
from app.corpus.service import QueryRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--tenant", default="tenant-000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime = standalone_runtime(get_settings())
    processor = EventProcessor(OutboxEventSource(runtime.session_factory), runtime.session_factory, runtime.indexes)
    auth = AuthContext(args.tenant, "freshness-benchmark", frozenset({"public"}))
    spec, run = SourceSpec(uri="benchmark"), uuid.uuid4().hex[:8]
    lags: dict[str, list[float]] = {"insert": [], "update": [], "delete": []}
    processing: list[float] = []

    def visible(word: str, document_id: str) -> bool:
        request = QueryRequest(query=f"{word} freshness probe", strategy="bm25", top_k=20)
        chunks = runtime.service.retrieve(auth, request).chunks      # through the cache, like a caller
        return any(c.document_id == document_id and word in c.text for c in chunks)

    def change(row: dict[str, object], check) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        doc, reason, _ = normalize_row(row, spec, "benchmark", datetime.utcnow())
        if doc is None:
            raise SystemExit(f"benchmark row rejected: {reason}")
        started = time.perf_counter()
        with runtime.session_factory() as session:
            publish(session, doc)
            session.commit()
        t0 = time.perf_counter()
        processor.process_batch()
        spent = time.perf_counter() - t0
        while not check(doc.document_id):
            if time.perf_counter() - started > args.timeout:
                raise SystemExit("change did not become visible within the timeout")
            time.sleep(0.01)
        return time.perf_counter() - started, spent

    for i in range(args.rounds):
        first, second = f"zq{run}a{i}", f"zq{run}b{i}"
        base = {"tenant": args.tenant, "source_system": "freshness_benchmark", "record_id": f"FRESH-{run}-{i}"}
        for kind, row, check in (
            ("insert", {**base, "version": "1", "body": f"{first} freshness probe document."},
             lambda d, w=first: visible(w, d)),
            ("update", {**base, "version": "2", "body": f"{second} freshness probe document, revised."},
             lambda d, old=first, new=second: visible(new, d) and not visible(old, d)),
            ("delete", {**base, "version": "3", "body": "", "deleted": True},
             lambda d, w=second: not visible(w, d)),
        ):
            lag, spent = change(row, check)
            lags[kind].append(lag * 1000.0)
            processing.append(spent * 1000.0)

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {"p50": round(statistics.median(ordered), 1), "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
                "max": round(ordered[-1], 1), "n": len(ordered)}

    report = {
        "what": "commit of a change event -> change visible to a query (one event per processor batch)",
        "excludes": "the indexer worker's poll interval (adds up to one interval)",
        "search_backend": runtime.backend.name, "database": get_settings().database_url.split(":", 1)[0],
        "rounds": args.rounds, "visibility_lag_ms": {kind: summary(values) for kind, values in lags.items()},
        "processor_batch_ms": summary(processing),
        "configured_bound_seconds": runtime.config.consistency_interval_seconds,
        "pending_after": runtime.indexes.pending(), "consistent_after": runtime.indexes.consistency_check()["consistent"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
