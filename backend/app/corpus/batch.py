"""Durable offline RAG jobs.

A job answers a file of questions with exactly the service the online API uses
(``CorpusService.answer``), under the permissions of the user who created it.

Durability model:

* items are processed in order, in fixed batches; each batch is written to its own
  output part (``part-00003.jsonl``) through a temp file and an atomic rename, and
  only then is the job's progress advanced in the database;
* a crashed or stopped worker therefore resumes at the first batch whose progress
  was not committed — rewriting that part replaces the same file, so completed
  outputs are never duplicated;
* a worker claims a queued job with a conditional update, so two workers cannot run
  the same job; a failure budget stops a job whose items keep failing.

Batch work is isolated from online traffic by running in separate worker processes
with their own concurrency limit and (optionally) a lower request priority at the
model server.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import AppError, NotFoundError, ValidationFailure
from app.core.hashing import file_sha256
from app.corpus.retrieval import AuthContext
from app.corpus.service import CorpusService, QueryRequest
from app.corpus.telemetry import BATCH_ITEMS
from app.db.models import CorpusBatchJob

TERMINAL = {"completed", "failed", "cancelled"}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def read_items(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the questions of a JSONL or Parquet input file."""
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def count_items(path: Path) -> int:
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


class BatchJobManager:
    def __init__(self, service: CorpusService, session_factory: sessionmaker[Session], work_dir: Path,
                 batch_size: int = 16, concurrency: int = 4, max_failure_fraction: float = 0.5) -> None:
        self.service, self.session_factory = service, session_factory
        self.work_dir = Path(work_dir)
        self.input_dir, self.output_dir = self.work_dir / "inputs", self.work_dir / "outputs"
        self.batch_size, self.concurrency, self.max_failure_fraction = batch_size, concurrency, max_failure_fraction

    # ----------------------------------------------------------------- create
    def _resolve_input(self, input_uri: str) -> Path:
        """Inputs must live inside the batch input directory: no arbitrary file access."""
        candidate = (self.input_dir / input_uri).resolve()
        if self.input_dir.resolve() not in candidate.parents or not candidate.is_file():
            raise ValidationFailure("input file not found in the batch input directory", {"input": input_uri})
        if candidate.suffix not in (".jsonl", ".parquet"):
            raise ValidationFailure("batch input must be a .jsonl or .parquet file")
        return candidate

    def create(self, auth: AuthContext, input_uri: str | None = None, items: list[dict[str, Any]] | None = None,
               strategy: str | None = None, response_length: str = "medium", priority: int = 0) -> dict[str, Any]:
        job_id = "job_" + uuid.uuid4().hex[:20]
        self.input_dir.mkdir(parents=True, exist_ok=True)
        if items is not None:
            path = self.input_dir / f"{job_id}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for item in items:
                    fh.write(json.dumps(item) + "\n")
        elif input_uri:
            path = self._resolve_input(input_uri)
        else:
            raise ValidationFailure("either 'items' or 'input_uri' is required")
        if strategy:
            try:
                self.service.config.strategy(strategy)
            except KeyError as exc:
                raise NotFoundError(f"unknown retrieval strategy '{strategy}'") from exc
        total = count_items(path)
        if total == 0:
            raise ValidationFailure("the batch input contains no items")
        job = CorpusBatchJob(
            job_id=job_id, tenant_id=auth.tenant_id, created_by=auth.user_id, scopes=sorted(auth.groups),
            status="queued", priority=priority, input_uri=str(path), output_uri=str(self.output_dir / job_id),
            total_items=total,
            config={"strategy": strategy, "response_length": response_length, "batch_size": self.batch_size,
                    "input_sha256": file_sha256(path), "model_version": self.service.model_version,
                    "index_version": self.service.indexes.active()})
        with self.session_factory() as session:
            session.add(job)
            session.commit()
        return self.describe(job)

    # ------------------------------------------------------------------ read
    @staticmethod
    def describe(job: CorpusBatchJob) -> dict[str, Any]:
        return {"job_id": job.job_id, "status": job.status, "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(), "total_items": job.total_items,
                "completed_items": job.completed_items, "failed_items": job.failed_items,
                "model_version": job.config.get("model_version"), "index_version": job.config.get("index_version"),
                "strategy": job.config.get("strategy"), "priority": job.priority,
                "input_dataset": Path(job.input_uri).name, "output_artifact_uri": job.output_uri,
                "error": job.error, "cancel_requested": job.cancel_requested}

    def _load(self, session: Session, job_id: str, auth: AuthContext | None) -> CorpusBatchJob:
        job = session.get(CorpusBatchJob, job_id)
        # a job of another tenant is indistinguishable from a missing one
        if job is None or (auth is not None and job.tenant_id != auth.tenant_id):
            raise NotFoundError(f"batch job '{job_id}' not found")
        return job

    def get(self, job_id: str, auth: AuthContext | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            return self.describe(self._load(session, job_id, auth))

    def cancel(self, job_id: str, auth: AuthContext | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            job = self._load(session, job_id, auth)
            if job.status == "queued":
                job.status = "cancelled"
            elif job.status == "running":
                job.cancel_requested = True  # the worker stops after the batch it is processing
            job.updated_at = _now()
            session.commit()
            return self.describe(job)

    # ------------------------------------------------------------------ claim
    def claim_next(self) -> str | None:
        """Atomically move the best queued job to ``running`` and return its id."""
        with self.session_factory() as session:
            for job_id in session.scalars(select(CorpusBatchJob.job_id).where(CorpusBatchJob.status == "queued")
                                          .order_by(CorpusBatchJob.priority.desc(), CorpusBatchJob.created_at)
                                          .limit(5)):
                claimed = session.execute(update(CorpusBatchJob).where(
                    CorpusBatchJob.job_id == job_id, CorpusBatchJob.status == "queued")
                    .values(status="running", updated_at=_now()))
                session.commit()
                if claimed.rowcount == 1:  # type: ignore[attr-defined]
                    return str(job_id)
        return None

    # -------------------------------------------------------------------- run
    def run(self, job_id: str, max_batches: int | None = None) -> dict[str, Any]:
        """Process a job (or continue it).  ``max_batches`` bounds one call, for tests and time slices."""
        return asyncio.run(self.run_async(job_id, max_batches))

    async def run_async(self, job_id: str, max_batches: int | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            job = self._load(session, job_id, None)
            if job.status in TERMINAL:
                return self.describe(job)
            job.status, job.updated_at = "running", _now()
            session.commit()
            auth = AuthContext(job.tenant_id, job.created_by, frozenset(job.scopes))
            config, input_path, output_dir = dict(job.config), Path(job.input_uri), Path(job.output_uri)
            done = job.completed_items + job.failed_items
            total = job.total_items
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(config["batch_size"])
        slots = asyncio.Semaphore(self.concurrency)
        items = read_items(input_path)
        for _ in range(done):  # resume: skip what earlier runs committed
            next(items, None)
        batches_run = 0
        while done < total:
            if max_batches is not None and batches_run >= max_batches:
                break
            with self.session_factory() as session:
                if self._load(session, job_id, None).cancel_requested:
                    return self._finish(job_id, "cancelled")
            batch = [item for item in (next(items, None) for _ in range(min(batch_size, total - done)))
                     if item is not None]
            if not batch:
                break

            async def one(index: int, item: dict[str, Any]) -> dict[str, Any]:
                async with slots:
                    try:
                        request = QueryRequest(query=str(item.get("query") or item.get("question") or ""),
                                               strategy=item.get("strategy") or config.get("strategy"),
                                               document_types=item.get("document_types") or [],
                                               response_length=config.get("response_length") or "medium")
                        answer = await self.service.answer(auth, request)
                        BATCH_ITEMS.labels(outcome="ok").inc()
                        return {"item_index": index, "id": item.get("id") or item.get("query_id"), "status": "ok",
                                **{k: answer[k] for k in ("answer", "citations", "abstained", "abstention_reason",
                                                          "model_version", "index_version", "strategy", "trace_id",
                                                          "total_ms", "usage", "retrieved_chunks", "degraded", "mock")}}
                    except Exception as exc:  # one bad item must not stop the job
                        BATCH_ITEMS.labels(outcome="error").inc()
                        code = exc.code if isinstance(exc, AppError) else type(exc).__name__
                        return {"item_index": index, "id": item.get("id") or item.get("query_id"),
                                "status": "error", "error": code}

            results = await asyncio.gather(*(one(done + i, item) for i, item in enumerate(batch)))
            part = output_dir / f"part-{done // batch_size:05d}.jsonl"
            tmp = part.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in results), encoding="utf-8")
            tmp.replace(part)  # the part exists completely or not at all
            failed = sum(1 for r in results if r["status"] == "error")
            with self.session_factory() as session:  # progress is committed only after the part is durable
                job = self._load(session, job_id, None)
                job.completed_items += len(results) - failed
                job.failed_items += failed
                job.updated_at = _now()
                session.commit()
                done = job.completed_items + job.failed_items
                failure_fraction = job.failed_items / max(1, done)
            batches_run += 1
            if done >= 2 * batch_size and failure_fraction > self.max_failure_fraction:
                return self._finish(job_id, "failed", "failure budget exceeded")
        if done >= total:
            return self._finish(job_id, "completed")
        return self.get(job_id)

    def _finish(self, job_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            job = self._load(session, job_id, None)
            job.status, job.error, job.updated_at = status, error, _now()
            session.commit()
            output_dir = Path(job.output_uri)
            parts = sorted(p.name for p in output_dir.glob("part-*.jsonl"))
            manifest = {**self.describe(job), "config": job.config, "parts": parts,
                        "finished_at": job.updated_at.isoformat()}
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                                                      encoding="utf-8")
            return self.describe(job)

    def requeue_stale(self, older_than_seconds: float = 900.0) -> list[str]:
        """Return jobs stuck in ``running`` (a worker died) to the queue so another worker resumes them."""
        cutoff = datetime.fromtimestamp(_now().timestamp() - older_than_seconds)
        with self.session_factory() as session:
            stale = list(session.scalars(select(CorpusBatchJob).where(CorpusBatchJob.status == "running",
                                                                     CorpusBatchJob.updated_at < cutoff)))
            for job in stale:
                job.status = "queued"
            session.commit()
            return [job.job_id for job in stale]

    def work(self, max_jobs: int | None = None) -> list[dict[str, Any]]:
        """Worker loop body: claim and run queued jobs until the queue is empty."""
        finished: list[dict[str, Any]] = []
        while max_jobs is None or len(finished) < max_jobs:
            job_id = self.claim_next()
            if job_id is None:
                break
            finished.append(self.run(job_id))
        return finished
