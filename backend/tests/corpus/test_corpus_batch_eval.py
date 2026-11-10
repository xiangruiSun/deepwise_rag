"""Durable batch jobs, the evaluation platform, their API endpoints and the Redis cache backend."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.container import AppContext
from app.core.errors import NotFoundError, ValidationFailure
from app.corpus.batch import BatchJobManager
from app.corpus.cache import CorpusCache, RedisCache
from app.corpus.config import CacheParams
from app.corpus.evaluation import EvalDataset, ExperimentTracker, compare, evaluate_strategy, score_query
from app.main import create_app
from tests.corpus.conftest import Corpus

KEYS = {
    "t0-key": {"user": "alice", "role": "researcher", "scopes": [], "tenant": "tenant-000"},
    "t1-key": {"user": "bob", "role": "researcher", "scopes": [], "tenant": "tenant-001"},
    "t0-admin": {"user": "root", "role": "admin", "scopes": [], "tenant": "tenant-000"},
    "t1-admin": {"user": "root1", "role": "admin", "scopes": [], "tenant": "tenant-001"},
}


def _manager(corpus: Corpus, tmp_path: Path, **kw: Any) -> BatchJobManager:
    return BatchJobManager(corpus.runtime.service, corpus.runtime.session_factory, tmp_path / "batch",
                           **{"batch_size": 4, "concurrency": 3, **kw})


def _items(corpus: Corpus, n: int) -> list[dict[str, Any]]:
    queries = [q for q in corpus.by_category("keyword") + corpus.by_category("abbreviation")
               if q["tenant_id"] == "tenant-000"]
    return [{"id": f"item-{i}", "query": queries[i % len(queries)]["question"]} for i in range(n)]


def _outputs(job: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in sorted(Path(job["output_artifact_uri"]).glob("part-*.jsonl")):
        rows.extend(json.loads(line) for line in part.read_text().splitlines())
    return rows


# ----------------------------------------------------------------------- batch
def test_batch_job_survives_interruption_and_never_duplicates_outputs(corpus: Corpus, tmp_path: Path) -> None:
    manager = _manager(corpus, tmp_path)
    job = manager.create(corpus.auth(), items=_items(corpus, 18), strategy="hybrid")
    assert (job["status"], job["total_items"], job["completed_items"]) == ("queued", 18, 0)
    assert job["index_version"] == corpus.runtime.indexes.active() and job["model_version"].startswith("extractive")
    assert manager.claim_next() == job["job_id"] and manager.claim_next() is None      # one worker per job
    partial = manager.run(job["job_id"], max_batches=2)                                # the worker stops here
    assert (partial["status"], partial["completed_items"]) == ("running", 8)
    # a part was written but the worker died before committing progress for it
    orphan = Path(job["output_artifact_uri"]) / "part-00002.jsonl"
    orphan.write_text('{"item_index": 8, "status": "ok", "answer": "half written"}\n')
    resumed = _manager(corpus, tmp_path).run(job["job_id"])                            # a fresh worker process
    assert (resumed["status"], resumed["completed_items"], resumed["failed_items"]) == ("completed", 18, 0)
    rows = _outputs(resumed)
    assert [r["item_index"] for r in rows] == list(range(18))                          # each item exactly once
    assert rows[8]["answer"] != "half written" and all(r["status"] == "ok" for r in rows)
    assert all(r["index_version"] == job["index_version"] and r["trace_id"] for r in rows)
    assert any(r["citations"] for r in rows)
    manifest = json.loads((Path(job["output_artifact_uri"]) / "manifest.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["parts"]) == 5
    assert len(manifest["config"]["input_sha256"]) == 64 and manifest["config"]["strategy"] == "hybrid"
    assert manager.run(job["job_id"])["completed_items"] == 18                         # a finished job is not re-run
    assert len(_outputs(resumed)) == 18


def test_batch_jobs_respect_ownership_cancellation_and_failure_budget(corpus: Corpus, tmp_path: Path) -> None:
    manager = _manager(corpus, tmp_path)
    job = manager.create(corpus.auth(), items=_items(corpus, 6), priority=1)
    with pytest.raises(NotFoundError):
        manager.get(job["job_id"], corpus.auth("tenant-001"))                          # invisible to other tenants
    with pytest.raises(ValidationFailure):
        manager.create(corpus.auth(), input_uri="../../../etc/passwd")
    with pytest.raises(ValidationFailure):
        manager.create(corpus.auth(), items=[])
    with pytest.raises(NotFoundError):
        manager.create(corpus.auth(), items=_items(corpus, 1), strategy="nope")
    urgent = manager.create(corpus.auth(), items=_items(corpus, 2), priority=9)
    assert manager.claim_next() == urgent["job_id"]                                    # higher priority first
    assert manager.cancel(job["job_id"], corpus.auth())["status"] == "cancelled"
    assert manager.run(job["job_id"])["status"] == "cancelled"
    manager.run(urgent["job_id"], max_batches=0)
    assert manager.cancel(urgent["job_id"])["cancel_requested"] is True                # running: stop after the batch
    assert manager.run(urgent["job_id"])["status"] == "cancelled"
    # items that cannot be processed are recorded; too many of them fail the job
    bad = manager.create(corpus.auth(), items=[{"id": f"bad-{i}", "query": " " * 5000} for i in range(12)])
    failed = manager.run(bad["job_id"])
    assert failed["status"] == "failed" and failed["error"] == "failure budget exceeded" and failed["failed_items"] == 8
    assert all(r["status"] == "error" and r["error"] for r in _outputs(failed))
    # a job left "running" by a dead worker goes back to the queue
    stuck = manager.create(corpus.auth(), items=_items(corpus, 2))
    manager.claim_next()
    assert manager.requeue_stale(older_than_seconds=-1) == [stuck["job_id"]] and manager.work() != []
    assert manager.get(stuck["job_id"])["status"] == "completed"
    # file input (Parquet) and the creator's permissions: a restricted document is never cited
    import pyarrow as pa
    import pyarrow.parquet as pq

    restricted = corpus.by_category("authorized_restricted")[0]
    manager.input_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"query": [restricted["question"]] * 3}), manager.input_dir / "questions.parquet")
    done = manager.run(manager.create(corpus.auth(restricted["tenant_id"]), input_uri="questions.parquet")["job_id"])
    cited = {c["document_id"] for r in _outputs(done) for c in r["citations"]}
    assert done["status"] == "completed" and not cited & set(restricted["relevant_documents"])


# ------------------------------------------------------------------ evaluation
def test_evaluation_dataset_is_validated_and_versioned_by_content(corpus: Corpus) -> None:
    dataset = EvalDataset.load(corpus.data_dir / "eval" / "queries.jsonl")
    assert dataset.version.startswith("evalset-") and dataset.provenance == {"synthetic": len(dataset.queries)}
    assert set(dataset.categories()) >= {"exact_identifier", "paraphrase", "cross_tenant", "prompt_injection"}
    edited = [dict(q) for q in dataset.queries]
    edited[0]["question"] += " ?"
    assert EvalDataset.from_rows(edited).version != dataset.version
    assert EvalDataset.from_rows([dict(q) for q in dataset.queries]).version == dataset.version
    row = dataset.queries[0]
    for broken, message in (([{k: v for k, v in row.items() if k != "provenance"}], "missing fields"),
                            ([row, row], "duplicate query_id"), ([{**row, "provenance": "guess"}], "provenance")):
        with pytest.raises(ValueError, match=message):
            EvalDataset.from_rows(broken)


def test_per_query_scores_match_hand_computation() -> None:
    query = {"query_id": "q", "category": "freshness_superseded", "relevant_documents": {"A": 3, "B": 3},
             "stale_documents": ["S"], "forbidden_documents": ["F"], "required_facts": ["resolved", "taper"],
             "expects_abstention": False, "forbidden_text": ["CANARY"]}
    answer = {"ranked_documents": ["X", "A", "S", "B"], "context_documents": ["X", "A", "S", "A"],
              "citations": [{"document_id": "A"}, {"document_id": "X"}], "abstained": False,
              "answer": "Status resolved. [S1]", "sentences": [{"text": "Status resolved.", "supported": True}],
              "removed_sentences": 1, "context": {"chunks": 4, "duplicate_pairs": 1, "tokens": 120}}
    m = score_query(query, answer, (1, 5))
    assert m["recall@1"] == 0.0 and m["recall@5"] == 1.0 and m["hit_rate@5"] == 1.0 and m["mrr@10"] == 0.5
    assert m["precision@5"] == pytest.approx(2 / 5)
    assert m["context_precision"] == 0.5 and m["context_recall"] == 0.5 and m["duplicate_context_rate"] == 0.25
    assert m["freshness_error"] == 1.0 and m["leakage"] == 0.0 and m["abstention_correct"] == 1.0
    assert m["citation_precision"] == 0.5 and m["citation_coverage"] == 0.5 and m["fact_coverage"] == 0.5
    assert m["unsupported_claim_rate"] == 0.5 and m["hallucination"] == 0.0 and m["injection_success"] == 0.0
    assert m["temporal_consistency"] == 0.0
    leaked = score_query(query, {**answer, "ranked_documents": ["F"], "answer": "canary here"}, (1,))
    assert leaked["leakage"] == 1.0 and leaked["injection_success"] == 1.0
    absent = {"query_id": "n", "category": "no_answer", "relevant_documents": {}, "expects_abstention": True}
    assert score_query(absent, {"abstained": True, "answer": ""}, (1,))["hallucination"] == 0.0
    wrong = score_query(absent, {"abstained": False, "answer": "Made up.", "sentences": [{"supported": True}]}, (1,))
    assert wrong["hallucination"] == 1.0 and wrong["abstention_correct"] == 0.0 and "ndcg@1" not in wrong


def test_experiment_run_is_reproducible_gated_and_explains_differences(corpus: Corpus, tmp_path: Path) -> None:
    dataset = EvalDataset.load(corpus.data_dir / "eval" / "queries.jsonl")
    tracker = ExperimentTracker(corpus.runtime.service, tmp_path / "runs")
    run = tracker.run(dataset, ["bm25", "hybrid", "hybrid_rerank_routed", "no_freshness"], baseline="bm25", run_id="r1")
    again = tracker.run(dataset, ["bm25", "hybrid", "hybrid_rerank_routed", "no_freshness"], baseline="bm25", run_id="r2")
    by_name = {s["strategy"]: s for s in run["strategies"]}
    quality = [k for k in by_name["bm25"]["aggregate"] if "context_tokens" not in k]
    for name, summary in by_name.items():                                              # deterministic metrics
        other = next(s for s in again["strategies"] if s["strategy"] == name)
        assert {k: summary["aggregate"][k] for k in quality} == {k: other["aggregate"][k] for k in quality}
    assert run["config_hash"] == again["config_hash"] and run["dataset_version"] == dataset.version
    assert run["gates_passed"] and all(s["gates"] == {"leakage": True, "injection_success": True, "no_errors": True}
                                       for s in run["strategies"])
    assert all(s["aggregate"]["leakage"] == 0.0 for s in run["strategies"])
    assert by_name["hybrid"]["aggregate"]["freshness_error"] == 0.0
    assert by_name["no_freshness"]["aggregate"]["freshness_error"] > 0.0               # the filter is what prevents it
    # unanswerable and forbidden questions must be declined by every strategy
    for summary in run["strategies"]:
        assert summary["by_category"]["no_answer"]["abstention_correct"] == 1.0
        assert summary["by_category"]["cross_tenant"]["abstention_correct"] == 1.0
        assert summary["aggregate"]["hallucination"] == 0.0
    for field in ("git_commit", "embedding", "embedding_identity", "generation", "prompt_version", "index_version",
                  "strategies", "seed", "chunking", "dataset_version"):
        assert field in run["config"]
    assert run["config"]["strategies"]["hybrid"]["rrf_k"] == 60 and run["hardware"]["cpu_count"]
    lat = by_name["hybrid"]["latency_ms"]
    assert lat["end_to_end"]["p50"] <= lat["end_to_end"]["p95"] <= lat["end_to_end"]["p99"] and lat["retrieval"]["n"] > 0
    routed = next(c for c in run["comparisons"] if c["candidate"] == "hybrid_rerank_routed")
    assert routed["baseline"] == "bm25" and routed["metrics"]["ndcg@10"]["n_queries"] > 50
    assert routed["metrics"]["ndcg@10"]["verdict"] in {"improved", "regressed", "no credible difference"}
    assert routed["metrics"]["ndcg@10"]["ci_low"] <= routed["metrics"]["ndcg@10"]["mean_difference"] <= \
        routed["metrics"]["ndcg@10"]["ci_high"]
    assert isinstance(routed["improved_queries"], list) and isinstance(routed["regressed_by_category"], dict)
    run_dir = Path(run["artifact_dir"])
    assert {p.name for p in run_dir.iterdir()} == {"run.json", "report.md", "per_query.jsonl", "per_query.parquet"}
    report = (run_dir / "report.md").read_text()
    assert "Synthetic data and labels" in report and "Paired comparison against `bm25`" in report
    assert "nDCG@10 by query category" in report and "no_freshness" in report
    rows = [json.loads(line) for line in (run_dir / "per_query.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * len(dataset.queries) and "question" not in rows[0]         # identifiers, not text
    # regression detection against an earlier run
    worse = json.loads(json.dumps(again))
    worse["strategies"][0]["aggregate"]["ndcg@10"] -= 0.2
    worse["strategies"][1]["gates"]["leakage"] = False
    found = ExperimentTracker.regressions(run, worse)
    assert {"strategy": "bm25", "metric": "ndcg@10"}.items() <= found[0].items()
    assert any(f["metric"] == "release_gates" and f["failed"] == ["leakage"] for f in found)
    assert ExperimentTracker.regressions(run, again) == []
    with pytest.raises(ValueError, match="baseline"):
        tracker.run(dataset, ["bm25"], baseline="hybrid")
    one = evaluate_strategy(corpus.runtime.service, dataset, "bm25", limit=20)
    assert compare(one, one)["metrics"]["ndcg@10"]["verdict"] == "no credible difference"


# ------------------------------------------------------------------------- API
@pytest.fixture()
def client(ctx: AppContext, corpus: Corpus) -> Iterator[TestClient]:
    settings = ctx.settings.model_copy(update={"api_keys_json": json.dumps(KEYS)})
    with TestClient(create_app(dataclasses.replace(ctx, settings=settings))) as test_client:
        test_client.app.state.corpus = corpus.runtime  # type: ignore[attr-defined]
        yield test_client


def test_batch_job_endpoints(client: TestClient, corpus: Corpus) -> None:
    headers = {"X-API-Key": "t0-key"}
    items = [{"id": i["id"], "query": i["query"]} for i in _items(corpus, 5)]
    created = client.post("/v1/batch-jobs", json={"items": items, "strategy": "bm25", "run_now": True}, headers=headers)
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    status = client.get(f"/v1/batch-jobs/{job_id}", headers=headers).json()            # background task has run
    assert (status["status"], status["completed_items"], status["total_items"]) == ("completed", 5, 5)
    assert status["strategy"] == "bm25" and status["output_artifact_uri"]
    assert client.get(f"/v1/batch-jobs/{job_id}", headers={"X-API-Key": "t1-key"}).status_code == 404
    assert client.post(f"/v1/batch-jobs/{job_id}/cancel", headers={"X-API-Key": "t1-key"}).status_code == 404
    queued = client.post("/v1/batch-jobs", json={"items": items}, headers=headers).json()
    assert queued["status"] == "queued"
    assert client.post(f"/v1/batch-jobs/{queued['job_id']}/cancel", headers=headers).json()["status"] == "cancelled"
    assert client.post("/v1/batch-jobs", json={"items": items, "input_uri": "x.jsonl"}, headers=headers).status_code == 422
    assert client.post("/v1/batch-jobs", json={}, headers=headers).status_code == 422
    assert client.post("/v1/batch-jobs", json={"input_uri": "../secrets.jsonl"}, headers=headers).status_code == 422
    assert client.post("/v1/batch-jobs", json={"items": items}).status_code == 401


def test_evaluation_endpoints_are_admin_only_and_report_metrics(client: TestClient, corpus: Corpus) -> None:
    corpus.runtime.eval_dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(corpus.data_dir / "eval" / "queries.jsonl", corpus.runtime.eval_dataset_dir / "synthetic.jsonl")
    body = {"dataset": "synthetic.jsonl", "strategies": ["bm25", "hybrid"], "baseline": "bm25", "limit": 40}
    assert client.post("/v1/evaluations", json=body, headers={"X-API-Key": "t0-key"}).status_code == 403
    started = client.post("/v1/evaluations", json=body, headers={"X-API-Key": "t0-admin"})
    assert started.status_code == 202, started.text
    run_id = started.json()["run_id"]
    result = client.get(f"/v1/evaluations/{run_id}", headers={"X-API-Key": "t0-admin"}).json()
    assert result["status"] == "completed" and result["dataset_version"].startswith("evalset-")
    assert [s["strategy"] for s in result["metrics"]["strategies"]] == ["bm25", "hybrid"]
    assert result["metrics"]["gates_passed"] is True and Path(result["artifact_uri"], "report.md").is_file()
    assert client.get(f"/v1/evaluations/{run_id}", headers={"X-API-Key": "t1-admin"}).status_code == 403
    assert client.get(f"/v1/evaluations/{run_id}", headers={"X-API-Key": "t0-key"}).status_code == 403
    assert client.get("/v1/evaluations/eval_missing", headers={"X-API-Key": "t0-admin"}).status_code == 404
    for bad in ({**body, "dataset": "../../etc/passwd"}, {**body, "dataset": "nope.jsonl"},
                {**body, "strategies": ["nope"]}):
        assert client.post("/v1/evaluations", json=bad, headers={"X-API-Key": "t0-admin"}).status_code == 404
    assert client.post("/v1/evaluations", json={**body, "baseline": "dense"},
                       headers={"X-API-Key": "t0-admin"}).status_code == 422


# ----------------------------------------------------------------------- Redis
def test_redis_cache_backend_and_degradation(monkeypatch) -> None:
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url, **kw: fakeredis.FakeRedis(server=server)))
    backend = RedisCache("redis://cache:6379/0")
    cache = CorpusCache(backend, CacheParams(backend="redis", retrieval_ttl_seconds=30))
    assert backend.ping() and backend.get("missing") is None
    key = cache.retrieval_key("tenant-a", "scope", "idx", "emb", "rr", {"q": "x"})
    cache.put_retrieval(key, {"chunks": [{"chunk_id": "c"}], "candidates": {}, "warnings": []})
    assert cache.get_retrieval(key)["chunks"] == [{"chunk_id": "c"}]
    assert 0 < fakeredis.FakeRedis(server=server).ttl("corpus:" + key) <= 30          # bounded lifetime, namespaced
    second = CorpusCache(RedisCache("redis://cache:6379/0"), cache.params)             # another API process
    assert second.get_retrieval(key) is not None
    second.invalidate_tenant("tenant-a")                                               # seen by every process
    assert cache.retrieval_key("tenant-a", "scope", "idx", "emb", "rr", {"q": "x"}) != key

    class Down:
        name = "redis"

        def get(self, key: str) -> bytes | None:
            raise ConnectionError("redis down")

        set = incr = get  # type: ignore[assignment]

        def ping(self) -> bool:
            return False

    broken = CorpusCache(Down(), CacheParams(backend="redis"))                         # type: ignore[arg-type]
    assert broken.retrieval_key("t", "s", "i", "e", "r", {}) is None                   # no key -> no read, no write
    assert broken.get_retrieval(None) is None and broken.get_query_embedding("e", "q") is None
    broken.put_retrieval(None, {})
    assert broken.invalidate_tenant("t") == -1


@pytest.mark.redis
def test_live_redis_round_trip() -> None:
    url = os.environ.get("INSPECT_RAG_TEST_REDIS_URL")
    if not url:
        pytest.skip("set INSPECT_RAG_TEST_REDIS_URL to run against a real Redis")
    backend = RedisCache(url, prefix="corpus-test:")
    assert backend.ping()
    backend.set("k", b"v", 5)
    assert backend.get("k") == b"v" and backend.incr("counter") >= 1
