"""Load benchmark of a RUNNING corpus API (``/v1/query``, ``/v1/query/stream``).

    python scripts/corpus_benchmark.py --base-url http://127.0.0.1:8000 \
        --queries data/corpus_demo/eval/queries.jsonl --keys keys.json \
        --concurrency 1 4 16 32 --requests 400 --out experiments/enterprise/benchmarks/api.json

``--keys`` is a JSON file in the format of INSPECT_RAG_API_KEYS_JSON; every query is sent
with a key of its own tenant.  The script reports only what it measured: latency
percentiles, throughput, error and rejection counts, time to first token and the share
of requests served from the retrieval cache.  It makes no statement about other hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path
from typing import Any

import httpx


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    ordered = sorted(values)

    def at(q: float) -> float:
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 2)

    return {"p50": at(0.50), "p95": at(0.95), "p99": at(0.99), "mean": round(sum(ordered) / len(ordered), 2),
            "max": round(ordered[-1], 2)}


async def one_query(client: httpx.AsyncClient, key: str, body: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post("/v1/query", json=body, headers={"X-API-Key": key})
        elapsed = (time.perf_counter() - started) * 1000.0
        data = response.json() if response.status_code == 200 else {}
        return {"status": response.status_code, "ms": elapsed, "cache_hit": bool(data.get("cache_hit")),
                "abstained": bool(data.get("abstained")), "server_ms": data.get("total_ms")}
    except httpx.HTTPError as exc:
        return {"status": type(exc).__name__, "ms": (time.perf_counter() - started) * 1000.0, "cache_hit": False}


async def one_stream(client: httpx.AsyncClient, key: str, body: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    first_token = first_event = None
    try:
        async with client.stream("POST", "/v1/query/stream", json=body, headers={"X-API-Key": key}) as response:
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    now = (time.perf_counter() - started) * 1000.0
                    first_event = first_event or now
                    if line.split(":", 1)[1].strip() == "token" and first_token is None:
                        first_token = now
            return {"status": response.status_code, "ms": (time.perf_counter() - started) * 1000.0,
                    "ttft_ms": first_token, "first_event_ms": first_event}
    except httpx.HTTPError as exc:
        return {"status": type(exc).__name__, "ms": (time.perf_counter() - started) * 1000.0, "ttft_ms": None,
                "first_event_ms": None}


async def run_level(base_url: str, work: list[tuple[str, dict[str, Any]]], concurrency: int, requests: int,
                    stream: bool, timeout: float) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(work[i % len(work)])
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=timeout) as client:
        async def worker() -> None:
            while True:
                try:
                    key, body = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await (one_stream if stream else one_query)(client, key, body))

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - started
    ok = [r for r in results if r["status"] == 200]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    out: dict[str, Any] = {
        "concurrency": concurrency, "requests": len(results), "seconds": round(elapsed, 3),
        "throughput_rps": round(len(ok) / elapsed, 2) if elapsed else None, "statuses": statuses,
        "error_rate": round(1 - len(ok) / max(1, len(results)), 4), "latency_ms": percentiles([r["ms"] for r in ok])}
    if stream:
        out["time_to_first_token_ms"] = percentiles([r["ttft_ms"] for r in ok if r["ttft_ms"] is not None])
        out["time_to_first_event_ms"] = percentiles([r["first_event_ms"] for r in ok if r["first_event_ms"] is not None])
        out["streams_with_tokens"] = sum(1 for r in ok if r["ttft_ms"] is not None)
    else:
        out["cache_hit_rate"] = round(sum(1 for r in ok if r["cache_hit"]) / max(1, len(ok)), 4)
        out["abstention_rate"] = round(sum(1 for r in ok if r.get("abstained")) / max(1, len(ok)), 4)
        out["server_side_ms"] = percentiles([float(r["server_ms"]) for r in ok if r.get("server_ms") is not None])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--distinct", type=int, default=0, help="use only the first N queries (more cache hits)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    keys = json.loads(args.keys.read_text())
    key_of = {spec["tenant"]: key for key, spec in keys.items() if "tenant" in spec}
    rows = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    work = []
    for q in rows:
        # questions that need an extra permission group are left out: the benchmark keys are public-only
        if q["tenant_id"] in key_of and set(q.get("groups") or []) <= {"public"}:
            body: dict[str, Any] = {"query": q["question"]}
            if args.strategy:
                body["strategy"] = args.strategy
            work.append((key_of[q["tenant_id"]], body))
    if args.distinct:
        work = work[:args.distinct]
    if not work:
        raise SystemExit("no query matches a tenant in the key file")

    health = httpx.get(args.base_url + "/readyz", timeout=10).json()
    report: dict[str, Any] = {
        "label": args.label, "base_url": args.base_url, "strategy": args.strategy or "default",
        "distinct_queries": len(work), "requests_per_level": args.requests, "readiness": health,
        "client": {"machine": platform.machine(), "platform": platform.platform(),
                   "note": "client and server ran on the same machine"},
        "query": [], "stream": []}
    asyncio.run(run_level(args.base_url, work, 2, min(40, len(work)), False, args.timeout))  # warm-up, not reported
    for concurrency in args.concurrency:
        level = asyncio.run(run_level(args.base_url, work, concurrency, args.requests, False, args.timeout))
        report["query"].append(level)
        print(json.dumps(level), flush=True)
    for concurrency in args.concurrency[:3]:
        level = asyncio.run(run_level(args.base_url, work, concurrency, max(50, args.requests // 4), True, args.timeout))
        report["stream"].append(level)
        print(json.dumps(level), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
