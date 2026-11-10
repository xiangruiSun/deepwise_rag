"""Locust load test of the corpus API.

    CORPUS_QUERIES=data/corpus_demo/eval/queries.jsonl CORPUS_KEYS=keys.json \
      locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
             --headless -u 20 -r 5 -t 60s --csv experiments/enterprise/benchmarks/locust

Traffic mix: 60 % answers, 20 % streamed answers, 20 % retrieval only.  Users belong to
tenants (one API key each); every user only asks questions of its own tenant.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from locust import HttpUser, between, events, task

_QUERIES = [json.loads(line) for line in Path(os.environ["CORPUS_QUERIES"]).read_text().splitlines() if line.strip()]
_KEYS = json.loads(Path(os.environ["CORPUS_KEYS"]).read_text())
_BY_TENANT: dict[str, list[str]] = {}
for _q in _QUERIES:
    if set(_q.get("groups") or []) <= {"public"}:  # the load-test keys carry no extra permission group
        _BY_TENANT.setdefault(_q["tenant_id"], []).append(_q["question"])
_USERS = [(key, spec["tenant"]) for key, spec in _KEYS.items() if spec.get("tenant") in _BY_TENANT]


class CorpusUser(HttpUser):
    wait_time = between(0.05, 0.3)

    def on_start(self) -> None:
        self.key, self.tenant = random.choice(_USERS)
        self.questions = _BY_TENANT[self.tenant]
        self.headers = {"X-API-Key": self.key}

    def _body(self) -> dict[str, str]:
        return {"query": random.choice(self.questions)}

    @task(6)
    def query(self) -> None:
        with self.client.post("/v1/query", json=self._body(), headers=self.headers, name="/v1/query",
                              catch_response=True) as response:
            if response.status_code == 429:
                response.success()  # back-pressure is an expected, controlled answer under overload
            elif response.status_code != 200:
                response.failure(f"status {response.status_code}")

    @task(2)
    def stream(self) -> None:
        started = time.perf_counter()
        first_token = None
        with self.client.post("/v1/query/stream", json=self._body(), headers=self.headers, stream=True,
                              name="/v1/query/stream", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return
            for line in response.iter_lines():
                if first_token is None and line and line.decode().startswith("event: token"):
                    first_token = (time.perf_counter() - started) * 1000.0
        if first_token is not None:
            events.request.fire(request_type="TTFT", name="/v1/query/stream first token",
                                response_time=first_token, response_length=0, exception=None, context={})

    @task(2)
    def retrieve(self) -> None:
        self.client.post("/v1/retrieve", json={**self._body(), "top_k": 10}, headers=self.headers, name="/v1/retrieve")
