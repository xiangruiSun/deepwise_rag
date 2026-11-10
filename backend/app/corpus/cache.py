"""Tenant-safe caching for the corpus platform (in-process or Redis).

Every key of tenant data is derived from the caller's permission set, the index
version, the model identity and the full request shape, so an entry written for one
tenant or permission set can never be read by another.  Invalidation is O(1): each
tenant has a generation counter that is part of its keys and is bumped whenever its
documents change, which orphans all of that tenant's cached results at once.

Cached retrieval results contain document text.  The cache therefore belongs to the
same trust zone as the search index; answers are not cached unless a TTL is set.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

import numpy as np

from app.core.errors import DependencyUnavailable
from app.core.hashing import sha256_text, stable_hash
from app.corpus.config import CacheParams
from app.corpus.telemetry import CACHE_EVENTS


class CacheBackend(Protocol):
    name: str

    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...
    def incr(self, key: str) -> int: ...
    def ping(self) -> bool: ...


class NullCache:
    name = "none"

    def get(self, key: str) -> bytes | None:
        return None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        return None

    def incr(self, key: str) -> int:
        return 0

    def ping(self) -> bool:
        return True


class MemoryCache:
    """Bounded LRU with per-entry TTL (single process)."""

    name = "memory"

    def __init__(self, max_items: int = 10_000) -> None:
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float | None, bytes]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires is not None and expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl_seconds if ttl_seconds > 0 else None, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)  # evict the least recently used entry

    def incr(self, key: str) -> int:
        with self._lock:
            _expires, raw = self._data.get(key, (None, b"0"))
            value = int(raw) + 1
            self._data[key] = (None, str(value).encode())
            return value

    def ping(self) -> bool:
        return True


class RedisCache:
    name = "redis"

    def __init__(self, url: str, prefix: str = "corpus:") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailable("the 'redis' package is not installed") from exc
        self.prefix = prefix
        self._client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)

    def get(self, key: str) -> bytes | None:
        value: Any = self._client.get(self.prefix + key)
        return bytes(value) if value is not None else None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self._client.set(self.prefix + key, value, ex=ttl_seconds if ttl_seconds > 0 else None)

    def incr(self, key: str) -> int:
        return int(self._client.incr(self.prefix + key))

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


def build_cache_backend(params: CacheParams) -> CacheBackend:
    if params.backend == "redis":
        return RedisCache(params.redis_url)
    if params.backend == "memory":
        return MemoryCache(params.max_memory_items)
    return NullCache()


class CorpusCache:
    """Typed cache operations.  A failing cache backend degrades to a miss, never to an error."""

    def __init__(self, backend: CacheBackend, params: CacheParams) -> None:
        self.backend, self.params = backend, params

    # ----------------------------------------------------------------- safety
    def _get(self, cache: str, key: str) -> bytes | None:
        try:
            value = self.backend.get(key)
        except Exception:
            CACHE_EVENTS.labels(cache=cache, result="error").inc()
            return None
        CACHE_EVENTS.labels(cache=cache, result="hit" if value is not None else "miss").inc()
        return value

    def _set(self, key: str, value: bytes, ttl: int) -> None:
        try:
            self.backend.set(key, value, ttl)
        except Exception:
            CACHE_EVENTS.labels(cache="write", result="error").inc()

    # ------------------------------------------------------------ generations
    def tenant_generation(self, tenant_id: str) -> int:
        try:
            raw = self.backend.get(f"gen:{tenant_id}")
        except Exception:
            return -1  # unknown generation: keys built with it are never written by a healthy path
        return int(raw) if raw else 0

    def invalidate_tenant(self, tenant_id: str) -> int:
        """Orphan every cached result of ``tenant_id`` (called when its documents change)."""
        try:
            return self.backend.incr(f"gen:{tenant_id}")
        except Exception:
            return -1

    # ------------------------------------------------------- query embeddings
    def get_query_embedding(self, embedding_identity: str, text: str) -> np.ndarray | None:
        raw = self._get("query_embedding", f"qe:{embedding_identity}:{sha256_text(text)}")
        return np.frombuffer(raw, dtype=np.float32).copy() if raw is not None else None

    def put_query_embedding(self, embedding_identity: str, text: str, vector: np.ndarray) -> None:
        self._set(f"qe:{embedding_identity}:{sha256_text(text)}", np.asarray(vector, dtype=np.float32).tobytes(),
                  self.params.query_embedding_ttl_seconds)

    # -------------------------------------------------------------- retrieval
    def retrieval_key(self, tenant_id: str, scope_key: str, index_version: str, embedding_identity: str,
                      reranker_identity: str, request: dict[str, Any]) -> str | None:
        generation = self.tenant_generation(tenant_id)
        if generation < 0:
            return None
        shape = stable_hash({"index": index_version, "embedding": embedding_identity,
                             "reranker": reranker_identity, "request": request})
        return f"ret:{tenant_id}:{generation}:{scope_key}:{shape}"

    def get_retrieval(self, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        raw = self._get("retrieval", key)
        return json.loads(raw) if raw is not None else None

    def put_retrieval(self, key: str | None, payload: dict[str, Any]) -> None:
        if key is not None:
            self._set(key, json.dumps(payload).encode("utf-8"), self.params.retrieval_ttl_seconds)

    # ---------------------------------------------------------------- answers
    def answer_key(self, retrieval_key: str | None, prompt_version: str, model: str, extra: dict[str, Any]) -> str | None:
        if retrieval_key is None or self.params.answer_ttl_seconds <= 0:
            return None
        return "ans:" + retrieval_key[4:] + ":" + stable_hash({"prompt": prompt_version, "model": model, **extra})

    def get_answer(self, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        raw = self._get("answer", key)
        return json.loads(raw) if raw is not None else None

    def put_answer(self, key: str | None, answer: dict[str, Any]) -> None:
        if key is not None:
            self._set(key, json.dumps(answer, default=str).encode("utf-8"), self.params.answer_ttl_seconds)
