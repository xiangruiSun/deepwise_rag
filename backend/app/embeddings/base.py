"""Embedding provider interface, tokenizer protocol and persistent cache."""

from __future__ import annotations

import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.core.config import EmbeddingConfig
from app.core.hashing import sha256_text


class Tokenizer(Protocol):
    """Model tokenizer view used by chunkers: character spans of content tokens."""

    def token_spans(self, text: str) -> list[tuple[int, int]]: ...


_WORD = re.compile(r"\w+|[^\w\s]")


class WhitespaceTokenizer:
    """Word/punctuation tokenizer (used by the hashing provider and as a test double)."""

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


class EmbeddingProvider(ABC):
    """One embedding configuration.  Query and document encoders may differ."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.truncation = {"documents": 0, "documents_truncated": 0, "queries": 0, "queries_truncated": 0}

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        """Tokenizer of the document encoder (used to enforce input limits when chunking)."""

    @abstractmethod
    def resolved_revision(self) -> dict[str, str | None]:
        """Exact model revision(s) actually loaded."""

    def embedding_dimension(self) -> int:
        return self.config.dimension

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer().token_spans(text))

    # special tokens ([CLS]/[SEP]) reserved by transformer encoders
    special_tokens: int = 0

    @property
    def max_document_content_tokens(self) -> int:
        return self.config.max_document_tokens - self.special_tokens

    def _track(self, role: str, texts: list[str], limit: int, prefix: str = "") -> None:
        tok = self.tokenizer()
        for text in texts:
            self.truncation[role] += 1
            if len(tok.token_spans(prefix + text)) > limit - self.special_tokens:
                self.truncation[f"{role}_truncated"] += 1

    def model_metadata(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "key": cfg.key,
            "provider": cfg.provider,
            "model_id": cfg.model_id,
            "query_model_id": cfg.query_model_id,
            "revision": self.resolved_revision(),
            "device": self.device(),
            "batch_size": cfg.batch_size,
            "max_document_tokens": cfg.max_document_tokens,
            "max_query_tokens": cfg.max_query_tokens,
            "pooling": cfg.pooling,
            "normalize": cfg.normalize,
            "similarity": cfg.similarity,
            "dimension": self.embedding_dimension(),
            "query_prefix": cfg.query_prefix,
            "document_prefix": cfg.document_prefix,
            "preprocessing_version": cfg.preprocessing_version,
        }

    def device(self) -> str:
        return "cpu"

    def identity(self) -> str:
        """Identity used for cache keys and index compatibility checks."""
        rev = self.resolved_revision()
        return "|".join(
            [
                self.config.model_id,
                str(rev.get("document")),
                str(self.config.query_model_id),
                str(rev.get("query")),
                self.config.preprocessing_version,
                str(self.config.max_document_tokens),
                str(self.config.max_query_tokens),
                self.config.pooling,
                str(self.config.normalize),
                self.config.document_prefix,
                self.config.query_prefix,
            ]
        )


def materialize_parameters(model: Any) -> Any:
    """Copy memory-mapped weights into ordinary, aligned tensors.

    Weights loaded from ``.safetensors`` are memory-mapped.  On at least one tested
    stack (PyTorch 2.15 / Apple Silicon) such tensors made a BERT cross-encoder return
    NaN logits while an identical state dict loaded from ``pytorch_model.bin`` worked.
    Copying the parameters removes the dependency on the mapping's alignment.
    """
    import torch

    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.clone().contiguous()
    return model


def resolve_commit_hash(model_id: str, revision: str | None, loaded_config: Any = None) -> str | None:
    """Exact commit of the weights actually loaded (from the model config or the local
    Hugging Face cache); falls back to the configured revision."""
    commit = getattr(loaded_config, "_commit_hash", None)
    if commit:
        return str(commit)
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_id, "config.json", revision=revision)
        if isinstance(path, str) and "/snapshots/" in path:
            return path.split("/snapshots/")[1].split("/")[0]
    except Exception:  # resolution is best effort; never fail a run on it
        pass
    return revision


def assert_finite(values: np.ndarray, what: str) -> None:
    """Model outputs must be finite; a NaN must fail loudly instead of corrupting rankings."""
    if not np.isfinite(values).all():
        from app.core.errors import DependencyUnavailable

        raise DependencyUnavailable(f"{what} produced non-finite values (NaN/inf); refusing to use them")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class EmbeddingCache:
    """SQLite-backed cache keyed by content hash, model identity and encoding role.

    Embeddings from incompatible models can never be mixed: the model identity
    (model id, exact revision, preprocessing, limits, prefixes) is part of the key.
    """

    def __init__(self, path: Path | None) -> None:
        self._lock = threading.Lock()
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "key TEXT PRIMARY KEY, identity TEXT NOT NULL, role TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_emb_content ON embeddings(content_hash)")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(identity: str, role: str, content_hash: str) -> str:
        return sha256_text(f"{identity}|{role}|{content_hash}")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" * len(chunk))
                for key, blob in self._conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", chunk
                ):
                    out[key] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, rows: list[tuple[str, str, str, str, np.ndarray]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings(key, identity, role, content_hash, dim, vector) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (key, identity, role, chash, int(vec.shape[0]), vec.astype(np.float32).tobytes())
                    for key, identity, role, chash, vec in rows
                ],
            )
            self._conn.commit()

    def invalidate_content(self, content_hashes: list[str]) -> int:
        """Drop cached embeddings of obsolete content (called on source changes)."""
        removed = 0
        with self._lock:
            for start in range(0, len(content_hashes), 500):
                chunk = content_hashes[start : start + 500]
                marks = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"DELETE FROM embeddings WHERE content_hash IN ({marks})", chunk
                )
                removed += cur.rowcount
            self._conn.commit()
        return removed

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


class CachedEmbedder:
    """Wraps a provider with the cache.  Only cache misses reach the model."""

    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache, cache_queries: bool = True) -> None:
        self.provider = provider
        self.cache = cache
        # Benchmarks disable query caching so that measured query-encoding latency is real.
        self.cache_queries = cache_queries

    def _encode(self, texts: list[str], role: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.provider.embedding_dimension()), dtype=np.float32)
        identity = self.provider.identity()
        hashes = [sha256_text(t) for t in texts]
        keys = [self.cache.key(identity, role, h) for h in hashes]
        cached = self.cache.get_many(keys)
        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        # de-duplicate identical texts inside the batch
        unique: dict[str, int] = {}
        for i in missing_idx:
            unique.setdefault(keys[i], i)
        if unique:
            order = list(unique.values())
            batch = [texts[i] for i in order]
            encoded = (
                self.provider.encode_documents(batch)
                if role == "document"
                else self.provider.encode_queries(batch)
            )
            rows = []
            for vec, i in zip(encoded, order, strict=True):
                cached[keys[i]] = vec
                rows.append((keys[i], identity, role, hashes[i], vec))
            self.cache.put_many(rows)
        self.cache.hits += len(texts) - len(missing_idx)
        self.cache.misses += len(unique)
        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not self.cache_queries:
            return self.provider.encode_queries(texts).astype(np.float32)
        return self._encode(texts, "query")
