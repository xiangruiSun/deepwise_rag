"""Search backend abstraction and the dependency-free local implementation.

Two interchangeable backends implement ``SearchBackend``:

* ``ElasticsearchBackend`` (``es_backend.py``) — BM25, filters and approximate kNN;
* ``LocalSearchBackend`` (here) — the same contract with an in-process Okapi BM25
  and exact cosine / inner-product search, optionally persisted to disk.  It
  exists so the baseline runs CPU-only without infrastructure and so tests are
  hermetic.  It is a real retriever, not a stub.

In both backends structured filters are applied to the candidate population
*before* scoring (pre-filtering), for lexical and vector search alike.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.core.errors import IndexCompatibilityError, NotFoundError

VECTOR_FIELD = "embedding"
TEXT_FIELD = "content"


@dataclass
class FilterSpec:
    """Backend-neutral structured filter.

    ``terms``       field value must be one of the listed values (list fields: any overlap)
    ``all_terms``   list field must contain every listed value
    ``must_not``    field value must not be any of the listed values
    ``ranges``      inclusive numeric range ``(gte, lte)``; a missing value never matches
    ``exists``      field must be present and non-null
    """

    terms: dict[str, list[Any]] = field(default_factory=dict)
    all_terms: dict[str, list[Any]] = field(default_factory=dict)
    must_not: dict[str, list[Any]] = field(default_factory=dict)
    ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    exists: list[str] = field(default_factory=list)

    def merged(self, other: FilterSpec) -> FilterSpec:
        out = FilterSpec(
            terms=dict(self.terms), all_terms=dict(self.all_terms), must_not=dict(self.must_not),
            ranges=dict(self.ranges), exists=list(self.exists),
        )
        for key, values in other.terms.items():
            if key in out.terms:  # intersect: both constraints must hold
                out.terms[key] = [v for v in out.terms[key] if v in values]
            else:
                out.terms[key] = list(values)
        for key, values in other.all_terms.items():
            out.all_terms[key] = sorted({*out.all_terms.get(key, []), *values})
        for key, values in other.must_not.items():
            out.must_not[key] = sorted({*out.must_not.get(key, []), *values})
        for key, (lo, hi) in other.ranges.items():
            cur_lo, cur_hi = out.ranges.get(key, (None, None))
            new_lo = lo if cur_lo is None else cur_lo if lo is None else max(lo, cur_lo)
            new_hi = hi if cur_hi is None else cur_hi if hi is None else min(hi, cur_hi)
            out.ranges[key] = (new_lo, new_hi)
        out.exists = sorted({*out.exists, *other.exists})
        return out

    def matches(self, source: dict[str, Any]) -> bool:
        for key, allowed in self.terms.items():
            value = source.get(key)
            if isinstance(value, list):
                if not any(v in allowed for v in value):
                    return False
            elif value not in allowed:
                return False
        for key, required in self.all_terms.items():
            value = source.get(key)
            present = value if isinstance(value, list) else [value]
            if not all(r in present for r in required):
                return False
        for key, banned in self.must_not.items():
            value = source.get(key)
            values = value if isinstance(value, list) else [value]
            if any(v in banned for v in values):
                return False
        for key, (lo, hi) in self.ranges.items():
            value = source.get(key)
            if value is None or isinstance(value, (str, list)):
                return False
            if lo is not None and value < lo:
                return False
            if hi is not None and value > hi:
                return False
        return all(source.get(key) is not None for key in self.exists)


@dataclass
class Hit:
    id: str
    score: float
    source: dict[str, Any]


@dataclass
class IndexSpec:
    name: str
    kind: str  # "cases" | "reports" | "images"
    dimension: int | None = None
    similarity: str = "cosine"
    mapping_version: str = "v1"
    meta: dict[str, Any] = field(default_factory=dict)


class SearchBackend(Protocol):
    name: str

    def ping(self) -> bool: ...
    def create_index(self, spec: IndexSpec, recreate: bool = False) -> None: ...
    def delete_index(self, name: str) -> None: ...
    def index_exists(self, name: str) -> bool: ...
    def index_spec(self, name: str) -> IndexSpec: ...
    def list_indexes(self) -> list[str]: ...
    def bulk_index(self, name: str, docs: list[dict[str, Any]]) -> None: ...
    def delete_documents(self, name: str, ids: list[str]) -> int: ...
    def refresh(self, name: str) -> None: ...
    def get_document(self, name: str, doc_id: str) -> dict[str, Any] | None: ...
    def count(self, name: str, filters: FilterSpec | None = None) -> int: ...
    def bm25_search(self, name: str, query: str, filters: FilterSpec | None, size: int) -> list[Hit]: ...
    def knn_search(self, name: str, vector: np.ndarray, filters: FilterSpec | None, k: int,
                   num_candidates: int) -> list[Hit]: ...
    def filter_search(self, name: str, filters: FilterSpec | None, size: int, offset: int = 0,
                      sort_field: str = "case_id") -> tuple[list[Hit], int]: ...
    def put_alias(self, alias: str, index: str) -> None: ...
    def get_alias(self, alias: str) -> str | None: ...
    def stats(self, name: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Local backend
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+")


def analyze(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens (approximates the Elasticsearch standard analyzer)."""
    return _TOKEN.findall(text.lower())


class _LocalIndex:
    K1, B = 1.2, 0.75

    def __init__(self, spec: IndexSpec) -> None:
        self.spec = spec
        self.docs: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self._dirty = True
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._vec_rows: list[int] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._lengths: np.ndarray = np.zeros(0)
        self._avg_len = 0.0

    def _build(self) -> None:
        if not self._dirty:
            return
        self._ids = sorted(self.docs)
        self._postings = {}
        lengths = []
        for row, doc_id in enumerate(self._ids):
            tokens = analyze(str(self.docs[doc_id].get(TEXT_FIELD) or ""))
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, {})[row] = tf
        self._lengths = np.asarray(lengths, dtype=np.float64)
        self._avg_len = float(self._lengths.mean()) if lengths else 0.0
        self._vec_rows = [row for row, doc_id in enumerate(self._ids) if doc_id in self.vectors]
        if self._vec_rows:
            matrix = np.vstack([self.vectors[self._ids[row]] for row in self._vec_rows]).astype(np.float32)
            if self.spec.similarity == "cosine":
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                matrix = matrix / norms
            self._matrix = matrix
        else:
            self._matrix = None
        self._dirty = False

    def _mask(self, filters: FilterSpec | None) -> np.ndarray:
        if filters is None:
            return np.ones(len(self._ids), dtype=bool)
        return np.fromiter((filters.matches(self.docs[i]) for i in self._ids), dtype=bool, count=len(self._ids))

    def bm25(self, query: str, filters: FilterSpec | None, size: int) -> list[Hit]:
        self._build()
        n_docs = len(self._ids)
        if n_docs == 0:
            return []
        mask = self._mask(filters)
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in set(analyze(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            # corpus-wide statistics, as in Elasticsearch (filters do not change idf)
            idf = math.log(1.0 + (n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for row, tf in posting.items():
                denom = tf + self.K1 * (1 - self.B + self.B * self._lengths[row] / (self._avg_len or 1.0))
                scores[row] += idf * tf * (self.K1 + 1) / denom
        scores[~mask] = 0.0
        rows = [r for r in np.argsort(-scores, kind="stable")[:size] if scores[r] > 0]
        return [Hit(self._ids[r], float(scores[r]), self.docs[self._ids[r]]) for r in rows]

    def knn(self, vector: np.ndarray, filters: FilterSpec | None, k: int) -> list[Hit]:
        self._build()
        if self._matrix is None:
            return []
        if self.spec.dimension is not None and vector.shape[-1] != self.spec.dimension:
            raise IndexCompatibilityError(
                f"query vector has dimension {vector.shape[-1]} but index '{self.spec.name}' "
                f"stores {self.spec.dimension}-dimensional vectors"
            )
        mask = self._mask(filters)[self._vec_rows]
        if not mask.any():
            return []
        query = vector.astype(np.float32)
        if self.spec.similarity == "cosine":
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm
        sims = self._matrix @ query
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims, kind="stable")[:k]
        return [
            Hit(self._ids[self._vec_rows[i]], float(sims[i]), self.docs[self._ids[self._vec_rows[i]]])
            for i in order
            if np.isfinite(sims[i])
        ]


class LocalSearchBackend:
    name = "local"

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._indexes: dict[str, _LocalIndex] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        if self.directory and self.directory.exists():
            self._load_all()

    # ------------------------------------------------------------ persistence
    def _load_all(self) -> None:
        assert self.directory is not None
        alias_file = self.directory / "aliases.json"
        if alias_file.exists():
            self._aliases = json.loads(alias_file.read_text(encoding="utf-8"))
        for meta_file in sorted(self.directory.glob("*/meta.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            index = _LocalIndex(IndexSpec(**meta))
            docs_file = meta_file.parent / "docs.jsonl"
            if docs_file.exists():
                with docs_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        doc = json.loads(line)
                        index.docs[doc["_id"]] = doc["_source"]
            vec_file = meta_file.parent / "vectors.npz"
            if vec_file.exists():
                data = np.load(vec_file, allow_pickle=False)
                for doc_id, vec in zip(data["ids"].tolist(), data["vectors"], strict=True):
                    index.vectors[doc_id] = vec
            self._indexes[index.spec.name] = index

    def _persist(self, name: str) -> None:
        if self.directory is None:
            return
        index = self._indexes[name]
        folder = self.directory / name
        folder.mkdir(parents=True, exist_ok=True)
        spec = index.spec
        (folder / "meta.json").write_text(
            json.dumps({"name": spec.name, "kind": spec.kind, "dimension": spec.dimension,
                        "similarity": spec.similarity, "mapping_version": spec.mapping_version,
                        "meta": spec.meta}, sort_keys=True),
            encoding="utf-8",
        )
        with (folder / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for doc_id in sorted(index.docs):
                fh.write(json.dumps({"_id": doc_id, "_source": index.docs[doc_id]}, sort_keys=True,
                                    default=str) + "\n")
        if index.vectors:
            ids = sorted(index.vectors)
            np.savez(folder / "vectors.npz", ids=np.asarray(ids),
                     vectors=np.vstack([index.vectors[i] for i in ids]).astype(np.float32))
        elif (folder / "vectors.npz").exists():
            (folder / "vectors.npz").unlink()

    def _persist_aliases(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "aliases.json").write_text(json.dumps(self._aliases, sort_keys=True),
                                                         encoding="utf-8")

    # ---------------------------------------------------------------- contract
    def _resolve(self, name: str) -> _LocalIndex:
        target = self._aliases.get(name, name)
        try:
            return self._indexes[target]
        except KeyError as exc:
            raise NotFoundError(f"search index '{name}' does not exist") from exc

    def ping(self) -> bool:
        return True

    def create_index(self, spec: IndexSpec, recreate: bool = False) -> None:
        with self._lock:
            if spec.name in self._indexes:
                if not recreate:
                    existing = self._indexes[spec.name].spec
                    if (existing.dimension, existing.similarity) != (spec.dimension, spec.similarity):
                        raise IndexCompatibilityError(
                            f"index '{spec.name}' exists with dimension={existing.dimension}, "
                            f"similarity={existing.similarity}; refusing to mix incompatible vectors"
                        )
                    existing.meta.update(spec.meta)
                    return
                self.delete_index(spec.name)
            self._indexes[spec.name] = _LocalIndex(spec)
            self._persist(spec.name)

    def delete_index(self, name: str) -> None:
        with self._lock:
            self._indexes.pop(name, None)
            self._aliases = {a: i for a, i in self._aliases.items() if i != name}
            if self.directory is not None:
                folder = self.directory / name
                if folder.exists():
                    for f in folder.iterdir():
                        f.unlink()
                    folder.rmdir()
                self._persist_aliases()

    def index_exists(self, name: str) -> bool:
        return self._aliases.get(name, name) in self._indexes

    def index_spec(self, name: str) -> IndexSpec:
        return self._resolve(name).spec

    def list_indexes(self) -> list[str]:
        return sorted(self._indexes)

    def bulk_index(self, name: str, docs: list[dict[str, Any]]) -> None:
        with self._lock:
            index = self._resolve(name)
            for doc in docs:
                source = dict(doc)
                doc_id = str(source.pop("_id"))
                vector = source.pop(VECTOR_FIELD, None)
                if vector is not None:
                    arr = np.asarray(vector, dtype=np.float32)
                    if index.spec.dimension is not None and arr.shape[0] != index.spec.dimension:
                        raise IndexCompatibilityError(
                            f"document vector has dimension {arr.shape[0]}; index '{name}' "
                            f"expects {index.spec.dimension}"
                        )
                    index.vectors[doc_id] = arr
                else:
                    index.vectors.pop(doc_id, None)
                index.docs[doc_id] = source
            index._dirty = True

    def delete_documents(self, name: str, ids: list[str]) -> int:
        with self._lock:
            index = self._resolve(name)
            removed = 0
            for doc_id in ids:
                if index.docs.pop(doc_id, None) is not None:
                    removed += 1
                index.vectors.pop(doc_id, None)
            index._dirty = True
            return removed

    def refresh(self, name: str) -> None:
        with self._lock:
            index = self._resolve(name)
            index._build()
            self._persist(index.spec.name)

    def get_document(self, name: str, doc_id: str) -> dict[str, Any] | None:
        return self._resolve(name).docs.get(doc_id)

    def count(self, name: str, filters: FilterSpec | None = None) -> int:
        with self._lock:
            index = self._resolve(name)
            index._build()
            return int(index._mask(filters).sum())

    def bm25_search(self, name: str, query: str, filters: FilterSpec | None, size: int) -> list[Hit]:
        with self._lock:
            return self._resolve(name).bm25(query, filters, size)

    def knn_search(self, name: str, vector: np.ndarray, filters: FilterSpec | None, k: int,
                   num_candidates: int) -> list[Hit]:
        # exact search: num_candidates is irrelevant but accepted for contract parity
        with self._lock:
            return self._resolve(name).knn(vector, filters, k)

    def filter_search(self, name: str, filters: FilterSpec | None, size: int, offset: int = 0,
                      sort_field: str = "case_id") -> tuple[list[Hit], int]:
        with self._lock:
            index = self._resolve(name)
            index._build()
            mask = index._mask(filters)
            ids = [doc_id for doc_id, keep in zip(index._ids, mask, strict=True) if keep]
            ids.sort(key=lambda i: (str(index.docs[i].get(sort_field, "")), i))
            page = ids[offset : offset + size]
            return [Hit(i, 0.0, index.docs[i]) for i in page], len(ids)

    def put_alias(self, alias: str, index: str) -> None:
        with self._lock:
            if index not in self._indexes:
                raise NotFoundError(f"cannot alias missing index '{index}'")
            self._aliases[alias] = index
            self._persist_aliases()

    def get_alias(self, alias: str) -> str | None:
        return self._aliases.get(alias)

    def stats(self, name: str) -> dict[str, Any]:
        index = self._resolve(name)
        vector_bytes = sum(v.nbytes for v in index.vectors.values())
        text_bytes = sum(len(json.dumps(d, default=str)) for d in index.docs.values())
        return {
            "backend": self.name,
            "documents": len(index.docs),
            "vectors": len(index.vectors),
            "size_bytes": int(vector_bytes + text_bytes),
            "dimension": index.spec.dimension,
            "similarity": index.spec.similarity,
            "health": "green",
        }
