"""Elasticsearch 8.x implementation of ``SearchBackend``.

VALIDATION STATUS: the query DSL builders are unit tested; the live integration
tests in ``tests/integration/test_elasticsearch_live.py`` run only when an
Elasticsearch node is reachable (``make up``) and are skipped otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.config import IndexConfig, Settings
from app.core.errors import DependencyUnavailable, IndexCompatibilityError, NotFoundError
from app.indexing.backend import VECTOR_FIELD, FilterSpec, Hit, IndexSpec
from app.indexing.mappings import (
    build_bm25_query,
    build_filter_clauses,
    build_knn_query,
    build_mapping,
    raw_similarity,
)

MAX_RESULT_WINDOW = 10_000  # Elasticsearch default for index.max_result_window
SEARCH_AFTER_PAGE = 5_000


class ElasticsearchBackend:
    name = "elasticsearch"

    def __init__(self, settings: Settings, index_config: IndexConfig | None = None) -> None:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailable("the 'elasticsearch' client package is not installed") from exc
        kwargs: dict[str, Any] = {"request_timeout": settings.request_timeout_seconds}
        if settings.elasticsearch_api_key:
            kwargs["api_key"] = settings.elasticsearch_api_key
        elif settings.elasticsearch_username and settings.elasticsearch_password:
            kwargs["basic_auth"] = (settings.elasticsearch_username, settings.elasticsearch_password)
        self.client = Elasticsearch(settings.elasticsearch_url, **kwargs)
        self.index_config = index_config or IndexConfig()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    # ----------------------------------------------------------------- indexes
    def create_index(self, spec: IndexSpec, recreate: bool = False) -> None:
        exists = bool(self.client.indices.exists(index=spec.name))
        if exists and recreate:
            self.client.indices.delete(index=spec.name)
            exists = False
        if exists:
            existing = self.index_spec(spec.name)
            if (existing.dimension, existing.similarity) != (spec.dimension, spec.similarity):
                raise IndexCompatibilityError(
                    f"index '{spec.name}' exists with dimension={existing.dimension}, "
                    f"similarity={existing.similarity}; refusing to mix incompatible vectors"
                )
            return
        body = build_mapping(spec, self.index_config.shards, self.index_config.replicas)
        self.client.indices.create(index=spec.name, settings=body["settings"], mappings=body["mappings"])

    def delete_index(self, name: str) -> None:
        self.client.indices.delete(index=name, ignore_unavailable=True)

    def index_exists(self, name: str) -> bool:
        return bool(self.client.indices.exists(index=name))

    def index_spec(self, name: str) -> IndexSpec:
        if not self.index_exists(name):
            raise NotFoundError(f"search index '{name}' does not exist")
        response = self.client.indices.get_mapping(index=name)
        concrete, body = next(iter(response.items()))
        mappings = body["mappings"]
        meta = dict(mappings.get("_meta", {}))
        vector = mappings.get("properties", {}).get(VECTOR_FIELD)
        return IndexSpec(
            name=concrete,
            kind=meta.pop("kind", "reports"),
            dimension=int(vector["dims"]) if vector else None,
            similarity=str((vector or {}).get("similarity") or meta.pop("similarity", "cosine")),
            mapping_version=meta.pop("mapping_version", "v1"),
            meta=meta,
        )

    def list_indexes(self) -> list[str]:
        rows = self.client.cat.indices(format="json", expand_wildcards="open")
        names = [str(r["index"]) for r in list(rows)]  # type: ignore[index, call-overload]
        return sorted(n for n in names if not n.startswith("."))

    # --------------------------------------------------------------- documents
    def bulk_index(self, name: str, docs: list[dict[str, Any]]) -> None:
        from elasticsearch import helpers

        def actions():  # type: ignore[no-untyped-def]
            for doc in docs:
                source = dict(doc)
                doc_id = source.pop("_id")
                if source.get(VECTOR_FIELD) is not None:
                    source[VECTOR_FIELD] = [float(x) for x in source[VECTOR_FIELD]]
                else:
                    source.pop(VECTOR_FIELD, None)
                yield {"_op_type": "index", "_index": name, "_id": doc_id, "_source": source}

        helpers.bulk(self.client, actions(), chunk_size=500, refresh=False)

    def delete_documents(self, name: str, ids: list[str]) -> int:
        from elasticsearch import helpers

        actions = ({"_op_type": "delete", "_index": name, "_id": i} for i in ids)
        ok, _errors = helpers.bulk(self.client, actions, raise_on_error=False, refresh=False)
        return int(ok)

    def refresh(self, name: str) -> None:
        self.client.indices.refresh(index=name)

    def get_document(self, name: str, doc_id: str) -> dict[str, Any] | None:
        response = self.client.options(ignore_status=404).get(
            index=name, id=doc_id, source_excludes=[VECTOR_FIELD]
        )
        return response.get("_source") if response.get("found") else None

    def count(self, name: str, filters: FilterSpec | None = None) -> int:
        return int(self.client.count(index=name, query=build_filter_clauses(filters))["count"])

    # ------------------------------------------------------------------ search
    @staticmethod
    def _hits(response: Any, transform=None) -> list[Hit]:  # type: ignore[no-untyped-def]
        out = []
        for h in response["hits"]["hits"]:
            score = float(h["_score"]) if h.get("_score") is not None else 0.0
            out.append(Hit(h["_id"], transform(score) if transform else score, h.get("_source", {})))
        return out

    def bm25_search(self, name: str, query: str, filters: FilterSpec | None, size: int) -> list[Hit]:
        return self._hits(self.client.search(index=name, **build_bm25_query(query, filters, size)))

    def knn_search(self, name: str, vector: np.ndarray, filters: FilterSpec | None, k: int,
                   num_candidates: int) -> list[Hit]:
        spec = self.index_spec(name)
        if spec.dimension is not None and vector.shape[-1] != spec.dimension:
            raise IndexCompatibilityError(
                f"query vector has dimension {vector.shape[-1]} but index '{name}' stores "
                f"{spec.dimension}-dimensional vectors"
            )
        body = build_knn_query([float(x) for x in vector], filters, k, num_candidates)
        response = self.client.search(index=name, **body)
        return self._hits(response, lambda s: raw_similarity(s, spec.similarity))

    def filter_search(self, name: str, filters: FilterSpec | None, size: int, offset: int = 0,
                      sort_field: str = "case_id") -> tuple[list[Hit], int]:
        query = build_filter_clauses(filters)
        sort = [{sort_field: "asc"}, {"_doc": "asc"}]
        if offset + size <= MAX_RESULT_WINDOW:
            response = self.client.search(
                index=name, query=query, size=size, from_=offset, sort=sort, track_total_hits=True,
                source_excludes=[VECTOR_FIELD],
            )
            return self._hits(response), int(response["hits"]["total"]["value"])
        # Deep pages: from+size may not exceed index.max_result_window, so walk the sorted
        # result set with search_after instead.
        hits: list[Hit] = []
        total = 0
        skipped = 0
        after: list[Any] | None = None
        while len(hits) < size:
            kwargs: dict[str, Any] = {"search_after": after} if after is not None else {}
            response = self.client.search(
                index=name, query=query, size=SEARCH_AFTER_PAGE, sort=sort, track_total_hits=True,
                source_excludes=[VECTOR_FIELD], **kwargs,
            )
            total = int(response["hits"]["total"]["value"])
            page = response["hits"]["hits"]
            if not page:
                break
            after = page[-1]["sort"]
            for raw, hit in zip(page, self._hits(response), strict=True):
                del raw
                if skipped < offset:
                    skipped += 1
                elif len(hits) < size:
                    hits.append(hit)
        return hits, total

    # ----------------------------------------------------------------- aliases
    def put_alias(self, alias: str, index: str) -> None:
        actions: list[dict[str, Any]] = []
        if self.client.indices.exists_alias(name=alias):
            for old in self.client.indices.get_alias(name=alias):
                actions.append({"remove": {"index": old, "alias": alias}})
        actions.append({"add": {"index": index, "alias": alias}})
        self.client.indices.update_aliases(actions=actions)  # atomic promotion

    def get_alias(self, alias: str) -> str | None:
        if not self.client.indices.exists_alias(name=alias):
            return None
        return next(iter(self.client.indices.get_alias(name=alias)), None)

    def stats(self, name: str) -> dict[str, Any]:
        stats = self.client.indices.stats(index=name)["_all"]["primaries"]
        health = self.client.cluster.health(index=name)
        spec = self.index_spec(name)
        return {
            "backend": self.name,
            "documents": int(stats["docs"]["count"]),
            "size_bytes": int(stats["store"]["size_in_bytes"]),
            "dimension": spec.dimension,
            "similarity": spec.similarity,
            "health": health.get("status"),
        }
