"""Elasticsearch mappings and query-DSL construction (pure functions, unit tested)."""

from __future__ import annotations

from typing import Any

from app.indexing.backend import TEXT_FIELD, VECTOR_FIELD, FilterSpec, IndexSpec

_CASE_METADATA: dict[str, Any] = {
    "case_id": {"type": "keyword"},
    "patient_id": {"type": "keyword"},
    "dataset_split": {"type": "keyword"},
    "access_scope": {"type": "keyword"},
    "age": {"type": "float"},
    "gender": {"type": "keyword"},
    "race": {"type": "keyword"},
    "ethnicity": {"type": "keyword"},
    "diagnosis_codes": {"type": "keyword"},
    "study_modality": {"type": "keyword"},
    "has_report": {"type": "boolean"},
    "has_image": {"type": "boolean"},
}

# every ``lab_<concept>`` field is a float usable in numeric range filters
_DYNAMIC_TEMPLATES = [
    {"lab_values": {"match": "lab_*", "mapping": {"type": "float"}}},
    {"strings_as_keywords": {"match_mapping_type": "string", "mapping": {"type": "keyword"}}},
]


def _vector_field(dimension: int, similarity: str) -> dict[str, Any]:
    return {"type": "dense_vector", "dims": dimension, "index": True, "similarity": similarity}


def build_mapping(spec: IndexSpec, shards: int = 1, replicas: int = 0) -> dict[str, Any]:
    properties: dict[str, Any] = dict(_CASE_METADATA)
    if spec.kind == "reports":
        properties.update(
            {
                "document_id": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "parent_document_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunking": {"type": "keyword"},
                TEXT_FIELD: {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                "source_timestamp": {"type": "date", "ignore_malformed": True},
                "embedding_model_id": {"type": "keyword"},
                "embedding_revision": {"type": "keyword"},
                "source_version": {"type": "keyword"},
                "content_hash": {"type": "keyword"},
            }
        )
    elif spec.kind == "images":
        properties.update(
            {
                "study_id": {"type": "keyword"},
                "provider_id": {"type": "keyword"},
                "provider_version": {"type": "keyword"},
            }
        )
    if spec.dimension is not None:
        properties[VECTOR_FIELD] = _vector_field(spec.dimension, spec.similarity)
    return {
        "settings": {"number_of_shards": shards, "number_of_replicas": replicas},
        "mappings": {
            "dynamic": True,
            "dynamic_templates": _DYNAMIC_TEMPLATES,
            "_meta": {"kind": spec.kind, "mapping_version": spec.mapping_version,
                      "similarity": spec.similarity, **spec.meta},
            "properties": properties,
        },
    }


def build_filter_clauses(filters: FilterSpec | None) -> dict[str, Any]:
    """Compile a ``FilterSpec`` into an Elasticsearch ``bool`` filter (non-scoring)."""
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []
    if filters is not None:
        for key in sorted(filters.terms):
            must.append({"terms": {key: list(filters.terms[key])}})
        for key in sorted(filters.all_terms):
            for value in filters.all_terms[key]:
                must.append({"term": {key: value}})
        for key in sorted(filters.must_not):
            must_not.append({"terms": {key: list(filters.must_not[key])}})
        for key in sorted(filters.ranges):
            lo, hi = filters.ranges[key]
            bounds = {}
            if lo is not None:
                bounds["gte"] = lo
            if hi is not None:
                bounds["lte"] = hi
            must.append({"range": {key: bounds}})
        for key in filters.exists:
            must.append({"exists": {"field": key}})
    return {"bool": {"filter": must, "must_not": must_not}}


def build_bm25_query(query: str, filters: FilterSpec | None, size: int) -> dict[str, Any]:
    clauses = build_filter_clauses(filters)["bool"]
    return {
        "size": size,
        "_source": {"excludes": [VECTOR_FIELD]},
        "query": {
            "bool": {
                "must": [{"match": {TEXT_FIELD: {"query": query}}}],
                "filter": clauses["filter"],
                "must_not": clauses["must_not"],
            }
        },
    }


def build_knn_query(vector: list[float], filters: FilterSpec | None, k: int,
                    num_candidates: int) -> dict[str, Any]:
    """Approximate kNN with the filter applied *during* graph traversal (pre-filter)."""
    return {
        "size": k,
        "_source": {"excludes": [VECTOR_FIELD]},
        "knn": {
            "field": VECTOR_FIELD,
            "query_vector": vector,
            "k": k,
            "num_candidates": max(num_candidates, k),
            "filter": build_filter_clauses(filters),
        },
    }


def raw_similarity(es_score: float, similarity: str) -> float:
    """Invert Elasticsearch's kNN ``_score`` transform back to the raw similarity."""
    if similarity == "cosine":
        return 2.0 * es_score - 1.0
    if similarity == "max_inner_product":
        return es_score - 1.0 if es_score >= 1.0 else 1.0 - 1.0 / es_score
    return es_score
