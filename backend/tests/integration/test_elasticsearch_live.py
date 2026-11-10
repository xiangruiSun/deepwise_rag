"""Live Elasticsearch integration.  Skipped unless a node is reachable
(``make up``; override the address with INSPECT_RAG_ELASTICSEARCH_URL)."""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from app.core.config import Settings
from app.indexing.backend import FilterSpec, IndexSpec

pytestmark = pytest.mark.elasticsearch
URL = os.environ.get("INSPECT_RAG_ELASTICSEARCH_URL", "http://localhost:9200")


@pytest.fixture(scope="module")
def es():  # type: ignore[no-untyped-def]
    from app.indexing.es_backend import ElasticsearchBackend

    backend = ElasticsearchBackend(Settings(elasticsearch_url=URL, request_timeout_seconds=10))
    if not backend.ping():
        pytest.skip(f"Elasticsearch is not reachable at {URL}")
    name = f"inspect_rag_test_{uuid.uuid4().hex[:8]}"
    backend.create_index(IndexSpec(name=name, kind="reports", dimension=2))
    backend.bulk_index(name, [
        {"_id": "d1", "case_id": "c1", "patient_id": "p1", "content": "acute pulmonary embolism", "age": 60.0,
         "access_scope": "research", "lab_d_dimer": 3000.0, "embedding": [1.0, 0.0]},
        {"_id": "d2", "case_id": "c2", "patient_id": "p2", "content": "no pulmonary embolism", "age": 40.0,
         "access_scope": "research", "embedding": [0.9, 0.1]},
        {"_id": "d3", "case_id": "c3", "patient_id": "p3", "content": "saddle pulmonary embolism", "age": 70.0,
         "access_scope": "restricted", "lab_d_dimer": 9000.0, "embedding": [0.99, 0.01]},
    ])
    backend.refresh(name)
    yield backend, name
    backend.delete_index(name)


def test_bm25_knn_and_prefilters(es) -> None:  # type: ignore[no-untyped-def]
    backend, name = es
    scope = FilterSpec(terms={"access_scope": ["research"]})
    assert backend.count(name) == 3 and backend.count(name, scope) == 2
    assert "d3" not in [h.id for h in backend.bm25_search(name, "saddle pulmonary embolism", scope, 10)]
    knn = backend.knn_search(name, np.asarray([0.99, 0.01]), scope, 1, 10)
    assert [h.id for h in knn] == ["d1"] and -1.0 <= knn[0].score <= 1.0001
    ranged = backend.knn_search(name, np.asarray([0.0, 1.0]), FilterSpec(ranges={"lab_d_dimer": (2000.0, None),
                                                                             "age": (None, 65.0)}), 5, 10)
    assert [h.id for h in ranged] == ["d1"]
    hits, total = backend.filter_search(name, scope, 10)
    assert total == 2 and [h.id for h in hits] == ["d1", "d2"]
    spec = backend.index_spec(name)
    assert spec.dimension == 2 and spec.similarity == "cosine"
    alias = f"{name}_current"
    backend.put_alias(alias, name)
    assert backend.get_alias(alias) == name and backend.stats(name)["documents"] == 3
    assert backend.delete_documents(name, ["d2"]) == 1
