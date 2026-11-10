"""The Haystack pipeline must reproduce the native retrieval service's ranking."""

from __future__ import annotations

import pytest

from app.core.container import AppContext
from app.core.security import AccessPolicy
from app.indexing.backend import FilterSpec
from app.schemas.search import SearchRequest

pytest.importorskip("haystack")


def test_haystack_pipeline_matches_native_hybrid_retrieval(ctx: AppContext, policy: AccessPolicy) -> None:
    from app.retrieval.haystack_components import build_hybrid_pipeline
    from app.retrieval.retrievers import BM25Retriever, DenseRetriever

    recipe = ctx.config.recipes["hybrid_rrf_hashing"]
    pipeline = build_hybrid_pipeline(
        BM25Retriever(ctx.backend, "report_chunks_v1_lexical"),
        DenseRetriever(ctx.backend, "report_chunks_v1_hashing", ctx.embeddings.embedder("hashing")),
        recipe.fusion, recipe.aggregation, recipe.bm25_top_k, recipe.dense_top_k,
        recipe.dense_num_candidates, recipe.fused_top_k,
    )
    query = "segmental filling defects without right heart strain"
    filters = FilterSpec(terms={"access_scope": policy.allowed_scopes})
    result = pipeline.run({"bm25": {"query": query, "filters": filters}, "dense": {"query": query, "filters": filters}})
    cases = result["aggregation"]["cases"]
    native = ctx.retrieval.search(SearchRequest(query_mode="text", query_text=query, top_k=10,
                                                retrieval_recipe_id="hybrid_rrf_hashing"), policy)
    assert [c["case_id"] for c in cases[:10]] == [r.case_id for r in native.results]
    assert cases[0]["score"] == pytest.approx(native.results[0].final_score)
    assert len({c["case_id"] for c in cases}) == len(cases)
