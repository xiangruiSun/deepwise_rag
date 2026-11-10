"""Application wiring: builds every service from settings + configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import AppConfig, Settings, get_settings, load_app_config
from app.core.errors import DependencyUnavailable
from app.db.base import create_all, make_engine, make_session_factory, session_scope
from app.embeddings.registry import EmbeddingRegistry
from app.imaging.providers import ImageEmbeddingProvider, ImageFeaturePipeline, build_image_provider
from app.indexing.backend import LocalSearchBackend, SearchBackend
from app.indexing.indexer import Indexer
from app.ingestion.pipeline import IngestionPipeline, IngestionReport
from app.rag.generators import build_llm_client
from app.rag.pipeline import ComparisonService, RagPipeline
from app.ranking.rerankers import RerankerRegistry
from app.retrieval.features import CaseFeatureService
from app.retrieval.query_enhancement import QueryEnhancer
from app.retrieval.service import RetrievalService


@dataclass
class AppContext:
    settings: Settings
    config: AppConfig
    engine: Engine
    session_factory: sessionmaker[Session]
    backend: SearchBackend
    embeddings: EmbeddingRegistry
    rerankers: RerankerRegistry
    image_provider: ImageEmbeddingProvider
    retrieval: RetrievalService
    rag: RagPipeline
    comparison: ComparisonService
    dataset_fingerprint: str | None = None

    @property
    def data_kind(self) -> str:
        return "synthetic" if self.config.dataset.adapter == "synthetic" else "inspect"


def build_backend(settings: Settings, config: AppConfig) -> SearchBackend:
    if settings.search_backend == "elasticsearch":
        from app.indexing.es_backend import ElasticsearchBackend

        backend: SearchBackend = ElasticsearchBackend(settings, config.index)
        if not backend.ping():
            raise DependencyUnavailable(
                f"Elasticsearch is not reachable at {settings.elasticsearch_url} (start it with `make up`)"
            )
        return backend
    return LocalSearchBackend(settings.local_index_dir)


def build_context(
    settings: Settings | None = None,
    config: AppConfig | None = None,
    backend: SearchBackend | None = None,
    create_schema: bool = True,
) -> AppContext:
    settings = settings or get_settings()
    config = config or load_app_config(settings.app_config, settings.config_dir)
    engine = make_engine(settings.database_url)
    if create_schema and settings.database_url.startswith("sqlite"):
        create_all(engine)  # PostgreSQL deployments are migrated with Alembic
    session_factory = make_session_factory(engine)
    backend = backend or build_backend(settings, config)
    embeddings = EmbeddingRegistry(config.embeddings, settings.embedding_cache_path)
    rerankers = RerankerRegistry()
    image_provider = build_image_provider(config.image)
    # A local LLM (when configured) also powers the optional query transformations.
    llm_client = build_llm_client(config.generation)
    retrieval = RetrievalService(session_factory, backend, config, embeddings, rerankers, image_provider,
                                 QueryEnhancer(llm_client))
    rag = RagPipeline(session_factory, retrieval, config, llm_client=llm_client)
    comparison = ComparisonService(session_factory, config)
    return AppContext(settings, config, engine, session_factory, backend, embeddings, rerankers,
                      image_provider, retrieval, rag, comparison)


def run_ingestion(ctx: AppContext, force: bool = False, actor: str = "system") -> IngestionReport:
    report = IngestionPipeline(ctx.config.dataset, ctx.session_factory, ctx.settings.artifact_dir, actor).run(force)
    ctx.dataset_fingerprint = report.fingerprint
    ctx.retrieval.clear_caches()
    return report


def build_indexes(
    ctx: AppContext,
    pairs: list[tuple[str | None, str]] | None = None,
    recreate: bool = False,
    with_images: bool = True,
    actor: str = "system",
) -> dict[str, Any]:
    """Fit training-only laboratory statistics, compute image features and (incrementally)
    build the case, report and image indexes."""
    summary: dict[str, Any] = {"lab_statistics": [], "report_indexes": []}
    dataset = ctx.config.dataset
    with session_scope(ctx.session_factory) as session:
        features = CaseFeatureService(session)
        for lookback in dataset.lookback_days:
            for aggregate in ("last", "mean"):
                # fitted on the training corpus split only
                summary["lab_statistics"].append(
                    features.fit_lab_statistics(dataset.corpus_split, lookback, aggregate)
                )
    if with_images and ctx.image_provider.enabled:
        with session_scope(ctx.session_factory) as session:
            summary["image_features"] = ImageFeaturePipeline(session, ctx.image_provider).run()
    indexer = Indexer(ctx.session_factory, ctx.backend, ctx.config, ctx.embeddings, actor,
                      ctx.dataset_fingerprint)
    summary["cases_index"] = indexer.sync_cases_index()
    for embedding_key, chunking_key in pairs if pairs is not None else indexer.required_report_indexes():
        summary["report_indexes"].append(indexer.sync_report_index(embedding_key, chunking_key, recreate))
    if with_images:
        summary["image_index"] = indexer.sync_image_index(ctx.image_provider)
    ctx.retrieval.clear_caches()
    return summary
