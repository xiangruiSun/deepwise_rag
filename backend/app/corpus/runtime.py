"""Wiring of the corpus platform from settings and ``configs/corpus.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.corpus.batch import BatchJobManager
from app.corpus.cache import CorpusCache, build_cache_backend
from app.corpus.config import CorpusConfig, load_corpus_config
from app.corpus.evaluation import ExperimentTracker
from app.corpus.generation import build_llm
from app.corpus.indexing import IndexManager
from app.corpus.retrieval import HybridRetriever
from app.corpus.service import CorpusService
from app.embeddings.registry import EmbeddingRegistry
from app.indexing.backend import SearchBackend
from app.ranking.rerankers import RerankerRegistry


@dataclass
class CorpusRuntime:
    config: CorpusConfig
    session_factory: sessionmaker[Session]
    backend: SearchBackend
    embeddings: EmbeddingRegistry
    indexes: IndexManager
    retriever: HybridRetriever
    cache: CorpusCache
    service: CorpusService
    batch: BatchJobManager
    tracker: ExperimentTracker
    eval_dataset_dir: Path


def load_acronyms(config: CorpusConfig, config_dir: Path) -> dict[str, str]:
    if not config.acronyms_file:
        return {}
    path = Path(config.acronyms_file)
    path = path if path.is_absolute() else config_dir / path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


def build_corpus_runtime(settings: Settings, session_factory: sessionmaker[Session], backend: SearchBackend,
                         config: CorpusConfig | None = None, rerankers: RerankerRegistry | None = None,
                         cache: CorpusCache | None = None) -> CorpusRuntime:
    config = config or load_corpus_config(settings.corpus_config, settings.config_dir)
    embeddings = EmbeddingRegistry({config.embedding.key: config.embedding}, settings.embedding_cache_path)
    embedder = embeddings.embedder(config.embedding.key)
    indexes = IndexManager(config, session_factory, backend, embedder)
    reranker = (rerankers or RerankerRegistry()).get(config.reranker)
    retriever = HybridRetriever(config, backend, embedder, reranker, load_acronyms(config, settings.config_dir))
    cache = cache or CorpusCache(build_cache_backend(config.cache), config.cache)
    shared_cache = cache

    def invalidate(tenants: set[str]) -> None:
        for tenant in sorted(tenants):
            shared_cache.invalidate_tenant(tenant)

    indexes.on_change = invalidate
    service = CorpusService(config, indexes, retriever, cache, build_llm(config.generation))
    work = settings.artifact_dir / "corpus"
    batch = BatchJobManager(service, session_factory, work / "batch")
    tracker = ExperimentTracker(service, work / "evaluations")
    return CorpusRuntime(config, session_factory, backend, embeddings, indexes, retriever, cache, service,
                         batch, tracker, work / "eval_datasets")


def standalone_runtime(settings: Settings, config_name: str | None = None) -> CorpusRuntime:
    """Runtime for CLI tools and workers: database, search backend and services from settings."""
    from app.core.config import IndexConfig
    from app.db.base import create_all, make_engine, make_session_factory
    from app.indexing.backend import LocalSearchBackend

    engine = make_engine(settings.database_url)
    if settings.database_url.startswith("sqlite"):
        create_all(engine)  # PostgreSQL deployments are migrated with Alembic
    backend: SearchBackend
    if settings.search_backend == "elasticsearch":
        from app.indexing.es_backend import ElasticsearchBackend

        backend = ElasticsearchBackend(settings, IndexConfig())
    else:
        backend = LocalSearchBackend(settings.local_index_dir)
    config = load_corpus_config(config_name or settings.corpus_config, settings.config_dir)
    return build_corpus_runtime(settings, make_session_factory(engine), backend, config)
