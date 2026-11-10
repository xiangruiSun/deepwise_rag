"""Fixtures of the document-corpus platform: a small synthetic corpus, ingested and indexed.

Hermetic by default: in-memory SQLite, the in-process search backend, hashing embeddings, no model.

Live mode (same switches as the rest of the suite): with INSPECT_RAG_TEST_DATABASE_URL every
registry is a fresh PostgreSQL database whose schema is created by the Alembic migrations, and
with INSPECT_RAG_TEST_ELASTICSEARCH_URL every runtime indexes into uniquely prefixed
Elasticsearch indexes.  Both are removed at session end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.corpus.config import CorpusConfig, load_corpus_config
from app.corpus.etl import SourceSpec, load_normalized, normalize_source
from app.corpus.generator import write_dataset
from app.corpus.retrieval import AuthContext
from app.corpus.runtime import CorpusRuntime, build_corpus_runtime
from app.db.base import create_all, make_engine, make_session_factory
from app.indexing.backend import LocalSearchBackend, SearchBackend
from tests.conftest import _LIVE_INDEX_PREFIXES, LIVE_DB_URL, LIVE_ES_URL, _live_database_url

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
N_RECORDS, N_GOLDEN = 900, 100


@dataclass
class Corpus:
    runtime: CorpusRuntime
    data_dir: Path
    queries: list[dict[str, Any]]
    info: dict[str, Any]
    settings: Settings

    def auth(self, tenant: str = "tenant-000", *groups: str) -> AuthContext:
        return AuthContext(tenant, "tester", frozenset({"public", *groups}))

    def by_category(self, category: str) -> list[dict[str, Any]]:
        return [q for q in self.queries if q["category"] == category]


def corpus_config(**updates: Any) -> CorpusConfig:
    return load_corpus_config("corpus.yaml", CONFIG_DIR).model_copy(update=updates)


def new_session_factory() -> sessionmaker[Session]:
    if LIVE_DB_URL:
        url = _live_database_url()
        # the schema comes from the migrations, exactly as in a PostgreSQL deployment
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parents[2], env={**os.environ, "INSPECT_RAG_DATABASE_URL": url})
        return make_session_factory(make_engine(url))
    engine = make_engine("sqlite://")
    create_all(engine)
    return make_session_factory(engine)


def build_runtime(tmp: Path, config: CorpusConfig | None = None,
                  session_factory: sessionmaker[Session] | None = None) -> tuple[CorpusRuntime, Settings]:
    settings = Settings(database_url="sqlite://", artifact_dir=tmp / "artifacts", config_dir=CONFIG_DIR,
                        environment="test", search_backend="elasticsearch" if LIVE_ES_URL else "local",
                        elasticsearch_url=LIVE_ES_URL or "http://localhost:9200")
    config = config or corpus_config()
    backend: SearchBackend = LocalSearchBackend()
    if LIVE_ES_URL:
        from app.indexing.es_backend import ElasticsearchBackend

        backend = ElasticsearchBackend(settings)
        if not config.index_prefix.startswith("irt"):  # a runtime that shares another one's indexes keeps its prefix
            prefix = f"irt{uuid.uuid4().hex[:8]}_"
            _LIVE_INDEX_PREFIXES.append(prefix)
            config = config.model_copy(update={"index_prefix": prefix + config.index_prefix})
    runtime = build_corpus_runtime(settings, session_factory or new_session_factory(), backend, config)
    return runtime, settings


def ingest(runtime: CorpusRuntime, raw: Path, normalized: Path) -> dict[str, Any]:
    first = normalize_source(SourceSpec(uri=str(raw)), normalized, runtime.session_factory)
    second = load_normalized(normalized, runtime.session_factory)
    return {"normalize": first.as_dict(), "load": second.as_dict()}


def build_and_activate(runtime: CorpusRuntime) -> str:
    name = runtime.indexes.create_version()
    runtime.indexes.build(name)
    assert runtime.indexes.validate(name)["valid"]
    runtime.indexes.activate(name)
    return name


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    tmp = tmp_path_factory.mktemp("corpus")
    info = write_dataset(tmp / "data", N_RECORDS, seed=13, n_tenants=4, n_golden=N_GOLDEN, rows_per_file=400)
    runtime, settings = build_runtime(tmp)
    ingest(runtime, tmp / "data" / "raw", tmp / "data" / "normalized")
    build_and_activate(runtime)
    queries = [json.loads(line) for line in (tmp / "data" / "eval" / "queries.jsonl").read_text().splitlines()]
    return Corpus(runtime, tmp / "data", queries, info, settings)
