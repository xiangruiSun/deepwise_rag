"""Fixtures of the document-corpus platform: a small synthetic corpus, ingested and indexed.

Hermetic: in-memory SQLite, the in-process search backend, hashing embeddings, no model."""

from __future__ import annotations

import json
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
from app.indexing.backend import LocalSearchBackend

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
    engine = make_engine("sqlite://")
    create_all(engine)
    return make_session_factory(engine)


def build_runtime(tmp: Path, config: CorpusConfig | None = None,
                  session_factory: sessionmaker[Session] | None = None) -> tuple[CorpusRuntime, Settings]:
    settings = Settings(database_url="sqlite://", artifact_dir=tmp / "artifacts", config_dir=CONFIG_DIR,
                        environment="test")
    runtime = build_corpus_runtime(settings, session_factory or new_session_factory(), LocalSearchBackend(),
                                   config or corpus_config())
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
