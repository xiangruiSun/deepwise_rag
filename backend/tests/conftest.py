"""Shared fixtures: a fully ingested and indexed synthetic cohort (hermetic, CPU-only)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig, Settings, load_app_config
from app.core.container import AppContext, build_context, build_indexes, run_ingestion
from app.core.security import AccessPolicy, Principal, Role
from app.indexing.backend import LocalSearchBackend
from app.indexing.indexer import report_index_name
from app.ingestion.synthetic_generator import generate_synthetic_dataset

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Optional live mode.  By default the suite is hermetic (in-process search backend, SQLite).
#   INSPECT_RAG_TEST_ELASTICSEARCH_URL=http://127.0.0.1:9200   run every context on Elasticsearch
#   INSPECT_RAG_TEST_DATABASE_URL=postgresql+psycopg://...      run every context on PostgreSQL
# Each context gets uniquely prefixed indexes and its own database, removed at session end.
LIVE_ES_URL = os.environ.get("INSPECT_RAG_TEST_ELASTICSEARCH_URL")
LIVE_DB_URL = os.environ.get("INSPECT_RAG_TEST_DATABASE_URL")
_LIVE_INDEX_PREFIXES: list[str] = []
_LIVE_DATABASES: list[str] = []


def cases_index(ctx: AppContext) -> str:
    return ctx.config.index.cases_index


def image_index(ctx: AppContext) -> str:
    return ctx.config.index.image_index


def report_index(ctx: AppContext, embedding: str | None, chunking: str = "whole_impression") -> str:
    return report_index_name(ctx.config.index.report_index_prefix, embedding, chunking)


def _live_database_url() -> str:
    """Create an empty database for one context on the configured PostgreSQL server."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    assert LIVE_DB_URL is not None
    name = f"inspect_rag_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(LIVE_DB_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    _LIVE_DATABASES.append(name)
    return make_url(LIVE_DB_URL).set(database=name).render_as_string(hide_password=False)


@pytest.fixture(scope="session", autouse=True)
def _live_cleanup():  # type: ignore[no-untyped-def]
    yield
    if LIVE_ES_URL and _LIVE_INDEX_PREFIXES:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(LIVE_ES_URL, request_timeout=30)
        for prefix in _LIVE_INDEX_PREFIXES:
            # wildcard deletes are disabled by default; resolve the names first
            for name in list(client.indices.get(index=f"{prefix}*")):
                client.indices.delete(index=name, ignore_unavailable=True)
    if LIVE_DB_URL and _LIVE_DATABASES:
        from sqlalchemy import create_engine, text

        admin = create_engine(LIVE_DB_URL, isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            for name in _LIVE_DATABASES:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic") / "data"
    generate_synthetic_dataset(root)
    return root


def make_config(root: Path) -> AppConfig:
    config = load_app_config("synthetic.yaml", CONFIG_DIR)
    evaluation = config.evaluation.model_copy(update={
        "query_set": str(root / "eval" / "queries.jsonl"),
        "qrels": str(root / "eval" / "qrels.jsonl"),
        "generation_set": str(root / "eval" / "generation_eval.jsonl"),
        "bootstrap_samples": 200,
    })
    return config.model_copy(update={
        "dataset": config.dataset.model_copy(update={"root": str(root)}),
        "evaluation": evaluation,
    })


def make_context(root: Path, artifact_dir: Path, config: AppConfig | None = None) -> AppContext:
    config = config or make_config(root)
    database_url = _live_database_url() if LIVE_DB_URL else "sqlite://"
    if LIVE_ES_URL:
        prefix = f"irt{uuid.uuid4().hex[:8]}_"
        _LIVE_INDEX_PREFIXES.append(prefix)
        config = config.model_copy(update={"index": config.index.model_copy(update={
            "cases_index": prefix + config.index.cases_index,
            "report_index_prefix": prefix + config.index.report_index_prefix,
            "image_index": prefix + config.index.image_index,
            "alias_prefix": prefix + config.index.alias_prefix,
        })})
    settings = Settings(database_url=database_url, artifact_dir=artifact_dir, config_dir=CONFIG_DIR,
                        search_backend="elasticsearch" if LIVE_ES_URL else "local",
                        elasticsearch_url=LIVE_ES_URL or "http://localhost:9200",
                        environment="test", dev_api_key="test-admin-key",
                        api_keys_json=json.dumps({
                            "researcher-key": {"user": "rita", "role": "researcher", "scopes": ["research"]},
                            "restricted-key": {"user": "omar", "role": "researcher",
                                               "scopes": ["research", "restricted"]},
                            "noscope-key": {"user": "nina", "role": "researcher", "scopes": []},
                            "annotator-key": {"user": "ann", "role": "annotator", "scopes": ["research"]},
                        }))
    backend: Any = LocalSearchBackend()
    if LIVE_ES_URL:
        from app.indexing.es_backend import ElasticsearchBackend

        backend = ElasticsearchBackend(settings, config.index)
    ctx = build_context(settings, config, backend=backend)
    if LIVE_DB_URL:
        from app.db.base import create_all

        create_all(ctx.engine)
    run_ingestion(ctx)
    build_indexes(ctx)
    return ctx


@pytest.fixture(scope="session")
def ctx(synthetic_root: Path, tmp_path_factory: pytest.TempPathFactory) -> AppContext:
    return make_context(synthetic_root, tmp_path_factory.mktemp("artifacts"))


@pytest.fixture(scope="session")
def policy() -> AccessPolicy:
    return AccessPolicy(Principal("tester", Role.RESEARCHER, frozenset({"research"})))


@pytest.fixture(scope="session")
def restricted_policy() -> AccessPolicy:
    return AccessPolicy(Principal("privileged", Role.RESEARCHER, frozenset({"research", "restricted"})))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="session")
def queries(synthetic_root: Path) -> list[dict[str, Any]]:
    return read_jsonl(synthetic_root / "eval" / "queries.jsonl")


@pytest.fixture(scope="session")
def truth(synthetic_root: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(synthetic_root / "_truth" / "latent_profiles.jsonl")}


@pytest.fixture(scope="session")
def case_query(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return next(q for q in queries if q["query_mode"] == "case" and q["stage"] == "dev")
