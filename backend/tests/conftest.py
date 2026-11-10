"""Shared fixtures: a fully ingested and indexed synthetic cohort (hermetic, CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig, Settings, load_app_config
from app.core.container import AppContext, build_context, build_indexes, run_ingestion
from app.core.security import AccessPolicy, Principal, Role
from app.indexing.backend import LocalSearchBackend
from app.ingestion.synthetic_generator import generate_synthetic_dataset

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


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
    settings = Settings(database_url="sqlite://", artifact_dir=artifact_dir, config_dir=CONFIG_DIR,
                        environment="test", dev_api_key="test-admin-key",
                        api_keys_json=json.dumps({
                            "researcher-key": {"user": "rita", "role": "researcher", "scopes": ["research"]},
                            "restricted-key": {"user": "omar", "role": "researcher",
                                               "scopes": ["research", "restricted"]},
                            "noscope-key": {"user": "nina", "role": "researcher", "scopes": []},
                            "annotator-key": {"user": "ann", "role": "annotator", "scopes": ["research"]},
                        }))
    ctx = build_context(settings, config or make_config(root), backend=LocalSearchBackend())
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
