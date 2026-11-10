"""Environment settings and YAML configuration loading.

Two layers exist on purpose:

* ``Settings`` — process/environment level (connection strings, secrets, paths).
* YAML configuration models — experiment level (dataset, embedding, index,
  retrieval, multimodal, generation, evaluation).  Nothing about the selected
  embedding model, retrieval algorithm, fusion weights, reranker or LLM is
  hardcoded in the application code; it always comes from one of these models.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.hashing import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Environment settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSPECT_RAG_", env_file=".env", extra="ignore", case_sensitive=False
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = f"sqlite:///{REPO_ROOT / 'artifacts' / 'inspect_rag.db'}"
    search_backend: Literal["local", "elasticsearch"] = "local"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str | None = None
    elasticsearch_username: str | None = None
    elasticsearch_password: str | None = None
    artifact_dir: Path = REPO_ROOT / "artifacts"
    config_dir: Path = REPO_ROOT / "configs"
    app_config: str = "synthetic.yaml"
    # JSON object: {"<api-key>": {"user": "...", "role": "...", "scopes": [...]}}
    api_keys_json: str | None = None
    # Only honoured when environment != production (documented in docs/deployment.md).
    dev_api_key: str | None = "dev-research-key"
    request_timeout_seconds: float = 60.0
    log_level: str = "INFO"
    log_clinical_text: bool = False
    otel_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    cors_origins: str = "http://localhost:5173"

    @property
    def embedding_cache_path(self) -> Path:
        return self.artifact_dir / "embedding_cache.sqlite"

    @property
    def local_index_dir(self) -> Path:
        return self.artifact_dir / "local_index"

    @property
    def experiment_dir(self) -> Path:
        return self.artifact_dir / "experiments"

    def api_keys(self) -> dict[str, dict[str, Any]]:
        if not self.api_keys_json:
            return {}
        parsed = json.loads(self.api_keys_json)
        if not isinstance(parsed, dict):
            raise ValueError("INSPECT_RAG_API_KEYS_JSON must be a JSON object")
        return parsed


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# YAML configuration models
# --------------------------------------------------------------------------- #
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    name: str
    adapter: Literal["synthetic", "inspect_legacy", "inspect_meds"]
    release: str | None = None
    root: str
    # adapter specific: explicit file locations and column mappings (never guessed)
    files: dict[str, str] = Field(default_factory=dict)
    columns: dict[str, dict[str, str]] = Field(default_factory=dict)
    code_prefixes: dict[str, str] = Field(default_factory=dict)
    default_access_scope: str = "research"
    batch_size: int = Field(default=5000, ge=1, le=1_000_000)
    lookback_days: list[int] = Field(default_factory=lambda: [7, 30, 90])
    default_lookback_days: int = 30
    corpus_split: str = "train"
    dev_query_split: str = "valid"
    test_query_split: str = "test"

    @model_validator(mode="after")
    def _lookback_default_listed(self) -> DatasetConfig:
        if self.default_lookback_days not in self.lookback_days:
            raise ValueError("default_lookback_days must be one of lookback_days")
        return self


class EmbeddingConfig(StrictModel):
    key: str
    provider: Literal["hashing", "sentence_transformers", "medcpt"]
    model_id: str
    revision: str | None = None
    query_model_id: str | None = None  # MedCPT uses separate query/article encoders
    query_revision: str | None = None
    dimension: int = Field(gt=0)
    device: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    max_document_tokens: int = Field(default=256, ge=8)
    max_query_tokens: int = Field(default=256, ge=8)
    pooling: Literal["mean", "cls", "hashing"] = "mean"
    normalize: bool = True
    similarity: Literal["cosine", "max_inner_product"] = "cosine"
    query_prefix: str = ""
    document_prefix: str = ""
    preprocessing_version: str = "text-v1"


class ChunkingConfig(StrictModel):
    key: str = "whole_impression"
    strategy: Literal[
        "whole_impression", "fixed", "sliding_window", "sentence", "structure", "parent_child"
    ] = "whole_impression"
    max_tokens: int = Field(default=256, ge=8)
    overlap: int = Field(default=0, ge=0)
    version: str = "chunk-v1"

    @model_validator(mode="after")
    def _overlap_smaller(self) -> ChunkingConfig:
        if self.overlap >= self.max_tokens:
            raise ValueError("chunk overlap must be smaller than max_tokens")
        return self


class IndexConfig(StrictModel):
    mapping_version: str = "v1"
    cases_index: str = "cases_v1"
    report_index_prefix: str = "report_chunks_v1"
    image_index: str = "image_features_v1"
    alias_prefix: str = "inspect_rag"
    knn_num_candidates: int = Field(default=200, ge=1)
    refresh: Literal["wait_for", "true", "false"] = "wait_for"
    shards: int = 1
    replicas: int = 0


class FusionConfig(StrictModel):
    method: Literal["rrf", "weighted_score"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    weights: dict[str, float] = Field(default_factory=dict)
    normalization: Literal["min_max", "z_score"] = "min_max"


class RerankerConfig(StrictModel):
    enabled: bool = False
    key: str = "none"
    provider: Literal["none", "cross_encoder", "medcpt_cross_encoder", "lexical_overlap"] = "none"
    model_id: str | None = None
    revision: str | None = None
    depth: int = Field(default=40, ge=1, le=500)
    batch_size: int = 16
    max_tokens: int = 512
    device: str = "auto"


class AggregationConfig(StrictModel):
    method: Literal["best_score", "best_rank", "top_n_mean", "case_rrf"] = "best_score"
    top_n: int = Field(default=3, ge=1)
    rrf_k: int = 60


class QueryEnhancementConfig(StrictModel):
    method: Literal["none", "rule_expansion", "llm_rewrite", "hyde"] = "none"
    max_expansion_terms: int = 8


class LaboratoryConfig(StrictModel):
    lookback_days: int = 30
    aggregate: Literal["last", "mean"] = "last"
    min_overlap: int = Field(default=2, ge=1)
    distance_scale: float = Field(default=1.0, gt=0)
    concept_weights: dict[str, float] = Field(default_factory=dict)


class ImageConfig(StrictModel):
    provider: Literal["disabled", "intensity_histogram", "precomputed", "inspect_pretrained"] = (
        "disabled"
    )
    feature_path: str | None = None
    checkpoint_path: str | None = None
    dimension: int = 64
    version: str = "img-v1"
    target_spacing_mm: tuple[float, float, float] = (4.0, 4.0, 4.0)
    hu_window: tuple[float, float] = (-1000.0, 900.0)


Modality = Literal["text", "laboratory", "image", "diagnosis"]
Branch = Literal["bm25", "dense"]


class MultimodalConfig(StrictModel):
    enabled: bool = False
    modalities: list[Modality] = Field(default_factory=lambda: list[Modality](["text"]))
    candidate_sources: list[Modality] = Field(default_factory=lambda: list[Modality](["text"]))
    family: Literal["rank", "score"] = "rank"
    rrf_k: int = 60
    weights: dict[str, float] = Field(
        default_factory=lambda: {"text": 0.45, "laboratory": 0.25, "image": 0.20, "diagnosis": 0.10}
    )
    missing_policy: Literal["renormalize", "zero", "exclude"] = "renormalize"
    candidate_depth: int = Field(default=100, ge=1)
    laboratory: LaboratoryConfig = Field(default_factory=LaboratoryConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)

    @field_validator("weights")
    @classmethod
    def _non_negative(cls, v: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in v.values()):
            raise ValueError("fusion weights must be non-negative")
        return v


class RetrievalRecipe(StrictModel):
    """A complete, selectable retrieval recipe (``retrieval_recipe_id``)."""

    recipe_id: str
    description: str = ""
    branches: list[Branch] = Field(default_factory=lambda: list[Branch](["bm25"]))
    embedding: str | None = None  # key into the embeddings registry
    chunking: str = "whole_impression"
    # How a query CASE's impression is embedded for dense retrieval: with the document
    # encoder (symmetric case-to-case similarity) or with the query encoder.  An evaluated
    # hyper-parameter, not an assumption.
    case_query_encoding: Literal["document", "query"] = "document"
    bm25_top_k: int = Field(default=50, ge=1, le=1000)
    dense_top_k: int = Field(default=50, ge=1, le=1000)
    dense_num_candidates: int = Field(default=200, ge=1, le=10000)
    fused_top_k: int = Field(default=100, ge=1, le=2000)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    query_enhancement: QueryEnhancementConfig = Field(default_factory=QueryEnhancementConfig)

    @model_validator(mode="after")
    def _dense_needs_embedding(self) -> RetrievalRecipe:
        if "dense" in self.branches and not self.embedding:
            raise ValueError(f"recipe {self.recipe_id}: dense branch requires an embedding key")
        if len(self.branches) != len(set(self.branches)):
            raise ValueError("duplicate retrieval branches")
        if not self.branches and "text" in self.multimodal.candidate_sources:
            raise ValueError("text candidate source requires at least one text branch")
        return self


class ExternalApiConfig(StrictModel):
    enabled: bool = False
    # Must be set to true by a human who has verified that the data and environment
    # are authorised for the external provider.  Disabled by default.
    i_confirm_data_is_authorized_for_this_provider: bool = False
    base_url: str | None = None
    api_key_env: str = "INSPECT_RAG_EXTERNAL_LLM_API_KEY"


class GenerationConfig(StrictModel):
    key: str = "extractive"
    backend: Literal["extractive", "ollama", "openai_compatible", "huggingface", "external"] = (
        "extractive"
    )
    model: str = "extractive-evidence-v1"
    base_url: str | None = None
    prompt_variant: Literal["basic", "structured", "structured_citations"] = (
        "structured_citations"
    )
    post_verification: bool = True
    max_cases: int = Field(default=5, ge=1, le=20)
    max_evidence_per_case: int = Field(default=8, ge=1, le=50)
    context_token_budget: int = Field(default=3000, ge=200)
    max_output_tokens: int = Field(default=800, ge=16)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = 120.0
    require_citations: bool = True
    external_api: ExternalApiConfig = Field(default_factory=ExternalApiConfig)

    @model_validator(mode="after")
    def _external_gate(self) -> GenerationConfig:
        if self.backend == "external":
            ext = self.external_api
            if not (ext.enabled and ext.i_confirm_data_is_authorized_for_this_provider):
                raise ValueError(
                    "external generation backend requires external_api.enabled=true and "
                    "external_api.i_confirm_data_is_authorized_for_this_provider=true"
                )
            if not ext.base_url:
                raise ValueError("external generation backend requires external_api.base_url")
        return self


class EvaluationConfig(StrictModel):
    query_set: str | None = None
    qrels: str | None = None
    generation_set: str | None = None
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    relevance_threshold: int = Field(default=2, ge=1, le=3)
    unjudged_policy: Literal["non_relevant", "condensed"] = "non_relevant"
    label_sources: list[str] = Field(default_factory=lambda: ["clinician"])
    seed: int = 13
    bootstrap_samples: int = Field(default=2000, ge=100)
    warmup_queries: int = Field(default=2, ge=0)
    concurrency: int = Field(default=1, ge=1)
    top_k: int = Field(default=10, ge=1)


class AppConfig(StrictModel):
    """Top-level application configuration (e.g. ``configs/synthetic.yaml``)."""

    dataset: DatasetConfig
    embeddings: dict[str, EmbeddingConfig]
    chunking: dict[str, ChunkingConfig] = Field(
        default_factory=lambda: {"whole_impression": ChunkingConfig()}
    )
    index: IndexConfig = Field(default_factory=IndexConfig)
    recipes: dict[str, RetrievalRecipe]
    default_recipe: str
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    index_embeddings: list[str] = Field(default_factory=list)
    index_chunkings: list[str] = Field(default_factory=lambda: ["whole_impression"])
    image: ImageConfig = Field(default_factory=ImageConfig)

    @model_validator(mode="after")
    def _cross_refs(self) -> AppConfig:
        if self.default_recipe not in self.recipes:
            raise ValueError(f"default_recipe '{self.default_recipe}' is not defined in recipes")
        for key, emb in self.embeddings.items():
            if emb.key != key:
                raise ValueError(f"embedding key mismatch: '{key}' vs '{emb.key}'")
        for key, ch in self.chunking.items():
            if ch.key != key:
                raise ValueError(f"chunking key mismatch: '{key}' vs '{ch.key}'")
        for rid, recipe in self.recipes.items():
            if recipe.recipe_id != rid:
                raise ValueError(f"recipe id mismatch: '{rid}' vs '{recipe.recipe_id}'")
            if recipe.embedding and recipe.embedding not in self.embeddings:
                raise ValueError(f"recipe {rid}: unknown embedding '{recipe.embedding}'")
            if recipe.chunking not in self.chunking:
                raise ValueError(f"recipe {rid}: unknown chunking '{recipe.chunking}'")
        for key in self.index_embeddings:
            if key not in self.embeddings:
                raise ValueError(f"index_embeddings: unknown embedding '{key}'")
        for key in self.index_chunkings:
            if key not in self.chunking:
                raise ValueError(f"index_chunkings: unknown chunking '{key}'")
        return self

    def config_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class ExperimentRunSpec(StrictModel):
    experiment_id: str
    recipe: str
    description: str = ""
    generation: str | None = None  # key into ExperimentSuite.generation_variants
    tags: dict[str, str] = Field(default_factory=dict)


class ExperimentSuite(StrictModel):
    suite_id: str
    description: str = ""
    base_config: str
    stage: Literal["validation", "test"] = "validation"
    baseline_experiment: str | None = None
    recipes: dict[str, RetrievalRecipe] = Field(default_factory=dict)
    generation_variants: dict[str, GenerationConfig] = Field(default_factory=dict)
    runs: list[ExperimentRunSpec]
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> ExperimentSuite:
        ids = [r.experiment_id for r in self.runs]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment ids within a suite must be unique")
        if self.baseline_experiment and self.baseline_experiment not in ids:
            raise ValueError("baseline_experiment must be one of the suite's runs")
        for run in self.runs:
            if run.generation and run.generation not in self.generation_variants:
                raise ValueError(f"run {run.experiment_id}: unknown generation variant")
        return self


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config_path(name: str | Path, config_dir: Path | None = None) -> Path:
    path = Path(name)
    if path.is_absolute() or path.exists():
        return path
    return (config_dir or get_settings().config_dir) / path


def load_app_config(name: str | Path, config_dir: Path | None = None) -> AppConfig:
    """Load and validate an application config.  ``extends: other.yaml`` is supported."""
    path = resolve_config_path(name, config_dir)
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent:
        base = load_app_config(path.parent / parent, config_dir).model_dump(mode="json")
        data = _deep_merge(base, data)
    return AppConfig.model_validate(data)


def load_experiment_suite(name: str | Path, config_dir: Path | None = None) -> ExperimentSuite:
    path = resolve_config_path(name, config_dir)
    return ExperimentSuite.model_validate(_read_yaml(path))


def resolve_suite(
    suite: ExperimentSuite, config_dir: Path | None = None
) -> tuple[AppConfig, EvaluationConfig]:
    """Merge suite-local recipes into the base app config and validate references."""
    base = load_app_config(suite.base_config, config_dir)
    merged = base.model_dump(mode="json")
    for rid, recipe in suite.recipes.items():
        merged["recipes"][rid] = recipe.model_dump(mode="json")
    app_config = AppConfig.model_validate(merged)
    for run in suite.runs:
        if run.recipe not in app_config.recipes:
            raise ValueError(f"run {run.experiment_id}: unknown recipe '{run.recipe}'")
    return app_config, suite.evaluation or app_config.evaluation
