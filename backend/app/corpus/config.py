"""Configuration of the document corpus platform (``configs/corpus.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from app.core.config import EmbeddingConfig, RerankerConfig, StrictModel


class CorpusChunkingConfig(StrictModel):
    strategy: Literal["fixed", "sliding_window", "sentence", "structure", "recursive", "semantic",
                      "parent_child"] = "sliding_window"
    max_tokens: int = Field(default=200, ge=8)
    overlap: int = Field(default=30, ge=0)
    # semantic chunking: start a new chunk where adjacent-sentence similarity drops below this
    semantic_breakpoint: float = Field(default=0.35, ge=-1.0, le=1.0)
    version: str = "c1"

    @property
    def identity(self) -> str:
        """Everything that changes the produced chunks; part of the index version."""
        extra = f"-b{self.semantic_breakpoint}" if self.strategy == "semantic" else ""
        return f"{self.strategy}-{self.max_tokens}-{self.overlap}{extra}-{self.version}"

    @model_validator(mode="after")
    def _overlap_smaller(self) -> CorpusChunkingConfig:
        if self.strategy == "sliding_window" and self.overlap >= self.max_tokens:
            raise ValueError("chunk overlap must be smaller than max_tokens")
        return self


class RetrievalParams(StrictModel):
    """One retrieval strategy.  Every stage is independently switchable for ablations."""

    strategy: Literal["bm25", "dense", "hybrid"] = "hybrid"
    bm25_top_k: int = Field(default=100, ge=1, le=1000)
    dense_top_k: int = Field(default=100, ge=1, le=1000)
    num_candidates: int = Field(default=200, ge=1, le=10000)
    rrf_k: int = Field(default=60, ge=1)
    rrf_weights: dict[str, float] = Field(default_factory=dict)
    fusion_window: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False
    rerank_depth: int = Field(default=20, ge=1, le=200)
    final_k: int = Field(default=8, ge=1, le=100)
    query_routing: bool = False
    query_rewriting: bool = False
    max_rewrites: int = Field(default=2, ge=0, le=5)
    adaptive_depth: bool = False
    freshness: bool = True
    # per-strategy override of context.selection, so context packing can be ablated like retrieval
    context_selection: Literal["top_k", "diverse"] | None = None


class ContextParams(StrictModel):
    selection: Literal["top_k", "diverse"] = "diverse"
    token_budget: int = Field(default=1500, ge=50)
    max_chunks: int = Field(default=8, ge=1, le=50)
    # two chunks whose token-set Jaccard similarity is at least this are near-duplicates
    duplicate_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    per_document_limit: int = Field(default=3, ge=1)
    # drop candidates scoring below this fraction of the best candidate's score
    min_relative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_expansion: bool = False
    # withhold sentences of retrieved text that address the model ("ignore previous instructions ...")
    quarantine_instructions: bool = True
    # flag pairs of context sentences from different documents that contradict each other
    detect_conflicts: bool = True


class GenerationParams(StrictModel):
    # "extractive": deterministic, model-free.  "openai_compatible": vLLM or any server with the
    # same API.  "mock": canned token stream for local demos — always labelled as mock output.
    backend: Literal["extractive", "openai_compatible", "mock"] = "extractive"
    model: str = "extractive-v1"
    base_url: str | None = None
    api_key_env: str = "INSPECT_RAG_LLM_API_KEY"
    prompt_version: str = "corpus-p1"
    max_output_tokens: int = Field(default=400, ge=16, le=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    queue_limit: int = Field(default=32, ge=0)
    max_retries: int = Field(default=1, ge=0, le=5)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)
    verify_citations: bool = True

    @model_validator(mode="after")
    def _needs_url(self) -> GenerationParams:
        if self.backend == "openai_compatible" and not self.base_url:
            raise ValueError("generation.backend=openai_compatible requires generation.base_url")
        return self


class CacheParams(StrictModel):
    backend: Literal["none", "memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    query_embedding_ttl_seconds: int = Field(default=3600, ge=1)
    retrieval_ttl_seconds: int = Field(default=120, ge=1)
    answer_ttl_seconds: int = Field(default=0, ge=0)  # 0 = never cache generated answers
    max_memory_items: int = Field(default=10_000, ge=1)


class CorpusConfig(StrictModel):
    name: str = "corpus"
    index_prefix: str = "corpus_chunks"
    mapping_version: str = "m1"
    embedding: EmbeddingConfig
    chunking: CorpusChunkingConfig = Field(default_factory=CorpusChunkingConfig)
    retrieval: RetrievalParams = Field(default_factory=RetrievalParams)
    # named alternative strategies selectable per request and compared in ablations
    strategies: dict[str, RetrievalParams] = Field(default_factory=dict)
    context: ContextParams = Field(default_factory=ContextParams)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheParams = Field(default_factory=CacheParams)
    # int8 quantisation of stored vectors in Elasticsearch (about 4x less vector memory)
    vector_quantization: Literal["none", "int8"] = "none"
    index_batch_size: int = Field(default=256, ge=1, le=10_000)
    # a deleted or updated document must stop being served within this many seconds
    consistency_interval_seconds: float = Field(default=120.0, gt=0)
    dead_letter_limit: int = Field(default=10_000, ge=0)
    # YAML mapping of abbreviation -> full form used by query rewriting (relative to config_dir)
    acronyms_file: str | None = None

    @property
    def alias(self) -> str:
        return f"{self.index_prefix}_current"

    def strategy(self, name: str | None) -> RetrievalParams:
        if not name or name == "default":
            return self.retrieval
        if name not in self.strategies:
            raise KeyError(name)
        return self.strategies[name]


def load_corpus_config(name: str | Path, config_dir: Path) -> CorpusConfig:
    path = Path(name)
    if not path.is_absolute():
        path = config_dir / path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CorpusConfig.model_validate(data)
