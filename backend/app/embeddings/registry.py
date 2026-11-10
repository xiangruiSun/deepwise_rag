"""Embedding registry: builds providers from configuration, lazily and once."""

from __future__ import annotations

from pathlib import Path

from app.core.config import EmbeddingConfig
from app.core.errors import ConfigurationError
from app.embeddings.base import CachedEmbedder, EmbeddingCache, EmbeddingProvider
from app.embeddings.providers import PROVIDERS


class EmbeddingRegistry:
    def __init__(self, configs: dict[str, EmbeddingConfig], cache_path: Path | None = None) -> None:
        self._configs = configs
        self._providers: dict[str, EmbeddingProvider] = {}
        self.cache = EmbeddingCache(cache_path)
        # set to False while benchmarking: query vectors are then always computed by the model
        self.cache_queries = True

    def keys(self) -> list[str]:
        return sorted(self._configs)

    def config(self, key: str) -> EmbeddingConfig:
        try:
            return self._configs[key]
        except KeyError as exc:
            raise ConfigurationError(f"unknown embedding configuration '{key}'") from exc

    def provider(self, key: str) -> EmbeddingProvider:
        if key not in self._providers:
            cfg = self.config(key)
            self._providers[key] = PROVIDERS[cfg.provider](cfg)
        return self._providers[key]

    def embedder(self, key: str) -> CachedEmbedder:
        return CachedEmbedder(self.provider(key), self.cache, cache_queries=self.cache_queries)

    def is_loaded(self, key: str) -> bool:
        return key in self._providers
