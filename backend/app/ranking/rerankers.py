"""Text rerankers (spec section 16).

A text cross-encoder reranks TEXT evidence only.  It never sees CT volumes or
typed laboratory values; those enter through ``MultimodalCaseRanker`` afterwards.
"""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from app.core.config import RerankerConfig
from app.core.errors import ConfigurationError, DependencyUnavailable
from app.embeddings.base import (
    assert_finite,
    materialize_parameters,
    resolve_commit_hash,
    resolve_device,
)


class Reranker(ABC):
    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document (higher is better; not a probability)."""

    def metadata(self) -> dict[str, Any]:
        return {"key": self.config.key, "provider": self.config.provider,
                "model_id": self.config.model_id, "revision": self.resolved_revision(),
                "depth": self.config.depth}

    def resolved_revision(self) -> str | None:
        return self.config.revision


class CrossEncoderReranker(Reranker):
    """SentenceTransformers ``CrossEncoder`` (e.g. cross-encoder/ms-marco-MiniLM-L6-v2)."""

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__(config)
        if not config.model_id:
            raise ConfigurationError("cross_encoder reranker requires model_id")
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise DependencyUnavailable("sentence-transformers is not installed") from exc
            self._model = CrossEncoder(self.config.model_id, revision=self.config.revision,
                                       max_length=self.config.max_tokens,
                                       device=resolve_device(self.config.device))
            materialize_parameters(self._model.model)
        return self._model

    def resolved_revision(self) -> str | None:
        assert self.config.model_id is not None
        return resolve_commit_hash(self.config.model_id, self.config.revision, self._load().model.config)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self._load().predict([(query, d) for d in documents],
                                      batch_size=self.config.batch_size, show_progress_bar=False)
        assert_finite(np.asarray(scores, dtype=np.float64), f"reranker '{self.config.model_id}'")
        return [float(s) for s in scores]


class MedCPTCrossEncoderReranker(Reranker):
    """``ncbi/MedCPT-Cross-Encoder``: sequence-classification logits for (query, article)."""

    def __init__(self, config: RerankerConfig) -> None:
        super().__init__(config)
        if not config.model_id:
            raise ConfigurationError("medcpt_cross_encoder reranker requires model_id")
        self._loaded: dict[str, Any] = {}
        self._device = resolve_device(config.device)

    def _load(self) -> dict[str, Any]:
        if not self._loaded:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise DependencyUnavailable("transformers / torch are not installed") from exc
            model_id = self.config.model_id
            assert model_id is not None
            self._loaded = {
                "torch": torch,
                "tok": AutoTokenizer.from_pretrained(model_id, revision=self.config.revision),
                "model": materialize_parameters(AutoModelForSequenceClassification.from_pretrained(
                    model_id, revision=self.config.revision
                )).to(self._device).eval(),
            }
        return self._loaded

    def resolved_revision(self) -> str | None:
        assert self.config.model_id is not None
        return resolve_commit_hash(self.config.model_id, self.config.revision, self._load()["model"].config)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        loaded = self._load()
        torch = loaded["torch"]
        out: list[float] = []
        for start in range(0, len(documents), self.config.batch_size):
            batch = documents[start : start + self.config.batch_size]
            enc = loaded["tok"]([[query, d] for d in batch], truncation=True, padding=True,
                                return_tensors="pt", max_length=self.config.max_tokens)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = loaded["model"](**enc).logits.squeeze(dim=1)
            out.extend(float(v) for v in logits.cpu())
        assert_finite(np.asarray(out, dtype=np.float64), f"reranker '{self.config.model_id}'")
        return out


_TOKEN = re.compile(r"[a-z0-9]+")
_NEGATIONS = {"no", "not", "without", "negative", "absent"}


class LexicalOverlapReranker(Reranker):
    """Dependency-free reranker: IDF-weighted bigram/unigram overlap with a penalty when
    exactly one side is negated.  Used for CPU-only smoke runs and tests; it is a real
    scoring function, not a stand-in for the neural rerankers in evaluations."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        q_tokens = _TOKEN.findall(query.lower())
        doc_tokens = [_TOKEN.findall(d.lower()) for d in documents]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        n = max(len(documents), 1)
        q_set, q_bigrams = set(q_tokens), set(itertools.pairwise(q_tokens))
        q_negated = bool(q_set & _NEGATIONS)
        scores = []
        for tokens in doc_tokens:
            d_set = set(tokens)
            unigram = sum(math.log(1 + n / (1 + df[t])) for t in q_set & d_set)
            bigram = len(q_bigrams & set(itertools.pairwise(tokens)))
            score = unigram + 1.5 * bigram
            if q_negated != bool(d_set & _NEGATIONS):
                score *= 0.6
            scores.append(score / (1.0 + 0.05 * abs(len(tokens) - len(q_tokens))))
        return scores


def build_reranker(config: RerankerConfig) -> Reranker | None:
    if not config.enabled or config.provider == "none":
        return None
    builders: dict[str, type[Reranker]] = {
        "cross_encoder": CrossEncoderReranker,
        "medcpt_cross_encoder": MedCPTCrossEncoderReranker,
        "lexical_overlap": LexicalOverlapReranker,
    }
    return builders[config.provider](config)


class RerankerRegistry:
    """Caches loaded rerankers by (provider, model, revision)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None, str | None, int], Reranker] = {}

    def get(self, config: RerankerConfig) -> Reranker | None:
        if not config.enabled or config.provider == "none":
            return None
        key = (config.provider, config.model_id, config.revision, config.max_tokens)
        if key not in self._cache:
            built = build_reranker(config)
            assert built is not None
            self._cache[key] = built
        reranker = self._cache[key]
        reranker.config = config  # depth / batch size may differ between recipes
        return reranker
