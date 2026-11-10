"""Concrete embedding providers.

* ``HashingEmbeddingProvider`` — dependency-free lexical-hashing baseline used for
  CPU-only smoke runs and tests.  It is NOT one of the four compared models.
* ``SentenceTransformerProvider`` — MiniLM, E5 (query:/passage: prefixes) and BGE
  (query instruction) through SentenceTransformers.
* ``MedCPTProvider`` — NCBI MedCPT with *separate* query and article encoders,
  CLS pooling, 64-token queries and 512-token articles, inner-product scoring.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from typing import Any

import numpy as np

from app.core.config import EmbeddingConfig
from app.core.errors import ConfigurationError, DependencyUnavailable
from app.embeddings.base import (
    EmbeddingProvider,
    Tokenizer,
    WhitespaceTokenizer,
    assert_finite,
    materialize_parameters,
    resolve_commit_hash,
    resolve_device,
)

_WORDS = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Hashing baseline
# --------------------------------------------------------------------------- #
class HashingEmbeddingProvider(EmbeddingProvider):
    """Signed feature hashing of word uni/bi-grams and character trigrams."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._tokenizer = WhitespaceTokenizer()

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def resolved_revision(self) -> dict[str, str | None]:
        return {"document": "hashing-v1", "query": "hashing-v1"}

    def _features(self, text: str) -> Counter[str]:
        words = _WORDS.findall(text.lower())
        feats: Counter[str] = Counter()
        for w in words:
            feats[f"w:{w}"] += 1.0  # type: ignore[assignment]
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                feats[f"c:{padded[i:i + 3]}"] += 0.3  # type: ignore[assignment]
        for a, b in itertools.pairwise(words):
            feats[f"b:{a}_{b}"] += 0.7  # type: ignore[assignment]
        return feats

    def _embed(self, text: str, limit: int) -> np.ndarray:
        spans = self._tokenizer.token_spans(text)
        if len(spans) > limit:
            text = text[: spans[limit - 1][1]]
        dim = self.config.dimension
        vec = np.zeros(dim, dtype=np.float32)
        for feat, tf in self._features(text).items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
        if self.config.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return vec

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self._track("documents", texts, self.config.max_document_tokens)
        return np.vstack([self._embed(t, self.config.max_document_tokens) for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self._track("queries", texts, self.config.max_query_tokens)
        return np.vstack([self._embed(t, self.config.max_query_tokens) for t in texts])


# --------------------------------------------------------------------------- #
# Hugging Face helpers
# --------------------------------------------------------------------------- #
class HFTokenizerAdapter:
    """Character spans of a Hugging Face fast tokenizer (special tokens excluded)."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailable(
            "PyTorch is not installed; install backend/requirements-ml.txt to use neural models"
        ) from exc
    return torch


class SentenceTransformerProvider(EmbeddingProvider):
    special_tokens = 2

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._device = resolve_device(config.device)

    def _load(self) -> Any:
        if self._model is None:
            _require_torch()
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DependencyUnavailable("sentence-transformers is not installed") from exc
            model = SentenceTransformer(self.config.model_id, revision=self.config.revision,
                                        device=self._device)
            materialize_parameters(model)
            model.max_seq_length = max(self.config.max_document_tokens, self.config.max_query_tokens)
            dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            actual = dimension()
            if actual != self.config.dimension:
                raise ConfigurationError(
                    f"embedding '{self.config.key}': configured dimension {self.config.dimension} "
                    f"does not match model dimension {actual}"
                )
            self._model = model
        return self._model

    def device(self) -> str:
        return self._device

    def tokenizer(self) -> Tokenizer:
        return HFTokenizerAdapter(self._load().tokenizer)

    def resolved_revision(self) -> dict[str, str | None]:
        model = self._load()
        rev = resolve_commit_hash(self.config.model_id, self.config.revision, model[0].auto_model.config)
        return {"document": rev, "query": rev}

    def _encode(self, texts: list[str], prefix: str, limit: int) -> np.ndarray:
        model = self._load()
        model.max_seq_length = limit
        vectors = model.encode(
            [prefix + t for t in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        assert_finite(vectors, f"embedding model '{self.config.model_id}'")
        return vectors

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self._track("documents", texts, self.config.max_document_tokens, self.config.document_prefix)
        return self._encode(texts, self.config.document_prefix, self.config.max_document_tokens)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self._track("queries", texts, self.config.max_query_tokens, self.config.query_prefix)
        return self._encode(texts, self.config.query_prefix, self.config.max_query_tokens)


class MedCPTProvider(EmbeddingProvider):
    """MedCPT asymmetric retriever (official procedure: CLS vector of the last hidden
    state; query encoder ``max_length=64``; article encoder ``max_length=512``).

    The article encoder was trained on ``[title, abstract]`` pairs of biomedical
    literature.  Radiology impressions have no title, so the impression is passed
    as a single segment — a documented deviation whose effect must be judged by
    the evaluation, not assumed.
    """

    special_tokens = 2

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        if not config.query_model_id:
            raise ConfigurationError("MedCPT requires query_model_id (the separate query encoder)")
        if config.pooling != "cls":
            raise ConfigurationError("MedCPT must use CLS pooling")
        self._device = resolve_device(config.device)
        self._loaded: dict[str, Any] = {}

    def _load(self) -> dict[str, Any]:
        if not self._loaded:
            torch = _require_torch()
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise DependencyUnavailable("transformers is not installed") from exc
            cfg = self.config
            query_model_id = cfg.query_model_id
            assert query_model_id is not None
            self._loaded = {
                "torch": torch,
                "doc_tok": AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision),
                "doc_model": materialize_parameters(
                    AutoModel.from_pretrained(cfg.model_id, revision=cfg.revision)).to(self._device).eval(),
                "query_tok": AutoTokenizer.from_pretrained(query_model_id, revision=cfg.query_revision),
                "query_model": materialize_parameters(
                    AutoModel.from_pretrained(query_model_id, revision=cfg.query_revision)
                ).to(self._device).eval(),
            }
            actual = int(self._loaded["doc_model"].config.hidden_size)
            if actual != cfg.dimension:
                raise ConfigurationError(
                    f"embedding '{cfg.key}': configured dimension {cfg.dimension} != model {actual}"
                )
        return self._loaded

    def device(self) -> str:
        return self._device

    def tokenizer(self) -> Tokenizer:
        return HFTokenizerAdapter(self._load()["doc_tok"])

    def resolved_revision(self) -> dict[str, str | None]:
        loaded = self._load()
        cfg = self.config
        assert cfg.query_model_id is not None
        return {
            "document": resolve_commit_hash(cfg.model_id, cfg.revision, loaded["doc_model"].config),
            "query": resolve_commit_hash(cfg.query_model_id, cfg.query_revision, loaded["query_model"].config),
        }

    def _encode(self, texts: list[str], which: str, limit: int) -> np.ndarray:
        loaded = self._load()
        torch = loaded["torch"]
        tok, model = loaded[f"{which}_tok"], loaded[f"{which}_model"]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            enc = tok(batch, truncation=True, padding=True, return_tensors="pt", max_length=limit)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                cls = model(**enc).last_hidden_state[:, 0, :]
                if self.config.normalize:
                    cls = torch.nn.functional.normalize(cls, dim=-1)
            out.append(cls.cpu().numpy().astype(np.float32))
        vectors = np.vstack(out)
        assert_finite(vectors, f"MedCPT {which} encoder")
        return vectors

    def _track_with(self, role: str, texts: list[str], tok: Any, limit: int) -> None:
        for text in texts:
            self.truncation[role] += 1
            if len(tok(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]) > limit:
                self.truncation[f"{role}_truncated"] += 1

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self._track_with("documents", texts, self._load()["doc_tok"], self.config.max_document_tokens)
        return self._encode(texts, "doc", self.config.max_document_tokens)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self._track_with("queries", texts, self._load()["query_tok"], self.config.max_query_tokens)
        return self._encode(texts, "query", self.config.max_query_tokens)


PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "hashing": HashingEmbeddingProvider,
    "sentence_transformers": SentenceTransformerProvider,
    "medcpt": MedCPTProvider,
}
