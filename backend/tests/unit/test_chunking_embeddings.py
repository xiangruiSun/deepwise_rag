"""Chunk boundaries and embedding encoding contracts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.core.config import ChunkingConfig, EmbeddingConfig
from app.core.errors import ConfigurationError
from app.embeddings.base import CachedEmbedder, EmbeddingCache, WhitespaceTokenizer
from app.embeddings.providers import (
    HashingEmbeddingProvider,
    MedCPTProvider,
    SentenceTransformerProvider,
)
from app.embeddings.registry import EmbeddingRegistry
from app.indexing.chunking import Chunker, ChunkValidationError, SourceDocument, chunk_statistics

LONG = " ".join(
    f"Sentence {i} describes segmental filling defects in the right lower lobe without strain." for i in range(1, 13)
)
SHORT = "No evidence of pulmonary embolism."


def cfg(strategy: str, max_tokens: int, overlap: int = 0) -> ChunkingConfig:
    return ChunkingConfig(key=f"{strategy}_{max_tokens}_{overlap}", strategy=strategy,  # type: ignore[arg-type]
                          max_tokens=max_tokens, overlap=overlap)


@pytest.mark.parametrize("strategy,max_tokens,overlap", [
    ("whole_impression", 512, 0), ("fixed", 32, 0), ("sliding_window", 32, 8), ("sentence", 32, 0),
    ("structure", 32, 0), ("parent_child", 16, 0),
])
def test_chunks_are_lossless_bounded_and_traceable(strategy: str, max_tokens: int, overlap: int) -> None:
    chunker = Chunker(WhitespaceTokenizer(), encoder_limit=64)
    doc = SourceDocument("rep1", LONG)
    chunks = chunker.split(doc, cfg(strategy, max_tokens, overlap))
    chunker.validate_chunks(chunks)  # offsets, coverage, sequential indexes
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert LONG[chunk.char_start:chunk.char_end] == chunk.text
        assert chunk.parent_id == "rep1" and chunker.get_parent_document(chunk.chunk_id).text == LONG
        assert chunk.sentence_offsets
        if strategy != "whole_impression":
            assert chunk.token_count <= min(max_tokens, 64)
    if strategy == "whole_impression":
        assert len(chunks) == 1 and chunks[0].exceeds_encoder_limit  # tracked, never silent
        assert chunk_statistics(chunks, 1)["pct_documents_truncated"] == 100.0
    else:
        assert len(chunks) > 1 and not any(c.exceeds_encoder_limit for c in chunks)


def test_short_impressions_are_never_split() -> None:
    chunker = Chunker(WhitespaceTokenizer(), encoder_limit=64)
    for strategy in ("fixed", "sliding_window", "sentence", "structure", "parent_child", "whole_impression"):
        chunks = chunker.split(SourceDocument("r", SHORT), cfg(strategy, 16))
        assert len(chunks) == 1 and chunks[0].text == SHORT and not chunks[0].exceeds_encoder_limit


def test_sliding_window_overlap_and_sentence_boundaries() -> None:
    chunker = Chunker(WhitespaceTokenizer(), encoder_limit=512)
    spans = WhitespaceTokenizer().token_spans(LONG)
    windows = chunker.split(SourceDocument("r", LONG), cfg("sliding_window", 32, 8))
    for first, second in itertools.pairwise(windows):
        shared = [s for s in spans if s[0] >= second.char_start and s[1] <= first.char_end]
        assert len(shared) == 8
    sentences = chunker.split(SourceDocument("r", LONG), cfg("sentence", 32))
    assert all(c.text.endswith(".") and c.text.startswith("Sentence") for c in sentences)


def test_structure_chunks_keep_sections_apart() -> None:
    text = "FINDINGS: " + LONG[:300] + ".\nIMPRESSION: " + LONG[300:]
    chunker = Chunker(WhitespaceTokenizer(), encoder_limit=512)
    chunks = chunker.split(SourceDocument("r", text), cfg("structure", 40))
    chunker.validate_chunks(chunks)
    assert {c.section for c in chunks} == {"FINDINGS", "IMPRESSION"}
    boundary = text.index("IMPRESSION:")
    assert all(c.char_end <= boundary or c.char_start >= boundary for c in chunks)


def test_validation_rejects_corrupt_chunks() -> None:
    chunker = Chunker(WhitespaceTokenizer(), encoder_limit=16)
    chunks = chunker.split(SourceDocument("r", LONG), cfg("fixed", 16))
    chunks[1].text = "tampered"
    with pytest.raises(ChunkValidationError, match="offsets"):
        chunker.validate_chunks(chunks)
    good = chunker.split(SourceDocument("r", LONG), cfg("fixed", 16))
    with pytest.raises(ChunkValidationError, match="not covered"):
        chunker.validate_chunks(good[:-1])
    with pytest.raises(ValueError):
        ChunkingConfig(key="x", strategy="fixed", max_tokens=16, overlap=16)


def hashing(dim: int = 64, **kw) -> HashingEmbeddingProvider:  # type: ignore[no-untyped-def]
    return HashingEmbeddingProvider(EmbeddingConfig(key="h", provider="hashing", model_id="hashing-ngram-v1",
                                                    dimension=dim, pooling="hashing", **kw))


def test_hashing_provider_contract() -> None:
    provider = hashing()
    docs = provider.encode_documents(["acute pulmonary embolism", "no pulmonary embolism", "pleural effusion"])
    query = provider.encode_queries(["acute pulmonary embolism"])
    assert docs.shape == (3, 64) and query.shape == (1, 64) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    sims = docs @ query[0]
    assert sims[0] == pytest.approx(1.0, abs=1e-5) and sims[0] > sims[1] > sims[2]
    assert np.array_equal(provider.encode_documents(["x y z"]), provider.encode_documents(["x y z"]))
    meta = provider.model_metadata()
    for key in ("model_id", "revision", "device", "batch_size", "max_document_tokens", "pooling",
                "normalize", "dimension", "query_prefix", "document_prefix"):
        assert key in meta


def test_truncation_is_tracked_not_silent() -> None:
    provider = hashing(max_document_tokens=8, max_query_tokens=8)
    provider.encode_documents(["one two three", " ".join(["tok"] * 20)])
    provider.encode_queries([" ".join(["q"] * 9)])
    assert provider.truncation == {"documents": 2, "documents_truncated": 1, "queries": 1, "queries_truncated": 1}


def test_cache_is_keyed_by_model_identity_and_role(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    a, b = hashing(64), hashing(64, preprocessing_version="text-v2")
    ea, eb = CachedEmbedder(a, cache), CachedEmbedder(b, cache)
    first = ea.encode_documents(["same text", "same text", "other"])
    assert cache.misses == 2 and cache.count() == 2  # duplicates in a batch are embedded once
    again = ea.encode_documents(["same text"])
    assert cache.hits >= 1 and np.array_equal(first[0], again[0])
    ea.encode_queries(["same text"])  # different role → different entry
    eb.encode_documents(["same text"])  # different preprocessing version → different entry
    assert cache.count() == 4
    from app.core.hashing import sha256_text

    assert cache.invalidate_content([sha256_text("same text")]) == 3 and cache.count() == 1


class _FakeST:
    """Stands in for a SentenceTransformer so the prefix contract can be tested offline."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.max_seq_length = 0
        self.tokenizer = None

    def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.extend(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def test_e5_and_bge_prefix_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EmbeddingConfig(key="e5", provider="sentence_transformers", model_id="intfloat/e5-base-v2",
                             dimension=4, query_prefix="query: ", document_prefix="passage: ")
    provider = SentenceTransformerProvider(config)
    fake = _FakeST()
    provider._model = fake
    monkeypatch.setattr(provider, "tokenizer", lambda: WhitespaceTokenizer())
    provider.encode_queries(["segmental embolus"])
    provider.encode_documents(["No embolus."])
    assert fake.seen == ["query: segmental embolus", "passage: No embolus."]


def test_medcpt_requires_separate_query_encoder_and_cls_pooling() -> None:
    base = {"key": "medcpt", "provider": "medcpt", "model_id": "ncbi/MedCPT-Article-Encoder", "dimension": 768}
    with pytest.raises(ConfigurationError, match="query_model_id"):
        MedCPTProvider(EmbeddingConfig(**base, pooling="cls"))
    with pytest.raises(ConfigurationError, match="CLS"):
        MedCPTProvider(EmbeddingConfig(**base, pooling="mean", query_model_id="ncbi/MedCPT-Query-Encoder"))
    ok = MedCPTProvider(EmbeddingConfig(**base, pooling="cls", query_model_id="ncbi/MedCPT-Query-Encoder",
                                        max_query_tokens=64, max_document_tokens=512, normalize=False,
                                        similarity="max_inner_product"))
    assert ok.config.max_query_tokens == 64 and ok.config.similarity == "max_inner_product"


def test_registry_rejects_unknown_embedding() -> None:
    registry = EmbeddingRegistry({"h": hashing().config})
    assert registry.provider("h") is registry.provider("h")
    with pytest.raises(ConfigurationError):
        registry.provider("nope")


@pytest.mark.neural
def test_real_models_follow_their_official_procedures() -> None:
    """Runs only when PyTorch and locally cached model weights are available."""
    pytest.importorskip("sentence_transformers")
    from app.core.config import load_app_config
    from tests.conftest import CONFIG_DIR

    configs = load_app_config("synthetic.yaml", CONFIG_DIR).embeddings
    registry = EmbeddingRegistry(configs)
    texts = ["Acute pulmonary embolism in segmental branches of the right lower lobe.",
             "No evidence of pulmonary embolism.", "Small left pleural effusion."]
    for key in ("minilm", "e5", "bge", "medcpt"):
        try:
            provider = registry.provider(key)
            docs = provider.encode_documents(texts)
        except Exception as exc:  # weights not cached / offline
            pytest.skip(f"{key}: model not available locally ({type(exc).__name__})")
        query = provider.encode_queries(["segmental pulmonary embolus"])
        assert docs.shape == (3, configs[key].dimension) and query.shape == (1, configs[key].dimension)
        if configs[key].normalize:
            assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3)
        assert int(np.argmax(docs @ query[0])) == 0
        revision = provider.resolved_revision()
        assert revision["document"] and revision["query"]
        if key == "medcpt":
            assert revision["document"] != revision["query"]  # two different encoders
