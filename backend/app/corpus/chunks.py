"""Chunking of canonical documents.

``fixed``, ``sliding_window``, ``sentence`` and ``structure`` reuse the token-span
chunker of ``app.indexing.chunking``.  Two strategies are added here:

* ``recursive`` — split on paragraphs, then sentences, then token windows, and
  greedily pack neighbours up to the limit (boundaries follow the text's own
  structure wherever it fits);
* ``semantic`` — start a new chunk where the similarity between adjacent
  sentences drops below a threshold.  It costs one embedding per sentence and
  is NOT assumed to be better: it is one more configuration to evaluate.

``parent_child`` produces small sentence-level chunks; the surrounding context
is restored at answer time by neighbour expansion (``app.corpus.context``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.core.config import ChunkingConfig
from app.corpus.config import CorpusChunkingConfig
from app.embeddings.base import Tokenizer
from app.indexing.chunking import Chunker, SourceDocument
from app.ingestion.text import section_spans, sentence_spans

_PARAGRAPH = re.compile(r"\n\s*\n|\n")
_REUSED = {"fixed": "fixed", "sliding_window": "sliding_window", "sentence": "sentence",
           "structure": "structure", "parent_child": "sentence"}


@dataclass
class Piece:
    text: str
    token_count: int
    section_path: str | None = None


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.token_spans(text))


def _windows(tokenizer: Tokenizer, text: str, limit: int) -> list[str]:
    spans = tokenizer.token_spans(text)
    return [text[spans[i][0]:spans[min(i + limit, len(spans)) - 1][1]] for i in range(0, len(spans), limit)]


def _recursive(text: str, tokenizer: Tokenizer, limit: int) -> list[Piece]:
    units: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(text)):
        if not paragraph:
            continue
        if _count(tokenizer, paragraph) <= limit:
            units.append(paragraph)
            continue
        for start, end in sentence_spans(paragraph):
            sentence = paragraph[start:end].strip()
            if sentence:
                units.extend([sentence] if _count(tokenizer, sentence) <= limit
                             else _windows(tokenizer, sentence, limit))
    pieces: list[Piece] = []
    current: list[str] = []
    size = 0
    for unit in units:
        n = _count(tokenizer, unit)
        if current and size + n > limit:
            pieces.append(Piece(" ".join(current), size))
            current, size = [], 0
        current.append(unit)
        size += n
    if current:
        pieces.append(Piece(" ".join(current), size))
    return pieces


def _semantic(text: str, tokenizer: Tokenizer, limit: int, breakpoint: float,
              embed: Callable[[list[str]], np.ndarray]) -> list[Piece]:
    sentences = [text[s:e].strip() for s, e in sentence_spans(text) if text[s:e].strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return _recursive(text, tokenizer, limit)
    vectors = np.asarray(embed(sentences), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    pieces: list[Piece] = []
    current, size = [sentences[0]], _count(tokenizer, sentences[0])
    for i in range(1, len(sentences)):
        n = _count(tokenizer, sentences[i])
        similarity = float(vectors[i - 1] @ vectors[i])
        if similarity < breakpoint or size + n > limit:
            pieces.extend(_recursive(" ".join(current), tokenizer, limit))
            current, size = [], 0
        current.append(sentences[i])
        size += n
    pieces.extend(_recursive(" ".join(current), tokenizer, limit))
    return pieces


def chunk_document(document_id: str, text: str, config: CorpusChunkingConfig, tokenizer: Tokenizer,
                   encoder_limit: int, embed: Callable[[list[str]], np.ndarray] | None = None) -> list[Piece]:
    """Split ``text`` into pieces of at most ``min(max_tokens, encoder_limit)`` tokens."""
    if not text.strip():
        return []
    limit = min(config.max_tokens, encoder_limit)
    if config.strategy == "recursive":
        return _recursive(text, tokenizer, limit)
    if config.strategy == "semantic":
        if embed is None:
            raise ValueError("semantic chunking needs a sentence embedding function")
        return _semantic(text, tokenizer, limit, config.semantic_breakpoint, embed)
    legacy = ChunkingConfig(key=config.identity, strategy=_REUSED[config.strategy],  # type: ignore[arg-type]
                            max_tokens=limit, overlap=min(config.overlap, limit - 1) if
                            config.strategy == "sliding_window" else 0)
    chunks = Chunker(tokenizer, encoder_limit).split(SourceDocument(document_id, text), legacy)
    return [Piece(c.text, c.token_count, c.section) for c in chunks if not c.is_parent and c.text.strip()]


def section_names(text: str) -> list[str]:
    return [name for name, _s, _e in section_spans(text)]
