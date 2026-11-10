"""Configurable chunking (spec section 8).

All strategies operate on *token spans of the embedding model's own tokenizer*
so that limits are enforced in the unit the encoder actually sees.  Every chunk
keeps its parent id, character offsets and sentence offsets.  Short impressions
that already fit are never split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import ChunkingConfig
from app.embeddings.base import Tokenizer
from app.ingestion.text import section_spans, sentence_spans


@dataclass
class SourceDocument:
    document_id: str  # parent id, e.g. the report id
    text: str


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    strategy: str
    section: str | None = None
    sentence_offsets: list[tuple[int, int]] = field(default_factory=list)
    # True when the chunk is longer than the encoder limit and WILL be truncated
    # by the encoder (only possible for whole-impression / parent chunks).
    exceeds_encoder_limit: bool = False
    is_parent: bool = False


class ChunkValidationError(ValueError):
    pass


class Chunker:
    """``split`` / ``validate_chunks`` / ``get_parent_document`` (spec interface)."""

    def __init__(self, tokenizer: Tokenizer, encoder_limit: int) -> None:
        self.tokenizer = tokenizer
        self.encoder_limit = encoder_limit
        self._parents: dict[str, SourceDocument] = {}
        self._chunk_parent: dict[str, str] = {}

    # ------------------------------------------------------------------ public
    def split(self, document: SourceDocument, configuration: ChunkingConfig) -> list[Chunk]:
        text = document.text
        spans = self.tokenizer.token_spans(text)
        self._parents[document.document_id] = document
        if not spans:
            return []
        limit = min(configuration.max_tokens, self.encoder_limit)
        strategy = configuration.strategy
        if strategy == "whole_impression" or len(spans) <= limit:
            # short impressions are never split, whatever the strategy
            pieces: list[tuple[int, int, str | None]] = [(0, len(spans), None)]
        elif strategy in ("fixed", "sliding_window"):
            overlap = 0 if strategy == "fixed" else configuration.overlap
            if strategy == "fixed" and configuration.overlap:
                overlap = configuration.overlap  # "fixed 256 / overlap 32" style configs
            pieces = [(a, b, None) for a, b in _windows(len(spans), limit, overlap)]
        elif strategy in ("sentence", "parent_child"):
            pieces = self._sentence_pieces(text, spans, limit, (0, len(text)), None)
        elif strategy == "structure":
            pieces = []
            for name, start, end in section_spans(text):
                pieces.extend(self._sentence_pieces(text, spans, limit, (start, end), name))
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown chunking strategy '{strategy}'")

        sentences = sentence_spans(text)
        chunks: list[Chunk] = []
        for index, (tok_start, tok_end, section) in enumerate(pieces):
            char_start, char_end = spans[tok_start][0], spans[tok_end - 1][1]
            count = tok_end - tok_start
            chunk = Chunk(
                chunk_id=f"{document.document_id}::{configuration.key}::{index}",
                parent_id=document.document_id,
                chunk_index=index,
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_count=count,
                strategy=configuration.key,
                section=section,
                sentence_offsets=[
                    (max(s, char_start), min(e, char_end))
                    for s, e in sentences
                    if s < char_end and e > char_start
                ],
                exceeds_encoder_limit=count > self.encoder_limit,
            )
            self._chunk_parent[chunk.chunk_id] = document.document_id
            chunks.append(chunk)
        return chunks

    def validate_chunks(self, chunks: list[Chunk]) -> None:
        """Raise ``ChunkValidationError`` unless chunks are well-formed and lossless."""
        by_parent: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_parent.setdefault(chunk.parent_id, []).append(chunk)
        for parent_id, group in by_parent.items():
            parent = self._parents.get(parent_id)
            if parent is None:
                raise ChunkValidationError(f"unknown parent document '{parent_id}'")
            group.sort(key=lambda c: c.chunk_index)
            covered = [False] * len(parent.text)
            for expected, chunk in enumerate(group):
                if chunk.chunk_index != expected:
                    raise ChunkValidationError(f"{parent_id}: non-sequential chunk index")
                if not chunk.text.strip():
                    raise ChunkValidationError(f"{chunk.chunk_id}: empty chunk")
                if parent.text[chunk.char_start : chunk.char_end] != chunk.text:
                    raise ChunkValidationError(f"{chunk.chunk_id}: offsets do not match parent text")
                if chunk.token_count > self.encoder_limit and not chunk.exceeds_encoder_limit:
                    raise ChunkValidationError(
                        f"{chunk.chunk_id}: {chunk.token_count} tokens silently exceed the "
                        f"encoder limit of {self.encoder_limit}"
                    )
                for i in range(chunk.char_start, chunk.char_end):
                    covered[i] = True
            for start, end in self.tokenizer.token_spans(parent.text):
                if not all(covered[start:end]):
                    raise ChunkValidationError(f"{parent_id}: source tokens are not covered by any chunk")

    def get_parent_document(self, chunk_id: str) -> SourceDocument:
        try:
            return self._parents[self._chunk_parent[chunk_id]]
        except KeyError as exc:
            raise KeyError(f"unknown chunk id '{chunk_id}'") from exc

    # ----------------------------------------------------------------- helpers
    def _sentence_pieces(
        self,
        text: str,
        spans: list[tuple[int, int]],
        limit: int,
        region: tuple[int, int],
        section: str | None,
    ) -> list[tuple[int, int, str | None]]:
        """Greedily pack whole sentences; an over-long sentence falls back to windows."""
        region_tokens = [i for i, (s, e) in enumerate(spans) if s >= region[0] and e <= region[1]]
        if not region_tokens:
            return []
        first, last = region_tokens[0], region_tokens[-1] + 1
        boundaries: list[int] = []
        for _s, e in sentence_spans(text[region[0] : region[1]]):
            end_char = region[0] + e
            idx = next((i for i in range(first, last) if spans[i][1] > end_char), last)
            if idx > first and (not boundaries or idx > boundaries[-1]):
                boundaries.append(idx)
        if not boundaries or boundaries[-1] != last:
            boundaries.append(last)
        pieces: list[tuple[int, int, str | None]] = []
        start = first
        current_end = first
        for boundary in boundaries:
            if boundary - start <= limit:
                current_end = boundary
                continue
            if current_end > start:
                pieces.append((start, current_end, section))
                start = current_end
            if boundary - start > limit:  # a single sentence longer than the limit
                for a, b in _windows(boundary - start, limit, 0):
                    pieces.append((start + a, start + b, section))
                start = boundary
            current_end = boundary
        if current_end > start:
            pieces.append((start, current_end, section))
        return pieces


def _windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    step = size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += step
    return windows


def chunk_statistics(chunks: list[Chunk], n_documents: int) -> dict[str, float]:
    if not chunks:
        return {"chunks": 0, "avg_tokens": 0.0, "max_tokens": 0, "pct_documents_truncated": 0.0}
    truncated_parents = {c.parent_id for c in chunks if c.exceeds_encoder_limit}
    return {
        "chunks": len(chunks),
        "avg_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 2),
        "max_tokens": max(c.token_count for c in chunks),
        "pct_documents_truncated": round(100.0 * len(truncated_parents) / max(n_documents, 1), 3),
    }
