"""Context construction: which retrieved chunks reach the generator, and in what form.

``top_k`` packs the ranking as it is (the baseline).  ``diverse`` removes
near-duplicates, limits chunks per document, applies a relevance floor and uses
Maximal Marginal Relevance so the token budget is spent on distinct evidence.
Optional neighbour expansion restores the text around a small chunk
(small-to-big / parent-document retrieval).  Every decision is counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.corpus.config import ContextParams
from app.corpus.retrieval import AuthContext, RetrievedChunk
from app.corpus.schema import make_chunk_id
from app.indexing.backend import SearchBackend
from app.rag.evidence import estimate_tokens, neutralize

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class ContextItem:
    source_id: str  # "S1", "S2", ... the only identifiers the generator may cite
    chunk: RetrievedChunk
    text: str  # neutralised text shown to the generator (may include expanded neighbours)
    tokens: int


@dataclass
class ContextBundle:
    items: list[ContextItem]
    tokens_used: int
    token_budget: int
    dropped: dict[str, int] = field(default_factory=dict)
    selection: str = "top_k"

    def by_source(self) -> dict[str, ContextItem]:
        return {item.source_id: item for item in self.items}

    def render(self) -> str:
        blocks = []
        for item in self.items:
            c = item.chunk
            when = f", dated {c.event_time[:10]}" if c.event_time else ""
            blocks.append(f"[{item.source_id}] ({c.title or c.document_type}{when}) <<<{item.text}>>>")
        return "\n".join(blocks)

    def stats(self) -> dict[str, Any]:
        texts = [_tokens(i.text) for i in self.items]
        pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
        duplicates = sum(1 for a, b in pairs if _jaccard(a, b) >= 0.85)
        return {"chunks": len(self.items), "tokens": self.tokens_used, "budget": self.token_budget,
                "documents": len({i.chunk.document_id for i in self.items}),
                "duplicate_pairs": duplicates, "dropped": self.dropped, "selection": self.selection}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def select_context(chunks: list[RetrievedChunk], params: ContextParams, final_k: int | None = None,
                   backend: SearchBackend | None = None, index: str | None = None,
                   auth: AuthContext | None = None) -> ContextBundle:
    limit = min(params.max_chunks, final_k or params.max_chunks)
    dropped = {"duplicate": 0, "per_document": 0, "low_score": 0, "over_budget": 0}
    candidates = list(chunks)
    if params.selection == "top_k":
        chosen: list[RetrievedChunk] = []
        used = 0
        for chunk in candidates[:limit]:
            cost = chunk.token_count or estimate_tokens(chunk.text)
            if used + cost > params.token_budget:
                dropped["over_budget"] += 1
                continue
            chosen.append(chunk)
            used += cost
    else:
        best = max((c.score for c in candidates), default=0.0)
        floor = params.min_relative_score * best if best > 0 else None
        pool: list[tuple[RetrievedChunk, frozenset[str], float]] = []
        for rank, chunk in enumerate(candidates):
            if floor is not None and chunk.score < floor:
                dropped["low_score"] += 1
                continue
            pool.append((chunk, _tokens(chunk.text), 1.0 / (1.0 + rank)))  # rank-based relevance
        chosen, used = [], 0
        chosen_tokens: list[frozenset[str]] = []
        per_doc: dict[str, int] = {}
        while pool and len(chosen) < limit:
            scored = []
            for i, (_candidate, toks, relevance) in enumerate(pool):
                redundancy = max((_jaccard(toks, other) for other in chosen_tokens), default=0.0)
                scored.append((params.mmr_lambda * relevance - (1 - params.mmr_lambda) * redundancy, -i, i,
                               redundancy))
            _score, _tie, index_in_pool, redundancy = max(scored)
            chunk, toks, _relevance = pool.pop(index_in_pool)
            if redundancy >= params.duplicate_threshold:
                dropped["duplicate"] += 1
                continue
            if per_doc.get(chunk.document_id, 0) >= params.per_document_limit:
                dropped["per_document"] += 1
                continue
            cost = chunk.token_count or estimate_tokens(chunk.text)
            if used + cost > params.token_budget:
                dropped["over_budget"] += 1
                continue
            chosen.append(chunk)
            chosen_tokens.append(toks)
            per_doc[chunk.document_id] = per_doc.get(chunk.document_id, 0) + 1
            used += cost
        # present evidence in ranking order, not in selection order
        order = {c.chunk_id: i for i, c in enumerate(candidates)}
        chosen.sort(key=lambda c: order[c.chunk_id])

    items: list[ContextItem] = []
    used = 0
    selected_ids = {c.chunk_id for c in chosen}
    for n, chunk in enumerate(chosen, start=1):
        text = chunk.text
        if params.parent_expansion and backend is not None and index is not None:
            text = _expand(chunk, backend, index, auth, selected_ids, params.token_budget - used)
        text = neutralize(text)
        tokens = estimate_tokens(text)
        items.append(ContextItem(f"S{n}", chunk, text, tokens))
        used += tokens
    return ContextBundle(items, used, params.token_budget, {k: v for k, v in dropped.items() if v},
                         params.selection)


def _expand(chunk: RetrievedChunk, backend: SearchBackend, index: str, auth: AuthContext | None,
            selected: set[str], room: int) -> str:
    """Add the previous and the next chunk of the same document when they fit the budget."""
    parts = [chunk.text]
    budget = room - (chunk.token_count or estimate_tokens(chunk.text))
    for offset, prepend in ((-1, True), (1, False)):
        position = chunk.chunk_index + offset
        if position < 0 or position >= chunk.chunk_count:
            continue
        neighbour_id = make_chunk_id(chunk.document_id, position)
        if neighbour_id in selected:
            continue  # it is already a context item of its own
        source = backend.get_document(index, neighbour_id)
        # same document, but still verified: version must match and the caller must be allowed
        if (source is None or int(source.get("document_version", -1)) != chunk.document_version
                or (auth is not None and not auth.allows(source))):
            continue
        neighbour = source.get("chunk_text") or ""
        cost = estimate_tokens(neighbour)
        if cost > budget:
            continue
        budget -= cost
        parts = [neighbour, *parts] if prepend else [*parts, neighbour]
    return " ".join(parts)
