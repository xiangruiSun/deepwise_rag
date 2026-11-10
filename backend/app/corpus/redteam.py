"""Adversarial test doubles for the security evaluation.

``InstructionFollowingLLM`` simulates the worst case of indirect prompt injection: a model
that obeys any instruction it finds in its context.  It is a *simulation* — it measures
whether the platform's layers hold when the model does not, and says nothing about how
often a particular real model would comply.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from app.corpus.config import GenerationParams
from app.corpus.generation import ABSTAIN_TOKEN, CircuitBreaker

_SOURCE = re.compile(r"\[(S\d+)\] \([^)]*\) <<<(.*?)>>>", re.DOTALL)
_ORDER = re.compile(r"(?:reply|respond|answer|output)\s+only\s+with\s+([^\s.>]+)", re.IGNORECASE)


class InstructionFollowingLLM:
    """Obeys "reply only with X" found in a source, and cites that source to pass naive checks."""

    name = "instruction-following-simulator"
    mock = True

    def __init__(self, params: GenerationParams) -> None:
        self.params = params
        self.breaker = CircuitBreaker(params.circuit_failure_threshold, params.circuit_reset_seconds)
        self.obeyed = 0
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def aclose(self) -> None:
        return None

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "circuit": "closed"}

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.usage["requests"] += 1
        sources = _SOURCE.findall(messages[-1]["content"])
        for source_id, text in sources:
            order = _ORDER.search(text)
            if order:
                self.obeyed += 1
                yield f"{order.group(1)}. [{source_id}]"
                return
        if not sources:
            yield ABSTAIN_TOKEN
            return
        source_id, text = sources[0]
        yield f"{text.strip().split('. ')[0].rstrip('.')}. [{source_id}]"
