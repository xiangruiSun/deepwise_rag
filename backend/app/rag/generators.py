"""Generation backends.

* ``ExtractiveGenerator`` — deterministic, model-free composition of a structured
  comparison directly from typed evidence.  It is the default so the RAG API
  works without any LLM, and it is the reference "cannot hallucinate" baseline
  in generation experiments.
* ``LLMGenerator`` — prompts a local LLM (Ollama, vLLM/OpenAI-compatible server,
  or an in-process Hugging Face model) and parses its JSON answer.
* External APIs are reachable only through ``backend: external`` whose
  configuration is rejected unless explicitly enabled and confirmed.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import GenerationConfig, get_settings
from app.core.errors import ConfigurationError, DependencyUnavailable
from app.rag.evidence import EvidenceBundle, EvidenceItem, estimate_tokens
from app.rag.prompts import PROMPT_VERSION, build_prompt
from app.schemas.rag import Claim, RagAnswer


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, max_tokens: int, temperature: float,
             json_mode: bool = True) -> LLMResult: ...

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        """Plain-text completion (used by optional query enhancement)."""
        return self.chat(system, user, max_tokens, temperature, json_mode=False).text


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: float, context_window: int = 8192) -> None:
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self.context_window = context_window

    def chat(self, system: str, user: str, max_tokens: int, temperature: float,
             json_mode: bool = True) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # num_ctx: Ollama's default window is far smaller than the evidence budget and an
            # oversized prompt is truncated from the front (the instructions) without an error.
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0,
                        "num_ctx": self.context_window},
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise DependencyUnavailable(
                f"Ollama request failed: {type(exc).__name__}", {"base_url": self.base_url}
            ) from exc
        if response.status_code == 404:
            raise DependencyUnavailable(
                f"model '{self.model}' is not available on the Ollama server; pull it first "
                f"(ollama pull {self.model})", {"base_url": self.base_url}
            )
        if response.status_code >= 400:
            raise DependencyUnavailable(f"Ollama request failed: HTTP {response.status_code}",
                                        {"base_url": self.base_url})
        data = response.json()
        input_tokens = int(data.get("prompt_eval_count") or estimate_tokens(system + user))
        if input_tokens + max_tokens > self.context_window:
            raise DependencyUnavailable(
                "the prompt does not fit the model context window together with the answer; raise "
                "generation.context_window or lower generation.context_token_budget",
                {"context_window": self.context_window, "prompt_tokens": input_tokens},
            )
        return LLMResult(
            text=data.get("message", {}).get("content", ""),
            input_tokens=input_tokens,
            output_tokens=int(data.get("eval_count") or 0),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class OpenAICompatibleClient(LLMClient):
    """vLLM (and other OpenAI-compatible) chat-completions servers."""

    def __init__(self, base_url: str, model: str, timeout: float, api_key: str | None = None) -> None:
        self.base_url, self.model, self.timeout, self.api_key = base_url.rstrip("/"), model, timeout, api_key

    def chat(self, system: str, user: str, max_tokens: int, temperature: float,
             json_mode: bool = True) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers,
                                  timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DependencyUnavailable(f"LLM server request failed: {type(exc).__name__}") from exc
        data = response.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(system + user)),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class HuggingFaceClient(LLMClient):
    """In-process ``transformers`` text-generation pipeline (local weights)."""

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailable("transformers is not installed") from exc
        self._pipe = pipeline("text-generation", model=model)

    def chat(self, system: str, user: str, max_tokens: int, temperature: float,
             json_mode: bool = True) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "return_full_text": False}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        out = self._pipe(messages, **kwargs)
        text = out[0]["generated_text"]
        if isinstance(text, list):  # chat pipelines may return the message list
            text = text[-1].get("content", "")
        return LLMResult(text, estimate_tokens(system + user), estimate_tokens(text),
                         (time.perf_counter() - t0) * 1000.0)


def build_llm_client(config: GenerationConfig) -> LLMClient | None:
    backend = config.backend
    if backend == "extractive":
        return None
    if backend == "ollama":
        return OllamaClient(config.base_url or get_settings().ollama_base_url, config.model,
                            config.timeout_seconds, config.context_window)
    if backend == "openai_compatible":
        if not config.base_url:
            raise ConfigurationError("openai_compatible generation backend requires base_url")
        return OpenAICompatibleClient(config.base_url, config.model, config.timeout_seconds)
    if backend == "huggingface":
        return HuggingFaceClient(config.model)
    if backend == "external":
        # GenerationConfig validation has already enforced the explicit confirmation flags.
        ext = config.external_api
        key = os.environ.get(ext.api_key_env)
        if not key:
            raise ConfigurationError(f"external generation requires the {ext.api_key_env} environment variable")
        assert ext.base_url is not None
        return OpenAICompatibleClient(ext.base_url, config.model, config.timeout_seconds, api_key=key)
    raise ConfigurationError(f"unknown generation backend '{backend}'")


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
class Generator(ABC):
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, bundle: EvidenceBundle) -> RagAnswer: ...


_FINDINGS: dict[str, tuple[str, ...]] = {
    "pulmonary embolism": ("pulmonary embol", "pulmonary thromboembolism", "acute thromboembolic",
                           "filling defect", "embolus"),
    "chronic thromboembolic disease": ("chronic thromboembolic", "chronic pulmonary embol", "webs and bands",
                                       "wall-adherent"),
    "right heart strain": ("right heart strain", "right ventricular strain", "right ventricular enlargement"),
    "pleural effusion": ("pleural effusion", "pleural fluid"),
    "consolidation": ("consolidation", "airspace opacity", "pneumonia"),
    "atelectasis": ("atelectasis",),
    "pulmonary nodule": ("nodule",),
    "emphysema": ("emphysema", "emphysematous"),
}
_NEGATION = re.compile(r"\b(no|not|without|negative for|absent|free of)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.;!?])\s+|\n+")


def finding_assertions(text: str) -> dict[str, str]:
    """Sentence-level assertion status of a small set of findings: ``affirmed`` or ``negated``.
    A finding that is affirmed in any sentence counts as affirmed."""
    return {finding: status for finding, (status, _phrase) in finding_mentions(text).items()}


def finding_mentions(text: str) -> dict[str, tuple[str, str]]:
    """``finding -> (status, source wording)``; the wording is quoted in claims so that
    every statement stays lexically grounded in its cited source."""
    status: dict[str, tuple[str, str]] = {}
    for sentence in _SENTENCE.split(text):
        lowered = sentence.lower()
        for finding, patterns in _FINDINGS.items():
            for pattern in patterns:
                idx = lowered.find(pattern)
                if idx < 0:
                    continue
                negated = bool(_NEGATION.search(lowered[:idx]))
                if not negated:
                    status[finding] = ("affirmed", pattern)
                else:
                    status.setdefault(finding, ("negated", pattern))
                break
    return status


class ExtractiveGenerator(Generator):
    """Builds every statement from typed evidence; each claim cites its source records."""

    def generate(self, bundle: EvidenceBundle) -> RagAnswer:
        t0 = time.perf_counter()
        answer = RagAnswer(status="answered")
        query_id = bundle.query_case_id
        reports = {cid: bundle.case_items(cid, "report") for cid in bundle.all_case_ids}
        labs = {cid: {i.data["concept_key"]: i for i in bundle.case_items(cid, "laboratory")}
                for cid in bundle.all_case_ids}

        for cid in bundle.all_case_ids:
            for item in reports[cid][:1]:
                role = "Query case" if cid == query_id else "Retrieved case"
                quote = item.text.split(": ", 1)[-1].strip()
                answer.summary.append(
                    Claim(text=f'{role} {cid}: the documented impression states "{_shorten(quote)}"',
                          citations=[item.source_id])
                )

        if query_id is not None and reports.get(query_id):
            q_item = reports[query_id][0]
            q_mentions = finding_mentions(q_item.text)
            q_findings = {k: v[0] for k, v in q_mentions.items()}
            for cid in bundle.retrieved_case_ids:
                if not reports[cid]:
                    continue
                c_item = reports[cid][0]
                c_mentions = finding_mentions(c_item.text)
                c_findings = {k: v[0] for k, v in c_mentions.items()}
                cites = [q_item.source_id, c_item.source_id]
                for label in sorted(set(q_findings) | set(c_findings)):
                    q_status, c_status = q_findings.get(label), c_findings.get(label)
                    wordings = list(dict.fromkeys(m[label][1] for m in (q_mentions, c_mentions) if label in m))
                    finding = f"{label} (source wording: {' / '.join(repr(w) for w in wordings)})"
                    if q_status == c_status:
                        verb = "document" if q_status == "affirmed" else "document no"
                        answer.similarities.append(
                            Claim(text=f"Both the query case and case {cid} {verb} {finding}.", citations=cites)
                        )
                    elif q_status and c_status:
                        answer.differences.append(
                            Claim(text=f"{finding.capitalize()} is {q_status} in the query case but "
                                       f"{c_status} in case {cid}.", citations=cites)
                        )
                    else:
                        where, status, other = (("the query case", q_status, f"case {cid}") if q_status
                                                else (f"case {cid}", c_status, "the query case"))
                        answer.differences.append(
                            Claim(text=f"{finding.capitalize()} is {status} in {where}; the impression of "
                                       f"{other} does not mention it.", citations=cites)
                        )
                self._lab_claims(answer, query_id, cid, labs)
        else:
            # no query case (free-text search or explicit comparison): compare retrieved cases pairwise
            ids = bundle.retrieved_case_ids
            findings = {cid: finding_mentions(reports[cid][0].text) for cid in ids if reports[cid]}
            for finding in sorted({f for fs in findings.values() for f in fs}):
                holders = [cid for cid in findings if findings[cid].get(finding, ("", ""))[0] == "affirmed"]
                if len(holders) >= 2:
                    wordings = list(dict.fromkeys(findings[cid][finding][1] for cid in holders))
                    answer.similarities.append(
                        Claim(text=f"{finding.capitalize()} (source wording: "
                                   f"{' / '.join(repr(w) for w in wordings)}) is documented in each of "
                                   f"these retrieved cases: {', '.join(holders)}.",
                              citations=[reports[cid][0].source_id for cid in holders])
                    )
            for first, second in itertools.pairwise(ids):
                self._lab_claims(answer, first, second, labs)

        for cid in bundle.all_case_ids:
            for gap in bundle.missing.get(cid, []):
                answer.missing_evidence.append(f"Case {cid}: {gap}.")
        if not answer.summary and not answer.similarities and not answer.differences:
            answer.status = "insufficient_evidence"
            answer.missing_evidence.append("No citable evidence is available for the selected cases.")
        answer.limitations.extend([
            "Extractive summary: statements are restricted to documented impressions and recorded "
            "laboratory values; no interpretation beyond the sources is provided.",
            "Finding detection uses a small fixed vocabulary with sentence-level negation handling "
            "and can miss or misread unusual phrasing.",
        ])
        answer.generation_metadata = {
            "backend": "extractive", "model": self.config.model, "prompt_variant": None,
            "input_tokens": bundle.token_count, "output_tokens": sum(estimate_tokens(c.text) for c in answer.claims()),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        return answer

    @staticmethod
    def _lab_claims(answer: RagAnswer, first: str, second: str,
                    labs: dict[str, dict[str, EvidenceItem]]) -> None:
        shared = sorted(set(labs.get(first, {})) & set(labs.get(second, {})))
        for key in shared:
            a, b = labs[first][key], labs[second][key]
            text = (f"{key}: case {first} {a.data['original_value']:g} {a.data['original_unit'] or ''}".rstrip()
                    + f" versus case {second} {b.data['original_value']:g} {b.data['original_unit'] or ''}".rstrip())
            if a.data["original_unit"] != b.data["original_unit"]:
                text += (f" (converted for comparison: {a.data['canonical_value']:g} vs "
                         f"{b.data['canonical_value']:g} {a.data['canonical_unit']})")
            target = answer.similarities if _close(a, b) else answer.differences
            target.append(Claim(text=text + ".", citations=[a.source_id, b.source_id]))


def _close(a: EvidenceItem, b: EvidenceItem) -> bool:
    x, y = float(a.data["canonical_value"]), float(b.data["canonical_value"])
    return abs(x - y) <= 0.2 * max(abs(x), abs(y), 1e-9)


def _shorten(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class LLMGenerator(Generator):
    def __init__(self, config: GenerationConfig, client: LLMClient) -> None:
        super().__init__(config)
        self.client = client

    def generate(self, bundle: EvidenceBundle) -> RagAnswer:
        system, user = build_prompt(bundle, self.config.prompt_variant)
        result = self.client.chat(system, user, self.config.max_output_tokens, self.config.temperature)
        metadata = {
            "backend": self.config.backend, "model": self.config.model,
            "prompt_variant": self.config.prompt_variant, "prompt_version": PROMPT_VERSION,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "latency_ms": round(result.latency_ms, 3),
            "temperature": self.config.temperature,
        }
        truncated = result.output_tokens >= self.config.max_output_tokens
        parsed = parse_llm_json(result.text, allow_truncated=truncated)
        if parsed is not None and parsed.pop("_truncated", False):
            metadata["output_truncated"] = True
        if parsed is None:
            return RagAnswer(
                status="insufficient_evidence",
                missing_evidence=["The language model did not return a parseable structured answer."],
                limitations=["Generation failed validation; no model text is shown."],
                generation_metadata={**metadata, "parse_error": True},
            )
        return RagAnswer(
            status="insufficient_evidence" if parsed.get("status") == "insufficient_evidence" else "answered",
            summary=_claims(parsed.get("summary")),
            similarities=_claims(parsed.get("similarities")),
            differences=_claims(parsed.get("differences")),
            missing_evidence=_notes(parsed.get("missing_evidence")),
            limitations=_notes(parsed.get("limitations")),
            warnings=(["The model answer reached the output limit; its incomplete last statement was "
                       "discarded."] if metadata.get("output_truncated") else []),
            generation_metadata=metadata,
        )


def parse_llm_json(text: str, allow_truncated: bool = False) -> dict[str, Any] | None:
    """Parse a model reply as one JSON object.

    With ``allow_truncated`` a reply that was cut off by the output-token limit is repaired by
    dropping the incomplete last element and closing the open brackets; the recovered object
    carries ``"_truncated": True`` so the caller can say so."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        value = None
        if match:
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                value = None
        if value is None and allow_truncated:
            value = _repair_truncated(text)
            if isinstance(value, dict):
                value["_truncated"] = True
    return value if isinstance(value, dict) else None


def _repair_truncated(text: str) -> Any:
    """Cut a truncated JSON document after its last complete element and close it."""
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]
    stack: list[str] = []
    in_string = escaped = False
    cut: tuple[int, list[str]] | None = None  # (index of the last element separator, open brackets there)
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
        elif ch == "," and stack:
            cut = (i, list(stack))
    if cut is None:
        return None
    index, open_brackets = cut
    try:
        return json.loads(text[:index] + "".join(reversed(open_brackets)))
    except json.JSONDecodeError:
        return None


_HAS_CONTENT = re.compile(r"[^\W_]")  # at least one letter or digit
_SOURCE_ID = re.compile(r"S\d+")
_INLINE_ID = re.compile(r"(?<![A-Za-z0-9_])(S\d+)(?![A-Za-z0-9_])")
_INLINE_TAG = re.compile(r"</?S\d+>|\[S\d+(?:\s*,\s*S\d+)*\]|\(S\d+(?:\s*,\s*S\d+)*\)")


def _notes(raw: Any) -> list[str]:
    """Free-text list entries; placeholder-only entries ("...", "-") are dropped."""
    if isinstance(raw, str):
        raw = [raw]
    return [str(m).strip() for m in raw or [] if _HAS_CONTENT.search(str(m))]


def _claims(raw: Any) -> list[Claim]:
    claims: list[Claim] = []
    if not isinstance(raw, list):
        return claims
    for entry in raw:
        if isinstance(entry, str):
            text, listed = entry, []
        elif isinstance(entry, dict):
            text = str(entry.get("text", ""))
            raw_cites = entry.get("citations") or []
            listed = [str(c).strip("[]<>/ ") for c in (raw_cites if isinstance(raw_cites, list) else [raw_cites])]
        else:
            continue
        # Models often cite inline ("... [S1]", "<S2>...</S2>") instead of, or as well as, in the
        # list.  Inline ids are citations too; they are validated like any other reference.
        cites = [c for c in listed if _SOURCE_ID.fullmatch(c)] + _INLINE_ID.findall(text)
        text = " ".join(_INLINE_TAG.sub(" ", text).split())
        if _HAS_CONTENT.search(text):  # a placeholder such as "..." is not a claim
            claims.append(Claim(text=text, citations=list(dict.fromkeys(cites))))
    return claims


def build_generator(config: GenerationConfig, client: LLMClient | None = None) -> Generator:
    if config.backend == "extractive":
        return ExtractiveGenerator(config)
    client = client or build_llm_client(config)
    assert client is not None
    return LLMGenerator(config, client)
