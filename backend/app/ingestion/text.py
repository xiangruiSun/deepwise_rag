"""Radiology text normalisation (spec section 7, stage D).

Only whitespace and encoding are normalised.  Negations, measurements,
laterality and anatomical terms are never removed or rewritten.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")

# Terms whose presence/absence must survive cleaning; used by ``meaning_preserved``.
PROTECTED_TERMS = (
    "no",
    "not",
    "without",
    "negative",
    "absent",
    "acute",
    "chronic",
    "left",
    "right",
    "bilateral",
    "segmental",
    "subsegmental",
    "normal",
    "abnormal",
)
_WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


def clean_impression(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n", text)
    return text.strip()


def content_tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def meaning_preserved(original: str, cleaned: str) -> bool:
    """Cleaning invariant: the multiset of word/number tokens is unchanged."""
    return sorted(content_tokens(unicodedata.normalize("NFKC", original))) == sorted(
        content_tokens(cleaned)
    )


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(])|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences.  Numbered list markers ("1.") stay attached."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENT_SPLIT.finditer(text):
        end = match.start()
        segment = text[start:end]
        # do not split right after an enumeration marker such as "1." or "2)"
        if re.fullmatch(r"\s*\(?\d{1,2}[.)]?", segment):
            continue
        if segment.strip():
            spans.append((start + (len(segment) - len(segment.lstrip())), end))
        start = match.end()
    tail = text[start:]
    if tail.strip():
        spans.append((start + (len(tail) - len(tail.lstrip())), start + len(tail.rstrip())))
    return spans


_SECTION = re.compile(r"(?m)^(?P<name>[A-Z][A-Z /&-]{2,40}):\s*")


def section_spans(text: str) -> list[tuple[str, int, int]]:
    """Recognised ``SECTION:`` blocks as ``(name, start, end)``; the whole text is one
    ``IMPRESSION`` section when no headers are present (the common INSPECT case)."""
    matches = list(_SECTION.finditer(text))
    if not matches:
        return [("IMPRESSION", 0, len(text))] if text.strip() else []
    spans: list[tuple[str, int, int]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        spans.append(("PREAMBLE", 0, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((match.group("name").strip(), match.start(), end))
    return spans
