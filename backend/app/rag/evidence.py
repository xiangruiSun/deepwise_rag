"""EvidenceBundle and the deterministic, token-budgeted context builder (spec 17.1).

Every piece of evidence is a reference to an authoritative record (report,
clinical event, imaging study) with an explicit source marker.  Nothing that
fails authorization is ever placed in a bundle, and generated text (e.g. HyDE)
can never become evidence because bundles are built from database records only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.config import GenerationConfig
from app.core.hashing import sha256_text
from app.core.security import AccessPolicy
from app.retrieval.features import CaseContext

_TOKEN = re.compile(r"\w+|[^\w\s]")
_MARKER = re.compile(r"\[(S\d+)\]")


def estimate_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def relative_time(event_time: datetime, procedure_time: datetime) -> str:
    """Time relative to the procedure.  Absolute dates are patient-specifically shifted
    and must not be compared across patients, so only offsets are shown."""
    hours = (event_time - procedure_time).total_seconds() / 3600.0
    direction = "before" if hours <= 0 else "after"
    magnitude = abs(hours)
    if magnitude < 48:
        return f"{magnitude:.1f} h {direction} procedure"
    return f"{magnitude / 24.0:.0f} d {direction} procedure"


def neutralize(text: str) -> str:
    """Retrieved clinical text is untrusted input: it must not be able to forge citation
    markers or break out of its evidence block."""
    return _MARKER.sub(r"(\1)", text).replace("<<<", "‹‹‹").replace(">>>", "›››")


@dataclass
class EvidenceItem:
    source_id: str
    record_type: str  # report | laboratory | diagnosis | imaging
    record_id: str
    case_id: str
    role: str  # query | retrieved
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    tokens: int = 0


@dataclass
class EvidenceBundle:
    question: str
    query: dict[str, Any]
    query_case_id: str | None
    retrieved_case_ids: list[str]
    items: list[EvidenceItem] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    token_count: int = 0
    truncated: bool = False

    def by_source_id(self) -> dict[str, EvidenceItem]:
        return {item.source_id: item for item in self.items}

    def case_items(self, case_id: str, record_type: str | None = None) -> list[EvidenceItem]:
        return [i for i in self.items
                if i.case_id == case_id and (record_type is None or i.record_type == record_type)]

    @property
    def all_case_ids(self) -> list[str]:
        return ([self.query_case_id] if self.query_case_id else []) + self.retrieved_case_ids

    def render(self) -> str:
        """Context text given to a generator, with explicit source markers."""
        blocks: list[str] = []
        for case_id in self.all_case_ids:
            role = "QUERY CASE" if case_id == self.query_case_id else "RETRIEVED CASE"
            rank = self.retrieval.get("ranks", {}).get(case_id)
            header = f"=== {role} {case_id}" + (f" (retrieval rank {rank})" if rank else "") + " ==="
            lines = [header]
            for item in self.case_items(case_id):
                lines.append(f"[{item.source_id}] ({item.record_type}) <<<{item.text}>>>")
            for gap in self.missing.get(case_id, []):
                lines.append(f"UNAVAILABLE: {gap}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


class EvidenceBuilder:
    """Round-robin over cases in rank order, highest-priority evidence first, so the
    budget is spread across cases instead of spent on one of them."""

    PRIORITY = ("report", "laboratory", "diagnosis", "imaging")

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def build(
        self,
        question: str,
        query: dict[str, Any],
        query_ctx: CaseContext | None,
        retrieved: list[CaseContext],
        policy: AccessPolicy,
        retrieval: dict[str, Any] | None = None,
        matched_documents: dict[str, list[str]] | None = None,
    ) -> EvidenceBundle:
        matched_documents = matched_documents or {}
        contexts: list[tuple[str, CaseContext]] = []
        if query_ctx is not None and policy.allows(query_ctx.case.access_scope):
            contexts.append(("query", query_ctx))
        for ctx in retrieved[: self.config.max_cases]:
            if policy.allows(ctx.case.access_scope):  # never include unauthorized records
                contexts.append(("retrieved", ctx))
        bundle = EvidenceBundle(
            question=question,
            query=query,
            query_case_id=contexts[0][1].case.case_id if contexts and contexts[0][0] == "query" else None,
            retrieved_case_ids=[c.case.case_id for role, c in contexts if role == "retrieved"],
            retrieval=retrieval or {},
        )
        queues: list[list[EvidenceItem]] = []
        for role, ctx in contexts:
            queues.append(self._candidates(role, ctx, matched_documents.get(ctx.case.case_id, [])))
            gaps = self._missing(ctx)
            if gaps:
                bundle.missing[ctx.case.case_id] = gaps

        budget = self.config.context_token_budget
        used = 0
        counter = 0
        taken = [0] * len(queues)
        progressed = True
        while progressed:
            progressed = False
            for qi, queue in enumerate(queues):
                if taken[qi] >= self.config.max_evidence_per_case or taken[qi] >= len(queue):
                    continue
                item = queue[taken[qi]]
                if used + item.tokens > budget:
                    bundle.truncated = True
                    taken[qi] = len(queue)  # stop this case; smaller later items stay out for determinism
                    continue
                counter += 1
                item.source_id = f"S{counter}"
                bundle.items.append(item)
                used += item.tokens
                taken[qi] += 1
                progressed = True
        # stable presentation order: by case, then by evidence priority
        order = {cid: i for i, cid in enumerate(bundle.all_case_ids)}
        bundle.items.sort(key=lambda i: (order[i.case_id], int(i.source_id[1:])))
        bundle.token_count = used
        if bundle.truncated:
            bundle.warnings.append("Evidence was truncated to fit the context token budget.")
        return bundle

    # ------------------------------------------------------------------ helpers
    def _item(self, role: str, ctx: CaseContext, record_type: str, record_id: str, text: str,
              data: dict[str, Any], raw_for_hash: str | None = None) -> EvidenceItem:
        safe = neutralize(text)
        return EvidenceItem(
            source_id="", record_type=record_type, record_id=record_id, case_id=ctx.case.case_id,
            role=role, text=safe, data=data, content_hash=sha256_text(raw_for_hash or text),
            tokens=estimate_tokens(safe) + 6,
        )

    def _candidates(self, role: str, ctx: CaseContext, matched_reports: list[str]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        seen_text: set[str] = set()
        reports = sorted(ctx.reports, key=lambda r: (r.report_id not in matched_reports, r.report_id))
        for report in reports:
            if report.impression_text in seen_text:  # do not repeat overlapping/duplicate text
                continue
            seen_text.add(report.impression_text)
            items.append(self._item(role, ctx, "report", report.report_id,
                                    f"Radiology impression: {report.impression_text}",
                                    {"report_id": report.report_id}, raw_for_hash=report.impression_text))
        if ctx.labs is not None:
            for key in sorted(ctx.labs.features):
                f = ctx.labs.features[key]
                unit = f.original_unit or ""
                text = (f"{f.display_name}: {f.original_value:g} {unit} "
                        f"({relative_time(f.event_time, ctx.case.procedure_time)})").replace("  ", " ")
                items.append(self._item(role, ctx, "laboratory", f.event_id, text, {
                    "concept_key": key, "original_value": f.original_value, "original_unit": f.original_unit,
                    "canonical_value": f.value, "canonical_unit": f.unit, "aggregate": f.aggregate,
                }))
        for event in sorted(ctx.diagnosis_events, key=lambda e: (e.event_time, e.event_id))[-6:]:
            text = (f"Documented diagnosis code {event.concept_code}"
                    + (f" ({event.display_name})" if event.display_name else "")
                    + f", recorded {relative_time(event.event_time, ctx.case.procedure_time)}")
            items.append(self._item(role, ctx, "diagnosis", event.event_id, text,
                                    {"concept_code": event.concept_code}))
        for study in ctx.studies:
            text = (f"Imaging study {study.study_id}: modality {study.modality or 'unknown'}, "
                    f"{study.slice_count or 'unknown'} slices, feature status {study.feature_status}. "
                    "No image findings can be derived from this reference.")
            items.append(self._item(role, ctx, "imaging", study.study_id, text, {"study_id": study.study_id}))
        rank = {t: i for i, t in enumerate(self.PRIORITY)}
        items.sort(key=lambda i: rank[i.record_type])  # stable: keeps within-type order
        return items

    def _missing(self, ctx: CaseContext) -> list[str]:
        gaps: list[str] = []
        if not ctx.reports:
            gaps.append("no radiology impression is available")
        if ctx.labs is None or not ctx.labs.features:
            gaps.append("no comparable laboratory measurements in the lookback window")
        if ctx.labs is not None:
            for key, reason in sorted(ctx.labs.unavailable.items()):
                gaps.append(f"laboratory '{key}' is not comparable ({reason})")
        if not ctx.diagnosis_events:
            gaps.append("no diagnosis codes documented before the procedure")
        if not any(s.feature_status != "missing_source" for s in ctx.studies):
            gaps.append("no imaging study is available")
        return gaps
