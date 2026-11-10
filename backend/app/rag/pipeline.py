"""RAG pipeline: scope guard → retrieval → evidence → generation → verification.

The pipeline never asks a model to compensate for missing evidence: with no
authorized cases, no citable evidence, or a question outside the supported
scope it returns an explicit ``insufficient_evidence`` answer.
"""

from __future__ import annotations

import re
import time
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import AppConfig, GenerationConfig
from app.core.errors import NotFoundError
from app.core.security import AccessPolicy
from app.observability.metrics import ACCESS_DENIED, INFERENCE_TOKENS, StageTimer
from app.rag.citations import CitationVerifier
from app.rag.evidence import EvidenceBuilder, EvidenceBundle
from app.rag.generators import Generator, LLMClient, build_generator
from app.retrieval.features import CaseFeatureService
from app.retrieval.service import RetrievalService
from app.schemas.rag import (
    CaseComparisonColumn,
    CompareRequest,
    CompareResponse,
    RagAnswer,
    RagRequest,
)
from app.schemas.search import SearchResponse

# Requests the system must not answer, whatever the evidence (spec 2.3 / 17.5).
_SCOPE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Treatment recommendations are outside the scope of this research system.",
     re.compile(r"\b(should (be|we|i) (given|give|treat|prescrib\w*|start\w*)|what (dose|dosage|treatment|therapy)"
                r"|which (\w+ )?(dose|drug|medication|anticoagulant|treatment|therapy)|recommend\w*|prescrib\w*)\b",
                re.IGNORECASE)),
    ("The system does not make or suggest diagnoses.",
     re.compile(r"\b(does (this|the) patient have|diagnose|what is the diagnosis|is this (a )?(pe|pulmonary embol))",
                re.IGNORECASE)),
    ("Risk or outcome prediction is outside the scope of this research system.",
     re.compile(r"\b(risk of|probability of|chance of|prognosis|will (this|the) patient|predict\w*|likelihood)\b",
                re.IGNORECASE)),
    ("Geographic residence is not available in the dataset.",
     re.compile(r"\b(zip ?code|postal code|address|county|city|neighbou?rhood|where (does|do) .* live|"
                r"resid\w+|rural|urban)\b", re.IGNORECASE)),
]
_IMAGING_QUESTION = re.compile(r"\b(image|imaging|ct|scan|lesion|clot burden|volume|density)\b", re.IGNORECASE)


def scope_guard(question: str) -> list[str]:
    """Reasons why (parts of) a question cannot be answered from this system's evidence."""
    return [reason for reason, pattern in _SCOPE_RULES if pattern.search(question)]


class RagPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        retrieval: RetrievalService,
        config: AppConfig,
        generation_variants: dict[str, GenerationConfig] | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.retrieval = retrieval
        self.config = config
        self.variants = {config.generation.key: config.generation, **(generation_variants or {})}
        self._llm_client = llm_client
        self._generators: dict[str, Generator] = {}

    def generator(self, key: str | None) -> Generator:
        key = key or self.config.generation.key
        if key not in self.variants:
            raise NotFoundError(f"unknown generation configuration '{key}'",
                                {"available": sorted(self.variants)})
        if key not in self._generators:
            self._generators[key] = build_generator(self.variants[key], self._llm_client)
        return self._generators[key]

    # ------------------------------------------------------------------ answer
    def answer(self, request: RagRequest, policy: AccessPolicy,
               search_response: SearchResponse | None = None) -> RagAnswer:
        """``search_response`` lets evaluations freeze retrieval while comparing generators."""
        timer = StageTimer()
        generator = self.generator(request.generation_key)
        gen_cfg = generator.config
        reasons = scope_guard(request.question)
        if reasons:
            return self._insufficient(reasons, gen_cfg, timer, ["Question is outside the supported scope."])

        with timer.stage("retrieval"):
            if search_response is None and request.search is not None:
                search_response = self.retrieval.search(request.search, policy)
        case_ids = (
            [r.case_id for r in search_response.results] if search_response is not None else list(request.case_ids)
        )
        query_case_id = request.query_case_id or (
            request.search.case_id if request.search and request.search.query_mode == "case" else None
        )
        if not case_ids:
            reasons = ["No authorized cases satisfy the request, so there is no evidence to summarise."]
            if search_response is not None:
                reasons.extend(search_response.warnings)
            return self._insufficient(reasons, gen_cfg, timer, [])

        with self.session_factory() as session:
            with timer.stage("context_assembly"):
                bundle = self._bundle(session, request, policy, gen_cfg, case_ids, query_case_id, search_response)
            if not bundle.items:
                return self._insufficient(
                    ["No citable evidence is available for the selected cases."], gen_cfg, timer, bundle.warnings
                )
            with timer.stage("generation"):
                answer = generator.generate(bundle)
            with timer.stage("citation_verification"):
                report = CitationVerifier(session, policy).verify(
                    answer, bundle, enforce=gen_cfg.post_verification,
                    require_citations=gen_cfg.require_citations,
                )
        if report.removed_claims:
            answer.warnings.append(
                f"{report.removed_claims} generated statement(s) were removed because they had no valid "
                "citation or were not supported by the cited evidence."
            )
        if _IMAGING_QUESTION.search(request.question):
            answer.missing_evidence.append(
                "Imaging-specific conclusions are limited to documented radiology impressions; "
                "image-vector similarity does not support statements about findings."
            )
        if answer.status == "answered" and not answer.claims():
            answer.status = "insufficient_evidence"
            answer.missing_evidence.append("No supported, citable statement could be produced.")
        answer.warnings.extend(bundle.warnings)
        answer.retrieved_case_ids = bundle.retrieved_case_ids
        answer.limitations.append(
            "Retrospective research data; not for diagnosis or treatment decisions."
        )
        meta = answer.generation_metadata
        INFERENCE_TOKENS.labels(direction="input").inc(float(meta.get("input_tokens", 0) or 0))
        INFERENCE_TOKENS.labels(direction="output").inc(float(meta.get("output_tokens", 0) or 0))
        answer.generation_metadata = {
            **meta,
            "generation_key": gen_cfg.key,
            "post_verification": gen_cfg.post_verification,
            "citation_report": report.as_dict(),
            "evidence_items": len(bundle.items),
            "context_tokens": bundle.token_count,
            "context_truncated": bundle.truncated,
            "recipe_id": search_response.recipe_id if search_response else None,
            "latency_ms": timer.finish(),
            "generation_latency_ms": meta.get("latency_ms"),
        }
        return answer

    def build_bundle(self, request: RagRequest, policy: AccessPolicy,
                     search_response: SearchResponse | None = None) -> EvidenceBundle | None:
        """Expose the constructed evidence (used by the evaluation framework)."""
        generator = self.generator(request.generation_key)
        case_ids = (
            [r.case_id for r in search_response.results] if search_response is not None else list(request.case_ids)
        )
        if not case_ids:
            return None
        query_case_id = request.query_case_id or (
            request.search.case_id if request.search and request.search.query_mode == "case" else None
        )
        with self.session_factory() as session:
            return self._bundle(session, request, policy, generator.config, case_ids, query_case_id, search_response)

    def _bundle(self, session: Session, request: RagRequest, policy: AccessPolicy, gen_cfg: GenerationConfig,
                case_ids: list[str], query_case_id: str | None,
                search_response: SearchResponse | None) -> EvidenceBundle:
        features = CaseFeatureService(session)
        lab_cfg = self.config.recipes[self.config.default_recipe].multimodal.laboratory
        lookback = lab_cfg.lookback_days
        wanted = [c for c in dict.fromkeys(case_ids) if c != query_case_id][: gen_cfg.max_cases]
        contexts = features.contexts(wanted + ([query_case_id] if query_case_id else []), lookback, lab_cfg.aggregate)
        missing = [c for c in wanted if c not in contexts]
        if missing:
            raise NotFoundError("one or more cases do not exist", {"case_ids": missing})
        for ctx in contexts.values():
            if not policy.allows(ctx.case.access_scope):
                ACCESS_DENIED.labels(resource="rag_context").inc()
            policy.assert_case_allowed(ctx.case.access_scope, ctx.case.case_id)
        matched: dict[str, list[str]] = {}
        retrieval_meta: dict[str, Any] = {}
        if search_response is not None:
            matched = {r.case_id: [e.source_id for e in r.evidence] for r in search_response.results}
            retrieval_meta = {
                "recipe_id": search_response.recipe_id,
                "recipe_hash": search_response.recipe_hash,
                "ranks": {r.case_id: r.rank for r in search_response.results},
            }
        query_meta = {
            "mode": request.search.query_mode if request.search else "explicit_cases",
            "query_text": request.search.query_text if request.search else None,
            "query_case_id": query_case_id,
        }
        return EvidenceBuilder(gen_cfg).build(
            request.question, query_meta, contexts.get(query_case_id) if query_case_id else None,
            [contexts[c] for c in wanted], policy, retrieval_meta, matched,
        )

    def _insufficient(self, reasons: list[str], gen_cfg: GenerationConfig, timer: StageTimer,
                      warnings: list[str]) -> RagAnswer:
        return RagAnswer(
            status="insufficient_evidence",
            missing_evidence=reasons,
            limitations=["No statement was generated because the available evidence cannot support one."],
            warnings=warnings,
            generation_metadata={
                "generation_key": gen_cfg.key, "backend": gen_cfg.backend, "model": gen_cfg.model,
                "abstained_before_generation": True, "input_tokens": 0, "output_tokens": 0,
                "latency_ms": timer.finish(), "generated_at": time.time(),
            },
        )


class ComparisonService:
    """Side-by-side structured comparison of selected cases (no LLM involved)."""

    def __init__(self, session_factory: sessionmaker[Session], config: AppConfig) -> None:
        self.session_factory = session_factory
        self.config = config

    def compare(self, request: CompareRequest, policy: AccessPolicy) -> CompareResponse:
        lookback = request.lookback_days or self.config.dataset.default_lookback_days
        warnings: list[str] = []
        with self.session_factory() as session:
            features = CaseFeatureService(session)
            contexts = features.contexts(list(dict.fromkeys(request.case_ids)), lookback)
            missing = [c for c in request.case_ids if c not in contexts]
            if missing:
                raise NotFoundError("one or more cases do not exist", {"case_ids": missing})
            columns: list[CaseComparisonColumn] = []
            for case_id in dict.fromkeys(request.case_ids):
                ctx = contexts[case_id]
                if not policy.allows(ctx.case.access_scope):
                    ACCESS_DENIED.labels(resource="compare").inc()
                policy.assert_case_allowed(ctx.case.access_scope, case_id)
                labs = ctx.labs
                follow_up = features.follow_up_events(ctx.case, limit=50)
                columns.append(
                    CaseComparisonColumn(
                        case_id=case_id,
                        demographics={
                            "age": ctx.age,
                            "gender": ctx.patient.gender if ctx.patient else None,
                            "race": ctx.patient.race if ctx.patient else None,
                            "ethnicity": ctx.patient.ethnicity if ctx.patient else None,
                            "dataset_split": ctx.case.dataset_split,
                        },
                        laboratory=[f.as_dict() for _, f in sorted(labs.features.items())] if labs else [],
                        laboratory_unavailable=dict(labs.unavailable) if labs else {},
                        diagnoses=[
                            {"event_id": e.event_id, "concept_code": e.concept_code,
                             "display_name": e.display_name,
                             "hours_from_procedure": round(
                                 (e.event_time - ctx.case.procedure_time).total_seconds() / 3600.0, 1)}
                            for e in ctx.diagnosis_events
                        ],
                        impressions=[
                            {"report_id": r.report_id, "impression_text": r.impression_text} for r in ctx.reports
                        ],
                        imaging=[
                            {"study_id": s.study_id, "modality": s.modality, "slice_count": s.slice_count,
                             "feature_status": s.feature_status, "available": s.feature_status != "missing_source"}
                            for s in ctx.studies
                        ],
                        follow_up=[
                            {"event_id": e.event_id, "event_type": e.event_type, "concept_code": e.concept_code,
                             "numeric_value": e.numeric_value, "unit": e.unit,
                             "hours_after_procedure": round(
                                 (e.event_time - ctx.case.procedure_time).total_seconds() / 3600.0, 1)}
                            for e in follow_up
                        ],
                    )
                )
        code_sets = [{d["concept_code"] for d in c.diagnoses} for c in columns]
        lab_sets = [{lab["concept_key"] for lab in c.laboratory} for c in columns]
        if any(not s for s in lab_sets):
            warnings.append("At least one case has no comparable laboratory measurement in the lookback window.")
        return CompareResponse(
            cases=columns,
            shared_diagnosis_codes=sorted(set.intersection(*code_sets)) if code_sets else [],
            comparable_laboratory_concepts=sorted(set.intersection(*lab_sets)) if lab_sets else [],
            lookback_days=lookback,
            warnings=warnings,
        )
