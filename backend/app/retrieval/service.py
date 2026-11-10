"""``RetrievalService.search`` — staged, patient-level, multimodal case retrieval.

Stages (each timed separately and independently testable):

 1. query parsing / recipe resolution
 2. query representation (case-based: temporally valid features of the query case)
 3. filter construction — authorization scope, structured criteria, same-patient exclusion
 4. eligibility count on the structured case index
 5. first-stage retrieval: BM25 and/or dense kNN, both PRE-filtered
 6. document-level fusion (RRF or normalised weighted scores)
 7. optional cross-encoder reranking of a bounded pool (text evidence only)
 8. aggregation of documents to distinct cases
 9. optional multimodal case-level fusion (laboratory, image, diagnosis-code similarity)
10. result assembly with evidence references, component scores, coverage and warnings

The service never needs an LLM.  Access control is enforced inside the search
filters and re-checked on the assembled results (defence in depth).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import AppConfig, RetrievalRecipe
from app.core.errors import DependencyUnavailable, NotFoundError, ValidationFailure
from app.core.hashing import stable_hash
from app.core.security import AccessPolicy
from app.embeddings.registry import EmbeddingRegistry
from app.imaging.providers import ImageEmbeddingProvider
from app.indexing.backend import FilterSpec, SearchBackend
from app.indexing.indexer import report_index_name
from app.ingestion.normalization import LabConceptRegistry, normalize_diagnosis_code
from app.observability.metrics import ACCESS_DENIED, StageTimer, get_logger
from app.ranking.aggregation import CaseAggregator, CaseCandidate, RetrievedDocument
from app.ranking.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.ranking.multimodal import MultimodalCaseRanker, MultimodalResult
from app.ranking.rerankers import RerankerRegistry
from app.retrieval.features import CaseContext, CaseFeatureService
from app.retrieval.laboratory import (
    CaseLabFeatures,
    LaboratorySimilarityScorer,
    LabSimilarity,
    RobustStat,
    diagnosis_similarity,
)
from app.retrieval.query_enhancement import EnhancedQuery, QueryEnhancer
from app.retrieval.retrievers import BM25Retriever, DenseRetriever, ImageRetriever, LaboratoryRetriever
from app.schemas.search import (
    SCORE_DESCRIPTIONS,
    CaseResult,
    ComponentScore,
    EvidenceRef,
    LabComparisonItem,
    SearchRequest,
    SearchResponse,
)

log = get_logger("retrieval")
MAX_EXHAUSTIVE_CANDIDATES = 50_000
EXCERPT_CHARS = 600


@dataclass
class _CandidateFeatures:
    patient_id: str
    labs: CaseLabFeatures
    diagnosis_codes: set[str]


@dataclass
class _State:
    """Mutable per-request state passed between stages."""

    request: SearchRequest
    recipe: RetrievalRecipe
    policy: AccessPolicy
    timer: StageTimer
    warnings: list[str] = field(default_factory=list)
    query_ctx: CaseContext | None = None
    query_text: str | None = None
    filters: FilterSpec = field(default_factory=FilterSpec)
    enhanced: EnhancedQuery | None = None
    text_score_type: str = "none"
    lab_details: dict[str, LabSimilarity] = field(default_factory=dict)
    multimodal: dict[str, MultimodalResult] = field(default_factory=dict)


class RetrievalService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        backend: SearchBackend,
        config: AppConfig,
        embeddings: EmbeddingRegistry,
        rerankers: RerankerRegistry | None = None,
        image_provider: ImageEmbeddingProvider | None = None,
        enhancer: QueryEnhancer | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.backend = backend
        self.config = config
        self.embeddings = embeddings
        self.rerankers = rerankers or RerankerRegistry()
        self.image_provider = image_provider
        self.enhancer = enhancer or QueryEnhancer()
        self.registry = LabConceptRegistry()
        self._feature_cache: dict[tuple[int, str], dict[str, _CandidateFeatures]] = {}
        self._image_cache: dict[str, np.ndarray] = {}
        self._stats_cache: dict[tuple[int, str], dict[str, RobustStat]] = {}

    # ------------------------------------------------------------------ public
    def clear_caches(self) -> None:
        self._feature_cache.clear()
        self._image_cache.clear()
        self._stats_cache.clear()

    def recipe(self, recipe_id: str | None) -> RetrievalRecipe:
        key = recipe_id or self.config.default_recipe
        try:
            return self.config.recipes[key]
        except KeyError as exc:
            raise ValidationFailure(
                f"unknown retrieval_recipe_id '{key}'", {"available": sorted(self.config.recipes)}
            ) from exc

    def search(
        self,
        request: SearchRequest,
        policy: AccessPolicy,
        recipe: RetrievalRecipe | None = None,
        query_id: str | None = None,
    ) -> SearchResponse:
        timer = StageTimer()
        with timer.stage("query_parsing"):
            recipe = recipe or self.recipe(request.retrieval_recipe_id)
            state = _State(request=request, recipe=recipe, policy=policy, timer=timer)
        with self.session_factory() as session:
            features = CaseFeatureService(session, self.registry)
            self._query_representation(state, features)
            with timer.stage("filter_construction"):
                state.filters = self._build_filters(state)
            with timer.stage("eligibility"):
                cases_index = self.config.index.cases_index
                if not self.backend.index_exists(cases_index):
                    raise DependencyUnavailable(f"case index '{cases_index}' has not been built")
                total = self.backend.count(cases_index, state.filters)

            results: list[CaseResult] = []
            if total == 0:
                state.warnings.append("No authorized cases satisfy the requested filters.")
            elif request.query_mode == "filter":
                results = self._filter_only(state, features)
            else:
                candidates, branch_docs = self._text_candidates(state)
                ranked = self._rank_cases(state, features, candidates)
                with timer.stage("result_assembly"):
                    results = self._assemble(state, features, ranked, candidates)
                if not results and not state.warnings:
                    state.warnings.append("No relevant retrievable evidence was found for this query.")
                del branch_docs

        coverage: dict[str, float] = {}
        if results:
            for modality in sorted({m for r in results for m in r.modality_coverage}):
                coverage[modality] = round(
                    sum(1 for r in results if r.modality_coverage.get(modality)) / len(results), 4
                )
        return SearchResponse(
            request_id=str(uuid.uuid4()),
            query_id=query_id or stable_hash(request.model_dump(mode="json"), 16),
            query_mode=request.query_mode,
            recipe_id=recipe.recipe_id,
            recipe_hash=stable_hash(recipe.model_dump(mode="json"), 16),
            recipe_summary=self._recipe_summary(recipe),
            total_eligible_cases=total,
            results=results,
            score_descriptions={k: v for k, v in SCORE_DESCRIPTIONS.items()
                                if k in self._score_types(state, results)},
            modality_coverage=coverage,
            latency_ms=timer.finish(),
            warnings=state.warnings,
            query_enhancement=state.enhanced.as_dict() if state.enhanced else None,
        )

    # ------------------------------------------------- stage 2: query features
    def _query_representation(self, state: _State, features: CaseFeatureService) -> None:
        request, lab_cfg = state.request, state.recipe.multimodal.laboratory
        with state.timer.stage("query_representation"):
            if request.query_mode == "case":
                assert request.case_id is not None
                contexts = features.contexts(
                    [request.case_id], lab_cfg.lookback_days, lab_cfg.aggregate, as_of=request.as_of_time
                )
                ctx = contexts.get(request.case_id)
                if ctx is None:
                    raise NotFoundError("query case not found", {"case_id": request.case_id})
                if not state.policy.allows(ctx.case.access_scope):
                    ACCESS_DENIED.labels(resource="search_query_case").inc()
                state.policy.assert_case_allowed(ctx.case.access_scope, request.case_id)
                state.query_ctx = ctx
                # Retrospective, report-available task: the query case's finalised impression
                # may be used.  Research labels and post-procedure events are never used.
                text = "\n".join(r.impression_text for r in ctx.reports).strip()
                state.query_text = text or None
                if state.query_text is None:
                    state.warnings.append(
                        "The query case has no radiology impression; text retrieval is unavailable."
                    )
            elif request.query_mode == "text":
                state.query_text = (request.query_text or "").strip()

    # ------------------------------------------------------- stage 3: filters
    def _build_filters(self, state: _State) -> FilterSpec:
        request = state.request
        spec = FilterSpec()
        # authorization scope always comes from the server-side principal
        spec.terms["access_scope"] = state.policy.allowed_scopes
        if request.age_min is not None or request.age_max is not None:
            spec.ranges["age"] = (request.age_min, request.age_max)
        if request.gender:
            spec.terms["gender"] = [request.gender]
        if request.diagnosis_codes:
            spec.all_terms["diagnosis_codes"] = [normalize_diagnosis_code(c) for c in request.diagnosis_codes]
        for lab in request.lab_filters:
            if self.registry.by_key(lab.concept) is None:
                raise ValidationFailure(
                    f"unknown laboratory concept '{lab.concept}'", {"available": self.registry.keys()}
                )
            spec.ranges[f"lab_{lab.concept}"] = (lab.min_value, lab.max_value)
        if request.study_modality:
            spec.terms["study_modality"] = [request.study_modality]
        if request.candidate_splits:
            spec.terms["dataset_split"] = list(request.candidate_splits)
        if state.query_ctx is not None:
            spec.must_not["case_id"] = [state.query_ctx.case.case_id]
            if request.exclude_same_patient:
                spec.must_not["patient_id"] = [state.query_ctx.case.patient_id]
        return spec

    # ---------------------------------------------------- filter-only retrieval
    def _filter_only(self, state: _State, features: CaseFeatureService) -> list[CaseResult]:
        request = state.request
        with state.timer.stage("structured_retrieval"):
            hits, _total = self.backend.filter_search(
                self.config.index.cases_index, state.filters, request.top_k, request.offset
            )
        lab_cfg = state.recipe.multimodal.laboratory
        with state.timer.stage("result_assembly"):
            contexts = features.contexts([h.id for h in hits], lab_cfg.lookback_days, lab_cfg.aggregate)
            results = []
            for position, hit in enumerate(hits, start=1):
                ctx = contexts.get(hit.id)
                if ctx is None or not state.policy.allows(ctx.case.access_scope):
                    continue
                result = self._base_result(ctx, request.offset + position, None, "none")
                result.why_retrieved = ["Matches all requested structured filters (not scored)."]
                results.append(result)
        return results

    # ------------------------------------- stages 5-8: text retrieval pipeline
    def _text_candidates(
        self, state: _State
    ) -> tuple[list[CaseCandidate], dict[str, list[RetrievedDocument]]]:
        recipe, timer = state.recipe, state.timer
        if not recipe.branches:
            if state.request.query_mode == "text":
                raise ValidationFailure(
                    f"recipe '{recipe.recipe_id}' has no text retriever and cannot answer free-text queries"
                )
            return [], {}
        if not state.query_text:
            return [], {}
        prefix = self.config.index.report_index_prefix
        enhanced = self.enhancer.enhance(state.query_text, recipe.query_enhancement)
        state.warnings.extend(enhanced.warnings)
        if recipe.query_enhancement.method != "none":
            state.enhanced = enhanced

        branch_docs: dict[str, list[RetrievedDocument]] = {}
        if "bm25" in recipe.branches:
            retriever = BM25Retriever(self.backend, report_index_name(prefix, None, recipe.chunking))
            with timer.stage("bm25"):
                branch_docs["bm25"] = retriever.retrieve(enhanced.lexical_query, state.filters, recipe.bm25_top_k)
        if "dense" in recipe.branches:
            assert recipe.embedding is not None
            provider = self.embeddings.provider(recipe.embedding)
            dense = DenseRetriever(
                self.backend, report_index_name(prefix, recipe.embedding, recipe.chunking),
                self.embeddings.embedder(recipe.embedding),
            )
            as_document = enhanced.dense_as_document or (
                state.request.query_mode == "case" and recipe.case_query_encoding == "document"
            )
            before = provider.truncation["documents_truncated" if as_document else "queries_truncated"]
            with timer.stage("query_embedding"):
                vector = dense.encode(enhanced.dense_query, as_document=as_document)
            after = provider.truncation["documents_truncated" if as_document else "queries_truncated"]
            if after > before:
                state.warnings.append(
                    "The query text exceeded the encoder's input limit and was truncated for dense retrieval."
                )
            with timer.stage("vector_search"):
                branch_docs["dense"] = dense.retrieve(
                    vector, state.filters, recipe.dense_top_k, recipe.dense_num_candidates
                )

        with timer.stage("fusion"):
            documents = self._fuse_documents(state, branch_docs)
        with timer.stage("reranking"):
            documents = self._rerank(state, documents)
        with timer.stage("aggregation"):
            exclude_patients: set[str] = set()
            exclude_cases: set[str] = set()
            if state.query_ctx is not None:
                exclude_cases.add(state.query_ctx.case.case_id)
                if state.request.exclude_same_patient:
                    exclude_patients.add(state.query_ctx.case.patient_id)
            aggregator = CaseAggregator(recipe.aggregation)
            candidates = aggregator.aggregate(documents, branch_docs, exclude_patients, exclude_cases)
        return candidates, branch_docs

    def _fuse_documents(
        self, state: _State, branch_docs: dict[str, list[RetrievedDocument]]
    ) -> list[RetrievedDocument]:
        """Document-level fusion.  (Case-level fusion happens in ``CaseAggregator``.)"""
        recipe = state.recipe
        active = dict(branch_docs.items())
        if len(active) == 1:
            name, docs = next(iter(active.items()))
            state.text_score_type = name
            return docs[: recipe.fused_top_k]
        by_id: dict[str, RetrievedDocument] = {}
        for docs in active.values():
            for doc in docs:
                merged = by_id.setdefault(doc.document_id, doc)
                if merged is not doc:
                    merged.branch_scores.update(doc.branch_scores)
                    merged.branch_ranks.update(doc.branch_ranks)
        if recipe.fusion.method == "rrf":
            fused = reciprocal_rank_fusion(
                {name: [d.document_id for d in docs] for name, docs in active.items()},
                k=recipe.fusion.rrf_k, weights=recipe.fusion.weights,
            )
            state.text_score_type = "rrf"
        else:
            fused = weighted_score_fusion(
                {name: {d.document_id: d.branch_scores[name] for d in docs} for name, docs in active.items()},
                weights=recipe.fusion.weights, normalization=recipe.fusion.normalization,
            )
            state.text_score_type = "weighted_score"
        out: list[RetrievedDocument] = []
        for rank, item in enumerate(fused[: recipe.fused_top_k], start=1):
            doc = by_id[item.item_id]
            doc.score, doc.score_type, doc.rank = item.score, state.text_score_type, rank
            out.append(doc)
        return out

    def _rerank(self, state: _State, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        config = state.recipe.reranker
        reranker = self.rerankers.get(config)
        if reranker is None or not documents or not state.query_text:
            return documents
        pool = documents[: config.depth]  # bounded candidate pool
        scores = reranker.score(state.query_text, [d.content for d in pool])
        for doc, score in zip(pool, scores, strict=True):
            doc.rerank_score = float(score)
        pool.sort(key=lambda d: (-(d.rerank_score or 0.0), d.rank, d.document_id))
        for rank, doc in enumerate(pool, start=1):
            doc.score, doc.score_type, doc.rank = float(doc.rerank_score or 0.0), "rerank", rank
        state.text_score_type = "rerank"
        return pool

    # ------------------------------------------- stage 9: case-level ranking
    def _lab_stats(self, features: CaseFeatureService, lookback: int, aggregate: str) -> dict[str, RobustStat]:
        key = (lookback, aggregate)
        if key not in self._stats_cache:
            self._stats_cache[key] = features.load_lab_statistics(
                self.config.dataset.corpus_split, lookback, aggregate
            )
        return self._stats_cache[key]

    def _candidate_features(
        self, features: CaseFeatureService, case_ids: list[str], lookback: int, aggregate: str
    ) -> dict[str, _CandidateFeatures]:
        cache = self._feature_cache.setdefault((lookback, aggregate), {})
        missing = [c for c in case_ids if c not in cache]
        for start in range(0, len(missing), 1000):
            contexts = features.contexts(missing[start : start + 1000], lookback, aggregate, with_reports=False)
            for case_id, ctx in contexts.items():
                assert ctx.labs is not None
                cache[case_id] = _CandidateFeatures(ctx.case.patient_id, ctx.labs, set(ctx.diagnosis_codes))
        return {c: cache[c] for c in case_ids if c in cache}

    def _image_vectors(self, features: CaseFeatureService, case_ids: list[str]) -> dict[str, np.ndarray]:
        provider = self.image_provider
        if provider is None or not provider.enabled:
            return {}
        missing = [c for c in case_ids if c not in self._image_cache]
        if missing:
            self._image_cache.update(features.image_vectors(missing, provider.provider_id, provider.version))
        return {c: self._image_cache[c] for c in case_ids if c in self._image_cache}

    def _eligible_case_ids(self, state: _State) -> list[str]:
        hits, total = self.backend.filter_search(
            self.config.index.cases_index, state.filters, MAX_EXHAUSTIVE_CANDIDATES, 0
        )
        if total > len(hits):
            state.warnings.append(
                f"Exhaustive structured candidate generation was capped at {MAX_EXHAUSTIVE_CANDIDATES} cases."
            )
        return [h.id for h in hits]

    def _rank_cases(
        self, state: _State, features: CaseFeatureService, candidates: list[CaseCandidate]
    ) -> list[tuple[str, float | None, str]]:
        """Returns ``[(case_id, final_score, score_type)]`` in final order."""
        recipe, mm, timer = state.recipe, state.recipe.multimodal, state.timer
        if not mm.enabled:
            return [(c.case_id, c.text_score, state.text_score_type) for c in candidates]
        query_ctx = state.query_ctx
        if query_ctx is None:
            if any(m != "text" for m in mm.modalities):
                state.warnings.append(
                    "Free-text queries carry no laboratory, imaging or diagnosis features; "
                    "only text relevance was used."
                )
            return [(c.case_id, c.text_score, state.text_score_type) for c in candidates]

        lab_cfg = mm.laboratory
        text_scores = {c.case_id: c.text_score for c in candidates}
        pool: dict[str, None] = dict.fromkeys(
            c.case_id for c in candidates[: mm.candidate_depth]
        ) if "text" in mm.candidate_sources else {}
        use_lab = "laboratory" in mm.modalities
        use_image = "image" in mm.modalities
        use_dx = "diagnosis" in mm.modalities
        scorer: LaboratorySimilarityScorer | None = None
        eligible: list[str] | None = None

        if use_lab:
            stats = self._lab_stats(features, lab_cfg.lookback_days, lab_cfg.aggregate)
            if not stats:
                state.warnings.append(
                    "Laboratory normalisation statistics are not fitted; laboratory similarity is unavailable."
                )
            scorer = LaboratorySimilarityScorer(stats, lab_cfg)
            if query_ctx.labs is None or not query_ctx.labs.features:
                state.warnings.append(
                    "The query case has no comparable laboratory measurements in the lookback window."
                )
            for concept, reason in sorted((query_ctx.labs.unavailable if query_ctx.labs else {}).items()):
                state.warnings.append(f"Query laboratory '{concept}' is not comparable: {reason}.")

        # --- candidate generation from non-text sources (all pre-filtered)
        if "laboratory" in mm.candidate_sources and scorer is not None and query_ctx.labs is not None:
            with timer.stage("laboratory_similarity"):
                eligible = self._eligible_case_ids(state)
                feats = self._candidate_features(features, eligible, lab_cfg.lookback_days, lab_cfg.aggregate)
                retrieved = LaboratoryRetriever(scorer).retrieve(
                    query_ctx.labs, {c: (f.patient_id, f.labs) for c, f in feats.items()}, mm.candidate_depth
                )
                for item in retrieved:
                    pool.setdefault(item.case_id, None)
                    state.lab_details[item.case_id] = item.detail
        query_image: np.ndarray | None = None
        image_scores: dict[str, float] = {}
        if use_image:
            query_image = self._image_vectors(features, [query_ctx.case.case_id]).get(query_ctx.case.case_id)
            retriever = ImageRetriever(self.backend, self.config.index.image_index)
            if self.image_provider is None or not self.image_provider.enabled or not retriever.available():
                state.warnings.append("Image modality is disabled or no image index is available.")
                query_image = None
            elif query_image is None:
                state.warnings.append("The query case has no usable image representation.")
            elif "image" in mm.candidate_sources:
                with timer.stage("image_similarity"):
                    for item in retriever.retrieve(query_image, state.filters, mm.candidate_depth,
                                                   self.config.index.knn_num_candidates):
                        pool.setdefault(item.case_id, None)
                        image_scores[item.case_id] = item.score
        if "diagnosis" in mm.candidate_sources and query_ctx.diagnosis_codes:
            with timer.stage("diagnosis_similarity"):
                eligible = eligible or self._eligible_case_ids(state)
                feats = self._candidate_features(features, eligible, lab_cfg.lookback_days, lab_cfg.aggregate)
                scored = [
                    (c, diagnosis_similarity(query_ctx.diagnosis_codes, f.diagnosis_codes)) for c, f in feats.items()
                ]
                for case_id, _sim in sorted(
                    ((c, s) for c, s in scored if s), key=lambda x: (-x[1], x[0])  # type: ignore[operator]
                )[: mm.candidate_depth]:
                    pool.setdefault(case_id, None)

        # --- score every pooled candidate on every enabled modality
        pool_ids = list(pool)
        feats = self._candidate_features(features, pool_ids, lab_cfg.lookback_days, lab_cfg.aggregate)
        scores: dict[str, dict[str, float | None]] = {c: {} for c in pool_ids if c in feats}
        if use_lab and scorer is not None and query_ctx.labs is not None:
            with timer.stage("laboratory_similarity"):
                for case_id in scores:
                    detail = state.lab_details.get(case_id) or scorer.score(query_ctx.labs, feats[case_id].labs)
                    state.lab_details[case_id] = detail
                    scores[case_id]["laboratory"] = detail.similarity if detail.available else None
        if use_image and query_image is not None:
            with timer.stage("image_similarity"):
                vectors = self._image_vectors(features, list(scores))
                qn = float(np.linalg.norm(query_image)) or 1.0
                for case_id in scores:
                    if case_id in image_scores:
                        scores[case_id]["image"] = image_scores[case_id]
                    elif case_id in vectors:
                        v = vectors[case_id]
                        scores[case_id]["image"] = float(v @ query_image / ((float(np.linalg.norm(v)) or 1.0) * qn))
                    else:
                        scores[case_id]["image"] = None
        for case_id in scores:
            if "text" in mm.modalities:
                scores[case_id]["text"] = text_scores.get(case_id)
            if use_dx:
                scores[case_id]["diagnosis"] = diagnosis_similarity(
                    query_ctx.diagnosis_codes, feats[case_id].diagnosis_codes
                )
            for modality in mm.modalities:
                scores[case_id].setdefault(modality, None)

        with timer.stage("multimodal_fusion"):
            ranked = MultimodalCaseRanker(mm).rank(scores)
        state.multimodal = {r.case_id: r for r in ranked}
        score_type = "multimodal_rank" if mm.family == "rank" else "multimodal_score"
        del recipe
        return [(r.case_id, r.score, score_type) for r in ranked]

    # --------------------------------------------------- stage 10: assembly
    def _base_result(self, ctx: CaseContext, rank: int, score: float | None, score_type: str) -> CaseResult:
        labs = ctx.labs
        return CaseResult(
            rank=rank,
            case_id=ctx.case.case_id,
            age=ctx.age,
            gender=ctx.patient.gender if ctx.patient else None,
            race=ctx.patient.race if ctx.patient else None,
            ethnicity=ctx.patient.ethnicity if ctx.patient else None,
            dataset_split=ctx.case.dataset_split,
            study_ids=[s.study_id for s in ctx.studies],
            study_modality=sorted({s.modality for s in ctx.studies if s.modality}),
            final_score=score,
            score_type=score_type,
            laboratory_values=[f.as_dict() for _, f in sorted(labs.features.items())] if labs else [],
            modality_coverage={
                "text": bool(ctx.reports),
                "laboratory": bool(labs and labs.features),
                "image": any(s.feature_status == "ready" for s in ctx.studies),
                "diagnosis": bool(ctx.diagnosis_codes),
            },
        )

    def _assemble(
        self,
        state: _State,
        features: CaseFeatureService,
        ranked: list[tuple[str, float | None, str]],
        candidates: list[CaseCandidate],
    ) -> list[CaseResult]:
        request, recipe = state.request, state.recipe
        lab_cfg = recipe.multimodal.laboratory
        top = ranked[: request.top_k]
        by_case = {c.case_id: c for c in candidates}
        contexts = features.contexts([c for c, _, _ in top], lab_cfg.lookback_days, lab_cfg.aggregate)
        query_ctx = state.query_ctx
        parent_child = self.config.chunking[recipe.chunking].strategy == "parent_child"
        scorer: LaboratorySimilarityScorer | None = None
        if query_ctx is not None and query_ctx.labs is not None and query_ctx.labs.features:
            scorer = LaboratorySimilarityScorer(
                self._lab_stats(features, lab_cfg.lookback_days, lab_cfg.aggregate), lab_cfg
            )
        results: list[CaseResult] = []
        for case_id, score, score_type in top:
            ctx = contexts.get(case_id)
            if ctx is None:
                continue
            # defence in depth: never return a case outside the caller's scope or the query patient
            if not state.policy.allows(ctx.case.access_scope):
                ACCESS_DENIED.labels(resource="search_result").inc()
                log.error("dropped unauthorized case from results", extra={"case_id": case_id})
                continue
            if query_ctx is not None and (
                case_id == query_ctx.case.case_id
                or (request.exclude_same_patient and ctx.case.patient_id == query_ctx.case.patient_id)
            ):
                continue
            result = self._base_result(ctx, len(results) + 1, score, score_type)
            candidate = by_case.get(case_id)
            reports = {r.report_id: r for r in ctx.reports}
            if candidate is not None:
                for doc in candidate.evidence:
                    parent = reports.get(doc.source_id)
                    full = parent.impression_text if (parent_child and parent is not None) else doc.content
                    result.evidence.append(
                        EvidenceRef(
                            document_id=doc.document_id, source_type=doc.source_type,
                            source_id=doc.source_id, chunk_index=doc.chunk_index,
                            excerpt=full[:EXCERPT_CHARS],
                            matched_text=doc.content[:EXCERPT_CHARS] if parent_child else None,
                            branch_scores={k: round(v, 6) for k, v in doc.branch_scores.items()},
                            branch_ranks=dict(doc.branch_ranks), rerank_score=doc.rerank_score,
                        )
                    )
                result.components.append(
                    ComponentScore(name="text", value=candidate.text_score, rank=candidate.text_rank,
                                   score_type=state.text_score_type,
                                   description=SCORE_DESCRIPTIONS[state.text_score_type])
                )
            self._add_structured_comparison(state, result, ctx, scorer)
            mm = state.multimodal.get(case_id)
            if mm is not None:
                result.effective_weights = {k: round(v, 4) for k, v in mm.effective_weights.items()}
                for modality in recipe.multimodal.modalities:
                    if modality == "text":
                        continue
                    result.components.append(
                        ComponentScore(
                            name=modality, value=mm.component_scores.get(modality),
                            rank=mm.component_ranks.get(modality), available=mm.coverage[modality],
                            score_type=modality, description=SCORE_DESCRIPTIONS[modality],
                        )
                    )
                result.missing_modalities = [m for m, ok in mm.coverage.items() if not ok]
            else:
                result.missing_modalities = [m for m, ok in result.modality_coverage.items() if not ok]
            result.why_retrieved = self._explain(state, result, candidate)
            results.append(result)
        return results

    def _add_structured_comparison(
        self, state: _State, result: CaseResult, ctx: CaseContext, scorer: LaboratorySimilarityScorer | None
    ) -> None:
        query_ctx = state.query_ctx
        if query_ctx is None:
            return
        result.shared_diagnosis_codes = sorted(query_ctx.diagnosis_codes & ctx.diagnosis_codes)
        if scorer is None or ctx.labs is None or query_ctx.labs is None:
            result.laboratory_note = "Laboratory comparison unavailable: the query case has no comparable measurements."
            return
        detail = state.lab_details.get(ctx.case.case_id) or scorer.score(query_ctx.labs, ctx.labs)
        result.laboratory_comparison = [LabComparisonItem(**item) for item in detail.per_concept]
        if not detail.available:
            result.laboratory_note = f"Laboratory similarity unavailable: {detail.reason}."
        else:
            result.laboratory_note = (
                f"{detail.n_comparable} of {detail.n_query_measurements} query measurements comparable "
                f"({detail.coverage:.0%}); similarity {detail.similarity:.3f} (not a probability)."
            )
        unavailable = sorted(ctx.labs.unavailable.items())
        if unavailable:
            result.laboratory_note += " Not comparable for this case: " + "; ".join(
                f"{k} ({v})" for k, v in unavailable
            ) + "."

    def _explain(self, state: _State, result: CaseResult, candidate: CaseCandidate | None) -> list[str]:
        """"Why retrieved?" — built only from actual retrieval evidence and features."""
        lines: list[str] = []
        if candidate is not None and candidate.evidence:
            best = candidate.evidence[0]
            parts = [f"{branch} rank {rank}" for branch, rank in sorted(best.branch_ranks.items())]
            if best.rerank_score is not None:
                parts.append(f"reranker score {best.rerank_score:.3f}")
            lines.append(
                f"Report impression {best.source_id} matched the query text ({', '.join(parts)}); "
                f"case-level text rank {candidate.text_rank}."
            )
        mm = state.multimodal.get(result.case_id)
        if mm is not None:
            if mm.coverage.get("laboratory"):
                closest = sorted(result.laboratory_comparison, key=lambda i: i.standardized_difference or 0.0)[:2]
                detail = "; ".join(
                    f"{i.display_name}: query {i.query_value:g} vs {i.candidate_value:g} {i.unit}" for i in closest
                )
                lines.append(
                    f"Laboratory similarity rank {mm.component_ranks.get('laboratory')} over "
                    f"{len(result.laboratory_comparison)} comparable measurement(s)" + (f" ({detail})." if detail else ".")
                )
            if mm.coverage.get("image"):
                lines.append(
                    f"Image representation similarity rank {mm.component_ranks.get('image')} "
                    "(experimental; not a validated lesion comparison)."
                )
            if mm.coverage.get("diagnosis") and result.shared_diagnosis_codes:
                lines.append("Shares documented pre-procedure diagnosis codes: "
                             + ", ".join(result.shared_diagnosis_codes) + ".")
            missing = [m for m, ok in mm.coverage.items() if not ok]
            if missing:
                lines.append("Unavailable for this case (not used in its score): " + ", ".join(missing) + ".")
        if not lines:
            lines.append("Retrieved by the configured ranking; no further evidence is available.")
        return lines

    # ------------------------------------------------------------------ helpers
    def _score_types(self, state: _State, results: list[CaseResult]) -> set[str]:
        types = {state.text_score_type}
        for result in results:
            types.add(result.score_type)
            types.update(c.score_type for c in result.components)
            for ev in result.evidence:
                types.update(ev.branch_scores)
                if ev.rerank_score is not None:
                    types.add("rerank")
        return types

    def _recipe_summary(self, recipe: RetrievalRecipe) -> dict[str, Any]:
        mm = recipe.multimodal
        return {
            "branches": list(recipe.branches),
            "embedding": recipe.embedding,
            "chunking": recipe.chunking,
            "fusion": recipe.fusion.method if len(recipe.branches) > 1 else None,
            "reranker": recipe.reranker.key if recipe.reranker.enabled else None,
            "rerank_depth": recipe.reranker.depth if recipe.reranker.enabled else None,
            "aggregation": recipe.aggregation.method,
            "multimodal": {
                "enabled": mm.enabled, "modalities": list(mm.modalities), "family": mm.family,
                "weights": {m: mm.weights.get(m) for m in mm.modalities},
                "missing_policy": mm.missing_policy,
            } if mm.enabled else {"enabled": False},
            "query_enhancement": recipe.query_enhancement.method,
        }
