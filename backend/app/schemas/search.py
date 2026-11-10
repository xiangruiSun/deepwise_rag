"""Search request / response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESEARCH_NOTICE = (
    "Research prototype on retrospective data. Not for diagnosis, treatment decisions or any "
    "clinical use. Scores are ranking signals, not clinical probabilities."
)


class LabFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept: str = Field(description="Laboratory concept key, e.g. 'd_dimer'")
    min_value: float | None = Field(default=None, description="Inclusive lower bound (canonical unit)")
    max_value: float | None = Field(default=None, description="Inclusive upper bound (canonical unit)")

    @model_validator(mode="after")
    def _bounds(self) -> LabFilter:
        if self.min_value is None and self.max_value is None:
            raise ValueError("a laboratory filter needs min_value and/or max_value")
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("min_value must not exceed max_value")
        return self


class SearchRequest(BaseModel):
    """Note: there is deliberately no user / permission field.  Identity and scope come
    from the server-side authorization layer only.  Research labels (PE status,
    outcomes) are deliberately not filterable."""

    model_config = ConfigDict(extra="forbid")

    query_mode: Literal["case", "text", "filter"]
    case_id: str | None = None
    query_text: str | None = Field(default=None, max_length=4000)
    age_min: float | None = Field(default=None, ge=0, le=130)
    age_max: float | None = Field(default=None, ge=0, le=130)
    gender: str | None = Field(default=None, max_length=32)
    diagnosis_codes: list[str] = Field(default_factory=list, max_length=20)
    lab_filters: list[LabFilter] = Field(default_factory=list, max_length=10)
    study_modality: str | None = Field(default=None, max_length=16)
    candidate_splits: list[Literal["train", "valid", "test"]] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=100_000)
    retrieval_recipe_id: str | None = None
    exclude_same_patient: bool = True
    as_of_time: datetime | None = None

    @model_validator(mode="after")
    def _mode_requirements(self) -> SearchRequest:
        if self.query_mode == "case" and not self.case_id:
            raise ValueError("query_mode 'case' requires case_id")
        if self.query_mode == "text" and not (self.query_text and self.query_text.strip()):
            raise ValueError("query_mode 'text' requires a non-empty query_text")
        if self.age_min is not None and self.age_max is not None and self.age_min > self.age_max:
            raise ValueError("age_min must not exceed age_max")
        return self


class ComponentScore(BaseModel):
    name: str
    value: float | None
    rank: int | None = None
    available: bool = True
    score_type: str
    description: str


class EvidenceRef(BaseModel):
    document_id: str
    source_type: str
    source_id: str
    chunk_index: int = 0
    excerpt: str
    matched_text: str | None = None
    branch_scores: dict[str, float] = Field(default_factory=dict)
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    rerank_score: float | None = None


class LabComparisonItem(BaseModel):
    concept_key: str
    display_name: str
    unit: str
    query_value: float | None = None
    candidate_value: float | None = None
    standardized_difference: float | None = None
    candidate_event_id: str | None = None
    query_event_id: str | None = None


class CaseResult(BaseModel):
    rank: int
    case_id: str
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    dataset_split: str | None = None
    study_ids: list[str] = Field(default_factory=list)
    study_modality: list[str] = Field(default_factory=list)
    final_score: float | None = None
    score_type: str
    components: list[ComponentScore] = Field(default_factory=list)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    laboratory_comparison: list[LabComparisonItem] = Field(default_factory=list)
    laboratory_values: list[dict[str, Any]] = Field(default_factory=list)
    laboratory_note: str | None = None
    shared_diagnosis_codes: list[str] = Field(default_factory=list)
    modality_coverage: dict[str, bool] = Field(default_factory=dict)
    missing_modalities: list[str] = Field(default_factory=list)
    why_retrieved: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    request_id: str
    query_id: str
    query_mode: str
    recipe_id: str
    recipe_hash: str
    recipe_summary: dict[str, Any]
    total_eligible_cases: int | None
    results: list[CaseResult]
    score_descriptions: dict[str, str]
    modality_coverage: dict[str, float] = Field(default_factory=dict)
    latency_ms: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    query_enhancement: dict[str, Any] | None = None
    notice: str = RESEARCH_NOTICE


SCORE_DESCRIPTIONS = {
    "bm25": "Okapi BM25 lexical relevance of the report text (unbounded, higher is better).",
    "dense": "Embedding similarity between query and report text (cosine or inner product).",
    "rrf": "Reciprocal Rank Fusion of ranked lists: sum of weight/(k+rank). Rank-based, unitless.",
    "weighted_score": "Weighted sum of per-branch min-max normalised scores in [0, 1].",
    "rerank": "Cross-encoder relevance logit for (query, report text). Not a probability.",
    "text": "Case-level text relevance after aggregating document scores per case.",
    "laboratory": "exp(-robust-scaled mean absolute difference) over overlapping laboratory "
                  "measurements, in (0, 1]. Not a probability of the same diagnosis.",
    "image": "Cosine similarity of experimental study-level CT representations. Not a validated "
             "statement about lesion similarity.",
    "diagnosis": "Jaccard overlap of diagnosis codes documented at or before the procedure.",
    "multimodal_rank": "Weighted RRF across modality rankings (missing-modality policy applied).",
    "multimodal_score": "Weighted sum of per-modality min-max normalised scores.",
    "none": "Filter-only retrieval: results are not scored.",
}
