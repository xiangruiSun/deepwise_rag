"""RAG and comparison contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.search import RESEARCH_NOTICE, SearchRequest


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source_id: str  # marker used in the answer, e.g. "S3"
    record_type: Literal["report", "laboratory", "diagnosis", "imaging"]
    record_id: str  # authoritative record: report id / event id / study id
    case_id: str
    role: Literal["query", "retrieved"]
    excerpt: str


class RagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    search: SearchRequest | None = None
    query_case_id: str | None = None
    case_ids: list[str] = Field(default_factory=list, max_length=20)
    generation_key: str | None = None

    @model_validator(mode="after")
    def _source(self) -> RagRequest:
        if self.search is None and not self.case_ids:
            raise ValueError("provide either 'search' (retrieve first) or explicit 'case_ids'")
        return self


class RagAnswer(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    summary: list[Claim] = Field(default_factory=list)
    similarities: list[Claim] = Field(default_factory=list)
    differences: list[Claim] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    retrieved_case_ids: list[str] = Field(default_factory=list)
    notice: str = RESEARCH_NOTICE

    def claims(self) -> list[Claim]:
        return [*self.summary, *self.similarities, *self.differences]


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_ids: list[str] = Field(min_length=2, max_length=8)
    lookback_days: int | None = Field(default=None, ge=1, le=3650)


class CaseComparisonColumn(BaseModel):
    case_id: str
    demographics: dict[str, Any]
    laboratory: list[dict[str, Any]]
    laboratory_unavailable: dict[str, str]
    diagnoses: list[dict[str, Any]]
    impressions: list[dict[str, Any]]
    imaging: list[dict[str, Any]]
    # recorded AFTER the procedure — retrospective follow-up, never query-time features
    follow_up: list[dict[str, Any]]


class CompareResponse(BaseModel):
    cases: list[CaseComparisonColumn]
    shared_diagnosis_codes: list[str]
    comparable_laboratory_concepts: list[str]
    lookback_days: int
    warnings: list[str] = Field(default_factory=list)
    notice: str = RESEARCH_NOTICE
