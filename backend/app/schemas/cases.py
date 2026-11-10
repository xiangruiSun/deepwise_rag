"""Case, experiment and admin response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.search import RESEARCH_NOTICE


class ErrorResponse(BaseModel):
    error: dict[str, Any]


class CaseSummary(BaseModel):
    case_id: str
    age: float | None = None
    gender: str | None = None
    dataset_split: str | None = None
    has_report: bool = False
    has_image: bool = False


class CaseListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CaseSummary]
    notice: str = RESEARCH_NOTICE


class CaseOverview(BaseModel):
    case_id: str
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    dataset_split: str | None = None
    source_version: str | None = None
    report_ids: list[str] = Field(default_factory=list)
    study_ids: list[str] = Field(default_factory=list)
    n_events_before_procedure: int = 0
    n_events_after_procedure: int = 0
    laboratory: list[dict[str, Any]] = Field(default_factory=list)
    laboratory_unavailable: dict[str, str] = Field(default_factory=dict)
    diagnosis_codes: list[str] = Field(default_factory=list)
    modality_coverage: dict[str, bool] = Field(default_factory=dict)
    notice: str = RESEARCH_NOTICE


class TimelineEvent(BaseModel):
    event_id: str
    event_type: str
    concept_code: str
    code_system: str | None = None
    display_name: str | None = None
    numeric_value: float | None = None
    text_value: str | None = None
    unit: str | None = None
    # Absolute dates are patient-specifically shifted; offsets are the meaningful quantity.
    hours_from_procedure: float
    phase: str  # "before_procedure" | "follow_up"


class TimelineResponse(BaseModel):
    case_id: str
    total: int
    limit: int
    offset: int
    events: list[TimelineEvent]


class ReportOut(BaseModel):
    report_id: str
    case_id: str
    study_id: str | None = None
    impression_text: str
    content_hash: str


class ImageOut(BaseModel):
    study_id: str
    case_id: str
    modality: str | None = None
    slice_count: int | None = None
    spacing: list[float] | None = None
    feature_status: str
    preview_available: bool
    note: str = "Image-derived similarity is experimental and carries no clinical findings."


class ExperimentRunOut(BaseModel):
    run_id: str
    suite_id: str | None
    experiment_id: str
    status: str
    stage: str
    data_kind: str
    config_hash: str
    dataset_fingerprint: str | None
    started_at: datetime
    finished_at: datetime | None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


class ExperimentDetail(ExperimentRunOut):
    config: dict[str, Any] = Field(default_factory=dict)
    model_revisions: dict[str, Any] = Field(default_factory=dict)
    index_versions: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    query_set_version: str | None = None
    qrels_version: str | None = None
    metrics_by_group: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ExperimentResults(BaseModel):
    run_id: str
    total: int
    limit: int
    offset: int
    rows: list[dict[str, Any]]


class IndexStatus(BaseModel):
    index_name: str
    kind: str
    alias: str | None = None
    exists: bool
    embedding_key: str | None = None
    embedding_model_id: str | None = None
    embedding_revision: str | None = None
    dimension: int | None = None
    chunking_key: str | None = None
    mapping_version: str | None = None
    document_count: int | None = None
    size_bytes: int | None = None
    health: str | None = None
    built_at: datetime | None = None


class IngestionRequest(BaseModel):
    force: bool = False
    rebuild_indexes: bool = True


class IngestionJob(BaseModel):
    job_id: str
    status: str
    submitted_by: str
    summary: dict[str, Any] | None = None
    error: str | None = None


class MetaResponse(BaseModel):
    """Which filters and recipes are actually supported by the loaded dataset/config."""

    dataset: str
    data_kind: str
    text_label: str = "Radiology impression"
    genders: list[str]
    laboratory_concepts: list[dict[str, str]]
    diagnosis_codes: list[str]
    study_modalities: list[str]
    age_range: list[float] | None
    recipes: list[dict[str, Any]]
    default_recipe: str
    generation_backend: str
    image_modality_enabled: bool
    notice: str = RESEARCH_NOTICE
