"""Canonical relational data model (spec section 6).

The relational store is the authoritative record.  Search indexes hold derived,
versioned copies; typed clinical values (laboratory numbers, units, timestamps)
are never replaced by generated text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    race: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ethnicity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_dataset: Mapped[str] = mapped_column(String(64))
    source_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    cases: Mapped[list[ClinicalCase]] = relationship(back_populates="patient")


class ClinicalCase(Base):
    __tablename__ = "clinical_cases"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"))
    procedure_time: Mapped[datetime] = mapped_column(DateTime)
    # Age at the procedure when the source provides it (nullable; never synthesised).
    age_at_procedure: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_split: Mapped[str | None] = mapped_column(String(16), nullable=True)
    study_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_patient_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_row_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    access_scope: Mapped[str] = mapped_column(String(32), default="research")
    ingestion_status: Mapped[str] = mapped_column(String(32), default="ingested")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    patient: Mapped[Patient] = relationship(back_populates="cases")

    __table_args__ = (
        UniqueConstraint("patient_id", "procedure_time", name="uq_case_patient_time"),
        Index("ix_case_patient", "patient_id"),
        Index("ix_case_study", "study_id"),
        Index("ix_case_split", "dataset_split"),
        Index("ix_case_procedure_time", "procedure_time"),
        Index("ix_case_scope", "access_scope"),
    )


class CaseLabel(Base):
    """Research labels shipped with the dataset (e.g. PE positive, outcomes).

    Kept in a separate table on purpose: they are never exposed as ordinary query
    filters and never enter query-time features (blinded retrieval evaluation)."""

    __tablename__ = "case_labels"

    case_id: Mapped[str] = mapped_column(ForeignKey("clinical_cases.case_id"), primary_key=True)
    label_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    label_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # "diagnostic" labels describe the index study, "outcome" labels describe the future.
    label_kind: Mapped[str] = mapped_column(String(16), default="diagnostic")


class ClinicalEvent(Base):
    __tablename__ = "clinical_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(64))
    case_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32))
    concept_code: Mapped[str] = mapped_column(String(128))
    code_system: Mapped[str | None] = mapped_column(String(32), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    event_time: Mapped[datetime] = mapped_column(DateTime)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    text_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_record_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)

    __table_args__ = (
        Index("ix_event_patient_time", "patient_id", "event_time"),
        Index("ix_event_patient_type", "patient_id", "event_type"),
        Index("ix_event_code", "concept_code"),
        Index("ix_event_source_file", "source_file"),
    )


class RadiologyReport(Base):
    __tablename__ = "radiology_reports"

    report_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("clinical_cases.case_id"))
    patient_id: Mapped[str] = mapped_column(String(64))
    study_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    impression_text: Mapped[str] = mapped_column(Text)
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))

    __table_args__ = (Index("ix_report_case", "case_id"), Index("ix_report_hash", "content_hash"))


class ImagingStudy(Base):
    __tablename__ = "imaging_studies"

    study_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("clinical_cases.case_id"))
    patient_id: Mapped[str] = mapped_column(String(64))
    modality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Path or controlled object-store reference; volumes are never put in a search index.
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    slice_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spacing: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    orientation_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    preprocessing_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feature_status: Mapped[str] = mapped_column(String(32), default="pending")
    feature_error: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (Index("ix_imaging_case", "case_id"),)


class ImageFeature(Base):
    """Versioned study-level image representation (experimental; not a validated
    lesion-similarity embedding)."""

    __tablename__ = "image_features"

    study_id: Mapped[str] = mapped_column(ForeignKey("imaging_studies.study_id"), primary_key=True)
    provider_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    provider_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64))
    dimension: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list[float]] = mapped_column(JSON)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RetrievalDocument(Base):
    __tablename__ = "retrieval_documents"

    document_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64))
    patient_id: Mapped[str] = mapped_column(String(64))
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(128))
    parent_document_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    section: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    # hash of the denormalised filter metadata stored with the indexed document
    metadata_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chunking_strategy: Mapped[str] = mapped_column(String(64))
    chunking_version: Mapped[str] = mapped_column(String(32))
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    embedding_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    index_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    access_scope: Mapped[str] = mapped_column(String(32), default="research")

    __table_args__ = (
        Index("ix_doc_case", "case_id"),
        Index("ix_doc_source", "source_type", "source_id"),
        Index("ix_doc_index_version", "index_version"),
    )


class SourceFileState(Base):
    """Per-source content hashes that drive incremental ingestion."""

    __tablename__ = "source_file_state"

    source_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(128))
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class LabNormalizationStat(Base):
    """Robust scaling statistics fitted on the training corpus only."""

    __tablename__ = "lab_normalization_stats"

    stats_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    concept_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_unit: Mapped[str] = mapped_column(String(32))
    median: Mapped[float] = mapped_column(Float)
    iqr: Mapped[float] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer)
    fitted_split: Mapped[str] = mapped_column(String(16))


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    suite_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    experiment_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running")
    stage: Mapped[str] = mapped_column(String(16), default="validation")
    data_kind: Mapped[str] = mapped_column(String(16), default="synthetic")
    config_hash: Mapped[str] = mapped_column(String(64))
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query_set_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    qrels_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_revisions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    index_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    metrics: Mapped[list[ExperimentMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[ExperimentArtifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_run_experiment", "experiment_id"), Index("ix_run_suite", "suite_id"))


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.run_id"))
    name: Mapped[str] = mapped_column(String(96))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    group: Mapped[str] = mapped_column(String(64), default="overall")

    run: Mapped[ExperimentRun] = relationship(back_populates="metrics")
    __table_args__ = (Index("ix_metric_run", "run_id"),)


class ExperimentArtifact(Base):
    __tablename__ = "experiment_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.run_id"))
    kind: Mapped[str] = mapped_column(String(48))
    path: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Artifacts holding per-query case ids are restricted to the scopes used by the run.
    access_scope: Mapped[str] = mapped_column(String(32), default="research")

    run: Mapped[ExperimentRun] = relationship(back_populates="artifacts")


class IndexManifest(Base):
    __tablename__ = "index_manifests"

    index_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    alias: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))
    embedding_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunking_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunking_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    mapping_version: Mapped[str] = mapped_column(String(32))
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    built_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditEvent(Base):
    """Audit trail.  ``details`` holds identifiers and counts only — never clinical text."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    outcome: Mapped[str] = mapped_column(String(16), default="ok")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_audit_time", "occurred_at"), Index("ix_audit_action", "action"))
