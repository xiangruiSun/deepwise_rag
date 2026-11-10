"""Canonical document and chunk contracts shared by every ingestion and serving path."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.hashing import sha256_text

SCHEMA_VERSION = "corpus-doc-1"
PUBLIC_GROUP = "public"


def make_document_id(tenant_id: str, source_system: str, source_record_id: str) -> str:
    """Deterministic identity of a logical document: retries and re-ingestion never duplicate it."""
    return "doc_" + sha256_text(f"{tenant_id}|{source_system}|{source_record_id}")[:24]


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Stable across document versions, so a new version overwrites its predecessor's chunks."""
    return f"{document_id}-{chunk_index:05d}"


def to_epoch(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


class CanonicalDocument(BaseModel):
    """One version of one source document after normalisation."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    document_id: str
    source_system: str
    source_record_id: str
    document_type: str
    title: str = ""
    content: str
    content_hash: str
    source_updated_at: datetime | None = None
    ingested_at: datetime
    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    document_version: int = Field(ge=1)
    schema_version: str = SCHEMA_VERSION
    # groups allowed to read the document inside its tenant
    access_policy: list[str] = Field(default_factory=lambda: [PUBLIC_GROUP])
    source_uri: str = ""
    is_deleted: bool = False
    # domain-specific fields (patient pseudonym, encounter, codes, department, ...)
    extension: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    chunk_text: str
    token_count: int
    section_path: str | None = None
    chunking_version: str
    embedding_model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # set for child chunks of parent-document retrieval
    parent_chunk_index: int | None = None
