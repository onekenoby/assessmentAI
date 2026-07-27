"""Schemi pubblici della Byte API."""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ServiceState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class JobResponse(ApiModel):
    job_id: UUID | str
    job_type: str
    status: str
    priority: int
    available_at: datetime | str


class CorpusUploadResponse(ApiModel):
    mode: str = "corpus"
    filename: str
    mime_type: str
    sha256: str
    file_size_bytes: int
    tier: str
    scope: str
    organization_id: int | None
    ontology_id: int
    ontology_code: str
    ontology_label: str
    file_blob_id: UUID | str
    document_id: UUID | str
    document_created: bool
    document_context_id: UUID | str
    context_created: bool
    jobs: list[JobResponse] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def mode_must_be_corpus(cls, value: str) -> str:
        if value != "corpus":
            raise ValueError("mode deve essere corpus")
        return value


class EvidenceUploadResponse(BaseModel):
    """Risposta evidence.

    La funzione PostgreSQL ufficiale può aggiungere campi applicativi nel tempo;
    per questo la risposta conserva campi extra senza perderli.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    mode: str = "evidence"
    filename: str
    mime_type: str
    sha256: str
    file_size_bytes: int
    document_id: UUID | str | None = None
    jobs: list[JobResponse] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def mode_must_be_evidence(cls, value: str) -> str:
        if value != "evidence":
            raise ValueError("mode deve essere evidence")
        return value


class DependencyHealth(ApiModel):
    ready: bool
    detail: str


class HealthResponse(ApiModel):
    service: str
    state: ServiceState
    ready: bool
    initialized: bool = False
    active_uploads: int = 0
    startup_error: str = ""
    dependencies: dict[str, DependencyHealth] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApiError(ApiModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
