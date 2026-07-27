"""Contratti JSON pubblici della Ingestion API."""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from core.config import settings


def utc_now() -> datetime:
    return datetime.now(UTC)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class ServiceState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class IngestionRunRequest(ApiModel):
    max_jobs: int = Field(
        default=settings.default_max_jobs,
        ge=1,
        le=settings.max_jobs_per_run,
        description="Numero massimo di job PENDING da reclamare dal Database A.",
    )


class JobExecutionResult(ApiModel):
    job_id: str | None = None
    status: str
    job_type: str | None = None
    document_id: str | None = None
    chunks: int = 0
    processing_time_ms: int = 0
    ontology_codes: list[str] = Field(default_factory=list)
    next_status: str | None = None
    error: str | None = None


class IngestionRunResponse(ApiModel):
    run_id: UUID
    state: RunState
    requested_max_jobs: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    jobs_claimed: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    queue_empty: bool | None = None
    processing_time_ms: int = 0
    jobs: list[JobExecutionResult] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class RunListResponse(ApiModel):
    items: list[IngestionRunResponse]


class DependencyHealth(ApiModel):
    ready: bool
    detail: str


class HealthResponse(ApiModel):
    service: str
    state: ServiceState
    ready: bool
    active_run_id: UUID | None = None
    initialized: bool = False
    startup_error: str = ""
    dependencies: dict[str, DependencyHealth] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=utc_now)


class ApiError(ApiModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
