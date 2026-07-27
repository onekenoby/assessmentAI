from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api.schemas import (
    CorpusUploadResponse,
    EvidenceUploadResponse,
    HealthResponse,
    JobResponse,
)
from tests.conftest import sample_corpus_response, sample_evidence_response


def test_job_response_accepts_uuid_and_datetime():
    model = JobResponse(
        job_id=uuid4(),
        job_type="CONTENT_INGESTION",
        status="PENDING",
        priority=100,
        available_at=datetime.now(UTC),
    )
    assert model.priority == 100


def test_corpus_response_validates_sample():
    model = CorpusUploadResponse.model_validate(sample_corpus_response())
    assert model.mode == "corpus"
    assert model.scope == "ACCOUNT"


def test_corpus_response_forbids_extra_fields():
    value = sample_corpus_response(unexpected=True)
    with pytest.raises(ValidationError):
        CorpusUploadResponse.model_validate(value)


def test_corpus_response_rejects_wrong_mode():
    with pytest.raises(ValidationError):
        CorpusUploadResponse.model_validate(sample_corpus_response(mode="evidence"))


def test_evidence_response_preserves_extra_fields():
    model = EvidenceUploadResponse.model_validate(sample_evidence_response(custom_id=123))
    assert model.model_extra["custom_id"] == 123


def test_evidence_response_rejects_wrong_mode():
    with pytest.raises(ValidationError):
        EvidenceUploadResponse.model_validate(sample_evidence_response(mode="corpus"))


def test_health_defaults_checked_at():
    model = HealthResponse(
        service="byte-api",
        state="ready",
        ready=True,
    )
    assert model.checked_at.tzinfo is not None


def test_health_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        HealthResponse(service="x", state="ready", ready=True, unknown=True)
