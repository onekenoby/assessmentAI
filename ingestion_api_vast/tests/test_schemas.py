from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api.schemas import HealthResponse, IngestionRunRequest, IngestionRunResponse


def test_run_request_forbids_tenant_and_file_fields():
    for field, value in {
        "organization_id": 8888,
        "scope": "ACCOUNT",
        "tier": "C",
        "tenant_key": "ORG:8888",
        "file": "document.pdf",
    }.items():
        with pytest.raises(ValidationError):
            IngestionRunRequest.model_validate({"max_jobs": 1, field: value})


def test_run_response_rejects_unknown_state():
    with pytest.raises(ValidationError):
        IngestionRunResponse.model_validate(
            {
                "run_id": uuid4(),
                "state": "cancelled",
                "requested_max_jobs": 1,
                "created_at": datetime.now(UTC),
            }
        )


def test_health_response_validates_dependency_contract():
    model = HealthResponse.model_validate(
        {
            "service": "ingestion-api",
            "state": "ready",
            "ready": True,
            "initialized": True,
            "dependencies": {
                "postgres_source": {"ready": True, "detail": "ok"}
            },
        }
    )
    assert model.dependencies["postgres_source"].ready is True


def test_models_forbid_extra_output_fields():
    with pytest.raises(ValidationError):
        HealthResponse.model_validate(
            {
                "service": "ingestion-api",
                "state": "ready",
                "ready": True,
                "secret": "must-not-leak",
            }
        )
