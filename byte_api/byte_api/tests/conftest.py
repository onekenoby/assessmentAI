from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api.routes_byte as routes
from main import create_app


def sample_job():
    return {
        "job_id": str(uuid4()),
        "job_type": "CONTENT_INGESTION",
        "status": "PENDING",
        "priority": 100,
        "available_at": datetime.now(UTC).isoformat(),
    }


def sample_corpus_response(**overrides):
    value = {
        "mode": "corpus",
        "filename": "test.pdf",
        "mime_type": "application/pdf",
        "sha256": "a" * 64,
        "file_size_bytes": 8,
        "tier": "C",
        "scope": "ACCOUNT",
        "organization_id": 9999,
        "ontology_id": 2,
        "ontology_code": "identify__risk_assessment",
        "ontology_label": "IDENTIFY / Risk Assessment",
        "file_blob_id": str(uuid4()),
        "document_id": str(uuid4()),
        "document_created": True,
        "document_context_id": str(uuid4()),
        "context_created": True,
        "jobs": [sample_job()],
    }
    value.update(overrides)
    return value


def sample_evidence_response(**overrides):
    value = {
        "mode": "evidence",
        "filename": "evidence.pdf",
        "mime_type": "application/pdf",
        "sha256": "b" * 64,
        "file_size_bytes": 9,
        "file_blob_id": str(uuid4()),
        "file_asset_id": str(uuid4()),
        "attachment_id": str(uuid4()),
        "document_id": str(uuid4()),
        "document_context_id": str(uuid4()),
        "jobs": [sample_job()],
    }
    value.update(overrides)
    return value


class FakeRouteService:
    def __init__(self):
        self.corpus_result = sample_corpus_response()
        self.evidence_result = sample_evidence_response()
        self.corpus_payloads = []
        self.evidence_payloads = []
        self.corpus_error = None
        self.evidence_error = None
        self.health = {
            "service": "byte-api",
            "state": "ready",
            "ready": True,
            "initialized": True,
            "active_uploads": 0,
            "startup_error": "",
            "dependencies": {
                "postgres_source": {"ready": True, "detail": "ok"}
            },
            "checked_at": datetime.now(UTC),
        }

    def upload_corpus(self, payload):
        self.corpus_payloads.append(payload)
        if self.corpus_error:
            raise self.corpus_error
        return self.corpus_result

    def upload_evidence(self, payload):
        self.evidence_payloads.append(payload)
        if self.evidence_error:
            raise self.evidence_error
        return self.evidence_result

    def health_snapshot(self, *, deep=False):
        result = dict(self.health)
        if not deep:
            result["dependencies"] = {}
        return result


@pytest.fixture
def route_service(monkeypatch):
    fake = FakeRouteService()
    monkeypatch.setattr(routes, "service", fake)
    return fake


@pytest.fixture
def client(route_service):
    return TestClient(create_app())


@pytest.fixture
def pdf_file():
    return {"file": ("test.pdf", b"%PDF-1.4", "application/pdf")}
