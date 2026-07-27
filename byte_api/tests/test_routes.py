from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import api.routes_byte as routes
import byte_engine
from core.service import ServiceBusyError, ServiceClosedError


def corpus_data(**overrides):
    data = {
        "tier": "C",
        "organization_id": "9999",
        "user_id": "123",
        "area": "IDENTIFY",
        "subarea": "Risk Assessment",
    }
    data.update(overrides)
    return data


def test_corpus_upload_returns_201(client, route_service, pdf_file):
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 201
    assert response.json()["mode"] == "corpus"
    payload = route_service.corpus_payloads[0]
    assert payload.tier == "C"
    assert payload.organization_id == 9999
    assert payload.file.data == b"%PDF-1.4"


def test_corpus_maps_all_optional_fields(client, route_service, pdf_file):
    response = client.post(
        "/api/v1/byte/corpus",
        data=corpus_data(
            ontology_code="custom__code",
            ontology_label="Custom Label",
            classification="confidential",
            pipeline_version="v2",
            corpus_version="2026-07",
            embedding_model="bge-m3",
            mime_type="application/x-pdf",
        ),
        files=pdf_file,
    )
    assert response.status_code == 201
    payload = route_service.corpus_payloads[0]
    assert payload.ontology_code == "custom__code"
    assert payload.classification == "confidential"
    assert payload.embedding_model == "bge-m3"
    assert payload.file.mime_type == "application/x-pdf"


def test_evidence_upload_returns_201(client, route_service, pdf_file):
    response = client.post(
        "/api/v1/byte/evidence",
        data={
            "organization_id": "9999",
            "user_id": "123",
            "assessment_id": "10",
            "response_id": "20",
            "encryption_required": "false",
        },
        files=pdf_file,
    )
    assert response.status_code == 201
    assert response.json()["mode"] == "evidence"
    payload = route_service.evidence_payloads[0]
    assert payload.assessment_id == 10
    assert payload.encryption_required is False


@pytest.mark.parametrize(
    "path,data",
    [
        ("/api/v1/byte/corpus", {}),
        ("/api/v1/byte/evidence", {"organization_id": "1"}),
    ],
)
def test_missing_required_form_fields_return_422(client, pdf_file, path, data):
    response = client.post(path, data=data, files=pdf_file)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_missing_file_returns_422(client):
    response = client.post("/api/v1/byte/corpus", data=corpus_data())
    assert response.status_code == 422


def test_engine_upload_validation_maps_to_422(client, route_service, pdf_file):
    route_service.corpus_error = byte_engine.UploadError("tier non valido")
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 422
    assert response.json()["code"] == "upload_validation_error"


def test_database_error_maps_to_503_without_details(client, route_service, pdf_file):
    route_service.corpus_error = byte_engine.DatabaseOperationError("password=secret")
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 503
    assert response.json()["code"] == "database_unavailable"
    assert "secret" not in str(response.json())


def test_busy_maps_to_409(client, route_service, pdf_file):
    route_service.corpus_error = ServiceBusyError("busy")
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 409
    assert response.json()["code"] == "service_busy"


def test_closed_maps_to_503(client, route_service, pdf_file):
    route_service.corpus_error = ServiceClosedError("closed")
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 503
    assert response.json()["code"] == "service_closed"


def test_file_too_large_maps_to_413(client, pdf_file, monkeypatch):
    monkeypatch.setattr(
        routes,
        "settings",
        SimpleNamespace(
            max_file_bytes=2,
            api_key="",
            api_key_header="X-Byte-Api-Key",
            expose_error_details=False,
        ),
    )
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_api_key_missing_wrong_and_valid(client, route_service, monkeypatch):
    monkeypatch.setattr(
        routes,
        "settings",
        SimpleNamespace(
            max_file_bytes=100,
            api_key="top-secret",
            api_key_header="X-Byte-Api-Key",
            expose_error_details=False,
        ),
    )
    missing = client.get("/health/live")
    # Health non è protetta.
    assert missing.status_code == 200

    bad = client.post(
        "/api/v1/byte/corpus",
        data=corpus_data(),
        files={"file": ("x.pdf", b"x", "application/pdf")},
        headers={"X-Byte-Api-Key": "wrong"},
    )
    good = client.post(
        "/api/v1/byte/corpus",
        data=corpus_data(),
        files={"file": ("x.pdf", b"x", "application/pdf")},
        headers={"X-Byte-Api-Key": "top-secret"},
    )
    assert bad.status_code == 401
    assert bad.headers["WWW-Authenticate"] == "ApiKey"
    assert good.status_code == 201


def test_valid_request_id_is_preserved(client):
    request_id = str(uuid4())
    response = client.get("/health/live", headers={"X-Request-ID": request_id})
    assert response.headers["X-Request-ID"] == request_id


def test_invalid_request_id_is_replaced(client):
    response = client.get("/health/live", headers={"X-Request-ID": "bad"})
    assert str(UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


@pytest.mark.parametrize("path", ["/health/live", "/health/ready?deep=true"])
def test_security_headers(client, path):
    response = client.get(path)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_live_is_process_liveness(client, route_service):
    route_service.health.update({"state": "failed", "ready": False})
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_ready_returns_503_when_db_fails(client, route_service):
    route_service.health.update(
        {
            "state": "failed",
            "ready": False,
            "dependencies": {"postgres_source": {"ready": False, "detail": "offline"}},
        }
    )
    response = client.get("/health/ready?deep=true")
    assert response.status_code == 503


def test_ready_returns_200(client):
    response = client.get("/health/ready?deep=true")
    assert response.status_code == 200


def test_openapi_exposes_only_byte_upload_and_health(client):
    response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/api/v1/byte/corpus" in paths
    assert "/api/v1/byte/evidence" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/api/v1/ingestion/runs" not in paths


def test_database_error_can_expose_details_when_enabled(client, route_service, pdf_file, monkeypatch):
    route_service.corpus_error = byte_engine.DatabaseOperationError("diagnostic")
    monkeypatch.setattr(
        routes,
        "settings",
        SimpleNamespace(
            max_file_bytes=100,
            api_key="",
            api_key_header="X-Byte-Api-Key",
            expose_error_details=True,
        ),
    )
    response = client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
    assert response.status_code == 503
    assert response.json()["details"]["error"] == "diagnostic"


def test_evidence_database_error_maps_to_503(client, route_service, pdf_file):
    route_service.evidence_error = byte_engine.DatabaseDependencyError("driver missing")
    response = client.post(
        "/api/v1/byte/evidence",
        data={
            "organization_id": "1",
            "user_id": "2",
            "assessment_id": "3",
            "response_id": "4",
        },
        files=pdf_file,
    )
    assert response.status_code == 503
    assert response.json()["code"] == "database_unavailable"


def test_unknown_service_exception_is_not_misclassified(client, route_service, pdf_file):
    route_service.corpus_error = RuntimeError("unexpected")
    with pytest.raises(RuntimeError, match="unexpected"):
        client.post("/api/v1/byte/corpus", data=corpus_data(), files=pdf_file)
