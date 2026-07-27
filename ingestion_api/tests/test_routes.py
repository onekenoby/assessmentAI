from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import api.routes_ingestion as routes
from core.service import RunNotFoundError, ServiceBusyError, ServiceClosedError
from tests.conftest import sample_record


def test_create_run_returns_202(client, route_service):
    response = client.post("/api/v1/ingestion/runs", json={"max_jobs": 1})

    assert response.status_code == 202
    assert response.json()["run_id"] == str(route_service.record["run_id"])
    assert route_service.submitted_max_jobs == [1]
    assert str(UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


def test_create_run_uses_default_max_jobs(client, route_service):
    response = client.post("/api/v1/ingestion/runs", json={})
    assert response.status_code == 202
    assert route_service.submitted_max_jobs == [1]


@pytest.mark.parametrize("body", [{"max_jobs": 0}, {"max_jobs": 101}, {"max_jobs": 1, "scope": "GLOBAL"}])
def test_validation_rejects_invalid_body(client, body):
    response = client.post("/api/v1/ingestion/runs", json=body)
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert payload["details"]["errors"]


def test_validation_rejects_malformed_json(client):
    response = client.post(
        "/api/v1/ingestion/runs",
        content="{",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_busy_service_returns_409(client, route_service):
    route_service.submit_error = ServiceBusyError("Esecuzione già attiva: abc")
    response = client.post("/api/v1/ingestion/runs", json={"max_jobs": 1})
    assert response.status_code == 409
    assert response.json()["code"] == "service_busy"


def test_closed_service_returns_503(client, route_service):
    route_service.submit_error = ServiceClosedError("closed")
    response = client.post("/api/v1/ingestion/runs", json={"max_jobs": 1})
    assert response.status_code == 503
    assert response.json()["code"] == "service_closed"
    assert "closed" not in response.json()["message"].lower()


def test_get_run_returns_record(client, route_service):
    run_id = uuid4()
    response = client.get(f"/api/v1/ingestion/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == str(run_id)


def test_get_run_not_found_returns_404(client, route_service):
    route_service.get_error = RunNotFoundError("missing")
    response = client.get(f"/api/v1/ingestion/runs/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["code"] == "run_not_found"


def test_invalid_run_uuid_is_validation_error(client):
    response = client.get("/api/v1/ingestion/runs/not-a-uuid")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_list_runs_is_wrapped_in_items(client, route_service):
    route_service.records = [sample_record(state="succeeded"), sample_record(state="failed")]
    response = client.get("/api/v1/ingestion/runs")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2


def test_api_key_authentication_missing_wrong_and_valid(client, route_service, monkeypatch):
    monkeypatch.setattr(
        routes,
        "settings",
        SimpleNamespace(api_key="top-secret", api_key_header="X-Ingestion-Api-Key"),
    )

    missing = client.get("/api/v1/ingestion/runs")
    wrong = client.get(
        "/api/v1/ingestion/runs",
        headers={"X-Ingestion-Api-Key": "wrong"},
    )
    valid = client.get(
        "/api/v1/ingestion/runs",
        headers={"X-Ingestion-Api-Key": "top-secret"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert valid.status_code == 200
    assert missing.headers["WWW-Authenticate"] == "ApiKey"
    assert missing.json()["code"] == "authentication_error"


def test_valid_request_id_is_preserved(client):
    request_id = str(uuid4())
    response = client.get("/health/live", headers={"X-Request-ID": request_id})
    assert response.headers["X-Request-ID"] == request_id


def test_invalid_request_id_is_replaced(client):
    response = client.get("/health/live", headers={"X-Request-ID": "invalid"})
    generated = response.headers["X-Request-ID"]
    assert generated != "invalid"
    assert str(UUID(generated)) == generated


@pytest.mark.parametrize(
    "path",
    [
        "/health/live",
        "/health/ready?deep=false",
        "/api/v1/ingestion/runs",
    ],
)
def test_security_headers_are_always_present(client, path):
    response = client.get(path)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_live_is_200_even_when_service_not_ready(client, route_service):
    route_service.health_snapshot.update(
        {"state": "failed", "ready": False, "initialized": False}
    )
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["state"] == "failed"


def test_live_reports_closed_as_not_live(client, route_service):
    route_service.health_snapshot.update({"state": "closed", "ready": False})
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_ready_returns_503_when_dependencies_are_not_ready(client, route_service):
    route_service.health_snapshot.update(
        {
            "state": "failed",
            "ready": False,
            "dependencies": {
                "postgres_source": {"ready": False, "detail": "offline"}
            },
        }
    )
    response = client.get("/health/ready?deep=true")
    assert response.status_code == 503
    assert response.json()["dependencies"]["postgres_source"]["ready"] is False


def test_ready_returns_200_when_service_ready(client):
    response = client.get("/health/ready?deep=true")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_openapi_exposes_expected_endpoints_and_safe_request_contract(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/api/v1/ingestion/runs" in paths
    assert "/api/v1/ingestion/runs/{run_id}" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths

    request_schema = schema["components"]["schemas"]["IngestionRunRequest"]
    assert set(request_schema["properties"]) == {"max_jobs"}
    serialized = str(request_schema).lower()
    for forbidden in ("organization_id", "tenant_key", "scope", "tier", "bytea", "file"):
        assert forbidden not in serialized
