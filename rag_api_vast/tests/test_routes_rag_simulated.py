from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import routes_rag
from core.models import RagServiceResult, RetrievalDebug
from core.tenant import TenantContext


class FakeRagService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def query(self, command, *, tenant_context):
        self.calls.append((command, tenant_context))
        if self.error:
            raise self.error
        return self.result


class FakeHealthSnapshot:
    def __init__(self, ready=True, degraded=False):
        self.ready = ready
        self.degraded = degraded
        self.dependencies = ()


def _app(tenant_context, fake_service):
    app = FastAPI()
    app.include_router(routes_rag.router)
    app.include_router(routes_rag.health_router)
    routes_rag.install_rag_exception_handlers(app)

    async def tenant_override():
        return tenant_context

    app.dependency_overrides[routes_rag.resolve_request_tenant] = tenant_override
    return app


def test_query_endpoint_maps_request_and_response(monkeypatch, tenant_context):
    result = RagServiceResult(
        request_id=UUID(tenant_context.request_id),
        conversation_id="conv-1",
        answer="**A) Risposta**\n\nOK\n\n**B) Evidenze**\n\n-\n\n**C) Limiti / Conflitti**\n\n-\n\n**D) Fonti**\n\n-",
        retrieval=RetrievalDebug(),
    )
    fake = FakeRagService(result=result)
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "Domanda",
                "conversation_id": "conv-1",
                "history": [],
                "options": {"include_sources": True, "include_debug": False},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["answer"].startswith("**A) Risposta**")
    assert response.headers["x-request-id"] == tenant_context.request_id
    command, context = fake.calls[0]
    assert command.query == "Domanda"
    assert context.organization_id == 1234


def test_query_validation_error_uses_uniform_contract(monkeypatch, tenant_context):
    fake = FakeRagService()
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == "validation_error"
    assert body["request_id"]


def test_service_error_is_mapped_to_safe_http_error(monkeypatch, tenant_context):
    from core.rag_service import RagServiceRetrievalError

    fake = FakeRagService(error=RagServiceRetrievalError("backend details"))
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": "Domanda"})

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "retrieval_error"
    assert "backend details" not in body["message"]
    assert body["retryable"] is True


def test_liveness_is_always_ok(tenant_context):
    app = _app(tenant_context, FakeRagService())
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_uses_simulated_snapshot(monkeypatch, tenant_context):
    monkeypatch.setattr(
        routes_rag.resources,
        "health_snapshot",
        lambda deep=False: FakeHealthSnapshot(ready=False),
    )
    app = _app(tenant_context, FakeRagService())
    with TestClient(app) as client:
        response = client.get("/health/ready?deep=true")
    assert response.status_code == 503
    assert response.json()["status"] == "down"
