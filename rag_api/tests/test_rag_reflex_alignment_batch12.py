from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import routes_rag
from core.capacity import (
    CapacityRejectedError,
    CapacitySnapshot,
    RequestCapacityLimiter,
)
from core.config import settings
from core.models import RagServiceResult, RetrievalDebug


class FakeRagService:
    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def query(self, command, *, tenant_context):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class FakeHealthSnapshot:
    ready = True
    degraded = False
    dependencies = ()


class RejectingLimiter:
    def snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            max_concurrent=1,
            max_queued=0,
            active=1,
            queued=0,
            available_slots=0,
            queue_available=0,
            accepting=False,
            saturated=True,
            closed=False,
        )

    @asynccontextmanager
    async def slot(self):
        raise CapacityRejectedError("queue_full")
        yield  # pragma: no cover


class OpenLimiter:
    def snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            max_concurrent=2,
            max_queued=4,
            active=0,
            queued=0,
            available_slots=2,
            queue_available=4,
            accepting=True,
            saturated=False,
            closed=False,
        )

    @asynccontextmanager
    async def slot(self):
        yield self.snapshot()


def _app(tenant_context, fake_service, *, limiter=None) -> FastAPI:
    app = FastAPI()
    app.include_router(routes_rag.router)
    app.include_router(routes_rag.health_router)
    routes_rag.install_rag_exception_handlers(app)
    if limiter is not None:
        app.state.rag_capacity_limiter = limiter

    async def tenant_override():
        return tenant_context

    app.dependency_overrides[routes_rag.resolve_request_tenant] = tenant_override
    return app


def _result(tenant_context) -> RagServiceResult:
    return RagServiceResult(
        request_id=UUID(tenant_context.request_id),
        answer=(
            "**A) Risposta**\n\nOK\n\n"
            "**B) Evidenze**\n\n-\n\n"
            "**C) Limiti / Conflitti**\n\n-\n\n"
            "**D) Fonti**\n\n-"
        ),
        retrieval=RetrievalDebug(),
    )


def test_capacity_settings_are_bounded_and_validated() -> None:
    assert settings.max_concurrent_queries >= 1
    assert settings.max_queued_queries >= 0
    assert settings.query_queue_timeout_seconds > 0

    with pytest.raises(ValidationError):
        settings.model_copy(update={"max_concurrent_queries": 0}, deep=True).__class__(
            **settings.model_dump(exclude={"max_concurrent_queries"}),
            max_concurrent_queries=0,
        )

    with pytest.raises(ValidationError):
        settings.model_copy(update={"max_queued_queries": -1}, deep=True).__class__(
            **settings.model_dump(exclude={"max_queued_queries"}),
            max_queued_queries=-1,
        )


@pytest.mark.asyncio
async def test_capacity_limiter_bounds_active_and_waiting_requests() -> None:
    limiter = RequestCapacityLimiter(
        max_concurrent=1,
        max_queued=1,
        acquire_timeout_seconds=1.0,
    )

    await limiter.acquire()
    waiter = asyncio.create_task(limiter.acquire())

    for _ in range(100):
        if limiter.snapshot().queued == 1:
            break
        await asyncio.sleep(0.001)

    snapshot = limiter.snapshot()
    assert snapshot.active == 1
    assert snapshot.queued == 1
    assert snapshot.accepting is False

    with pytest.raises(CapacityRejectedError) as rejected:
        await limiter.acquire()
    assert rejected.value.reason == "queue_full"

    await limiter.release()
    await waiter
    assert limiter.snapshot().active == 1
    assert limiter.snapshot().queued == 0
    await limiter.release()
    assert limiter.snapshot().active == 0


@pytest.mark.asyncio
async def test_capacity_queue_timeout_cleans_waiting_counter() -> None:
    limiter = RequestCapacityLimiter(
        max_concurrent=1,
        max_queued=1,
        acquire_timeout_seconds=0.02,
    )
    await limiter.acquire()

    with pytest.raises(CapacityRejectedError) as rejected:
        await limiter.acquire()

    assert rejected.value.reason == "queue_timeout"
    assert limiter.snapshot().queued == 0
    assert limiter.snapshot().active == 1
    await limiter.release()


def test_query_endpoint_returns_retryable_service_busy(monkeypatch, tenant_context) -> None:
    fake = FakeRagService(result=_result(tenant_context))
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake, limiter=RejectingLimiter())

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": "Domanda"})

    assert response.status_code == 503
    assert response.json()["code"] == "service_busy"
    assert response.json()["retryable"] is True
    assert response.headers["retry-after"] == "5"
    assert fake.calls == 0


def test_capacity_slot_is_released_when_service_raises(monkeypatch, tenant_context) -> None:
    from core.rag_service import RagServiceRetrievalError

    limiter = RequestCapacityLimiter(
        max_concurrent=1,
        max_queued=0,
        acquire_timeout_seconds=0.1,
    )
    fake = FakeRagService(error=RagServiceRetrievalError("internal backend detail"))
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake, limiter=limiter)

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": "Domanda"})

    assert response.status_code == 503
    assert limiter.snapshot().active == 0
    assert limiter.snapshot().accepting is True


def test_readiness_exposes_capacity_and_rejects_saturated_instance(
    monkeypatch,
    tenant_context,
) -> None:
    monkeypatch.setattr(
        routes_rag.resources,
        "health_snapshot",
        lambda deep=False: FakeHealthSnapshot(),
    )
    app = _app(
        tenant_context,
        FakeRagService(result=_result(tenant_context)),
        limiter=RejectingLimiter(),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "down"
    assert body["dependencies"]["request_capacity"]["state"] == "down"
    assert "active=1/1" in body["dependencies"]["request_capacity"]["detail"]


def test_readiness_is_ok_when_capacity_accepts_work(monkeypatch, tenant_context) -> None:
    monkeypatch.setattr(
        routes_rag.resources,
        "health_snapshot",
        lambda deep=False: FakeHealthSnapshot(),
    )
    app = _app(
        tenant_context,
        FakeRagService(result=_result(tenant_context)),
        limiter=OpenLimiter(),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["dependencies"]["request_capacity"]["state"] == "ok"
