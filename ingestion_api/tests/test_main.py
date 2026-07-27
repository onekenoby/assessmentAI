from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import api.routes_ingestion as routes
import main
from tests.conftest import RouteServiceStub


def test_valid_request_id_helper_preserves_uuid():
    value = str(uuid4())
    assert main._valid_request_id(value) == value


def test_valid_request_id_helper_replaces_invalid_values():
    generated = main._valid_request_id("not-a-uuid")
    assert generated != "not-a-uuid"
    assert str(UUID(generated)) == generated


def test_lifespan_initializes_and_closes_service(monkeypatch):
    stub = RouteServiceStub()
    monkeypatch.setattr(main, "service", stub)
    monkeypatch.setattr(routes, "service", stub)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(initialize_on_startup=True, startup_strict=True),
    )

    with TestClient(main.create_app()) as client:
        assert client.get("/health/live").status_code == 200

    assert stub.initialize_calls == [True]
    assert stub.closed is True


def test_non_strict_startup_failure_does_not_abort_app(monkeypatch):
    class FailingService(RouteServiceStub):
        def initialize(self, *, strict: bool = False):
            self.initialize_calls.append(strict)
            raise RuntimeError("postgres unavailable")

    stub = FailingService()
    monkeypatch.setattr(main, "service", stub)
    monkeypatch.setattr(routes, "service", stub)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(initialize_on_startup=True, startup_strict=False),
    )

    app = main.create_app()
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert "postgres unavailable" in app.state.startup_error

    assert stub.initialize_calls == [False]
    assert stub.closed is True


def test_strict_startup_failure_aborts_and_still_closes_service(monkeypatch):
    class FailingService(RouteServiceStub):
        def initialize(self, *, strict: bool = False):
            self.initialize_calls.append(strict)
            raise RuntimeError("contract mismatch")

    stub = FailingService()
    monkeypatch.setattr(main, "service", stub)
    monkeypatch.setattr(routes, "service", stub)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(initialize_on_startup=True, startup_strict=True),
    )

    with pytest.raises(RuntimeError, match="contract mismatch"):
        with TestClient(main.create_app()):
            pass

    assert stub.initialize_calls == [True]
    assert stub.closed is True
