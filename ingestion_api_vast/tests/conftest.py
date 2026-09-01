from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import api.routes_ingestion as routes
import main


TERMINAL_STATES = {"succeeded", "partial_failed", "failed"}


def wait_for_terminal(service: Any, run_id: UUID, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    snapshot = service.get_run(run_id)
    while snapshot["state"] not in TERMINAL_STATES and time.monotonic() < deadline:
        time.sleep(0.01)
        snapshot = service.get_run(run_id)
    assert snapshot["state"] in TERMINAL_STATES, snapshot
    return snapshot


def sample_record(*, state: str = "queued", run_id: UUID | None = None) -> dict[str, Any]:
    now = datetime.now(UTC)
    terminal = state in TERMINAL_STATES
    return {
        "run_id": run_id or uuid4(),
        "state": state,
        "requested_max_jobs": 1,
        "created_at": now,
        "started_at": now if state != "queued" else None,
        "completed_at": now if terminal else None,
        "jobs_claimed": 1 if terminal else 0,
        "jobs_completed": 1 if state == "succeeded" else 0,
        "jobs_failed": 1 if state in {"partial_failed", "failed"} else 0,
        "queue_empty": True if terminal else None,
        "processing_time_ms": 10 if terminal else 0,
        "jobs": [],
        "error_code": "ingestion_failed" if state == "failed" else None,
        "error_message": "errore" if state == "failed" else None,
    }


class RouteServiceStub:
    def __init__(self) -> None:
        self.record = sample_record()
        self.records = [self.record]
        self.submit_error: Exception | None = None
        self.get_error: Exception | None = None
        self.health_snapshot = {
            "service": "ingestion-api",
            "state": "ready",
            "ready": True,
            "active_run_id": None,
            "initialized": True,
            "startup_error": "",
            "dependencies": {
                "fake": {"ready": True, "detail": "ok"},
            },
            "checked_at": datetime.now(UTC),
        }
        self.submitted_max_jobs: list[int] = []
        self.initialize_calls: list[bool] = []
        self.closed = False

    def initialize(self, *, strict: bool = False) -> dict[str, Any]:
        self.initialize_calls.append(strict)
        return dict(self.health_snapshot)

    def submit(self, *, max_jobs: int) -> dict[str, Any]:
        self.submitted_max_jobs.append(max_jobs)
        if self.submit_error:
            raise self.submit_error
        return dict(self.record)

    def get_run(self, run_id: UUID) -> dict[str, Any]:
        if self.get_error:
            raise self.get_error
        record = dict(self.record)
        record["run_id"] = run_id
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.records]

    def health(self, *, deep: bool = False) -> dict[str, Any]:
        snapshot = dict(self.health_snapshot)
        snapshot["dependencies"] = {
            key: dict(value)
            for key, value in self.health_snapshot.get("dependencies", {}).items()
        }
        snapshot["checked_at"] = datetime.now(UTC)
        return snapshot

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def route_service(monkeypatch: pytest.MonkeyPatch) -> RouteServiceStub:
    stub = RouteServiceStub()
    monkeypatch.setattr(routes, "service", stub)
    monkeypatch.setattr(main, "service", stub)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(initialize_on_startup=False, startup_strict=False),
    )
    monkeypatch.setattr(
        routes,
        "settings",
        SimpleNamespace(
            api_key="",
            api_key_header="X-Ingestion-Api-Key",
        ),
    )
    return stub


@pytest.fixture
def client(route_service: RouteServiceStub):
    with TestClient(main.create_app()) as test_client:
        yield test_client
    assert route_service.closed is True
