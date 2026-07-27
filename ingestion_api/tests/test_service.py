from __future__ import annotations

from threading import Event
from typing import Any

import pytest

from core.config import IngestionApiSettings
from core.service import (
    IngestionService,
    InternalState,
    RunNotFoundError,
    ServiceBusyError,
    ServiceClosedError,
)
from tests.conftest import wait_for_terminal


class IngestionWorkerBusyError(RuntimeError):
    pass


class FakeEngine:
    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        init_error: Exception | None = None,
        run_error: Exception | None = None,
        health_error: Exception | None = None,
        blocking: bool = False,
    ) -> None:
        self.result = result or {
            "status": "DONE",
            "jobs_claimed": 1,
            "jobs_completed": 1,
            "jobs_failed": 0,
            "queue_empty": True,
            "processing_time_ms": 12,
            "jobs": [
                {
                    "job_id": "job-1",
                    "status": "DONE",
                    "job_type": "CONTENT_INGESTION",
                    "document_id": "doc-1",
                    "chunks": 5,
                    "processing_time_ms": 10,
                    "ontology_codes": ["iso27001"],
                }
            ],
        }
        self.init_error = init_error
        self.run_error = run_error
        self.health_error = health_error
        self.blocking = blocking
        self.started = Event()
        self.release = Event()
        self.closed = False
        self.initialize_calls: list[bool] = []
        self.run_calls: list[int] = []
        if not blocking:
            self.release.set()

    def initialize_ingestion_runtime(self, *, strict: bool = True):
        self.initialize_calls.append(strict)
        if self.init_error:
            raise self.init_error
        return {"ready": True}

    def runtime_healthcheck(self, deep: bool = False):
        if self.health_error:
            raise self.health_error
        return {
            "ready": True,
            "initialized": True,
            "dependencies": {"fake": {"ready": True, "detail": "ok"}},
        }

    def run_pending_jobs(self, max_jobs: int):
        self.run_calls.append(max_jobs)
        self.started.set()
        self.release.wait(timeout=3)
        if self.run_error:
            raise self.run_error
        return self.result

    def shutdown_ingestion_runtime(self):
        self.closed = True


def make_service(
    engine: FakeEngine | None = None,
    *,
    history_limit: int = 20,
    expose_error_details: bool = False,
) -> tuple[IngestionService, FakeEngine]:
    config = IngestionApiSettings(
        initialize_on_startup=False,
        startup_strict=False,
        default_max_jobs=1,
        max_jobs_per_run=10,
        run_history_limit=history_limit,
        expose_error_details=expose_error_details,
    )
    service = IngestionService(config)
    resolved_engine = engine or FakeEngine()
    service._engine = resolved_engine
    return service, resolved_engine


def test_submit_is_serialized_and_returns_result():
    service, engine = make_service(FakeEngine(blocking=True))
    try:
        first = service.submit(max_jobs=1)
        assert engine.started.wait(timeout=1)

        with pytest.raises(ServiceBusyError):
            service.submit(max_jobs=1)

        engine.release.set()
        current = wait_for_terminal(service, first["run_id"])

        assert current["state"] == "succeeded"
        assert current["jobs_completed"] == 1
        assert current["jobs"][0]["document_id"] == "doc-1"
        assert service.active_run_id is None
    finally:
        engine.release.set()
        service.close()


def test_health_uses_engine_snapshot_and_close_is_idempotent():
    service, engine = make_service()
    service.initialize(strict=True)
    snapshot = service.health(deep=True)
    assert snapshot["ready"] is True
    assert snapshot["dependencies"]["fake"]["ready"] is True

    service.close()
    service.close()
    assert engine.closed is True
    assert service.state == InternalState.CLOSED


@pytest.mark.parametrize("max_jobs", [0, -1, 11])
def test_submit_rejects_invalid_limits(max_jobs: int):
    service, _ = make_service()
    try:
        with pytest.raises(ValueError, match="max_jobs fuori intervallo"):
            service.submit(max_jobs=max_jobs)
    finally:
        service.close()


def test_initialize_non_strict_records_failure_without_raising():
    service, _ = make_service(FakeEngine(init_error=RuntimeError("db offline")))
    try:
        snapshot = service.initialize(strict=False)
        assert snapshot["state"] == "failed"
        assert snapshot["ready"] is False
        assert "RuntimeError: db offline" in snapshot["startup_error"]
    finally:
        service.close()


def test_initialize_strict_propagates_and_records_failure():
    service, _ = make_service(FakeEngine(init_error=RuntimeError("contract mismatch")))
    try:
        with pytest.raises(RuntimeError, match="contract mismatch"):
            service.initialize(strict=True)
        assert service.state == InternalState.FAILED
        assert "contract mismatch" in service.health()["startup_error"]
    finally:
        service.close()


def test_initialize_ready_non_strict_short_circuits_second_initialization():
    service, engine = make_service()
    try:
        service.initialize(strict=False)
        service.initialize(strict=False)
        assert engine.initialize_calls == [False]
    finally:
        service.close()


@pytest.mark.parametrize(
    ("engine_status", "expected_state", "error_code"),
    [
        ("DONE", "succeeded", None),
        ("PARTIAL_FAILED", "partial_failed", None),
        ("FAILED", "failed", "ingestion_failed"),
        ("UNKNOWN", "failed", "ingestion_failed"),
    ],
)
def test_engine_status_mapping(engine_status: str, expected_state: str, error_code: str | None):
    result = {
        "status": engine_status,
        "jobs_claimed": 1,
        "jobs_completed": 0,
        "jobs_failed": 1,
        "queue_empty": False,
        "jobs": [],
    }
    service, _ = make_service(FakeEngine(result=result))
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        assert final["state"] == expected_state
        assert final["error_code"] == error_code
    finally:
        service.close()


def test_worker_busy_exception_is_mapped_to_public_error():
    service, _ = make_service(FakeEngine(run_error=IngestionWorkerBusyError("lock 123")))
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        assert final["state"] == "failed"
        assert final["error_code"] == "worker_busy"
        assert final["error_message"] == (
            "Un altro worker sta già eseguendo la pipeline di ingestion."
        )
        assert "lock 123" not in final["error_message"]
    finally:
        service.close()


def test_runtime_exception_details_are_hidden_by_default():
    service, _ = make_service(FakeEngine(run_error=RuntimeError("password=secret")))
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        assert final["error_code"] == "runtime_error"
        assert final["error_message"] == (
            "Esecuzione non completata; consultare i log applicativi."
        )
        assert "secret" not in final["error_message"]
    finally:
        service.close()


def test_runtime_exception_details_can_be_exposed_explicitly():
    service, _ = make_service(
        FakeEngine(run_error=RuntimeError("diagnostic")),
        expose_error_details=True,
    )
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        assert final["error_message"] == "RuntimeError: diagnostic"
    finally:
        service.close()


def test_job_result_is_sanitized_and_negative_counts_are_clamped():
    result = {
        "status": "PARTIAL_FAILED",
        "jobs_claimed": 1,
        "jobs_completed": 0,
        "jobs_failed": 1,
        "jobs": [
            {
                "job_id": 123,
                "status": "FAILED",
                "chunks": -5,
                "processing_time_ms": -9,
                "ontology_codes": None,
                "error": "sensitive stack trace",
            }
        ],
    }
    service, _ = make_service(FakeEngine(result=result))
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        job = final["jobs"][0]
        assert job["job_id"] == "123"
        assert job["chunks"] == 0
        assert job["processing_time_ms"] == 0
        assert job["ontology_codes"] == []
        assert job["error"] == "Elaborazione job fallita; consultare i log applicativi."
    finally:
        service.close()


def test_get_unknown_run_raises_not_found():
    service, _ = make_service()
    try:
        from uuid import uuid4

        with pytest.raises(RunNotFoundError):
            service.get_run(uuid4())
    finally:
        service.close()


def test_history_is_reverse_chronological_and_trimmed():
    service, _ = make_service(history_limit=2)
    try:
        ids = []
        for _ in range(3):
            run = service.submit(max_jobs=1)
            ids.append(run["run_id"])
            wait_for_terminal(service, run["run_id"])

        listed = service.list_runs()
        assert [item["run_id"] for item in listed] == [ids[2], ids[1]]
        with pytest.raises(RunNotFoundError):
            service.get_run(ids[0])
    finally:
        service.close()


def test_snapshots_do_not_expose_mutable_internal_job_data():
    service, _ = make_service()
    try:
        run = service.submit(max_jobs=1)
        final = wait_for_terminal(service, run["run_id"])
        final["jobs"][0]["ontology_codes"].append("tampered")
        again = service.get_run(run["run_id"])
        assert again["jobs"][0]["ontology_codes"] == ["iso27001"]
    finally:
        service.close()


def test_health_reports_runtime_healthcheck_exception():
    service, _ = make_service(FakeEngine(health_error=RuntimeError("ping failed")))
    try:
        service._state = InternalState.READY
        snapshot = service.health(deep=True)
        assert snapshot["ready"] is False
        assert snapshot["dependencies"]["runtime"]["ready"] is False
        assert "ping failed" in snapshot["dependencies"]["runtime"]["detail"]
    finally:
        service.close()


def test_closed_service_rejects_initialize_and_submit():
    service, _ = make_service()
    service.close()
    with pytest.raises(ServiceClosedError):
        service.initialize(strict=False)
    with pytest.raises(ServiceClosedError):
        service.submit(max_jobs=1)
