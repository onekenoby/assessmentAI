"""Orchestrazione thread-safe del motore di ingestion.

Il motore pesante viene importato in modo lazy. Una sola esecuzione è ammessa
per processo; il motore applica inoltre un advisory lock PostgreSQL globale,
proteggendo GPU/Ollama anche con più processi o istanze API.
"""
from __future__ import annotations

import importlib
from copy import deepcopy
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

from core.config import IngestionApiSettings, settings

logger = logging.getLogger(__name__)


class IngestionServiceError(RuntimeError):
    pass


class ServiceBusyError(IngestionServiceError):
    pass


class RunNotFoundError(IngestionServiceError):
    pass


class ServiceClosedError(IngestionServiceError):
    pass


class InternalState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(slots=True)
class RunRecord:
    run_id: UUID
    requested_max_jobs: int
    state: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    jobs_claimed: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    queue_empty: bool | None = None
    processing_time_ms: int = 0
    jobs: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "requested_max_jobs": self.requested_max_jobs,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "jobs_claimed": self.jobs_claimed,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "queue_empty": self.queue_empty,
            "processing_time_ms": self.processing_time_ms,
            "jobs": [deepcopy(item) for item in self.jobs],
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class IngestionService:
    def __init__(self, config: IngestionApiSettings = settings) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ingestion-api-worker",
        )
        self._engine: ModuleType | None = None
        self._state = InternalState.NEW
        self._startup_error = ""
        self._active_run_id: UUID | None = None
        self._runs: OrderedDict[UUID, RunRecord] = OrderedDict()

    @property
    def state(self) -> InternalState:
        with self._lock:
            return self._state

    @property
    def active_run_id(self) -> UUID | None:
        with self._lock:
            return self._active_run_id

    def _load_engine(self) -> ModuleType:
        if self._engine is None:
            self._engine = importlib.import_module("ingestion_engine")
        return self._engine

    def initialize(self, *, strict: bool | None = None) -> dict[str, Any]:
        effective_strict = self._config.startup_strict if strict is None else strict
        with self._lock:
            if self._state == InternalState.CLOSED:
                raise ServiceClosedError("Servizio ingestion chiuso")
            if self._state == InternalState.READY and not effective_strict:
                return self.health(deep=False)
            self._state = InternalState.INITIALIZING
            self._startup_error = ""

        try:
            engine = self._load_engine()
            engine.initialize_ingestion_runtime(strict=effective_strict)
        except Exception as exc:
            with self._lock:
                self._state = InternalState.FAILED
                self._startup_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
            if effective_strict:
                raise
            logger.exception("Inizializzazione ingestion non strict fallita")
        else:
            with self._lock:
                self._state = InternalState.READY

        return self.health(deep=False)

    def submit(self, *, max_jobs: int) -> dict[str, Any]:
        if max_jobs <= 0 or max_jobs > self._config.max_jobs_per_run:
            raise ValueError("max_jobs fuori intervallo")

        with self._lock:
            if self._state == InternalState.CLOSED:
                raise ServiceClosedError("Servizio ingestion chiuso")
            if self._active_run_id is not None:
                raise ServiceBusyError(
                    f"Esecuzione già attiva: {self._active_run_id}"
                )

            record = RunRecord(run_id=uuid4(), requested_max_jobs=max_jobs)
            self._runs[record.run_id] = record
            self._active_run_id = record.run_id
            self._trim_history_locked()
            self._executor.submit(self._execute_run, record.run_id)
            return record.snapshot()

    def _execute_run(self, run_id: UUID) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.state = "running"
            record.started_at = datetime.now(UTC)

        try:
            self.initialize(strict=True)
            engine = self._load_engine()
            result = engine.run_pending_jobs(record.requested_max_jobs)
            jobs = [self._sanitize_job_result(item) for item in result.get("jobs", [])]

            with self._lock:
                record.jobs = jobs
                record.jobs_claimed = int(result.get("jobs_claimed", len(jobs)) or 0)
                record.jobs_completed = int(result.get("jobs_completed", 0) or 0)
                record.jobs_failed = int(result.get("jobs_failed", 0) or 0)
                record.queue_empty = bool(result.get("queue_empty", False))
                record.processing_time_ms = int(result.get("processing_time_ms", 0) or 0)
                status = str(result.get("status") or "FAILED").upper()
                if status == "DONE":
                    record.state = "succeeded"
                elif status == "PARTIAL_FAILED":
                    record.state = "partial_failed"
                else:
                    record.state = "failed"
                    record.error_code = "ingestion_failed"
                    record.error_message = "Uno o più job non sono stati completati."
        except Exception as exc:
            error_name = type(exc).__name__
            with self._lock:
                record.state = "failed"
                record.error_code = (
                    "worker_busy"
                    if error_name == "IngestionWorkerBusyError"
                    else "runtime_error"
                )
                record.error_message = self._public_error(exc)
            logger.exception("Esecuzione ingestion fallita | run_id=%s", run_id)
        finally:
            with self._lock:
                record.completed_at = datetime.now(UTC)
                if record.started_at and record.processing_time_ms <= 0:
                    delta = record.completed_at - record.started_at
                    record.processing_time_ms = max(0, int(delta.total_seconds() * 1000))
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _sanitize_job_result(self, item: dict[str, Any]) -> dict[str, Any]:
        clean = {
            "job_id": str(item.get("job_id") or "") or None,
            "status": str(item.get("status") or "UNKNOWN"),
            "job_type": str(item.get("job_type") or "") or None,
            "document_id": str(item.get("document_id") or "") or None,
            "chunks": max(0, int(item.get("chunks", 0) or 0)),
            "processing_time_ms": max(0, int(item.get("processing_time_ms", 0) or 0)),
            "ontology_codes": [str(v) for v in (item.get("ontology_codes") or [])],
            "next_status": str(item.get("next_status") or "") or None,
            "error": None,
        }
        if item.get("error"):
            clean["error"] = (
                str(item["error"])[:1000]
                if self._config.expose_error_details
                else "Elaborazione job fallita; consultare i log applicativi."
            )
        return clean

    def _public_error(self, exc: Exception) -> str:
        if self._config.expose_error_details:
            return f"{type(exc).__name__}: {str(exc)[:1000]}"
        if type(exc).__name__ == "IngestionWorkerBusyError":
            return "Un altro worker sta già eseguendo la pipeline di ingestion."
        return "Esecuzione non completata; consultare i log applicativi."

    def get_run(self, run_id: UUID) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise RunNotFoundError(str(run_id))
            return record.snapshot()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.snapshot() for record in reversed(self._runs.values())]

    def health(self, *, deep: bool = False) -> dict[str, Any]:
        with self._lock:
            state = self._state
            active_run_id = self._active_run_id
            startup_error = self._startup_error
            engine = self._engine

        dependencies: dict[str, Any] = {}
        initialized = state == InternalState.READY
        ready = state == InternalState.READY

        if engine is not None and state != InternalState.CLOSED:
            try:
                snapshot = engine.runtime_healthcheck(deep=deep)
            except Exception as exc:
                ready = False
                dependencies = {
                    "runtime": {
                        "ready": False,
                        "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                }
            else:
                dependencies = snapshot.get("dependencies", {})
                initialized = bool(snapshot.get("initialized", initialized))
                ready = ready and bool(snapshot.get("ready", False))

        return {
            "service": self._config.service_name,
            "state": state.value,
            "ready": ready,
            "active_run_id": active_run_id,
            "initialized": initialized,
            "startup_error": startup_error,
            "dependencies": dependencies,
            "checked_at": datetime.now(UTC),
        }

    def close(self) -> None:
        with self._lock:
            if self._state == InternalState.CLOSED:
                return
            self._state = InternalState.CLOSED
            engine = self._engine

        self._executor.shutdown(wait=True, cancel_futures=True)
        if engine is not None:
            try:
                engine.shutdown_ingestion_runtime()
            except Exception:
                logger.exception("Chiusura runtime ingestion fallita")

    def _trim_history_locked(self) -> None:
        while len(self._runs) > self._config.run_history_limit:
            oldest_id, oldest = next(iter(self._runs.items()))
            if oldest_id == self._active_run_id:
                break
            self._runs.popitem(last=False)


service = IngestionService()
