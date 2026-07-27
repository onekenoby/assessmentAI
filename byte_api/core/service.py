"""Orchestrazione applicativa della Byte API."""
from __future__ import annotations

import copy
import importlib
import threading
from datetime import UTC, datetime
from typing import Any

from api.schemas import ServiceState
from core.config import settings


class ServiceBusyError(RuntimeError):
    pass


class ServiceClosedError(RuntimeError):
    pass


class ByteUploadService:
    def __init__(self, *, engine_module: Any | None = None) -> None:
        self._engine_module = engine_module
        self._state = ServiceState.NEW
        self._startup_error = ""
        self._initialized = False
        self._active_uploads = 0
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_uploads)

    def _engine(self):
        if self._engine_module is None:
            self._engine_module = importlib.import_module("byte_engine")
        return self._engine_module

    @property
    def state(self) -> ServiceState:
        with self._lock:
            return self._state

    def initialize(self, *, strict: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._state is ServiceState.CLOSED:
                raise ServiceClosedError("Servizio chiuso")
            self._state = ServiceState.INITIALIZING
            self._startup_error = ""

        dependencies = self._engine().healthcheck(deep=True)
        ready = all(bool(item.get("ready")) for item in dependencies.values())

        with self._lock:
            self._initialized = ready
            self._state = ServiceState.READY if ready else ServiceState.FAILED
            if not ready:
                self._startup_error = "; ".join(
                    str(item.get("detail") or "dependency unavailable")
                    for item in dependencies.values()
                    if not item.get("ready")
                )[:1000]

        if strict and not ready:
            raise RuntimeError(self._startup_error or "Byte API non pronta")
        return self.health_snapshot(deep=False)

    def _enter_upload(self) -> None:
        with self._lock:
            if self._state is ServiceState.CLOSED:
                raise ServiceClosedError("Servizio chiuso")
        if not self._slots.acquire(blocking=False):
            raise ServiceBusyError("Numero massimo di upload concorrenti raggiunto")
        with self._lock:
            self._active_uploads += 1

    def _leave_upload(self) -> None:
        with self._lock:
            self._active_uploads = max(0, self._active_uploads - 1)
        self._slots.release()

    def upload_corpus(self, payload: Any) -> dict[str, Any]:
        self._enter_upload()
        try:
            result = self._engine().upload_corpus(
                payload,
                max_file_bytes=settings.max_file_bytes,
            )
            return copy.deepcopy(dict(result))
        finally:
            self._leave_upload()

    def upload_evidence(self, payload: Any) -> dict[str, Any]:
        self._enter_upload()
        try:
            result = self._engine().upload_evidence(
                payload,
                max_file_bytes=settings.max_file_bytes,
            )
            return copy.deepcopy(dict(result))
        finally:
            self._leave_upload()

    def health_snapshot(self, *, deep: bool = False) -> dict[str, Any]:
        with self._lock:
            state = self._state
            initialized = self._initialized
            active_uploads = self._active_uploads
            startup_error = self._startup_error

        dependencies: dict[str, Any] = {}
        if deep:
            dependencies = self._engine().healthcheck(deep=True)
            dependency_ready = all(
                bool(item.get("ready")) for item in dependencies.values()
            )
        else:
            dependency_ready = initialized or state is ServiceState.NEW

        ready = state is not ServiceState.CLOSED and dependency_ready
        if deep and not dependency_ready:
            ready = False

        return {
            "service": settings.service_name,
            "state": state,
            "ready": ready,
            "initialized": initialized,
            "active_uploads": active_uploads,
            "startup_error": startup_error,
            "dependencies": copy.deepcopy(dependencies),
            "checked_at": datetime.now(UTC),
        }

    def close(self) -> None:
        with self._lock:
            self._state = ServiceState.CLOSED
            self._initialized = False


service = ByteUploadService()
