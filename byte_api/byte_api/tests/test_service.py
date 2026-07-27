from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import core.service as service_module
from core.service import ByteUploadService, ServiceBusyError, ServiceClosedError


class FakeEngine:
    def __init__(self):
        self.corpus_result = {"value": [1]}
        self.evidence_result = {"value": [2]}
        self.corpus_calls = []
        self.evidence_calls = []
        self.health = {"postgres_source": {"ready": True, "detail": "ok"}}
        self.block = None

    def upload_corpus(self, payload, *, max_file_bytes):
        self.corpus_calls.append((payload, max_file_bytes))
        if self.block:
            self.block.wait(timeout=2)
        return self.corpus_result

    def upload_evidence(self, payload, *, max_file_bytes):
        self.evidence_calls.append((payload, max_file_bytes))
        return self.evidence_result

    def healthcheck(self, *, deep=False):
        return self.health


def test_initialize_ready():
    service = ByteUploadService(engine_module=FakeEngine())
    result = service.initialize(strict=True)
    assert result["state"] == "ready"
    assert result["initialized"] is True


def test_initialize_non_strict_failure():
    engine = FakeEngine()
    engine.health = {"postgres_source": {"ready": False, "detail": "offline"}}
    service = ByteUploadService(engine_module=engine)
    result = service.initialize(strict=False)
    assert result["state"] == "failed"
    assert result["initialized"] is False
    assert "offline" in result["startup_error"]


def test_initialize_strict_failure_raises():
    engine = FakeEngine()
    engine.health = {"postgres_source": {"ready": False, "detail": "offline"}}
    service = ByteUploadService(engine_module=engine)
    with pytest.raises(RuntimeError, match="offline"):
        service.initialize(strict=True)


def test_upload_corpus_returns_deep_copy():
    engine = FakeEngine()
    service = ByteUploadService(engine_module=engine)
    result = service.upload_corpus("payload")
    result["value"].append(99)
    assert engine.corpus_result == {"value": [1]}
    assert engine.corpus_calls[0][0] == "payload"


def test_upload_evidence_returns_deep_copy():
    engine = FakeEngine()
    service = ByteUploadService(engine_module=engine)
    result = service.upload_evidence("payload")
    result["value"].append(99)
    assert engine.evidence_result == {"value": [2]}


def test_active_upload_count_resets_after_error():
    class BrokenEngine(FakeEngine):
        def upload_corpus(self, payload, *, max_file_bytes):
            raise RuntimeError("boom")

    service = ByteUploadService(engine_module=BrokenEngine())
    with pytest.raises(RuntimeError):
        service.upload_corpus("x")
    assert service.health_snapshot()["active_uploads"] == 0


def test_busy_service_rejects_when_slot_unavailable(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "settings",
        SimpleNamespace(max_concurrent_uploads=1, max_file_bytes=10, service_name="byte-api"),
    )
    engine = FakeEngine()
    gate = threading.Event()
    engine.block = gate
    service = ByteUploadService(engine_module=engine)

    thread = threading.Thread(target=service.upload_corpus, args=("first",))
    thread.start()
    for _ in range(100):
        if service.health_snapshot()["active_uploads"] == 1:
            break
        threading.Event().wait(0.005)

    with pytest.raises(ServiceBusyError):
        service.upload_corpus("second")
    gate.set()
    thread.join(timeout=2)


def test_close_rejects_uploads():
    service = ByteUploadService(engine_module=FakeEngine())
    service.close()
    with pytest.raises(ServiceClosedError):
        service.upload_corpus("x")


def test_close_rejects_initialize():
    service = ByteUploadService(engine_module=FakeEngine())
    service.close()
    with pytest.raises(ServiceClosedError):
        service.initialize()


def test_deep_health_uses_engine_result():
    engine = FakeEngine()
    service = ByteUploadService(engine_module=engine)
    result = service.health_snapshot(deep=True)
    assert result["ready"] is True
    assert result["dependencies"]["postgres_source"]["ready"] is True


def test_deep_health_reports_failure():
    engine = FakeEngine()
    engine.health = {"postgres_source": {"ready": False, "detail": "offline"}}
    service = ByteUploadService(engine_module=engine)
    assert service.health_snapshot(deep=True)["ready"] is False


def test_state_property_returns_current_state():
    service = ByteUploadService(engine_module=FakeEngine())
    assert service.state == "new"
    service.initialize()
    assert service.state == "ready"


def test_engine_is_imported_lazily(monkeypatch):
    marker = FakeEngine()
    monkeypatch.setattr(service_module.importlib, "import_module", lambda name: marker)
    service = ByteUploadService()
    assert service._engine() is marker
    assert service._engine() is marker
