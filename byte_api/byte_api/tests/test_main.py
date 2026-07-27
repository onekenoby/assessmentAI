from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from main import _valid_request_id, create_app


def test_valid_request_id_roundtrip():
    value = "550e8400-e29b-41d4-a716-446655440000"
    assert _valid_request_id(value) == value


def test_invalid_request_id_generates_uuid():
    value = _valid_request_id("not-valid")
    assert str(UUID(value)) == value


def test_missing_request_id_generates_uuid():
    value = _valid_request_id(None)
    assert str(UUID(value)) == value


def test_app_metadata():
    app = create_app()
    assert app.title == "Multi-Tenant BYTEA Upload API"
    assert app.version == "1.0.0"


def test_app_has_expected_routes():
    paths = {route.path for route in create_app().routes}
    assert "/api/v1/byte/corpus" in paths
    assert "/api/v1/byte/evidence" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths


def test_lifespan_initializes_and_closes(monkeypatch):
    import main

    class FakeService:
        def __init__(self):
            self.initialized = []
            self.closed = 0

        def initialize(self, *, strict=False):
            self.initialized.append(strict)

        def close(self):
            self.closed += 1

    fake = FakeService()
    monkeypatch.setattr(main, "service", fake)
    monkeypatch.setattr(
        main,
        "settings",
        type("Settings", (), {"initialize_on_startup": True, "startup_strict": False})(),
    )
    app = create_app()

    async def run():
        async with main.lifespan(app):
            assert fake.initialized == [False]

    asyncio.run(run())
    assert fake.closed == 1


def test_lifespan_non_strict_records_startup_error(monkeypatch):
    import main

    class FakeService:
        def initialize(self, *, strict=False):
            raise RuntimeError("offline")

        def close(self):
            pass

    monkeypatch.setattr(main, "service", FakeService())
    monkeypatch.setattr(
        main,
        "settings",
        type("Settings", (), {"initialize_on_startup": True, "startup_strict": False})(),
    )
    app = create_app()

    async def run():
        async with main.lifespan(app):
            assert "offline" in app.state.startup_error

    asyncio.run(run())


def test_lifespan_strict_raises_and_still_closes(monkeypatch):
    import main

    class FakeService:
        def __init__(self):
            self.closed = 0

        def initialize(self, *, strict=False):
            raise RuntimeError("offline")

        def close(self):
            self.closed += 1

    fake = FakeService()
    monkeypatch.setattr(main, "service", fake)
    monkeypatch.setattr(
        main,
        "settings",
        type("Settings", (), {"initialize_on_startup": True, "startup_strict": True})(),
    )
    app = create_app()

    async def run():
        async with main.lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(run())
    assert fake.closed == 1
