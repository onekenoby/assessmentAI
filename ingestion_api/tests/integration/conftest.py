from __future__ import annotations

import os

import pytest


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session", autouse=True)
def require_real_services_enabled() -> None:
    if not env_bool("RUN_REAL_SERVICE_TESTS"):
        pytest.skip(
            "Test reali disabilitati. Impostare RUN_REAL_SERVICE_TESTS=1 "
            "solo nell'ambiente di test configurato."
        )


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.getenv("INGESTION_API_BASE_URL", "http://127.0.0.1:8010").rstrip("/")


@pytest.fixture(scope="session")
def api_headers() -> dict[str, str]:
    key = os.getenv("INGESTION_API_KEY", "").strip()
    return {"X-Ingestion-Api-Key": key} if key else {}
