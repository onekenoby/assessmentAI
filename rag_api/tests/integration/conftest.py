from __future__ import annotations

import os
from uuid import uuid4

import pytest

from core.config import RagSettings, settings
from core.tenant import TenantContext


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session", autouse=True)
def require_real_services_enabled() -> None:
    """Evita che i test di integrazione partano durante il normale pytest."""
    if not _env_bool("RUN_REAL_SERVICE_TESTS", False):
        pytest.skip(
            "Test reali disabilitati. Impostare RUN_REAL_SERVICE_TESTS=1 "
            "oppure usare tests/verify_real_services.py."
        )


@pytest.fixture(scope="session")
def integration_settings() -> RagSettings:
    return settings


@pytest.fixture(scope="session")
def integration_tenant(integration_settings: RagSettings) -> TenantContext:
    return TenantContext(
        organization_id=integration_settings.poc_organization_id,
        user_id="integration-test-user",
        roles=("auditor",),
        request_id=str(uuid4()),
        allowed_scopes=("GLOBAL", "ACCOUNT"),
    )


@pytest.fixture(scope="session")
def require_non_empty_data() -> bool:
    return _env_bool("RAG_INTEGRATION_REQUIRE_DATA", True)


@pytest.fixture(scope="session")
def service_timeout_seconds() -> int:
    raw = os.getenv("RAG_INTEGRATION_TIMEOUT_S", "300")
    try:
        value = int(raw)
    except ValueError:
        value = 300
    return max(5, value)


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@pytest.fixture(scope="session")
def require_second_organization() -> bool:
    return _env_bool("RAG_E2E_REQUIRE_SECOND_ORGANIZATION", False)


@pytest.fixture(scope="session")
def run_capacity_stress() -> bool:
    return _env_bool("RAG_E2E_RUN_CAPACITY_STRESS", False)
