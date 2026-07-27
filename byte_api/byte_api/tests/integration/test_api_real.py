from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration


def _headers() -> dict[str, str]:
    api_key = os.getenv("BYTE_API_KEY", "").strip()
    return {"X-Byte-Api-Key": api_key} if api_key else {}


def test_real_api_liveness_and_readiness():
    if os.getenv("BYTE_API_RUN_API_INTEGRATION") != "1":
        pytest.skip("Impostare BYTE_API_RUN_API_INTEGRATION=1")

    base_url = os.getenv("BYTE_API_BASE_URL", "http://127.0.0.1:8020").rstrip("/")
    with httpx.Client(timeout=30.0, headers=_headers()) as client:
        live = client.get(f"{base_url}/health/live")
        ready = client.get(f"{base_url}/health/ready", params={"deep": "true"})

    assert live.status_code == 200, live.text
    assert live.json()["ready"] is True
    assert ready.status_code == 200, ready.text
    assert ready.json()["dependencies"]["postgres_source"]["ready"] is True
