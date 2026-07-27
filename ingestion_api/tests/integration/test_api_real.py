from __future__ import annotations

import os
import time
from uuid import UUID, uuid4

import pytest
import requests

pytestmark = pytest.mark.integration


def test_live_endpoint_and_security_headers(api_base_url: str):
    response = requests.get(
        api_base_url + "/health/live",
        headers={"X-Request-ID": str(uuid4())},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["service"]
    assert payload["state"] != "closed"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert str(UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


def test_deep_readiness_reports_real_dependencies(api_base_url: str):
    response = requests.get(api_base_url + "/health/ready", params={"deep": "true"}, timeout=60)
    assert response.status_code == 200, response.text[:2000]
    payload = response.json()
    assert payload["ready"] is True
    required = {"postgres_source", "postgres_output", "ollama", "qdrant"}
    assert required.issubset(payload["dependencies"])
    for name in required:
        assert payload["dependencies"][name]["ready"] is True


@pytest.mark.destructive
def test_claim_and_ingest_one_job_only_when_explicitly_enabled(
    api_base_url: str,
    api_headers: dict[str, str],
):
    if os.getenv("RUN_INGESTION_CLAIM_TEST", "0").strip() != "1":
        pytest.skip(
            "Test distruttivo disabilitato: impostare RUN_INGESTION_CLAIM_TEST=1 "
            "solo su Database A di test con un job PENDING predisposto."
        )

    create = requests.post(
        api_base_url + "/api/v1/ingestion/runs",
        headers=api_headers,
        json={"max_jobs": 1},
        timeout=30,
    )
    assert create.status_code == 202, create.text[:2000]
    run = create.json()
    run_id = str(UUID(run["run_id"]))

    deadline = time.monotonic() + int(os.getenv("INGESTION_E2E_TIMEOUT_S", "1800"))
    final = run
    while final["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(2)
        response = requests.get(
            api_base_url + f"/api/v1/ingestion/runs/{run_id}",
            headers=api_headers,
            timeout=30,
        )
        response.raise_for_status()
        final = response.json()

    assert final["state"] in {"succeeded", "partial_failed", "failed"}
    assert final["completed_at"] is not None
    assert final["jobs_claimed"] <= 1
    assert final["state"] == "succeeded", final
