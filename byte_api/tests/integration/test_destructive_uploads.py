from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.destructive]


def _require_destructive() -> None:
    if os.getenv("BYTE_API_RUN_DESTRUCTIVE") != "1":
        pytest.skip("Impostare BYTE_API_RUN_DESTRUCTIVE=1: il test modifica il Database A")


def _base_url() -> str:
    return os.getenv("BYTE_API_BASE_URL", "http://127.0.0.1:8020").rstrip("/")


def _headers() -> dict[str, str]:
    api_key = os.getenv("BYTE_API_KEY", "").strip()
    return {"X-Byte-Api-Key": api_key} if api_key else {}


def _file_from_env(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"Variabile {name} non configurata")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"File inesistente: {path}")
    return path


def test_real_corpus_upload_creates_pending_job():
    _require_destructive()
    path = _file_from_env("BYTE_API_TEST_CORPUS_FILE")

    data = {
        "tier": os.getenv("BYTE_API_TEST_TIER", "C"),
        "organization_id": os.getenv("BYTE_API_TEST_ORGANIZATION_ID", "9999"),
        "user_id": os.getenv("BYTE_API_TEST_USER_ID", "123"),
        "area": os.getenv("BYTE_API_TEST_AREA", "IDENTIFY"),
        "subarea": os.getenv("BYTE_API_TEST_SUBAREA", "Risk Assessment"),
        "classification": os.getenv("BYTE_API_TEST_CLASSIFICATION", "internal"),
        "pipeline_version": os.getenv("BYTE_API_TEST_PIPELINE_VERSION", "v1"),
        "corpus_version": os.getenv("BYTE_API_TEST_CORPUS_VERSION", "v1"),
    }

    with path.open("rb") as stream, httpx.Client(timeout=120.0, headers=_headers()) as client:
        response = client.post(
            f"{_base_url()}/api/v1/byte/corpus",
            data=data,
            files={"file": (path.name, stream, "application/octet-stream")},
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["mode"] == "corpus"
    assert payload["document_id"]
    assert payload["jobs"]
    assert payload["jobs"][0]["status"] in {"PENDING", "RUNNING"}


def test_real_evidence_upload_creates_pending_job():
    _require_destructive()
    path = _file_from_env("BYTE_API_TEST_EVIDENCE_FILE")

    required = {
        "organization_id": os.getenv("BYTE_API_TEST_ORGANIZATION_ID", "").strip(),
        "user_id": os.getenv("BYTE_API_TEST_USER_ID", "").strip(),
        "assessment_id": os.getenv("BYTE_API_TEST_ASSESSMENT_ID", "").strip(),
        "response_id": os.getenv("BYTE_API_TEST_RESPONSE_ID", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        pytest.skip("Variabili evidence mancanti: " + ", ".join(missing))
    required["encryption_required"] = os.getenv(
        "BYTE_API_TEST_ENCRYPTION_REQUIRED", "true"
    )

    with path.open("rb") as stream, httpx.Client(timeout=120.0, headers=_headers()) as client:
        response = client.post(
            f"{_base_url()}/api/v1/byte/evidence",
            data=required,
            files={"file": (path.name, stream, "application/octet-stream")},
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["mode"] == "evidence"
    assert payload["document_id"]
    assert payload["jobs"]
    assert payload["jobs"][0]["status"] in {"PENDING", "RUNNING"}
