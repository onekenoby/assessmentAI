from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePath
from typing import Any
from uuid import UUID, uuid4

import pytest

from core.config import RagSettings
from core.tenant import TenantContext

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api_session() -> Iterator[Any]:
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "rag-api-batch13-e2e/1.0",
        }
    )
    try:
        yield session
    finally:
        session.close()


def _response_json(response: Any) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except Exception as exc:  # pragma: no cover - diagnostic path on real systems
        pytest.fail(
            f"Risposta non JSON da {response.request.method} {response.request.url}: "
            f"status={response.status_code}, body={response.text[:1000]!r}, error={exc}"
        )
    assert isinstance(payload, Mapping), f"Payload API non-object: {payload!r}"
    return payload


def _assert_request_id(value: Any) -> None:
    assert str(UUID(str(value))) == str(value)


def _normalize_document_name(value: str) -> str:
    name = PurePath(str(value or "").strip()).name.casefold()
    name = re.sub(r"\.(pdf|md|txt|docx|html|csv|xlsx)$", "", name)
    name = re.sub(r"[_\-\s]+(?:out|output)$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _sample_visible_chunk(
    config: RagSettings,
    tenant: TenantContext,
    *,
    require_data: bool,
) -> dict[str, Any] | None:
    import psycopg2

    conn = psycopg2.connect(
        host=config.pg_host,
        port=config.pg_port,
        dbname=config.pg_database,
        user=config.pg_user,
        password=config.pg_password,
        connect_timeout=10,
        application_name="rag-api-batch13-document-scope",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content_raw, content_semantic, metadata_json
                FROM public.document_chunks
                WHERE lower(status) = 'active'
                  AND (
                        (upper(scope) = 'GLOBAL' AND organization_id IS NULL AND upper(tier) = 'A')
                        OR
                        (upper(scope) = 'ACCOUNT' AND organization_id = %s AND upper(tier) IN ('B', 'C'))
                      )
                  AND coalesce(content_semantic, content_raw, '') <> ''
                ORDER BY length(coalesce(content_semantic, content_raw, '')) DESC,
                         ingestion_ts DESC
                LIMIT 100
                """,
                (tenant.organization_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    for raw, semantic, metadata_raw in rows:
        metadata = _metadata_dict(metadata_raw)
        filename = str(metadata.get("filename") or metadata.get("source_name") or "").strip()
        try:
            page = int(metadata.get("page") or metadata.get("page_no") or 0)
        except (TypeError, ValueError):
            page = 0
        content = str(semantic or raw or "").strip()
        if filename and page > 0 and content:
            words = []
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9_-]{4,}", content):
                normalized = token.casefold()
                if normalized not in words:
                    words.append(normalized)
                if len(words) >= 12:
                    break
            return {
                "filename": filename,
                "page": page,
                "content": content,
                "query": " ".join(words) or content[:250],
            }

    if require_data:
        pytest.fail(
            "Nessun chunk PostgreSQL visibile con filename, pagina e contenuto utilizzabili"
        )
    return None


@pytest.fixture(scope="module")
def visible_sample(
    integration_settings: RagSettings,
    integration_tenant: TenantContext,
    require_non_empty_data: bool,
) -> dict[str, Any] | None:
    if not integration_settings.pg_enrich_enabled:
        pytest.skip("Document/page E2E richiede PostgreSQL abilitato")
    return _sample_visible_chunk(
        integration_settings,
        integration_tenant,
        require_data=require_non_empty_data,
    )


def _post_query(
    session: Any,
    base_url: str,
    body: Mapping[str, Any],
    timeout_seconds: int,
) -> Any:
    return session.post(
        base_url + "/api/v1/rag/query",
        headers={"X-Request-ID": str(uuid4())},
        json=dict(body),
        timeout=(10, timeout_seconds),
    )


def test_live_endpoint_and_security_headers(
    api_session: Any,
    api_base_url: str,
    service_timeout_seconds: int,
) -> None:
    response = api_session.get(
        api_base_url + "/health/live",
        timeout=(10, service_timeout_seconds),
    )
    assert response.status_code == 200, response.text[:1000]
    payload = _response_json(response)
    assert payload["status"] == "ok"
    assert payload["service"] == "rag-api"
    assert response.headers.get("Cache-Control") == "no-store"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    _assert_request_id(response.headers.get("X-Request-ID"))


def test_deep_readiness_reports_real_dependencies(
    api_session: Any,
    api_base_url: str,
    integration_settings: RagSettings,
    service_timeout_seconds: int,
) -> None:
    response = api_session.get(
        api_base_url + "/health/ready",
        params={"deep": "true"},
        timeout=(10, service_timeout_seconds),
    )
    assert response.status_code == 200, response.text[:2000]
    payload = _response_json(response)
    assert payload["status"] in {"ok", "degraded"}

    dependencies = payload.get("dependencies")
    assert isinstance(dependencies, Mapping)
    required = {"embedder", "reranker", "ollama", "qdrant", "request_capacity"}
    if integration_settings.pg_enrich_enabled:
        required.add("postgresql")
    if integration_settings.neo4j_enabled:
        required.add("neo4j")
    assert required.issubset(dependencies.keys())
    for name in required:
        state = str((dependencies[name] or {}).get("state") or "")
        assert state in {"ok", "degraded"}, f"{name} non pronto: {dependencies[name]}"


def test_deterministic_math_route_is_end_to_end_and_model_free(
    api_session: Any,
    api_base_url: str,
    service_timeout_seconds: int,
) -> None:
    response = _post_query(
        api_session,
        api_base_url,
        {
            "query": (
                "Una checklist di 20 controlli contiene 12 controlli implementati e "
                "4 controlli parziali che valgono il 50%. Calcola la copertura complessiva."
            ),
            "conversation_id": "batch13-math",
            "history": [],
            "options": {
                "include_sources": True,
                "include_debug": True,
                "include_evaluation": True,
            },
        },
        service_timeout_seconds,
    )
    assert response.status_code == 200, response.text[:2000]
    payload = _response_json(response)
    assert payload["status"] == "success"
    assert payload["execution_mode"] == "math_direct"
    assert payload["deterministic"] is True
    assert payload["model"] == "not-used"
    assert "70" in str(payload["answer"])
    assert payload.get("evaluation", {}).get("verdict") == "PASS"
    _assert_request_id(payload["request_id"])
    assert response.headers.get("X-Request-ID") == str(payload["request_id"])


def test_real_rag_query_respects_document_page_and_public_provenance(
    api_session: Any,
    api_base_url: str,
    visible_sample: dict[str, Any] | None,
    require_non_empty_data: bool,
    service_timeout_seconds: int,
) -> None:
    if visible_sample is None:
        pytest.skip("Nessun chunk reale disponibile per document/page scope")

    body = {
        "query": (
            "Riporta esclusivamente le informazioni documentali pertinenti ai seguenti termini: "
            + str(visible_sample["query"])
        ),
        "conversation_id": "batch13-document-page",
        "history": [],
        "options": {
            "include_sources": True,
            "include_debug": True,
            "include_evaluation": False,
            "target_document": visible_sample["filename"],
            "target_pages": [visible_sample["page"]],
            "max_sources": 8,
        },
    }
    response = _post_query(
        api_session,
        api_base_url,
        body,
        service_timeout_seconds,
    )
    assert response.status_code == 200, response.text[:3000]
    payload = _response_json(response)
    assert payload["status"] == "success"
    assert str(payload.get("answer") or "").strip()
    assert payload["execution_mode"] == "rag_generation"

    sources = payload.get("sources")
    assert isinstance(sources, list)
    if require_non_empty_data:
        assert sources, f"La query scoped non ha restituito fonti: {payload}"

    forbidden_public_fields = {
        "organization_id",
        "status",
        "ingestion_run_id",
        "pg_log_id",
        "pg_chunk_id",
        "embedding_model",
        "content",
    }
    wanted_doc = _normalize_document_name(str(visible_sample["filename"]))
    for source in sources:
        assert isinstance(source, Mapping)
        assert not forbidden_public_fields.intersection(source.keys())
        assert _normalize_document_name(str(source.get("filename") or "")) == wanted_doc
        assert int(source.get("page") or 0) == int(visible_sample["page"])
        assert str(source.get("scope") or "") in {"GLOBAL", "ACCOUNT"}
        assert str(source.get("tier") or "") in {"A", "B", "C", "GRAPH", "USER"}
        assert "\n" not in str(source.get("filename") or "")

    debug = payload.get("debug")
    assert isinstance(debug, Mapping)
    retrieval = debug.get("retrieval")
    assert isinstance(retrieval, Mapping)
    assert _normalize_document_name(str(retrieval.get("target_document") or "")) == wanted_doc
    assert int(retrieval.get("final_sources") or 0) == len(sources)

    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    for secret in (
        os.getenv("PG_PASS", ""),
        os.getenv("NEO4J_PASSWORD", ""),
        os.getenv("OLLAMA_API_KEY", ""),
    ):
        if secret and secret not in {"ollama", "admin_password"}:
            assert secret.casefold() not in serialized


def test_public_api_rejects_client_tenant_fields(
    api_session: Any,
    api_base_url: str,
    service_timeout_seconds: int,
) -> None:
    response = _post_query(
        api_session,
        api_base_url,
        {
            "query": "Definisci il perimetro della richiesta.",
            "organization_id": 999999,
            "options": {"include_sources": False},
        },
        service_timeout_seconds,
    )
    assert response.status_code == 422
    payload = _response_json(response)
    assert payload["status"] == "error"
    assert payload["code"] == "validation_error"
    assert "999999" not in json.dumps(payload, ensure_ascii=False)


def test_live_capacity_returns_service_busy_when_opted_in(
    api_base_url: str,
    visible_sample: dict[str, Any] | None,
    run_capacity_stress: bool,
    service_timeout_seconds: int,
) -> None:
    if not run_capacity_stress:
        pytest.skip("Stress concorrente disabilitato; usare --capacity-stress")
    if visible_sample is None:
        pytest.skip("Nessun documento disponibile per lo stress concorrente")

    import requests

    barrier = threading.Barrier(3)
    body = {
        "query": (
            "Esegui un'analisi dettagliata e documentata dei seguenti contenuti: "
            + str(visible_sample["query"])
        ),
        "history": [],
        "options": {
            "include_sources": False,
            "include_debug": False,
            "target_document": visible_sample["filename"],
            "target_pages": [visible_sample["page"]],
        },
    }

    def _request() -> Any:
        with requests.Session() as session:
            barrier.wait(timeout=10)
            return _post_query(session, api_base_url, body, service_timeout_seconds)

    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(lambda _: _request(), range(3)))

    statuses = [response.status_code for response in responses]
    assert 200 in statuses, statuses
    assert 503 in statuses, (
        "Nessun service_busy osservato. Avviare l'API tramite verify_batch13.py "
        "--start-api --capacity-stress oppure configurare coda=0."
    )
    for response in responses:
        if response.status_code != 503:
            continue
        payload = _response_json(response)
        assert payload["code"] == "service_busy"
        assert payload["retryable"] is True
        assert response.headers.get("Retry-After")
