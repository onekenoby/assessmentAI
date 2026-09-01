from __future__ import annotations

import os

import pytest
import requests

pytestmark = pytest.mark.integration


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def test_source_postgres_ping_and_worker_contract():
    import psycopg2

    expected = {
        "fn_claim_next_ingestion_job": 1,
        "fn_get_claimed_job_payload": 3,
        "fn_heartbeat_ingestion_job": 3,
        "fn_complete_ingestion_job": 6,
        "fn_fail_ingestion_job": 6,
    }
    conn = psycopg2.connect(
        host=_env("SOURCE_PG_HOST", _env("PG_HOST", "127.0.0.1")),
        port=_env_int("SOURCE_PG_PORT", _env_int("PG_PORT", 5433)),
        dbname=_env("SOURCE_PG_DB", "assessment_gestio_tier"),
        user=_env("SOURCE_PG_USER", _env("PG_USER", "admin")),
        password=_env("SOURCE_PG_PASS", _env("PG_PASS", "admin_password")),
        connect_timeout=10,
        application_name="ingestion-api-integration-contract",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            database, user = cur.fetchone()
            assert database == _env("SOURCE_PG_DB", "assessment_gestio_tier")
            assert str(user)
            cur.execute(
                """
                SELECT p.proname, p.pronargs
                FROM pg_catalog.pg_proc p
                JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'rag_ingestion'
                  AND p.proname = ANY(%s)
                """,
                (list(expected),),
            )
            found: dict[str, set[int]] = {}
            for name, count in cur.fetchall():
                found.setdefault(str(name), set()).add(int(count))
        for name, count in expected.items():
            assert count in found.get(name, set()), f"Firma mancante: {name}/{count}"
    finally:
        conn.close()


def test_output_postgres_ping_and_required_tables():
    import psycopg2

    conn = psycopg2.connect(
        host=_env("PG_HOST", "127.0.0.1"),
        port=_env_int("PG_PORT", 5433),
        dbname=_env("PG_DB", "assessment_ingestion"),
        user=_env("PG_USER", "admin"),
        password=_env("PG_PASS", "admin_password"),
        connect_timeout=10,
        application_name="ingestion-api-integration-output",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.ingestion_logs'), "
                "to_regclass('public.document_chunks'), "
                "to_regclass('public.ingestion_images')"
            )
            assert all(value is not None for value in cur.fetchone())
    finally:
        conn.close()


def test_ollama_is_reachable_and_models_are_listed():
    base = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    response = requests.get(base + "/api/tags", timeout=10)
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload.get("models"), list)


def test_qdrant_is_reachable():
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=_env("QDRANT_HOST", "127.0.0.1"),
        port=_env_int("QDRANT_PORT", 6334),
        timeout=20,
    )
    try:
        collections = client.get_collections()
        assert collections is not None
    finally:
        if hasattr(client, "close"):
            client.close()


def test_neo4j_is_reachable_when_enabled():
    if _env("NEO4J_ENABLED", "1") != "1":
        pytest.skip("Neo4j disabilitato")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        _env("NEO4J_URI", "bolt://127.0.0.1:7688"),
        auth=(
            _env("NEO4J_USER", "neo4j"),
            _env("NEO4J_PASS", "admin_password"),
        ),
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()
