from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.config import RagSettings
from core.tenant import TenantContext

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_connection(integration_settings: RagSettings) -> Iterator[object]:
    import psycopg2

    conn = psycopg2.connect(
        host=integration_settings.pg_host,
        port=integration_settings.pg_port,
        dbname=integration_settings.pg_database,
        user=integration_settings.pg_user,
        password=integration_settings.pg_password,
        connect_timeout=10,
        application_name="rag-api-real-integration-test",
    )
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def test_postgres_ping_and_identity(pg_connection: object, integration_settings: RagSettings) -> None:
    with pg_connection.cursor() as cur:
        cur.execute(
            "SELECT 1, current_database(), current_user, "
            "current_setting('server_version')"
        )
        one, database, user, version = cur.fetchone()

    assert one == 1
    assert database == integration_settings.pg_database
    assert user == integration_settings.pg_user
    assert str(version).strip()


def test_postgres_required_schema_rls_and_policies(pg_connection: object) -> None:
    required_columns = {
        "status",
        "ingestion_run_id",
        "tenant_key",
        "corpus_version",
        "classification",
        "embedding_model",
        "organization_id",
        "tier",
        "scope",
        "chunk_uuid",
        "content_raw",
        "content_semantic",
        "metadata_json",
    }

    with pg_connection.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.document_chunks'), "
            "to_regclass('public.rag_query_audit')"
        )
        document_chunks, rag_query_audit = cur.fetchone()
        assert document_chunks is not None
        assert rag_query_audit is not None

        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'document_chunks'
            """
        )
        existing = {str(row[0]) for row in cur.fetchall()}
        assert not (required_columns - existing), (
            "Colonne mancanti: " + ", ".join(sorted(required_columns - existing))
        )

        for table_name in ("document_chunks", "rag_query_audit"):
            cur.execute(
                """
                SELECT relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE oid = %s::regclass
                """,
                (f"public.{table_name}",),
            )
            row = cur.fetchone()
            assert row is not None
            assert row == (True, True), f"RLS/FORCE RLS non attive su {table_name}: {row}"

        expected_policies = {
            ("document_chunks", "document_chunks_tenant_select"),
            ("rag_query_audit", "rag_query_audit_tenant_all"),
        }
        cur.execute(
            """
            SELECT tablename, policyname
            FROM pg_policies
            WHERE schemaname = 'public'
              AND (
                    (tablename = 'document_chunks' AND policyname = 'document_chunks_tenant_select')
                    OR
                    (tablename = 'rag_query_audit' AND policyname = 'rag_query_audit_tenant_all')
                  )
            """
        )
        found = {(str(row[0]), str(row[1])) for row in cur.fetchall()}
        assert found == expected_policies


def test_postgres_tenant_guc_roundtrip(
    pg_connection: object,
    integration_tenant: TenantContext,
) -> None:
    with pg_connection.cursor() as cur:
        cur.execute(
            "SELECT set_config('app.current_customer_account_id', %s, false), "
            "set_config('app.current_request_id', %s, false)",
            (str(integration_tenant.organization_id), integration_tenant.request_id),
        )
        cur.execute(
            "SELECT current_setting('app.current_customer_account_id', true), "
            "current_setting('app.current_request_id', true)"
        )
        organization_id, request_id = cur.fetchone()

        assert organization_id == str(integration_tenant.organization_id)
        assert request_id == integration_tenant.request_id

        cur.execute("RESET app.current_customer_account_id")
        cur.execute("RESET app.current_request_id")


def test_postgres_active_data_respects_tier_scope_invariants(
    pg_connection: object,
    integration_tenant: TenantContext,
    require_non_empty_data: bool,
) -> None:
    with pg_connection.cursor() as cur:
        cur.execute(
            """
            SELECT
                count(*) AS total_active,
                count(*) FILTER (
                    WHERE NOT (
                        (upper(scope) = 'GLOBAL' AND organization_id IS NULL AND upper(tier) = 'A')
                        OR
                        (upper(scope) = 'ACCOUNT' AND organization_id IS NOT NULL AND upper(tier) IN ('B', 'C'))
                    )
                ) AS invalid_active,
                count(*) FILTER (
                    WHERE
                        (upper(scope) = 'GLOBAL' AND organization_id IS NULL AND upper(tier) = 'A')
                        OR
                        (upper(scope) = 'ACCOUNT' AND organization_id = %s AND upper(tier) IN ('B', 'C'))
                ) AS visible_for_tenant
            FROM public.document_chunks
            WHERE lower(status) = 'active'
            """,
            (integration_tenant.organization_id,),
        )
        total_active, invalid_active, visible_for_tenant = map(int, cur.fetchone())

    assert invalid_active == 0, (
        f"Trovati {invalid_active} chunk active con combinazione scope/tier/org non valida"
    )
    if require_non_empty_data:
        assert total_active > 0, "document_chunks non contiene chunk active"
        assert visible_for_tenant > 0, (
            f"Nessun chunk visibile per organization_id={integration_tenant.organization_id}"
        )
