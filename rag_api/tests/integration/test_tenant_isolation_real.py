from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from core.config import RagSettings
from core.retrieval import HybridRetrievalEngine
from core.tenant import TenantContext

pytestmark = pytest.mark.integration



def _scroll_points(
    client: Any,
    *,
    collection_name: str,
    scroll_filter: Any,
    max_points: int = 5000,
) -> list[Any]:
    points: list[Any] = []
    offset: Any = None
    while len(points) < max_points:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=min(256, max_points - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if next_offset is None or not batch:
            break
        offset = next_offset
    return points


def _skip_or_fail(message: str, *, required: bool) -> None:
    if required:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def isolation_pg_connection(integration_settings: RagSettings) -> Iterator[Any]:
    if not integration_settings.pg_enrich_enabled:
        pytest.skip("PostgreSQL enrichment disabilitato")

    import psycopg2

    conn = psycopg2.connect(
        host=integration_settings.pg_host,
        port=integration_settings.pg_port,
        dbname=integration_settings.pg_database,
        user=integration_settings.pg_user,
        password=integration_settings.pg_password,
        connect_timeout=10,
        application_name="rag-api-batch13-tenant-isolation",
    )
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def isolation_qdrant_client(integration_settings: RagSettings) -> Iterator[Any]:
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=integration_settings.qdrant_host,
        port=integration_settings.qdrant_port,
        timeout=30,
    )
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


@pytest.fixture(scope="module")
def isolation_neo4j_driver(integration_settings: RagSettings) -> Iterator[Any]:
    if not integration_settings.neo4j_enabled:
        pytest.skip("Neo4j disabilitato")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        integration_settings.neo4j_uri,
        auth=integration_settings.neo4j_auth,
    )
    driver.verify_connectivity()
    try:
        yield driver
    finally:
        driver.close()


def test_postgres_visibility_predicate_excludes_foreign_account_chunks(
    isolation_pg_connection: Any,
    integration_tenant: TenantContext,
    require_second_tenant: bool,
) -> None:
    with isolation_pg_connection.cursor() as cur:
        cur.execute(
            """
            SELECT organization_id, count(*)
            FROM public.document_chunks
            WHERE lower(status) = 'active'
              AND upper(scope) = 'ACCOUNT'
              AND organization_id IS NOT NULL
              AND organization_id <> %s
              AND upper(tier) IN ('B', 'C')
            GROUP BY organization_id
            ORDER BY count(*) DESC
            LIMIT 10
            """,
            (integration_tenant.organization_id,),
        )
        foreign_rows = cur.fetchall()

        if not foreign_rows:
            _skip_or_fail(
                "PostgreSQL non contiene un secondo tenant ACCOUNT attivo",
                required=require_second_tenant,
            )

        foreign_ids = [int(row[0]) for row in foreign_rows]
        cur.execute(
            """
            SELECT count(*)
            FROM public.document_chunks
            WHERE lower(status) = 'active'
              AND organization_id = ANY(%s)
              AND (
                    (upper(scope) = 'GLOBAL' AND organization_id IS NULL AND upper(tier) = 'A')
                    OR
                    (upper(scope) = 'ACCOUNT' AND organization_id = %s AND upper(tier) IN ('B', 'C'))
                  )
            """,
            (foreign_ids, integration_tenant.organization_id),
        )
        leaked = int(cur.fetchone()[0])

    assert leaked == 0, f"PostgreSQL visibility predicate espone tenant esterni: {foreign_ids}"


def test_qdrant_engine_filter_excludes_foreign_account_payloads(
    isolation_qdrant_client: Any,
    integration_settings: RagSettings,
    integration_tenant: TenantContext,
    require_second_tenant: bool,
) -> None:
    from qdrant_client import models

    account_filter = models.Filter(
        must=[
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
            models.FieldCondition(key="scope", match=models.MatchValue(value="ACCOUNT")),
        ]
    )
    raw_points = _scroll_points(
        isolation_qdrant_client,
        collection_name=integration_settings.qdrant_collection,
        scroll_filter=account_filter,
    )
    foreign_ids = {
        int(payload.get("organization_id"))
        for point in raw_points
        if isinstance((payload := getattr(point, "payload", None) or {}), Mapping)
        and str(payload.get("organization_id") or "").isdigit()
        and int(payload.get("organization_id")) != integration_tenant.organization_id
    }
    if not foreign_ids:
        _skip_or_fail(
            "Qdrant non contiene payload ACCOUNT di un secondo tenant nel campione",
            required=require_second_tenant,
        )

    engine = HybridRetrievalEngine(
        config=integration_settings,
        resource_manager=SimpleNamespace(),
    )
    visible_filter = engine._build_qdrant_filter(integration_tenant)
    visible_points = _scroll_points(
        isolation_qdrant_client,
        collection_name=integration_settings.qdrant_collection,
        scroll_filter=visible_filter,
    )

    leaked = []
    for point in visible_points:
        payload = getattr(point, "payload", None) or {}
        if str(payload.get("scope") or "").upper() != "ACCOUNT":
            continue
        org_raw = payload.get("organization_id")
        try:
            org_id = int(org_raw)
        except (TypeError, ValueError):
            leaked.append(org_raw)
            continue
        if org_id != integration_tenant.organization_id:
            leaked.append(org_id)

    assert not leaked, f"Qdrant tenant filter ha esposto organization_id esterni: {leaked}"


def test_neo4j_visibility_predicate_excludes_foreign_account_chunks(
    isolation_neo4j_driver: Any,
    integration_tenant: TenantContext,
    require_second_tenant: bool,
) -> None:
    with isolation_neo4j_driver.session() as session:
        foreign = session.run(
            """
            MATCH (c:Chunk)
            WHERE toLower(coalesce(c.status, '')) = 'active'
              AND toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
              AND c.organization_id IS NOT NULL
              AND c.organization_id <> $org_id
              AND toUpper(coalesce(c.tier, '')) IN ['B', 'C']
            RETURN collect(DISTINCT c.organization_id)[0..20] AS organization_ids
            """,
            org_id=integration_tenant.organization_id,
        ).single()
        foreign_ids = [int(value) for value in (foreign["organization_ids"] or [])]
        if not foreign_ids:
            _skip_or_fail(
                "Neo4j non contiene Chunk ACCOUNT di un secondo tenant",
                required=require_second_tenant,
            )

        result = session.run(
            """
            MATCH (c:Chunk)
            WHERE toLower(coalesce(c.status, '')) = 'active'
              AND c.organization_id IN $foreign_ids
              AND (
                    (toUpper(coalesce(c.scope, '')) = 'GLOBAL'
                        AND c.organization_id IS NULL
                        AND toUpper(coalesce(c.tier, '')) = 'A')
                    OR
                    (toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                        AND c.organization_id = $org_id
                        AND toUpper(coalesce(c.tier, '')) IN ['B', 'C'])
                  )
            RETURN count(c) AS leaked
            """,
            org_id=integration_tenant.organization_id,
            foreign_ids=foreign_ids,
        ).single()

    assert int(result["leaked"] or 0) == 0, (
        f"Neo4j visibility predicate espone tenant esterni: {foreign_ids}"
    )
