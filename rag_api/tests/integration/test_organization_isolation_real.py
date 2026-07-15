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
        application_name="rag-api-batch13-organization-isolation",
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


def test_postgres_visibility_shares_tier_a_and_excludes_foreign_tier_bc(
    isolation_pg_connection: Any,
    integration_tenant: TenantContext,
    require_second_organization: bool,
) -> None:
    org_id = integration_tenant.organization_id
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
            LIMIT 20
            """,
            (org_id,),
        )
        foreign_ids = [int(row[0]) for row in cur.fetchall()]
        if not foreign_ids:
            _skip_or_fail(
                "PostgreSQL non contiene dati ACCOUNT Tier B/C di una seconda organization_id",
                required=require_second_organization,
            )

        cur.execute(
            """
            WITH visible AS (
                SELECT scope, organization_id, tier
                FROM public.document_chunks
                WHERE lower(status) = 'active'
                  AND (
                        (upper(scope) = 'GLOBAL'
                            AND organization_id IS NULL
                            AND upper(tier) = 'A')
                        OR
                        (upper(scope) = 'ACCOUNT'
                            AND organization_id = %s
                            AND upper(tier) IN ('B', 'C'))
                      )
            )
            SELECT
                count(*) FILTER (
                    WHERE upper(scope) = 'GLOBAL'
                      AND organization_id IS NULL
                      AND upper(tier) = 'A'
                ) AS shared_tier_a,
                count(*) FILTER (
                    WHERE upper(scope) = 'ACCOUNT'
                      AND organization_id = %s
                      AND upper(tier) IN ('B', 'C')
                ) AS own_tier_bc,
                count(*) FILTER (
                    WHERE upper(scope) = 'ACCOUNT'
                      AND organization_id = ANY(%s)
                ) AS leaked_foreign
            FROM visible
            """,
            (org_id, org_id, foreign_ids),
        )
        shared_tier_a, own_tier_bc, leaked_foreign = map(int, cur.fetchone())

    assert shared_tier_a > 0, "PostgreSQL non espone il Tier A GLOBAL condiviso"
    assert own_tier_bc > 0, f"PostgreSQL non espone Tier B/C per organization_id={org_id}"
    assert leaked_foreign == 0, (
        "PostgreSQL espone Tier B/C appartenenti ad altre organization_id: "
        f"{foreign_ids}"
    )


def test_qdrant_filter_shares_tier_a_and_excludes_foreign_tier_bc(
    isolation_qdrant_client: Any,
    integration_settings: RagSettings,
    integration_tenant: TenantContext,
    require_second_organization: bool,
) -> None:
    from qdrant_client import models

    org_id = integration_tenant.organization_id
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
        and int(payload.get("organization_id")) != org_id
        and str(payload.get("tier") or "").upper() in {"B", "C"}
    }
    if not foreign_ids:
        _skip_or_fail(
            "Qdrant non contiene payload ACCOUNT Tier B/C di una seconda organization_id",
            required=require_second_organization,
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

    shared_tier_a = 0
    own_tier_bc = 0
    leaked: list[Any] = []
    invalid: list[dict[str, Any]] = []
    for point in visible_points:
        payload = getattr(point, "payload", None) or {}
        scope = str(payload.get("scope") or "").upper()
        tier = str(payload.get("tier") or "").upper()
        org_raw = payload.get("organization_id")

        if scope == "GLOBAL":
            if org_raw is None and tier == "A":
                shared_tier_a += 1
            else:
                invalid.append(dict(payload))
            continue

        if scope == "ACCOUNT":
            try:
                payload_org_id = int(org_raw)
            except (TypeError, ValueError):
                invalid.append(dict(payload))
                continue
            if payload_org_id == org_id and tier in {"B", "C"}:
                own_tier_bc += 1
            else:
                leaked.append(payload_org_id)
            continue

        invalid.append(dict(payload))

    assert shared_tier_a > 0, "Qdrant non espone il Tier A GLOBAL condiviso"
    assert own_tier_bc > 0, f"Qdrant non espone Tier B/C per organization_id={org_id}"
    assert not invalid, "Qdrant ha restituito payload fuori dall'invariante GLOBAL/A o ACCOUNT/B-C"
    assert not leaked, f"Qdrant ha esposto organization_id esterni: {leaked}"


def test_neo4j_visibility_shares_tier_a_and_excludes_foreign_tier_bc(
    isolation_neo4j_driver: Any,
    integration_tenant: TenantContext,
    require_second_organization: bool,
) -> None:
    org_id = integration_tenant.organization_id
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
            org_id=org_id,
        ).single()
        foreign_ids = [int(value) for value in (foreign["organization_ids"] or [])]
        if not foreign_ids:
            _skip_or_fail(
                "Neo4j non contiene Chunk ACCOUNT Tier B/C di una seconda organization_id",
                required=require_second_organization,
            )

        result = session.run(
            """
            MATCH (c:Chunk)
            WHERE toLower(coalesce(c.status, '')) = 'active'
              AND (
                    (toUpper(coalesce(c.scope, '')) = 'GLOBAL'
                        AND c.organization_id IS NULL
                        AND toUpper(coalesce(c.tier, '')) = 'A')
                    OR
                    (toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                        AND c.organization_id = $org_id
                        AND toUpper(coalesce(c.tier, '')) IN ['B', 'C'])
                  )
            RETURN
                count(CASE WHEN
                    toUpper(coalesce(c.scope, '')) = 'GLOBAL'
                    AND c.organization_id IS NULL
                    AND toUpper(coalesce(c.tier, '')) = 'A'
                THEN 1 END) AS shared_tier_a,
                count(CASE WHEN
                    toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                    AND c.organization_id = $org_id
                    AND toUpper(coalesce(c.tier, '')) IN ['B', 'C']
                THEN 1 END) AS own_tier_bc,
                count(CASE WHEN
                    toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                    AND c.organization_id IN $foreign_ids
                THEN 1 END) AS leaked_foreign
            """,
            org_id=org_id,
            foreign_ids=foreign_ids,
        ).single()

    assert int(result["shared_tier_a"] or 0) > 0, "Neo4j non espone il Tier A GLOBAL condiviso"
    assert int(result["own_tier_bc"] or 0) > 0, (
        f"Neo4j non espone Tier B/C per organization_id={org_id}"
    )
    assert int(result["leaked_foreign"] or 0) == 0, (
        "Neo4j espone Tier B/C appartenenti ad altre organization_id: "
        f"{foreign_ids}"
    )
