from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from core.config import RagSettings
from core.tenant import TenantContext

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def neo4j_driver(integration_settings: RagSettings) -> Iterator[Any]:
    if not integration_settings.neo4j_enabled:
        pytest.skip("Neo4j disabilitato dalla configurazione")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        integration_settings.neo4j_uri,
        auth=integration_settings.neo4j_auth,
    )
    try:
        driver.verify_connectivity()
        yield driver
    finally:
        driver.close()


def test_neo4j_ping_and_server_info(neo4j_driver: Any) -> None:
    info = neo4j_driver.get_server_info()
    assert str(getattr(info, "agent", "")).strip()

    with neo4j_driver.session() as session:
        value = session.run("RETURN 1 AS value").single()["value"]
    assert value == 1


def test_neo4j_expected_graph_content(
    neo4j_driver: Any,
    require_non_empty_data: bool,
) -> None:
    with neo4j_driver.session() as session:
        row = session.run(
            """
            MATCH (n)
            WITH count(n) AS total_nodes
            OPTIONAL MATCH (c:Chunk)
            WITH total_nodes, count(c) AS chunks
            OPTIONAL MATCH (e:Entity)
            WITH total_nodes, chunks, count(e) AS entities
            OPTIONAL MATCH ()-[r]->()
            RETURN total_nodes, chunks, entities, count(r) AS relationships
            """
        ).single()

    total_nodes = int(row["total_nodes"] or 0)
    chunks = int(row["chunks"] or 0)
    entities = int(row["entities"] or 0)
    relationships = int(row["relationships"] or 0)

    assert min(total_nodes, chunks, entities, relationships) >= 0
    if require_non_empty_data:
        assert total_nodes > 0, "Neo4j non contiene nodi"
        assert chunks > 0, "Neo4j non contiene nodi :Chunk"
        assert entities > 0, "Neo4j non contiene nodi :Entity"


def test_neo4j_active_chunks_respect_tenant_invariants(
    neo4j_driver: Any,
    integration_tenant: TenantContext,
    require_non_empty_data: bool,
) -> None:
    with neo4j_driver.session() as session:
        row = session.run(
            """
            MATCH (c:Chunk)
            WHERE toLower(coalesce(c.status, '')) = 'active'
            RETURN
                count(c) AS total_active,
                count(CASE WHEN NOT (
                    (toUpper(coalesce(c.scope, '')) = 'GLOBAL'
                        AND c.organization_id IS NULL
                        AND toUpper(coalesce(c.tier, '')) = 'A')
                    OR
                    (toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                        AND c.organization_id IS NOT NULL
                        AND toUpper(coalesce(c.tier, '')) IN ['B', 'C'])
                ) THEN 1 END) AS invalid_active,
                count(CASE WHEN
                    (toUpper(coalesce(c.scope, '')) = 'GLOBAL'
                        AND c.organization_id IS NULL
                        AND toUpper(coalesce(c.tier, '')) = 'A')
                    OR
                    (toUpper(coalesce(c.scope, '')) = 'ACCOUNT'
                        AND c.organization_id = $org_id
                        AND toUpper(coalesce(c.tier, '')) IN ['B', 'C'])
                THEN 1 END) AS visible_for_tenant
            """,
            org_id=integration_tenant.organization_id,
        ).single()

    total_active = int(row["total_active"] or 0)
    invalid_active = int(row["invalid_active"] or 0)
    visible_for_tenant = int(row["visible_for_tenant"] or 0)

    assert invalid_active == 0, (
        f"Trovati {invalid_active} Chunk active con scope/tier/org non validi"
    )
    if require_non_empty_data:
        assert total_active > 0
        assert visible_for_tenant > 0, (
            f"Nessun Chunk Neo4j visibile per organization_id={integration_tenant.organization_id}"
        )


def test_neo4j_active_relationship_types_are_whitelisted(
    neo4j_driver: Any,
    integration_settings: RagSettings,
) -> None:
    with neo4j_driver.session() as session:
        row = session.run(
            """
            MATCH (:Entity)-[r]->(:Entity)
            WHERE toLower(coalesce(r.status, 'active')) = 'active'
              AND NOT type(r) IN $allowed
            RETURN count(r) AS invalid_count,
                   collect(DISTINCT type(r))[0..20] AS invalid_types
            """,
            allowed=list(integration_settings.neo4j_allowed_relationships),
        ).single()

    invalid_count = int(row["invalid_count"] or 0)
    invalid_types = list(row["invalid_types"] or [])
    assert invalid_count == 0, (
        f"Relazioni active fuori whitelist: count={invalid_count}, types={invalid_types}"
    )
