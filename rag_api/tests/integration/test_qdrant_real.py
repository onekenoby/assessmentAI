from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from core.config import RagSettings
from core.tenant import TenantContext, bind_tenant_context, qdrant_payload_is_visible

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qdrant_client(integration_settings: RagSettings) -> Iterator[Any]:
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


def _tenant_filter(tenant: TenantContext) -> Any:
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="active"),
            )
        ],
        should=[
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="scope",
                        match=models.MatchValue(value="GLOBAL"),
                    ),
                    models.FieldCondition(
                        key="tier",
                        match=models.MatchValue(value="A"),
                    ),
                ]
            ),
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="scope",
                        match=models.MatchValue(value="ACCOUNT"),
                    ),
                    models.FieldCondition(
                        key="organization_id",
                        match=models.MatchValue(value=tenant.organization_id),
                    ),
                    models.FieldCondition(
                        key="tier",
                        match=models.MatchAny(any=["B", "C"]),
                    ),
                ]
            ),
        ],
    )


def test_qdrant_collection_exists_and_is_green(
    qdrant_client: Any,
    integration_settings: RagSettings,
) -> None:
    info = qdrant_client.get_collection(integration_settings.qdrant_collection)
    assert info is not None

    status = str(getattr(info, "status", "")).lower()
    assert "green" in status or "ok" in status, f"Collection status inatteso: {status}"

    points_count = int(getattr(info, "points_count", 0) or 0)
    assert points_count >= 0


def test_qdrant_visible_payloads_respect_tenant_policy(
    qdrant_client: Any,
    integration_settings: RagSettings,
    integration_tenant: TenantContext,
    require_non_empty_data: bool,
) -> None:
    points, _ = qdrant_client.scroll(
        collection_name=integration_settings.qdrant_collection,
        scroll_filter=_tenant_filter(integration_tenant),
        limit=20,
        with_payload=True,
        with_vectors=False,
    )

    if require_non_empty_data:
        assert points, "Nessun point Qdrant visibile per il tenant di test"

    with bind_tenant_context(integration_tenant):
        for point in points:
            payload = getattr(point, "payload", None) or {}
            assert isinstance(payload, Mapping)
            assert qdrant_payload_is_visible(payload, context=integration_tenant)
            assert str(payload.get("status") or "").lower() == "active"
            assert str(payload.get("scope") or "").upper() in {"GLOBAL", "ACCOUNT"}
            assert str(payload.get("tier") or "").upper() in {"A", "B", "C"}


def test_qdrant_real_vector_query_returns_visible_points(
    qdrant_client: Any,
    integration_settings: RagSettings,
    integration_tenant: TenantContext,
    require_non_empty_data: bool,
) -> None:
    points, _ = qdrant_client.scroll(
        collection_name=integration_settings.qdrant_collection,
        scroll_filter=_tenant_filter(integration_tenant),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        if require_non_empty_data:
            pytest.fail("Impossibile eseguire vector query: collection visibile vuota")
        pytest.skip("Collection visibile vuota")

    point = points[0]
    raw_vector = getattr(point, "vector", None)
    using: str | None = None
    query_vector: Any = raw_vector

    if isinstance(raw_vector, Mapping):
        if not raw_vector:
            pytest.fail("Il point di esempio non contiene vettori")
        using, query_vector = next(iter(raw_vector.items()))

    if query_vector is None:
        pytest.fail("Il point di esempio non contiene un vettore utilizzabile")

    kwargs: dict[str, Any] = {
        "collection_name": integration_settings.qdrant_collection,
        "query": query_vector,
        "query_filter": _tenant_filter(integration_tenant),
        "limit": 3,
        "with_payload": True,
        "with_vectors": False,
    }
    if using:
        kwargs["using"] = using

    response = qdrant_client.query_points(**kwargs)
    hits = list(getattr(response, "points", ()) or ())
    assert hits, "La vector query reale non ha restituito point"

    with bind_tenant_context(integration_tenant):
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            assert qdrant_payload_is_visible(payload, context=integration_tenant)
