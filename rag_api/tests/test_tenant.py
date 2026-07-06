from __future__ import annotations

from uuid import uuid4

import pytest

from core.tenant import (
    TenantContextError,
    TrustedTenantIdentity,
    bind_tenant_context,
    build_tenant_key,
    current_organization_id,
    filter_visible_records,
    get_tenant_context,
    resolve_tenant_context,
    tenant_metadata,
    tenant_record_is_visible,
    validate_tier_scope_invariants,
)


def test_context_is_fail_closed_without_binding() -> None:
    with pytest.raises(TenantContextError):
        get_tenant_context()


def test_identity_and_context_are_normalized() -> None:
    identity = TrustedTenantIdentity(
        organization_id=1234,
        user_id=" user-1 ",
        roles=("Auditor", "auditor", " USER "),
        allowed_scopes=("ACCOUNT", "GLOBAL", "ACCOUNT"),
    )

    context = resolve_tenant_context(
        identity=identity,
        request_id=str(uuid4()),
    )

    assert context.organization_id == 1234
    assert context.user_id == "user-1"
    assert context.roles == ("auditor", "user")
    assert context.allowed_scopes == ("ACCOUNT", "GLOBAL")
    assert context.tenant_key == "ORG:1234"


def test_binding_restores_previous_context(tenant_context, other_tenant_context) -> None:
    with bind_tenant_context(tenant_context):
        assert current_organization_id() == 1234
        with bind_tenant_context(other_tenant_context):
            assert current_organization_id() == 9999
        assert current_organization_id() == 1234

    with pytest.raises(TenantContextError):
        current_organization_id()


@pytest.mark.parametrize(
    ("scope", "organization_id", "tier", "expected"),
    [
        ("GLOBAL", None, "A", ("GLOBAL", None, "A")),
        ("ACCOUNT", 1234, "B", ("ACCOUNT", 1234, "B")),
        ("ACCOUNT", "1234", "C", ("ACCOUNT", 1234, "C")),
    ],
)
def test_tier_scope_invariants_accept_valid_combinations(
    scope, organization_id, tier, expected
) -> None:
    assert validate_tier_scope_invariants(
        scope=scope,
        organization_id=organization_id,
        tier=tier,
    ) == expected


@pytest.mark.parametrize(
    ("scope", "organization_id", "tier"),
    [
        ("GLOBAL", 1234, "A"),
        ("GLOBAL", None, "B"),
        ("ACCOUNT", None, "C"),
        ("ACCOUNT", 1234, "A"),
    ],
)
def test_tier_scope_invariants_reject_invalid_combinations(
    scope, organization_id, tier
) -> None:
    with pytest.raises(TenantContextError):
        validate_tier_scope_invariants(
            scope=scope,
            organization_id=organization_id,
            tier=tier,
        )


def test_visibility_rules_are_fail_closed(tenant_context) -> None:
    assert tenant_record_is_visible(
        scope="GLOBAL",
        organization_id=None,
        tier="A",
        context=tenant_context,
    )
    assert tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="B",
        context=tenant_context,
    )
    assert tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="C",
        context=tenant_context,
    )

    assert not tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=9999,
        tier="C",
        context=tenant_context,
    )
    assert not tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="C",
        status="inactive",
        context=tenant_context,
    )
    assert not tenant_record_is_visible(
        scope="GLOBAL",
        organization_id=None,
        tier="GRAPH",
        context=tenant_context,
    )
    assert tenant_record_is_visible(
        scope="GLOBAL",
        organization_id=None,
        tier="GRAPH",
        context=tenant_context,
        allow_graph_tier=True,
    )


def test_user_sources_require_allow_flag_and_known_id(tenant_context) -> None:
    assert not tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="USER",
        source_id="user_input",
        context=tenant_context,
    )
    assert tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="USER",
        source_id="user_input",
        context=tenant_context,
        allow_user_tier=True,
    )
    assert not tenant_record_is_visible(
        scope="ACCOUNT",
        organization_id=1234,
        tier="USER",
        source_id="arbitrary-id",
        context=tenant_context,
        allow_user_tier=True,
    )


def test_filter_visible_records_removes_foreign_tenant(
    tenant_context, source_a, source_b, source_c, foreign_source
) -> None:
    visible = filter_visible_records(
        [source_a, source_b, source_c, foreign_source],
        context=tenant_context,
    )
    assert [source.id for source in visible] == ["source-a", "source-b", "source-c"]


def test_tenant_metadata_is_normalized() -> None:
    metadata = tenant_metadata(
        scope=" account ",
        organization_id="1234",
        tier="c",
        corpus_version="v-test",
    )
    assert metadata == {
        "scope": "ACCOUNT",
        "organization_id": 1234,
        "tier": "C",
        "tenant_key": "ORG:1234",
        "status": "active",
        "corpus_version": "v-test",
    }
    assert build_tenant_key("GLOBAL", None) == "GLOBAL"
