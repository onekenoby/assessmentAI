from __future__ import annotations

from uuid import uuid4

import pytest

from core.models import RetrievalCandidate, SourceItem
from core.tenant import TenantContext


@pytest.fixture
def tenant_context() -> TenantContext:
    return TenantContext(
        organization_id=1234,
        user_id="unit-test-user",
        roles=("Auditor", "auditor", "USER"),
        request_id=str(uuid4()),
        allowed_scopes=("GLOBAL", "ACCOUNT"),
    )


@pytest.fixture
def other_tenant_context() -> TenantContext:
    return TenantContext(
        organization_id=9999,
        user_id="other-tenant-user",
        roles=("user",),
        request_id=str(uuid4()),
        allowed_scopes=("GLOBAL", "ACCOUNT"),
    )


@pytest.fixture
def source_a() -> SourceItem:
    return SourceItem(
        id="source-a",
        content="Il requisito normativo impone una procedura formalizzata.",
        filename="normativa.pdf",
        page=2,
        page_chunk_index=1,
        doc_id="doc-a",
        type="text",
        tier="A",
        scope="GLOBAL",
        organization_id=None,
        status="active",
        corpus_version="v1",
        db_origin="Qdrant",
    )


@pytest.fixture
def source_b() -> SourceItem:
    return SourceItem(
        id="source-b",
        content="La policy interna definisce ruoli e responsabilità.",
        filename="policy.pdf",
        page=4,
        page_chunk_index=1,
        doc_id="doc-b",
        type="text",
        tier="B",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        corpus_version="v1",
        db_origin="PostgresBM25",
    )


@pytest.fixture
def source_c() -> SourceItem:
    return SourceItem(
        id="source-c",
        content="Il log dimostra che il test della procedura è stato eseguito.",
        filename="evidenza_test.pdf",
        page=7,
        page_chunk_index=2,
        doc_id="doc-c",
        type="text",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        corpus_version="v1",
        db_origin="Qdrant",
    )


@pytest.fixture
def foreign_source() -> SourceItem:
    return SourceItem(
        id="foreign-source",
        content="Contenuto appartenente a un altro tenant.",
        filename="foreign.pdf",
        page=1,
        doc_id="foreign-doc",
        type="text",
        tier="C",
        scope="ACCOUNT",
        organization_id=9999,
        status="active",
        corpus_version="v1",
        db_origin="Qdrant",
    )


@pytest.fixture
def candidate_factory():
    def _factory(
        identifier: str,
        *,
        filename: str = "documento.pdf",
        page: int = 1,
        page_chunk_index: int = 0,
        doc_id: str | None = None,
        tier: str = "C",
        scope: str | None = None,
        organization_id: int | None = None,
        score_vec: float = 0.0,
        score_bm25: float = 0.0,
        score_graph: float = 0.0,
        final_score: float = 0.0,
        source_type: str = "text",
        content: str | None = None,
    ) -> RetrievalCandidate:
        resolved_scope = scope or ("GLOBAL" if tier == "A" else "ACCOUNT")
        resolved_org = (
            None if resolved_scope == "GLOBAL" else (organization_id or 1234)
        )
        return RetrievalCandidate(
            id=identifier,
            content=content or f"Contenuto del candidato {identifier}",
            filename=filename,
            page=page,
            page_chunk_index=page_chunk_index,
            doc_id=doc_id or f"doc-{identifier}",
            type=source_type,
            tier=tier,
            scope=resolved_scope,
            organization_id=resolved_org,
            status="active",
            origin="unit-test",
            score_vec=score_vec,
            score_bm25=score_bm25,
            score_graph=score_graph,
            final_score=final_score,
        )

    return _factory
