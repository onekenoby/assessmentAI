from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import settings
from core.models import RagAnswerMode, RagIntent, RetrievalCandidate
from core.retrieval import HybridRetrievalEngine, RetrievalBackendError


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True):
        return [0.1, 0.2, 0.3]


class FakeQdrant:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.hits)


class FakeResources:
    def __init__(self, qdrant=None, neo4j=None):
        self.qdrant = qdrant
        self.neo4j = neo4j

    def get_embedder(self):
        return FakeEmbedder()

    def get_qdrant_client(self):
        return self.qdrant

    def get_neo4j_driver(self, required=False):
        return self.neo4j


class StubEngine(HybridRetrievalEngine):
    def __init__(self, *, config, resource_manager, qdrant_hits=(), pg_hits=(), fail_qdrant=False, fail_pg=False):
        super().__init__(config=config, resource_manager=resource_manager)
        self.stub_qdrant_hits = list(qdrant_hits)
        self.stub_pg_hits = list(pg_hits)
        self.fail_qdrant = fail_qdrant
        self.fail_pg = fail_pg

    def _search_qdrant(self, *args, **kwargs):
        if self.fail_qdrant:
            raise RuntimeError("qdrant down")
        return list(self.stub_qdrant_hits)

    def _search_pg_bm25(self, *args, **kwargs):
        if self.fail_pg:
            raise RuntimeError("pg down")
        return list(self.stub_pg_hits)

    def _search_pg_exact_phrases(self, *args, **kwargs):
        return []

    def _search_pg_document_scope(self, *args, **kwargs):
        return []

    def _search_pg_glossary_term(self, *args, **kwargs):
        return []

    def _fetch_pg_chunks_by_uuid(self, *args, **kwargs):
        return {}


def _candidate(identifier, *, org=1234, filename="doc.pdf", page=1, score_vec=0.5, score_bm25=0):
    return RetrievalCandidate(
        id=identifier,
        content=f"Contenuto {identifier}",
        filename=filename,
        page=page,
        doc_id=f"doc-{identifier}",
        tier="C",
        scope="ACCOUNT",
        organization_id=org,
        status="active",
        origin="stub",
        score_vec=score_vec,
        score_bm25=score_bm25,
    )


def test_qdrant_search_filters_foreign_tenant(tenant_context):
    valid_payload = {
        "text_sem": "contenuto valido",
        "filename": "valid.pdf",
        "page": 1,
        "doc_id": "doc-valid",
        "tier": "C",
        "scope": "ACCOUNT",
        "organization_id": 1234,
        "status": "active",
    }
    foreign_payload = {**valid_payload, "filename": "foreign.pdf", "organization_id": 9999}
    hits = [
        SimpleNamespace(id="valid", score=0.9, payload=valid_payload),
        SimpleNamespace(id="foreign", score=0.99, payload=foreign_payload),
    ]
    client = FakeQdrant(hits)
    engine = HybridRetrievalEngine(
        config=settings.model_copy(update={"pg_enrich_enabled": False, "neo4j_enabled": False}),
        resource_manager=FakeResources(qdrant=client),
    )

    result = engine._search_qdrant("query", limit=10, tenant_context=tenant_context)

    assert [item.id for item in result] == ["valid"]
    assert client.calls[0]["collection_name"] == settings.qdrant_collection
    assert client.calls[0]["with_payload"] is True


@pytest.mark.asyncio
async def test_retrieval_merges_same_candidate_from_qdrant_and_postgres(tenant_context):
    q = _candidate("same", score_vec=0.8)
    pg = _candidate("same", score_bm25=2.0)
    config = settings.model_copy(update={"pg_enrich_enabled": True, "neo4j_enabled": False})
    engine = StubEngine(config=config, resource_manager=FakeResources(), qdrant_hits=[q], pg_hits=[pg])

    candidates, debug = await engine.retrieve_candidates(
        query="incident response",
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        target_document=None,
        target_pages=(),
        wants_evidence=True,
        graph_relation_mode=False,
        formula_mode=False,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert len(candidates) == 1
    assert candidates[0].score_vec == pytest.approx(0.8)
    assert candidates[0].score_bm25 == pytest.approx(2.0)
    assert debug.qdrant_hits == 1
    assert debug.postgres_bm25_hits == 1


@pytest.mark.asyncio
async def test_retrieval_applies_document_and_page_scope(tenant_context):
    hits = [
        _candidate("a", filename="evidence.pdf", page=2),
        _candidate("b", filename="other.pdf", page=2),
        _candidate("c", filename="evidence.pdf", page=3),
    ]
    config = settings.model_copy(update={"pg_enrich_enabled": False, "neo4j_enabled": False})
    engine = StubEngine(config=config, resource_manager=FakeResources(), qdrant_hits=hits)

    candidates, debug = await engine.retrieve_candidates(
        query="verifica",
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        target_document="evidence.pdf",
        target_pages=(2,),
        wants_evidence=True,
        graph_relation_mode=False,
        formula_mode=False,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert [item.id for item in candidates] == ["a"]
    assert any("Document scope" in warning for warning in debug.warnings)
    assert any("Page scope" in warning for warning in debug.warnings)


@pytest.mark.asyncio
async def test_retrieval_degrades_when_one_backend_fails(tenant_context):
    config = settings.model_copy(update={"pg_enrich_enabled": True, "neo4j_enabled": False})
    engine = StubEngine(
        config=config,
        resource_manager=FakeResources(),
        qdrant_hits=[],
        pg_hits=[_candidate("pg", score_bm25=1.5)],
        fail_qdrant=True,
    )

    candidates, debug = await engine.retrieve_candidates(
        query="audit log",
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        target_document=None,
        target_pages=(),
        wants_evidence=True,
        graph_relation_mode=False,
        formula_mode=False,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert [item.id for item in candidates] == ["pg"]
    assert any("Qdrant non disponibile" in warning for warning in debug.warnings)


@pytest.mark.asyncio
async def test_retrieval_raises_when_all_attempted_backends_fail(tenant_context):
    config = settings.model_copy(update={"pg_enrich_enabled": True, "neo4j_enabled": False})
    engine = StubEngine(
        config=config,
        resource_manager=FakeResources(),
        fail_qdrant=True,
        fail_pg=True,
    )

    with pytest.raises(RetrievalBackendError):
        await engine.retrieve_candidates(
            query="audit log",
            intent=RagIntent.AUDIT,
            answer_mode=RagAnswerMode.AUDIT,
            target_document=None,
            target_pages=(),
            wants_evidence=True,
            graph_relation_mode=False,
            formula_mode=False,
            exhaustive_formula_lookup=False,
            tenant_context=tenant_context,
        )
