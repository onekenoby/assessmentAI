from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api import routes_rag
from core.config import settings
from core.models import GraphEntity, RagExecutionMode, RagServiceResult, RetrievalDebug, SourceItem
from core.rag_service import RagQueryCommand, RagService


BATCH13_CONTROL_COVERAGE_QUERY = (
    "Una checklist di 20 controlli contiene 12 controlli implementati e "
    "4 controlli parziali che valgono il 50%. Calcola la copertura complessiva."
)


class _NeverRetriever:
    async def retrieve_candidates(self, **kwargs):
        raise AssertionError("math_direct non deve eseguire retrieval")

    async def lookup_glossary(self, **kwargs):
        raise AssertionError("math_direct non deve eseguire glossary lookup")


class _NeverGenerator:
    async def generate_async(self, prompt):
        raise AssertionError("math_direct non deve invocare Ollama")


class _ResourcesWithoutReranker:
    @staticmethod
    def get_reranker():
        return None


class _NeverEvaluator:
    async def evaluate_async(self, **kwargs):
        raise AssertionError("math_direct non deve invocare il judge")


class _NoopAuditor:
    async def persist_query_audit_async(self, audit, **kwargs):
        return SimpleNamespace(success=True, skipped=True)


class _FakeRagService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def query(self, command, *, tenant_context):
        if self.error is not None:
            raise self.error
        return self.result


def _app(tenant_context, fake_service) -> FastAPI:
    app = FastAPI()
    app.include_router(routes_rag.router)
    routes_rag.install_rag_exception_handlers(app)

    async def tenant_override():
        return tenant_context

    app.dependency_overrides[routes_rag.resolve_request_tenant] = tenant_override
    routes_rag.rag_service = fake_service
    return app


def _overlong_source(tenant_context) -> SourceItem:
    return SourceItem(
        id="source-" + "x" * 400,
        doc_id="document-" + "y" * 400,
        content="contenuto " * 800,
        filename="documento-" + "z" * 700 + ".pdf",
        page=2,
        page_chunk_index=1,
        type="text",
        score=0.9,
        graph_context=[
            GraphEntity(
                name="entity-" + "n" * 390,
                type="type-" + "t" * 130,
                relation="REL-" + "r" * 130,
            )
        ],
        section_hint="section-" + "s" * 900,
        tier="B",
        scope="ACCOUNT",
        organization_id=tenant_context.organization_id,
        classification="internal",
        request_id=tenant_context.request_id,
        db_origin="origin-" + "o" * 170,
    )


@pytest.mark.asyncio
async def test_deterministic_requested_evaluation_is_pass_even_when_judge_disabled(
    tenant_context,
) -> None:
    service = RagService(
        config=settings.model_copy(
            update={
                "audit_enabled": False,
                "evaluation_enabled": False,
            }
        ),
        resource_manager=_ResourcesWithoutReranker(),
        retriever=_NeverRetriever(),
        llm_generator=_NeverGenerator(),
        evaluator=_NeverEvaluator(),
        auditor=_NoopAuditor(),
    )

    result = await service.query(
        RagQueryCommand(
            query=BATCH13_CONTROL_COVERAGE_QUERY,
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.MATH_DIRECT
    assert result.evaluation is not None
    assert str(result.evaluation.verdict) == "PASS"
    assert result.model == "not-used"


def test_public_source_mapping_bounds_real_provenance_fields(tenant_context) -> None:
    source = _overlong_source(tenant_context)

    public = routes_rag._source_to_response(source)

    assert len(public.source_id) <= 256
    assert len(public.document_id) <= 256
    assert len(public.filename) <= 512
    assert len(public.excerpt) <= 5_000
    assert len(public.section_hint) <= 500
    assert len(public.database_origin) <= 150
    assert len(public.graph_context[0].name) <= 300
    assert len(public.graph_context[0].type) <= 100
    assert len(public.graph_context[0].relation) <= 100
    assert public.source_id == routes_rag._source_to_response(source).source_id


def test_http_response_with_overlong_internal_source_is_not_false_422(
    monkeypatch,
    tenant_context,
) -> None:
    result = RagServiceResult(
        request_id=UUID(tenant_context.request_id),
        answer=(
            "**A) Risposta**\n\nOK\n\n**B) Evidenze**\n\n- Fonte.\n\n"
            "**C) Limiti / Conflitti**\n\n- Nessuno.\n\n**D) Fonti**\n\n- Fonte."
        ),
        sources=(_overlong_source(tenant_context),),
        retrieval=RetrievalDebug(final_sources=1),
        model="fallback-model",
    )
    fake = _FakeRagService(result=result)
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "Domanda documentale",
                "options": {"include_sources": True},
            },
        )

    assert response.status_code == 200, response.text
    assert len(response.json()["sources"][0]["section_hint"]) <= 500


def test_internal_value_error_is_not_misreported_as_client_validation(
    monkeypatch,
    tenant_context,
) -> None:
    fake = _FakeRagService(error=ValueError("internal projection failure"))
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": "Domanda"})

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "internal projection failure" not in response.text


def test_command_construction_value_error_remains_safe_422(
    monkeypatch,
    tenant_context,
) -> None:
    fake = _FakeRagService(result=None)
    monkeypatch.setattr(routes_rag, "rag_service", fake)
    monkeypatch.setattr(
        routes_rag,
        "_request_to_command",
        lambda payload: (_ for _ in ()).throw(ValueError("bad command")),
    )
    app = _app(tenant_context, fake)

    with TestClient(app) as client:
        response = client.post("/api/v1/rag/query", json={"query": "Domanda"})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert "bad command" not in response.text
