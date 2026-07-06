from __future__ import annotations

from hashlib import sha256

import pytest

from core.audit import AuditSink, AuditSinkOutcome, AuditWriteResult
from core.config import settings
from core.generation import GenerationMetrics, GenerationResult
from core.models import RagAnswerMode, RagExecutionMode, RagIntent, RetrievalDebug
from core.rag_service import RagQueryCommand, RagService, RagServiceGenerationError
from core.validation import RagEvalResult


class FakeResources:
    def __init__(self, reranker=None):
        self.reranker = reranker

    def get_reranker(self):
        return self.reranker


class FakeRetriever:
    def __init__(self, candidates=(), debug=None, glossary=None):
        self.candidates = tuple(candidates)
        self.debug = debug or RetrievalDebug()
        self.glossary = glossary
        self.calls = 0

    async def retrieve_candidates(self, **kwargs):
        self.calls += 1
        return self.candidates, self.debug

    async def lookup_glossary(self, **kwargs):
        return self.glossary


class FakeGenerator:
    def __init__(self, content=None, error=None):
        self.content = content or (
            "**A) Risposta**\n\nLa procedura è formalizzata.\n\n"
            "**B) Evidenze**\n\nEvidenza disponibile.\n\n"
            "**C) Limiti / Conflitti**\n\nNessuno.\n\n"
            "**D) Fonti**\n\n- placeholder"
        )
        self.error = error
        self.calls = 0

    async def generate_async(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return GenerationResult(
            content=self.content,
            model="gemma4:12b",
            request_id="",
            attempts=1,
            elapsed_ms=5,
            response_sha256=sha256(self.content.encode()).hexdigest(),
            metrics=GenerationMetrics(),
        )


class FakeAuditor:
    def __init__(self):
        self.calls = 0

    async def persist_query_audit_async(self, audit, **kwargs):
        self.calls += 1
        return AuditWriteResult(
            request_id=str(audit.request_id),
            outcomes=(AuditSinkOutcome(
                sink=AuditSink.QUERY_JSONL,
                attempted=False,
                success=True,
                skipped=True,
            ),),
        )


class FakeEvaluator:
    async def evaluate_async(self, **kwargs):
        return RagEvalResult.disabled()


def _service(*, retriever, generator, auditor=None):
    return RagService(
        config=settings.model_copy(update={"audit_enabled": False, "evaluation_enabled": False}),
        resource_manager=FakeResources(),
        retriever=retriever,
        llm_generator=generator,
        evaluator=FakeEvaluator(),
        auditor=auditor or FakeAuditor(),
    )


@pytest.mark.asyncio
async def test_math_direct_bypasses_retrieval_and_generation(tenant_context):
    retriever = FakeRetriever()
    generator = FakeGenerator()
    service = _service(retriever=retriever, generator=generator)

    result = await service.query(
        RagQueryCommand(
            query="Calcola la copertura di una checklist di 100 controlli: 70 implementati e 20 parziali che valgono al 50%.",
        ),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.MATH_DIRECT
    assert result.deterministic is True
    assert retriever.calls == 0
    assert generator.calls == 0
    assert "80.00%" in result.answer


@pytest.mark.asyncio
async def test_documental_query_runs_retrieval_generation_validation_and_audit(
    tenant_context, candidate_factory
):
    candidate = candidate_factory(
        "c1",
        filename="policy.pdf",
        tier="B",
        score_vec=0.8,
        content="La policy definisce una procedura formalizzata.",
    )
    debug = RetrievalDebug(qdrant_hits=1, kept_after_quality_filters=1)
    retriever = FakeRetriever(candidates=(candidate,), debug=debug)
    generator = FakeGenerator()
    auditor = FakeAuditor()
    service = _service(retriever=retriever, generator=generator, auditor=auditor)

    result = await service.query(
        RagQueryCommand(query="Esiste una procedura formalizzata?", max_sources=4),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.RAG_GENERATION
    assert result.sources[0].filename == "policy.pdf"
    assert "policy.pdf" in result.answer
    assert retriever.calls == 1
    assert generator.calls == 1
    assert auditor.calls == 1


@pytest.mark.asyncio
async def test_no_sources_returns_safe_fallback_without_llm(tenant_context):
    retriever = FakeRetriever(candidates=(), debug=RetrievalDebug())
    generator = FakeGenerator()
    service = _service(retriever=retriever, generator=generator)

    result = await service.query(
        RagQueryCommand(query="Descrivi il controllo documentale"),
        tenant_context=tenant_context,
    )

    assert generator.calls == 0
    assert result.sources == ()
    assert "Non ho trovato evidenze sufficienti" in result.answer


@pytest.mark.asyncio
async def test_generation_error_is_wrapped_and_failed_request_is_audited(
    tenant_context, candidate_factory
):
    from core.generation import GenerationTransportError

    candidate = candidate_factory("c1", tier="B", score_vec=0.7)
    retriever = FakeRetriever(candidates=(candidate,), debug=RetrievalDebug(qdrant_hits=1))
    auditor = FakeAuditor()
    generator = FakeGenerator(error=GenerationTransportError("timeout", retryable=True))
    service = _service(retriever=retriever, generator=generator, auditor=auditor)

    with pytest.raises(RagServiceGenerationError):
        await service.query(
            RagQueryCommand(query="Valuta la policy"),
            tenant_context=tenant_context,
        )

    assert auditor.calls == 1


@pytest.mark.asyncio
async def test_foreign_candidate_is_rejected_by_service_guard(
    tenant_context, candidate_factory
):
    foreign = candidate_factory("foreign", tier="C", organization_id=9999)
    retriever = FakeRetriever(candidates=(foreign,), debug=RetrievalDebug(qdrant_hits=1))
    service = _service(retriever=retriever, generator=FakeGenerator())

    with pytest.raises(Exception) as exc_info:
        await service.query(
            RagQueryCommand(query="Valuta evidenza"),
            tenant_context=tenant_context,
        )

    assert "fuori dal perimetro tenant" in str(exc_info.value)
