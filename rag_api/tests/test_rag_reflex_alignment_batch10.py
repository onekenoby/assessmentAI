from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from core.audit import AuditSink, AuditSinkOutcome, AuditWriteResult
from core.config import settings
from core.generation import EmptyGenerationError, GenerationTransportError
from core.models import EvaluationVerdict, RagExecutionMode, RetrievalDebug
from core.rag_service import (
    RagQueryCommand,
    RagService,
    RagServiceGenerationError,
)


class FakeResources:
    def get_reranker(self):
        return None


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)

    async def retrieve_candidates(self, **kwargs):
        return self.candidates, RetrievalDebug(
            qdrant_hits=len(self.candidates),
            kept_after_quality_filters=len(self.candidates),
        )

    async def lookup_glossary(self, **kwargs):
        return None


class FailingGenerator:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def generate_async(self, prompt):
        self.calls += 1
        raise self.error


class CapturingAuditor:
    def __init__(self):
        self.calls = 0
        self.last_audit = None

    async def persist_query_audit_async(self, audit, **kwargs):
        self.calls += 1
        self.last_audit = audit
        return AuditWriteResult(
            request_id=str(audit.request_id),
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.QUERY_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )


class CountingEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate_async(self, **kwargs):
        self.calls += 1
        raise AssertionError("Il judge non deve essere chiamato sul fallback tecnico")


@dataclass
class _SkippedAuditResult:
    success: bool = True
    skipped: bool = True


def _service(*, candidate, error, auditor=None, evaluator=None, **config_updates):
    config = settings.model_copy(
        update={
            "audit_enabled": False,
            "evaluation_enabled": False,
            "generation_failure_fallback_enabled": True,
            **config_updates,
        }
    )
    return RagService(
        config=config,
        resource_manager=FakeResources(),
        retriever=FakeRetriever((candidate,)),
        llm_generator=FailingGenerator(error),
        auditor=auditor or CapturingAuditor(),
        evaluator=evaluator or CountingEvaluator(),
    )


def test_generation_failure_fallback_is_enabled_by_default() -> None:
    assert settings.generation_failure_fallback_enabled is True


@pytest.mark.asyncio
async def test_transport_failure_returns_structured_fallback_and_preserves_sources(
    tenant_context,
    candidate_factory,
):
    candidate = candidate_factory(
        "generation-fallback",
        filename="policy.pdf",
        tier="B",
        score_vec=0.9,
        content="La policy definisce il processo di approvazione.",
    )
    auditor = CapturingAuditor()
    generator_error = GenerationTransportError(
        "timeout verso http://ollama.internal:11434?token=secret",
        retryable=True,
    )
    service = _service(candidate=candidate, error=generator_error, auditor=auditor)

    result = await service.query(
        RagQueryCommand(query="Valuta la policy di approvazione"),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.RAG_GENERATION
    assert result.deterministic is True
    assert result.model == settings.llm_model_name
    assert len(result.sources) == 1
    assert result.sources[0].filename == "policy.pdf"
    assert "modello generativo non ha prodotto una risposta valida" in result.answer
    assert "policy.pdf" in result.answer
    assert "secret" not in result.answer
    assert "ollama.internal" not in result.answer
    assert any("GenerationTransportError" in warning for warning in result.warnings)

    assert auditor.calls == 1
    assert auditor.last_audit is not None
    assert auditor.last_audit.llm_model == settings.llm_model_name
    assert auditor.last_audit.filters["generation_attempted"] is True
    assert auditor.last_audit.filters["generation_failed"] is True
    assert auditor.last_audit.filters["generation_fallback_applied"] is True
    assert auditor.last_audit.filters["generation_error_type"] == "GenerationTransportError"


@pytest.mark.asyncio
async def test_empty_generation_fallback_does_not_expose_thinking_or_raw_error(
    tenant_context,
    candidate_factory,
):
    candidate = candidate_factory("empty-generation", filename="evidence.pdf", tier="C")
    service = _service(
        candidate=candidate,
        error=EmptyGenerationError(
            "message.content vuoto; thinking=ragionamento-segreto",
            thinking_chars=22,
        ),
    )

    result = await service.query(
        RagQueryCommand(query="Analizza l'evidenza tecnica"),
        tenant_context=tenant_context,
    )

    assert "EmptyGenerationError" in result.answer
    assert "ragionamento-segreto" not in result.answer
    assert "thinking=" not in result.answer
    assert result.sources[0].filename == "evidence.pdf"


@pytest.mark.asyncio
async def test_generation_fallback_uses_local_warn_evaluation_without_judge(
    tenant_context,
    candidate_factory,
    monkeypatch,
):
    async def _skip_eval_audit(**kwargs: Any):
        return _SkippedAuditResult()

    monkeypatch.setattr(
        "core.rag_service.append_rag_eval_log_async",
        _skip_eval_audit,
    )

    candidate = candidate_factory("eval-fallback", filename="audit.pdf", tier="B")
    evaluator = CountingEvaluator()
    service = _service(
        candidate=candidate,
        error=GenerationTransportError("timeout", retryable=True),
        evaluator=evaluator,
        evaluation_enabled=True,
    )

    result = await service.query(
        RagQueryCommand(
            query="Valuta il controllo documentale",
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert evaluator.calls == 0
    assert result.evaluation is not None
    assert str(result.evaluation.verdict) == EvaluationVerdict.WARN
    assert result.evaluation.faithfulness == 1.0
    assert result.evaluation.hallucination_risk == 0.0
    assert "GenerationTransportError" in result.evaluation.reason
    assert "Evaluation locale WARN" in "\n".join(result.warnings)


@pytest.mark.asyncio
async def test_generation_failure_can_still_be_configured_as_http_error(
    tenant_context,
    candidate_factory,
):
    candidate = candidate_factory("strict-generation", filename="strict.pdf", tier="B")
    auditor = CapturingAuditor()
    service = _service(
        candidate=candidate,
        error=GenerationTransportError("timeout", retryable=True),
        auditor=auditor,
        generation_failure_fallback_enabled=False,
    )

    with pytest.raises(RagServiceGenerationError):
        await service.query(
            RagQueryCommand(query="Valuta la policy"),
            tenant_context=tenant_context,
        )

    assert auditor.calls == 1
    assert auditor.last_audit.filters["generation_failed"] is True
