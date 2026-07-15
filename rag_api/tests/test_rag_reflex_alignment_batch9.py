from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import pytest

from core.audit import AuditSink, AuditSinkOutcome, AuditWriteResult
from core.config import settings
from core.generation import GenerationMetrics, GenerationResult
from core.models import (
    EvaluationVerdict,
    RagEvalResult,
    RagExecutionMode,
    RetrievalDebug,
)
from core.rag_service import RagQueryCommand, RagService


class FakeRetriever:
    def __init__(self, candidates=()):
        self.candidates = tuple(candidates)
        self.calls = 0

    async def retrieve_candidates(self, **kwargs):
        self.calls += 1
        return self.candidates, RetrievalDebug(
            qdrant_hits=len(self.candidates),
            kept_after_quality_filters=len(self.candidates),
        )

    async def lookup_glossary(self, **kwargs):
        return None


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    async def generate_async(self, prompt):
        self.calls += 1
        content = (
            "**A) Risposta**\n\nLa procedura è formalizzata.\n\n"
            "**B) Evidenze**\n\n- Evidenza presente nella fonte recuperata.\n\n"
            "**C) Limiti / Conflitti**\n\n- Verifica limitata al contesto.\n\n"
            "**D) Fonti**\n\n- placeholder"
        )
        return GenerationResult(
            content=content,
            model="gemma4:12b",
            request_id="",
            attempts=1,
            elapsed_ms=5,
            response_sha256=sha256(content.encode()).hexdigest(),
            metrics=GenerationMetrics(),
        )


class CountingEvaluator:
    def __init__(self, result: RagEvalResult | None = None):
        self.calls = 0
        self.result = result or RagEvalResult(
            faithfulness=0.95,
            answer_relevance=0.95,
            context_support=0.95,
            hallucination_risk=0.05,
            verdict=EvaluationVerdict.PASS,
            reason="Judge executed.",
        )

    async def evaluate_async(self, **kwargs):
        self.calls += 1
        return self.result


class CapturingAuditor:
    def __init__(self):
        self.last_audit = None

    async def persist_query_audit_async(self, audit, **kwargs):
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


def _service(*, config, retriever, generator, evaluator, auditor):
    return RagService(
        config=config,
        resource_manager=SimpleNamespace(get_reranker=lambda: None),
        retriever=retriever,
        llm_generator=generator,
        evaluator=evaluator,
        auditor=auditor,
    )


def test_deterministic_eval_result_is_explicit_pass_not_disabled():
    result = RagEvalResult.deterministic_pass(
        execution_mode=RagExecutionMode.MATH_DIRECT,
        source_count=0,
    )

    assert result.verdict == EvaluationVerdict.PASS
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_support == 1.0
    assert result.hallucination_risk == 0.0
    assert result.source_scope_violation is False
    assert "input utente validato" in result.reason
    assert "LLM judge bypassed" in result.reason


@pytest.mark.asyncio
async def test_math_direct_bypasses_judge_and_cannot_be_strict_blocked(
    tenant_context,
    monkeypatch,
):
    config = settings.model_copy(
        update={
            "audit_enabled": False,
            "evaluation_enabled": True,
            "evaluation_strict_block": True,
        }
    )
    retriever = FakeRetriever()
    generator = FakeGenerator()
    evaluator = CountingEvaluator(
        RagEvalResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_support=0.0,
            hallucination_risk=1.0,
            verdict=EvaluationVerdict.FAIL,
            reason="This result must never be used.",
        )
    )
    auditor = CapturingAuditor()
    eval_audit_kwargs = {}

    async def fake_eval_audit(**kwargs):
        eval_audit_kwargs.update(kwargs)
        return AuditWriteResult(
            request_id=tenant_context.request_id,
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.EVALUATION_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )

    monkeypatch.setattr("core.rag_service.append_rag_eval_log_async", fake_eval_audit)

    service = _service(
        config=config,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        auditor=auditor,
    )

    result = await service.query(
        RagQueryCommand(
            query=(
                "Calcola la copertura di una checklist di 100 controlli: "
                "70 implementati e 20 parziali che valgono al 50%."
            ),
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert retriever.calls == 0
    assert generator.calls == 0
    assert evaluator.calls == 0
    assert result.execution_mode == RagExecutionMode.MATH_DIRECT
    assert result.deterministic is True
    assert result.evaluation is not None
    assert result.evaluation.verdict == EvaluationVerdict.PASS
    assert "80.00%" in result.answer
    assert "non ha superato" not in result.answer
    assert result.model == "not-used"
    assert auditor.last_audit is not None
    assert auditor.last_audit.llm_model == "not-used"
    assert "Modello: `not-used`" in result.audit_markdown
    assert eval_audit_kwargs["llm_model"] == "not-used"
    assert eval_audit_kwargs["evaluation_model"] == "not-used"


@pytest.mark.asyncio
async def test_no_sources_fallback_bypasses_judge_and_is_marked_deterministic(
    tenant_context,
    monkeypatch,
):
    config = settings.model_copy(
        update={
            "audit_enabled": False,
            "evaluation_enabled": True,
            "evaluation_strict_block": True,
        }
    )
    retriever = FakeRetriever()
    generator = FakeGenerator()
    evaluator = CountingEvaluator()
    auditor = CapturingAuditor()

    async def fake_eval_audit(**kwargs):
        return AuditWriteResult(
            request_id=tenant_context.request_id,
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.EVALUATION_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )

    monkeypatch.setattr("core.rag_service.append_rag_eval_log_async", fake_eval_audit)

    service = _service(
        config=config,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        auditor=auditor,
    )

    result = await service.query(
        RagQueryCommand(
            query="Descrivi la procedura formalizzata.",
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert retriever.calls == 1
    assert generator.calls == 0
    assert evaluator.calls == 0
    assert result.deterministic is True
    assert result.evaluation is not None
    assert result.evaluation.verdict == EvaluationVerdict.PASS
    assert "Non ho trovato evidenze sufficienti" in result.answer
    assert result.sources == ()
    assert auditor.last_audit.llm_model == "not-used"


@pytest.mark.asyncio
async def test_generated_rag_answer_still_uses_llm_judge(
    tenant_context,
    candidate_factory,
    monkeypatch,
):
    candidate = candidate_factory(
        "generated-source",
        filename="policy.pdf",
        tier="B",
        score_vec=0.9,
        content="La policy definisce una procedura formalizzata.",
    )
    config = settings.model_copy(
        update={
            "audit_enabled": False,
            "evaluation_enabled": True,
            "evaluation_strict_block": False,
        }
    )
    retriever = FakeRetriever((candidate,))
    generator = FakeGenerator()
    evaluator = CountingEvaluator()
    auditor = CapturingAuditor()
    eval_audit_kwargs = {}

    async def fake_eval_audit(**kwargs):
        eval_audit_kwargs.update(kwargs)
        return AuditWriteResult(
            request_id=tenant_context.request_id,
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.EVALUATION_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )

    monkeypatch.setattr("core.rag_service.append_rag_eval_log_async", fake_eval_audit)

    service = _service(
        config=config,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        auditor=auditor,
    )

    result = await service.query(
        RagQueryCommand(
            query="Descrivi la procedura formalizzata.",
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert retriever.calls == 1
    assert generator.calls == 1
    assert evaluator.calls == 1
    assert result.deterministic is False
    assert result.evaluation is not None
    assert result.evaluation.reason == "Judge executed."
    assert result.model == "gemma4:12b"
    assert auditor.last_audit.llm_model == "gemma4:12b"
    assert eval_audit_kwargs["llm_model"] == "gemma4:12b"
    assert eval_audit_kwargs["evaluation_model"] == config.evaluation_model_name


@pytest.mark.asyncio
async def test_disabled_judge_still_returns_local_pass_for_deterministic_branch(
    tenant_context,
    monkeypatch,
):
    config = settings.model_copy(
        update={
            "audit_enabled": False,
            "evaluation_enabled": False,
        }
    )
    evaluator = CountingEvaluator()

    async def fake_eval_audit(**kwargs):
        return AuditWriteResult(
            request_id=tenant_context.request_id,
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.EVALUATION_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )

    monkeypatch.setattr("core.rag_service.append_rag_eval_log_async", fake_eval_audit)

    service = _service(
        config=config,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        evaluator=evaluator,
        auditor=CapturingAuditor(),
    )

    result = await service.query(
        RagQueryCommand(
            query=(
                "Calcola la copertura di una checklist di 100 controlli: "
                "70 implementati e 20 parziali che valgono al 50%."
            ),
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert evaluator.calls == 0
    assert result.evaluation is not None
    assert result.evaluation.verdict == EvaluationVerdict.PASS
