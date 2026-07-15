from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS, settings
from core.models import RagExecutionMode
from core.rag_service import RagQueryCommand, RagQueryRouter, RagService
from core.solvers import SolverName, solve_deterministic_query


BATCH13_CONTROL_COVERAGE_QUERY = (
    "Una checklist di 20 controlli contiene 12 controlli implementati e "
    "4 controlli parziali che valgono il 50%. Calcola la copertura complessiva."
)


class _NeverRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve_candidates(self, **kwargs):
        self.calls += 1
        raise AssertionError("Il ramo math_direct non deve eseguire retrieval")

    async def lookup_glossary(self, **kwargs):
        raise AssertionError("Il ramo math_direct non deve eseguire glossary lookup")


class _NeverGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_async(self, prompt):
        self.calls += 1
        raise AssertionError("Il ramo math_direct non deve invocare Ollama")


class _ResourcesWithoutReranker:
    @staticmethod
    def get_reranker():
        return None


class _NeverEvaluator:
    async def evaluate_async(self, **kwargs):
        raise AssertionError("Il judge non deve essere invocato per math_direct")


class _NoopAuditor:
    async def persist_query_audit_async(self, audit, **kwargs):
        return SimpleNamespace(success=True, skipped=True)


def test_has_access_to_is_canonical_but_connects_to_remains_disallowed() -> None:
    assert "HAS_ACCESS_TO" in DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS
    assert "CONNECTS_TO" not in DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS


def test_batch13_control_coverage_phrase_is_solved_deterministically() -> None:
    result = solve_deterministic_query(BATCH13_CONTROL_COVERAGE_QUERY)

    assert result is not None
    assert result.solver == SolverName.CONTROL_COVERAGE
    assert "70.00%" in result.answer
    assert "Controlli implementati = `12`" in result.answer
    assert "Controlli parziali = `4`" in result.answer


def test_batch13_control_coverage_phrase_routes_math_direct() -> None:
    decision = RagQueryRouter().route(
        RagQueryCommand(query=BATCH13_CONTROL_COVERAGE_QUERY)
    )

    assert decision.execution_mode == RagExecutionMode.MATH_DIRECT
    assert decision.solver_result is not None
    assert decision.formula_strict_mode is False
    assert decision.math_needs_document_context is False


@pytest.mark.asyncio
async def test_batch13_math_query_bypasses_retrieval_generation_and_judge(
    tenant_context,
) -> None:
    retriever = _NeverRetriever()
    generator = _NeverGenerator()
    service = RagService(
        config=settings.model_copy(
            update={
                "audit_enabled": False,
                "evaluation_enabled": False,
            }
        ),
        resource_manager=_ResourcesWithoutReranker(),
        retriever=retriever,
        llm_generator=generator,
        evaluator=_NeverEvaluator(),
        auditor=_NoopAuditor(),
    )

    result = await service.query(
        RagQueryCommand(
            query=BATCH13_CONTROL_COVERAGE_QUERY,
            include_evaluation=False,
        ),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.MATH_DIRECT
    assert result.deterministic is True
    assert result.model == "not-used"
    assert "70.00%" in result.answer
    assert retriever.calls == 0
    assert generator.calls == 0


def test_control_coverage_still_rejects_incomplete_narrative() -> None:
    assert solve_deterministic_query(
        "La checklist contiene 12 controlli implementati e alcuni controlli parziali."
    ) is None
