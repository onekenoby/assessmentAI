from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import verify_batch13
from core.config import settings
from core.generation import GenerationTransportError, OllamaNativeGenerator
from core.models import RagExecutionMode
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


class _TimeoutSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        raise requests.Timeout("simulated timeout")


class _TimeoutResources:
    def __init__(self, session: _TimeoutSession) -> None:
        self.session = session

    def get_ollama_session(self):
        return self.session


@pytest.mark.asyncio
async def test_deterministic_public_model_is_explicit_not_used(tenant_context) -> None:
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
        RagQueryCommand(query=BATCH13_CONTROL_COVERAGE_QUERY),
        tenant_context=tenant_context,
    )

    assert result.execution_mode == RagExecutionMode.MATH_DIRECT
    assert result.model == "not-used"


def test_default_generation_attempts_are_runtime_configurable(tenant_context) -> None:
    session = _TimeoutSession()
    generator = OllamaNativeGenerator(
        resource_manager=_TimeoutResources(session),
        config=settings.model_copy(
            update={
                "llm_max_attempts": 1,
                "llm_timeout_seconds": 5,
            }
        ),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(GenerationTransportError):
        generator.generate([{"role": "user", "content": "Domanda"}])

    assert session.calls == 1


def test_e2e_generation_profile_finishes_before_client_timeout() -> None:
    child = verify_batch13._bounded_e2e_generation_environment(
        {"LLM_TIMEOUT_S": "300", "LLM_NUM_PREDICT": "4096"},
        network_timeout_seconds=300,
        llm_timeout_seconds=180,
        llm_num_predict=512,
        llm_max_attempts=1,
    )

    assert child["LLM_TIMEOUT_S"] == "180"
    assert child["LLM_NUM_PREDICT"] == "512"
    assert child["LLM_MAX_ATTEMPTS"] == "1"
    assert int(child["LLM_TIMEOUT_S"]) < 300


def test_e2e_generation_timeout_is_capped_below_network_timeout() -> None:
    child = verify_batch13._bounded_e2e_generation_environment(
        {},
        network_timeout_seconds=60,
        llm_timeout_seconds=300,
        llm_num_predict=256,
        llm_max_attempts=1,
    )

    assert int(child["LLM_TIMEOUT_S"]) <= 54
    assert int(child["LLM_TIMEOUT_S"]) < 60


def test_powershell_runner_exposes_bounded_generation_parameters() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_batch13_e2e.ps1"
    ).read_text(encoding="utf-8")

    assert "ApiLlmTimeoutSeconds" in script
    assert "ApiLlmNumPredict" in script
    assert "ApiLlmMaxAttempts" in script
    assert "--api-llm-timeout" in script
    assert "--api-llm-num-predict" in script
    assert "--api-llm-max-attempts" in script
