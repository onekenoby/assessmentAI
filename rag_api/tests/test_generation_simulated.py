from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import requests

from core.config import settings
from core.generation import (
    EmptyGenerationError,
    GenerationHttpError,
    GenerationOptions,
    GenerationProtocolError,
    OllamaNativeGenerator,
)
from core.prompting import PromptMessage
from core.tenant import bind_tenant_context


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, *, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = "fake-response"

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@dataclass
class FakeResources:
    session: FakeSession

    def get_ollama_session(self):
        return self.session


def _payload(content: str, *, thinking: str = "", model: str = "gemma4:12b") -> dict[str, Any]:
    return {
        "model": model,
        "created_at": "2026-07-02T10:00:00Z",
        "message": {"role": "assistant", "content": content, "thinking": thinking},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 500_000_000,
        "eval_duration": 1_000_000_000,
        "prompt_eval_count": 20,
        "eval_count": 10,
    }


def test_generation_success_builds_native_chat_payload(tenant_context):
    session = FakeSession([FakeResponse(200, _payload("Risposta finale"))])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        result = generator.generate([
            PromptMessage(role="system", content="Sistema"),
            PromptMessage(role="user", content="Domanda"),
        ])

    assert result.content == "Risposta finale"
    assert result.request_id == tenant_context.request_id
    assert result.attempts == 1
    assert result.metrics.eval_count == 10
    assert result.metrics.tokens_per_second == pytest.approx(10.0)

    sent = session.calls[0]["json"]
    assert sent["stream"] is False
    assert sent["think"] is False
    assert sent["model"] == settings.llm_model_name
    assert sent["options"]["num_ctx"] == settings.llm_num_ctx
    assert sent["messages"][0]["role"] == "system"


def test_empty_content_retries_and_never_uses_thinking(tenant_context):
    session = FakeSession([
        FakeResponse(200, _payload("", thinking="ragionamento privato")),
        FakeResponse(200, _payload("Risposta dopo retry")),
    ])
    sleeps: list[float] = []
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=sleeps.append,
    )

    with bind_tenant_context(tenant_context):
        result = generator.generate(
            [{"role": "user", "content": "Domanda"}],
            options=GenerationOptions(max_attempts=2, retry_backoff_seconds=0),
        )

    assert result.content == "Risposta dopo retry"
    assert result.attempts == 2
    assert "ragionamento privato" not in result.content
    assert len(session.calls) == 2
    retry_messages = session.calls[1]["json"]["messages"]
    assert "Return the final answer now" in retry_messages[-1]["content"]
    assert any("message.content vuoto" in warning for warning in result.warnings)


def test_empty_content_without_retry_raises(tenant_context):
    session = FakeSession([FakeResponse(200, _payload("", thinking="solo thinking"))])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        with pytest.raises(EmptyGenerationError) as exc_info:
            generator.generate(
                [{"role": "user", "content": "Domanda"}],
                options=GenerationOptions(max_attempts=1),
            )

    assert exc_info.value.thinking_chars == len("solo thinking")


def test_transient_http_error_retries(tenant_context):
    session = FakeSession([
        FakeResponse(503, {"error": "temporaneo"}, headers={"Retry-After": "0"}),
        FakeResponse(200, _payload("OK")),
    ])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        result = generator.generate(
            [{"role": "user", "content": "Domanda"}],
            options=GenerationOptions(max_attempts=2, retry_backoff_seconds=0),
        )

    assert result.content == "OK"
    assert result.attempts == 2


def test_non_retryable_http_error_does_not_retry(tenant_context):
    session = FakeSession([FakeResponse(400, {"error": "bad request"})])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        with pytest.raises(GenerationHttpError) as exc_info:
            generator.generate(
                [{"role": "user", "content": "Domanda"}],
                options=GenerationOptions(max_attempts=2, retry_backoff_seconds=0),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False
    assert len(session.calls) == 1


def test_invalid_json_is_protocol_error(tenant_context):
    session = FakeSession([FakeResponse(200, ValueError("invalid json"))])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        with pytest.raises(GenerationProtocolError):
            generator.generate(
                [{"role": "user", "content": "Domanda"}],
                options=GenerationOptions(max_attempts=1),
            )


def test_transport_timeout_is_retryable_and_eventually_raises(tenant_context):
    session = FakeSession([
        requests.Timeout("timeout-1"),
        requests.Timeout("timeout-2"),
    ])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        with pytest.raises(Exception) as exc_info:
            generator.generate(
                [{"role": "user", "content": "Domanda"}],
                options=GenerationOptions(max_attempts=2, retry_backoff_seconds=0),
            )

    assert getattr(exc_info.value, "retryable", False) is True
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_generate_async_uses_same_contract(tenant_context):
    session = FakeSession([FakeResponse(200, _payload("Async OK"))])
    generator = OllamaNativeGenerator(
        resource_manager=FakeResources(session),
        config=settings,
        sleep_fn=lambda _: None,
    )

    with bind_tenant_context(tenant_context):
        result = await generator.generate_async([
            {"role": "user", "content": "Domanda asincrona"}
        ])

    assert result.content == "Async OK"
    assert result.request_id == tenant_context.request_id
