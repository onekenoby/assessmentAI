from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from core.config import RagSettings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ollama_session() -> Iterator[Any]:
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "rag-api-real-integration-test/1.0",
        }
    )
    try:
        yield session
    finally:
        session.close()


def _ollama_base_url(config: RagSettings) -> str:
    suffix = "/api/chat"
    value = config.ollama_native_chat_url.rstrip("/")
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value.rsplit("/api/", 1)[0]


def test_ollama_tags_contains_configured_model(
    ollama_session: Any,
    integration_settings: RagSettings,
    service_timeout_seconds: int,
) -> None:
    url = _ollama_base_url(integration_settings) + "/api/tags"
    response = ollama_session.get(url, timeout=(10, service_timeout_seconds))
    response.raise_for_status()
    payload = response.json()

    models = payload.get("models") if isinstance(payload, Mapping) else None
    assert isinstance(models, list)
    names = {
        str(item.get("name") or item.get("model") or "")
        for item in models
        if isinstance(item, Mapping)
    }
    assert integration_settings.llm_model_name in names, (
        f"Modello {integration_settings.llm_model_name!r} non presente: {sorted(names)}"
    )


def test_ollama_native_chat_returns_non_empty_content(
    ollama_session: Any,
    integration_settings: RagSettings,
    service_timeout_seconds: int,
) -> None:
    expected_token = "OK_OLLAMA_REAL_TEST"
    payload = {
        "model": integration_settings.llm_model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sei un endpoint di test. Rispondi soltanto con il token "
                    f"{expected_token}, senza altro testo."
                ),
            },
            {
                "role": "user",
                "content": f"Restituisci esattamente: {expected_token}",
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": min(integration_settings.llm_num_ctx, 4096),
            "num_predict": 32,
            "repeat_penalty": integration_settings.llm_repeat_penalty,
        },
    }

    response = ollama_session.post(
        integration_settings.ollama_native_chat_url,
        json=payload,
        timeout=(10, service_timeout_seconds),
    )
    response.raise_for_status()
    body = response.json()

    assert isinstance(body, Mapping)
    message = body.get("message")
    assert isinstance(message, Mapping)
    content = str(message.get("content") or "").strip()

    assert content, "Ollama ha restituito message.content vuoto"
    assert expected_token in content, f"Token atteso non trovato nella risposta: {content!r}"
    assert bool(body.get("done", True)) is True
