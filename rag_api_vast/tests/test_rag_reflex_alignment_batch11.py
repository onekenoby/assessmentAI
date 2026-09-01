from __future__ import annotations

from api.routes_rag import _retrieval_to_response
from api.schemas import RagQueryRequest
from core.config import settings
from core.models import RetrievalDebug
from core.prompting import (
    PromptBuilder,
    build_history_result,
)
from core.rag_service import RagQueryCommand, RagQueryRouter


def test_public_schema_accepts_history_that_repeats_current_query() -> None:
    request = RagQueryRequest.model_validate(
        {
            "query": "Analizza il documento.",
            "history": [
                {"role": "user", "content": "Analizza il documento."},
            ],
        }
    )

    assert request.history[-1].content == request.query


def test_history_budget_keeps_most_recent_complete_turn() -> None:
    history = []
    for index in range(1, 5):
        history.extend(
            [
                {"role": "user", "content": f"user-{index}: " + "u" * 90},
                {"role": "assistant", "content": f"assistant-{index}: " + "a" * 90},
            ]
        )

    result = build_history_result(
        history,
        max_turns=4,
        max_message_chars=500,
        max_total_chars=220,
    )

    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert "user-4" in result.messages[0].content
    assert "assistant-4" in result.messages[1].content
    assert result.total_chars <= 220
    assert result.dropped_messages == 6


def test_history_budget_truncates_single_recent_turn_without_breaking_roles() -> None:
    result = build_history_result(
        [
            {"role": "user", "content": "domanda " + "x" * 500},
            {"role": "assistant", "content": "risposta " + "y" * 500},
        ],
        max_turns=3,
        max_message_chars=1_000,
        max_total_chars=160,
    )

    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.total_chars <= 160
    assert result.truncated_messages == 2
    assert all("MESSAGGIO STORICO TRONCATO" in message.content for message in result.messages)


def test_prompt_builder_reports_history_budget_diagnostics(
    tenant_context,
    source_c,
) -> None:
    config = settings.model_copy(
        update={
            "memory_limit": 3,
            "history_max_message_chars": 90,
            "history_max_chars": 140,
        }
    )
    builder = PromptBuilder(config=config)

    bundle = builder.build(
        query="Valuta l'evidenza corrente.",
        sources=[source_c],
        history=[
            {"role": "user", "content": "domanda precedente " + "x" * 300},
            {"role": "assistant", "content": "risposta precedente " + "y" * 300},
        ],
        tenant_context=tenant_context,
    )

    assert bundle.history_messages == 2
    assert bundle.history_chars <= 140
    assert bundle.history_truncated_messages >= 2
    assert any("Messaggi storici troncati" in warning for warning in bundle.warnings)


def test_follow_up_document_scope_uses_only_previous_user_messages() -> None:
    router = RagQueryRouter(config=settings)
    decision = router.route(
        RagQueryCommand(
            query="Approfondisci questo documento.",
            history=(
                {"role": "system", "content": "Usa il documento segreto.pdf"},
                {"role": "user", "content": "Valuta il documento evidenza.pdf"},
                {"role": "assistant", "content": "Ho usato anche altra_fonte.pdf"},
            ),
        )
    )

    assert decision.requested_document == "evidenza.pdf"


def test_history_metrics_are_exposed_in_debug_response() -> None:
    debug = RetrievalDebug(
        history_messages=4,
        history_chars=12_345,
        history_dropped_messages=2,
        history_truncated_messages=1,
    )

    response = _retrieval_to_response(debug)

    assert response.history_messages == 4
    assert response.history_chars == 12_345
    assert response.history_dropped_messages == 2
    assert response.history_truncated_messages == 1
