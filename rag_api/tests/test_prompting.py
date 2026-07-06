from __future__ import annotations

from core.models import RagAnswerMode, RagIntent
from core.prompting import (
    PromptBuildOptions,
    PromptBuilder,
    build_alternating_history,
    build_tier_context_blocks,
)


def test_history_ignores_system_collapses_roles_and_removes_current_query() -> None:
    history = [
        {"role": "system", "content": "Non deve entrare"},
        {"role": "user", "content": "Prima domanda"},
        {"role": "user", "content": "Domanda aggiornata"},
        {"role": "assistant", "content": "Risposta"},
        {"role": "user", "content": "Domanda corrente"},
    ]

    normalized = build_alternating_history(
        history,
        max_turns=3,
        current_query="Domanda corrente",
    )

    assert [(message.role, message.content) for message in normalized] == [
        ("user", "Domanda aggiornata"),
        ("assistant", "Risposta"),
    ]


def test_tier_context_filters_foreign_sources(
    tenant_context, source_a, source_b, source_c, foreign_source
) -> None:
    blocks = build_tier_context_blocks(
        [source_a, source_b, source_c, foreign_source],
        max_chars=10_000,
        context=tenant_context,
    )

    assert blocks.source_count == 3
    assert blocks.dropped_sources == 1
    assert blocks.tier_counts == {"A": 1, "B": 1, "C": 1}
    assert "normativa.pdf" in blocks.tier_a
    assert "policy.pdf" in blocks.tier_b
    assert "evidenza_test.pdf" in blocks.tier_c
    assert "foreign.pdf" not in blocks.combined()


def test_requested_document_limits_prompt_context(
    tenant_context, source_b, source_c
) -> None:
    blocks = build_tier_context_blocks(
        [source_b, source_c],
        max_chars=10_000,
        context=tenant_context,
        requested_document="evidenza_test.pdf",
    )

    assert blocks.source_count == 1
    assert blocks.included_source_ids == ("source-c",)
    assert "policy.pdf" not in blocks.combined()


def test_prompt_builder_builds_tenant_safe_evidence_relevance_prompt(
    tenant_context, source_c, foreign_source
) -> None:
    bundle = PromptBuilder().build(
        query="Valuta se il documento è attinente alla domanda di assessment",
        sources=[source_c, foreign_source],
        history=[{"role": "user", "content": "Contesto precedente"}],
        options=PromptBuildOptions(
            intent=RagIntent.AUDIT,
            answer_mode=RagAnswerMode.EVIDENCE_RELEVANCE,
            requested_document="evidenza_test.pdf",
            strict_checklist_mode=True,
            wants_evidence=True,
        ),
        tenant_context=tenant_context,
    )

    payload = bundle.to_ollama_messages()
    combined = "\n".join(message["content"] for message in payload)

    assert payload[0]["role"] == "system"
    assert payload[-1]["role"] == "user"
    assert "| Livello di attinenza | Percentuale stimata | Esito sintetico |" in combined
    assert "EVIDENCE RELEVANCE MODE — HIGHEST PRIORITY" in combined
    assert "evidenza_test.pdf" in combined
    assert "foreign.pdf" not in combined
    assert bundle.context.source_count == 1
    assert len(bundle.prompt_sha256) == 64


def test_prompt_rejects_document_path(tenant_context, source_c) -> None:
    builder = PromptBuilder()
    try:
        builder.build(
            query="test",
            sources=[source_c],
            options=PromptBuildOptions(requested_document="../secret.pdf"),
            tenant_context=tenant_context,
        )
    except ValueError as exc:
        assert "nome file" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Un percorso documentale deve essere rifiutato")
