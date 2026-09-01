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

def test_prompt_builder_uses_audit_specific_mode_instructions(
    tenant_context,
    source_a,
    source_b,
    source_c,
) -> None:
    bundle = PromptBuilder().build(
        query=(
            "Esegui un audit di conformità e identifica "
            "eventuali gap di implementazione"
        ),
        sources=[
            source_a,
            source_b,
            source_c,
        ],
        options=PromptBuildOptions(
            intent=RagIntent.AUDIT,
            answer_mode=RagAnswerMode.AUDIT,
            wants_evidence=True,
        ),
        tenant_context=tenant_context,
    )

    user_content = bundle.user_content

    assert "### ANSWER MODE ###\naudit" in user_content
    assert "AUDIT MODE:" in user_content

    assert (
        "The absence of a Tier B or Tier C source is not, by itself, proof of non-compliance."
        in user_content
    )

    assert (
        "Never present a Tier B policy or planned procedure as proof that a Tier C control has actually been implemented."
        in user_content
    )

    assert "KNOWLEDGE MODE:" not in user_content
    assert "When answer mode is knowledge" not in user_content


def test_prompt_builder_keeps_knowledge_mode_non_audit_guardrail(
    tenant_context,
    source_a,
) -> None:
    bundle = PromptBuilder().build(
        query="Spiega il significato del requisito recuperato",
        sources=[source_a],
        options=PromptBuildOptions(
            intent=RagIntent.TEXT,
            answer_mode=RagAnswerMode.KNOWLEDGE,
        ),
        tenant_context=tenant_context,
    )

    user_content = bundle.user_content

    assert "### ANSWER MODE ###\nknowledge" in user_content
    assert "KNOWLEDGE MODE:" in user_content
    assert "AUDIT MODE:" not in user_content
    assert "EVIDENCE RELEVANCE MODE" not in user_content
    

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

def test_tier_context_preserves_substantial_source_body(
    tenant_context,
    source_a,
) -> None:
    long_content = "A" * 500

    sources = []

    for index in range(20):
        source_id = f"body-source-{index}"

        sources.append(
            source_a.model_copy(
                update={
                    "id": source_id,
                    "doc_id": f"body-doc-{index}",
                    "filename": f"body-document-{index}.pdf",
                    "page": index + 1,
                    "page_chunk_index": 1,
                    "content": (
                        long_content
                        if index == 0
                        else f"Contenuto breve della fonte {index}."
                    ),
                    "dedupe_key": (
                        f"body-doc-{index}::{index + 1}::1::{source_id}"
                    ),
                }
            )
        )

    blocks = build_tier_context_blocks(
        sources,
        max_chars=10_000,
        context=tenant_context,
    )

    combined = blocks.combined()

    # Con il precedente minimo di 200 caratteri questa sequenza
    # veniva troncata.
    assert long_content in combined
    assert "A" * 200 + "..." not in combined
    
    
def test_tier_context_skips_oversized_source_and_keeps_later_source(
    tenant_context,
    source_a,
    source_b,
) -> None:
    oversized = source_a.model_copy(
        update={
            "id": "oversized-source",
            "doc_id": "oversized-doc",
            "filename": "oversized.pdf",
            "section_hint": "S" * 1_200,
            "content": "Fonte con intestazione eccessivamente grande.",
            "dedupe_key": (
                "oversized-doc::1::1::oversized-source"
            ),
        }
    )

    compact = source_b.model_copy(
        update={
            "id": "compact-source",
            "doc_id": "compact-doc",
            "filename": "compact.pdf",
            "page": 2,
            "content": (
                "Questa fonte compatta deve essere inclusa "
                "anche se la precedente non entra nel budget."
            ),
            "dedupe_key": (
                "compact-doc::2::1::compact-source"
            ),
        }
    )

    blocks = build_tier_context_blocks(
        [oversized, compact],
        max_chars=900,
        context=tenant_context,
    )

    assert "oversized-source" not in blocks.included_source_ids
    assert "compact-source" in blocks.included_source_ids
    assert "compact.pdf" in blocks.combined()
    assert blocks.truncated is True
    
    