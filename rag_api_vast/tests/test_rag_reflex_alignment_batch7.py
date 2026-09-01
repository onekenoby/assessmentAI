from __future__ import annotations

from core.models import RagAnswerMode, RagIntent
from core.prompting import PromptBuildOptions, PromptBuilder, build_tier_context_blocks


TRUNCATION_MARKER = "[... CONTENUTO TRONCATO DAL BACKEND ...]"


def test_formula_intent_prioritizes_formula_source_before_ranked_text(
    tenant_context,
    source_c,
) -> None:
    ranked_text = source_c.model_copy(
        update={
            "id": "ranked-text",
            "doc_id": "ranked-text-doc",
            "filename": "ranked-text.pdf",
            "content": "Testo documentale ripetuto. " * 100,
            "type": "text",
            "page": 1,
        }
    )
    formula = source_c.model_copy(
        update={
            "id": "formula-source",
            "doc_id": "formula-doc",
            "filename": "formula.pdf",
            "content": "LaTeX: R = P \\times I\nPlain: R = P x I",
            "type": "formula",
            "page": 2,
        }
    )

    blocks = build_tier_context_blocks(
        [ranked_text, formula],
        max_chars=650,
        context=tenant_context,
        intent=RagIntent.FORMULA,
    )

    assert blocks.included_source_ids[0] == "formula-source"
    assert "LaTeX: R = P \\times I" in blocks.combined()
    assert blocks.included_source_ids.index("formula-source") < blocks.included_source_ids.index("ranked-text")


def test_prompt_builder_propagates_intent_priority_to_context_selection(
    tenant_context,
    source_c,
) -> None:
    long_text = source_c.model_copy(
        update={
            "id": "long-text",
            "doc_id": "long-text-doc",
            "filename": "long-text.pdf",
            "content": "Contenuto testuale. " * 120,
            "type": "text",
        }
    )
    formula = source_c.model_copy(
        update={
            "id": "builder-formula",
            "doc_id": "builder-formula-doc",
            "filename": "builder-formula.pdf",
            "content": "LaTeX: A = B + C\nPlain: A = B + C",
            "type": "formula",
        }
    )

    bundle = PromptBuilder().build(
        query="Mostra la formula recuperata.",
        sources=[long_text, formula],
        options=PromptBuildOptions(
            intent=RagIntent.FORMULA,
            answer_mode=RagAnswerMode.KNOWLEDGE,
            max_context_chars=650,
        ),
        tenant_context=tenant_context,
    )

    assert bundle.context.included_source_ids[0] == "builder-formula"
    assert "builder-formula.pdf" in bundle.user_content
    assert bundle.context.included_source_ids.index("builder-formula") < bundle.context.included_source_ids.index("long-text")
    assert bundle.user_content.index("builder-formula.pdf") < bundle.user_content.index("long-text.pdf")


def test_audit_context_prioritizes_tier_c_evidence_under_tight_budget(
    tenant_context,
    source_a,
    source_c,
) -> None:
    normative = source_a.model_copy(
        update={
            "id": "long-normative",
            "doc_id": "long-normative-doc",
            "filename": "long-normative.pdf",
            "content": "Requisito normativo generale. " * 100,
        }
    )
    evidence = source_c.model_copy(
        update={
            "id": "technical-evidence",
            "doc_id": "technical-evidence-doc",
            "filename": "technical-evidence.log",
            "content": "Log tecnico: controllo eseguito con esito positivo.",
        }
    )

    blocks = build_tier_context_blocks(
        [normative, evidence],
        max_chars=650,
        context=tenant_context,
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        wants_evidence=True,
    )

    assert blocks.included_source_ids[0] == "technical-evidence"
    assert "technical-evidence.log" in blocks.tier_c
    assert blocks.included_source_ids.index("technical-evidence") < blocks.included_source_ids.index("long-normative")


def test_merged_formula_tail_is_preserved_without_cutting_formula_lines(
    tenant_context,
    source_c,
) -> None:
    formula_tail = (
        "--- Formule collegate dal Knowledge Graph ---\n"
        "LaTeX: R = P \\times I\n"
        "Plain: R = P x I\n"
        "Meaning: rischio uguale probabilità per impatto"
    )
    mixed = source_c.model_copy(
        update={
            "id": "mixed-formula",
            "doc_id": "mixed-formula-doc",
            "filename": "mixed-formula.pdf",
            "content": ("Prosa documentale molto estesa. " * 100) + "\n\n" + formula_tail,
            "type": "text",
        }
    )

    blocks = build_tier_context_blocks(
        [mixed],
        max_chars=760,
        context=tenant_context,
        intent=RagIntent.FORMULA,
    )
    rendered = blocks.combined()

    assert TRUNCATION_MARKER in rendered
    assert "LaTeX: R = P \\times I" in rendered
    assert "Plain: R = P x I" in rendered
    assert "Meaning: rischio uguale probabilità per impatto" in rendered
    assert "LaTeX: R = P \\times" in rendered


def test_markdown_table_truncation_keeps_header_separator_and_complete_rows(
    tenant_context,
    source_c,
) -> None:
    rows = [
        f"| Controllo {index:02d} | Evidenza completa numero {index:02d} |"
        for index in range(1, 30)
    ]
    table_content = "\n".join(
        [
            "Premessa descrittiva. " * 20,
            "| Controllo | Evidenza |",
            "|---|---|",
            *rows,
        ]
    )
    table_source = source_c.model_copy(
        update={
            "id": "table-source",
            "doc_id": "table-doc",
            "filename": "controls-table.md",
            "content": table_content,
            "type": "table",
        }
    )

    blocks = build_tier_context_blocks(
        [table_source],
        max_chars=850,
        context=tenant_context,
        intent=RagIntent.TABLE,
    )
    rendered = blocks.combined()

    assert "| Controllo | Evidenza |" in rendered
    assert "|---|---|" in rendered
    assert TRUNCATION_MARKER in rendered

    table_lines = [line for line in rendered.splitlines() if line.startswith("|")]
    assert len(table_lines) >= 3
    assert all(line.endswith("|") for line in table_lines)
    assert all(
        line in {"| Controllo | Evidenza |", "|---|---|"}
        or line.startswith("| Controllo ") and " | Evidenza completa numero " in line
        for line in table_lines
    )
    assert "| Controllo 29 |" not in rendered


def test_prompt_source_provenance_envelope_is_complete_when_body_is_truncated(
    tenant_context,
    source_c,
) -> None:
    source = source_c.model_copy(
        update={
            "id": "provenance-source",
            "doc_id": "provenance-doc",
            "filename": "provenance.pdf",
            "page": 19,
            "type": "text",
            "db_origin": "Qdrant + PostgresCanonicalEnrich + Neo4jFormulaSearch",
            "content": "Frase completa per il troncamento sicuro. " * 100,
        }
    )

    blocks = build_tier_context_blocks(
        [source],
        max_chars=650,
        context=tenant_context,
    )
    rendered = blocks.combined()

    assert "Filename: provenance.pdf" in rendered
    assert "Page: 19" in rendered
    assert "Tier: C" in rendered
    assert "Type: text" in rendered
    assert "Origin: Qdrant + PostgresCanonicalEnrich + Neo4jFormulaSearch" in rendered
    assert "--- RETRIEVED SOURCE [1] END ---" in rendered
    assert TRUNCATION_MARKER in rendered
    assert blocks.context_chars <= 650
