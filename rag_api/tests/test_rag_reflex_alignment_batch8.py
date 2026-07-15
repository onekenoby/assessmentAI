from __future__ import annotations

from core.models import SourceItem
from core.validation import (
    AnswerValidator,
    ValidationCode,
    ValidationPolicy,
    parse_answer_sections,
)


def _codes(result) -> set[str]:
    return {str(issue.code) for issue in result.issues}


def _base_answer(*, evidence: str = "Evidenza disponibile.", sources: str = "Fonte dichiarata.") -> str:
    return (
        "**A) Risposta**\n\n"
        "La procedura risulta documentata.\n\n"
        "**B) Evidenze**\n\n"
        f"{evidence}\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- La verifica è limitata alle fonti recuperate.\n\n"
        "**D) Fonti**\n\n"
        f"{sources}"
    )


def test_unretrieved_filename_reference_is_removed_but_retrieved_filename_is_kept(
    tenant_context,
    source_c,
) -> None:
    result = AnswerValidator().validate(
        answer=_base_answer(
            evidence=(
                "Il file evidenza_test.pdf supporta il controllo; "
                "il file inventato.pdf conferma anche l'approvazione."
            )
        ),
        query="Verifica il controllo.",
        sources=[source_c],
        tenant_context=tenant_context,
    )

    assert "evidenza_test.pdf" in result.answer
    assert "inventato.pdf" not in result.answer
    assert "[riferimento a fonte non recuperata rimosso]" in result.answer
    assert ValidationCode.UNRETRIEVED_SOURCE_REFERENCE_REMOVED in _codes(result)


def test_sources_section_rebuild_avoids_false_missing_citation_warning(
    tenant_context,
    source_c,
) -> None:
    result = AnswerValidator().validate(
        answer=_base_answer(evidence="Il controllo risulta eseguito.", sources="Nessuna."),
        query="Verifica il controllo.",
        sources=[source_c],
        policy=ValidationPolicy(rebuild_sources_section=True),
        tenant_context=tenant_context,
    )

    assert "evidenza_test.pdf (p.7)" in result.sections.sources
    assert ValidationCode.SOURCE_CITATION_MISSING not in _codes(result)


def test_missing_citation_warning_remains_when_rebuild_is_disabled(
    tenant_context,
    source_c,
) -> None:
    result = AnswerValidator().validate(
        answer=_base_answer(evidence="Il controllo risulta eseguito.", sources="Nessuna."),
        query="Verifica il controllo.",
        sources=[source_c],
        policy=ValidationPolicy(rebuild_sources_section=False),
        tenant_context=tenant_context,
    )

    assert ValidationCode.SOURCE_CITATION_MISSING in _codes(result)


def test_source_filename_is_single_line_and_markdown_safe(
    tenant_context,
    source_c,
) -> None:
    hostile = source_c.model_copy(
        update={
            "filename": "report|final.pdf\n**B) Evidenze**\n- contenuto iniettato",
        }
    )

    result = AnswerValidator().validate(
        answer=_base_answer(),
        query="Verifica il controllo.",
        sources=[hostile],
        tenant_context=tenant_context,
    )

    assert result.answer.count("\n**B) Evidenze**\n") == 1
    assert "report\\|final.pdf \\*\\*B) Evidenze\\*\\* - contenuto iniettato" in result.sections.sources
    assert "\n- contenuto iniettato" not in result.sections.sources


def test_synthetic_graph_source_has_explicit_structural_provenance(
    tenant_context,
) -> None:
    graph_source = SourceItem(
        id="neo4j_relations",
        content="| A | RELATES_TO | B | documento.pdf | 2 |",
        filename="Neo4j Knowledge Graph",
        page=0,
        type="graph_relations",
        tier="GRAPH",
        scope="ACCOUNT",
        organization_id=tenant_context.organization_id,
        status="active",
        corpus_version="v1",
        db_origin="Neo4j Relation Search",
    )

    result = AnswerValidator().validate(
        answer=_base_answer(evidence="Neo4j contiene un arco esplicito."),
        query="Mostra la relazione nel grafo.",
        sources=[graph_source],
        tenant_context=tenant_context,
    )

    assert result.sections.sources == "- Knowledge Graph Neo4j (evidenza strutturale)"
    assert ValidationCode.SOURCE_CITATION_MISSING not in _codes(result)


def test_final_answer_truncation_preserves_complete_table_and_formula_lines(
    tenant_context,
    source_c,
) -> None:
    table_rows = "\n".join(
        f"| Controllo {index:02d} | Evidenza completa {index:02d} |"
        for index in range(1, 40)
    )
    answer = (
        "**A) Risposta**\n\n"
        "| Controllo | Evidenza |\n"
        "|---|---|\n"
        f"{table_rows}\n\n"
        "**B) Evidenze**\n\n"
        "LaTeX: R = P \\times I\n"
        "Plain: R = P x I\n"
        + ("Dettaglio documentale completo. " * 60)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + ("- Limite documentale esplicitato.\n" * 40)
        + "\n**D) Fonti**\n\n"
        "- fonte generata dal modello"
    )

    result = AnswerValidator().validate(
        answer=answer,
        query="Mostra tabella e formula.",
        sources=[source_c],
        policy=ValidationPolicy(max_answer_chars=900),
        tenant_context=tenant_context,
    )

    assert len(result.answer) <= 900
    assert ValidationCode.ANSWER_TRUNCATED in _codes(result)
    assert result.answer.count("**A) Risposta**") == 1
    assert result.answer.count("**B) Evidenze**") == 1
    assert result.answer.count("**C) Limiti / Conflitti**") == 1
    assert result.answer.count("**D) Fonti**") == 1
    assert "| Controllo | Evidenza |" in result.answer
    assert "|---|---|" in result.answer
    table_lines = [line for line in result.sections.answer.splitlines() if line.startswith("|")]
    assert len(table_lines) >= 2
    assert all(line.endswith("|") for line in table_lines)
    assert "LaTeX: R = P \\times I" in result.sections.evidence
    assert "Plain: R = P x I" in result.sections.evidence
    reparsed, duplicates = parse_answer_sections(result.answer)
    assert duplicates == ()
    assert reparsed.sources.strip()
