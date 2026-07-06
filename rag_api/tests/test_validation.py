from __future__ import annotations

from core.models import RagAnswerMode, RagExecutionMode, RagIntent
from core.validation import (
    AnswerValidator,
    ValidationCode,
    ValidationPolicy,
)


def _codes(result) -> set[str]:
    return {str(issue.code) for issue in result.issues}


def test_structure_is_repaired_and_sources_are_rebuilt(tenant_context, source_c) -> None:
    result = AnswerValidator().validate(
        answer="La procedura è documentata.",
        query="Esiste una procedura?",
        sources=[source_c],
        policy=ValidationPolicy(require_sources=True),
        tenant_context=tenant_context,
    )

    assert result.blocked is False
    assert result.repaired is True
    assert "**A) Risposta**" in result.answer
    assert "**D) Fonti**" in result.answer
    assert "evidenza_test.pdf (p.7)" in result.answer
    assert ValidationCode.STRUCTURE_REPAIRED in _codes(result)
    assert ValidationCode.SOURCES_SECTION_REBUILT in _codes(result)


def test_internal_metadata_and_external_url_are_removed(tenant_context, source_c) -> None:
    answer = """
**A) Risposta**
<thinking>segreto</thinking>
La risposta è disponibile su https://example.com.
request_id: 123e4567-e89b-12d3-a456-426614174000

**B) Evidenze**
evidenza_test.pdf

**C) Limiti / Conflitti**
Nessuno.

**D) Fonti**
evidenza_test.pdf
"""
    result = AnswerValidator().validate(
        answer=answer,
        query="test",
        sources=[source_c],
        tenant_context=tenant_context,
    )

    assert "segreto" not in result.answer
    assert "https://example.com" not in result.answer
    assert "request_id" not in result.answer
    assert ValidationCode.INTERNAL_METADATA_REMOVED in _codes(result)
    assert ValidationCode.EXTERNAL_URL_REMOVED in _codes(result)


def test_cross_tenant_source_blocks_response(
    tenant_context, source_c, foreign_source
) -> None:
    result = AnswerValidator().validate(
        answer="""
**A) Risposta**
Risposta.
**B) Evidenze**
Evidenze.
**C) Limiti / Conflitti**
Limiti.
**D) Fonti**
Fonti.
""",
        query="test",
        sources=[source_c, foreign_source],
        tenant_context=tenant_context,
    )

    assert result.blocked is True
    assert result.valid is False
    assert result.visible_sources == ()
    assert ValidationCode.TENANT_SOURCE_VIOLATION in _codes(result)


def test_requested_document_not_found_returns_safe_fallback(
    tenant_context, source_c
) -> None:
    result = AnswerValidator().validate(
        answer="Risposta generata",
        query="test",
        sources=[],
        policy=ValidationPolicy(requested_document="inesistente.pdf"),
        tenant_context=tenant_context,
    )

    assert result.valid is False
    assert result.blocked is False
    assert "inesistente.pdf" in result.answer
    assert ValidationCode.REQUESTED_DOCUMENT_SCOPE_VIOLATION not in _codes(result)
    assert ValidationCode.REQUESTED_DOCUMENT_NOT_FOUND in _codes(result)


def test_evidence_relevance_table_is_repaired(tenant_context, source_c) -> None:
    answer = """
**A) Risposta**
Livello di attinenza: 2
Percentuale stimata: 70%
Esito sintetico: Parzialmente attinente

**B) Evidenze**
Il documento evidenza_test.pdf a pagina 7 dimostra un test della procedura.

**C) Limiti / Conflitti**
Manca l'approvazione formale. Remediation: integrare la firma del responsabile.

**D) Fonti**
evidenza_test.pdf
"""
    result = AnswerValidator().validate(
        answer=answer,
        query="Valuta l'attinenza dell'evidenza",
        sources=[source_c],
        policy=ValidationPolicy(
            intent=RagIntent.AUDIT,
            answer_mode=RagAnswerMode.EVIDENCE_RELEVANCE,
            requested_document="evidenza_test.pdf",
        ),
        tenant_context=tenant_context,
    )

    assert result.blocked is False
    assert "| Livello di attinenza | Percentuale stimata | Esito sintetico |" in result.answer
    assert "| 2 | 70% | Parzialmente attinente |" in result.answer
    assert ValidationCode.EVIDENCE_TABLE_REPAIRED in _codes(result)


def test_formula_mode_requires_visible_formula(tenant_context, source_c) -> None:
    result = AnswerValidator().validate(
        answer="""
**A) Risposta**
Il risultato è disponibile.
**B) Evidenze**
evidenza_test.pdf
**C) Limiti / Conflitti**
Nessuno.
**D) Fonti**
evidenza_test.pdf
""",
        query="Calcola il valore",
        sources=[source_c],
        policy=ValidationPolicy(
            intent=RagIntent.FORMULA,
            execution_mode=RagExecutionMode.FORMULA_STRICT,
        ),
        tenant_context=tenant_context,
    )

    assert result.valid is False
    assert ValidationCode.FORMULA_NOT_VISIBLE in _codes(result)
