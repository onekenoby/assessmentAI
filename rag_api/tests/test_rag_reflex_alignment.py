from __future__ import annotations

import pytest

from core.models import RagExecutionMode, RagIntent
from core.rag_service import RagQueryCommand, RagQueryRouter


@pytest.fixture()
def router() -> RagQueryRouter:
    return RagQueryRouter()


def test_unquoted_document_filename_is_extracted_without_sentence_prefix(
    router: RagQueryRouter,
) -> None:
    decision = router.route(
        RagQueryCommand(
            query=(
                "Valuta se il documento evidenza.pdf è attinente "
                "alla domanda di assessment e indica i gap."
            )
        )
    )

    assert decision.requested_document == "evidenza.pdf"


def test_quoted_document_filename_is_extracted(router: RagQueryRouter) -> None:
    decision = router.route(
        RagQueryCommand(
            query='Analizza il documento "Guida_operativa.pdf" e riassumi i requisiti.'
        )
    )

    assert decision.requested_document == "Guida_operativa.pdf"


def test_explicit_target_document_has_priority_over_query_inference(
    router: RagQueryRouter,
) -> None:
    decision = router.route(
        RagQueryCommand(
            query="Confronta old.pdf con i requisiti recuperati.",
            target_document="target.pdf",
        )
    )

    assert decision.requested_document == "target.pdf"


@pytest.mark.parametrize(
    "query",
    [
        "Descrivi il controllo documentale.",
        "Spiega la relazione tra rischio e controllo.",
        "Describe the access control objective.",
    ],
)
def test_single_weak_checklist_concept_does_not_enable_strict_checklist(
    router: RagQueryRouter,
    query: str,
) -> None:
    decision = router.route(RagQueryCommand(query=query))

    assert decision.strict_checklist_mode is False


@pytest.mark.parametrize(
    "query",
    [
        "Elenca i controlli e le evidenze richieste.",
        "List the controls and evidence required for the audit.",
    ],
)
def test_two_distinct_checklist_concepts_enable_strict_checklist(
    router: RagQueryRouter,
    query: str,
) -> None:
    decision = router.route(RagQueryCommand(query=query))

    assert decision.strict_checklist_mode is True


@pytest.mark.parametrize(
    "query",
    [
        "Mostra un grafico del rischio.",
        "Disegna un diagramma del processo.",
        "Crea un flowchart della procedura.",
        "Descrivi l'architettura del sistema.",
    ],
)
def test_chart_intent_uses_the_same_general_vocabulary_as_reflex(
    router: RagQueryRouter,
    query: str,
) -> None:
    decision = router.route(RagQueryCommand(query=query))

    assert decision.intent == RagIntent.CHART


def test_operational_calculation_remains_math_direct_not_formula_lookup(
    router: RagQueryRouter,
) -> None:
    decision = router.route(
        RagQueryCommand(
            query=(
                "Calcola la copertura di una checklist di 100 controlli: "
                "70 implementati e 20 parziali che valgono al 50%."
            )
        )
    )

    assert decision.execution_mode == RagExecutionMode.MATH_DIRECT
    assert decision.formula_strict_mode is False


def test_explicit_neo4j_relation_query_remains_graph_strict(
    router: RagQueryRouter,
) -> None:
    decision = router.route(
        RagQueryCommand(
            query=(
                "Usando Neo4j, verifica le relazioni esplicite tra Asset, "
                "compromissione della triade CID e funzione Respond."
            )
        )
    )

    assert decision.execution_mode == RagExecutionMode.GRAPH_RELATION_STRICT
    assert decision.graph_search_mode is True
    assert decision.graph_relation_mode is True


def test_explanatory_relation_query_does_not_force_graph_strict(
    router: RagQueryRouter,
) -> None:
    decision = router.route(
        RagQueryCommand(query="Spiega la relazione tra rischio e controllo.")
    )

    assert decision.graph_relation_mode is False
    assert decision.execution_mode == RagExecutionMode.RAG_GENERATION


def test_page_range_matches_reflex_behavior(router: RagQueryRouter) -> None:
    decision = router.route(
        RagQueryCommand(query="Analizza il documento policy.pdf, pag. 8-10.")
    )

    assert decision.requested_document == "policy.pdf"
    assert decision.requested_pages == (8, 9, 10)
