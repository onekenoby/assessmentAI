from __future__ import annotations

from core.models import GraphEntity, SourceItem
from core.rag_service import _dedupe_sources


def _source(
    identifier: str,
    *,
    content: str,
    source_type: str = "text",
    tier: str = "C",
    scope: str = "ACCOUNT",
    organization_id: int | None = 1234,
    score: float = 0.0,
    origin: str = "Qdrant",
    graph_context: list[GraphEntity] | None = None,
    classification: str = "internal",
    **updates,
) -> SourceItem:
    return SourceItem(
        id=identifier,
        content=content,
        filename=updates.pop("filename", "risk_policy.pdf"),
        page=updates.pop("page", 12),
        page_chunk_index=updates.pop("page_chunk_index", 2),
        doc_id=updates.pop("doc_id", "doc-risk"),
        type=source_type,
        tier=tier,
        scope=scope,
        organization_id=organization_id,
        status="active",
        score=score,
        db_origin=origin,
        graph_context=graph_context or [],
        classification=classification,
        **updates,
    )


def test_final_dedupe_merges_text_and_formula_for_same_chunk():
    text = _source(
        "chunk-shared",
        content="Il documento definisce il modello quantitativo del rischio.",
        score=1.2,
        origin="Qdrant + PostgresCanonicalEnrich",
    )
    formula = _source(
        "chunk-shared",
        content=r"Formula from Knowledge Graph: R = P \times I",
        source_type="formula",
        tier="GRAPH",
        score=5.0,
        origin="Neo4jFormulaSearch",
    )

    merged = _dedupe_sources((text, formula))

    assert len(merged) == 1
    source = merged[0]
    assert source.type == "formula"
    assert source.tier == "C"
    assert source.score == 5.0
    assert source.content.startswith("Il documento definisce")
    assert source.content.count(r"R = P \times I") == 1
    assert "Formule collegate dal Knowledge Graph" in source.content
    assert source.db_origin == (
        "Qdrant + PostgresCanonicalEnrich + Neo4jFormulaSearch"
    )


def test_final_dedupe_unions_graph_context_without_duplicates():
    first = _source(
        "chunk-graph",
        content="Controllo degli accessi.",
        graph_context=[
            GraphEntity(name="MFA", type="Control", relation="IMPLEMENTS"),
        ],
    )
    second = _source(
        "chunk-graph",
        content="Controllo degli accessi con autenticazione forte.",
        origin="Neo4jEntitySearch",
        graph_context=[
            GraphEntity(name="MFA", type="Control", relation="IMPLEMENTS"),
            GraphEntity(name="Identity", type="Asset", relation="PROTECTS"),
        ],
    )

    source = _dedupe_sources((first, second))[0]

    assert source.content == "Controllo degli accessi con autenticazione forte."
    assert [(e.name, e.type, e.relation) for e in source.graph_context] == [
        ("MFA", "Control", "IMPLEMENTS"),
        ("Identity", "Asset", "PROTECTS"),
    ]
    assert source.db_origin == "Qdrant + Neo4jEntitySearch"


def test_final_dedupe_preserves_postgres_provenance_and_stricter_classification():
    qdrant = _source(
        "chunk-pg",
        content="Testo semantico breve.",
        score=0.8,
        classification="internal",
    )
    postgres = _source(
        "chunk-pg",
        content="Testo raw canonico più completo recuperato da PostgreSQL.",
        score=0.6,
        origin="PostgresCanonicalEnrich",
        classification="confidential",
        pg_ingestion_ts="2026-07-14T10:30:00+00:00",
        pg_source_name="risk_policy.pdf",
        pg_source_type="pdf",
        pg_log_id=71,
        pg_chunk_id=9,
        pg_page_chunk_index=2,
        pg_toon_type="text",
        embedding_model="BAAI/bge-m3",
    )

    source = _dedupe_sources((qdrant, postgres))[0]

    assert source.content == postgres.content
    assert source.score == 0.8
    assert source.classification == "confidential"
    assert source.pg_ingestion_ts == "2026-07-14T10:30:00+00:00"
    assert source.pg_source_name == "risk_policy.pdf"
    assert source.pg_source_type == "pdf"
    assert source.pg_log_id == 71
    assert source.pg_chunk_id == 9
    assert source.pg_page_chunk_index == 2
    assert source.pg_toon_type == "text"
    assert source.embedding_model == "BAAI/bge-m3"


def test_final_dedupe_does_not_collapse_distinct_generic_graph_rows():
    first = _source(
        "neo4j_relations",
        content="Asset | THREATENS | Availability",
        source_type="graph_relations",
        tier="GRAPH",
        page_chunk_index=0,
        doc_id="",
        origin="Neo4jRelationSearch",
    )
    second = _source(
        "neo4j_relations",
        content="Incident | TRIGGERS | Respond",
        source_type="graph_relations",
        tier="GRAPH",
        page_chunk_index=0,
        doc_id="",
        origin="Neo4jRelationSearch",
    )

    merged = _dedupe_sources((first, second))

    assert len(merged) == 2
    assert [source.content for source in merged] == [first.content, second.content]


def test_final_dedupe_identity_is_bound_to_tenant_scope():
    global_source = _source(
        "same-id",
        content="Contenuto globale.",
        tier="A",
        scope="GLOBAL",
        organization_id=None,
    )
    account_source = _source(
        "same-id",
        content="Contenuto account.",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
    )

    merged = _dedupe_sources((global_source, account_source))

    assert len(merged) == 2


def test_final_dedupe_is_idempotent_and_preserves_first_ranking_position():
    first = _source(
        "first",
        content="Prima fonte.",
        score=3.0,
    )
    duplicate = _source(
        "first",
        content="Prima fonte con dettaglio aggiuntivo.",
        score=2.0,
        origin="PostgresBM25",
    )
    second = _source(
        "second",
        content="Seconda fonte.",
        score=1.0,
        page=13,
        doc_id="doc-second",
    )

    once = _dedupe_sources((first, duplicate, second))
    twice = _dedupe_sources(once)

    assert [source.id for source in once] == ["first", "second"]
    assert once == twice
    assert once[0].content == "Prima fonte con dettaglio aggiuntivo."
    assert once[0].score == 3.0
    assert once[0].db_origin == "Qdrant + PostgresBM25"
