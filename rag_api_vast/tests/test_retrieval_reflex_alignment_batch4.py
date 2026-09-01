from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import settings
from core.models import RagAnswerMode, RagIntent, RetrievalCandidate
from core.retrieval import HybridRetrievalEngine, _merge_candidates


def _text_candidate(*, origin: str = "Qdrant") -> RetrievalCandidate:
    return RetrievalCandidate(
        id="chunk-shared",
        content="Contenuto canonico del documento.",
        filename="risk_policy.pdf",
        page=12,
        page_chunk_index=2,
        doc_id="doc-risk",
        type="text",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        ingestion_run_id="run-qdrant",
        corpus_version="v1",
        classification="internal",
        embedding_model="BAAI/bge-m3",
        origin=origin,
        score_vec=0.82,
        metadata={"filename": "risk_policy.pdf"},
    )


def _formula_candidate(key: str, latex: str) -> RetrievalCandidate:
    # Python 3.11 non consente backslash nell'espressione di una f-string.
    # Calcoliamo quindi la variante plain prima della formattazione.
    plain = latex.replace("\\times", "x")

    return RetrievalCandidate(
        id="chunk-shared",
        content=(
            "Formula from Knowledge Graph:\n"
            f"LaTeX: {latex} | Plain: {plain}"
        ),
        filename="risk_policy.pdf",
        page=12,
        page_chunk_index=2,
        doc_id="doc-risk",
        type="formula",
        tier="GRAPH",
        source_tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        ingestion_run_id="run-graph",
        corpus_version="v1",
        classification="internal",
        origin="Neo4jFormulaSearch",
        score_graph=5.0,
        metadata={"formula_key": key},
    )


def _pg_row() -> dict[str, object]:
    return {
        "content_raw": "Testo raw canonico recuperato da PostgreSQL.",
        "content_semantic": "Testo semantico canonico.",
        "metadata": {
            "filename": "risk_policy.pdf",
            "page": 12,
            "page_chunk_index": 2,
            "doc_id": "doc-risk",
            "source_name": "risk_policy.pdf",
            "source_type": "pdf",
            "log_id": 71,
            "chunk_index": 9,
            "toon_type": "text",
            "tier": "C",
            "scope": "ACCOUNT",
            "organization_id": 1234,
            "status": "active",
        },
        "ingestion_ts": "2026-07-14T10:30:00+00:00",
        "scope": "ACCOUNT",
        "organization_id": 1234,
        "tier": "C",
        "status": "active",
        "ingestion_run_id": "run-postgres",
        "corpus_version": "v2",
        "classification": "confidential",
        "embedding_model": "BAAI/bge-m3",
    }


def test_merge_same_chunk_preserves_all_distinct_graph_formulas_and_keys():
    merged = _merge_candidates(_text_candidate(), _formula_candidate("risk", r"R = P \times I"))
    merged = _merge_candidates(merged, _formula_candidate("ale", r"ALE = SLE \times ARO"))

    assert merged.type == "formula"
    assert merged.tier == "C"
    assert merged.source_tier == "C"
    assert merged.content.count("R = P \\times I") == 1
    assert merged.content.count("ALE = SLE \\times ARO") == 1
    assert merged.metadata["graph_formula_keys"] == ["risk", "ale"]
    assert len(merged.metadata["graph_formula_contents"]) == 2
    assert merged.origin == "Qdrant + Neo4jFormulaSearch"
    assert merged.score_vec == pytest.approx(0.82)
    assert merged.score_graph == pytest.approx(5.0)


def test_merge_formula_is_idempotent_for_duplicate_neo4j_rows():
    formula = _formula_candidate("risk", r"R = P \times I")
    merged = _merge_candidates(_merge_candidates(_text_candidate(), formula), formula)

    assert merged.content.count("R = P \\times I") == 1
    assert merged.metadata["graph_formula_keys"] == ["risk"]
    assert merged.origin == "Qdrant + Neo4jFormulaSearch"


def test_postgres_canonical_enrichment_does_not_erase_direct_graph_formula():
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())
    merged = _merge_candidates(
        _text_candidate(),
        _formula_candidate("risk", r"R = P \times I"),
    )

    enriched = engine._enrich_candidate_from_pg(merged, _pg_row(), formula_mode=True)

    assert enriched.content.startswith("Testo raw canonico recuperato da PostgreSQL.")
    assert enriched.content.count("R = P \\times I") == 1
    assert enriched.type == "formula"
    assert enriched.tier == "C"
    assert enriched.source_tier == "C"
    assert enriched.origin == (
        "Qdrant + Neo4jFormulaSearch + PostgresCanonicalEnrich"
    )
    assert enriched.metadata["graph_formula_keys"] == ["risk"]
    assert len(enriched.metadata["graph_formula_contents"]) == 1


def test_final_source_materialization_preserves_postgres_provenance_fields():
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())
    enriched = engine._enrich_candidate_from_pg(
        _text_candidate(),
        _pg_row(),
        formula_mode=False,
    )

    source = enriched.to_source_item(request_id="request-batch4")

    assert source.pg_ingestion_ts == "2026-07-14T10:30:00+00:00"
    assert source.pg_source_name == "risk_policy.pdf"
    assert source.pg_source_type == "pdf"
    assert source.pg_log_id == 71
    assert source.pg_chunk_id == 9
    assert source.pg_page_chunk_index == 2
    assert source.pg_toon_type == "text"
    assert source.db_origin == "Qdrant + PostgresCanonicalEnrich"
    assert source.request_id == "request-batch4"


class _Batch4Resources:
    def get_neo4j_driver(self, *, required=False):
        return object()


class _Batch4Retrieval(HybridRetrievalEngine):
    def _search_qdrant(self, *args, **kwargs):
        return [_text_candidate()]

    def _search_pg_bm25(self, *args, **kwargs):
        return []

    def _search_pg_exact_phrases(self, *args, **kwargs):
        return []

    def _search_pg_document_scope(self, *args, **kwargs):
        return []

    def _search_pg_glossary_term(self, *args, **kwargs):
        return []

    def _search_neo4j_entities(self, *args, **kwargs):
        return []

    def _search_neo4j_formulas(self, *args, **kwargs):
        return [
            _formula_candidate("risk", r"R = P \times I"),
            _formula_candidate("ale", r"ALE = SLE \times ARO"),
        ]

    def _fetch_pg_chunks_by_uuid(self, *args, **kwargs):
        return {"chunk-shared": _pg_row()}

    def _get_graph_entities(self, *args, **kwargs):
        return {}

    def _get_formulas_for_chunks(self, *args, **kwargs):
        # Simula un enrichment Neo4j contestuale non disponibile: le formule
        # ottenute dalla ricerca diretta devono comunque sopravvivere.
        return {}


@pytest.mark.asyncio
async def test_full_retrieval_merge_keeps_formula_and_canonical_provenance(tenant_context):
    config = settings.model_copy(
        update={
            "pg_enrich_enabled": True,
            "neo4j_enabled": True,
            "graph_expand_enabled": False,
        }
    )
    engine = _Batch4Retrieval(config=config, resource_manager=_Batch4Resources())

    candidates, debug = await engine.retrieve_candidates(
        query="Elenca le formule del rischio nel documento risk_policy.pdf",
        intent=RagIntent.FORMULA,
        answer_mode=RagAnswerMode.KNOWLEDGE,
        target_document="risk_policy.pdf",
        target_pages=(12,),
        wants_evidence=False,
        graph_relation_mode=False,
        formula_mode=True,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.content.startswith("Testo raw canonico recuperato da PostgreSQL.")
    assert candidate.content.count("R = P \\times I") == 1
    assert candidate.content.count("ALE = SLE \\times ARO") == 1
    assert candidate.metadata["graph_formula_keys"] == ["risk", "ale"]
    assert candidate.origin == (
        "Qdrant + Neo4jFormulaSearch + PostgresCanonicalEnrich"
    )
    assert candidate.tier == "C"
    assert candidate.scope == "ACCOUNT"
    assert candidate.organization_id == 1234
    assert candidate.ingestion_run_id == "run-postgres"
    assert candidate.corpus_version == "v2"
    assert candidate.classification == "confidential"
    assert debug.qdrant_hits == 1
    assert debug.neo4j_direct_hits == 2
