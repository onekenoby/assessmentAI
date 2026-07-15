from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.config import settings
from core.models import RagAnswerMode, RagIntent, RetrievalCandidate
from core.retrieval import HybridRetrievalEngine, _exact_phrases, _expand_query


class _Cursor:
    def __init__(self, rows):
        self._rows = list(rows)
        self.executed_sql = ""
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.executed_sql = str(sql)
        self.executed_params = params

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _PostgresResources:
    def __init__(self, rows):
        self.cursor_obj = _Cursor(rows)

    @contextmanager
    def postgres_connection(self, context):
        yield _Connection(self.cursor_obj)


class _Neo4jSession:
    def __init__(self, rows):
        self._rows = list(rows)
        self.cypher = ""
        self.params = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher, **params):
        self.cypher = str(cypher)
        self.params = dict(params)
        return list(self._rows)


class _Neo4jDriver:
    def __init__(self, rows):
        self.session_obj = _Neo4jSession(rows)

    def session(self):
        return self.session_obj


class _StubRetrieval(HybridRetrievalEngine):
    def __init__(self, *, config, doc_hits):
        super().__init__(config=config, resource_manager=SimpleNamespace())
        self._doc_hits = list(doc_hits)

    def _search_qdrant(self, *args, **kwargs):
        return []

    def _search_pg_bm25(self, *args, **kwargs):
        return []

    def _search_pg_exact_phrases(self, *args, **kwargs):
        return []

    def _search_pg_document_scope(self, *args, **kwargs):
        return list(self._doc_hits)

    def _search_pg_glossary_term(self, *args, **kwargs):
        return []

    def _fetch_pg_chunks_by_uuid(self, *args, **kwargs):
        return {}


def _candidate(identifier: str, filename: str = "policy.pdf") -> RetrievalCandidate:
    return RetrievalCandidate(
        id=identifier,
        content="Contenuto documentale pertinente",
        filename=filename,
        page=3,
        doc_id="doc-policy",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        origin="PostgresDocScope",
        score_bm25=0.75,
        metadata={"score_doc_scope": 1.0},
    )


def test_postgres_document_scope_select_has_aligned_provenance_columns(tenant_context):
    ingestion_ts = datetime(2026, 7, 14, 10, 30, tzinfo=timezone.utc)
    metadata = {
        "filename": "policy.pdf",
        "page": 3,
        "doc_id": "doc-policy",
        "tier": "C",
        "scope": "ACCOUNT",
        "organization_id": 1234,
        "status": "active",
    }
    row = (
        "chunk-1",
        "raw canonico",
        "semantic canonico",
        metadata,
        "ACCOUNT",
        1234,
        "C",
        "active",
        "run-1",
        "v1",
        "internal",
        "BAAI/bge-m3",
        ingestion_ts,
        0.73,
    )
    resources = _PostgresResources([row])
    engine = HybridRetrievalEngine(config=settings, resource_manager=resources)

    results = engine._search_pg_document_scope(
        "policy.pdf",
        "requisiti di sicurezza",
        limit=25,
        tenant_context=tenant_context,
    )

    sql = resources.cursor_obj.executed_sql
    assert not __import__("re").search(
        r"classification\s*,\s*classification\s*,",
        sql,
        flags=__import__("re").IGNORECASE,
    )
    assert len(results) == 1
    assert results[0].embedding_model == "BAAI/bge-m3"
    assert results[0].score_bm25 == pytest.approx(0.73)
    assert results[0].metadata["pg_ingestion_ts"] == ingestion_ts.isoformat()


def test_formula_metric_query_expands_notification_threshold_aliases():
    query = (
        "Elenca le metriche e le soglie relative all'obbligo di notifica "
        "degli incidenti significativi e agli utenti impattati."
    )

    expanded = _expand_query(query, formula_mode=True)
    phrases = _exact_phrases(query)

    assert "affected users" in expanded.casefold()
    assert "incident notification" in expanded.casefold()
    assert "significant incident" in expanded.casefold()
    assert any(item.casefold() == "obbligo di notifica" for item in phrases)


def test_non_formula_query_is_not_expanded_with_threshold_aliases():
    query = "Descrivi l'obbligo di notifica degli incidenti significativi."

    assert _expand_query(query, formula_mode=False) == query


def test_neo4j_formula_document_scope_is_applied_inside_query_before_limit(
    tenant_context,
):
    row = {
        "chunk_id": "chunk-policy-formula",
        "doc_id": "doc-policy",
        "filename": "policy.pdf",
        "page": 7,
        "page_chunk_index": 1,
        "scope": "ACCOUNT",
        "organization_id": 1234,
        "source_tier": "C",
        "status": "active",
        "ingestion_run_id": "run-1",
        "corpus_version": "v1",
        "classification": "internal",
        "latex": "R = P \\times I",
        "plain": "R = P x I",
        "meaning": "Risk score",
        "formula_key": "risk-score",
    }
    driver = _Neo4jDriver([row])
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())

    results = engine._search_neo4j_formulas(
        "Elenca le formule nel documento policy.pdf",
        limit=6,
        target_document="policy.pdf",
        tenant_context=tenant_context,
        driver=driver,
    )

    session = driver.session_obj
    where_pos = session.cypher.find("$requested_doc_norm")
    limit_pos = session.cypher.find("LIMIT $limit")
    assert 0 <= where_pos < limit_pos
    assert session.params["requested_doc_norm"] == "policy"
    assert session.params["requested_doc_lower"] == "policy.pdf"
    assert [item.filename for item in results] == ["policy.pdf"]


@pytest.mark.asyncio
async def test_retrieval_debug_reports_document_scope_hits(tenant_context):
    config = settings.model_copy(
        update={
            "pg_enrich_enabled": True,
            "neo4j_enabled": False,
            "graph_expand_enabled": False,
        }
    )
    engine = _StubRetrieval(config=config, doc_hits=[_candidate("doc-hit")])

    candidates, debug = await engine.retrieve_candidates(
        query="Analizza il documento policy.pdf",
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        target_document="policy.pdf",
        target_pages=(),
        wants_evidence=True,
        graph_relation_mode=False,
        formula_mode=False,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert [item.id for item in candidates] == ["doc-hit"]
    assert debug.postgres_document_scope_hits == 1
