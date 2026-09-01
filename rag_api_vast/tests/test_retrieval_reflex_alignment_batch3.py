from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from core.config import settings
from core.models import RagAnswerMode, RagIntent, RetrievalCandidate
from core.retrieval import HybridRetrievalEngine


class _Cursor:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.calls: list[tuple[str, tuple | list]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.calls.append((str(sql), params))

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _PostgresResources:
    def __init__(self, rows=()):
        self.cursor_obj = _Cursor(rows)

    @contextmanager
    def postgres_connection(self, context):
        yield _Connection(self.cursor_obj)


class _Neo4jSession:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher, **params):
        self.calls.append((str(cypher), dict(params)))
        return list(self._rows)


class _Neo4jDriver:
    def __init__(self, rows=()):
        self.session_obj = _Neo4jSession(rows)

    def session(self):
        return self.session_obj


class _GraphResources:
    def __init__(self, driver):
        self._driver = driver

    def get_neo4j_driver(self, required=False):
        return self._driver


class _ScopedGraphEngine(HybridRetrievalEngine):
    def __init__(self, *, config, resource_manager, qdrant_hits, expanded_hits):
        super().__init__(config=config, resource_manager=resource_manager)
        self._qdrant_hits = list(qdrant_hits)
        self._expanded_hits = list(expanded_hits)
        self.seed_ids: list[str] = []
        self.neighbour_scope: dict = {}
        self.retrieve_scope: dict = {}

    def _search_qdrant(self, *args, **kwargs):
        return list(self._qdrant_hits)

    def _search_neo4j_entities(self, *args, **kwargs):
        return []

    def _search_neo4j_formulas(self, *args, **kwargs):
        return []

    def _search_neo4j_relations(self, *args, **kwargs):
        return []

    def _get_neighbor_chunk_ids(self, seed_ids, **kwargs):
        self.seed_ids = list(seed_ids)
        self.neighbour_scope = dict(kwargs)
        return [item.id for item in self._expanded_hits]

    def _retrieve_qdrant_points_by_ids(self, ids, **kwargs):
        self.retrieve_scope = dict(kwargs)
        pages = set(kwargs.get("target_pages") or ())
        document = kwargs.get("target_document")
        return [
            item
            for item in self._expanded_hits
            if (not document or item.filename == document)
            and (not pages or item.page in pages)
        ]

    def _get_graph_entities(self, *args, **kwargs):
        return {}


def _candidate(
    identifier: str,
    *,
    filename: str,
    page: int,
    score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        id=identifier,
        content=f"Contenuto {identifier}",
        filename=filename,
        page=page,
        doc_id=f"doc-{identifier}",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        origin="test",
        score_vec=score,
    )


def test_qdrant_filter_pushes_page_scope_into_payload_filter(tenant_context):
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())

    qfilter = engine._build_qdrant_filter(tenant_context, target_pages=(9, 8, 9))
    if hasattr(qfilter, "model_dump"):
        # qdrant-client/Pydantic versions expose different optional fields
        # (range, geo_*, values_count, ...), all set to None.  They are not
        # part of the page-scope semantics being tested.
        dumped = qfilter.model_dump(exclude_none=True)
    elif hasattr(qfilter, "dict"):
        dumped = qfilter.dict(exclude_none=True)
    else:
        dumped = qfilter

    page_branch = dumped["must"][1]["should"]
    assert page_branch == [
        {"key": "page", "match": {"any": [8, 9]}},
        {"key": "page_no", "match": {"any": [8, 9]}},
    ]


def test_postgres_bm25_and_exact_phrase_push_page_scope_before_limit(tenant_context):
    resources = _PostgresResources()
    engine = HybridRetrievalEngine(config=settings, resource_manager=resources)

    engine._search_pg_bm25(
        "incident response",
        limit=20,
        tenant_context=tenant_context,
        target_pages=(4, 5),
    )
    bm25_sql, bm25_params = resources.cursor_obj.calls[-1]
    assert "::int = ANY(%s)" in bm25_sql
    assert bm25_sql.find("::int = ANY(%s)") < bm25_sql.find("LIMIT %s")
    assert bm25_params[-2] == [4, 5]

    engine._search_pg_exact_phrases(
        'Trova "incident response"',
        limit=12,
        tenant_context=tenant_context,
        target_pages=(7,),
    )
    exact_sql, exact_params = resources.cursor_obj.calls[-1]
    assert "::int = ANY(%s)" in exact_sql
    assert exact_sql.find("::int = ANY(%s)") < exact_sql.find("LIMIT %s")
    assert exact_params[-2] == [7]


def test_postgres_document_scope_has_exact_provenance_projection_and_page_pushdown(
    tenant_context,
):
    resources = _PostgresResources()
    engine = HybridRetrievalEngine(config=settings, resource_manager=resources)

    engine._search_pg_document_scope(
        "policy.pdf",
        "requisiti",
        limit=25,
        tenant_context=tenant_context,
        target_pages=(3,),
    )

    sql, params = resources.cursor_obj.calls[-1]
    projection = sql.split("SELECT", 3)[-1].split("FROM ranked", 1)[0]
    assert projection.count("content_raw") == 1
    assert projection.count("content_semantic") == 1
    assert projection.count("embedding_model") == 1
    assert "::int = ANY(%s)" in sql
    assert sql.find("filename_norm = %s") < sql.find("::int = ANY(%s)") < sql.find("LIMIT %s")
    assert params[-2] == [3]


@pytest.mark.parametrize(
    ("method_name", "query"),
    [
        ("_search_neo4j_entities", "Mostra le entità incident response"),
        ("_search_neo4j_formulas", "Elenca le formule di rischio"),
        ("_search_neo4j_relations", "Mostra le relazioni tra rischio e controllo"),
    ],
)
def test_neo4j_direct_search_pushes_document_and_page_scope_before_limit(
    tenant_context,
    method_name,
    query,
):
    driver = _Neo4jDriver()
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())
    method = getattr(engine, method_name)

    kwargs = {
        "limit": 10,
        "target_document": "policy.pdf",
        "target_pages": (8, 9),
        "tenant_context": tenant_context,
        "driver": driver,
    }
    method(query, **kwargs)

    cypher, params = driver.session_obj.calls[-1]
    assert cypher.find("$requested_doc_norm") < cypher.find("LIMIT $limit")
    assert cypher.find("$requested_pages") < cypher.find("LIMIT $limit")
    assert params["requested_doc_norm"] == "policy"
    assert params["requested_doc_lower"] == "policy.pdf"
    assert params["requested_pages"] == [8, 9]


def test_neighbor_search_pushes_scope_on_c2_before_limit(tenant_context):
    driver = _Neo4jDriver([{"chunk_id": "neighbour-1"}])
    engine = HybridRetrievalEngine(config=settings, resource_manager=SimpleNamespace())

    ids = engine._get_neighbor_chunk_ids(
        ["seed-1"],
        limit=4,
        tenant_context=tenant_context,
        driver=driver,
        target_document="policy.pdf",
        target_pages=(11,),
    )

    cypher, params = driver.session_obj.calls[-1]
    assert "coalesce(c2.filename" in cypher
    assert "coalesce(c2.page, 0) IN $requested_pages" in cypher
    assert cypher.find("coalesce(c2.page, 0) IN $requested_pages") < cypher.find("LIMIT $limit")
    assert params["requested_pages"] == [11]
    assert ids == ["neighbour-1"]


@pytest.mark.asyncio
async def test_graph_expansion_uses_only_document_and_page_scoped_seeds(tenant_context):
    driver = _Neo4jDriver()
    resources = _GraphResources(driver)
    config = settings.model_copy(
        update={
            "pg_enrich_enabled": False,
            "neo4j_enabled": True,
            "graph_expand_enabled": True,
            "graph_max_neighbor_chunks": 4,
        }
    )
    hits = [
        _candidate("foreign-high", filename="other.pdf", page=8, score=0.99),
        _candidate("wrong-page", filename="policy.pdf", page=7, score=0.98),
        _candidate("scoped-seed", filename="policy.pdf", page=8, score=0.50),
    ]
    expanded = [
        _candidate("expanded-good", filename="policy.pdf", page=8, score=0.0),
        _candidate("expanded-wrong-page", filename="policy.pdf", page=9, score=0.0),
    ]
    engine = _ScopedGraphEngine(
        config=config,
        resource_manager=resources,
        qdrant_hits=hits,
        expanded_hits=expanded,
    )

    candidates, debug = await engine.retrieve_candidates(
        query="Analizza policy.pdf pagina 8",
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        target_document="policy.pdf",
        target_pages=(8,),
        wants_evidence=True,
        graph_relation_mode=False,
        formula_mode=False,
        exhaustive_formula_lookup=False,
        tenant_context=tenant_context,
    )

    assert engine.seed_ids == ["scoped-seed"]
    assert engine.neighbour_scope["target_document"] == "policy.pdf"
    assert engine.neighbour_scope["target_pages"] == (8,)
    assert engine.retrieve_scope["target_document"] == "policy.pdf"
    assert engine.retrieve_scope["target_pages"] == (8,)
    assert {item.id for item in candidates} == {"scoped-seed", "expanded-good"}
    assert debug.graph_expand_used is True
    assert debug.neo4j_expanded_hits == 1
