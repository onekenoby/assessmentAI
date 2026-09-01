from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import byte_engine as engine


@pytest.mark.parametrize(
    "value, expected",
    [
        ("doc.pdf", "doc.pdf"),
        (r"C:\\temp\\doc.pdf", "doc.pdf"),
        ("/tmp/doc.md", "doc.md"),
        ("  report.markdown  ", "report.markdown"),
    ],
)
def test_sanitize_filename(value, expected):
    assert engine.sanitize_filename(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "..", ".", "bad\x00.pdf"])
def test_sanitize_filename_rejects_invalid(value):
    with pytest.raises(engine.UploadError):
        engine.sanitize_filename(value)


@pytest.mark.parametrize(
    "filename, override, expected",
    [
        ("a.pdf", None, "application/pdf"),
        ("a.md", None, "text/markdown"),
        ("a.markdown", None, "text/markdown"),
        ("a.txt", None, "text/plain"),
        (
            "a.docx",
            None,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("archive.zip", None, "application/zip"),
        ("file.custom", None, "application/octet-stream"),
        ("a.pdf", " APPLICATION/X-PDF ", "application/x-pdf"),
    ],
)
def test_detect_mime_type(filename, override, expected):
    assert engine.detect_mime_type(filename, override) == expected


def test_prepare_file_calculates_hash_and_size():
    prepared = engine.prepare_file(
        engine.UploadFileData("a.pdf", b"abc"),
        max_file_bytes=10,
    )
    assert prepared.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert prepared.size == 3
    assert prepared.suffix == ".pdf"


@pytest.mark.parametrize(
    "filename,data,expected_suffix,expected_mime",
    [
        ("document.txt", b"testo", ".txt", "text/plain"),
        (
            "document.docx",
            b"PK\x03\x04test",
            ".docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("archive.zip", b"PK\x03\x04archive", ".zip", "application/zip"),
        ("image.png", b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
        (
            "file.custom",
            b"custom content",
            ".custom",
            "application/octet-stream",
        ),
    ],
)
def test_prepare_file_accepts_arbitrary_extensions(
    filename, data, expected_suffix, expected_mime
):
    prepared = engine.prepare_file(
        engine.UploadFileData(filename, data),
        max_file_bytes=1024,
    )
    assert prepared.filename == filename
    assert prepared.suffix == expected_suffix
    assert prepared.data == data
    assert prepared.size == len(data)
    assert prepared.mime_type == expected_mime
    assert prepared.sha256 == hashlib.sha256(data).hexdigest()


def test_prepare_file_accepts_file_without_extension():
    prepared = engine.prepare_file(
        engine.UploadFileData("LICENSE", b"license content"),
        max_file_bytes=100,
    )
    assert prepared.filename == "LICENSE"
    assert prepared.suffix == ""
    assert prepared.mime_type == "application/octet-stream"


@pytest.mark.parametrize(
    "filename,mime_type,expected",
    [
        ("document.pdf", "application/pdf", "pdf"),
        (
            "document.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
        ("archive.zip", "application/zip", "zip"),
        ("file.custom", "application/octet-stream", "custom"),
        ("LICENSE", "application/octet-stream", "binary"),
        ("README", "text/plain", "plain"),
    ],
)
def test_detect_source_format(filename, mime_type, expected):
    assert engine.detect_source_format(filename, mime_type) == expected


@pytest.mark.parametrize(
    "file, limit, message",
    [
        (engine.UploadFileData("a.pdf", b""), 10, "vuoto"),
        (engine.UploadFileData("a.pdf", b"123"), 2, "troppo grande"),
    ],
)
def test_prepare_file_rejects_invalid(file, limit, message):
    with pytest.raises(engine.UploadError, match=message):
        engine.prepare_file(file, max_file_bytes=limit)


def test_database_config_reads_environment(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db")
    monkeypatch.setenv("PG_PORT", "5544")
    monkeypatch.setenv("SOURCE_PG_DB", "source")
    monkeypatch.setenv("PG_USER", "user")
    monkeypatch.setenv("PG_PASS", "secret")
    assert engine.database_config() == {
        "host": "db",
        "port": 5544,
        "dbname": "source",
        "user": "user",
        "password": "secret",
    }


class Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


def test_resolve_ontology_uses_explicit_code_without_sql():
    cur = Cursor([])
    code, label = engine.resolve_ontology_code(
        cur,
        ontology_code=" Risk__Score ",
        ontology_label=None,
        area=None,
        subarea=None,
    )
    assert code == "risk__score"
    assert label == "risk / score"
    assert cur.executed == []


def test_resolve_ontology_builds_code_from_area_subarea():
    cur = Cursor([{"ontology_code": "identify__risk_assessment"}])
    code, label = engine.resolve_ontology_code(
        cur,
        ontology_code=None,
        ontology_label=None,
        area="IDENTIFY",
        subarea="Risk Assessment",
    )
    assert code == "identify__risk_assessment"
    assert label == "IDENTIFY / Risk Assessment"
    assert "fn_build_ontology_code" in cur.executed[0][0]


def test_resolve_ontology_requires_input():
    with pytest.raises(engine.UploadError):
        engine.resolve_ontology_code(
            Cursor([]),
            ontology_code=None,
            ontology_label=None,
            area=None,
            subarea=None,
        )


def test_verify_database_contract_success():
    row = {
        "database_name": "assessment_gestio_tier",
        "session_user": "admin",
        "has_blob": True,
        "has_document": True,
        "has_context": True,
        "has_job": True,
    }
    assert engine.verify_database_contract(Cursor([row]))["database_name"] == "assessment_gestio_tier"


def test_verify_database_contract_fails_closed():
    row = {
        "database_name": "x",
        "session_user": "x",
        "has_blob": True,
        "has_document": False,
        "has_context": True,
        "has_job": True,
    }
    with pytest.raises(engine.UploadError, match="incompleto"):
        engine.verify_database_contract(Cursor([row]))


@pytest.mark.parametrize("value", [None, 0, -1, "bad"])
def test_positive_int_validation(value):
    with pytest.raises(engine.UploadError):
        engine._normalize_positive_int(value, "field")


def test_positive_int_accepts_numeric_string():
    assert engine._normalize_positive_int("12", "field") == 12


def test_detect_mime_type_unknown_extension_uses_mimetypes():
    assert engine.detect_mime_type("data.json", None) == "application/json"


def test_set_tenant_context_executes_worker_function():
    cur = Cursor([])
    engine.set_tenant_context(cur, 9999, 123)
    query, params = cur.executed[0]
    assert "fn_set_tenant_context" in query
    assert params[0:2] == (9999, 123)
    assert len(params[2]) == 36


def test_clear_tenant_context_executes_function():
    cur = Cursor([])
    engine.clear_tenant_context(cur)
    assert "fn_clear_tenant_context" in cur.executed[0][0]


def test_clear_tenant_context_swallows_error():
    class BrokenCursor:
        def execute(self, *args, **kwargs):
            raise RuntimeError("ignored")

    engine.clear_tenant_context(BrokenCursor())


def test_resolve_ontology_rejects_empty_database_result():
    with pytest.raises(engine.UploadError, match="non ha restituito"):
        engine.resolve_ontology_code(
            Cursor([None]),
            ontology_code=None,
            ontology_label=None,
            area="A",
            subarea="B",
        )


def test_get_or_create_manual_ontology_returns_existing():
    cur = Cursor([{"ontology_id": 7}])
    value = engine.get_or_create_manual_ontology(
        cur,
        tier="C",
        scope="ACCOUNT",
        organization_id=9,
        user_id=3,
        ontology_code="x",
        ontology_label="X",
        area="A",
        subarea="B",
    )
    assert value == 7
    assert len(cur.executed) == 1


def test_get_or_create_manual_ontology_inserts_new():
    cur = Cursor([None, {"ontology_id": 8}])
    value = engine.get_or_create_manual_ontology(
        cur,
        tier="C",
        scope="ACCOUNT",
        organization_id=9,
        user_id=3,
        ontology_code="x",
        ontology_label="X",
        area="A",
        subarea="B",
    )
    assert value == 8
    assert "INSERT INTO rag_ingestion.rag_ontology" in cur.executed[1][0]


def test_get_or_create_blob_returns_existing():
    cur = Cursor([{"file_blob_id": "blob-1"}])
    value = engine.get_or_create_blob(
        cur,
        Binary=lambda value: value,
        Json=lambda value: value,
        scope="ACCOUNT",
        organization_id=9,
        user_id=3,
        data=b"x",
        sha256="a" * 64,
        size=1,
        mime_type="application/pdf",
        original_filename="x.pdf",
    )
    assert value == "blob-1"
    assert len(cur.executed) == 1


def test_get_or_create_blob_inserts_new():
    cur = Cursor([None, {"file_blob_id": "blob-2"}])
    value = engine.get_or_create_blob(
        cur,
        Binary=lambda value: ("binary", value),
        Json=lambda value: ("json", value),
        scope="ACCOUNT",
        organization_id=9,
        user_id=3,
        data=b"x",
        sha256="a" * 64,
        size=1,
        mime_type="application/pdf",
        original_filename="x.pdf",
    )
    assert value == "blob-2"
    query, params = cur.executed[1]
    assert "INSERT INTO rag_ingestion.rag_file_blob" in query
    assert params[2] == ("binary", b"x")
    assert params[6][0] == "json"
    assert params[6][1]["source"] == "carica_documento_bytea_rag.py"


def test_get_or_create_document_returns_existing():
    cur = Cursor([{"document_id": "doc-1", "processing_status": "DONE"}])
    value = engine.get_or_create_document(
        cur,
        file_blob_id="blob",
        tier="C",
        scope="ACCOUNT",
        organization_id=9,
        classification="internal",
        source_format="pdf",
        pipeline_version="v1",
        corpus_version="v1",
        embedding_model=None,
    )
    assert value == ("doc-1", False, "DONE")


def test_get_or_create_document_inserts_new():
    cur = Cursor([None, {"document_id": "doc-2", "processing_status": "PENDING"}])
    value = engine.get_or_create_document(
        cur,
        file_blob_id="blob",
        tier="C",
        scope="ACCOUNT",
        organization_id=9,
        classification="internal",
        source_format="pdf",
        pipeline_version="v1",
        corpus_version="v1",
        embedding_model="bge-m3",
    )
    assert value == ("doc-2", True, "PENDING")
    assert "INSERT INTO rag_ingestion.rag_document" in cur.executed[1][0]


def test_get_or_create_context_returns_existing():
    cur = Cursor([{"document_context_id": "ctx-1"}])
    assert engine.get_or_create_corpus_context(
        cur, document_id="doc", ontology_id=2, organization_id=9
    ) == ("ctx-1", False)


def test_get_or_create_context_inserts_new():
    cur = Cursor([None, {"document_context_id": "ctx-2"}])
    assert engine.get_or_create_corpus_context(
        cur, document_id="doc", ontology_id=2, organization_id=9
    ) == ("ctx-2", True)
    assert "INSERT INTO rag_ingestion.rag_document_context" in cur.executed[1][0]


def test_connect_uses_database_config(monkeypatch):
    class FakePg:
        def __init__(self):
            self.kwargs = None

        def connect(self, **kwargs):
            self.kwargs = kwargs
            return "connection"

    pg = FakePg()
    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (pg, None, None, None))
    monkeypatch.setattr(engine, "database_config", lambda: {"host": "db"})
    assert engine.connect() == "connection"
    assert pg.kwargs == {"host": "db"}


def test_connect_wraps_driver_error(monkeypatch):
    class FakePg:
        @staticmethod
        def connect(**kwargs):
            raise RuntimeError("secret")

    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (FakePg(), None, None, None))
    with pytest.raises(engine.DatabaseOperationError, match="Database A"):
        engine.connect()


def test_healthcheck_shallow_success(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (None, None, None, None))
    result = engine.healthcheck(deep=False)
    assert result["postgres_source"]["ready"] is True


def test_healthcheck_shallow_failure(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    result = engine.healthcheck(deep=False)
    assert result["postgres_source"]["ready"] is False
    assert "missing" in result["postgres_source"]["detail"]


def test_healthcheck_deep_success(monkeypatch):
    class HealthCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class HealthConn:
        def __init__(self):
            self.rollbacks = 0
            self.closed = 0

        def cursor(self, **kwargs):
            return HealthCursor()

        def rollback(self):
            self.rollbacks += 1

        def close(self):
            self.closed += 1

    conn = HealthConn()
    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (None, None, None, object()))
    monkeypatch.setattr(engine, "connect", lambda: conn)
    monkeypatch.setattr(
        engine,
        "verify_database_contract",
        lambda cur: {"database_name": "source", "session_user": "admin"},
    )
    result = engine.healthcheck(deep=True)
    assert result["postgres_source"]["ready"] is True
    assert "db=source" in result["postgres_source"]["detail"]
    assert conn.rollbacks == 1
    assert conn.closed == 1


def test_healthcheck_deep_failure_closes_connection(monkeypatch):
    class HealthConn:
        def __init__(self):
            self.closed = 0

        def cursor(self, **kwargs):
            raise RuntimeError("offline")

        def close(self):
            self.closed += 1

    conn = HealthConn()
    monkeypatch.setattr(engine, "_load_psycopg2", lambda: (None, None, None, object()))
    monkeypatch.setattr(engine, "connect", lambda: conn)
    result = engine.healthcheck(deep=True)
    assert result["postgres_source"]["ready"] is False
    assert "offline" in result["postgres_source"]["detail"]
    assert conn.closed == 1
