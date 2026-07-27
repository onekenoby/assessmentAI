from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

import byte_engine as engine


class FakeCursor:
    def __init__(self, *, one=None, all_rows=None):
        self.one = list(one or [])
        self.all_rows = list(all_rows or [])
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.one.pop(0) if self.one else None

    def fetchall(self):
        return list(self.all_rows)


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self, **kwargs):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


def fake_driver():
    return object(), lambda value: ("BINARY", value), lambda value: ("JSON", value), object()


def corpus_payload(**overrides):
    values = {
        "file": engine.UploadFileData("doc.pdf", b"%PDF"),
        "tier": "C",
        "organization_id": 9999,
        "user_id": 123,
        "area": "IDENTIFY",
        "subarea": "Risk Assessment",
    }
    values.update(overrides)
    return engine.CorpusUpload(**values)


def evidence_payload(**overrides):
    values = {
        "file": engine.UploadFileData("evidence.pdf", b"%PDF"),
        "organization_id": 9999,
        "user_id": 123,
        "assessment_id": 10,
        "response_id": 20,
    }
    values.update(overrides)
    return engine.EvidenceUpload(**values)


def patch_corpus_helpers(monkeypatch, *, document_created=True, processing_status="PENDING"):
    calls = []
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: {"ok": True})
    monkeypatch.setattr(engine, "set_tenant_context", lambda cur, org, user: calls.append(("set", org, user)))
    monkeypatch.setattr(engine, "clear_tenant_context", lambda cur: calls.append(("clear",)))
    monkeypatch.setattr(
        engine,
        "resolve_ontology_code",
        lambda cur, **kwargs: ("identify__risk_assessment", "IDENTIFY / Risk Assessment"),
    )
    monkeypatch.setattr(engine, "get_or_create_manual_ontology", lambda cur, **kwargs: 2)
    monkeypatch.setattr(engine, "get_or_create_blob", lambda cur, **kwargs: str(uuid4()))
    monkeypatch.setattr(
        engine,
        "get_or_create_document",
        lambda cur, **kwargs: (str(uuid4()), document_created, processing_status),
    )
    monkeypatch.setattr(
        engine,
        "get_or_create_corpus_context",
        lambda cur, **kwargs: (str(uuid4()), True),
    )
    return calls


def test_upload_corpus_account_commits_and_returns_job(monkeypatch):
    calls = patch_corpus_helpers(monkeypatch)
    job = {
        "job_id": str(uuid4()),
        "job_type": "CONTENT_INGESTION",
        "status": "PENDING",
        "priority": 100,
        "available_at": datetime.now(UTC),
    }
    cursor = FakeCursor(all_rows=[job])
    conn = FakeConnection(cursor)
    monkeypatch.setattr(engine, "connect", lambda: conn)

    result = engine.upload_corpus(corpus_payload(), max_file_bytes=100)

    assert result["scope"] == "ACCOUNT"
    assert result["organization_id"] == 9999
    assert result["jobs"] == [job]
    assert calls == [("set", 9999, 123), ("clear",)]
    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert conn.closed == 1


def test_upload_corpus_global_does_not_set_tenant_context(monkeypatch):
    calls = patch_corpus_helpers(monkeypatch)
    conn = FakeConnection(FakeCursor(all_rows=[]))
    monkeypatch.setattr(engine, "connect", lambda: conn)

    result = engine.upload_corpus(
        corpus_payload(tier="A", organization_id=None, user_id=None),
        max_file_bytes=100,
    )

    assert result["scope"] == "GLOBAL"
    assert result["organization_id"] is None
    assert calls == []


def test_existing_pending_document_recreates_missing_job(monkeypatch):
    patch_corpus_helpers(monkeypatch, document_created=False, processing_status="PENDING")
    cursor = FakeCursor(all_rows=[])
    conn = FakeConnection(cursor)
    monkeypatch.setattr(engine, "connect", lambda: conn)

    engine.upload_corpus(corpus_payload(), max_file_bytes=100)
    assert any("INSERT INTO rag_ingestion.rag_ingestion_job" in query for query, _ in cursor.executed)


def test_existing_done_document_does_not_recreate_job(monkeypatch):
    patch_corpus_helpers(monkeypatch, document_created=False, processing_status="DONE")
    cursor = FakeCursor(all_rows=[])
    conn = FakeConnection(cursor)
    monkeypatch.setattr(engine, "connect", lambda: conn)

    engine.upload_corpus(corpus_payload(), max_file_bytes=100)
    assert not any("INSERT INTO rag_ingestion.rag_ingestion_job" in query for query, _ in cursor.executed)


@pytest.mark.parametrize(
    "payload,message",
    [
        (corpus_payload(tier="Z"), "A, B oppure C"),
        (corpus_payload(tier="A"), "GLOBAL"),
        (corpus_payload(organization_id=None), "organization_id"),
        (corpus_payload(user_id=None), "user_id"),
        (corpus_payload(classification="secret"), "classification"),
        (corpus_payload(pipeline_version=""), "pipeline_version"),
        (corpus_payload(corpus_version=""), "corpus_version"),
    ],
)
def test_upload_corpus_validates_before_database(monkeypatch, payload, message):
    monkeypatch.setattr(engine, "connect", lambda: pytest.fail("connect non deve essere chiamata"))
    with pytest.raises(engine.UploadError, match=message):
        engine.upload_corpus(payload, max_file_bytes=100)


def test_upload_corpus_rolls_back_and_wraps_database_failure(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: (_ for _ in ()).throw(RuntimeError("db")))
    conn = FakeConnection(FakeCursor())
    monkeypatch.setattr(engine, "connect", lambda: conn)

    with pytest.raises(engine.DatabaseOperationError):
        engine.upload_corpus(corpus_payload(), max_file_bytes=100)
    assert conn.rollbacks == 1
    assert conn.closed == 1


def test_upload_corpus_preserves_upload_error(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: (_ for _ in ()).throw(engine.UploadError("schema")))
    conn = FakeConnection(FakeCursor())
    monkeypatch.setattr(engine, "connect", lambda: conn)

    with pytest.raises(engine.UploadError, match="schema"):
        engine.upload_corpus(corpus_payload(), max_file_bytes=100)
    assert conn.rollbacks == 1


def test_upload_evidence_calls_official_function_and_commits(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: {"ok": True})
    contexts = []
    monkeypatch.setattr(engine, "set_tenant_context", lambda cur, org, user: contexts.append(("set", org, user)))
    monkeypatch.setattr(engine, "clear_tenant_context", lambda cur: contexts.append(("clear",)))
    document_id = str(uuid4())
    result_row = {"document_id": document_id, "file_blob_id": str(uuid4())}
    job = {
        "job_id": str(uuid4()),
        "job_type": "CONTENT_INGESTION",
        "status": "PENDING",
        "priority": 100,
        "available_at": datetime.now(UTC),
    }
    cursor = FakeCursor(one=[result_row], all_rows=[job])
    conn = FakeConnection(cursor)
    monkeypatch.setattr(engine, "connect", lambda: conn)

    result = engine.upload_evidence(evidence_payload(encryption_required=False), max_file_bytes=100)

    official_query, params = next(
        (query, params)
        for query, params in cursor.executed
        if "fn_upload_response_evidence" in query
    )
    assert params[0:3] == (10, 20, 123)
    assert params[-1] is False
    assert result["document_id"] == document_id
    assert result["jobs"] == [job]
    assert contexts == [("set", 9999, 123), ("clear",)]
    assert conn.commits == 1
    assert conn.closed == 1


def test_upload_evidence_empty_function_result_rolls_back(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: {"ok": True})
    monkeypatch.setattr(engine, "set_tenant_context", lambda *args: None)
    conn = FakeConnection(FakeCursor(one=[None]))
    monkeypatch.setattr(engine, "connect", lambda: conn)

    with pytest.raises(engine.UploadError, match="non ha restituito"):
        engine.upload_evidence(evidence_payload(), max_file_bytes=100)
    assert conn.rollbacks == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("organization_id", 0),
        ("user_id", -1),
        ("assessment_id", 0),
        ("response_id", 0),
    ],
)
def test_upload_evidence_validates_positive_ids(monkeypatch, field, value):
    monkeypatch.setattr(engine, "connect", lambda: pytest.fail("connect non deve essere chiamata"))
    with pytest.raises(engine.UploadError):
        engine.upload_evidence(evidence_payload(**{field: value}), max_file_bytes=100)


def test_upload_evidence_wraps_database_failure(monkeypatch):
    monkeypatch.setattr(engine, "_load_psycopg2", fake_driver)
    monkeypatch.setattr(engine, "verify_database_contract", lambda cur: (_ for _ in ()).throw(RuntimeError("db")))
    conn = FakeConnection(FakeCursor())
    monkeypatch.setattr(engine, "connect", lambda: conn)

    with pytest.raises(engine.DatabaseOperationError):
        engine.upload_evidence(evidence_payload(), max_file_bytes=100)
    assert conn.rollbacks == 1
    assert conn.closed == 1
