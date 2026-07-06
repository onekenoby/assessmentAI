from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from core.audit import AuditIdentityError, AuditService, AuditSink, create_query_audit
from core.config import settings
from core.models import RagAnswerMode, RagExecutionMode, RagIntent, RetrievalDebug


class FakeCursor:
    def __init__(self, *, existing=None):
        self.existing = existing
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self.existing


class FakeConnection:
    def __init__(self, *, existing=None):
        self.cursor_obj = FakeCursor(existing=existing)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _factory(conn):
    @contextmanager
    def connection_factory(*, context=None):
        yield conn
    return connection_factory


def _audit(tenant_context, source_c):
    return create_query_audit(
        query="query segreta",
        sources=(source_c,),
        intent=RagIntent.AUDIT,
        answer_mode=RagAnswerMode.AUDIT,
        execution_mode=RagExecutionMode.RAG_GENERATION,
        retrieval=RetrievalDebug(query="query segreta"),
        filters={"password": "non salvare", "target_document": "evidence.pdf"},
        prompt_sha256="a" * 64,
        context_chars=100,
        deterministic=False,
        context=tenant_context,
        config=settings,
    )


def test_audit_writes_redacted_jsonl_and_postgres(tmp_path, tenant_context, source_c):
    conn = FakeConnection(existing=None)
    config = settings.model_copy(update={
        "audit_enabled": True,
        "pg_enrich_enabled": True,
        "audit_log_path": tmp_path / "rag_audit.jsonl",
    })
    service = AuditService(config=config, postgres_connection_factory=_factory(conn))
    audit = _audit(tenant_context, source_c)

    result = service.persist_query_audit(audit, context=tenant_context)

    assert result.success is True
    assert {outcome.sink for outcome in result.outcomes} == {
        AuditSink.QUERY_JSONL,
        AuditSink.POSTGRES,
    }
    line = json.loads((tmp_path / "rag_audit.jsonl").read_text(encoding="utf-8"))
    assert line["query"] in {"", "[REDACTED]"}
    assert "query segreta" not in json.dumps(line, ensure_ascii=False)
    assert line["filters"]["password"] == "[REDACTED]"
    assert conn.commits == 1
    assert any("INSERT INTO public.rag_query_audit" in sql for sql, _ in conn.cursor_obj.calls)


def test_audit_postgres_duplicate_is_not_failure(tmp_path, tenant_context, source_c):
    conn = FakeConnection(existing=(123,))
    config = settings.model_copy(update={
        "audit_enabled": True,
        "pg_enrich_enabled": True,
        "audit_log_path": tmp_path / "rag_audit.jsonl",
    })
    service = AuditService(config=config, postgres_connection_factory=_factory(conn))

    result = service.persist_query_audit(_audit(tenant_context, source_c), context=tenant_context)

    pg = next(item for item in result.outcomes if item.sink == AuditSink.POSTGRES)
    assert pg.success is True
    assert pg.duplicate is True
    assert conn.rollbacks >= 1


def test_audit_disabled_skips_all_sinks(tmp_path, tenant_context, source_c):
    config = settings.model_copy(update={
        "audit_enabled": False,
        "audit_log_path": tmp_path / "never.jsonl",
    })
    service = AuditService(config=config, postgres_connection_factory=_factory(FakeConnection()))

    result = service.persist_query_audit(_audit(tenant_context, source_c), context=tenant_context)

    assert result.skipped is True
    assert all(item.skipped for item in result.outcomes)
    assert not (tmp_path / "never.jsonl").exists()


def test_audit_rejects_identity_mismatch(tmp_path, tenant_context, other_tenant_context, source_c):
    config = settings.model_copy(update={
        "audit_enabled": True,
        "pg_enrich_enabled": False,
        "audit_log_path": tmp_path / "rag_audit.jsonl",
    })
    service = AuditService(config=config, postgres_connection_factory=_factory(FakeConnection()))

    with pytest.raises(AuditIdentityError):
        service.persist_query_audit(
            _audit(tenant_context, source_c),
            context=other_tenant_context,
        )


@pytest.mark.asyncio
async def test_audit_async_wrapper(tmp_path, tenant_context, source_c):
    config = settings.model_copy(update={
        "audit_enabled": True,
        "pg_enrich_enabled": False,
        "audit_log_path": tmp_path / "rag_audit.jsonl",
    })
    service = AuditService(config=config, postgres_connection_factory=_factory(FakeConnection()))

    result = await service.persist_query_audit_async(
        _audit(tenant_context, source_c),
        context=tenant_context,
    )

    assert result.success is True
    assert (tmp_path / "rag_audit.jsonl").exists()
