from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.config import settings
from core.resources import (
    ResourceInitializationError,
    ResourceManager,
    ResourceState,
)


class Closable:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeNeo4j(Closable):
    def __init__(self, *, fail=False):
        super().__init__()
        self.fail = fail
        self.probes = 0

    def verify_connectivity(self):
        self.probes += 1
        if self.fail:
            raise RuntimeError("neo4j down")


class FakePool:
    def __init__(self, conn=None):
        self.conn = conn
        self.closed = 0
        self.returned = []

    def getconn(self):
        return self.conn

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))

    def closeall(self):
        self.closed += 1


class FakeCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return ("app", False, False)

    def fetchall(self):
        return []


class FakeConnection:
    closed = False

    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeSession(Closable):
    pass


class FakeManager(ResourceManager):
    def __init__(self, config, *, neo4j_fail=False):
        super().__init__(config)
        self.neo4j_fail = neo4j_fail
        self.created = {}

    def _create_ollama_session(self):
        self.created["ollama_session"] = FakeSession()
        return self.created["ollama_session"]

    def _create_openai_client(self):
        self.created["openai"] = Closable()
        return self.created["openai"]

    def _create_qdrant_client(self):
        self.created["qdrant"] = Closable()
        return self.created["qdrant"]

    def _create_neo4j_driver(self):
        self.created["neo4j"] = FakeNeo4j(fail=self.neo4j_fail)
        return self.created["neo4j"]

    def _create_postgres_pool(self):
        self.created["pg"] = FakePool(FakeConnection())
        return self.created["pg"]

    def _create_embedder(self):
        self.created["embedder"] = object()
        return self.created["embedder"]

    def _create_reranker(self):
        self.created["reranker"] = object()
        return self.created["reranker"]

    def _verify_ollama(self, session, *, require_model):
        return None

    def _verify_qdrant(self, client, *, require_collection):
        return None

    def _ensure_postgres_rag_security(self, pool):
        return None

    def _verify_postgres(self, pool):
        return None


def test_resource_manager_initializes_atomically_ready():
    config = settings.model_copy(update={"neo4j_enabled": True, "pg_enrich_enabled": True})
    manager = FakeManager(config)

    manager.initialize(strict=True)

    assert manager.state == ResourceState.READY
    assert manager.is_ready is True
    assert manager.get_embedder() is manager.created["embedder"]
    assert manager.get_qdrant_client() is manager.created["qdrant"]


def test_resource_manager_degrades_when_optional_neo4j_fails():
    config = settings.model_copy(update={"neo4j_enabled": True, "pg_enrich_enabled": False})
    manager = FakeManager(config, neo4j_fail=True)

    manager.initialize(strict=False)

    assert manager.state == ResourceState.DEGRADED
    assert manager.is_ready is True
    assert manager.get_neo4j_driver(required=False) is None


def test_resource_manager_strict_neo4j_failure_fails_startup():
    config = settings.model_copy(update={"neo4j_enabled": True, "pg_enrich_enabled": False})
    manager = FakeManager(config, neo4j_fail=True)

    with pytest.raises(ResourceInitializationError):
        manager.initialize(strict=True)

    assert manager.state == ResourceState.FAILED


def test_postgres_connection_sets_and_resets_tenant_gucs(tenant_context):
    config = settings.model_copy(update={"neo4j_enabled": False, "pg_enrich_enabled": True})
    manager = FakeManager(config)
    manager.initialize(strict=True)
    conn = manager.created["pg"].conn

    with manager.postgres_connection(context=tenant_context) as yielded:
        assert yielded is conn

    sql_text = "\n".join(call[0] for call in conn.cursor_obj.calls)
    assert "set_config('app.current_customer_account_id'" in sql_text
    assert "RESET app.current_customer_account_id" in sql_text
    assert "RESET app.current_request_id" in sql_text
    assert conn.commits >= 2
    assert conn.rollbacks >= 1
    assert manager.created["pg"].returned[-1][1] is False


def test_health_snapshot_deep_uses_simulated_probes():
    config = settings.model_copy(update={"neo4j_enabled": True, "pg_enrich_enabled": True})
    manager = FakeManager(config)
    manager.initialize(strict=True)

    snapshot = manager.health_snapshot(deep=True)

    assert snapshot.ready is True
    assert {item.name for item in snapshot.dependencies} == {
        "embedder", "reranker", "ollama", "qdrant", "neo4j", "postgresql"
    }
    assert all(item.ready for item in snapshot.dependencies if item.enabled)


def test_close_is_idempotent():
    config = settings.model_copy(update={"neo4j_enabled": True, "pg_enrich_enabled": True})
    manager = FakeManager(config)
    manager.initialize(strict=True)

    manager.close()
    manager.close()

    assert manager.state == ResourceState.CLOSED
    assert manager.created["pg"].closed == 1
