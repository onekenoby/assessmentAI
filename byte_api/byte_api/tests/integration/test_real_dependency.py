from __future__ import annotations

import os

import pytest

import byte_engine

pytestmark = pytest.mark.integration


def test_real_database_a_contract():
    if os.getenv("BYTE_API_RUN_INTEGRATION") != "1":
        pytest.skip("Impostare BYTE_API_RUN_INTEGRATION=1")

    result = byte_engine.healthcheck(deep=True)
    postgres = result["postgres_source"]
    assert postgres["ready"] is True, postgres["detail"]
    assert "rag_ingestion" in postgres["detail"]
