from __future__ import annotations

import ast
from pathlib import Path


ENGINE_PATH = Path(__file__).resolve().parents[1] / "ingestion_engine.py"
ENGINE_SOURCE = ENGINE_PATH.read_text(encoding="utf-8")
ENGINE_TREE = ast.parse(ENGINE_SOURCE)


def _top_level_functions() -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ENGINE_TREE.body
        if isinstance(node, ast.FunctionDef)
    }


def test_engine_exposes_api_runtime_contract():
    functions = _top_level_functions()
    expected = {
        "initialize_ingestion_runtime",
        "runtime_healthcheck",
        "run_pending_jobs",
        "process_next_db_job",
        "shutdown_ingestion_runtime",
        "claim_next_db_job",
        "get_claimed_db_job_payload",
        "complete_db_job",
        "fail_db_job",
    }
    assert expected.issubset(functions)


def test_run_pending_jobs_has_explicit_max_jobs_parameter():
    node = _top_level_functions()["run_pending_jobs"]
    assert [arg.arg for arg in node.args.args] == ["max_jobs"]
    assert len(node.args.defaults) == 1
    assert isinstance(node.args.defaults[0], ast.Constant)
    assert node.args.defaults[0].value == 1


def test_worker_contract_uses_only_protected_rag_ingestion_functions():
    required_sql_functions = {
        "fn_claim_next_ingestion_job",
        "fn_get_claimed_job_payload",
        "fn_heartbeat_ingestion_job",
        "fn_complete_ingestion_job",
        "fn_fail_ingestion_job",
    }
    for function_name in required_sql_functions:
        assert f"rag_ingestion.{function_name}" in ENGINE_SOURCE



def test_global_cross_process_lock_is_present_in_run_loop():
    node = _top_level_functions()["run_pending_jobs"]
    source = ast.get_source_segment(ENGINE_SOURCE, node) or ""

    assert "with ingestion_global_lock()" in source
    assert "process_next_db_job()" in source


def test_run_pending_jobs_supports_producer_consumer():
    node = _top_level_functions()["run_pending_jobs"]
    source = ast.get_source_segment(ENGINE_SOURCE, node) or ""

    assert 'os.getenv("USE_PRODUCER_CONSUMER", "1")' in source
    assert 'os.getenv("DOC_QUEUE_MAXSIZE", "1")' in source

    assert "queue.Queue" in source
    assert "threading.Thread" in source
    assert "consumer_worker" in source

    assert "claim_next_db_job()" in source
    assert "prepare_claimed_job_item" in source
    assert "consume_claimed_job_item" in source

    assert "doc_queue.put(" in source
    assert "doc_queue.put(None)" in source
    assert "doc_queue.join()" in source
    assert "consumer_thread.join()" in source


def test_producer_consumer_keeps_single_consumer():
    node = _top_level_functions()["run_pending_jobs"]
    source = ast.get_source_segment(ENGINE_SOURCE, node) or ""

    assert 'name="ingestion-document-consumer"' in source
    assert "max_workers" not in source


def test_engine_does_not_define_http_upload_routes():
    lowered = ENGINE_SOURCE.lower()
    assert "uploadfile" not in lowered
    assert "@app.post" not in lowered
    assert "@router.post" not in lowered


def test_tenant_context_is_derived_from_job_metadata():
    node = _top_level_functions()["set_output_db_tenant_context"]
    source = ast.get_source_segment(ENGINE_SOURCE, node) or ""
    assert "organization_id = meta.get(\"organization_id\")" in source
    assert "tier = str(meta.get(\"tier\")" in source
    assert "scope = str(meta.get(\"scope\")" in source
    assert "expected_tenant_key" in source
