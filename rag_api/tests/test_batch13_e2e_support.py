from __future__ import annotations

import json
from pathlib import Path

from e2e_report import parse_junit_xml, safe_environment_snapshot, write_batch13_report
import verify_batch13
import verify_real_services


def test_real_service_runner_resolves_project_root() -> None:
    assert verify_real_services.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (verify_real_services.PROJECT_ROOT / "main.py").exists()
    assert "api" in verify_real_services.SERVICE_FILES
    assert "isolation" in verify_real_services.SERVICE_FILES


def test_batch13_manifest_contains_unit_and_real_closure_tests() -> None:
    assert "tests/test_rag_reflex_alignment_batch13_4.py" in verify_batch13.UNIT_FILES
    assert "tests/test_rag_reflex_alignment_batch13_3.py" in verify_batch13.UNIT_FILES
    assert "tests/test_rag_reflex_alignment_batch12.py" in verify_batch13.UNIT_FILES
    assert "tests/test_batch13_e2e_support.py" in verify_batch13.UNIT_FILES
    assert "tests/integration/test_rag_api_e2e_real.py" in verify_batch13.REAL_FILES
    assert "tests/integration/test_organization_isolation_real.py" in verify_batch13.REAL_FILES


def test_parse_junit_xml_sums_multiple_suites(tmp_path: Path) -> None:
    xml = tmp_path / "junit.xml"
    xml.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
        <testsuites>
          <testsuite name='unit' tests='4' failures='1' errors='0' skipped='1' time='1.25'/>
          <testsuite name='real' tests='3' failures='0' errors='1' skipped='0' time='2.75'/>
        </testsuites>
        """,
        encoding="utf-8",
    )
    summary = parse_junit_xml(xml)
    assert summary.tests == 7
    assert summary.passed == 4
    assert summary.failures == 1
    assert summary.errors == 1
    assert summary.skipped == 1
    assert summary.time_seconds == 4.0
    assert summary.successful is False


def test_safe_environment_snapshot_never_records_secrets() -> None:
    snapshot = safe_environment_snapshot(
        {
            "PG_HOST": "127.0.0.1",
            "PG_USER": "rag_user",
            "PG_PASS": "secret-password",
            "NEO4J_PASSWORD": "secret-neo4j",
            "OLLAMA_API_KEY": "secret-token",
            "RAG_API_BASE_URL": "http://127.0.0.1:8000",
        }
    )
    assert snapshot["PG_HOST"] == "127.0.0.1"
    assert snapshot["PG_USER"] == "rag_user"
    assert snapshot["RAG_API_BASE_URL"] == "http://127.0.0.1:8000"
    assert "PG_PASS" not in snapshot
    assert "NEO4J_PASSWORD" not in snapshot
    assert "OLLAMA_API_KEY" not in snapshot
    assert "secret" not in json.dumps(snapshot)


def test_batch13_report_is_written_and_sanitized(tmp_path: Path) -> None:
    junit = tmp_path / "result.xml"
    junit.write_text(
        "<testsuite tests='5' failures='0' errors='0' skipped='1' time='3.5'/>",
        encoding="utf-8",
    )
    json_path, md_path = write_batch13_report(
        report_dir=tmp_path / "report",
        junit_xml=junit,
        command=["python", "-m", "pytest"],
        exit_code=0,
        base_url="http://127.0.0.1:8000",
        started_api=True,
        capacity_stress=False,
        second_organization_required=False,
        environ={"PG_HOST": "db", "PG_PASS": "do-not-record"},
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["status"] == "PASS"
    assert payload["junit"]["passed"] == 4
    assert payload["safe_environment"] == {"PG_HOST": "db"}
    assert "do-not-record" not in json_path.read_text(encoding="utf-8")
    assert "Overall result:** `PASS`" in markdown
    assert "Cross-organization leakage checks" in markdown


def test_batch13_start_mode_refuses_to_reuse_existing_api() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="già occupata"):
        verify_batch13._resolve_api_launch_plan(
            start_requested=True,
            api_live=True,
            base_url="http://127.0.0.1:8000",
        )


def test_batch13_launch_plan_is_explicit() -> None:
    assert verify_batch13._resolve_api_launch_plan(
        start_requested=True,
        api_live=False,
        base_url="http://127.0.0.1:8013",
    ) == "start"
    assert verify_batch13._resolve_api_launch_plan(
        start_requested=False,
        api_live=True,
        base_url="http://127.0.0.1:8000",
    ) == "reuse"
