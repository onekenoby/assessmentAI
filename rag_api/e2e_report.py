"""Utilities for producing the Batch 13 end-to-end validation report.

The module is deliberately dependency-free so the report can still be written
when one of the optional RAG clients is unavailable.  It parses the JUnit XML
emitted by pytest and records only non-sensitive configuration metadata.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class JUnitSummary:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time_seconds: float = 0.0

    @property
    def passed(self) -> int:
        return max(0, self.tests - self.failures - self.errors - self.skipped)

    @property
    def successful(self) -> bool:
        return self.tests > 0 and self.failures == 0 and self.errors == 0


SAFE_ENV_KEYS = (
    "POC_MODE",
    "ORGANIZATION_ID",
    "CORPUS_VERSION",
    "PG_HOST",
    "PG_PORT",
    "PG_DB",
    "PG_USER",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_ENABLED",
    "OLLAMA_NATIVE_CHAT_URL",
    "LLM_MODEL_NAME",
    "LLM_TIMEOUT_S",
    "LLM_NUM_PREDICT",
    "LLM_MAX_ATTEMPTS",
    "EMBEDDING_MODEL_NAME",
    "RERANKER_MODEL_NAME",
    "RAG_MAX_CONCURRENT_QUERIES",
    "RAG_MAX_QUEUED_QUERIES",
    "RAG_QUERY_QUEUE_TIMEOUT_S",
    "RAG_API_BASE_URL",
)


def parse_junit_xml(path: str | Path) -> JUnitSummary:
    xml_path = Path(path)
    if not xml_path.exists() or xml_path.stat().st_size == 0:
        return JUnitSummary()

    root = ElementTree.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    def _int_attr(node: ElementTree.Element, name: str) -> int:
        try:
            return int(float(node.attrib.get(name, "0") or 0))
        except (TypeError, ValueError):
            return 0

    def _float_attr(node: ElementTree.Element, name: str) -> float:
        try:
            return float(node.attrib.get(name, "0") or 0)
        except (TypeError, ValueError):
            return 0.0

    return JUnitSummary(
        tests=sum(_int_attr(node, "tests") for node in suites),
        failures=sum(_int_attr(node, "failures") for node in suites),
        errors=sum(_int_attr(node, "errors") for node in suites),
        skipped=sum(_int_attr(node, "skipped") for node in suites),
        time_seconds=sum(_float_attr(node, "time") for node in suites),
    )


def safe_environment_snapshot(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = environ if environ is not None else os.environ
    snapshot: dict[str, str] = {}
    for key in SAFE_ENV_KEYS:
        value = str(source.get(key, "")).strip()
        if value:
            snapshot[key] = value[:1000]
    return snapshot


def _alignment_matrix(*, capacity_stress: bool, second_organization_required: bool) -> list[dict[str, str]]:
    return [
        {
            "area": "PostgreSQL schema, RLS and organization visibility",
            "evidence": "real service integration tests",
            "status": "covered",
        },
        {
            "area": "Qdrant collection, vector query and payload filtering",
            "evidence": "real service integration tests",
            "status": "covered",
        },
        {
            "area": "Neo4j connectivity, graph invariants and relationship whitelist",
            "evidence": "real service integration tests",
            "status": "covered",
        },
        {
            "area": "Ollama configured model and native /api/chat response",
            "evidence": "real service integration tests",
            "status": "covered",
        },
        {
            "area": "Live API health, readiness, deterministic and RAG query contracts",
            "evidence": "HTTP end-to-end tests",
            "status": "covered",
        },
        {
            "area": "Document/page scope and public source provenance",
            "evidence": "HTTP end-to-end tests with a real PostgreSQL chunk",
            "status": "covered",
        },
        {
            "area": "Cross-organization leakage checks",
            "evidence": "real-store isolation tests",
            "status": "required" if second_organization_required else "covered when a second organization_id exists",
        },
        {
            "area": "Generation timeout/empty response fallback",
            "evidence": "Batch 10 deterministic fault simulation",
            "status": "covered",
        },
        {
            "area": "Bounded capacity and service_busy",
            "evidence": "Batch 12 unit tests" + (" plus live stress" if capacity_stress else ""),
            "status": "covered",
        },
    ]


def write_batch13_report(
    *,
    report_dir: str | Path,
    junit_xml: str | Path,
    command: list[str],
    exit_code: int,
    base_url: str,
    started_api: bool,
    capacity_stress: bool,
    second_organization_required: bool,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = parse_junit_xml(junit_xml)
    now = datetime.now(UTC).isoformat()
    status = "PASS" if exit_code == 0 and summary.successful else "FAIL"
    matrix = _alignment_matrix(
        capacity_stress=capacity_stress,
        second_organization_required=second_organization_required,
    )

    payload: dict[str, Any] = {
        "batch": 13,
        "status": status,
        "generated_at_utc": now,
        "exit_code": int(exit_code),
        "base_url": base_url,
        "api_started_by_runner": bool(started_api),
        "capacity_stress_enabled": bool(capacity_stress),
        "second_organization_required": bool(second_organization_required),
        "junit": asdict(summary) | {
            "passed": summary.passed,
            "successful": summary.successful,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "safe_environment": safe_environment_snapshot(environ),
        "command": command,
        "alignment_matrix": matrix,
    }

    json_path = output_dir / "batch13_report.json"
    markdown_path = output_dir / "batch13_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Batch 13 — End-to-End Alignment Report",
        "",
        f"**Overall result:** `{status}`",
        "",
        f"- Generated at: `{now}`",
        f"- API base URL: `{base_url}`",
        f"- API started by runner: `{str(started_api).lower()}`",
        f"- Pytest exit code: `{exit_code}`",
        f"- Tests: `{summary.tests}`",
        f"- Passed: `{summary.passed}`",
        f"- Failed: `{summary.failures}`",
        f"- Errors: `{summary.errors}`",
        f"- Skipped: `{summary.skipped}`",
        f"- Duration: `{summary.time_seconds:.2f}s`",
        "",
        "## Alignment matrix",
        "",
        "| Area | Evidence | Status |",
        "|---|---|---|",
    ]
    for item in matrix:
        lines.append(
            f"| {item['area']} | {item['evidence']} | {item['status']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS result closes the static and executable alignment work through Batch 13. "
            "Skipped cross-organization tests are acceptable only when the stores contain ACCOUNT Tier B/C data for a single organization_id; "
            "use `--require-second-organization` to require a foreign organization_id for the isolation proof.",
            "",
            "The report never records database passwords, API keys, bearer tokens, raw prompts or retrieved document content.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "JUnitSummary",
    "SAFE_ENV_KEYS",
    "parse_junit_xml",
    "safe_environment_snapshot",
    "write_batch13_report",
]
