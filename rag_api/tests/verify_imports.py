"""Verifica import, dipendenze e bootstrap leggero del backend RAG.

Esecuzione dalla radice ``rag_api``::

    python tests/verify_imports.py

Il controllo non apre connessioni verso Ollama, PostgreSQL, Qdrant o Neo4j e
non carica i modelli di embedding/reranking.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    detail: str = ""
    version: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"


RUNTIME_DEPENDENCIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("fastapi", "fastapi", ("FastAPI", "APIRouter")),
    ("pydantic", "pydantic", ("BaseModel", "ConfigDict", "field_validator", "model_validator")),
    ("uvicorn", "uvicorn", ("run",)),
    ("requests", "requests", ("Session", "post")),
    ("psycopg2-binary", "psycopg2", ()),
    ("qdrant-client", "qdrant_client", ("QdrantClient", "models")),
    ("neo4j", "neo4j", ("GraphDatabase",)),
    ("openai", "openai", ("OpenAI",)),
    ("sentence-transformers", "sentence_transformers", ("SentenceTransformer", "CrossEncoder")),
    ("torch", "torch", ("cuda",)),
)

TEST_DEPENDENCIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("httpx", "httpx", ("Client",)),
    ("pytest", "pytest", ("main",)),
)

INTERNAL_MODULES: tuple[str, ...] = (
    "core.config",
    "api.schemas",
    "core.tenant",
    "core.models",
    "core.solvers",
    "core.reranking",
    "core.prompting",
    "core.generation",
    "core.validation",
    "core.audit",
    "core.resources",
    "core.retrieval",
    "core.rag_service",
    "api.routes_rag",
    "main",
)


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return ""


def check_python() -> CheckResult:
    current = sys.version_info
    detail = f"{platform.python_implementation()} {platform.python_version()}"
    if current < (3, 11):
        return CheckResult("python", "FAIL", detail + "; richiesto Python >= 3.11")
    if current[:2] != (3, 11):
        return CheckResult(
            "python",
            "WARN",
            detail + "; il container di progetto è validato su Python 3.11.x",
        )
    return CheckResult("python", "OK", detail)


def check_dependency(
    distribution_name: str,
    module_name: str,
    symbols: tuple[str, ...],
    *,
    required: bool,
) -> CheckResult:
    package_version = _package_version(distribution_name)
    if importlib.util.find_spec(module_name) is None:
        return CheckResult(
            distribution_name,
            "FAIL" if required else "WARN",
            f"modulo '{module_name}' non installato",
            package_version,
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - il checker deve mostrare ogni errore di import
        return CheckResult(
            distribution_name,
            "FAIL" if required else "WARN",
            f"{type(exc).__name__}: {exc}",
            package_version,
        )

    missing_symbols = [symbol for symbol in symbols if not hasattr(module, symbol)]
    if missing_symbols:
        return CheckResult(
            distribution_name,
            "FAIL" if required else "WARN",
            "simboli mancanti: " + ", ".join(missing_symbols),
            package_version,
        )

    return CheckResult(
        distribution_name,
        "OK",
        f"import '{module_name}' riuscito",
        package_version,
    )


def check_internal_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            module_name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(module_name, "OK", "import riuscito")


def check_no_startup_side_effect() -> CheckResult:
    try:
        from core.resources import ResourceState, resources
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "resource-import-side-effect",
            "FAIL",
            f"impossibile leggere ResourceManager: {type(exc).__name__}: {exc}",
        )

    if resources.state != ResourceState.NEW:
        return CheckResult(
            "resource-import-side-effect",
            "FAIL",
            f"stato risorse dopo import: {resources.state.value}; atteso: new",
        )

    return CheckResult(
        "resource-import-side-effect",
        "OK",
        "nessun modello o client inizializzato durante l'import",
    )


def check_fastapi_smoke() -> CheckResult:
    try:
        from fastapi.testclient import TestClient
        from main import create_app

        app = create_app(initialize_on_startup=False)
        with TestClient(app) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            openapi = client.get("/openapi.json")

        if live.status_code != 200:
            return CheckResult(
                "fastapi-smoke",
                "FAIL",
                f"/health/live ha restituito {live.status_code}",
            )
        if ready.status_code != 503:
            return CheckResult(
                "fastapi-smoke",
                "FAIL",
                f"/health/ready senza startup ha restituito {ready.status_code}; atteso 503",
            )
        if openapi.status_code != 200:
            return CheckResult(
                "fastapi-smoke",
                "FAIL",
                f"/openapi.json ha restituito {openapi.status_code}",
            )

        return CheckResult(
            "fastapi-smoke",
            "OK",
            "liveness=200, readiness=503, openapi=200",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "fastapi-smoke",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


def run_checks() -> list[CheckResult]:
    # Impedisce al test di dipendere da variabili esterne non controllate.
    os.environ.setdefault("POC_MODE", "1")

    results: list[CheckResult] = [check_python()]

    for distribution_name, module_name, symbols in RUNTIME_DEPENDENCIES:
        results.append(
            check_dependency(
                distribution_name,
                module_name,
                symbols,
                required=True,
            )
        )

    for distribution_name, module_name, symbols in TEST_DEPENDENCIES:
        results.append(
            check_dependency(
                distribution_name,
                module_name,
                symbols,
                required=False,
            )
        )

    for module_name in INTERNAL_MODULES:
        results.append(check_internal_import(module_name))

    results.append(check_no_startup_side_effect())

    if all(result.ok for result in results if result.name in {"fastapi", "httpx"}):
        results.append(check_fastapi_smoke())
    else:
        results.append(
            CheckResult(
                "fastapi-smoke",
                "WARN",
                "test non eseguito perché FastAPI/httpx non sono disponibili",
            )
        )

    return results


def print_report(results: list[CheckResult]) -> None:
    print("\nRAG API - verifica import e dipendenze")
    print("=" * 78)
    for result in results:
        version_text = f" [{result.version}]" if result.version else ""
        detail_text = f" - {result.detail}" if result.detail else ""
        print(f"{result.status:5} {result.name:32}{version_text}{detail_text}")

    failures = [result for result in results if result.status == "FAIL"]
    warnings = [result for result in results if result.status == "WARN"]
    print("-" * 78)
    print(f"OK={sum(r.status == 'OK' for r in results)} WARN={len(warnings)} FAIL={len(failures)}")


def write_json_report(results: list[CheckResult]) -> Path:
    report_path = PROJECT_ROOT / "tests" / "import_dependency_report.json"
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "results": [asdict(result) for result in results],
        "ok": not any(result.status == "FAIL" for result in results),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    results = run_checks()
    print_report(results)
    report_path = write_json_report(results)
    print(f"Report JSON: {report_path}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
