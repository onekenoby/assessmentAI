from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    current = Path(__file__).resolve().parent
    if (current / "main.py").exists():
        return current
    if (current.parent / "main.py").exists():
        return current.parent
    raise RuntimeError("Impossibile individuare la root del progetto RAG API")


PROJECT_ROOT = _project_root()
INTEGRATION_DIR = PROJECT_ROOT / "tests" / "integration"

SERVICE_FILES = {
    "postgres": INTEGRATION_DIR / "test_postgres_real.py",
    "qdrant": INTEGRATION_DIR / "test_qdrant_real.py",
    "neo4j": INTEGRATION_DIR / "test_neo4j_real.py",
    "ollama": INTEGRATION_DIR / "test_ollama_real.py",
    "isolation": INTEGRATION_DIR / "test_organization_isolation_real.py",
    "api": INTEGRATION_DIR / "test_rag_api_e2e_real.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue test reali non distruttivi dei servizi RAG."
    )
    parser.add_argument(
        "--service",
        choices=["all", *SERVICE_FILES.keys()],
        default="all",
        help="Servizio o gruppo da testare.",
    )
    parser.add_argument(
        "--allow-empty-data",
        action="store_true",
        help="Non fallire quando gli store sono raggiungibili ma vuoti.",
    )
    parser.add_argument(
        "--require-second-organization",
        action="store_true",
        help="Rende obbligatoria la presenza di dati ACCOUNT Tier B/C associati a una seconda organization_id.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000"),
        help="URL base usato dai test HTTP API.",
    )
    parser.add_argument(
        "--capacity-stress",
        action="store_true",
        help="Abilita il test HTTP concorrente service_busy.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout di rete in secondi.",
    )
    parser.add_argument("--quiet", action="store_true", help="Output pytest compatto.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    env["RUN_REAL_SERVICE_TESTS"] = "1"
    env["RAG_INTEGRATION_REQUIRE_DATA"] = "0" if args.allow_empty_data else "1"
    env["RAG_INTEGRATION_TIMEOUT_S"] = str(max(5, args.timeout))
    env["RAG_API_BASE_URL"] = str(args.base_url).rstrip("/")
    env["RAG_E2E_REQUIRE_SECOND_ORGANIZATION"] = "1" if args.require_second_organization else "0"
    env["RAG_E2E_RUN_CAPACITY_STRESS"] = "1" if args.capacity_stress else "0"

    selected = (
        list(SERVICE_FILES.values())
        if args.service == "all"
        else [SERVICE_FILES[args.service]]
    )

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q" if args.quiet else "-v",
        "-s",
        *[str(path) for path in selected],
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
