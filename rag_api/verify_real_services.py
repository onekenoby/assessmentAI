from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_DIR = PROJECT_ROOT / "tests" / "integration"

SERVICE_FILES = {
    "postgres": INTEGRATION_DIR / "test_postgres_real.py",
    "qdrant": INTEGRATION_DIR / "test_qdrant_real.py",
    "neo4j": INTEGRATION_DIR / "test_neo4j_real.py",
    "ollama": INTEGRATION_DIR / "test_ollama_real.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue i test reali non distruttivi dei servizi RAG."
    )
    parser.add_argument(
        "--service",
        choices=["all", *SERVICE_FILES.keys()],
        default="all",
        help="Servizio da testare (default: all).",
    )
    parser.add_argument(
        "--allow-empty-data",
        action="store_true",
        help="Non fallire quando PostgreSQL/Qdrant/Neo4j sono raggiungibili ma vuoti.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout di rete per Ollama e probe, in secondi.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Usa output pytest compatto.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    env["RUN_REAL_SERVICE_TESTS"] = "1"
    env["RAG_INTEGRATION_REQUIRE_DATA"] = "0" if args.allow_empty_data else "1"
    env["RAG_INTEGRATION_TIMEOUT_S"] = str(max(5, args.timeout))

    selected = (
        list(SERVICE_FILES.values())
        if args.service == "all"
        else [SERVICE_FILES[args.service]]
    )

    print("\nRAG API - test reali PostgreSQL, Qdrant, Neo4j e Ollama")
    print("=" * 78)
    print(f"Servizio: {args.service}")
    print(f"Richiedi dati non vuoti: {not args.allow_empty_data}")
    print(f"Timeout: {max(5, args.timeout)}s")
    print("Modalità: read-only / non distruttiva")
    print()

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q" if args.quiet else "-v",
        "-s",
        "-m",
        "integration",
        *[str(path) for path in selected],
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
