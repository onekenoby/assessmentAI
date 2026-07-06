from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "tests/test_generation_simulated.py",
    "tests/test_audit_simulated.py",
    "tests/test_resources_simulated.py",
    "tests/test_retrieval_simulated.py",
    "tests/test_rag_service_simulated.py",
    "tests/test_routes_rag_simulated.py",
]


def main() -> int:
    print("RAG API - test con servizi simulati")
    print("=" * 78)
    command = [sys.executable, "-m", "pytest", "-q", *FILES]
    print("Comando:", " ".join(command))
    print()
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
