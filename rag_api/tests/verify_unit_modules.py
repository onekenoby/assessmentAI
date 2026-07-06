from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TEST_FILES = (
    "tests/test_tenant.py",
    "tests/test_models.py",
    "tests/test_solvers.py",
    "tests/test_reranking.py",
    "tests/test_prompting.py",
    "tests/test_validation.py",
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q", *TEST_FILES]

    print("RAG API - test unitari moduli core (fase 1)")
    print("=" * 78)
    print("Project root:", project_root)
    print("Comando:", " ".join(command))
    print()

    completed = subprocess.run(command, cwd=project_root, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
