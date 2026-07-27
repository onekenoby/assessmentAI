from __future__ import annotations

import os
import subprocess
import sys


if os.getenv("RUN_REAL_SERVICE_TESTS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
    print("Impostare RUN_REAL_SERVICE_TESTS=1 per abilitare i test reali.")
    raise SystemExit(2)

raise SystemExit(
    subprocess.call(
        [sys.executable, "-m", "pytest", "tests/integration", "-m", "integration", "-v"]
    )
)
