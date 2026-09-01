from __future__ import annotations

import subprocess
import sys


raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "not integration",
            "--cov=api",
            "--cov=core",
            "--cov=main",
            "--cov-report=term-missing",
        ]
    )
)
