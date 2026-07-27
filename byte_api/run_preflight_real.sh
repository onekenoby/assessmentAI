#!/usr/bin/env bash
set -euo pipefail
BYTE_API_RUN_INTEGRATION=1 python -m pytest \
  tests/integration/test_real_dependency.py \
  -m "integration and not destructive" -v
