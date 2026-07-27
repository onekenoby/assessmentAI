#!/usr/bin/env bash
set -euo pipefail
BYTE_API_RUN_API_INTEGRATION=1 python -m pytest \
  tests/integration/test_api_real.py \
  -m "integration and not destructive" -v
