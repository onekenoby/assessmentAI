#!/usr/bin/env sh
set -eu
RUN_REAL_SERVICE_TESTS=1 python -m pytest tests/integration/test_api_real.py -m "integration and not destructive" -v
