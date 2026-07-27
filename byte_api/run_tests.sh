#!/usr/bin/env bash
set -euo pipefail
python -m pytest -m "not integration" \
  --cov=api \
  --cov=core \
  --cov=main \
  --cov=byte_engine \
  --cov-report=term-missing
