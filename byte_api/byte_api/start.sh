#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn main:app --host 0.0.0.0 --port 8020 --workers 1
