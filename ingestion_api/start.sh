#!/usr/bin/env sh
set -eu
export PYTHONUNBUFFERED=1
exec python -m uvicorn main:app --host 0.0.0.0 --port 8010 --workers 1
