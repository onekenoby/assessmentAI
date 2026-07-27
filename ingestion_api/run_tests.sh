#!/usr/bin/env sh
set -eu
python -m pytest -m "not integration" --cov=api --cov=core --cov=main --cov-report=term-missing
