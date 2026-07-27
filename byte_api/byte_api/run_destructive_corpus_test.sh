#!/usr/bin/env bash
set -euo pipefail
if [[ "${BYTE_API_RUN_DESTRUCTIVE:-}" != "1" ]]; then
  echo "Impostare BYTE_API_RUN_DESTRUCTIVE=1. Il test modifica il Database A." >&2
  exit 2
fi
python -m pytest \
  tests/integration/test_destructive_uploads.py::test_real_corpus_upload_creates_pending_job \
  -m "integration and destructive" -v
