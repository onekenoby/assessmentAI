#!/usr/bin/env sh
set -eu

export PYTHONUNBUFFERED=1
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://<VAST_PUBLIC_IP>:<VAST_EXTERNAL_PORT>}"
export USE_REMOTE_OLLAMA=1
export OLLAMA_AUTOSTART=0
export LLM_MODEL_NAME="llama3.1:8b"
export VISION_MODEL_NAME="ministral-3:8b"
export OLLAMA_TIMEOUT_S=240
export OLLAMA_VISION_TIMEOUT_S=240
export OLLAMA_KG_TIMEOUT_S=180
export OLLAMA_RETRIES=1
export USE_PRODUCER_CONSUMER=1
export DOC_QUEUE_MAXSIZE=1

exec python -m uvicorn main:app --host 0.0.0.0 --port 8010 --workers 1
