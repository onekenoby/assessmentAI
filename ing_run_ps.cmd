$env:LLM_MODEL_NAME="llama3.1:8b"
$env:VISION_MODEL_NAME="ministral-3:8b"

$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:USE_REMOTE_OLLAMA="1"
$env:OLLAMA_AUTOSTART="0"

$env:OLLAMA_API_CHAT="http://127.0.0.1:11434/api/chat"
$env:OLLAMA_API_GENERATE="http://127.0.0.1:11434/api/generate"

$env:PG_HOST="127.0.0.1"
$env:PG_PORT="5433"
$env:PG_DB="assessment_ingestion"
$env:PG_USER="admin"
$env:PG_PASS="admin_password"

$env:QDRANT_HOST="127.0.0.1"
$env:QDRANT_PORT="6334"
$env:QDRANT_COLLECTION="assessment_docs"

$env:NEO4J_URI="bolt://127.0.0.1:7688"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASS="admin_password"

python ingestion.py