# =====================================================
# GUI REFLEX - CONFIGURAZIONE MODELLI
# =====================================================
$env:LLM_MODEL_NAME="gemma4:12b"
$env:VISION_MODEL_NAME="gemma4:12b"

# =====================================================
# OLLAMA CONTAINERIZZATO
# =====================================================
$env:OLLAMA_URL="http://127.0.0.1:11434/v1"
$env:OLLAMA_NATIVE_CHAT_URL="http://127.0.0.1:11434/api/chat"
$env:OLLAMA_API_KEY="ollama"

# =====================================================
# PARAMETRI LLM
# =====================================================
$env:LLM_NUM_CTX="16384"
$env:LLM_NUM_PREDICT="4096"
$env:LLM_TIMEOUT_S="300"

# =====================================================
# DATABASE / VECTOR DB / GRAPH DB
# Allineati al docker-compose assessment
# =====================================================
$env:PG_ENRICH_ENABLED="1"
$env:PG_HOST="127.0.0.1"
$env:PG_PORT="5433"
$env:PG_DB="assessment_ingestion"
$env:PG_USER="admin"
$env:PG_PASS="admin_password"

$env:QDRANT_HOST="127.0.0.1"
$env:QDRANT_PORT="6334"
$env:QDRANT_COLLECTION="assessment_docs"

$env:NEO4J_ENABLED="1"
$env:NEO4J_URI="bolt://127.0.0.1:7688"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASS="admin_password"

# =====================================================
# DEVICE MODELLI LOCALI
# Conviene CPU per non competere con Ollama sulla GPU
# =====================================================
$env:EMBED_DEVICE="cpu"
$env:RERANK_DEVICE="cpu"

# =====================================================
# LOG
# =====================================================
$env:AUDIT_ENABLED="1"
$env:RAG_LOG_DIR="$env:USERPROFILE\ai_rag_logs"

# =====================================================
# AVVIO GUI REFLEX
# =====================================================
reflex run