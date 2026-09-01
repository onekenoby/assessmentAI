param(
    [string]$OllamaBaseUrl = "http://127.0.0.1:11435",
    [int]$ApiPort = 8013,
    [string]$EmbeddingModel = "E:/Dev/assessmentAI/models/bge-m3",
    [string]$RerankerModel = "E:/Dev/assessmentAI/models/ms-marco-reranker"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:TOKENIZERS_PARALLELISM = "false"

# ------------------------------------------------------------
# Ollama remoto su Vast.ai tramite lo stesso tunnel dell'ingestion:
# 127.0.0.1:11435 -> Vast.ai 127.0.0.1:11434
# ------------------------------------------------------------
$env:OLLAMA_BASE_URL = $OllamaBaseUrl.TrimEnd("/")
$env:LLM_MODEL_NAME = "gemma4:12b"
$env:EVAL_MODEL_NAME = "gemma4:12b"
$env:OLLAMA_WARMUP_ON_STARTUP = "1"
$env:OLLAMA_WARMUP_TIMEOUT_S = "600"
$env:OLLAMA_REQUEST_KEEP_ALIVE = "-1m"
$env:OLLAMA_CONNECT_TIMEOUT_S = "30"
$env:LLM_TIMEOUT_S = "600"
$env:LLM_MAX_ATTEMPTS = "2"
$env:LLM_NUM_CTX = "12288"
$env:LLM_NUM_PREDICT = "3072"
$env:LLM_TEMPERATURE = "0.15"
$env:LLM_REPEAT_PENALTY = "1.15"


# -----------------------------------------------------------
# QUDRANT ETC...
# -----------------------------------------------------------
# Retrieval veloce conservativo.
$env:QDRANT_CANDIDATES = "80"
$env:RERANK_CANDIDATES = "28"
$env:FINAL_SOURCES = "8"
$env:EVAL_ENABLED = "0"

# ------------------------------------------------------------
# Modelli locali Windows. Devono coincidere con quelli usati
# dall'ingestion per mantenere allineato lo spazio vettoriale.
# ------------------------------------------------------------
$env:EMBEDDING_MODEL_NAME = $EmbeddingModel
$env:RERANKER_MODEL_NAME = $RerankerModel
$env:EMBED_DEVICE = "cpu"
$env:RERANK_DEVICE = "cpu"
$env:EMBED_CPU_THREADS = "8"

# ------------------------------------------------------------
# Data stores locali: stessi endpoint dell'ingestion.
# ------------------------------------------------------------
$env:PG_ENRICH_ENABLED = "1"
$env:PG_HOST = "127.0.0.1"
$env:PG_PORT = "5433"
$env:PG_DB = "assessment_ingestion"
$env:PG_USER = "admin"
$env:PG_PASS = "admin_password"
$env:PG_MIN_CONN = "1"
$env:PG_MAX_CONN = "8"

$env:QDRANT_HOST = "127.0.0.1"
$env:QDRANT_PORT = "6334"
$env:QDRANT_COLLECTION = "assessment_docs"

$env:NEO4J_ENABLED = "1"
$env:NEO4J_URI = "bolt://127.0.0.1:7688"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASS = "admin_password"

# ------------------------------------------------------------
# Profilo PoC tenant-safe coerente con il progetto esistente.
# ------------------------------------------------------------
$env:POC_MODE = "1"
$env:POC_ORGANIZATION_ID = "1234"
$env:RAG_USER_ID = "service-user"
$env:RAG_USER_ROLES = "user,auditor"
$env:RAG_ALLOWED_SCOPES = "GLOBAL,ACCOUNT"
$env:RAG_DEFAULT_TIERS = "A,B,C"
$env:CORPUS_VERSION = "v1"

# Una sola query LLM pesante alla volta sulla GPU remota.
$env:RAG_MAX_CONCURRENT_QUERIES = "1"
$env:RAG_MAX_QUEUED_QUERIES = "4"
$env:RAG_QUERY_QUEUE_TIMEOUT_S = "30"

# ------------------------------------------------------------
# Preflight: tunnel e modello Ollama.
# ------------------------------------------------------------
Write-Host ""
Write-Host "=== Preflight RAG Vast.ai ===" -ForegroundColor Cyan
Write-Host "OLLAMA_BASE_URL:        $env:OLLAMA_BASE_URL"
Write-Host "LLM_MODEL_NAME:         $env:LLM_MODEL_NAME"
Write-Host "EMBEDDING_MODEL_NAME:   $env:EMBEDDING_MODEL_NAME"
Write-Host "RERANKER_MODEL_NAME:    $env:RERANKER_MODEL_NAME"
Write-Host "RAG API:                http://127.0.0.1:$ApiPort"
Write-Host ""

if (-not (Test-Path -LiteralPath $env:EMBEDDING_MODEL_NAME)) {
    throw "Modello embedding non trovato: $env:EMBEDDING_MODEL_NAME"
}
if (-not (Test-Path -LiteralPath $env:RERANKER_MODEL_NAME)) {
    throw "Modello reranker non trovato: $env:RERANKER_MODEL_NAME"
}

try {
    $tags = Invoke-RestMethod `
        -Method Get `
        -Uri "$($env:OLLAMA_BASE_URL)/api/tags" `
        -TimeoutSec 20
} catch {
    throw "Ollama Vast.ai non raggiungibile su $env:OLLAMA_BASE_URL. Verificare che il tunnel SSH 11435 -> 11434 sia attivo. Dettaglio: $($_.Exception.Message)"
}

$modelNames = @(
    $tags.models | ForEach-Object {
        if ($_.name) { [string]$_.name }
        elseif ($_.model) { [string]$_.model }
    }
)
if ($modelNames -notcontains $env:LLM_MODEL_NAME) {
    throw "Il modello $env:LLM_MODEL_NAME non risulta installato su Vast.ai. Modelli disponibili: $($modelNames -join ', ')"
}

Write-Host "Tunnel Ollama raggiungibile e modello presente." -ForegroundColor Green
Write-Host "Il modello verrà caricato in VRAM durante lo startup." -ForegroundColor Green
Write-Host ""

$pythonExe = "python"
foreach ($candidate in @(".\venv\Scripts\python.exe", "..\venv\Scripts\python.exe")) {
    if (Test-Path -LiteralPath $candidate) {
        $pythonExe = $candidate
        break
    }
}

& $pythonExe -u -m uvicorn main:app `
    --host 127.0.0.1 `
    --port $ApiPort `
    --workers 1 `
    --log-level info `
    --access-log

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
