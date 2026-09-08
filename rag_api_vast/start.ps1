param(
    [int]$ApiPort = 8000,

    [string]$OllamaBaseUrl = "http://127.0.0.1:11435",

    [string]$RerankerModel = "E:/Modelli/ms-marco-reranker"
)


$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = "1"
$env:TOKENIZERS_PARALLELISM = "false"


# ============================================================
# BGE-M3 REMOTO - VAST
# Windows 127.0.0.1:18002
#       -> SSH tunnel
# Vast    127.0.0.1:8002
# ============================================================

$env:EMBEDDING_PROVIDER = "remote"
$env:EMBEDDING_BASE_URL = "http://127.0.0.1:18002"
$env:EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
$env:EMBEDDING_DIMENSION = "1024"
$env:EMBEDDING_TIMEOUT_S = "120"

# Mantenuto per compatibilità con RagSettings.
# Non viene utilizzato dal provider remote.
$env:EMBED_DEVICE = "cpu"


# ============================================================
# RERANKER LOCALE - CPU
# ============================================================

$env:RERANKER_MODEL_NAME = $RerankerModel
$env:RERANK_DEVICE = "cpu"


# ============================================================
# OLLAMA REMOTO - VAST
# Windows 127.0.0.1:11435
#       -> SSH tunnel
# Vast    127.0.0.1:11434
# ============================================================

$env:OLLAMA_BASE_URL = $OllamaBaseUrl.TrimEnd("/")
$env:OLLAMA_NATIVE_CHAT_URL = "$($env:OLLAMA_BASE_URL)/api/chat"

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


# ============================================================
# RETRIEVAL
# ============================================================

$env:QDRANT_CANDIDATES = "80"
$env:RERANK_CANDIDATES = "28"
$env:FINAL_SOURCES = "8"

$env:EVAL_ENABLED = "0"


# ============================================================
# POSTGRESQL LOCALE
# ============================================================

$env:PG_ENRICH_ENABLED = "1"

$env:PG_HOST = "127.0.0.1"
$env:PG_PORT = "5433"
$env:PG_DB = "assessment_ingestion"

$env:PG_USER = "admin"
$env:PG_PASS = "admin_password"

$env:PG_MIN_CONN = "1"
$env:PG_MAX_CONN = "8"


# ============================================================
# QDRANT LOCALE
# ============================================================

$env:QDRANT_HOST = "127.0.0.1"
$env:QDRANT_PORT = "6334"
$env:QDRANT_COLLECTION = "assessment_docs"


# ============================================================
# NEO4J LOCALE
# ============================================================

$env:NEO4J_ENABLED = "1"

$env:NEO4J_URI = "bolt://127.0.0.1:7688"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASS = "admin_password"


# ============================================================
# PROFILO POC TENANT-SAFE
# ============================================================

$env:POC_MODE = "1"

# organization_id NON è statico.
# Deve essere fornito da ogni singola query tramite:
# X-RAG-Organization-ID
#
# RAG_USER_ID e ruoli sono identità tecnica/audit del PoC.
# NON costituiscono la barriera di segregazione documentale.

$env:RAG_USER_ID = "service-user"
$env:RAG_USER_ROLES = "user,auditor"

$env:RAG_ALLOWED_SCOPES = "GLOBAL,ACCOUNT"
$env:RAG_DEFAULT_TIERS = "A,B,C"

$env:CORPUS_VERSION = "v1"


# ============================================================
# CAPACITY
# Una sola query LLM pesante alla volta sulla GPU remota
# ============================================================

$env:RAG_MAX_CONCURRENT_QUERIES = "1"
$env:RAG_MAX_QUEUED_QUERIES = "4"
$env:RAG_QUERY_QUEUE_TIMEOUT_S = "30"


# ============================================================
# PREFLIGHT
# ============================================================

Write-Host ""
Write-Host "=== Preflight RAG Vast.ai ===" -ForegroundColor Cyan

Write-Host "OLLAMA_BASE_URL:         $env:OLLAMA_BASE_URL"
Write-Host "OLLAMA_NATIVE_CHAT_URL:  $env:OLLAMA_NATIVE_CHAT_URL"

Write-Host "LLM_MODEL_NAME:          $env:LLM_MODEL_NAME"

Write-Host "EMBEDDING_PROVIDER:      $env:EMBEDDING_PROVIDER"
Write-Host "EMBEDDING_BASE_URL:      $env:EMBEDDING_BASE_URL"
Write-Host "EMBEDDING_MODEL_NAME:    $env:EMBEDDING_MODEL_NAME"

Write-Host "RERANKER_MODEL_NAME:     $env:RERANKER_MODEL_NAME"
Write-Host "RERANK_DEVICE:           $env:RERANK_DEVICE"

Write-Host "TENANT ORGANIZATION_ID:  dinamico via X-RAG-Organization-ID"

Write-Host "RAG API:                 http://127.0.0.1:$ApiPort"

Write-Host ""


# ============================================================
# PREFLIGHT RERANKER LOCALE
# ============================================================

if (-not (Test-Path -LiteralPath $env:RERANKER_MODEL_NAME)) {

    throw "Modello reranker non trovato: $env:RERANKER_MODEL_NAME"
}

Write-Host "Reranker locale presente." -ForegroundColor Green


# ============================================================
# PREFLIGHT BGE-M3 REMOTO
# ============================================================

try {

    $bgeHealth = Invoke-RestMethod `
        -Method Get `
        -Uri "$($env:EMBEDDING_BASE_URL)/health" `
        -TimeoutSec 20

}
catch {

    throw "BGE-M3 Vast non raggiungibile su $env:EMBEDDING_BASE_URL. Verificare che il tunnel SSH 18002 -> 8002 sia attivo. Dettaglio: $($_.Exception.Message)"
}


if (-not $bgeHealth.ready) {

    throw "BGE-M3 Vast raggiungibile ma non READY."
}


if ($bgeHealth.model -ne "BAAI/bge-m3") {

    throw "Modello BGE inatteso: $($bgeHealth.model)"
}


if ([int]$bgeHealth.dimension -ne 1024) {

    throw "Dimensione BGE inattesa: $($bgeHealth.dimension)"
}


Write-Host "BGE-M3 Vast READY." -ForegroundColor Green


# ============================================================
# PREFLIGHT OLLAMA / GEMMA4
# ============================================================

try {

    $tags = Invoke-RestMethod `
        -Method Get `
        -Uri "$($env:OLLAMA_BASE_URL)/api/tags" `
        -TimeoutSec 20

}
catch {

    throw "Ollama Vast.ai non raggiungibile su $env:OLLAMA_BASE_URL. Verificare che il tunnel SSH 11435 -> 11434 sia attivo. Dettaglio: $($_.Exception.Message)"
}


$modelNames = @(
    $tags.models | ForEach-Object {

        if ($_.name) {

            [string]$_.name

        }
        elseif ($_.model) {

            [string]$_.model
        }
    }
)


if ($modelNames -notcontains $env:LLM_MODEL_NAME) {

    throw "Il modello $env:LLM_MODEL_NAME non risulta installato su Vast.ai. Modelli disponibili: $($modelNames -join ', ')"
}


Write-Host "Ollama Vast raggiungibile." -ForegroundColor Green
Write-Host "Modello $env:LLM_MODEL_NAME presente." -ForegroundColor Green

Write-Host ""
Write-Host "Preflight completato con successo." -ForegroundColor Green
Write-Host "Avvio RAG API..." -ForegroundColor Cyan
Write-Host ""


# ============================================================
# PYTHON
# ============================================================

$pythonExe = "python"

foreach ($candidate in @(
    ".\venv\Scripts\python.exe",
    "..\venv\Scripts\python.exe"
)) {

    if (Test-Path -LiteralPath $candidate) {

        $pythonExe = $candidate
        break
    }
}


# ============================================================
# UVICORN
# ============================================================

& $pythonExe -u -m uvicorn main:app `
    --host 127.0.0.1 `
    --port $ApiPort `
    --workers 1 `
    --log-level info `
    --access-log


if ($LASTEXITCODE -ne 0) {

    exit $LASTEXITCODE
}