$env:PYTHONUNBUFFERED = "1"

# ------------------------------------------------------------
# Ollama remoto attraverso il tunnel SSH:
# 127.0.0.1:11435 -> Vast.ai 127.0.0.1:11434
# ------------------------------------------------------------
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11435"

$env:EMBEDDING_PROVIDER = "remote"
$env:EMBEDDING_BASE_URL = "http://127.0.0.1:18002"
$env:EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
$env:EMBEDDING_DIMENSION = "1024"
$env:EMBEDDING_TIMEOUT_S = "120"


$env:USE_REMOTE_OLLAMA = "1"
$env:OLLAMA_AUTOSTART = "0"

$env:LLM_MODEL_NAME = "llama3.1:8b"
$env:VISION_MODEL_NAME = "ministral-3:8b"

# Mantiene i modelli caricati durante la run.
$env:OLLAMA_REQUEST_KEEP_ALIVE = "30m"

# Timeout.
$env:OLLAMA_TIMEOUT_S = "240"
$env:OLLAMA_VISION_TIMEOUT_S = "240"
$env:OLLAMA_KG_TIMEOUT_S = "180"
$env:OLLAMA_RETRIES = "1"

# Metriche Ollama nel log.
$env:OLLAMA_LOG_TIMINGS = "1"

# ------------------------------------------------------------
# Profilo RTX 5090
# ------------------------------------------------------------
$env:VISION_PAGE_DPI = "160"
$env:VISION_PAGE_NUM_CTX = "8192"
$env:VISION_PAGE_MAX_TOKENS = "3200"

$env:FORMULA_RENDER_DPI = "210"
$env:FORMULA_VISION_NUM_CTX = "8192"
$env:FORMULA_VISION_MAX_TOKENS = "1500"
$env:FORMULA_VISION_MAX_SIDE = "2400"

# Gate Vision più selettivi.
$env:PDF_VECTOR_VISION_THRESHOLD = "40"
$env:PDF_VECTOR_ONLY_MAX_TEXT_LEN = "80"
$env:PDF_TABLE_DIGIT_DENSITY = "0.15"


$env:PDF_SKIP_FORMULA_AFTER_PAGE_VISION = "1"

# Corruzione reale del text layer.
$env:PDF_CORRUPTION_BAD_CHAR_RATIO = "0.01"
$env:PDF_CORRUPTION_MIN_BAD_CHARS = "3"

# Asset embedded.
# Esclude loghi, icone, banner e immagini uniformi.
$env:PDF_EMBEDDED_MIN_WIDTH = "300"
$env:PDF_EMBEDDED_MIN_HEIGHT = "180"
$env:PDF_EMBEDDED_MIN_AREA = "100000"
$env:PDF_EMBEDDED_MAX_ASPECT_RATIO = "6.0"
$env:PDF_EMBEDDED_MIN_TONAL_RANGE = "12"
$env:PDF_EMBEDDED_SKIP_DUPLICATES = "1"
$env:PDF_MAX_IMAGES_PER_PAGE = "5"

# Chart Vision.
# Tutte le chiamate Ministral usano lo stesso contesto.
$env:CHART_VISION_NUM_CTX = "8192"
$env:CHART_REQUIRE_OCR_TEXT = "1"
$env:CHART_MIN_OCR_CHARS = "12"
$env:CHART_VISION_RETRY_ENABLED = "1"
$env:CHART_VISION_RETRY_ON_EMPTY = "0"


# ------------------------------------------------------------
# Parallelismo controllato
# ------------------------------------------------------------

# La Vision resta seriale.
$env:VISION_PARALLEL_WORKERS = "1"
$env:OLLAMA_VISION_PARALLELISM = "1"

# Due task KG e due richieste testuali Ollama concorrenti.
$env:KG_WORKERS = "4"
$env:OLLAMA_TEXT_PARALLELISM = "4"
$env:OLLAMA_NUM_PARALLEL = "4"

# Embeddings BGE-M3 locali.
# Il PDF non forza più il batch a 8.
$env:PDF_EMBED_BATCH_SIZE = "16"

# Estrazione KG deterministica:
# una sola chiamata JSON per finestra.
$env:KG_INPUT_MAX_CHARS = "2600"
$env:KG_NUM_CTX = "8192"
$env:KG_MAX_OUTPUT_TOKENS = "1800"
$env:KG_MAX_NODES_PER_WINDOW = "20"
$env:KG_MAX_EDGES_PER_WINDOW = "30"

# Pipeline locale.
$env:USE_PRODUCER_CONSUMER = "1"
$env:DOC_QUEUE_MAXSIZE = "1"

# Batch embeddings DOCX/Markdown.
$env:EMBED_BATCH_SIZE = "32"

$env:INGESTION_VERBOSE_CHUNKS = "0"

# $env:EMBED_DEVICE = "cuda:0"
# $env:EMBED_DEVICE = "cpu"

Write-Host ""
Write-Host "=== Configurazione ingestion ===" -ForegroundColor Cyan
Write-Host "OLLAMA_BASE_URL:               $env:OLLAMA_BASE_URL"
Write-Host "LLM_MODEL_NAME:                $env:LLM_MODEL_NAME"
Write-Host "VISION_MODEL_NAME:             $env:VISION_MODEL_NAME"
Write-Host "OLLAMA_REQUEST_KEEP_ALIVE:     $env:OLLAMA_REQUEST_KEEP_ALIVE"
Write-Host "VISION_PAGE_DPI:               $env:VISION_PAGE_DPI"
Write-Host "VISION_PAGE_NUM_CTX:           $env:VISION_PAGE_NUM_CTX"
Write-Host "VISION_PAGE_MAX_TOKENS:        $env:VISION_PAGE_MAX_TOKENS"
Write-Host "FORMULA_RENDER_DPI:            $env:FORMULA_RENDER_DPI"

Write-Host "PDF_VECTOR_VISION_THRESHOLD:   $env:PDF_VECTOR_VISION_THRESHOLD"
Write-Host "PDF_VECTOR_ONLY_MAX_TEXT_LEN:  $env:PDF_VECTOR_ONLY_MAX_TEXT_LEN"
Write-Host "PDF_SKIP_FORMULA_AFTER_VISION: $env:PDF_SKIP_FORMULA_AFTER_PAGE_VISION"

Write-Host "PDF_EMBEDDED_MIN_WIDTH:        $env:PDF_EMBEDDED_MIN_WIDTH"
Write-Host "PDF_EMBEDDED_MIN_HEIGHT:       $env:PDF_EMBEDDED_MIN_HEIGHT"
Write-Host "PDF_EMBEDDED_MIN_AREA:         $env:PDF_EMBEDDED_MIN_AREA"

Write-Host "CHART_VISION_NUM_CTX:          $env:CHART_VISION_NUM_CTX"
Write-Host "CHART_REQUIRE_OCR_TEXT:        $env:CHART_REQUIRE_OCR_TEXT"
Write-Host "CHART_VISION_RETRY_ON_EMPTY:   $env:CHART_VISION_RETRY_ON_EMPTY"

Write-Host "VISION_PARALLEL_WORKERS:       $env:VISION_PARALLEL_WORKERS"
Write-Host "OLLAMA_VISION_PARALLELISM:     $env:OLLAMA_VISION_PARALLELISM"

Write-Host "KG_WORKERS:                    $env:KG_WORKERS"
Write-Host "OLLAMA_TEXT_PARALLELISM:       $env:OLLAMA_TEXT_PARALLELISM"
Write-Host "OLLAMA_NUM_PARALLEL:           $env:OLLAMA_NUM_PARALLEL"



Write-Host "PDF_EMBED_BATCH_SIZE:          $env:PDF_EMBED_BATCH_SIZE"
Write-Host "EMBED_BATCH_SIZE:              $env:EMBED_BATCH_SIZE"
Write-Host "PDF_VECTOR_ONLY_MAX_TEXT_LEN:  $env:PDF_VECTOR_ONLY_MAX_TEXT_LEN"

Write-Host "KG_INPUT_MAX_CHARS:            $env:KG_INPUT_MAX_CHARS"
Write-Host "KG_NUM_CTX:                    $env:KG_NUM_CTX"
Write-Host "KG_MAX_OUTPUT_TOKENS:          $env:KG_MAX_OUTPUT_TOKENS"

# Write-Host "EMBED_DEVICE:                  $env:EMBED_DEVICE"

Write-Host "==============================="
Write-Host ""



python -u -m uvicorn main:app `
    --host 0.0.0.0 `
    --port 8010 `
    --workers 1 `
    --log-level info `
    --access-log