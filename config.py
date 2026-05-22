import os

# --- FIX ANTI-BLOCCO ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# valori "safe" (evitano oversubscription e spesso evitano freeze)
CPU_THREADS = os.environ.get("EMBED_CPU_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = CPU_THREADS
os.environ["MKL_NUM_THREADS"] = CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = CPU_THREADS

# -------------------------
# KG LIMITS (coherent names)
# -------------------------
MIN_ENTITY_DENSITY = int(os.getenv("MIN_ENTITY_DENSITY", "1"))
MIN_FIN_KEYWORDS = int(os.getenv("MIN_FIN_KEYWORDS", "1"))
KG_TEXT_MAX_CHARS = int(os.getenv("KG_TEXT_MAX_CHARS", "2600"))
KG_MAX_TRIPLES = int(os.getenv("KG_MAX_TRIPLES", "50"))
KG_TIMEOUT = int(os.getenv("KG_TIMEOUT", "600"))

KG_CHARS_LIMIT = KG_TEXT_MAX_CHARS
KG_MAX_CHARS = KG_TEXT_MAX_CHARS
KG_MAX_TRIPLES_PER_PAGE = KG_MAX_TRIPLES
MAX_TRIPLES_KG = KG_MAX_TRIPLES
KG_TASK_TIMEOUT = KG_TIMEOUT
KG_TASK_TIMEOUT_PER_PAGE = KG_TIMEOUT
KG_TIMEOUT_PER_PAGE = KG_TIMEOUT

BASE_DATA_DIR = os.getenv("BASE_DIR", "./data/assessment")
INBOX_DIR = os.path.join(BASE_DATA_DIR, "INBOX")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/assessment/processed")
FAILED_DIR = os.getenv("FAILED_DIR", "./data/assessment/failed")

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "40"))

CONTEXT_WINDOW_CHARS = int(os.getenv("CONTEXT_WINDOW_CHARS", "260"))
INCLUDE_CONTEXT_IN_KG = os.getenv("INCLUDE_CONTEXT_IN_KG", "1") == "1"

DB_FLUSH_SIZE = int(os.getenv("DB_FLUSH_SIZE", "200"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# Vision switches
PDF_VISION_ENABLED = os.getenv("PDF_VISION_ENABLED", "1") == "1"
PDF_VISION_ONLY_IF_TEXT_SCARSO = False
PDF_MIN_TEXT_LEN_FOR_NO_VISION = 0

VISION_DPI = int(os.getenv("VISION_DPI", "130"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "2000000"))
VISION_MAX_FORMULAS_PER_PAGE = int(os.getenv("VISION_MAX_FORMULAS_PER_PAGE", "10"))

PDF_EXTRACT_EMBEDDED_IMAGES = True
PDF_VISION_ON_EMBEDDED_IMAGES = True
PDF_MAX_IMAGES_PER_PAGE = int(os.getenv("PDF_MAX_IMAGES_PER_PAGE", "8"))
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", "1"))
MIN_ASSET_SIZE = int(os.getenv("MIN_ASSET_SIZE", "2000"))

VISION_PARALLEL_WORKERS = 1
OLLAMA_NUM_PARALLEL=1
VISION_CACHE_MAX = int(os.getenv("VISION_CACHE_MAX", "5000"))

PG_COMMIT_EVERY_N_PAGES = int(os.getenv("PG_COMMIT_EVERY_N_PAGES", "25"))

KG_ENABLED = os.getenv("KG_ENABLED", "1") == "1"
KG_MIN_LEN = int(os.getenv("KG_MIN_LEN", "300"))
MAX_KG_CHUNKS_PER_DOC = int(os.getenv("MAX_KG_CHUNKS_PER_DOC", "50"))

PDF_TEXT_EXTRACTOR = "fitz"
FULLPAGE_DPI = 110 
CROP_DPI = 160 
KG_WORKERS = 1

REL_CANON_CACHE_PATH = os.getenv("REL_CANON_CACHE_PATH", "relation_canon_cache.json")
REL_CANON_MAX_TOKENS = int(os.getenv("REL_CANON_MAX_TOKENS", "700"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6334"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assessment_docs")

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB = os.getenv("PG_DB", "assessment_ingestion")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin_password")
PG_MIN_CONN = int(os.getenv("PG_MIN_CONN", "1"))
PG_MAX_CONN = int(os.getenv("PG_MAX_CONN", "8"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "admin_password")
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "1") == "1"

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1:8b") 
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "ministral-3:8b")
EMBEDDING_MODEL_NAME = "E:/Modelli/bge-m3"

QDRANT_TEXT_MAX_CHARS = int(os.getenv("QDRANT_TEXT_MAX_CHARS", "2500"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1300"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "2"))

OLLAMA_API_GENERATE = os.getenv("OLLAMA_API_GENERATE", "http://127.0.0.1:11434/api/generate")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "600"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
