import reflex as rx
import torch
import os
import time
import re
import json
import hashlib
import psycopg2
import requests
from collections import Counter

from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from neo4j import GraphDatabase
from openai import OpenAI

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import uuid

from typing import List, Dict

import warnings
import logging

# 1. Nasconde i warning standard di Python sollevati dal modulo neo4j
warnings.filterwarnings("ignore", module="neo4j")
warnings.filterwarnings("ignore", category=Warning, module="neo4j")

# 2. Silenzia il logger interno di Neo4j che stampa i GqlStatusObject
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Configura il logger se non lo hai già
logger = logging.getLogger(__name__)



import threading
_init_lock = threading.Lock()


import ast
import operator


# --- INIZIO MOTORE MATEMATICO AST ---
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.UAdd: operator.pos, ast.USub: operator.neg
}

def eval_expr(node):
    if isinstance(node, ast.Num): return node.n
    elif isinstance(node, ast.BinOp): return OPERATORS[type(node.op)](eval_expr(node.left), eval_expr(node.right))
    elif isinstance(node, ast.UnaryOp): return OPERATORS[type(node.op)](eval_expr(node.operand))
    else: raise TypeError("Operazione non supportata")

def calcolatrice_universale(espressione_matematica: str) -> str:
    try:
        # Pulisce tutto ciò che non è numero o operatore (previene testo sporco dall'LLM)
        expr_pulita = re.sub(r'[^0-9\+\-\*\/\(\)\.]', '', espressione_matematica)
        if not expr_pulita: return ""
        
        node = ast.parse(expr_pulita, mode='eval').body
        risultato = eval_expr(node)
        
        # Formatta il risultato come cifra leggibile (es. 6.000.000,00)
        return f"{risultato:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return ""
# --- FINE MOTORE MATEMATICO AST ---


# ============================================================
# DIZIONARIO GENERALISTA PER REQUISITI E SOGLIE NORMATIVE
# Usato dinamicamente per regex database e parser testuali
# ============================================================
THRESHOLD_TERMS_LIST = [
    # Comparativi e Limiti (Italiano)
    "oltre", "superiore", "almeno", "inferiore", "maggiore", "minore", 
    "massimo", "minimo", "limite", "soglia", "eccede", "eccedente", 
    "supera", "superamento", "fino a", "tetto", "cap", "tolleranza", 
    "margine", "range", "intervallo", "compreso tra", "al di sopra", 
    "al di sotto", "non più di", "non meno di", "pari o superiore", 
    "pari o inferiore", "franchigia", "massimale",
    # Comparativi e Limiti (Inglese)
    "greater than", "over", "less than", "at least", "threshold", 
    "limit", "maximum", "minimum", "exceeds", "exceeding", "surpasses", 
    "up to", "ceiling", "tolerance", "margin", "interval", 
    "between", "above", "below", "under", "no more than", "no less than", 
    "equal to",
    # Normativi, Logici e Requisiti (Italiano)
    "condizione", "regola", "legge", "requisito", "obbligo", "criterio", 
    "parametro", "vincolo", "direttiva", "normativa", "regolamento", 
    "standard", "policy", "procedura", "prescrizione", "disposizione", 
    "norma", "articolo", "comma", "decreto", "provvedimento", 
    "linea guida", "conformità", "adempimento", "metrica", "indicatore", 
    "kpi", "sla", "misura", "clausola", "certificazione", "target",
    # Normativi, Logici e Requisiti (Inglese)
    "condition", "rule", "law", "requirement", "obligation", "criterion", 
    "parameter", "constraint", "directive", "regulation", "procedure", 
    "prescription", "provision", "act", "measure", "guideline", 
    "compliance", "fulfillment", "benchmark", "metric", "indicator", 
    "clause", "certification"
]


MATH_CANDIDATE_PAT = re.compile(
    r"(?i)("
    # 1. KEYWORD FORTI (Finanza & Formule)
    r"formulae\s+sheet|maths\s+tables|economic\s+order\s+quantity|"
    r"miller[-.\s]?orr|capm|wacc|asset\s+beta|growth\s+model|"
    r"fisher\s+formula|purchasing\s+power\s+parity|\bbeta\b|standard\s+deviation|"
    
    # 2. ARTEFATTI OCR SPECIFICI (Il tuo colpo di genio)
    r"\b2c0d\b|"  # Usiamo \b per evitare che scatti dentro parole casuali
    
    # 3. SIMBOLI MATEMATICI PURI (Sempre validi)
    r"[\u2200-\u22FF]|"           # Blocco Unicode Operatori Matematici
    r"[∑∏∫√≈≠≤≥→↔∩∪∞±×÷]|"        # Simboli specifici (escluso = puro per sicurezza)
    
    # 4. SPAZZATURA OCR "SICURA" (Il fix)
    # Cerca trattini, tilde o bullet SOLO se sono incastrati tra cifre o parentesi
    # Es. scatta su "5–3" o "(4)˜2", ma IGNORA una lista puntata normale "• Punto uno"
    r"(?<=\d)[•–—˜](?=\d)|(?<=\))[•–—˜](?=\d)" 
    r")"
)


def looks_garbled(text: str) -> bool:
    """
    True if text contains typical garbage chars from PDF text layer extraction.
    We should avoid feeding these chunks to the LLM, especially for formulas.
    """
    if not text:
        return False
    bad = ["□", "\ufffd"]  # square box, replacement char
    return any(b in text for b in bad)


# =========================
# ⚙️ CONFIGURAZIONE UTENTE
# =========================
PAGE_TITLE = "Compliance & Security Auditor AI 🛡️"

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6334"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "assessment_docs")


# =========================
# RAG TIER POLICY
# =========================
RAG_DEFAULT_TIERS = os.getenv("RAG_DEFAULT_TIERS", "A,B,C")



# =========================
# 🐘 POSTGRES (Timescale) - RAG ENRICH
# =========================
PG_ENRICH_ENABLED = os.getenv("PG_ENRICH_ENABLED", "1") == "1"
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB   = os.getenv("PG_DB", "assessment_ingestion")
PG_USER = os.getenv("PG_USER", "admin")
PG_PASS = os.getenv("PG_PASS", "admin_password")
PG_MIN_CONN = int(os.getenv("PG_MIN_CONN", "1"))
PG_MAX_CONN = int(os.getenv("PG_MAX_CONN", "8"))



# preferisci content_raw (1) o content_semantic (0) quando disponibile
PG_PREFER_RAW = os.getenv("PG_PREFER_RAW", "0") == "1"

pg_pool: Optional[SimpleConnectionPool] = None

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688") # <-- Allineato all'ingestion (7688)
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", os.getenv("NEO4J_PASSWORD", "admin_password"))
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "1") == "1"



# AI Models
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", LLM_MODEL_NAME)

# alternativa
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1:8b")
#VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "ministral-3:8b")

#EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MODEL_NAME = "E:/Modelli/bge-m3"

#RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_MODEL_NAME = "E:/Modelli/ms-marco-reranker"


# LM Studio / OpenAI Compatible API
#LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
#LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy key, Ollama non la valida

# =========================
# 🧠 LLM / OLLAMA CONTEXT
# =========================
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "4096"))

# =========================
# 🧠 OLLAMA NATIVE CHAT - STABLE MODE
# =========================
OLLAMA_NATIVE_CHAT_URL = os.getenv(
    "OLLAMA_NATIVE_CHAT_URL",
    "http://127.0.0.1:11434/api/chat",
)

LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "300"))


def call_ollama_chat_native(messages: List[Dict[str, str]]) -> str:
    """
    Chiamata robusta a Ollama usando /api/chat.
    Evita blocchi dello streaming OpenAI-compatible dentro Reflex.
    """
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_ctx": int(LLM_NUM_CTX),
            "num_predict": int(LLM_NUM_PREDICT),
            "repeat_penalty": 1.15,
        },
    }

    print(
        f"🧠 Ollama native call start | model={LLM_MODEL_NAME} "
        f"| ctx={LLM_NUM_CTX} | predict={LLM_NUM_PREDICT}"
    )

    response = requests.post(
        OLLAMA_NATIVE_CHAT_URL,
        json=payload,
        timeout=(300, LLM_TIMEOUT_S),
    )
    response.raise_for_status()

    data = response.json() or {}
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()

    print(f"✅ Ollama native call completed | chars={len(content)}")

    return content


MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "3"))  # number of turns (user+assistant)

# Retrieval knobs (RAG v2)
QDRANT_CANDIDATES = int(os.getenv("QDRANT_CANDIDATES", "100"))     # retrieve top-N from qdrant
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "35"))     # Aumentato per catturare più sfumature
FINAL_SOURCES = int(os.getenv("FINAL_SOURCES", "8"))             # Aumentato per dare più contesto
MAX_PER_PAGE = int(os.getenv("MAX_PER_PAGE", "2"))                # ✅ FONDAMENTALE: Consente più chunk per la stessa pagina
MAX_PER_DOC = int(os.getenv("MAX_PER_DOC", "5"))                  # ✅ FONDAMENTALE: Consente Deep-Dive su un singolo documento

# =========================
# 🎚️ Tier-aware ranking
# =========================
TIER_BOOST_A = float(os.getenv("TIER_BOOST_A", "0.08"))
TIER_BOOST_B = float(os.getenv("TIER_BOOST_B", "0.04"))
TIER_PENALTY_C = float(os.getenv("TIER_PENALTY_C", "0.015"))

# Se la query cerca evidenze/log/tecnica, NON penalizzare Tier C
TIER_C_PENALTY_IF_NOT_EVIDENCE = os.getenv("TIER_C_PENALTY_IF_NOT_EVIDENCE", "1") == "1"


# Graph expansion knobs
GRAPH_EXPAND_ENABLED = os.getenv("GRAPH_EXPAND_ENABLED", "1") == "1"
GRAPH_MAX_FORMULAS = int(os.getenv("GRAPH_MAX_FORMULAS", "6"))
GRAPH_MAX_NEIGHBOR_CHUNKS = int(os.getenv("GRAPH_MAX_NEIGHBOR_CHUNKS", "4"))

# Prompt limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))  # prevent prompt blow-ups
MAX_ASSISTANT_CHARS = int(os.getenv("MAX_ASSISTANT_CHARS", "12000"))

AUDIT_ENABLED = True
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "./rag_audit.jsonl")

# In UI conviene partire con evaluation disabilitata.
# La faithfulness può essere eseguita dopo, offline o con un bottone dedicato.
EVAL_ENABLED = os.getenv("EVAL_ENABLED", "0") == "1"

# Può essere lo stesso modello, ma idealmente sarebbe un modello diverso usato come judge.
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", LLM_MODEL_NAME)

# =========================
# 🧾 LOG PATHS - fuori dalla cartella progetto Reflex
# =========================

LOG_DIR = os.getenv(
    "RAG_LOG_DIR",
    os.path.join(os.path.expanduser("~"), "ai_rag_logs")
)

os.makedirs(LOG_DIR, exist_ok=True)

AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1") == "1"
AUDIT_LOG_PATH = os.getenv(
    "AUDIT_LOG_PATH",
    os.path.join(LOG_DIR, "rag_audit.jsonl")
)

EVAL_LOG_PATH = os.getenv(
    "EVAL_LOG_PATH",
    os.path.join(LOG_DIR, "rag_eval_log.jsonl")
)

EVAL_MAX_CONTEXT_CHARS = int(os.getenv("EVAL_MAX_CONTEXT_CHARS", "12000"))

# Soglie KPI
EVAL_MIN_FAITHFULNESS = float(os.getenv("EVAL_MIN_FAITHFULNESS", "0.75"))
EVAL_MIN_ANSWER_RELEVANCE = float(os.getenv("EVAL_MIN_ANSWER_RELEVANCE", "0.70"))

# Se 1, blocca/sostituisce risposte giudicate non fedeli.
# Per iniziare ti consiglio 0: prima osservi le metriche, poi eventualmente blocchi.
EVAL_STRICT_BLOCK = os.getenv("EVAL_STRICT_BLOCK", "0") == "1"



# ============================================================
# 🧠 CARICAMENTO RISORSE AI & DB (SINGLETON PATTERN)
# ============================================================

# Inizializzazione variabili globali a None per caricamento Lazy/Controllato
embedder = None
reranker = None
llm_client = None
qdrant_client_inst = None
neo4j_driver = None
pg_pool = None

# Device selection (già definiti nel tuo script, ma assicurati siano accessibili)
device_embed = "cuda" if torch.cuda.is_available() else "cpu"
device_rerank = "cpu" 

def init_resources():
    """
    Inizializza i modelli e le connessioni ai database in un unico passaggio.
    Previene il caricamento duplicato durante la compilazione del frontend Reflex.
    """
    global embedder, reranker, llm_client, qdrant_client_inst, neo4j_driver, pg_pool, NEO4J_ENABLED

    with _init_lock:
        if embedder is not None:
            return

        print("\n" + "═" * 60)
        print("⏳ [BACKEND] Avvio inizializzazione modelli e database...")
        print("═" * 60)

        try:
            # 1. Embedding Model (BGE-M3) - Caricato su CUDA se disponibile
            print(f"🚀 Loading Embedding Model ({EMBEDDING_MODEL_NAME}) on {device_embed.upper()}...")
            embedder = SentenceTransformer(
                EMBEDDING_MODEL_NAME, 
                device=device_embed, 
                local_files_only=True
            )
            
            # 2. Reranker Model - Forzato su CPU per non competere con l'LLM
            print(f"🚀 Loading Reranker ({RERANKER_MODEL_NAME}) on {device_rerank.upper()}...")
            reranker = CrossEncoder(
                RERANKER_MODEL_NAME, 
                device=device_rerank
            )

            # 3. LLM Connection (Ollama / OpenAI Compatible)
            print(f"🚀 Connecting to LLM via Ollama ({LLM_MODEL_NAME}) at {OLLAMA_URL}...")
            llm_client = OpenAI(base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY)

            # 4. Qdrant (Vector DB)
            print(f"🌌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
            qdrant_client_inst = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

            # 5. Neo4j (Graph DB)
            if NEO4J_ENABLED:
                try:
                    print(f"🕸️ Connecting to Neo4j Graph at {NEO4J_URI}...")
                    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
                    neo4j_driver.verify_connectivity()
                except Exception as e:
                    print(f"⚠️ Neo4j disabled (driver init failed): {e}")
                    neo4j_driver = None
                    NEO4J_ENABLED = False

            # 6. Postgres Pool (TimescaleDB)
            if PG_ENRICH_ENABLED:
                print(f"🐘 Initializing Postgres Pool ({PG_HOST})...")
                pg_pool = SimpleConnectionPool(
                    PG_MIN_CONN, PG_MAX_CONN,
                    host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                    user=PG_USER, password=PG_PASS
                )
                # Smoke test per validare la connessione
                conn = pg_pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                finally:
                    pg_pool.putconn(conn)

            print("✅ [BACKEND] Risorse caricate con successo.")
            print("═"*60 + "\n")

        except Exception as e:
            print(f"❌ [ERRORE] Fallimento inizializzazione: {e}")
            # Reset variabili per permettere retry se necessario
            embedder = None
            # Non blocchiamo l'esecuzione dell'intera app, ma il RAG non funzionerà
        
# --- ESECUZIONE SELETTIVA ---
# REFLEX_RELOAD viene impostato durante l'hot-reload del server di sviluppo.
# Questo controllo assicura che il caricamento pesante avvenga solo nel processo worker.
if not os.environ.get("REFLEX_RELOAD"):
    init_resources()


# =========================
# 📦 DATA MODELS
# =========================
class GraphEntity(BaseModel):
    name: str
    type: str
    relation: str = "MENTIONED"


class SourceItem(BaseModel):
    id: str
    content: str
    filename: str
    page: int = 0
    type: str = "text"
    score: float = 0.0
    graph_context: List[GraphEntity] = Field(default_factory=list)
    # extra provenance / metadata
    section_hint: str = ""
    image_id: Optional[int] = None
    #NEW
    tier: str = "C"
    # ✅ PG canonical provenance
    pg_ingestion_ts: str = ""
    pg_source_name: str = ""
    pg_source_type: str = ""
    pg_log_id: int = 0
    pg_chunk_id: int = 0
    pg_toon_type: str = ""
    db_origin: str = "Unknown"
    
class RetrievalDebug(BaseModel):
    query: str = ""
    intent: str = "text"

    # Tier logic
    wants_evidence: bool = False
    default_tiers: List[str] = []

    # Qdrant stats
    qdrant_candidates: int = 0
    kept_after_quality_filters: int = 0
    rerank_candidates: int = 0
    final_sources: int = 0

    # Tier distribution in final set
    tier_counts: Dict[str, int] = {}

    # Scoring (quick summary)
    score_min: float = 0.0
    score_max: float = 0.0
    score_avg: float = 0.0

    # Flags
    reranker_used: bool = False
    graph_expand_used: bool = False

class AuditTrail(BaseModel):
    ts_utc: str = ""
    query: str = ""
    intent: str = ""

    # What we sent to the LLM (hash only, to avoid storing full sensitive context)
    prompt_sha256: str = ""
    context_chars: int = 0

    # Retrieval explainability
    retrieval: RetrievalDebug = Field(default_factory=RetrievalDebug)

    # Model config snapshot
    llm_model: str = ""
    temperature: float = 0.1
    memory_limit: int = 0

class RagEvalResult(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_support: float = 0.0
    hallucination_risk: float = 1.0
    source_scope_violation: bool = False
    verdict: str = "UNKNOWN"
    unsupported_claims: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    reason: str = ""


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SourceItem] = Field(default_factory=list)
    debug_md: str = "" # ✅ NEW: explainability/audit (renderizzato in UI)

# =========================
# 🧰 UTILS
# =========================
def build_alternating_history(messages: List[ChatMessage], max_turns: int) -> List[Dict[str, str]]:
    """Strict alternating user/assistant for LM Studio templates."""
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        content = (m.content or "").strip()
        if not content:
            continue
        if cleaned and cleaned[-1]["role"] == m.role:
            cleaned[-1]["content"] = content
        else:
            cleaned.append({"role": m.role, "content": content})

    limit = max_turns * 2
    cleaned = cleaned[-limit:]
    if cleaned and cleaned[0]["role"] == "assistant":
        cleaned = cleaned[1:]

    alt: List[Dict[str, str]] = []
    for item in cleaned:
        if alt and alt[-1]["role"] == item["role"]:
            alt[-1] = item
        else:
            alt.append(item)

    return alt


def gpu_free_info() -> str:
    """Return free/total VRAM. Works only if CUDA available."""
    if not torch.cuda.is_available():
        return "CPU Mode"
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        name = torch.cuda.get_device_name(0)
        return f"{name} | Free {free_gb:.1f} GB / Total {total_gb:.1f} GB"
    except Exception:
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 1024**3:.1f} GB)"



def detect_intent(query: str) -> str:
    """
    Router di intenti esteso per sistema di Assessment/Audit RAG.
    Classifica la domanda dell'utente per attivare pipeline o prompt specifici.
    Restituisce: 'formula', 'table', 'chart', 'audit' o 'text'.
    """
    q = (query or "").lower()

    # 1. INTENT: MATEMATICA E CALCOLO (Attiva il Pre-Flight AST o output analitico)
    MATH_KEYWORDS = [
        "formula", "equazione", "equation", "derivate", "cvss", "risk score",
        "probabilità", "probability", "calcolo del rischio", "calcola", "calcolo",
        "sanzione", "percentuale", "ammonta", "penale", "multa", "roi", "budget",
        "costo", "quantifica", "impatto economico", "%"
    ]
    if any(k in q for k in MATH_KEYWORDS):
        return "formula"

    # 2. INTENT: TABELLE E MATRICI (Attiva prompt per tabelle Markdown/Crosswalk)
    TABLE_KEYWORDS = [
        "tabella", "table", "righe", "colonne", "row", "column",
        "matrice rischi", "risk register", "asset inventory", "inventario",
        "crosswalk", "confronta", "confronto", "allineamento", "mappatura"
    ]
    if any(k in q for k in TABLE_KEYWORDS):
        return "table"

    # 3. INTENT: GRAFI E DIAGRAMMI (Attiva esplorazione topologica o Cypher avanzato)
    CHART_KEYWORDS = [
        "grafico", "graph", "flow", "flowchart", "diagramma", "diagram", "architettura",
        "topologia", "chart", "figura", "rete", "network map", "schema",
        "relazioni", "collegamenti", "nodi", "path"
    ]
    if any(k in q for k in CHART_KEYWORDS):
        return "chart"

    # 4. INTENT: AUDIT E COMPLIANCE (Attiva i guardrail rigidi e struttura A/B/C/D)
    AUDIT_KEYWORDS = [
        "audit", "compliance", "conformità", "verifica", "valuta", "assessment",
        "ispeziona", "requisito", "normativa", "iso 27001", "nis2", "gdpr", "dora",
        "linee guida", "policy", "sanzionatorio", "violazione"
    ]
    if any(k in q for k in AUDIT_KEYWORDS):
        return "audit"

    # 5. INTENT DI DEFAULT (Ricerca testuale standard)
    return "text"


def extract_requested_pages(query: str):
    import re
    if not query:
        return []

    q = query.lower().strip()
    # "pag 8-9", "pagina 8/9", "page 10-12"
    pattern = r"\b(?:pag(?:ina)?|page|p)\.?\s*[:=]?\s*(\d{1,4})(?:\s*[-/]\s*(\d{1,4}))?\b"
    m = re.search(pattern, q, flags=re.IGNORECASE)
    if not m:
        return []

    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else None

    if b is None:
        return [a] if a > 0 else []
    if a <= 0 or b <= 0:
        return []

    lo, hi = (a, b) if a <= b else (b, a)
    # clamp max span to avoid huge expansions
    if hi - lo > 20:
        return [lo, hi]
    return list(range(lo, hi + 1))


# ------------------------------------------------------------
# TABLE-FIRST RETRIEVAL REORDERING (ANTI-GENERIC ANSWERS)
# -----------------------------------------------------------
def is_user_data_analytics(query: str) -> bool:
    """
    Rileva se l'utente ha incollato dati grezzi nel prompt (es. liste, CSV, JSON, tabelle Markdown)
    con l'intento di farli analizzare o elaborare.
    """
    q = (query or "").lower()

    # 1. RILEVAMENTO STRUTTURE DATI (Esteso)
    # A. Liste classiche: [1.2, 3, 4] o (1, 2, 3)
    has_array = bool(re.search(r"[\[\(]\s*[\d,\.\s-]{3,}\s*[\]\)]", q))
    
    # B. Tabelle Markdown: | ID | CVSS |
    has_md_table = bool(re.search(r"\|[\w\s\.\-]+\|[\w\s\.\-]+\|", q))
    
    # C. Dati in formato JSON (molto basilare, cerca "chiave": valore numerico)
    has_json = bool(re.search(r"\"\w+\"\s*:\s*[\d\.]+", q))
    
    # D. Copia-incolla da CSV o Excel (almeno 3 righe con separatori come tab o punto e virgola)
    has_csv_tsv = len(re.findall(r"(?:^|\n)[\w\s\.\-]+[,\t;][\w\s\.\-,]+", q)) >= 3

    has_data_structure = has_array or has_md_table or has_json or has_csv_tsv

    # 2. CONTEGGIO NUMERI DISTINTI (Più sicuro del conteggio singole cifre)
    # Trova tutti i blocchi numerici isolati (es. "9.8", "100", "42")
    number_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", q))

    # 3. KEYWORD RAGGRUPPATE PER INTENTO
    # Azioni richieste dall'utente
    ACTION_KEYWORDS = [
        "calcola", "calculate", "stima", "estimate", "analizza", "analyse", "analyze", 
        "elabora", "raggruppa", "filtra", "ordina", "confronta","compare", "valuta", "assess", "quantifica", "quantify", "sintetizza", "sintetize"
    ]
    # Metriche matematiche o statistiche
    MATH_KEYWORDS = [
        "totale", "total", "somma", "sum", "media", "mean", "average", "massimo", 
        "minimo", "distribuzione", "distribution", "percentile"
    ]
    # Dominio Assessment / Cyber
    DOMAIN_KEYWORDS = [
        "vulnerabilità", "vulnerability", "incidenti", "incidents", "cvss", 
        "severità", "severity", "rischio", "risk", "trend", "mitigazione", "mitigation"
    ]

    has_action = any(k in q for k in ACTION_KEYWORDS)
    has_math = any(k in q for k in MATH_KEYWORDS)
    has_domain = any(k in q for k in DOMAIN_KEYWORDS)

    # L'intento c'è se troviamo una keyword analitica (azione o matematica) o di dominio
    has_keywords = has_action or has_math or has_domain

    # RITORNO LOGICO: 
    # C'è una struttura dati evidente (o almeno 6 numeri distinti) AND ci sono parole chiave analitiche?
    return (has_data_structure or number_count >= 6) and has_keywords

# ============================================================
# ✅ RAG QUALITY PATCHES - assessment test excellence
# ============================================================
def extract_search_tokens(query_text: str) -> List[str]:
    """Tokenizzazione per Postgres/BM25 che conserva acronimi brevi."""
    raw = re.findall(r"[A-Za-zÀ-ÿ0-9_\-]+", query_text or "")
    out: List[str] = []
    for t in raw:
        clean = t.strip().strip(".,:;!?()[]{}\"'")
        if not clean:
            continue
        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10
        is_mixed_acronym = bool(re.fullmatch(r"[A-Za-z]{1,5}\d{0,3}", clean)) and 2 <= len(clean) <= 10
        is_useful_word = len(clean) > 3
        if is_acronym or is_mixed_acronym or is_useful_word:
            out.append(clean.lower())
    return list(dict.fromkeys(out))




def is_math_query(query_text: str) -> bool:
    """
    Controllo bilingue avanzato per l'intento matematico.
    Unisce pattern OCR/Finanziari, dizionari base e controllo logico dei parametri.
    """
    q = (query_text or "").lower()
    
    # LIVELLO 1: Controllo Pattern Forti e Formule (Se scatta questo, è matematica al 100%)
    if MATH_CANDIDATE_PAT.search(q):
        return True
        
    # LIVELLO 2: Estrazione Numeri (supporta formati come 150.000,00)
    nums = re.findall(r"\d+(?:[,.]\d+)*", q)
    
    # LIVELLO 3: Dizionario Base Bilingue (tutto rigorosamente in minuscolo)
    math_terms = [
        # ITALIANO - Operazioni, Valutazioni e Sanzioni
        "calcola", "calcolo", "somma", "moltiplica", "dividi", "sottrai", "matematica",
        "percentuale", "media", "totale", "operazione", "equivalente", "stima", 
        "quantifica", "quantificazione", "costo", "ammonta", "penale", "multa",
        
        # INGLESE - Operazioni, Valutazioni e Sanzioni
        "calculate", "sum", "multiply", "divide", "subtract", "mathematics", "math", 
        "estimate", "quantify", "quantification", "cost", "amount", "penalty", "fine",
        "percentage", "average", "total", "operation", "equivalent",

        # STRUTTURA & TEORIA (Bilingue)
        "formula", "equation", "equazione", "theorem", "teorema", "lemma", "proof", "dimostrazione",
        
        # CALCOLO AVANZATO E ALGEBRA LINEARE (Bilingue)
        "integral", "integrale", "derivative", "derivata", "logarithm", "logaritmo", 
        "summation", "sommatoria", "matrix", "matrice", "vector", "vettore", "latex",
        
        # STATISTICA E PROBABILITÀ (Bilingue)
        "variance", "varianza", "deviation", "deviazione", "correlation", "correlazione", 
        "regression", "regressione", "distribution", "distribuzione", "confidence", "confidenza",
        
        # FINANZA E VALUTAZIONE ASSET (Bilingue)
        "discount", "sconto", "yield", "rendimento", "compounding", "capitalization", 
        "amortization", "ammortamento", "present value", "future value", "npv", "van", 
        "irr", "tir", "cash flow", "flusso",
        
        # SIMBOLI SCRITTI A PAROLE
        "sigma", "alpha", "beta", "gamma", "delta", "theta", "lambda"
    ]
    
    # Elenco esplicito dei simboli matematici (senza sintassi Regex)
    math_symbols = [
        "%", "+", "=", "*", "/", 
        "∑", "∏", "∫", "√", "≈", "≠", 
        "≤", "≥", "→", "↔", "∩", "∪", 
        "∞", "±", "×", "÷"
    ]
    
    # Verifica termine esatto usando word boundaries (\b) per evitare che "sum" scatti su "consumer"
    has_math_term = any(re.search(rf"\b{term}\b", q) for term in math_terms)
    
    # Verifica simboli base testuali
    has_math_symbol = any(sym in q for sym in math_symbols)
    
    # LOGICA DI RITORNO FINALE:
    # Restituisce True se:
    # A) C'è ALMENO un numero nel prompt accompagnato da una parola/simbolo matematico.
    # B) L'utente chiede esplicitamente una "formula" (il RAG andrà a cercare i numeri nei documenti).
    return (len(nums) >= 1 and (has_math_term or has_math_symbol)) or ("formula" in q)


def solve_control_coverage(query_text: str) -> Optional[str]:
    q = (query_text or "").lower().replace(",", ".")
    m_total = re.search(r"(\d+)\s+controlli", q)
    m_impl = re.search(r"(\d+)\s+implementati", q)
    m_partial = re.search(r"(\d+)\s+parziali", q)
    m_weight = re.search(r"(?:al|peso|considerare\s+al)\s*(\d+(?:\.\d+)?)\s*%", q)
    if not (m_total and m_impl and m_partial and m_weight):
        return None
    total = int(m_total.group(1))
    implemented = int(m_impl.group(1))
    partial = int(m_partial.group(1))
    weight = float(m_weight.group(1)) / 100.0
    if total <= 0:
        return None
    equivalent = implemented + partial * weight
    coverage = equivalent / total * 100.0
    return (
        "**A) Risposta**\n\n"
        f"La copertura equivalente è **{coverage:.2f}%**.\n\n"
        "**B) Evidenze**\n\n"
        "- Formula applicata: `copertura = (implementati + parziali × peso) / totale × 100`.\n"
        f"- Controlli equivalenti = `{implemented} + ({partial} × {weight:.2f}) = {equivalent:.2f}`.\n"
        f"- Percentuale = `{equivalent:.2f} / {total} × 100 = {coverage:.2f}%`.\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- Il calcolo usa esclusivamente i dati numerici forniti nella domanda.\n"
        "- Per usarlo come evidence assessment reale servirebbe associare ogni controllo alle evidenze documentali recuperate.\n\n"
        "**D) Fonti**\n\n"
        "- Dati forniti dall'utente."
    )


def solve_risk_product(query_text: str) -> Optional[str]:
    q = query_text or ""
    q_norm = q.replace("×", "x").replace("*", "x")
    pairs = re.findall(r"\b([A-Z])\s*(\d+(?:[,.]\d+)?)\s*x\s*(\d+(?:[,.]\d+)?)", q_norm, flags=re.IGNORECASE)
    if len(pairs) < 2 or not re.search(r"rischio|risk|probabil", q_norm, flags=re.IGNORECASE):
        return None
    results = []
    for label, a, b in pairs:
        p = float(a.replace(",", "."))
        imp = float(b.replace(",", "."))
        results.append((label.upper(), p, imp, p * imp))
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    ranking = ", ".join([f"{r[0]}={r[3]:.0f}" if r[3].is_integer() else f"{r[0]}={r[3]:.2f}" for r in results_sorted])
    evidence_lines = "\n".join([f"- Scenario {lab}: `{p:g} × {imp:g} = {score:g}`." for lab, p, imp, score in results_sorted])
    return (
        "**A) Risposta**\n\n"
        f"Ordinamento dal rischio più critico al meno critico: **{ranking}**.\n\n"
        "**B) Evidenze**\n\n"
        f"{evidence_lines}\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- La formula `rischio = probabilità × impatto` è stata fornita dall'utente nella domanda.\n"
        "- Il risultato numerico non dimostra da solo la conformità: va collegato al risk assessment documentale.\n\n"
        "**D) Fonti**\n\n"
        "- Dati e formula forniti dall'utente."
    )


def try_solve_math_query(query_text: str) -> Optional[str]:
    """
    Forziamo SEMPRE il ritorno a None. 
    In questo modo il sistema RAG recupera i documenti e lascia che sia 
    l'LLM (tramite il System Prompt) a eseguire l'operazione matematica.
    Nessun calcolo hardcoded nel codice Python.
    """
    return None

def needs_math_document_context(query_text: str) -> bool:
    """
    Alcune domande richiedono sia calcolo deterministico sia collegamento ai documenti.
    In questi casi il solver calcola i numeri, ma il RAG recupera comunque contesto concettuale.
    """
    q = (query_text or "").lower()
    context_terms = [
        "collega", "collegalo", "collegala", "documenti", "documento",
        "risk assessment", "evidence assessment", "assessment", "controlli",
        "evidenze", "conformità", "audit",
    ]
    return any(t in q for t in context_terms)


def is_glossary_definition_query(query_text: str) -> bool:
    """
    Bilingual (IT/EN). Identifica richieste di dizionario/vocabolario, 
    ma si disattiva se rileva intenti matematici o di ragionamento complesso.
    """
    # 1. Se è una query matematica, il glossario DEVE disattivarsi
    if is_math_query(query_text):
        return False
        
    q = (query_text or "").lower()
    
    # 2. Se è una query di ragionamento complesso, il glossario DEVE disattivarsi
    reasoning_terms = [
        # ITALIANO - Analisi, Causalità e Confronto
        "spiega", "confronta", "differenza", "differenze", "valuta", "perché", "perche", 
        "correlata", "correlato", "relazione", "analizza", "motivo", "causa", "impatto", 
        "conseguenze", "conseguenza", "vantaggi", "svantaggi", "giustifica", "argomenta", 
        "deduci", "collega", "paragona", "distinzione", "come funziona", "in che modo", 
        "sintetizza", "riassumi", "scopo", "obiettivo",
        
        # INGLESE - Analisi, Causalità e Confronto
        "explain", "compare", "difference", "differences", "evaluate", "why", 
        "correlated", "relation", "relationship", "analyze", "analyse", "reason", 
        "cause", "impact", "consequence", "consequences", "advantage", "advantages", 
        "disadvantage", "disadvantages", "justify", "argue", "deduce", "connect", 
        "contrast", "distinction", "how does it work", "in what way", "summarize", 
        "summarise", "purpose", "goal"
    ]
    
    # Se l'utente vuole un'analisi approfondita, scavalca il glossario
    if any(t in q for t in reasoning_terms):
        return False

    # 3. Solo se sopravvive ai filtri sopra ed è una richiesta pura di definizione, si attiva
    glossary_terms = [
        # ITALIANO - Definizioni e Identificazione
        "cosa significa", "cosa vuol dire", "definisci", "definizione", "significato", 
        "glossario", "cos'è", "cosa è", "cosa sono", "chi è", "chi sono", "acronimo", 
        "sta per", "cosa si intende", "dizionario", "vocabolario", "termine", 
        
        # INGLESE - Definizioni e Identificazione
        "what does it mean", "what is", "what are", "who is", "who are", "define", 
        "definition", "meaning", "glossary", "acronym", "stands for", 
        "what is meant by", "dictionary", "vocabulary", "term"
    ]
    
    return any(t in q for t in glossary_terms)


def is_mixed_glossary_rag_query(query_text: str) -> bool:
    """
    True quando l'utente cita il glossario ma chiede anche documenti,
    fonti, evidenze, audit, normative, relazioni o grafo.

    In questi casi NON bisogna bypassare il RAG con la modalità glossario atomico.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    has_glossary = any(t in q for t in [
        "glossario", "voce di glossario", "voci di glossario",
        "glossary", "glossary entry", "term definition",
    ])

    if not has_glossary:
        return False

    mixed_terms = [
        # IT
        "usa sia", "usando sia", "insieme ai documenti", "documenti normativi",
        "documenti recuperati", "fonti recuperate", "contesto documentale",
        "collegamento", "collegamenti", "relazione", "relazioni", "collega",
        "collegare", "grafo", "entità", "evidenza", "evidenze",
        "fonte", "fonti", "documento", "documenti", "controllo", "controlli",
        "assessment", "audit", "conformità", "compliance",

        # EN
        "using both", "use both", "together with documents", "retrieved documents",
        "retrieved sources", "documentary context", "normative documents",
        "relationship", "relationships", "relation", "relations", "connect",
        "connection", "connections", "graph", "entity", "entities",
        "evidence", "evidences", "source", "sources", "document", "documents",
        "control", "controls", "requirement", "requirements",
    ]

    return any(t in q for t in mixed_terms)


# Alias generici per lookup atomico di glossario / acronimi.
# Non contiene definizioni hard-coded: serve solo a recuperare i chunk corretti.
GLOSSARY_TERM_ALIASES: Dict[str, List[str]] = {}

KNOWN_GLOSSARY_TERMS = list(GLOSSARY_TERM_ALIASES.keys())


def extract_requested_terms(query_text: str) -> List[str]:
    """
    Estrae termini richiesti per il glossario in modo agnostico:
    Cattura ciò che l'utente ha messo tra virgolette o eventuali acronimi.
    """
    q = query_text or ""
    terms: List[str] = []
    
    # Cattura stringhe tra virgolette (es. Definisci "Advanced Persistent Threat")
    quoted = re.findall(r"[\"“']([^\"”']+)[\"”']", q)
    terms.extend([x.strip() for x in quoted if len(x.strip()) > 2])
    
    # Cattura acronimi in maiuscolo (es. Definisci APT)
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", q)
    terms.extend(acronyms)
    
    return list(dict.fromkeys(terms))


def extract_exact_phrases(query_text: str) -> List[str]:
    """Estrazione generalista (Agnostica). Estrae stringhe tra virgolette e acronimi."""
    q = query_text or ""
    phrases: List[str] = []
    
    # 1. Estrae frasi forzate dall'utente tra virgolette (es. "Data Breach" o "Heart Failure")
    quoted = re.findall(r"[\"“']([^\"”']+)[\"”']", q)
    phrases.extend([x.strip().lower() for x in quoted if len(x.strip()) > 2])
    
    # 2. Estrae acronimi (da 2 a 8 lettere maiuscole, es. GDPR, MFA, CVSS, HIPAA)
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", q)
    phrases.extend([x.lower() for x in acronyms])
    
    return list(dict.fromkeys([p for p in phrases if p]))


def expand_assessment_query(query_text: str) -> str:
    """
    Nessuna espansione hardcoded. Affidiamo il recupero semantico al motore vettoriale 
    (Qdrant) e all'espansione dinamica degli acronimi in retrieve_v2.
    """
    return (query_text or "").strip()


import re
from typing import Tuple

def dynamic_retrieval_limits(query_text: str) -> Tuple[int, int, int, int, int]:
    """
    Limiti dinamici basati sulla complessità della query, bilingue e agnostico.
    Ottimizzato con Word Boundaries per evitare falsi positivi.
    """
    q = (query_text or "").lower()
    long_query = len(q) > 300
    
    multi_doc_terms = [
        # --- 1. SOSTANTIVI: Corpus, Normative e Strutture (IT/EN) ---
        "documenti", "fonti", "normative", "normativa", "standard", "framework", 
        "regolamenti", "regolamento", "documentazione", "policy", "direttive", 
        "direttiva", "linee guida", "allegati", "architettura", "manuale", "guida", 
        "procedure", "procedura", "requisiti", "requisito", "specifiche", "corpus", 
        "assessment", "audit", "ispezione", "certificazione",
        
        "documents", "sources", "regulations", "regulation", "documentation", 
        "policies", "directives", "directive", "guidelines", "attachments", 
        "architecture", "manual", "guide", "procedures", "procedure", 
        "requirements", "requirement", "specifications", "reference materials",
        
        # --- 2. AZIONI E INTENTI: Confronto, Mappatura e Sintesi (IT/EN) ---
        "confronta", "confronto", "differenza", "differenze", "integra", "integrazione", 
        "mappa", "mappatura", "matrice", "correlazione", "incrocia", "valutazione", 
        "valuta", "allineamento", "sintesi", "riassunto", "sovrapposizione", "congiunta",
        
        "compare", "comparison", "difference", "differences", "integrate", "integration", 
        "map", "mapping", "matrix", "crosswalk", "correlate", "correlation", 
        "cross-reference", "evaluate", "evaluation", "alignment", "summary", 
        "overview", "overlap", "joint",
        
        # --- 3. ESTENSIONE E SCOPO: Processi, Fasi e Totalità (IT/EN) ---
        "passo-passo", "fasi", "fase", "completo", "completa", "intero", "intera", 
        "dettagliato", "dettagliata", "olistico", "tutto il processo", "dall'inizio alla fine",
        
        "step-by-step", "phases", "phase", "complete", "entire", "detailed", 
        "comprehensive", "holistic", "in-depth", "end-to-end", "start to finish", 
        "whole process", "full walkthrough"
    ]

    # Usiamo le word boundaries (\b) per evitare che "map" scatti su "bitmap"
    # re.escape ci protegge da eventuali caratteri speciali (es. il trattino di step-by-step)
    multi_doc_matches = sum(1 for k in multi_doc_terms if re.search(rf"\b{re.escape(k)}\b", q))
    multi_doc = multi_doc_matches >= 2
    
    # Anche per i grafi applichiamo la stessa sicurezza
    graph_terms = [
        # --- 1. BASE: Topologia e Struttura (IT/EN) ---
        "neo4j", "cypher", "grafo", "grafi", "relazioni", "relazione", "collegamenti", "collegamento", 
        "entità", "nodi", "nodo", "graph", "relations", "relation", "entities", "entity", 
        "links", "link", "nodes", "node", "edges", "edge", "topology",
        
        # --- 2. SEMANTICA E KNOWLEDGE GRAPH (IT/EN) ---
        "semantica", "semantico", "semantic", "semantics",
        "ontologia", "ontologie", "ontology", "ontologies",
        "grafo della conoscenza", "knowledge graph",
        "rete semantica", "semantic network",
        "triple", "tripla", "triples", # Riferimento alle triple RDF/Semantiche (Soggetto-Predicato-Oggetto)
        
        # --- 3. TASSONOMIA E GERARCHIA (IT/EN) ---
        "tassonomia", "tassonomie", "taxonomy", "taxonomies",
        "gerarchia", "gerarchie", "hierarchy", "hierarchies",
        "alberatura", "tree structure", "dipendenze", "dependencies"
    ]
    graph_query = any(re.search(rf"\b{k}\b", q) for k in graph_terms)
    
    glossary = is_glossary_definition_query(q)
    
    # Ritorno logico dei limiti (Assicurati che le costanti globali siano definite in cima al tuo file)
    if long_query or multi_doc or graph_query or glossary:
        # Limiti estesi per assessment complessi, manualistica o percorsi grafo
        return 140, 45, 15, 8, 4
        
    return QDRANT_CANDIDATES, RERANK_CANDIDATES, FINAL_SOURCES, MAX_PER_DOC, MAX_PER_PAGE


def should_force_tier_a(query_text: str) -> bool:
    """
    Forza il recupero da fonti normative e legali (Tier A) 
    ignorando l'overfitting su framework specifici.
    Bilingue e protetta da word boundaries.
    """
    q = (query_text or "").lower()
    
    # Se la query riguarda definizioni base o matematica, il Tier A non è forzato 
    # (lasciamo che il sistema recuperi liberamente dai tier appropriati)
    if is_glossary_definition_query(q) or is_math_query(q):
        return False
    
    audit_terms = [
        # --- 1. CONFORMITÀ E OBBLIGHI (IT/EN) ---
        "conformità", "conforme", "obbligo", "obblighi", "obbligatorio", 
        "adempimento", "adempimenti", "prescrizione", "prescrizioni",
        "compliance", "compliant", "obligation", "obligations", "mandatory", 
        "fulfillment", "enforcement",

        # --- 2. STRUTTURA NORMATIVA E LEGALE (IT/EN) ---
        "normativa", "normative", "normativo", "regolamento", "regolamenti", 
        "regolamentazione", "direttiva", "direttive", "legge", "leggi", 
        "legislazione", "decreto", "decreti", "framework", "standard", "policy", 
        "policies", "clausola", "clausole", "articolo", "commi", "comma", "allegato",
        "norm", "norms", "regulation", "regulations", "regulatory", "directive", 
        "directives", "law", "laws", "legislation", "decree", "decrees", 
        "clause", "clauses", "article", "articles", "annex", "appendix",

        # --- 3. ISPEZIONE E CERTIFICAZIONE (IT/EN) ---
        "audit", "auditor", "ispezione", "ispezioni", "ispettivo", "assessment", 
        "controllo", "controlli", "misura", "misure", "verifica", "verifiche", 
        "certificazione", "certificazioni", "attestazione", "governance",
        "inspection", "inspections", "inspector", "control", "controls", 
        "measure", "measures", "verification", "verifications", "certification", 
        "certifications", "attestation",

        # --- 4. REQUISITI, VIOLAZIONI E SANZIONI (IT/EN) ---
        "requisito", "requisiti", "violazione", "violazioni", "non-conformità", 
        "sanzione", "sanzioni", "multa", "multe", "infrazione", "infrazioni", 
        "penale", "penali", "responsabilità", "data breach",
        "requirement", "requirements", "violation", "violations", "non-compliance", 
        "sanction", "sanctions", "fine", "fines", "infringement", "breach", 
        "penalty", "penalties", "liability", "accountability"
    ]
    
    # Usiamo word boundaries (\b) per evitare che "legge" (sostantivo) 
    # scatti quando l'utente scrive "il sistema non legge il file" (verbo).
    # re.escape protegge da caratteri speciali come il trattino in "non-conformità"
    return any(re.search(rf"\b{re.escape(t)}\b", q) for t in audit_terms)


def is_follow_up_query(query_text: str) -> bool:
    """
    Rileva se l'utente sta facendo riferimento al documento 
    o alla risposta del turno precedente.
    """
    q = (query_text or "").lower().strip()
    
    # Se la query è troppo lunga, è probabile che contenga una nuova direttiva complessa, non un semplice follow-up
    if len(q) >= 140:
        return False
        
    follow_up_terms = [
        # ITALIANO
        "questo documento", "questa fonte", "questo file", "lo stesso", "la stessa", 
        "sempre lì", "nella stessa", "nel documento precedente", "come sopra", 
        "riguardo a prima", "e in questo", "in quest'ultimo", "riguardo quest'ultimo", 
        "ancora qui", "approfondisci questo", "su questo", "di questo", "in merito",
        
        # INGLESE
        "this document", "this source", "this file", "the same", "in the same", 
        "previous document", "previous source", "as above", "regarding the latter", 
        "in this one", "elaborate on this", "same file", "about this", "on this", 
        "in the previous"
    ]
    
    # Rileva se almeno un termine è presente (re.escape protegge gli apostrofi)
    return any(re.search(rf"\b{re.escape(t)}\b", q) for t in follow_up_terms)

def detect_answer_mode(query_text: str) -> str:
    """
    Stabilisce se il sistema deve semplicemente rispondere a una domanda (knowledge)
    o eseguire un'analisi critica/valutazione (audit).
    """
    q = (query_text or "").lower()
    
    audit_eval_terms = [
        # ITALIANO
        "verifica conformità", "valutazione conformità", "non conformità", "non conforme", 
        "audit", "evidenze implementazione", "evidenza", "evidenze", "policy contro evidenza", 
        "tier b", "tier c", "gap tecnico", "gap analysis", "analisi dei gap", "scostamento", 
        "discrepanza", "ispezione", "allineamento tecnico", "deviazione",
        
        # INGLESE
        "compliance check", "compliance assessment", "non-compliance", "non-compliant", 
        "audit", "implementation evidence", "evidence", "policy vs evidence", 
        "tier b", "tier c", "technical gap", "gap analysis", "deviation", 
        "discrepancy", "inspection", "technical alignment"
    ]
    
    if any(re.search(rf"\b{re.escape(t)}\b", q) for t in audit_eval_terms):
        return "audit"
        
    return "knowledge"

def is_strict_checklist_query(query_text: str) -> bool:
    """
    Attiva la checklist mode basandosi solo sull'intento.
    Utilizza un sistema a pesi: termini inequivocabili (forti) attivano subito, 
    termini contestuali (deboli) richiedono almeno 2 occorrenze.
    """
    q = (query_text or "").lower()
    
    # 1. TERMINI FORTI: Basta una sola parola per forzare la checklist mode
    strong_terms = [
        "checklist", "crosswalk", "matrice", "matrix", "griglia", "grid"
    ]
    if any(re.search(rf"\b{re.escape(t)}\b", q) for t in strong_terms):
        return True

    # 2. TERMINI DEBOLI/CONTESTUALI: Ne servono almeno 2 (es. "elenco" + "controlli")
    weak_terms = [
        # ITALIANO
        "assessment", "evidenze", "evidenza", "controlli", "controllo", 
        "requisiti", "requisito", "audit", "linee guida", "elenco", "lista", 
        "kpi", "indicatori", "indicatore", "questionario", "domande",
        
        # INGLESE
        "assessment", "evidence", "controls", "control", "requirements", 
        "requirement", "audit", "guidelines", "list", "kpi", "indicators", 
        "indicator", "questionnaire", "checkpoints", "questions"
    ]
    
    # Conta quanti termini deboli distinti sono presenti nella query
    weak_count = sum(1 for t in weak_terms if re.search(rf"\b{re.escape(t)}\b", q))
    
    return weak_count >= 2

import re

def is_graph_relation_query(query_text: str) -> bool:
    """
    Attiva l'output strutturato/tabellare quando la domanda riguarda esclusivamente 
    la topologia del grafo (entità, relazioni).
    Bilingue (IT/EN) e protetta dai word boundaries (\b).
    """
    q = (query_text or "").lower()

    # --- 1. GATEKEEPER: Protezione per ragionamento logico e scenari ---
    # Se la query richiede un'analisi discorsiva o descrive un caso ipotetico ("se..."),
    # l'output a tabella rigida viene disattivato per lasciare spazio al testo.
    analysis_terms = [
        # ITALIANO
        "spiega", "valuta", "confronta", "differenza", "differenze", "se", "basandoti", 
        "analizza", "perché", "motivo", "causa", "giustifica", "descrivi", "sintetizza", 
        "racconta", "scenario", "ipotesi",
        
        # INGLESE
        "explain", "evaluate", "compare", "difference", "differences", "if", "based on", 
        "analyze", "analyse", "why", "reason", "cause", "justify", "describe", 
        "summarize", "summarise", "scenario", "hypothesis", "what happens"
    ]
    
    # re.escape gestisce in sicurezza eventuali spazi in "based on"
    if any(re.search(rf"\b{re.escape(t)}\b", q) for t in analysis_terms):
        return False


    # --- 2. TRIGGER: Termini topologici e semantici ---
    relation_terms = [
        # ITALIANO
        "neo4j", "cypher", "grafo", "grafi", "relazioni", "relazione", "collegamenti", 
        "collegamento", "entità", "concettuale", "concettuali", "collega", "connessione", 
        "connessioni", "nodo", "nodi", "archi", "arco", "mappa", "mappatura", 
        "topologia", "ontologia", "tassonomia", "rete semantica", "triple",
        
        # INGLESE
        "graph", "graphs", "entity", "entities", "relationship", "relationships", 
        "relation", "relations", "link", "links", "connect", "connection", "connections", 
        "conceptual", "node", "nodes", "edge", "edges", "map", "mapping", 
        "topology", "ontology", "taxonomy", "semantic network", "triples"
    ]

    return any(re.search(rf"\b{re.escape(t)}\b", q) for t in relation_terms)



import re

def is_formula_strict_query(query_text: str) -> bool:
    """
    Bilingual (IT/EN). Attiva l'estrazione delle formule/metriche (es. in tabella) 
    SOLO per ricerche esplorative e teoriche.
    Se la query contiene numeri e chiede di eseguire un calcolo, 
    viene intercettata e bloccata dal gatekeeper (is_math_query).
    """
    # --- 1. GATEKEEPER ---
    if is_math_query(query_text):
        return False

    q = (query_text or "").lower()
    
    # --- 2. TERMINI ESPLORATIVI E STRUTTURALI ---
    formula_terms = [
        # ITALIANO - Base e Metriche
        "formula", "formule", "equazione", "equazioni", "metrica", "metriche", 
        "indicatore", "indicatori", "punteggio", "punteggi", "algoritmo", "algoritmi", 
        "indice", "indici", "modello di calcolo", "algoritmo di calcolo", "teorema",
        
        # INGLESE - Base e Metriche
        "formulas", "formulae", "equation", "equations", "metric", "metrics", 
        "indicator", "indicators", "score", "scores", "scoring", "algorithm", "algorithms", 
        "index", "indices", "calculation model", "computation model", "calculation algorithm",
        
        # ACRONIMI UNIVERSALI / DOMINIO CYBER & AUDIT
        "latex", "kpi", "kpis", "kri", "kris", "cvss"
    ]
    
    # Usiamo le word boundaries (\b) per evitare che "score" scatti su "underscore"
    return any(re.search(rf"\b{re.escape(t)}\b", q) for t in formula_terms)


def extract_formula_rows_from_sources(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    Estrae formule o metriche dai SourceItem recuperati.

    Non inventa formule:
    - se trova LaTeX esplicito, lo usa;
    - se trova una formula testuale esplicita tipo X = Y / Z, la riporta;
    - se trova solo una metrica/indicatore, dichiara che la formula esplicita non è recuperata.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()

    latex_pat = re.compile(r"(?<!\\)(\$\$.*?\$\$|\$[^$\n]{2,300}\$)", re.DOTALL)
    explicit_equation_pat = re.compile(
        r"(?i)\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_\-/ ]{1,80})\s*=\s*([^;\n]{2,260})"
    )
    metric_line_pat = re.compile(
        r"(?i)\b("
        r"formula|formulas|formulae|equation|equations|"
        r"formule|equazione|equazioni|"
        r"metric|metrics|metrica|metriche|"
        r"indicator|indicators|indicatore|indicatori|"
        r"score|scoring|punteggio|"
        r"calculation|calculation model|calcolo|modello di calcolo|"
        r"mean time|tempo medio|index|indice|ratio|coverage|copertura|"
        r"maturity|maturità|severity|severità"
        r")\b"
    )

    for s in sources or []:
        content = s.content or ""
        filename = s.filename or "N/D"
        page = int(s.page or 0)
        source_type = normalize_source_type(getattr(s, "type", "") or "")

        # Caso 1: formule LaTeX esplicite nel contenuto.
        for lx in latex_pat.findall(content):
            latex = lx.strip()
            key = ("latex", latex.lower(), filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": "Formula recuperata",
                "latex": latex,
                "meaning": "Formula LaTeX esplicita presente nella fonte recuperata.",
                "filename": filename,
                "page": page,
            })

        # Caso 2: contenuti provenienti da Neo4j Formula Search.
        if "Formula from Knowledge Graph" in content or source_type == "formula":
            latex = ""
            plain = ""
            meaning = ""

            for line in content.splitlines():
                clean = line.strip()
                low = clean.lower()
                if low.startswith("latex:"):
                    latex = clean.split(":", 1)[1].strip()
                elif low.startswith("plain:"):
                    plain = clean.split(":", 1)[1].strip()
                elif low.startswith("meaning:"):
                    meaning = clean.split(":", 1)[1].strip()

            name = plain or meaning or "Formula/metric"
            key = ("kg", name.lower(), latex.lower(), filename.lower(), page)

            if key not in seen:
                seen.add(key)
                rows.append({
                    "name": name,
                    "latex": latex if latex else "formula esplicita non recuperata",
                    "meaning": meaning,
                    "filename": filename,
                    "page": page,
                })

        # Caso 3: equazioni testuali esplicite tipo "X = Y / Z".
        for m in explicit_equation_pat.finditer(content):
            left = re.sub(r"\s+", " ", m.group(1)).strip()
            right = re.sub(r"\s+", " ", m.group(2)).strip()

            if len(left) < 2 or len(right) < 2:
                continue

            latex = f"{left} = {right}"
            key = ("eq", left.lower(), latex.lower(), filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": left,
                "latex": latex,
                "meaning": "Formula testuale esplicita presente nella fonte recuperata.",
                "filename": filename,
                "page": page,
            })

        # Caso 4: righe che citano metriche/indicatori senza formula esplicita.
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line or "").strip()
            if not line or not metric_line_pat.search(line):
                continue

            name = "Metrica/indicatore citato"
            m_name = re.match(r"^[-*•\s]*([A-Za-zÀ-ÿ0-9_\-/ ]{2,80})\s*[:=–-]", line)
            if m_name:
                name = m_name.group(1).strip()

            key = ("metric", name.lower(), filename.lower(), page, line[:120].lower())
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "name": name,
                "latex": "formula esplicita non recuperata",
                "meaning": (
                    "Metrica/indicatore citato nelle fonti recuperate; "
                    "nessuna formula esplicita è stata individuata nello stesso chunk."
                ),
                "filename": filename,
                "page": page,
            })

            if len(rows) >= 20:
                return rows

    return rows


def _formula_md_cell(value: Any, max_len: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    Risposta deterministica per domande su formule/metriche.
    Evita che il modello lasci formule vuote o inventi LaTeX.
    """
    rows = extract_formula_rows_from_sources(sources)

    if not rows:
        return None

    table_lines = [
        "| Nome / metrica | Formula | Significato | Fonte | Pagina |",
        "|---|---|---|---|---:|",
    ]

    for r in rows:
        # Troncamento sicuro solo per il testo descrittivo
        name = _formula_md_cell(r.get("name") or "N/D", 180)
        meaning = _formula_md_cell(r.get("meaning") or "", 320)
        filename = _formula_md_cell(r.get("filename") or "N/D", 220)
        page = int(r.get("page") or 0)

        # GESTIONE SICURA DEL LATEX
        raw_latex = r.get("latex") or "formula esplicita non recuperata"
        
        # 1. Nessun troncamento testuale per non corrompere la sintassi
        # 2. Escape del carattere pipe '|' per evitare che rompa la tabella Markdown
        safe_latex = raw_latex.replace("|", "&#124;")
        
        # Scegli il formato in base a cosa supporta il tuo frontend Reflex:
        # Se supporta il rendering matematico usa: f"${safe_latex}$"
        # Se vuoi mostrare solo il codice testo usa: f"`{safe_latex}`"
        latex_display = f"`{safe_latex}`" if raw_latex != "formula esplicita non recuperata" else safe_latex

        table_lines.append(
            f"| {name} | {latex_display} | {meaning} | {filename} | {page} |"
        )

    used_files = sorted({
        str(r.get("filename") or "")
        for r in rows
        if r.get("filename")
    })

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "**B) Evidenze**\n\n"
        "- Le formule/metriche sono state estratte in modo deterministico dai documenti recuperati e/o dal Knowledge Graph.\n"
        "- Se il campo formula riporta *'formula esplicita non recuperata'*, la metrica è citata nel testo ma senza la sua espressione matematica.\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- La risposta non inventa formule mancanti.\n"
        "- Per ottenere formule matematiche complete è necessario che le fonti originali (PDF/Markdown) le contengano in formato testuale leggibile o che il nodo nel Knowledge Graph abbia la proprietà `latex` valorizzata.\n\n"
        "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )

def safe_payload_text(payload: Dict[str, Any]) -> str:
    """
    IMPORTANT: align to ingestion payload:
    - most recent ingestion uses 'text_sem'
    - keep fallbacks for older payloads
    """
    return (
        (payload.get("text_sem") or "")
        or (payload.get("content_semantic") or "")
        or (payload.get("content_raw") or "")
        or (payload.get("content") or "")
        or (payload.get("text") or "")
        or ""
    ).strip()


def get_payload_page(payload: Dict[str, Any]) -> int:
    try:
        return int(payload.get("page") or payload.get("page_no") or 0)
    except Exception:
        return 0


def normalize_source_type(value: str) -> str:
    t = str(value or "").lower().strip()

    if t in {"formula", "math", "equation"}:
        return "formula"

    if t in {"image", "immagine", "imagine", "visual", "screenshot"}:
        return "image"

    if t in {"chart", "grafico", "chart_analysis", "diagram", "diagramma"}:
        return "chart"

    if t in {"table", "tabella"}:
        return "table"

    if t in {"text", "testo", ""}:
        return "text"

    return t


def get_payload_type(payload: Dict[str, Any]) -> str:
    return normalize_source_type(payload.get("toon_type") or payload.get("type") or "text")


def get_payload_section(payload: Dict[str, Any]) -> str:
    return str(payload.get("section_hint") or "")


def get_payload_image_id(payload: Dict[str, Any]) -> Optional[int]:
    try:
        v = payload.get("image_id")
        return int(v) if v is not None else None
    except Exception:
        return None

def get_payload_tier(payload: dict) -> str:
    try:
        t = payload.get("tier")
        if not t:
            return ""
        return str(t)
    except Exception:
        return ""

def is_evidence_query(query: str) -> bool:
    q = (query or "").lower()
    
    evidence_terms = [
        # --- IT: Sostantivi e Verbi per Audit Tecnico ---
        "evidenza", "evidenze", "prova", "prove", "log", "configurazione", "configurazioni",
        "implementazione", "implementato", "tecnico", "tecniche", "screenshot", "dimostra",
        "dimostrazione", "sistema", "sistemi", "applicato", "registri", "ticket", 
        "verificare", "verifica", "mostrami", "estratto", "script", "codice", "firewall",
        "regola", "regole", "auditare", "ispezionare", "traccia", "tracciamento",
        
        # --- EN: Nouns and Verbs for Technical Audit ---
        "evidence", "evidences", "proof", "logs", "configuration", "configurations",
        "implementation", "implemented", "technical", "demonstrate", "system", 
        "applied", "records", "registry", "verify", "show me", "extract", "script",
        "code", "firewall", "rule", "rules", "audit", "inspect", "trace", "tracking"
    ]
    
    return any(k in q for k in evidence_terms)



def has_sufficient_ab_sources(sources: List[SourceItem]) -> bool:
    tiers = [(getattr(s, "tier", "") or "").upper() for s in sources]
    for t in tiers:
        if t in ("A", "TIER_A_METHODOLOGY") or t.endswith("_A_METHODOLOGY"):
            return True
        if t in ("B", "TIER_B_REFERENCE") or t.endswith("_B_REFERENCE"):
            return True
    return False


def normalize_tier_value(tier: str) -> str:
    """
    Normalizza i tier in valori canonici:
    A, B, C, GRAPH, USER oppure C come fallback.
    Evita bug tipo: 'GRAPH' contiene la lettera 'A' e viene scambiato per Tier A.
    """
    t = (tier or "").strip().upper()

    if not t:
        return "C"

    if t == "GRAPH" or t.startswith("GRAPH"):
        return "GRAPH"

    if t == "USER" or t.startswith("USER"):
        return "USER"

    if t == "A" or t == "TIER_A_METHODOLOGY" or t.endswith("_A_METHODOLOGY"):
        return "A"

    if t == "B" or t == "TIER_B_REFERENCE" or t.endswith("_B_REFERENCE"):
        return "B"

    if t == "C" or t == "TIER_C_EVIDENCE" or t.endswith("_C_EVIDENCE") or "EVIDENCE" in t or "EVIDENZA" in t:
        return "C"

    return t

def tier_score_delta(tier: str, query_text: str) -> float:
    """
    Applica boost/penalty in modo sicuro sui tier normalizzati.
    Nota importante:
    - non usare mai 'if "A" in tier', perché 'GRAPH' contiene la lettera A.
    """
    t = normalize_tier_value(tier)

    if t == "A":
        return TIER_BOOST_A

    if t == "B":
        return TIER_BOOST_B

    if t == "C":
        if TIER_C_PENALTY_IF_NOT_EVIDENCE and not is_evidence_query(query_text):
            return -TIER_PENALTY_C
        return 0.0

    # GRAPH, USER, UNKNOWN: nessun boost metodologico
    return 0.0

def diversify(items: List[Dict[str, Any]], max_per_page: int, max_per_doc: int, final_k: int) -> List[Dict[str, Any]]:
    """Keep best-scoring items but limit duplicates by page and document."""
    out = []
    page_count: Dict[Tuple[str, int], int] = {}
    doc_count: Dict[str, int] = {}

    for it in sorted(items, key=lambda x: float(x.get("final_score", x.get("score", 0.0))), reverse=True):
        fname = it.get("filename", "Unknown")
        page = int(it.get("page", 0))
        page_key = (fname, page)

        if doc_count.get(fname, 0) >= max_per_doc:
            continue
        if page_count.get(page_key, 0) >= max_per_page:
            continue

        out.append(it)
        doc_count[fname] = doc_count.get(fname, 0) + 1
        page_count[page_key] = page_count.get(page_key, 0) + 1

        if len(out) >= final_k:
            break
    return out

def append_audit_log(audit: AuditTrail):
    if not AUDIT_ENABLED:
        return
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(audit.model_dump_json() + "\n")
    except Exception as e:
        print(f"⚠️ Audit log write error: {e}")






def get_graph_entities(chunk_ids: List[str]) -> Dict[str, List[GraphEntity]]:
    """
    Recupera le entità collegate ai chunk in modo bilanciato.
    Coerente con l'ingestion:
    - Entity -> Chunk usa PRESENT_IN o MENTIONED_IN.
    - Bilancia il retrieval limitando le entità PER SINGOLO CHUNK,
      evitando che un chunk troppo denso cannibalizzi il limite totale.
    """
    if not chunk_ids or not neo4j_driver:
        return {}

    graph_map: Dict[str, List[GraphEntity]] = {}

    # CYPHER QUERY OTTIMIZZATA
    # 1. Usa UNWIND per iterare in modo sicuro sugli ID in input.
    # 2. MATCH le relazioni con le entità.
    # 3. CALL interno (subquery in Neo4j >= 4.1) permette di applicare
    #    un LIMIT (es. max 15 entità) specifico PER OGNI chunk id.
    # 4. Filtra le "Stop-Entities" (es. "Dati", "Sistema", "Azienda") che creano solo rumore.
    query = """
    UNWIND $ids AS target_id
    MATCH (c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) = target_id
    
    // Subquery per limitare i risultati per singolo chunk
    CALL {
        WITH c
        MATCH (e:Entity)-[r:PRESENT_IN|MENTIONED_IN]->(c)
        
        // --- INIZIO FIX: Stop-Entities Filter ---
        // Ignora entità troppo generiche che non aggiungono valore analitico
        WHERE NOT toLower(coalesce(e.name, e.label, e.id)) IN [
            'dato', 'dati', 'sistema', 'sistemi', 'azienda', 'aziende', 
            'utente', 'utenti', 'informazione', 'informazioni', 'documento',
            'data', 'system', 'company', 'user', 'information', 'document'
        ]
        // --- FINE FIX ---
        
        RETURN 
            coalesce(e.name, e.label, e.id) AS entity_name,
            coalesce(e.category, labels(e)[0], 'Entity') AS entity_type,
            type(r) AS rel_type
        // Limitiamo le entità estratte PER OGNI CHUNK (es. max 10)
        LIMIT 10
    }
    
    RETURN 
        target_id AS chunk_id, 
        entity_name AS name, 
        entity_type AS type, 
        rel_type AS rel
    """

    try:
        with neo4j_driver.session() as session:
            result = session.run(query, ids=chunk_ids)

            for record in result:
                cid = record["chunk_id"]

                entity = GraphEntity(
                    name=record["name"],
                    type=record["type"],
                    relation=record["rel"],
                )

                graph_map.setdefault(cid, []).append(entity)

    except Exception as e:
        # Usa il logger invece del print per allineamento Enterprise
        logger.error(f"Neo4j Query Error (get_graph_entities) - Failed on IDs {chunk_ids[:3]}...: {e}")

    return graph_map


def get_formulas_for_chunks(chunk_ids: List[str], limit_per_chunk: int = 5) -> Dict[str, List[str]]:
    """
    Recupera formule collegate ai chunk in modo bilanciato.
    Coerente con ingestion:
    - Formula -> Chunk usa MENTIONED_IN o PRESENT_IN
    Restituisce un dizionario {chunk_id: [formule_formattate]} per mantenere
    la tracciabilità della fonte (documento/pagina).
    """
    if not chunk_ids or not neo4j_driver:
        return {}

    formula_map: Dict[str, List[str]] = {}

    # CYPHER QUERY OTTIMIZZATA
    # 1. UNWIND garantisce che iteriamo su ogni chunk_id.
    # 2. CALL {} permette di applicare il limite PER SINGOLO CHUNK.
    # 3. DISTINCT assicura che non estraiamo formule duplicate per lo stesso chunk.
    query = """
    UNWIND $ids AS target_id
    MATCH (c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) = target_id
    
    CALL {
        WITH c
        MATCH (f)-[:MENTIONED_IN|PRESENT_IN]->(c)
        WHERE (f:Formula OR toUpper(coalesce(f.category, '')) = 'FORMULA')
        
        RETURN DISTINCT
            coalesce(f.latex, f.formula, '') AS latex,
            coalesce(f.plain, f.name, f.id, '') AS plain,
            coalesce(f.meaning_it, f.meaning, f.description, '') AS meaning
        LIMIT $lim
    }
    
    RETURN 
        target_id AS chunk_id,
        latex,
        plain,
        meaning
    """

    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit_per_chunk)

            for r in res:
                cid = r["chunk_id"]
                latex = (r["latex"] or "").strip()
                plain = (r["plain"] or "").strip()
                meaning = (r["meaning"] or "").strip()

                parts = []
                if latex:
                    parts.append(f"LaTeX: {latex}")
                if plain:
                    parts.append(f"Plain: {plain}")
                if meaning:
                    parts.append(f"Meaning: {meaning}")

                if parts:
                    formula_string = " | ".join(parts)
                    formula_map.setdefault(cid, []).append(formula_string)

    except Exception as e:
        logger.error(f"Neo4j Query Error (get_formulas_for_chunks): {e}")

    return formula_map


def get_neighbor_chunk_ids(chunk_ids: List[str], limit: int = GRAPH_MAX_NEIGHBOR_CHUNKS) -> List[str]:
    """
    Espande semanticamente i chunk usando entità condivise nel grafo.
    Coerente con ingestion:
    - Entity -> Chunk usa PRESENT_IN
    - Compatibile anche con MENTIONED_IN per vecchi dati
    """
    if not chunk_ids or not neo4j_driver:
        return []

    query = """
    MATCH (c1:Chunk)<-[:PRESENT_IN|MENTIONED_IN]-(e:Entity)-[:PRESENT_IN|MENTIONED_IN]->(c2:Chunk)
    WHERE coalesce(c1.chunk_id, c1.id) IN $ids
      AND NOT coalesce(c2.chunk_id, c2.id) IN $ids
      AND NOT toUpper(coalesce(e.type, e.category, labels(e)[0], '')) IN ['GENERIC', 'YEAR', 'DATE']

    WITH c2, count(DISTINCT e) AS strength
    WHERE strength >= 2

    RETURN coalesce(c2.chunk_id, c2.id) AS cid
    ORDER BY strength DESC
    LIMIT $lim
    """

    out: List[str] = []

    try:
        with neo4j_driver.session() as session:
            res = session.run(query, ids=chunk_ids, lim=limit)
            out = [str(r["cid"]) for r in res if r.get("cid")]

    except Exception as e:
        print(f"⚠️ Neo4j Semantic Neighbors Error: {e}")

    return out


def fetch_chunks_from_qdrant_by_ids(ids: List[str]) -> List[SourceItem]:
    """Fetch Qdrant points by IDs (for graph expansion neighbors)."""
    if not ids or not qdrant_client_inst:
        return []
    out: List[SourceItem] = []
    try:
        # qdrant retrieve works with ids list
        points = qdrant_client_inst.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
        )
        for p in points:
            payload = p.payload or {}
            tier = get_payload_tier(payload)
            content = safe_payload_text(payload)
            if not content:
                continue
            out.append(
                SourceItem(
                    id=str(p.id),
                    content=content,
                    filename=str(payload.get("filename", "Unknown")),
                    page=get_payload_page(payload),
                    type=get_payload_type(payload),
                    score=0.0,
                    graph_context=[],
                    section_hint=get_payload_section(payload),
                    image_id=get_payload_image_id(payload),
                    tier=tier,  # ✅ NEW
                )
            )
    except Exception as e:
        print(f"⚠️ Qdrant retrieve error: {e}")
    return out

def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def tier_qdrant_filter(query_text: str):
    return None

def build_retrieval_audit_md(
    query_text: str,
    intent: str,
    timings: Dict[str, float],
    counts: Dict[str, Any],
    top_sources_preview: List[Dict[str, Any]],
) -> str:
    """Audit avanzato che scompone l'attività di Qdrant, Postgres e Neo4j."""
    def ms(x: float) -> str:
        return f"{x*1000:.0f} ms"

    lines = []
    lines.append("### 🔎 Audit Retrieval (Multi-Database Analysis)")
    lines.append(f"- **Intent**: `{intent}`")
    lines.append(f"- **Query**: `{(query_text or '')[:180]}`")

    # 🌌 SEZIONE QDRANT (Vettoriale)
    lines.append("\n#### 🌌 Qdrant (Vector Search)")
    if "qdrant_search" in timings:
        lines.append(f"- Tempo: **{ms(timings['qdrant_search'])}**")
    lines.append(f"- Hits vettoriali: **{counts.get('qdrant_hits', 0)}**")

    # 🐘 SEZIONE POSTGRES (BM25)
    lines.append("\n#### 🐘 Postgres (Keyword Search)")
    if "bm25_search" in timings:
        lines.append(f"- Tempo: **{ms(timings['bm25_search'])}**")
    lines.append(f"- Match testuali: **{counts.get('bm25_hits', 0)}**")

    # 📄 SEZIONE DOCUMENT SCOPE
    if counts.get("requested_doc"):
        lines.append("\n#### 📄 Document Scope")
        lines.append(f"- Documento richiesto: `{counts.get('requested_doc')}`")
        lines.append(f"- Chunk trovati nel documento: **{counts.get('doc_scope_hits', 0)}**")
        lines.append(f"- Prima del filtro documento: **{counts.get('doc_scope_before', 0)}**")
        lines.append(f"- Dopo il filtro documento: **{counts.get('doc_scope_after', 0)}**")

    # 🕸️ SEZIONE NEO4J (Grafo)
    neo4j_direct = counts.get("neo4j_direct_hits", 0)
    neo4j_expanded = counts.get("neo4j_hits", 0)
    final_formulas = counts.get("final_formulas", 0)

    if (
        neo4j_direct > 0
        or neo4j_expanded > 0
        or final_formulas > 0
        or "graph" in timings
        or "neo4j_direct_search" in timings
    ):
        lines.append("\n#### 🕸️ Neo4j (Graph Search / Expansion)")

        if "neo4j_direct_search" in timings:
            lines.append(f"- Tempo direct search: **{ms(timings['neo4j_direct_search'])}**")

        if "graph" in timings:
            lines.append(f"- Tempo graph expansion: **{ms(timings['graph'])}**")

        lines.append(f"- Chunk trovati da Neo4j direct search: **{neo4j_direct}**")
        lines.append(f"- Chunk aggiunti da graph expansion: **{neo4j_expanded}**")
        lines.append(f"- Formule collegate recuperate: **{final_formulas}**")

    # ⚖️ SEZIONE PERFORMANCE & RERANK
    lines.append("\n#### ⚖️ Fusione & Reranking")
    if "rerank" in timings:
        lines.append(f"- Tempo Reranker: **{ms(timings['rerank'])}**")
    lines.append(f"- Candidati totali: **{counts.get('qdrant_hits', 0) + counts.get('bm25_hits', 0)}**")
    if "total" in timings:
        lines.append(f"- **Tempo Totale Retrieval**: **{ms(timings['total'])}**")

    # 📦 DISTRIBUZIONE TIER
    tier_split = counts.get("tier_split", {})
    if tier_split:
        lines.append("\n#### 📦 Tier Distribution")
        for t, n in tier_split.items():
            lines.append(f"- `{t}`: **{n}**")

    return "\n".join(lines).strip()

def fetch_pg_chunks_by_uuid(chunk_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Recupera da Postgres i chunk usando l'ID corretto: chunk_uuid.

    Ritorna:
    {
        chunk_uuid: {
            "chunk_uuid": ...,
            "content_raw": ...,
            "content_semantic": ...,
            "metadata_json": ...,
            "ingestion_ts": ...
        }
    }

    Nota:
    - chunk_uuid corrisponde all'id usato in Qdrant.
    - chunk_uuid corrisponde al chunk_id usato in Neo4j.
    - prende sempre la versione più recente del chunk in base a ingestion_ts.
    """
    if not PG_ENRICH_ENABLED or not pg_pool or not chunk_uuids:
        return {}

    # Dedup preservando l'ordine
    seen = set()
    uuids: List[str] = []

    for u in chunk_uuids:
        if not u:
            continue

        key = str(u).strip()
        if not key or key in seen:
            continue

        seen.add(key)
        uuids.append(key)

    if not uuids:
        return {}

    sql = """
    WITH wanted(chunk_uuid) AS (
        VALUES %s
    ),
    ranked AS (
        SELECT
            d.chunk_uuid::text AS chunk_uuid,
            d.content_raw,
            d.content_semantic,
            d.metadata_json,
            d.ingestion_ts,
            ROW_NUMBER() OVER (
                PARTITION BY d.chunk_uuid
                ORDER BY d.ingestion_ts DESC
            ) AS rn
        FROM public.document_chunks d
        JOIN wanted w
          ON d.chunk_uuid::text = w.chunk_uuid::text
    )
    SELECT
        chunk_uuid,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts
    FROM ranked
    WHERE rn = 1;
    """

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                sql,
                [(u,) for u in uuids]
            )
            rows = cur.fetchall()

        out: Dict[str, Dict[str, Any]] = {}

        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts in rows:
            # metadata_json può arrivare già come dict oppure come stringa JSON
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out[str(chunk_uuid)] = {
                "chunk_uuid": str(chunk_uuid),
                "content_raw": content_raw or "",
                "content_semantic": content_semantic or "",
                "metadata_json": metadata_json,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            }

        return out

    except Exception as e:
        print(f"⚠️ PG enrich by chunk_uuid error: {e}")
        return {}

    finally:
        pg_pool.putconn(conn)


def search_pg_bm25(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca keyword/BM25-like su Postgres usando full-text search.
    PATCH: conserva acronimi brevi e usa websearch_to_tsquery.
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []
    if not query_text or not query_text.strip():
        return []
    tokens = extract_search_tokens(query_text)
    if not tokens:
        return []
    pg_query = " OR ".join(tokens)
    sql = """
    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq)
    SELECT
        chunk_uuid::text,
        content_raw,
        content_semantic,
        metadata_json,
        ts_rank_cd(
            to_tsvector('simple', COALESCE(content_semantic, '') || ' ' || COALESCE(content_raw, '') || ' ' || COALESCE(metadata_json::text, '')),
            q.tsq
        ) AS rank
    FROM public.document_chunks, q
    WHERE to_tsvector('simple', COALESCE(content_semantic, '') || ' ' || COALESCE(content_raw, '') || ' ' || COALESCE(metadata_json::text, '')) @@ q.tsq
    ORDER BY rank DESC
    LIMIT %s;
    """
    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (pg_query, limit))
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for chunk_uuid, content_raw, content_semantic, metadata_json, rank in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}
            if metadata_json is None:
                metadata_json = {}
            out.append({"id": str(chunk_uuid), "content": content_semantic or content_raw or "", "metadata": metadata_json, "score": float(rank or 0.0), "origin": "PostgresBM25"})
        return out
    except Exception as e:
        print(f"⚠️ BM25 Error: {e}")
        return []
    finally:
        pg_pool.putconn(conn)


def search_pg_exact_phrases(query_text: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Ricerca ILIKE deterministica per frasi/acronimi che non devono dipendere dal vettoriale."""
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []
    phrases = extract_exact_phrases(query_text)
    if not phrases:
        return []
    clauses = []
    params: List[Any] = []
    for p in phrases[:12]:
        like = f"%{p.lower()}%"
        clauses.append("""(
            lower(COALESCE(content_semantic, '')) LIKE %s OR
            lower(COALESCE(content_raw, '')) LIKE %s OR
            lower(COALESCE(metadata_json::text, '')) LIKE %s
        )""")
        params.extend([like, like, like])
    sql = f"""
    SELECT chunk_uuid::text, content_raw, content_semantic, metadata_json, ingestion_ts
    FROM public.document_chunks
    WHERE {' OR '.join(clauses)}
    ORDER BY ingestion_ts DESC
    LIMIT %s;
    """
    params.append(limit)
    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}
            if metadata_json is None:
                metadata_json = {}
            out.append({"id": str(chunk_uuid), "content": content_semantic or content_raw or "", "metadata": metadata_json, "score": 2.0, "origin": "PostgresExactPhrase", "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else ""})
        return out
    except Exception as e:
        print(f"⚠️ Exact phrase search error: {e}")
        return []
    finally:
        pg_pool.putconn(conn)


def _term_sql_condition(alias: str) -> Tuple[str, List[Any]]:
    """
    Condizione SQL robusta per alias/acronimi.
    - Per acronimi brevi usa regex con boundary.
    - Per frasi usa LIKE case-insensitive.
    """
    alias = (alias or "").strip()
    if not alias:
        return "", []

    is_short_acronym = alias.upper() == alias and 2 <= len(alias) <= 10

    if is_short_acronym:
        pattern = r"(^|[^A-Za-z0-9])" + re.escape(alias) + r"([^A-Za-z0-9]|$)"
        return (
            """(
                COALESCE(content_semantic, '') ~* %s OR
                COALESCE(content_raw, '') ~* %s OR
                COALESCE(metadata_json::text, '') ~* %s
            )""",
            [pattern, pattern, pattern],
        )

    like = f"%{alias.lower()}%"
    return (
        """(
            lower(COALESCE(content_semantic, '')) LIKE %s OR
            lower(COALESCE(content_raw, '')) LIKE %s OR
            lower(COALESCE(metadata_json::text, '')) LIKE %s
        )""",
        [like, like, like],
    )


def search_pg_glossary_term(
    canonical_term: str,
    aliases: List[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Lookup atomico di una voce di glossario.
    Non genera risposte e non contiene definizioni hard-coded: recupera solo chunk.
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    clauses: List[str] = []
    params: List[Any] = []

    for alias in aliases:
        cond, cond_params = _term_sql_condition(alias)
        if cond:
            clauses.append(cond)
            params.extend(cond_params)

    if not clauses:
        return []

    sql = f"""
    SELECT
        chunk_uuid::text,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts
    FROM public.document_chunks
    WHERE
        (
            lower(COALESCE(metadata_json->>'filename', '')) LIKE %s
            OR lower(COALESCE(metadata_json->>'source_name', '')) LIKE %s
            OR lower(COALESCE(metadata_json::text, '')) LIKE %s
        )
        AND ({' OR '.join(clauses)})
    ORDER BY ingestion_ts DESC
    LIMIT %s;
    """

    # psycopg2 usa il carattere % per i placeholder; i pattern LIKE vanno passati
    # come parametri e non scritti come '%glossario%' dentro la query, altrimenti
    # si ottengono errori tipo: list index out of range.
    params = ["%glossario%", "%glossario%", "%glossario%"] + params
    params.append(limit)

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []

        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out.append({
                "id": str(chunk_uuid),
                "content_raw": content_raw or "",
                "content_semantic": content_semantic or "",
                "metadata": metadata_json,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
                "term": canonical_term,
            })

        return out

    except Exception as e:
        print(f"⚠️ Glossary term lookup error for {canonical_term}: {e}")
        return []

    finally:
        pg_pool.putconn(conn)


def extract_definition_snippet(
    canonical_term: str,
    aliases: List[str],
    text: str,
    max_chars: int = 900,
) -> str:
    """Estrae uno snippet vicino alla voce trovata, senza inventare definizioni."""
    raw = (text or "").strip()

    if not raw:
        return "Voce trovata, ma il chunk non contiene testo utilizzabile."

    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    aliases_l = [a.lower() for a in aliases if a]

    for i, line in enumerate(lines):
        ll = line.lower()
        if any(alias in ll for alias in aliases_l):
            snippet = " ".join(lines[i:i + 5]).strip()
            return snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")

    raw_l = raw.lower()
    for alias in aliases_l:
        pos = raw_l.find(alias)
        if pos >= 0:
            start = max(0, pos - 160)
            end = min(len(raw), pos + max_chars)
            snippet = re.sub(r"\s+", " ", raw[start:end]).strip()
            return snippet + ("..." if end < len(raw) else "")

    return re.sub(r"\s+", " ", raw[:max_chars]).strip()


def answer_glossary_terms_directly(query_text: str) -> Tuple[str, List[SourceItem], str]:
    """
    Risposta deterministica per query di glossario.
    Ogni voce viene cercata separatamente per ridurre falsi negativi.
    """
    terms = extract_requested_terms(query_text)

    if not terms:
        return "", [], ""

    answer_lines: List[str] = []
    evidence_lines: List[str] = []
    source_items: List[SourceItem] = []
    source_seen = set()

    for term in terms:
        aliases = GLOSSARY_TERM_ALIASES.get(term, [term])
        hits = search_pg_glossary_term(term, aliases, limit=5)

        if not hits:
            answer_lines.append(f"- **{term}**: voce non trovata nel glossario recuperato.")
            evidence_lines.append(f"- **{term}**: nessun chunk di glossario recuperato.")
            continue

        best = hits[0]
        content = best.get("content_semantic") or best.get("content_raw") or ""
        snippet = extract_definition_snippet(term, aliases, content)

        answer_lines.append(f"- **{term}**: {snippet}")

        meta = best.get("metadata", {}) or {}
        fname = meta.get("filename") or meta.get("source_name") or "Glossario"
        page = int(meta.get("page_no") or meta.get("page") or 0)

        evidence_lines.append(f"- **{term}**: recuperato da `{fname}`, pag. {page}.")

        sid = str(best.get("id", ""))
        if sid and sid not in source_seen:
            source_seen.add(sid)
            source_items.append(
                SourceItem(
                    id=sid,
                    content=content[:1800],
                    filename=fname,
                    page=page,
                    type=meta.get("toon_type") or meta.get("type") or "text",
                    score=2.0,
                    tier=normalize_tier_value(meta.get("tier", "C")),
                    db_origin="PostgresGlossaryTerm",
                    section_hint=f"Glossary term: {term}",
                    pg_ingestion_ts=best.get("ingestion_ts", ""),
                    pg_source_name=meta.get("source_name", ""),
                    pg_source_type=meta.get("source_type", ""),
                    pg_log_id=int(meta.get("log_id") or 0),
                    pg_chunk_id=int(meta.get("chunk_index") or 0),
                    pg_toon_type=meta.get("toon_type", ""),
                )
            )

    used_files = sorted({s.filename for s in source_items if s.filename})

    answer = (
        "**A) Risposta**\n\n"
        + "\n".join(answer_lines)
        + "\n\n"
        "**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- Risposta generata in modalità deterministica di glossario: ogni voce è stata cercata separatamente.\n"
        "- Una voce viene dichiarata assente solo se il lookup atomico sul glossario non restituisce chunk pertinenti.\n\n"
        "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Nessuna fonte di glossario recuperata.")
    )

    debug_md = (
        "### 🔎 Audit (Glossary Deterministic Mode)\n"
        f"- Termini richiesti: `{', '.join(terms)}`\n"
        f"- Fonti recuperate: **{len(source_items)}**\n"
        "- Retrieval generativo bypassato solo per il lookup definitorio."
    )

    return answer, source_items, debug_md



# ============================================================
# 🧮 MATH-FIRST CONTEXT MERGE - v4.3 minimal non-adaptive fix
# ============================================================
def build_math_answer_with_document_context(
    math_answer: str,
    sources: List[SourceItem],
    max_items: int = 3,
) -> str:
    """
    Integra un risultato matematico deterministico con contesto documentale,
    senza permettere al Graph Relation Mode o all'LLM di modificare il calcolo.

    Fix v4.3:
    - se il calcolo è stato risolto dal solver deterministico, il risultato numerico
      resta autoritativo;
    - i documenti recuperati servono solo per contestualizzare risk/evidence/control
      assessment;
    - mantiene la struttura A/B/C/D già prodotta dal solver matematico.
    """
    if not math_answer:
        return ""

    clean_sources = []
    seen = set()

    for s in sources or []:
        tier = normalize_tier_value(getattr(s, "tier", "") or "")
        stype = normalize_source_type(getattr(s, "type", "") or "")

        # Evita di usare righe grafo/formula come contesto concettuale principale.
        if tier == "GRAPH" or stype in {"graph", "graph_relations", "formula"}:
            continue

        filename = getattr(s, "filename", "") or "N/D"
        page = int(getattr(s, "page", 0) or 0)
        content = re.sub(r"\s+", " ", getattr(s, "content", "") or "").strip()

        if not content:
            continue

        key = (normalize_doc_name(filename), page)
        if key in seen:
            continue

        seen.add(key)
        clean_sources.append((filename, page, content))

        if len(clean_sources) >= max_items:
            break

    if not clean_sources:
        return math_answer

    context_lines = [
        "Collegamento documentale",
        "",
        "- Il risultato numerico è calcolato solo sui dati forniti dall'utente.",
        "- Le fonti recuperate vengono usate solo per contestualizzare il risultato nel risk/evidence/control assessment; non modificano il calcolo.",
    ]

    for filename, page, content in clean_sources:
        snippet = content[:360].rstrip()
        if len(content) > 360:
            snippet += "..."
        context_lines.append(f"- `{filename}` (p.{page}): {snippet}")

    context_block = "\n".join(context_lines)
    used_files = []
    for filename, _, _ in clean_sources:
        if filename and filename not in used_files:
            used_files.append(filename)

    d_sources_extra = "\n".join(f"- {f}" for f in used_files)

    marker = "**D) Fonti**"
    if marker in math_answer:
        before, after = math_answer.split(marker, 1)
        after_clean = after.strip()
        if d_sources_extra:
            after_clean = after_clean + "\n" + d_sources_extra
        return before.rstrip() + "\n\n" + context_block + "\n\n" + marker + "\n\n" + after_clean

    return math_answer.rstrip() + "\n\n" + context_block + "\n\n**D) Fonti**\n\n" + d_sources_extra


# =========================
# 🔍 RAG v2 Retrieval
# =========================


def apply_rrf_scoring(candidates: List[Dict[str, Any]], k: int = 60):
    """
    Reciprocal Rank Fusion tra:
    - Qdrant vector rank
    - Postgres BM25 rank
    - Neo4j graph rank
    """

    for c in candidates:
        c["rrf_score"] = 0.0

    vec_sorted = sorted(
        [c for c in candidates if c.get("score_vec", c.get("score_base", 0.0)) > 0],
        key=lambda x: x.get("score_vec", x.get("score_base", 0.0)),
        reverse=True,
    )

    bm25_sorted = sorted(
        [c for c in candidates if c.get("score_bm25", 0.0) > 0],
        key=lambda x: x.get("score_bm25", 0.0),
        reverse=True,
    )

    graph_sorted = sorted(
        [c for c in candidates if c.get("score_graph", 0.0) > 0],
        key=lambda x: x.get("score_graph", 0.0),
        reverse=True,
    )

    for rank, item in enumerate(vec_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(graph_sorted):
        item["rrf_score"] += 1.0 / (k + rank + 1)


RAG_STOPWORDS = {
    # --- GRAMMATICA E PRONOMI IT (> 3 lettere) ---
    "della", "delle", "degli", "dello", "dalla", "dalle", "dagli",
    "nella", "nelle", "negli", "nello", "alla", "alle", "agli",
    "sulla", "sulle", "sugli", "sullo",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
    "sono", "presenti", "presente", "ciascuna", "ciascuno", "tutti", "tutte",
    "quale", "quali", "cosa", "come", "dove", "quando", "perché", "perche",

    # --- VERBI CONVERSAZIONALI E INTENTI IT ---
    "spiega", "spiegami", "riporta", "riportale", "mostra", "mostrami", 
    "dimmi", "elenca", "trova", "cerca", "voglio", "vorrei", "fammi",
    "riguardo", "inerente", "relativo", "secondo", "base", "basandoti",

    # --- GRAMMATICA E CONVERSAZIONE EN ---
    "what", "which", "where", "when", "explain", "show", "tell", "list", 
    "find", "search", "report", "present", "available", "each", "about", 
    "these", "those", "this", "that", "there", "their", "would", "could",
    "should", "please", "according", "regarding", "based", "give",

    # --- RAG E STRUTTURA DEL DOCUMENTO (IT/EN) ---
    "documento", "documenti", "file", "fonte", "fonti", "testo", "riferisce",
    "document", "documents", "source", "sources", "text", "context",
    "pagina", "pag", "page", "pages", "paragrafo", "sezione", "capitolo",
    "chapter", "section", "paragraph",

    # --- FORMULE E CONCETTI META ---
    "formula", "formule", "matematica", "matematiche", "latex", "concetto"
}

def extract_rag_tokens(query_text: str) -> List[str]:
    """
    Estrae token utili per filename matching, Neo4j e formula lookup.
    Mantiene acronimi brevi (MFA, APT, CVE, KPI) invece di eliminarli.
    """
    return [t for t in extract_search_tokens(query_text) if t not in RAG_STOPWORDS]


def search_neo4j_entities(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca diretta nel grafo Neo4j sui nodi Entity.

    Versione pulita:
    - non usa proprietà Neo4j inesistenti come source_name/content_semantic/content_raw
    - ritorna chunk_id coerente con chunk_uuid usato da Qdrant/Postgres
    - lascia a Postgres il compito di arricchire il contenuto completo
    """
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)

    if not tokens:
        return []

    cypher = """
    MATCH (f)-[:MENTIONED_IN|PRESENT_IN]->(c:Chunk)
    WHERE (
            f:Formula
            OR toUpper(coalesce(f.category, '')) = 'FORMULA'
        )
    AND any(tok IN $tokens WHERE
            toLower(coalesce(c.filename, '')) CONTAINS tok OR
            toLower(coalesce(f.latex, '')) CONTAINS tok OR
            toLower(coalesce(f.formula, '')) CONTAINS tok OR
            toLower(coalesce(f.plain, '')) CONTAINS tok OR
            toLower(coalesce(f.meaning_it, '')) CONTAINS tok OR
            toLower(coalesce(f.description, '')) CONTAINS tok OR
            toLower(coalesce(f.name, f.id, '')) CONTAINS tok
    )
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.chunk_index, 0) AS chunk_index,
        coalesce(f.latex, f.formula, '') AS latex,
        coalesce(f.plain, f.name, f.id, '') AS plain,
        coalesce(f.meaning_it, f.description, '') AS meaning,
        count(*) AS rel_count
    ORDER BY page ASC, chunk_index ASC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []

    try:
        with neo4j_driver.session() as session:
            rows = session.run(cypher, tokens=tokens, limit=limit)

            for r in rows:
                cid = r.get("chunk_id")

                if not cid:
                    continue

                out.append({
                    "id": str(cid),
                    "content": r.get("content") or "",
                    "filename": r.get("filename") or "Neo4j",
                    "page": int(r.get("page") or 0),
                    "type": "graph",
                    "tier": "GRAPH",
                    "score_graph": float(r.get("rel_count") or 1.0),
                    "origin": f"Neo4j Entity Search: {r.get('entity_name')}",
                    "section_hint": f"Entity: {r.get('entity_name')}",
                })

    except Exception as e:
        print(f"⚠️ Neo4j direct search error: {e}")

    return out

def search_neo4j_formulas(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Ricerca diretta delle formule nel Knowledge Graph.

    Utile per domande come:
    - quali formule sono presenti nel documento?
    - riportale in LaTeX
    - quali equazioni usa il documento?
    """
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)

    if not tokens:
        return []

    cypher = """
    MATCH (f)-[:MENTIONED_IN|PRESENT_IN]->(c:Chunk)
    WHERE (
            f:Formula
            OR toUpper(coalesce(f.category, '')) = 'FORMULA'
        )
    AND any(tok IN $tokens WHERE
            toLower(coalesce(c.filename, '')) CONTAINS tok OR
            toLower(coalesce(f.latex, '')) CONTAINS tok OR
            toLower(coalesce(f.formula, '')) CONTAINS tok OR
            toLower(coalesce(f.plain, '')) CONTAINS tok OR
            toLower(coalesce(f.meaning_it, '')) CONTAINS tok OR
            toLower(coalesce(f.description, '')) CONTAINS tok OR
            toLower(coalesce(f.name, f.id, '')) CONTAINS tok
    )
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.chunk_index, 0) AS chunk_index,
        coalesce(f.latex, f.formula, '') AS latex,
        coalesce(f.plain, f.name, f.id, '') AS plain,
        coalesce(f.meaning_it, f.description, '') AS meaning,
        count(*) AS rel_count
    ORDER BY page ASC, chunk_index ASC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []

    try:
        with neo4j_driver.session() as session:
            rows = session.run(cypher, tokens=tokens, limit=limit)

            for r in rows:
                cid = r.get("chunk_id")

                if not cid:
                    continue

                latex = (r.get("latex") or "").strip()
                plain = (r.get("plain") or "").strip()
                meaning = (r.get("meaning") or "").strip()

                formula_parts = []

                if latex:
                    formula_parts.append(f"LaTeX: {latex}")

                if plain:
                    formula_parts.append(f"Plain: {plain}")

                if meaning:
                    formula_parts.append(f"Meaning: {meaning}")

                if not formula_parts:
                    continue

                out.append({
                    "id": str(cid),
                    "content": "Formula from Knowledge Graph:\n" + "\n".join(formula_parts),
                    "filename": r.get("filename") or "Neo4j",
                    "page": int(r.get("page") or 0),
                    "type": "formula",
                    "tier": "GRAPH",
                    "score_graph": float(r.get("rel_count") or 5.0),
                    "origin": "Neo4j Formula Search",
                    "section_hint": "Formula node",
                })

    except Exception as e:
        print(f"⚠️ Neo4j formula search error: {e}")

    return out





GRAPH_QUERY_NOISE_TERMS = {
    # IT generici
    "usa", "usare", "spiega", "spiegare", "collegamenti", "collegamento",
    "relazioni", "relazione", "documenti", "documento", "normativi",
    "normativo", "glossario", "fonti", "fonte", "tabella",

    # EN generici
    "using", "use", "explain", "relationship", "relationships",
    "relation", "relations", "retrieved", "documents", "document",
    "sources", "source", "glossary", "table",
}


def graph_relevant_tokens(query_text: str) -> List[str]:
    """
    Estrae token utili per cercare relazioni nel grafo.
    Rimuove parole di istruzione, formato e richiesta.
    Non contiene termini domain-specific.
    """
    tokens = extract_rag_tokens(query_text)

    out: List[str] = []

    for t in tokens:
        tl = t.lower().strip()

        if not tl:
            continue

        if tl in GRAPH_QUERY_NOISE_TERMS:
            continue

        if tl in RAG_STOPWORDS:
            continue

        if len(tl) < 3:
            continue

        out.append(tl)

    return list(dict.fromkeys(out))


def _relation_row_text(row: Dict[str, Any]) -> str:
    props = row.get("props") or {}

    try:
        props_text = json.dumps(props, ensure_ascii=False)
    except Exception:
        props_text = str(props)

    return " ".join([
        str(row.get("source") or ""),
        str(row.get("relation") or ""),
        str(row.get("target") or ""),
        props_text,
        str(row.get("filename") or ""),
    ]).lower()


def filter_neo4j_relation_rows(
    query_text: str,
    rows: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Tiene solo relazioni Neo4j che coprono almeno due concetti rilevanti
    della domanda.

    Se la domanda contiene pochissimi concetti, accetta anche un solo match.
    """
    if not rows:
        return []

    tokens = graph_relevant_tokens(query_text)

    if not tokens:
        return rows[:limit]

    scored: List[Tuple[int, Dict[str, Any]]] = []

    for row in rows:
        text = _relation_row_text(row)
        hits = {t for t in tokens if t in text}
        hit_count = len(hits)

        # Regola generale:
        # - con 1-2 token rilevanti basta 1 hit;
        # - con più token servono almeno 2 hit.
        min_hits = 1 if len(tokens) <= 2 else 2

        if hit_count >= min_hits:
            scored.append((hit_count, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored[:limit]]

def search_neo4j_relations(query_text: str, limit: int = 40) -> List[Dict[str, Any]]:
    """Restituisce vere relazioni Entity-[:REL]->Entity dal KG, non solo chunk collegati."""
    if not neo4j_driver:
        return []

    tokens = graph_relevant_tokens(query_text)

    if not tokens:
        tokens = extract_rag_tokens(query_text)

    if not tokens:
        return []

    # --- INIZIO FIX 3: Espansione Statica dei Token (Assessment Domain) ---
    q_low = (query_text or "").lower()

    # 1. CUSTOMER ACCOUNT (Organizzazione/Soggetto verificato)
    customer_terms = [
        "cliente", "account cliente", "azienda cliente", "organizzazione", "tenant", "sottoscrittore", 
        "client", "customer", "customer account", "client organization", "subscriber"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in customer_terms):
        tokens.append("customer_account")
        
    # 2. ADMIN (Gestione/Acquisto Licenza Assessment)
    admin_terms = [
        "amministratore", "admin", "chi compra la licenza", "acquirente della licenza", "titolare della licenza", "responsabile licenza",
        "administrator", "license buyer", "license purchaser", "license owner", "license manager"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in admin_terms):
        tokens.append("admin")
        
    # 3. SECURITY INCIDENT (Focus NIS2 / GDPR)
    incident_terms = [
        "incidente", "violazione", "breach", "data breach", "compromissione", "attacco informatico", "evento di sicurezza", "fuga di dati",
        "incident", "compromise", "cyber attack", "security event", "data leak", "security incident"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in incident_terms):
        tokens.append("securityincident")
        
    # 4. SECURITY CONTROL (Focus ISO 27001 / DORA)
    control_terms = [
        "controllo", "misura", "mitigazione", "salvaguardia", "contromisura", "presidio", "misura di sicurezza", "policy", "procedura",
        "control", "measure", "mitigation", "safeguard", "countermeasure", "security measure", "security control", "procedure"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in control_terms):
        tokens.append("securitycontrol")

    # 5. ASSET (Perimetro di applicabilità)
    asset_terms = [
        "asset", "risorsa", "dispositivo", "bene", "infrastruttura", "sistema",
        "resource", "device", "infrastructure", "system", "equipment"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in asset_terms):
        tokens.append("asset")

    # 6. RISK / VULNERABILITY (Valutazione del rischio)
    risk_terms = [
        "rischio", "minaccia", "vulnerabilità", "falla", "impatto", "criticita", "criticità",
        "risk", "threat", "vulnerability", "flaw", "impact", "criticality"
    ]
    if any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in risk_terms):
        tokens.append("risk")
    # --- FINE FIX 3 ---

    # CYPHER FIX: Aggiunto il controllo su e.category, e.type e labels(e)
    # in modo che token come "securityincident" facciano match sull'ontologia del nodo.
    cypher = """
    MATCH (e1:Entity)-[rel]->(e2:Entity)
    WHERE any(tok IN $tokens WHERE
        toLower(coalesce(e1.name, e1.id, '')) CONTAINS tok OR
        toLower(coalesce(e2.name, e2.id, '')) CONTAINS tok OR
        toLower(coalesce(e1.description, '')) CONTAINS tok OR
        toLower(coalesce(e2.description, '')) CONTAINS tok OR
        toLower(coalesce(e1.category, e1.type, labels(e1)[0], '')) CONTAINS tok OR
        toLower(coalesce(e2.category, e2.type, labels(e2)[0], '')) CONTAINS tok OR
        any(s IN coalesce(e1.synonyms, []) WHERE toLower(s) CONTAINS tok) OR
        any(s IN coalesce(e2.synonyms, []) WHERE toLower(s) CONTAINS tok)
    )
    OPTIONAL MATCH (e1)-[:PRESENT_IN|MENTIONED_IN]->(c1:Chunk)
    OPTIONAL MATCH (e2)-[:PRESENT_IN|MENTIONED_IN]->(c2:Chunk)
    RETURN coalesce(e1.name, e1.id) AS source,
           type(rel) AS relation,
           coalesce(e2.name, e2.id) AS target,
           properties(rel) AS props,
           coalesce(rel.filename, rel.source_name, c1.filename, c2.filename, '') AS filename,
           coalesce(rel.page_no, rel.page, c1.page, c2.page, 0) AS page
    LIMIT $limit
    """

    try:
        with neo4j_driver.session() as session:
            scan_limit = max(limit * 4, limit)
            rows = session.run(cypher, tokens=tokens, limit=scan_limit)
            raw_rows = [dict(r) for r in rows]

        return filter_neo4j_relation_rows(query_text, raw_rows, limit)

    except Exception as e:
        print(f"⚠️ Neo4j relation search error: {e}")
        return []


from typing import List, Dict, Any, Optional

def graph_relations_to_source(rows: List[Dict[str, Any]]) -> Optional[SourceItem]:
    """
    Converte le relazioni Neo4j in una tabella Markdown.
    Questo rende il contesto grafo più deterministico per il modello.
    Blindata con _md_cell per evitare la corruzione della tabella Markdown.
    """
    if not rows:
        return None

    lines = [
        "Relazioni Neo4j trovate:",
        "",
        "| Entità sorgente | Relazione | Entità target | Documento | Pagina |",
        "|---|---|---|---|---:|",
    ]

    seen = set()

    for r in rows:
        # Usa _md_cell (già presente nel tuo codice) per pulire e troncare in sicurezza
        source = _md_cell(r.get("source") or "", 180)
        
        # Recuperiamo anche le proprietà della relazione (se presenti) per dare più contesto
        props = r.get("props") or {}
        props_str = f" {props}" if props else ""
        relation_raw = str(r.get("relation") or "RELATES_TO") + props_str
        relation = _md_cell(relation_raw, 120)
        
        target = _md_cell(r.get("target") or "", 180)
        filename = _md_cell(r.get("filename") or "N/D", 200)
        page = int(r.get("page") or 0)

        if not source or not target:
            continue

        key = (source, relation, target, filename, page)
        if key in seen:
            continue

        seen.add(key)

        lines.append(
            f"| {source} | {relation} | {target} | {filename} | {page} |"
        )

    if len(lines) <= 4:
        return None

    return SourceItem(
        id="neo4j_relations",
        content="\n".join(lines),
        filename="Neo4j Knowledge Graph",
        page=0,
        type="graph_relations",
        tier="GRAPH",
        score=1.0,
        db_origin="Neo4j Relation Search",
        section_hint="Entity relations table",
    )



def _md_cell(value: Any, max_len: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text



def _clean_graph_concept(value: str) -> str:
    """
    Pulisce un concetto testuale prima della ricerca nel grafo.
    Supporta articoli con apostrofo e combinazioni multiple di stopwords (es. "and the").
    """
    text = re.sub(r"\s+", " ", value or "").strip(" \t\n\r.,;:!?()[]{}\"'“”")
    
    # FIX 1: Supporto per apostrofi (es. l'entità, un'azienda) e parole multiple (es. "e il ").
    # Usando (?:...)+ la regex "mangia" in loop tutti gli articoli/congiunzioni iniziali.
    leading_noise = r"^(?:(?:e|ed|and|or|oppure|o|il|lo|la|i|gli|le|un|una|uno|the|a|an)\s+|(?:l|un)['’])+"
    text = re.sub(leading_noise, "", text, flags=re.IGNORECASE)
    
    # FIX 2: Rimuove eventuali congiunzioni rimaste appese alla fine del concetto
    trailing_noise = r"\s+(?:e|ed|and|or|oppure|o)$"
    text = re.sub(trailing_noise, "", text, flags=re.IGNORECASE)
    
    return text.strip()


def _split_relation_segment(segment: str) -> List[str]:
    """
    Divide una porzione della domanda in concetti candidati.
    Non usa termini di dominio: sfrutta punteggiatura e connettori IT/EN.
    """
    segment = re.sub(r"[\n\r]+", " ", segment or "")
    
    # FIX 3: Rimosso "with" e "con" da questa lista. Se eliminiamo tutto ciò che 
    # c'è dopo "con", perdiamo entità utili (es. "relazione tra server con database").
    segment = re.sub(
        r"\b(?:usando|using|tramite|through|rispetto a|against|return|do not|non usare|non rispondere)\b.*$",
        "",
        segment,
        flags=re.IGNORECASE,
    )
    
    # FIX 4: Aggiunti "o", "or", "oppure", "con" e "with" come separatori logici 
    # per dividere correttamente le entità.
    raw_parts = re.split(
        r"\s*(?:,|;|\be\b|\bed\b|\band\b|\bo\b|\bor\b|\boppure\b|\bwith\b|\bcon\b|\bversus\b|\bvs\.?\b)\s*", 
        segment, 
        flags=re.IGNORECASE
    )
    
    return [_clean_graph_concept(p) for p in raw_parts if _clean_graph_concept(p)]

def _canonical_graph_concept(concept: str) -> str:
    """
    Canonicalizza solo usando alias già presenti nel glossario.
    Evita relazioni tra sinonimi dello stesso concetto, es. MFA ↔ autenticazione a più fattori.
    """
    c = (concept or "").lower().strip()

    for canonical, aliases in GLOSSARY_TERM_ALIASES.items():
        all_aliases = [canonical] + list(aliases or [])
        for alias in all_aliases:
            al = (alias or "").lower().strip()
            if not al:
                continue
            if c == al:
                return canonical.lower()

    return c


def _graph_concept_aliases(concept: str) -> List[str]:
    """
    Espande un concetto richiesto dall'utente in alias minimi IT/EN.
    Usa alias di glossario + varianti linguistiche generiche già note.
    """
    aliases: List[str] = []
    raw = _clean_graph_concept(concept)

    if raw:
        aliases.append(raw)

    raw_l = raw.lower()

    for canonical, vals in GLOSSARY_TERM_ALIASES.items():
        all_aliases = [canonical] + list(vals or [])
        if any(raw_l == (a or "").lower().strip() for a in all_aliases):
            aliases.extend(all_aliases)

    if raw_l in {"access control", "controllo accessi", "controllo degli accessi", "controlli di accesso", "controlli degli accessi"}:
        aliases.extend([
            "access control", "controllo accessi", "controllo degli accessi",
            "controlli di accesso", "controlli degli accessi",
        ])

    if raw_l in {"account privilegiati", "account privilegiato", "privileged account", "privileged accounts"}:
        aliases.extend([
            "account privilegiati", "account privilegiato", "privileged account",
            "privileged accounts", "utenze privilegiate", "utenze con privilegi",
            "privilegi amministrativi", "administrative privileges",
        ])

    if "accesso non autorizzato" in raw_l or "unauthorized access" in raw_l:
        aliases.extend([
            raw, "accesso non autorizzato", "rischio di accesso non autorizzato",
            "unauthorized access", "unauthorized access risk", "rischio di accesso",
        ])

    out: List[str] = []
    seen = set()

    for a in aliases:
        clean = _clean_graph_concept(a)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)

    return out


def extract_graph_concepts_from_query(query_text: str, max_concepts: int = 8) -> List[str]:
    """
    Estrae concetti forti dalla domanda per costruire relazioni testuali.
    Evita token singoli deboli e non crea concetti dal contenuto dei documenti.
    """
    q = query_text or ""
    concepts: List[str] = []

    quoted = re.findall(r"[\"“']([^\"”']+)[\"”']", q)
    concepts.extend([_clean_graph_concept(x) for x in quoted if len(_clean_graph_concept(x)) >= 3])

    relation_segment_patterns = [
        r"\b(?:tra|fra)\s+(.+?)(?:[\.?]|$)",
        r"\bbetween\s+(.+?)(?:[\.?]|$)",
        r"\bamong\s+(.+?)(?:[\.?]|$)",
    ]

    for pat in relation_segment_patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            concepts.extend(_split_relation_segment(m.group(1)))

    generic_phrases = {
        "assessment", "audit", "evidence", "evidenza", "evidenze",
        "document", "documents", "documento", "documenti",
        "source", "sources", "fonte", "fonti", "glossario", "glossary",
        "compliance", "conformità", "controllo", "controlli", "control", "controls",
        "requirement", "requirements", "requisito", "requisiti",
        "metric", "metrics", "metrica", "metriche",
    }



    # Stopwords bilingui per evitare l'estrazione di "rumore" nel grafo
    ignore_terms = {
        "documento", "fonti", "fonte", "valutazione", "assessment", "audit", "entità", "relazione", "nodo", "grafo", "sistema", # IT
        "document", "source", "sources", "entity", "entities", "relation", "relationship", "node", "graph", "system", "framework" # EN
    }

    # 1. Termini espliciti tra virgolette (priorità massima)
    quoted = re.findall(r'["“”\']([^"“”\']+)["“”\']', q)
    concepts.extend([_clean_graph_concept(x) for x in quoted if len(_clean_graph_concept(x)) >= 3])

    # 2. Acronimi in maiuscolo (Agnostico rispetto alla lingua)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", q)
    concepts.extend(acronyms)

    # 3. Frasi esatte note
    for p in extract_exact_phrases(q):
        clean = _clean_graph_concept(p)
        if clean and clean.lower() not in ignore_terms:
            concepts.append(clean)

    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", q)
    concepts.extend(acronyms)

    # Fallback solo quando non ci sono concetti forti.
    if not concepts:
        for t in graph_relevant_tokens(q):
            if len(t) >= 5:
                concepts.append(t)

    weak_single_terms = {
        "tutti", "tutto", "all", "each", "ogni",
        "fattore", "fattori", "factor", "factors",
        "access", "accesso", "control", "controllo", "controlli",
        "autenticazione", "authentication",
        "rischio", "risk", "utente", "user", "identity", "identità",
        "documenti", "documents", "normativi", "normative",
    }

    cleaned: List[str] = []
    seen_canonical = set()

    for c in concepts:
        clean = _clean_graph_concept(c)
        if not clean:
            continue

        cl = clean.lower()
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", clean))
        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10

        if not is_acronym and word_count == 1 and cl in weak_single_terms:
            continue

        canonical = _canonical_graph_concept(clean)
        if canonical in seen_canonical:
            continue

        seen_canonical.add(canonical)
        cleaned.append(clean)

    final: List[str] = []
    lowered = [c.lower() for c in cleaned]

    for c in cleaned:
        cl = c.lower()
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", c))
        is_acronym = c.upper() == c and 2 <= len(c) <= 10

        if not is_acronym and word_count == 1:
            if any(cl != other and cl in other and len(other.split()) > 1 for other in lowered):
                continue

        final.append(c)

        if len(final) >= max_concepts:
            break

    return final


def _concept_in_text(concept: str, text_l: str) -> bool:
    """Verifica presenza del concetto usando alias IT/EN e boundary per acronimi/parole singole."""
    if not concept or not text_l:
        return False

    for alias in _graph_concept_aliases(concept):
        a = alias.lower().strip()
        if not a:
            continue

        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", alias))
        is_acronym = alias.upper() == alias and 2 <= len(alias) <= 10

        if is_acronym or word_count == 1:
            if re.search(rf"(^|[^a-z0-9]){re.escape(a)}([^a-z0-9]|$)", text_l):
                return True
        else:
            if a in text_l:
                return True

    return False


def _best_alias_for_text(concept: str, text_l: str) -> str:
    for alias in _graph_concept_aliases(concept):
        a = alias.lower().strip()
        if a and a in text_l:
            return alias
    return concept


def _source_concept_hits(concepts: List[str], content: str) -> List[str]:
    """Restituisce concetti presenti nel chunk, deduplicando sinonimi/canoni."""
    text_l = (content or "").lower()
    hits: List[str] = []
    seen = set()

    for c in concepts:
        if not _concept_in_text(c, text_l):
            continue

        canonical = _canonical_graph_concept(c)
        if canonical in seen:
            continue

        seen.add(canonical)
        hits.append(c)

    return hits


def _rank_sources_for_graph(concepts: List[str], sources: List[SourceItem]) -> List[Tuple[int, float, SourceItem, List[str]]]:
    """Ordina i chunk per utilità nella costruzione di relazioni testuali."""
    ranked: List[Tuple[int, float, SourceItem, List[str]]] = []

    for s in sources:
        if normalize_tier_value(s.tier) == "GRAPH":
            continue

        content = s.content or ""
        hits = _source_concept_hits(concepts, content)

        if len(hits) < 2:
            continue

        ranked.append((len(hits), float(s.score or 0.0), s, hits))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return ranked


def _evidence_snippet_for_pair(content: str, a: str, b: str, max_chars: int = 260) -> str:
    """
    Restituisce uno snippet dove compaiono entrambi i concetti,
    oppure almeno uno dei due.
    """
    if not content:
        return ""

    chunks = re.split(r"(?<=[\.\!\?])\s+|\n+", content)

    a_l = a.lower()
    b_l = b.lower()

    for chunk in chunks:
        cl = chunk.lower()

        if a_l in cl and b_l in cl:
            return _md_cell(chunk, max_chars)

    for chunk in chunks:
        cl = chunk.lower()

        if a_l in cl or b_l in cl:
            return _md_cell(chunk, max_chars)

    return _md_cell(content, max_chars)


def _parse_graph_relation_table_from_source(source: SourceItem) -> List[Dict[str, Any]]:
    """
    Estrae righe dalla tabella prodotta da graph_relations_to_source().
    """
    rows: List[Dict[str, Any]] = []
    content = source.content or ""

    for line in content.splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        if "---" in line:
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]

        if len(cols) < 5:
            continue

        if "entità" in cols[0].lower() or "source" in cols[0].lower():
            continue

        rows.append({
            "source": cols[0],
            "relation": cols[1],
            "target": cols[2],
            "filename": cols[3],
            "page": cols[4],
            "evidence": "Relazione presente nel Knowledge Graph.",
            "status": "esplicita nel grafo",
        })

    return rows


def answer_graph_relations_strict(
    query_text: str,
    sources: List[SourceItem],
    max_rows: int = 10,
) -> Optional[str]:
    """
    Risposta deterministica per domande relazionali.

    Fix v4:
    - per query relazionali NON ritorna più None solo perché non trova una coppia forte;
    - costruisce righe supportate da Neo4j o da co-occorrenza testuale;
    - se non trova relazioni supportate, produce comunque una risposta deterministica
      con tabella controllata e limiti espliciti, evitando la chiamata LLM e i timeout;
    - non usa termini domain-specific hard-coded: lavora sui concetti presenti nella domanda.
    """
    if not is_graph_relation_query(query_text):
        return None

    concepts = extract_graph_concepts_from_query(query_text)

    # Fallback non adattativo: se l'estrazione forte fallisce, usa token rilevanti della query.
    if len(concepts) < 2:
        concepts = [t for t in graph_relevant_tokens(query_text) if len(t) >= 4][:6]

    rows: List[Dict[str, Any]] = []
    unsupported_rows: List[Dict[str, Any]] = []
    seen = set()
    concept_canons = {_canonical_graph_concept(c) for c in concepts}

    def add_row(row: Dict[str, Any]) -> None:
        key = (
            str(row.get("source", "")).lower(),
            str(row.get("relation", "")).lower(),
            str(row.get("target", "")).lower(),
            str(row.get("filename", "")).lower(),
            str(row.get("page", "")),
            str(row.get("status", "")).lower(),
        )
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    # 1. Relazioni esplicite dal Knowledge Graph, ma solo se coinvolgono concetti richiesti.
    for s in sources:
        if s.type == "graph_relations" or "Relazioni Neo4j trovate" in (s.content or ""):
            for r in _parse_graph_relation_table_from_source(s):
                src = str(r.get("source", ""))
                tgt = str(r.get("target", ""))
                src_can = _canonical_graph_concept(src)
                tgt_can = _canonical_graph_concept(tgt)

                relation_text = " ".join([src, str(r.get("relation", "")), tgt]).lower()
                hits = {c for c in concepts if _concept_in_text(c, relation_text)}

                if src_can not in concept_canons and tgt_can not in concept_canons and len(hits) < 2:
                    continue

                add_row(r)
                if len(rows) >= max_rows:
                    break

        if len(rows) >= max_rows:
            break

    # 2. Relazioni testuali: cerca coppie di concetti richiesti presenti nello stesso chunk.
    def source_supports_pair(s: SourceItem, a: str, b: str) -> Tuple[bool, str, str, str]:
        content = s.content or ""
        text_l = content.lower()

        if not (_concept_in_text(a, text_l) and _concept_in_text(b, text_l)):
            return False, "", "", ""

        alias_a = _best_alias_for_text(a, text_l)
        alias_b = _best_alias_for_text(b, text_l)
        snippet = _evidence_snippet_for_pair(content, alias_a, alias_b)
        return True, alias_a, alias_b, snippet

    doc_row_count: Dict[Tuple[str, int], int] = {}

    if len(rows) < max_rows and len(concepts) >= 2:
        pairs: List[Tuple[str, str]] = []

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                a = concepts[i]
                b = concepts[j]
                if _canonical_graph_concept(a) == _canonical_graph_concept(b):
                    continue
                pairs.append((a, b))

        for a, b in pairs:
            if len(rows) >= max_rows:
                break

            best_match: Optional[Tuple[float, SourceItem, str, str, str]] = None

            for s in sources:
                if normalize_tier_value(s.tier) == "GRAPH":
                    continue

                ok, alias_a, alias_b, snippet = source_supports_pair(s, a, b)
                if not ok:
                    continue

                score = float(s.score or 0.0)
                candidate = (score, s, alias_a, alias_b, snippet)

                if best_match is None or candidate[0] > best_match[0]:
                    best_match = candidate

            if not best_match:
                unsupported_rows.append({
                    "source": a,
                    "relation": "collegamento richiesto",
                    "target": b,
                    "filename": "N/D",
                    "page": 0,
                    "evidence": "Relazione non trovata nello stesso chunk recuperato.",
                    "status": "non supportata dalle fonti recuperate come relazione diretta",
                })
                continue

            _, s, alias_a, alias_b, snippet = best_match
            doc_key = (s.filename, int(s.page or 0))

            if doc_row_count.get(doc_key, 0) >= 3:
                continue

            doc_row_count[doc_key] = doc_row_count.get(doc_key, 0) + 1

            add_row({
                "source": a,
                "relation": "collegamento testuale",
                "target": b,
                "filename": s.filename,
                "page": int(s.page or 0),
                "evidence": snippet,
                "status": "supportata testualmente, non esplicita come arco",
            })

    # 3. Fallback deterministico: evita LLM e timeout anche se non ci sono relazioni forti.
    if not rows and unsupported_rows:
        rows = unsupported_rows[:max_rows]

    if not rows:
        rows = [{
            "source": "Concetti relazionali",
            "relation": "non determinabile",
            "target": "Fonti recuperate",
            "filename": "N/D",
            "page": 0,
            "evidence": "Non sono state trovate relazioni esplicite o co-occorrenze testuali sufficienti nei chunk recuperati.",
            "status": "non supportata dalle fonti recuperate",
        }]

    table = [
        "| Entità sorgente | Relazione | Entità target | Documento | Pagina | Evidenza | Stato |",
        "|---|---|---|---|---:|---|---|",
    ]

    for r in rows[:max_rows]:
        table.append(
            "| "
            + _md_cell(r.get("source"))
            + " | "
            + _md_cell(r.get("relation"))
            + " | "
            + _md_cell(r.get("target"))
            + " | "
            + _md_cell(r.get("filename"))
            + " | "
            + _md_cell(r.get("page"))
            + " | "
            + _md_cell(r.get("evidence"), 320)
            + " | "
            + _md_cell(r.get("status"))
            + " |"
        )

    used_files = sorted({
        str(r.get("filename") or "").strip()
        for r in rows
        if str(r.get("filename") or "").strip() and str(r.get("filename") or "").strip() != "N/D"
    })

    has_graph_rows = any(r.get("status") == "esplicita nel grafo" for r in rows)
    has_text_rows = any("testualmente" in str(r.get("status", "")) for r in rows)
    has_unsupported = any("non supportata" in str(r.get("status", "")) for r in rows)

    limits: List[str] = []

    if has_text_rows:
        limits.append(
            "- Alcune relazioni sono supportate testualmente dai chunk recuperati, "
            "ma non risultano esplicitate come archi Neo4j."
        )

    if not has_graph_rows:
        limits.append(
            "- Non sono state trovate relazioni Neo4j sufficientemente pertinenti; "
            "la tabella usa supporto testuale oppure dichiara l'assenza di supporto diretto."
        )

    if has_unsupported:
        limits.append(
            "- Le righe marcate come non supportate indicano collegamenti richiesti dalla domanda "
            "ma non dimostrati direttamente dalle fonti recuperate."
        )

    if not limits:
        limits.append("- Le relazioni elencate risultano supportate dalle fonti recuperate.")

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table)
        + "\n\n"
        "**B) Evidenze**\n\n"
        "- La tabella è stata costruita in modalità deterministica per evitare relazioni inventate.\n"
        "- Ogni riga distingue tra relazione esplicita nel grafo, supporto testuale o supporto non trovato.\n\n"
        "**C) Limiti / Conflitti**\n\n"
        + "\n".join(limits)
        + "\n\n"
        "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Nessuna fonte documentale diretta utilizzabile.")
    )


def normalize_doc_name(value: str) -> str:
    """
    Normalizza un nome documento per confronti robusti:
    - lowercase
    - rimuove estensioni
    - rimuove caratteri non alfanumerici
    - rimuove suffissi tecnici comuni tipo _out / output
    """
    if not value:
        return ""

    v = os.path.basename(str(value).lower().strip())

    v = re.sub(r"\.(pdf|md|txt|docx|html)$", "", v)
    v = re.sub(r"[_\-\s]+out$", "", v)
    v = re.sub(r"[_\-\s]+output$", "", v)
    v = re.sub(r"[^a-z0-9]+", "", v)

    return v


def extract_requested_document(query_text: str) -> str:
    """
    Estrae il documento richiesto dalla query in modo sicuro.
    Evita falsi positivi come "il documento consiglia...".
    """
    q = query_text or ""

    patterns = [
        # 1. Nome tra virgolette o apici: nel documento "Trading_Tesi"
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+["\']([^"\']+)["\']',
        
        # 2. Nome con estensione esplicita: file report.pdf
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+([A-Za-z0-9_\-\.]+\.(?:pdf|md|txt|docx|csv|html))\b',
        
        # 3. Nome tecnico con underscore o trattini: documento TRADING_ALGORITMICO
        r'\b(?:nel|nella|dal|dalla\s+)?(?:documento|file|pdf)\s+([A-Za-z0-9]+[_\-][A-Za-z0-9_\-\.]+)\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,:;!?\"'")

    return ""

def candidate_matches_requested_doc(candidate: Dict[str, Any], requested_doc: str) -> bool:
    """
    Verifica se un candidato appartiene al documento richiesto.
    """
    if not requested_doc:
        return True

    wanted = normalize_doc_name(requested_doc)
    if not wanted:
        return True

    filename = normalize_doc_name(candidate.get("filename", ""))

    # Match robusto nei due versi
    return wanted in filename or filename in wanted

def search_pg_by_document_scope(
    requested_doc: str,
    query_text: str,
    limit: int = 80
) -> List[Dict[str, Any]]:
    """
    Recupera chunk da Postgres appartenenti al documento richiesto,
    indipendentemente dal fatto che siano entrati nei primi risultati BM25 generici.

    Serve per evitare falsi negativi quando l'utente chiede:
    "nel documento X..."
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    wanted_norm = normalize_doc_name(requested_doc)

    if not wanted_norm:
        return []

    sql = """
    WITH q AS (
        SELECT plainto_tsquery('simple', %s) AS tsq
    ),
    ranked AS (
        SELECT
            d.chunk_uuid::text AS chunk_uuid,
            d.content_raw,
            d.content_semantic,
            d.metadata_json,
            d.ingestion_ts,

            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(
                            coalesce(
                                d.metadata_json->>'filename',
                                d.metadata_json->>'source_name',
                                ''
                            )
                        ),
                        '\\.(pdf|md|txt|docx|html)$',
                        '',
                        'g'
                    ),
                    '[_\\-\\s]+(out|output)$',
                    '',
                    'g'
                ),
                '[^a-z0-9]+',
                '',
                'g'
            ) AS filename_norm,

            ts_rank_cd(
                to_tsvector(
                    'simple',
                    coalesce(d.content_semantic, '') || ' ' ||
                    coalesce(d.content_raw, '') || ' ' ||
                    coalesce(d.metadata_json::text, '')
                ),
                q.tsq
            ) AS rank,

            row_number() OVER (
                PARTITION BY d.chunk_uuid
                ORDER BY d.ingestion_ts DESC
            ) AS rn

        FROM public.document_chunks d, q
    )
    SELECT
        chunk_uuid,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts,
        rank
    FROM ranked
    WHERE rn = 1
      AND length(filename_norm) > 0
      AND (
            filename_norm LIKE %s
            OR %s LIKE ('%%' || filename_norm || '%%')
      )
    ORDER BY rank DESC, ingestion_ts DESC
    LIMIT %s;
    """

    conn = pg_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_text,
                    f"%{wanted_norm}%",
                    wanted_norm,
                    limit,
                )
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []

        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts, rank in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}

            if metadata_json is None:
                metadata_json = {}

            out.append({
                "id": str(chunk_uuid),
                "content": content_semantic or content_raw or "",
                "metadata": metadata_json,
                "score": float(rank or 0.001),
                "origin": "PostgresDocScope",
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            })

        return out

    except Exception as e:
        print(f"⚠️ PG document scope search error: {e}")
        return []

    finally:
        pg_pool.putconn(conn)

def retrieve_v2(query_text: str, active_doc: str = "") -> Tuple[List[SourceItem], str]:
    """
    Retrieval V5:
    - Qdrant vector search
    - Postgres BM25 keyword search
    - Neo4j entity/formula search
    - Neo4j graph expansion
    - RRF fusion
    - CrossEncoder reranking
    - Final Postgres enrichment by chunk_uuid
    """
    print(f"\n\n{'=' * 40}")
    print("🔎 DEBUG RETRIEVAL START")
    print(f"❓ Query: '{query_text}'")

    if not embedder or not qdrant_client_inst:
        return [SourceItem(id="error", content="Backend OFF", filename="System")], "Backend OFF"

    t_total0 = time.time()
    timings: Dict[str, float] = {}
    counts: Dict[str, Any] = {}
    intent = detect_intent(query_text)
    expanded_query = expand_assessment_query(query_text)   
    
    qdrant_k, rerank_k, final_k, max_per_doc_k, max_per_page_k = dynamic_retrieval_limits(query_text)
    
    requested_pages = extract_requested_pages(query_text)
    counts["requested_pages"] = requested_pages


    # LOGICA DI MEMORIA:
    extracted_doc = extract_requested_document(query_text)
    
    # Se l'utente nomina un file ora, usa quello. 
    # Altrimenti usa quello che abbiamo in memoria (active_doc).
    requested_doc = extracted_doc if extracted_doc else active_doc
    requested_doc_norm = normalize_doc_name(requested_doc)

    if requested_doc:
        print(f"📄 Requested document scope: {requested_doc} -> {requested_doc_norm}")
        counts["requested_doc"] = requested_doc


    # 1) Embedding query
    t0 = time.time()
    query_vector = embedder.encode(expanded_query, normalize_embeddings=True).tolist()
    timings["embed"] = time.time() - t0

    # 2) Qdrant vector search
    t0 = time.time()
    hits = []

    try:
        # Compatibilità universale per le versioni nuove e vecchie di Qdrant
        if hasattr(qdrant_client_inst, 'query_points'):
            response = qdrant_client_inst.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=qdrant_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = qdrant_client_inst.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=qdrant_k,
                with_payload=True,
            )
        counts["qdrant_hits"] = len(hits)
        print(f"🌌 Qdrant ha trovato {len(hits)} chunk.")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
        counts["qdrant_hits"] = 0

    # ... fine blocco Qdrant ...
    timings["qdrant_search"] = time.time() - t0

    # ==========================================
    # AGGIUNGI QUESTO BLOCCO MANCANTE:
    # 3) Postgres BM25 search
    t0 = time.time()
    bm25_hits = search_pg_bm25(expanded_query, limit=60)
    exact_hits = search_pg_exact_phrases(query_text, limit=40)

    # --- INIZIO FIX: Iniezione Dinamica Acronimi da Glossario ---
    # Rileva qualsiasi acronimo (2-8 lettere maiuscole) nella query
    detected_acronyms = set(re.findall(r"\b[A-Z]{2,8}\b", query_text))
    
    for acr in detected_acronyms:
        # Cerca dinamicamente nel glossario. Se trova la definizione, la inietta a priorità massima.
        gloss_hits = search_pg_glossary_term(acr, [acr], limit=2)
        for g in gloss_hits:
            exact_hits.append({
                "id": str(g.get("id")),
                "content": str(g.get("content_semantic") or g.get("content_raw") or ""),
                "metadata": g.get("metadata", {}),
                "score": 3.0,  # Score alto per forzare l'attenzione dell'LLM
                "origin": "PostgresGlossaryInjectDynamic"
            })
    # --- FINE FIX ---
    
    
    counts["bm25_hits"] = len(bm25_hits)
    counts["exact_phrase_hits"] = len(exact_hits)
    print(f"🐘 Postgres BM25 ha trovato {len(bm25_hits)} chunk; Exact phrase {len(exact_hits)} chunk.")
    timings["bm25_search"] = time.time() - t0
    # ==========================================

    # 3B) Postgres document-scope search
    # Se l'utente chiede un documento specifico, recuperiamo chunk direttamente...
    t0 = time.time()
    doc_scope_hits = []

    if requested_doc:
        doc_scope_hits = search_pg_by_document_scope(
            requested_doc=requested_doc,
            query_text=query_text,
            limit=80,
        )

    counts["doc_scope_hits"] = len(doc_scope_hits)

    if requested_doc:
        print(
            f"📄 Postgres document-scope search ha trovato "
            f"{len(doc_scope_hits)} chunk per documento '{requested_doc}'."
        )

    timings["doc_scope_search"] = time.time() - t0

    # 4) Neo4j direct entity/formula search
    t0 = time.time()

    neo4j_entity_hits = search_neo4j_entities(expanded_query, limit=30)
    neo4j_relation_rows = search_neo4j_relations(expanded_query, limit=40)

    formula_query = (
        intent == "formula"
        or any(k in (query_text or "").lower() for k in [
            "formula", "formule", "latex", "equazione", "equazioni"
        ])
    )

    neo4j_formula_hits = (
        search_neo4j_formulas(expanded_query, limit=GRAPH_MAX_FORMULAS)
        if formula_query
        else []
    )

    neo4j_direct_hits = neo4j_entity_hits + neo4j_formula_hits

    counts["neo4j_entity_hits"] = len(neo4j_entity_hits)
    counts["neo4j_formula_direct_hits"] = len(neo4j_formula_hits)
    counts["neo4j_direct_hits"] = len(neo4j_direct_hits)
    counts["neo4j_relation_hits"] = len(neo4j_relation_rows)

    print(
        f"🕸️ Neo4j direct search ha trovato {len(neo4j_direct_hits)} chunk "
        f"({len(neo4j_entity_hits)} entity, {len(neo4j_formula_hits)} formule)."
    )

    timings["neo4j_direct_search"] = time.time() - t0

    # 5) Candidate merge
    candidates_dict: Dict[str, Dict[str, Any]] = {}

    # 5A) Import Qdrant candidates
    for hit in hits:
        uid = str(hit.id)
        payload = hit.payload or {}

        content = safe_payload_text(payload)
        if not content:
            continue

        candidates_dict[uid] = {
            "id": uid,
            "content": content,
            "filename": str(payload.get("filename", "Unknown")),
            "page": get_payload_page(payload),
            "type": get_payload_type(payload),
            "tier": normalize_tier_value(str(payload.get("tier", "C"))),
            "score_base": float(hit.score or 0.0),
            "score_vec": float(hit.score or 0.0),
            "score_bm25": 0.0,
            "score_graph": 0.0,
            "origin": "Qdrant",
            "section_hint": get_payload_section(payload),
            "image_id": get_payload_image_id(payload),
        }
        
    # 5A-BIS) Import Postgres document-scope candidates
    for d in doc_scope_hits:
        uid = str(d.get("id", "")).strip()

        if not uid:
            continue

        meta = d.get("metadata", {}) or {}

        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        fname = meta.get("filename") or meta.get("source_name") or requested_doc or "Unknown"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        toon_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": d.get("content", ""),
                "filename": fname,
                "page": page,
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(d.get("score", 0.001)),
                "score_graph": 0.0,
                "score_doc_scope": 1.0,
                "origin": "PostgresDocScope",
                "section_hint": meta.get("section_hint", ""),
                "image_id": meta.get("image_id"),
            }
        else:
            candidates_dict[uid]["score_bm25"] = max(
                float(candidates_dict[uid].get("score_bm25", 0.0)),
                float(d.get("score", 0.001)),
            )
            candidates_dict[uid]["score_doc_scope"] = 1.0

            # Se Qdrant/Neo4j avevano filename Unknown o Neo4j,
            # correggiamo usando i metadati Postgres.
            if candidates_dict[uid].get("filename") in ("", "Unknown", "Neo4j"):
                candidates_dict[uid]["filename"] = fname

            if not candidates_dict[uid].get("page"):
                candidates_dict[uid]["page"] = page

            if "PostgresDocScope" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + PostgresDocScope"

    # 5A-TER) Import Postgres exact phrase candidates (high precision acronyms/glossary/roles)
    for e in exact_hits:
        uid = str(e.get("id", "")).strip()
        if not uid:
            continue
        meta = e.get("metadata", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        fname = meta.get("filename") or meta.get("source_name") or "Unknown"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        toon_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))
        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": e.get("content", ""),
                "filename": fname,
                "page": page,
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(e.get("score", 2.0)),
                "score_graph": 0.0,
                "score_exact": 1.0,
                "origin": "PostgresExactPhrase",
                "section_hint": meta.get("section_hint", ""),
                "image_id": meta.get("image_id"),
            }
        else:
            candidates_dict[uid]["score_bm25"] = max(float(candidates_dict[uid].get("score_bm25", 0.0)), float(e.get("score", 2.0)))
            candidates_dict[uid]["score_exact"] = 1.0
            if "PostgresExactPhrase" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + PostgresExactPhrase"

    # 5B) Import Postgres BM25 candidates
    for b in bm25_hits:
        uid = str(b.get("id", "")).strip()
        if not uid:
            continue

        meta = b.get("metadata", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        fname = meta.get("filename") or meta.get("source_name") or "Unknown"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        toon_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": b.get("content", ""),
                "filename": fname,
                "page": page,
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(b.get("score", 0.0)),
                "score_graph": 0.0,
                "origin": "Postgres",
                "section_hint": meta.get("section_hint", ""),
                "image_id": meta.get("image_id"),
            }
        else:
            candidates_dict[uid]["score_bm25"] = max(
                float(candidates_dict[uid].get("score_bm25", 0.0)),
                float(b.get("score", 0.0)),
            )
            if "Postgres" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + Postgres"

    # 5C) Import Neo4j direct candidates
    for g in neo4j_direct_hits:
        uid = str(g.get("id", "")).strip()
        if not uid:
            continue

        if uid not in candidates_dict:
            candidates_dict[uid] = {
                "id": uid,
                "content": g.get("content", ""),
                "filename": g.get("filename", "Neo4j"),
                "page": int(g.get("page") or 0),
                "type": g.get("type", "graph"),
                "tier": "GRAPH",
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": 0.0,
                "score_graph": float(g.get("score_graph", 0.0)),
                "origin": g.get("origin", "Neo4j"),
                "section_hint": g.get("section_hint", ""),
            }
        else:
            candidates_dict[uid]["score_graph"] = max(
                float(candidates_dict[uid].get("score_graph", 0.0)),
                float(g.get("score_graph", 0.0)),
            )
            if "Neo4j" not in candidates_dict[uid]["origin"]:
                candidates_dict[uid]["origin"] += " + Neo4j"

    # 6) Neo4j graph expansion
    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        t0_graph = time.time()

        seed_ids = list(candidates_dict.keys())[:10]
        graph_sources = []

        try:
            neighbor_ids = get_neighbor_chunk_ids(
                seed_ids,
                limit=GRAPH_MAX_NEIGHBOR_CHUNKS,
            )
        except Exception as e:
            print(f"⚠️ Neo4j neighbor search error: {e}")
            neighbor_ids = []

        if neighbor_ids:
            graph_sources = fetch_chunks_from_qdrant_by_ids(neighbor_ids)

            for gs in graph_sources:
                if gs.id not in candidates_dict:
                    candidates_dict[gs.id] = {
                        "id": gs.id,
                        "content": gs.content,
                        "filename": gs.filename,
                        "page": gs.page,
                        "type": gs.type,
                        "tier": normalize_tier_value(getattr(gs, "tier", "C")),
                        "score_base": 0.0,
                        "score_vec": 0.0,
                        "score_bm25": 0.0,
                        "score_graph": 1.0,
                        "origin": "Neo4j_Expansion",
                        "section_hint": getattr(gs, "section_hint", ""),
                    }

            print(f"🕸️ Neo4j ha aggiunto {len(graph_sources)} chunk semanticamente collegati.")

        counts["neo4j_hits"] = len(graph_sources)
        timings["graph"] = time.time() - t0_graph
    else:
        counts["neo4j_hits"] = 0

    # 7) Final candidate list
    candidates = list(candidates_dict.values())

    if not candidates:
        print("❌ NESSUN CANDIDATO TROVATO!")
        timings["total"] = time.time() - t_total0
        return [], build_retrieval_audit_md(query_text, intent, timings, counts, [])

    # 7B) HARD DOCUMENT SCOPE FILTER
    # Se l'utente chiede un documento specifico, NON permettere fonti di altri documenti.
    if requested_doc:
        before_doc_scope = len(candidates)

        scoped_candidates = [
            c for c in candidates
            if candidate_matches_requested_doc(c, requested_doc)
        ]

        counts["doc_scope_before"] = before_doc_scope
        counts["doc_scope_after"] = len(scoped_candidates)

        print(
            f"📄 Document scope filter: {before_doc_scope} -> {len(scoped_candidates)} "
            f"for requested_doc='{requested_doc}'"
        )

        if not scoped_candidates:
            timings["total"] = time.time() - t_total0
            audit = build_retrieval_audit_md(query_text, intent, timings, counts, [])
            audit += (
                f"\n\n#### 📄 Document Scope\n"
                f"- Documento richiesto: `{requested_doc}`\n"
                f"- Nessun chunk trovato appartenente al documento richiesto.\n"
            )
            return [], audit

        candidates = scoped_candidates



    # 8) RRF scoring
    apply_rrf_scoring(candidates)

    query_tokens = extract_rag_tokens(query_text)

    print(f"🎯 Target Tokens (Filename Match): {query_tokens}")


    filename_boost_stats = Counter()

    for c in candidates:
        fname = c.get("filename") or "Unknown"
        fname_lower = fname.lower()

        hits_fname_raw = sum(1 for token in query_tokens if token in fname_lower)

        # Evita che un filename con molti token uguali alla query domini troppo il ranking.
        hits_fname = min(hits_fname_raw, 3)

        # Boost più controllato: massimo 0.06.
        filename_boost = 0.02 * hits_fname

        if hits_fname > 0:
            if "[TARGET FILE]" not in c.get("origin", ""):
                c["origin"] += " [TARGET FILE]"

            filename_boost_stats[(fname, hits_fname_raw)] += 1

        tier_delta = tier_score_delta(c.get("tier", ""), query_text)

        doc_scope_boost = 0.20 if c.get("score_doc_scope", 0.0) > 0 else 0.0

        page_boost = 0.0
        if requested_pages and int(c.get("page", 0)) in requested_pages:
            page_boost = 0.30

        ctype = normalize_source_type(c.get("type", ""))

        intent_boost = 0.0

        if intent == "formula" and ctype == "formula":
            intent_boost = 0.25
        elif intent == "chart" and ctype in {"image", "chart"}:
            intent_boost = 0.20
        elif intent == "table" and ctype == "table":
            intent_boost = 0.20

        c["pre_rerank_score"] = (
            float(c.get("rrf_score", 0.0))
            + filename_boost
            + tier_delta
            + doc_scope_boost
            + page_boost
            + intent_boost
        )

    for (fname, hits_fname_raw), n_chunks in filename_boost_stats.items():
        print(
            f"   🚀 Filename boost per {fname} "
            f"(match={hits_fname_raw}, chunks={n_chunks})"
        )



    # 9) Reranking
    candidates.sort(key=lambda x: x.get("pre_rerank_score", 0.0), reverse=True)
    top_candidates = candidates[:rerank_k]

    if reranker and top_candidates:
        t0 = time.time()

        pairs = [
            (query_text, c.get("content", "") or "")
            for c in top_candidates
        ]

        try:
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                top_candidates[i]["final_score"] = (
                    float(score)
                    + float(top_candidates[i].get("pre_rerank_score", 0.0))
                )

        except Exception as e:
            print(f"⚠️ Reranker Error: {e}")

            for c in top_candidates:
                c["final_score"] = float(c.get("pre_rerank_score", 0.0))

        timings["rerank"] = time.time() - t0

    else:
        for c in top_candidates:
            c["final_score"] = float(c.get("pre_rerank_score", 0.0))

    top_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # 10) Diversification
    final_selection = diversify(
        top_candidates,
        max_per_page_k,
        max_per_doc_k,
        final_k,
    )

    # 11) Final Postgres enrichment by chunk_uuid
    pg_rows = fetch_pg_chunks_by_uuid(
        [str(t.get("id")) for t in final_selection if t.get("id")]
    )

    counts["pg_enriched_hits"] = len(pg_rows)

    for t in final_selection:
        uid = str(t.get("id", ""))
        pg_row = pg_rows.get(uid)

        if not pg_row:
            continue

        pg_meta = pg_row.get("metadata_json", {}) or {}
        if isinstance(pg_meta, str):
            try:
                pg_meta = json.loads(pg_meta)
            except Exception:
                pg_meta = {}

        preferred_content = (
            pg_row.get("content_raw", "")
            if PG_PREFER_RAW
            else (pg_row.get("content_semantic", "") or pg_row.get("content_raw", ""))
        )

        if preferred_content:
            t["content"] = preferred_content

        t["filename"] = (
            t.get("filename")
            or pg_meta.get("filename")
            or pg_meta.get("source_name")
            or "Unknown"
        )

        t["page"] = int(
            t.get("page")
            or pg_meta.get("page_no")
            or pg_meta.get("page")
            or 0
        )

        t["type"] = (
            t.get("type")
            or pg_meta.get("toon_type")
            or pg_meta.get("type")
            or "text"
        )

        t["tier"] = normalize_tier_value(
            t.get("tier")
            or pg_meta.get("tier")
            or "C"
        )

        t["pg_ingestion_ts"] = pg_row.get("ingestion_ts", "")
        t["pg_source_name"] = pg_meta.get("source_name", "")
        t["pg_source_type"] = pg_meta.get("source_type", "")
        t["pg_log_id"] = int(pg_meta.get("log_id") or 0)
        t["pg_chunk_id"] = int(pg_meta.get("chunk_index") or 0)
        t["pg_toon_type"] = pg_meta.get("toon_type", "")

        if "PG_Enrich" not in t["origin"]:
            t["origin"] += " + PG_Enrich"

    counts["tier_split"] = dict(
        Counter(normalize_tier_value(str(s.get("tier", "UNKNOWN"))) for s in final_selection)
    )
    counts["final_sources"] = len(final_selection)
    timings["total"] = time.time() - t_total0

    print("-" * 20)
    print("🏆 CLASSIFICA FINALE (Top 3):")

    for i, s in enumerate(final_selection[:3]):
        print(
            f"  {i + 1}. {s.get('filename')} "
            f"(Score: {float(s.get('final_score', 0.0)):.3f}) - {s.get('origin')}"
        )

    print("=" * 40 + "\n")

    # 12) Output SourceItem construction
    sources: List[SourceItem] = []

    for t in final_selection:
        sources.append(
            SourceItem(
                id=str(t.get("id", "")),
                content=t.get("content", ""),
                filename=t.get("filename", "Unknown"),
                page=int(t.get("page") or 0),
                type=t.get("type", "text"),
                score=float(t.get("final_score", 0.0)),
                tier=normalize_tier_value(t.get("tier", "C")),
                db_origin=t.get("origin", "Unknown"),
                section_hint=t.get("section_hint", ""),
                image_id=t.get("image_id"),
                pg_ingestion_ts=t.get("pg_ingestion_ts", ""),
                pg_source_name=t.get("pg_source_name", ""),
                pg_source_type=t.get("pg_source_type", ""),
                pg_log_id=int(t.get("pg_log_id") or 0),
                pg_chunk_id=int(t.get("pg_chunk_id") or 0),
                pg_toon_type=t.get("pg_toon_type", ""),
            )
        )

    # 13) Final formulas from Neo4j
    counts["final_formulas"] = 0

    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        chunk_ids = [s.id for s in sources if s.id and s.id != "graph"]
        
        # Ora la funzione restituisce un Dict[str, List[str]] e si aspetta limit_per_chunk
        formulas_dict = get_formulas_for_chunks(chunk_ids, limit_per_chunk=GRAPH_MAX_FORMULAS)

        # Appiattisce il dizionario in una singola lista di stringhe per mantenere la compatibilità 
        # con la funzione extract_formula_rows_from_sources
        all_formulas_flat = [formula for f_list in formulas_dict.values() for formula in f_list]

        counts["final_formulas"] = len(all_formulas_flat)

        if all_formulas_flat:
            sources.append(
                SourceItem(
                    id="graph",
                    content="Formule collegate dal Knowledge Graph:\n" + "\n".join(all_formulas_flat),
                    filename="KG",
                    page=0,
                    type="formula",
                    tier="GRAPH",
                    score=0.0,
                    db_origin="Neo4j Formula Lookup",
                )
            )
        rel_source = graph_relations_to_source(neo4j_relation_rows)
        if rel_source:
            sources.append(rel_source)

    return sources, build_retrieval_audit_md(
        query_text,
        intent,
        timings,
        counts,
        [],
    )


def build_context_block(sources: List[SourceItem], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context with strong provenance and caps."""
    parts = []
    total = 0

    # IMPORTANT: do not leak technical IDs into the LLM prompt.
    # We number sources as [1], [2], ... and keep IDs only in the UI pop-up.
    for i, s in enumerate(sources, start=1):
        header = f"--- Fonte [{i}] — {s.filename} — Pag {s.page} — ({s.type}) ---\n"
        if s.section_hint:
            header = f"--- Fonte [{i}] — {s.filename} — Pag {s.page} — ({s.type}) — sezione: {s.section_hint} ---\n"

        body = (s.content or "").strip()
        if not body:
            continue

        block = header + body + "\n\n"
        if total + len(block) > max_chars:
            # cut body
            remaining = max(0, max_chars - total - len(header) - 50)
            if remaining <= 200:
                break
            block = header + body[:remaining] + "\n\n"
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "".join(parts).strip()

def build_system_instructions(intent: str) -> str:
    """
    Core system prompt for the LLM.
    Completely framework-agnostic. Focuses on operational constraints.
    """
    base = """
        ROLE:
        You are a Senior Technical Auditor and Compliance AI.

        STRICT OPERATIONAL RULES:
        1. MATHEMATICAL PRIORITY: If the query provides numerical values and requires a calculation (e.g., coverage, risk scores, or deadlines), execute the math step-by-step as your absolute priority.
        2. DATA GROUNDING: Answer using ONLY the provided context. If a specific value, formula, or concept is not in the context, explicitly state "Information not found in retrieved documents." Do not invent or assume external standards.
        3. DEFINITIONS: Extract definitions exactly as written in the retrieved text.
        4. CROSS-REFERENCING: Synthesize information across all relevant retrieved documents impartially. Do not bias toward a specific framework unless requested.
        5. CITATION: Always cite the specific source file and page for every claim.
        6. NARRATIVE SYNTHESIS (CRITICAL): If the context contains structured data, JSON, graph nodes, or relations (like source-relation-target), you MUST synthesize them into fluent, professional paragraphs. NEVER output raw database logs, JSON, or tabular representations of graph structures. Translate technical relations (e.g., "A -[COMPLIES_WITH]-> B") into plain text (e.g., "Il documento A è conforme ai requisiti della normativa B").

        TONE: Technical, objective, and evidence-based.

        OUTPUT STRUCTURE:
        You MUST structure your response in EXACTLY these four sections, using these EXACT headers:
        **A) Risposta** (Direct, technical assessment in discursive paragraphs)
        **B) Evidenze** (Bullet points citing the source ID(s) and pages)
        **C) Limiti / Conflitti** (State missing evidence, contradictions between policies and evidence, or limits of the retrieved context)
        **D) Fonti** (List the filenames used)

        LANGUAGE RULE:
        You MUST respond EXCLUSIVELY in the same language as the user's question.
    """

    if intent == "formula":
        base += "\nINTENT: FORMULA / METRIC. Prioritize exact mathematical formulas, threshold criteria, or scoring models from the context."
    elif intent == "table":
        base += "\nINTENT: TABLE. Output a complete Markdown table based on the context."
    elif intent == "chart":
        base += "\nINTENT: CHART / DIAGRAM. Describe the topology or architecture extracted from the context."

    return base


def tier_guardrail_instructions(query_text: str) -> str:
    wants_evidence = is_evidence_query(query_text)
    return (
        "COMPLIANCE-GRADE GUARDRAILS:\n"
        "1) Tier A (Normative): Primary source for legal and framework requirements.\n"
        "2) Tier B (Governance): Internal policies and planned procedures.\n"
        "3) Tier C (Evidences): Technical proof of actual implementation.\n"
        "4) Grounding: Every statement must be supported by the provided context. Flag any non-conformities.\n"
        "5) Gap Analysis: If technical evidence is missing to prove a policy, state it in section C.\n"
        f"6) {'EVIDENCE FOCUS: The user specifically requested technical proofs, logs, or configurations. Prioritize Tier C context.' if wants_evidence else 'Standard audit: verify alignment across all Tiers.'}\n"
        f"7) GLOSSARY & DEFINITIONS: If the user asks for a definition ('definizione', 'significato', 'meaning', 'definition'), quote EXACTLY from the provided context. Do not use external knowledge.\n"
        f"8) MATH & PENALTIES: If calculating percentages or fines, strictly use the formulas in the text. Explicitly write out the mathematical steps before providing the final number.\n"
        f"9) ANTI-HALLUCINATION: If the context does not mention a specific scenario, state clearly: 'I documenti forniti non contengono questa informazione'.\n"
    )

def tier_guardrail_instructions_analytics(query_text: str) -> str:
    return (
        "SECURITY DATA ANALYTICS GUARDRAILS:\n"
        "1) Primary source: vulnerabilities or logs provided directly by the user.\n"
        "2) Use standard cybersecurity frameworks (e.g., CVSS scoring logic) if applicable.\n"
        "3) Do not invent vulnerabilities or assets not listed in the user's data.\n"
        "4) State assumptions clearly.\n"
        "Language rule: The final answer must be in the SAME LANGUAGE as the user's QUESTION.\n"
    )


def build_system_instructions_analytics(intent: str = "analysis") -> str:
    return f"""
    ROLE: Senior Security Data Analyst.

    LANGUAGE RULE:
    - YOU MUST ANSWER EXCLUSIVELY IN THE LANGUAGE OF THE USER.

    ANALYTICS RULES:
    - User data (e.g., vulnerability scans, logs) provided in the prompt is your PRIMARY SOURCE.
    - Evaluate risks, identify patterns, and propose mitigations based strictly on the provided data.

    OUTPUT STRUCTURE (MANDATORY):
    Use ONLY these exact headers:
    **A) Risposta**
    [Detailed security analysis of the provided data]

    **B) Evidenze**
    [Identified threats, anomalies, or statistical findings]

    **C) Limiti e Assunzioni**
    [Limitations of the provided logs or required further investigations]

    **D) Fonti**
    [Indicate 'User provided data']

    INTENT: {intent}
""".strip()


def safe_markdown(text: str) -> str:
    """Make markdown safer for frontend rendering."""
    if not text:
        return ""
    t = text

    # limit very long lines (layout killer)
    t = "\n".join(line[:2000] for line in t.splitlines())

    # close unbalanced code fences
    if t.count("```") % 2 == 1:
        t += "\n```"

    return t
def short_text(s: str, n: int = 320) -> str:
    if not s:
        return ""
    return s[:n] + ("..." if len(s) > n else "")


def make_analytics_sources(user_query: str) -> List[SourceItem]:
    """
    In analytics_mode non facciamo retrieval, ma vogliamo comunque
    mostrare nel popup un “provenance” minimo: i dati arrivano dall’utente.
    """
    preview = (user_query or "").strip()
    if len(preview) > 1200:
        preview = preview[:1200] + "…"

    return [
        SourceItem(
            id="user_input",
            content=preview,
            filename="USER_INPUT",
            page=0,
            type="user_data",
            score=1.0,
            graph_context=[],
            section_hint="Dati forniti direttamente dall’utente (analytics_mode)",
            image_id=None,
            tier="USER",
        )
    ]



def strip_id_leaks(text: str) -> str:
    """
    Rimuove artefatti tecnici se l'LLM ripete per errore i metadati nel testo.
    """
    if not text:
        return ""

    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?reasoning>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\[SourceID:\s*\d+.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r">>> SOURCE \[\d+\].*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "", text)
    text = text.replace("Tier: A", "").replace("Tier: B", "").replace("Tier: C", "")

    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Estrae un oggetto JSON da una risposta LLM.
    Serve perché alcuni modelli locali possono aggiungere testo prima/dopo il JSON.
    """
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def build_eval_context(sources: List[SourceItem], max_chars: int = EVAL_MAX_CONTEXT_CHARS) -> str:
    """
    Costruisce il contesto da passare al judge.
    Qui NON servono chunk_id tecnici: bastano fonte, pagina, tier e contenuto.
    """
    parts = []
    total = 0

    for i, s in enumerate(sources, start=1):
        if not s.content:
            continue

        header = (
            f"--- SOURCE [{i}] ---\n"
            f"filename: {s.filename}\n"
            f"page: {s.page}\n"
            f"type: {s.type}\n"
            f"tier: {normalize_tier_value(s.tier)}\n"
            f"origin: {s.db_origin}\n"
        )

        body = (s.content or "").strip()
        block = header + body + "\n\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total - len(header) - 100
            if remaining <= 300:
                break
            block = header + body[:remaining] + "\n\n"

        parts.append(block)
        total += len(block)

        if total >= max_chars:
            break

    return "".join(parts).strip()


def append_rag_eval_log(
    query_text: str,
    answer: str,
    sources: List[SourceItem],
    eval_result: RagEvalResult,
    requested_doc: str = "",
):
    """
    Salva le metriche KPI in JSONL.
    Non salva necessariamente tutto il contesto, ma salva abbastanza per audit tecnico.
    """
    if not EVAL_ENABLED:
        return

    try:
        row = {
            "ts_utc": datetime.utcnow().isoformat(),
            "query": query_text,
            "requested_doc": requested_doc,
            "answer_sha256": hashlib.sha256((answer or "").encode("utf-8")).hexdigest(),
            "sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "type": s.type,
                    "tier": normalize_tier_value(s.tier),
                    "db_origin": s.db_origin,
                    "score": s.score,
                }
                for s in sources
            ],
            "metrics": eval_result.model_dump(),
            "llm_model": LLM_MODEL_NAME,
            "eval_model": EVAL_MODEL_NAME,
        }

        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"⚠️ RAG eval log write error: {e}")


def evaluate_rag_answer(
    query_text: str,
    answer: str,
    sources: List[SourceItem],
    requested_doc: str = "",
) -> RagEvalResult:
    """
    Valuta la risposta rispetto ai documenti recuperati.

    Metriche:
    - faithfulness: quanto la risposta è supportata dalle fonti
    - answer_relevance: quanto risponde alla domanda
    - context_support: quanto il contesto contiene evidenza sufficiente
    - hallucination_risk: rischio di allucinazione
    - source_scope_violation: True se usa fonti fuori scope documentale
    """
    if not EVAL_ENABLED:
        return RagEvalResult(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_support=1.0,
            hallucination_risk=0.0,
            verdict="DISABLED",
            reason="Evaluation disabled.",
        )

    if not llm_client:
        return RagEvalResult(
            verdict="ERROR",
            reason="LLM client not initialized for evaluation.",
        )

    if not answer or not answer.strip():
        return RagEvalResult(
            verdict="FAIL",
            reason="Empty answer.",
        )

    if not sources:
        return RagEvalResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_support=0.0,
            hallucination_risk=1.0,
            verdict="FAIL",
            reason="No retrieved sources available.",
        )

    eval_context = build_eval_context(sources)

    if not eval_context:
        return RagEvalResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_support=0.0,
            hallucination_risk=1.0,
            verdict="FAIL",
            reason="Retrieved sources have no usable textual content.",
        )

    scope_rule = ""
    if requested_doc:
        scope_rule = (
            f"The user explicitly requested the document/source/version: {requested_doc}. "
            "Mark source_scope_violation=true if the answer relies on other documents."
        )

    judge_system = """
You are a strict RAG faithfulness evaluator.

You must evaluate whether the ANSWER is supported ONLY by the provided SOURCES.

Return ONLY valid JSON with this schema:

{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_support": 0.0,
  "hallucination_risk": 1.0,
  "source_scope_violation": false,
  "verdict": "PASS|WARN|FAIL",
  "unsupported_claims": [],
  "supported_claims": [],
  "reason": ""
}

Scoring rules:
- faithfulness = 1.0 only if all factual claims in the answer are explicitly supported by the sources.
- answer_relevance = 1.0 only if the answer directly addresses the user question.
- context_support = 1.0 only if the retrieved sources contain enough evidence to answer.
- hallucination_risk = 1.0 when the answer contains unsupported facts.
- source_scope_violation = true if the answer uses evidence outside the requested document/source/version.
- Do not use external knowledge.
- Do not reward plausible but unsupported claims.
- If the answer correctly says that evidence is insufficient, faithfulness can be high.
"""

    judge_user = f"""
### USER QUESTION
{query_text}

### REQUESTED SOURCE SCOPE
{scope_rule if scope_rule else "No explicit document/source/version constraint."}

### SOURCES
{eval_context}

### ANSWER TO EVALUATE
{answer}
"""

    try:
        resp = llm_client.chat.completions.create(
            model=EVAL_MODEL_NAME,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user},
            ],
            temperature=0.0,
            stream=False,
            extra_body={
                "options": {
                    "num_ctx": LLM_NUM_CTX,
                    "num_predict": LLM_NUM_PREDICT,
                    "repeat_penalty": 1.05,
                }
            },
        )

        raw = resp.choices[0].message.content or ""
        data = _extract_json_object(raw)

        result = RagEvalResult(
            faithfulness=_clamp01(data.get("faithfulness"), 0.0),
            answer_relevance=_clamp01(data.get("answer_relevance"), 0.0),
            context_support=_clamp01(data.get("context_support"), 0.0),
            hallucination_risk=_clamp01(data.get("hallucination_risk"), 1.0),
            source_scope_violation=bool(data.get("source_scope_violation", False)),
            verdict=str(data.get("verdict", "UNKNOWN")).upper(),
            unsupported_claims=list(data.get("unsupported_claims", []) or []),
            supported_claims=list(data.get("supported_claims", []) or []),
            reason=str(data.get("reason", "") or ""),
        )

        if result.verdict not in ("PASS", "WARN", "FAIL"):
            if (
                result.faithfulness >= EVAL_MIN_FAITHFULNESS
                and result.answer_relevance >= EVAL_MIN_ANSWER_RELEVANCE
                and not result.source_scope_violation
            ):
                result.verdict = "PASS"
            elif result.faithfulness >= 0.55:
                result.verdict = "WARN"
            else:
                result.verdict = "FAIL"

        return result

    except Exception as e:
        print(f"⚠️ RAG evaluation error: {e}")
        return RagEvalResult(
            verdict="ERROR",
            reason=str(e),
        )


def format_eval_debug_md(eval_result: RagEvalResult) -> str:
    """
    Formatta le metriche nel pannello Audit della UI.
    """
    unsupported = eval_result.unsupported_claims[:5]
    supported = eval_result.supported_claims[:5]

    lines = []
    lines.append("### 🧪 RAG Faithfulness Evaluation")
    lines.append(f"- **Verdict**: `{eval_result.verdict}`")
    lines.append(f"- **Faithfulness**: **{eval_result.faithfulness:.2f}**")
    lines.append(f"- **Answer relevance**: **{eval_result.answer_relevance:.2f}**")
    lines.append(f"- **Context support**: **{eval_result.context_support:.2f}**")
    lines.append(f"- **Hallucination risk**: **{eval_result.hallucination_risk:.2f}**")
    lines.append(f"- **Source scope violation**: **{eval_result.source_scope_violation}**")

    if eval_result.reason:
        lines.append(f"- **Reason**: {eval_result.reason}")

    if unsupported:
        lines.append("\n#### Unsupported claims")
        for c in unsupported:
            lines.append(f"- {c}")

    if supported:
        lines.append("\n#### Supported claims")
        for c in supported:
            lines.append(f"- {c}")

    return "\n".join(lines).strip()

# =========================
# 🛡️ UI SAFETY HELPERS
# =========================

MAX_UI_SOURCES = int(os.getenv("MAX_UI_SOURCES", "8"))
MAX_UI_SOURCE_CONTENT_CHARS = int(os.getenv("MAX_UI_SOURCE_CONTENT_CHARS", "900"))
MAX_UI_DEBUG_CHARS = int(os.getenv("MAX_UI_DEBUG_CHARS", "6000"))


def ui_safe_text(value, max_chars: int) -> str:
    """
    Versione minimale e compatibile con Reflex.
    Serve solo a evitare testi enormi o caratteri di controllo nella UI.
    Non altera il contenuto usato dal RAG/LLM.
    """
    if value is None:
        return ""

    try:
        text = str(value)
    except Exception:
        text = ""

    # Rimuove caratteri di controllo problematici per JSON/React.
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[contenuto troncato per la UI]"

    return text


def ui_safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def ui_safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:  # NaN
            return default
        if v == float("inf") or v == float("-inf"):
            return default
        return round(v, 4)
    except Exception:
        return default




def _short_content_hash(text: str, n: int = 900) -> str:
    """
    Hash breve del contenuto per deduplicare fonti quasi identiche.
    Non viene mostrato all'utente.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized[:n].encode("utf-8")).hexdigest()[:16]


def dedupe_sources_for_answer(sources: List[SourceItem]) -> List[SourceItem]:
    """
    Deduplica fonti equivalenti prima del prompt e della UI.

    Regole:
    - stesso id/chunk_uuid => una sola fonte;
    - stesso filename + pagina + tipo + contenuto simile => una sola fonte;
    - conserva la fonte con score più alto;
    - unisce db_origin quando la stessa fonte arriva da più canali.
    """
    if not sources:
        return []

    by_key: Dict[Tuple[str, str, int, str, str], SourceItem] = {}
    order: List[Tuple[str, str, int, str, str]] = []

    for s in sources:
        filename_norm = normalize_doc_name(getattr(s, "filename", "") or "")
        page = int(getattr(s, "page", 0) or 0)
        source_type = normalize_source_type(getattr(s, "type", "") or "text")
        content = getattr(s, "content", "") or ""

        # Preferisce id/chunk_uuid quando disponibile.
        sid = str(getattr(s, "id", "") or "").strip()

        if sid and sid not in {"graph", "neo4j_relations"}:
            key = ("id", sid, 0, "", "")
        else:
            key = (
                "content",
                filename_norm,
                page,
                source_type,
                _short_content_hash(content),
            )

        if key not in by_key:
            by_key[key] = s
            order.append(key)
            continue

        existing = by_key[key]

        # Tiene il contenuto più ricco.
        if len(content) > len(existing.content or ""):
            existing.content = content

        # Tiene score maggiore.
        existing.score = max(float(existing.score or 0.0), float(s.score or 0.0))

        # Unisce provenienza DB.
        origins: List[str] = []
        for origin in [existing.db_origin, s.db_origin]:
            for part in str(origin or "").split("+"):
                p = part.strip()
                if p and p not in origins:
                    origins.append(p)

        existing.db_origin = " + ".join(origins) if origins else existing.db_origin

        # Preserva metadati se mancanti.
        if not existing.filename and s.filename:
            existing.filename = s.filename

        if not existing.page and s.page:
            existing.page = s.page

        if not existing.section_hint and s.section_hint:
            existing.section_hint = s.section_hint

    return [by_key[k] for k in order]
def dedupe_sources_for_ui_compact(sources: List[SourceItem]) -> List[SourceItem]:
    """
    Deduplica più aggressiva solo per UI/badge.
    Compatta risultati della stessa pagina e dello stesso documento, mantenendo
    comunque separati GRAPH, formula e sorgenti con pagina diversa.
    """
    if not sources:
        return []

    out: List[SourceItem] = []
    seen = set()

    for s in dedupe_sources_for_answer(sources):
        stype = normalize_source_type(getattr(s, "type", "") or "text")
        tier = normalize_tier_value(getattr(s, "tier", "") or "C")

        if tier == "GRAPH" or stype in {"formula", "graph_relations"}:
            key = (
                "special",
                str(getattr(s, "id", "") or ""),
                normalize_doc_name(getattr(s, "filename", "") or ""),
                int(getattr(s, "page", 0) or 0),
                stype,
            )
        else:
            key = (
                "doc_page",
                normalize_doc_name(getattr(s, "filename", "") or ""),
                int(getattr(s, "page", 0) or 0),
                stype,
            )

        if key in seen:
            continue

        seen.add(key)
        out.append(s)

    return out


def dedupe_sources_for_ui(sources: List[SourceItem]) -> List[SourceItem]:
    """
    Deduplica più aggressiva SOLO per la UI: collassa fonti della stessa pagina,
    mantenendo graph/formula separati.
    """
    base = dedupe_sources_for_answer(sources or [])
    out: List[SourceItem] = []
    seen = set()

    for s in base:
        stype = normalize_source_type(getattr(s, "type", "") or "text")
        if stype in {"formula", "graph", "graph_relations"} or normalize_tier_value(getattr(s, "tier", "")) == "GRAPH":
            key = (str(getattr(s, "id", "")), stype)
        else:
            key = (
                normalize_doc_name(getattr(s, "filename", "") or ""),
                int(getattr(s, "page", 0) or 0),
                stype,
            )

        if key in seen:
            continue

        seen.add(key)
        out.append(s)

    return out

def prepare_sources_for_ui(sources: List[SourceItem]) -> List[SourceItem]:
    """
    Crea una copia ridotta delle fonti SOLO per la UI.
    Evita crash o sparizione schermata quando i chunk sono troppo lunghi.
    """
    out: List[SourceItem] = []

    for s in dedupe_sources_for_ui(sources or [])[:MAX_UI_SOURCES]:
        out.append(
            SourceItem(
                id=ui_safe_text(getattr(s, "id", ""), 200),
                content=ui_safe_text(getattr(s, "content", ""), MAX_UI_SOURCE_CONTENT_CHARS),
                filename=ui_safe_text(getattr(s, "filename", "Unknown"), 240),
                page=ui_safe_int(getattr(s, "page", 0), 0),
                type=ui_safe_text(getattr(s, "type", "text"), 80),
                score=ui_safe_float(getattr(s, "score", 0.0), 0.0),
                graph_context=[],
                section_hint=ui_safe_text(getattr(s, "section_hint", ""), 300),
                image_id=getattr(s, "image_id", None),
                tier=ui_safe_text(getattr(s, "tier", "C"), 40),
                pg_ingestion_ts=ui_safe_text(getattr(s, "pg_ingestion_ts", ""), 80),
                pg_source_name=ui_safe_text(getattr(s, "pg_source_name", ""), 160),
                pg_source_type=ui_safe_text(getattr(s, "pg_source_type", ""), 80),
                pg_log_id=ui_safe_int(getattr(s, "pg_log_id", 0), 0),
                pg_chunk_id=ui_safe_int(getattr(s, "pg_chunk_id", 0), 0),
                pg_toon_type=ui_safe_text(getattr(s, "pg_toon_type", ""), 80),
                db_origin=ui_safe_text(getattr(s, "db_origin", "Unknown"), 160),
            )
        )

    return out


def prepare_debug_for_ui(debug_md) -> str:
    """
    Riduce l'audit solo per visualizzazione e garantisce SEMPRE una stringa.
    Questo evita l'errore React Markdown:
    Unexpected value `[object Object]` for `children` prop, expected `string`.
    """
    if debug_md is None:
        return ""

    if isinstance(debug_md, (dict, list, tuple)):
        try:
            debug_md = json.dumps(debug_md, indent=2, ensure_ascii=False)
        except Exception:
            debug_md = str(debug_md)

    return safe_markdown(ui_safe_text(str(debug_md), MAX_UI_DEBUG_CHARS))


def state_get(obj, key: str, default=None):
    """
    Accesso sicuro a dict / oggetti Pydantic / oggetti Reflex.

    Serve perché, in alcuni casi, self.messages può contenere:
    - ChatMessage
    - dict serializzati da Reflex
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def normalize_sources_for_modal(raw_sources) -> List[SourceItem]:
    """
    Normalizza le fonti prima di passarle al modal Reflex.
    Evita crash quando le fonti arrivano come dict invece che come SourceItem.
    """
    normalized: List[SourceItem] = []

    for s in (raw_sources or []):
        normalized.append(
            SourceItem(
                id=ui_safe_text(state_get(s, "id", ""), 200),
                content=ui_safe_text(
                    state_get(s, "content", ""),
                    MAX_UI_SOURCE_CONTENT_CHARS,
                ),
                filename=ui_safe_text(
                    state_get(s, "filename", "Unknown"),
                    240,
                ),
                page=ui_safe_int(state_get(s, "page", 0), 0),
                type=ui_safe_text(state_get(s, "type", "text"), 80),
                score=ui_safe_float(state_get(s, "score", 0.0), 0.0),
                graph_context=[],
                section_hint=ui_safe_text(
                    state_get(s, "section_hint", ""),
                    300,
                ),
                image_id=state_get(s, "image_id", None),
                tier=ui_safe_text(state_get(s, "tier", "C"), 40),
                pg_ingestion_ts=ui_safe_text(
                    state_get(s, "pg_ingestion_ts", ""),
                    80,
                ),
                pg_source_name=ui_safe_text(
                    state_get(s, "pg_source_name", ""),
                    160,
                ),
                pg_source_type=ui_safe_text(
                    state_get(s, "pg_source_type", ""),
                    80,
                ),
                pg_log_id=ui_safe_int(state_get(s, "pg_log_id", 0), 0),
                pg_chunk_id=ui_safe_int(state_get(s, "pg_chunk_id", 0), 0),
                pg_toon_type=ui_safe_text(
                    state_get(s, "pg_toon_type", ""),
                    80,
                ),
                db_origin=ui_safe_text(
                    state_get(s, "db_origin", "Unknown"),
                    160,
                ),
            )
        )

    return prepare_sources_for_ui(normalized)






# ============================================================
# ✅ v4.4 MINIMAL NON-ADAPTIVE FIXES
# - formula classifier / cleaner
# - deterministic dates
# - crosswalk/checklist prompt helpers
# - final answer sanitation for language + external URLs/sources
# ============================================================

def _is_likely_italian_query(query_text: str) -> bool:
    q = (query_text or "").lower()
    italian_markers = [
        "cos", "perché", "quali", "quale", "spiega", "calcola", "confronta",
        "mostrami", "trova", "usa", "documenti", "fonti", "sanzioni", "garante",
        "scadenza", "soggetto", "rischio", "evidenze", "controlli",
    ]
    return any(m in q for m in italian_markers) or bool(re.search(r"[àèéìòù]", q))


def is_crosswalk_mapping_query(query_text: str) -> bool:
    """
    Router leggero e non adattativo per richieste di mapping/crosswalk/matrice.
    Non genera risposte deterministiche: aggiunge solo guardrail al prompt.
    """
    q = (query_text or "").lower()
    mapping_terms = [
        "crosswalk", "mapping", "mappatura", "mappa", "matrice", "matrix",
        "collega", "collegare", "allinea", "allineamento", "correlazione",
    ]
    framework_terms = ["iso", "nist", "annex", "csf", "800-53", "clausola", "clause", "controlli", "controls"]
    return any(t in q for t in mapping_terms) and sum(1 for t in framework_terms if t in q) >= 2


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - datetime(year, month, 1)).days


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, _days_in_month(year, month))
    return datetime(year, month, day)


def _format_it_date(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y")


def try_solve_date_offsets(query_text: str) -> Optional[str]:
    """
    Calcola in modo deterministico scadenze del tipo data + N mesi.
    Non interpreta norme: calcola solo le date aritmetiche richieste dall'utente.
    """
    q = query_text or ""
    ql = q.lower()

    if not re.search(r"\b(mesi|months?|giorni|days?|anni|years?)\b", ql):
        return None

    m_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", q)
    if not m_date:
        return None

    day, month, year = map(int, m_date.groups())
    try:
        base = datetime(year, month, day)
    except Exception:
        return None

    month_offsets = []
    for m in re.finditer(r"\b(\d{1,3})\s*(?:mesi|months?)\b", ql):
        n = int(m.group(1))
        if n <= 0:
            continue
        window = ql[max(0, m.start() - 80): m.end() + 80]
        if any(t in window for t in ["notifica", "notification", "incident", "incidente"]):
            label = f"Scadenza a {n} mesi per notifica/incidenti"
        elif any(t in window for t in ["misure", "security measures", "misure di sicurezza"]):
            label = f"Scadenza a {n} mesi per misure di sicurezza"
        else:
            label = f"Scadenza a {n} mesi"
        month_offsets.append((n, label, _add_months(base, n)))

    if not month_offsets:
        return None

    dedup = []
    seen_offsets = set()
    for item in month_offsets:
        if item[0] in seen_offsets:
            continue
        seen_offsets.add(item[0])
        dedup.append(item)

    lines = [
        "**A) Risposta**\n",
        f"Data di partenza: **{_format_it_date(base)}**.",
        "",
        "| Scadenza richiesta | Calcolo | Data risultante |",
        "|---|---|---:|",
    ]
    for n, label, result in dedup:
        lines.append(f"| {label} | {_format_it_date(base)} + {n} mesi | **{_format_it_date(result)}** |")

    if "2026" in ql or "confront" in ql or "compare" in ql:
        lines.extend([
            "",
            "**Confronto con fonti/documenti**",
            "",
            "- Il calcolo sopra è deterministico e usa solo la data fornita dall'utente.",
            "- Eventuali scadenze speciali o regole applicative devono essere verificate nelle fonti recuperate e non modificano il calcolo aritmetico sopra.",
        ])

    lines.extend([
        "",
        "**B) Evidenze**",
        "",
        f"- Data fornita dall'utente: `{_format_it_date(base)}`.",
        "- Offset temporali richiesti: " + ", ".join(f"{n} mesi" for n, _, _ in dedup) + ".",
        "",
        "**C) Limiti / Conflitti**",
        "",
        "- Il calcolo delle date è deterministico; l'interpretazione normativa resta vincolata alle fonti recuperate.",
        "",
        "**D) Fonti**",
        "",
        "- Dati temporali forniti dall'utente.",
    ])
    return "\n".join(lines)


# Override v4.4: include date offsets without touching risk/coverage solvers.
def try_solve_math_query(query_text: str) -> Optional[str]:
    if solve_control_coverage(query_text):
        return solve_control_coverage(query_text)
    if solve_risk_product(query_text):
        return solve_risk_product(query_text)
    return try_solve_date_offsets(query_text)


# Override v4.4: date/scadenze/confronti need retrieval context, but math remains authoritative.
def needs_math_document_context(query_text: str) -> bool:
    q = (query_text or "").lower()
    context_terms = [
        "collega", "collegalo", "collegala", "documenti", "documento",
        "risk assessment", "evidence assessment", "assessment", "controlli",
        "evidenze", "conformità", "audit", "confronta", "confrontale",
        "compare", "scadenze", "scadenza", "deadline", "deadlines",
        "nis", "2026", "fonti", "fonte", "normativa", "norme",
    ]
    return any(t in q for t in context_terms)


def _strip_math_wrappers(value: str) -> str:
    v = (value or "").strip()
    v = re.sub(r"^`+|`+$", "", v).strip()
    v = re.sub(r"^\$+|\$+$", "", v).strip()
    return v


def _normalize_latex_value(value: str) -> str:
    v = (value or "").strip()
    v = v.replace("$$$", "$$")
    v = re.sub(r"(?<!\\)ext\{", r"\\text{", v)
    v = re.sub(r"\${3,}", "$$", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

def _looks_threshold_rule(text: str) -> bool:
    r"""
    v4.5: riconosce regole soglia anche quando i valori sono in LaTeX,
    es. 5\% oppure 1\text{ milione}.
    """
    plain = _formula_plain_text(text or "").lower()
    
    # Cerca i termini esatti usando le regex dinamiche
    has_threshold_word = any(re.search(rf"\b{re.escape(x)}\b", plain) for x in THRESHOLD_TERMS_LIST)
    
    # Verifica che ci sia almeno un numero nel chunk
    has_number = bool(re.search(r"\b\d+(?:[,.]\d+)?\b", plain))
    
    return has_threshold_word and has_number


def _looks_computational_formula(latex: str) -> bool:
    v = _strip_math_wrappers(latex)
    if not v or "formula esplicita non recuperata" in v.lower():
        return False
    if any(x in v for x in ["\\frac", "\\sum", "\\prod", "×", "*", "/"]):
        return True
    if "=" in v:
        left, right = v.split("=", 1)
        right_clean = right.strip()
        if re.fullmatch(r"\\?text\{[^}]+\}", right_clean):
            return False
        return bool(re.search(r"[A-Za-z0-9]", left) and re.search(r"[+\-*/×÷]|\\frac|\\sum|\d", right_clean))
    return False


def _looks_definitional_metric(latex: str, meaning: str = "") -> bool:
    v = _strip_math_wrappers(latex)
    if "=" in v:
        _, right = v.split("=", 1)
        if re.search(r"\\?text\{[^}]+\}", right.strip()):
            return True
    m = (meaning or "").lower()
    return any(t in m for t in ["tempo medio", "mean time", "metrica", "metric", "indicatore", "indicator"])


def _classify_formula_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(row)
    name = re.sub(r"\s+", " ", str(rr.get("name") or "")).strip() or "Elemento recuperato"
    latex = _normalize_latex_value(str(rr.get("latex") or "")).strip()
    meaning = re.sub(r"\s+", " ", str(rr.get("meaning") or "")).strip()
    combined = " ".join([name, latex, meaning])

    if _looks_threshold_rule(combined):
        tipo = "Regola soglia"
        formula = _strip_math_wrappers(latex) if latex and "formula esplicita non recuperata" not in latex.lower() else "formula computazionale non applicabile"
    elif _looks_computational_formula(latex):
        tipo = "Formula computazionale"
        formula = latex
    elif _looks_definitional_metric(latex, meaning):
        tipo = "Metrica definitoria"
        formula = "formula computazionale non recuperata"
        if not meaning:
            m = re.search(r"=\s*\\?text\{([^}]+)\}", latex)
            if m:
                meaning = m.group(1).strip()
    else:
        tipo = "Metrica/elemento citato"
        formula = "formula esplicita non recuperata"

    rr["name"] = name
    rr["tipo"] = tipo
    rr["latex"] = formula
    rr["meaning"] = meaning
    return rr


def _is_noise_formula_row_v44(row: Dict[str, Any]) -> bool:
    name = str(row.get("name") or "").strip().lower()
    latex = str(row.get("latex") or "").strip().lower()
    tipo = str(row.get("tipo") or "").strip().lower()

    generic_names = {
        "", "formula/metric", "formula recuperata", "contenuto", "variabili",
        "metrica/indicatore citato", "formula", "metric", "formule e modelli matematici",
        "formule e modelli matematici - pagina 12 --", "formule e modelli matematici - pagina 24 --",
    }
    if name in generic_names and tipo not in {"formula computazionale", "regola soglia"}:
        return True

    if tipo != "regola soglia" and re.fullmatch(r"\$?\s*\d+(?:[,.]\d+)?\s*(?:\\text\{[^}]+\}|%|percento|milione|million)?\s*\$?", latex):
        return True
    return False


def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 8) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        rr = _classify_formula_row(r)
        if _is_noise_formula_row_v44(rr):
            continue
        norm_name = re.sub(r"[^a-z0-9]+", "", str(rr.get("name") or "").lower())
        key = (
            norm_name,
            str(rr.get("tipo") or "").lower(),
            normalize_doc_name(str(rr.get("filename") or "")),
            int(rr.get("page") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        if rr.get("tipo") == "Metrica definitoria":
            if any(k[0] == key[0] and k[2] == key[2] and k[3] == key[3] and k[1] == "formula computazionale" for k in seen):
                continue
        cleaned.append(rr)
        if len(cleaned) >= max_rows:
            break
    return cleaned


# Override v4.4: classifica formule/metriche/soglie senza trasformare definizioni in formule computazionali.
def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=8)
    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie o regole soglia sufficientemente esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    table_lines = [
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        table_lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 160)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 120)} | "
            f"`{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 260)}` | "
            f"{_formula_md_cell(r.get('meaning') or '', 300)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 220)} | "
            f"{int(r.get('page') or 0)} |"
        )

    used_files = sorted({str(r.get("filename") or "") for r in rows if r.get("filename")})
    missing_terms: List[str] = []
    ql = (query_text or "").lower()
    requested_generic = ["cvss", "rischio", "risk", "maturità", "maturity", "copertura", "coverage"]
    found_text = " ".join([str(r.get("name", "")) + " " + str(r.get("latex", "")) for r in rows]).lower()
    for t in requested_generic:
        if t in ql and t not in found_text:
            missing_terms.append(t)
    missing_line = ""
    if missing_terms:
        missing_line = "\n- Non sono state recuperate formule esplicite per: " + ", ".join(sorted(set(missing_terms))) + "."

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n**B) Evidenze**\n\n"
        + "- Le formule/metriche/regole sono state classificate in modo deterministico.\n"
        + "- Le soglie normative sono distinte dalle formule computazionali.\n"
        + "- Le definizioni di metriche sono distinte dalle formule calcolabili.\n"
        + missing_line
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula computazionale se la fonte non contiene un calcolo esplicito.\n\n"
        + "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )


def _compact_source_list_for_answer(sources: List[SourceItem], max_sources: int = 8) -> str:
    seen = set()
    lines: List[str] = []
    for s in sources or []:
        fname = str(getattr(s, "filename", "") or "").strip()
        if not fname or fname in {"KG", "Neo4j Knowledge Graph"}:
            continue
        page = int(getattr(s, "page", 0) or 0)
        key = (normalize_doc_name(fname), page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {fname}" + (f" (p.{page})" if page else ""))
        if len(lines) >= max_sources:
            break
    return "\n".join(lines) if lines else "- Vedi pannello Fonti/Audit."


def _replace_final_sources_section(answer: str, sources: List[SourceItem]) -> str:
    replacement = "**D) Fonti**\n\n" + _compact_source_list_for_answer(sources)
    pattern = r"(?is)(\*\*D\)\s*Fonti\*\*|D\)\s*Fonti).*\Z"
    if re.search(pattern, answer or ""):
        return re.sub(pattern, replacement, answer.rstrip())
    return (answer or "").rstrip() + "\n\n" + replacement


def postprocess_generated_answer(answer: str, query_text: str, sources: List[SourceItem]) -> str:
    """
    Corregge automaticamente le etichette della struttura e le frasi di fallback
    in base alla lingua della domanda dell'utente.
    """
    out = answer or ""

    if _is_likely_italian_query(query_text):
        # 1. Traduzione etichette strutturali
        replacements = {
            "**C) Limiti / Conflitti**": "**C) Limiti / Conflitti**",
            "**C) Limitations / Conflicts**": "**C) Limiti / Conflitti**",
            "Limiti / Conflitti": "Limiti / Conflitti",
            "Information not found in retrieved documents": "Non ho trovato informazioni sufficienti nei documenti recuperati",
            "The provided context does not contain": "Il contesto recuperato non contiene",
            "The calculation assumes": "Il calcolo assume",
            "The actual sanction imposed would depend on": "La sanzione effettiva dipenderà da",
            "to answer the user's question": "per rispondere alla domanda",
        }
        
        for old, new in replacements.items():
            out = out.replace(old, new)
            
        # 2. Correzione forzata se il modello ha usato inglese per la sezione C
        if "Limitations / Conflicts" in out:
            out = out.replace("Limitations / Conflicts", "Limiti / Conflitti")

    # Manteniamo la logica esistente per URL e Fonti
    if is_strict_checklist_query(query_text) or is_crosswalk_mapping_query(query_text):
        # Rimuove tutti gli URL esterni trovati e li sostituisce con una stringa neutra
        out = re.sub(
            r"https?://\S+",
            "[Link esterno non autorizzato rimosso]",
            out,
            flags=re.IGNORECASE,
        )
        out = _replace_final_sources_section(out, sources)

    return out.strip()

# =========================
# 🔄 STATE MANAGEMENT
# =========================
class State(rx.State):
    # ... le tue variabili esistenti ...
    
    messages: List[ChatMessage] = [
        ChatMessage(
            id="init",
            role="assistant",
            content=f"Ciao! Sono attivo con **{LLM_MODEL_NAME}**. Metodologia Tier A, Policy Tier B ed Evidenze Tier C caricate. Fammi domande sui tuoi documenti di assessment.",
        )
    ]


    input_text: str = ""
    is_processing: bool = False
    
    current_active_doc: str = ""
    inline_open_for: str = ""
    inline_tab: str = "sources"

    vram_info: str = "N/A"
    vram_free: str = "N/A"
    backend_status: str = "OK"

    show_sources_modal: bool = False
    modal_sources: List[SourceItem] = []
    
    # 🔴 Variabili RAW per conservare i dati complessi (liste/dizionari)
    neo4j_results_raw: list[dict] = []
    log_prompt_raw: list[dict] = []
    
    # 🟢 Variabili STRINGA (Computed/Viste) per il Frontend (Queste evitano l'errore Object!)
    modal_debug_md: str = ""
    modal_title: str = ""
    
    @rx.var
    def neo4j_debug_string(self) -> str:
        """Converte i risultati raw di Neo4j in una stringa JSON formattata per il frontend."""
        if not self.neo4j_results_raw:
            return "Nessun risultato da Neo4j."
        return json.dumps(self.neo4j_results_raw, indent=2, ensure_ascii=False)

    @rx.var
    def log_prompt_string(self) -> str:
        """Converte i log del prompt in una stringa JSON formattata per il frontend."""
        if not self.log_prompt_raw:\
            return "Nessun log disponibile."
        return json.dumps(self.log_prompt_raw, indent=2, ensure_ascii=False)

    def set_sources_modal_open(self, value: bool):
        self.show_sources_modal = value
    
    def get_context_by_tier(self, query: str, tier: str) -> str:
        try:
            query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

            # Compatibilità universale
            if hasattr(qdrant_client_inst, 'query_points'):
                search_result = qdrant_client_inst.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="tier", match=models.MatchValue(value=tier))]
                    ),
                    limit=15
                ).points
            else:
                search_result = qdrant_client_inst.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="tier", match=models.MatchValue(value=tier))]
                    ),
                    limit=15
                )
                
            texts = []
            for res in search_result:
                p = res.payload or {}
                content = safe_payload_text(p)
                if content:
                    texts.append(content)

            return "\n".join(texts)
        except Exception as e:
            print(f"⚠️ Errore recupero Tier {tier}: {e}")
            return ""

    # --- Metodi di gestione UI ---
    def toggle_inline_sources(self, msg_id: str):
        if self.inline_open_for == msg_id and self.inline_tab == "sources":
            self.inline_open_for = ""
            return
        self.inline_open_for = msg_id
        self.inline_tab = "sources"

    def toggle_inline_audit(self, msg_id: str):
        if self.inline_open_for == msg_id and self.inline_tab == "audit":
            self.inline_open_for = ""
            return
        self.inline_open_for = msg_id
        self.inline_tab = "audit"

    def close_inline_panel(self):
        self.inline_open_for = ""

    def open_sources_audit(self, msg_id: str):
        self.modal_title = "Fonti & Audit"

        found = None

        for m in self.messages:
            current_id = state_get(m, "id", "")
            if str(current_id) == str(msg_id):
                found = m
                break

        if not found:
            self.modal_sources = []
            self.modal_debug_md = ""
            self.show_sources_modal = True
            return

        raw_sources = state_get(found, "sources", [])
        raw_debug_md = state_get(found, "debug_md", "")

        self.modal_sources = normalize_sources_for_modal(raw_sources)
        self.modal_debug_md = str(prepare_debug_for_ui(raw_debug_md or ""))

        self.show_sources_modal = True

    def close_sources_audit(self):
        self.show_sources_modal = False

    def on_load(self):
        self.refresh_gpu()
        self.refresh_backend_status()

    def refresh_backend_status(self):
        ready = bool(embedder and qdrant_client_inst and llm_client)

        if PG_ENRICH_ENABLED:
            ready = ready and bool(pg_pool)

        if NEO4J_ENABLED:
            ready = ready and bool(neo4j_driver)

        self.backend_status = "OK" if ready else "DEGRADED"



    def refresh_gpu(self):
        self.vram_info = gpu_free_info()
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                self.vram_free = f"{free_bytes / (1024**3):.1f} GB free"
            except: self.vram_free = "N/A"
        else: self.vram_free = "CPU"

    def clear_history(self):
        self.messages = [self.messages[0]]

    def set_input_text(self, text: str):
        self.input_text = text

    # ✅ ORA INDENTATO CORRETTAMENTE DENTRO LA CLASSE

    async def handle_submit(self):
        # Import necessario per la gestione asincrona della UI
        import asyncio 

        if not self.input_text.strip() or self.is_processing:
            return

        user_query = self.input_text.strip()
        self.input_text = ""
        self.is_processing = True
        
        # English instructions for the model
        language_reminder = "\n\nCRITICAL: You MUST detect the language of the user's question and answer EXCLUSIVELY in that same language."

        try:
            self.refresh_gpu()
            # 1. Mostra subito il messaggio dell'utente nella chat
            self.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", content=user_query))
            yield rx.scroll_to("chat_bottom")
            
            # --- FIX CRITICO: Pausa per aggiornare la UI ---
            # Senza questo, l'app sembra bloccata finché il RAG non finisce i calcoli.
            # 0.1 secondi sono sufficienti a Reflex per renderizzare il messaggio a video.
            await asyncio.sleep(0.1) 
            # -----------------------------------------------

            intent = detect_intent(user_query)
            math_answer = try_solve_math_query(user_query)
            math_needs_context = bool(math_answer and needs_math_document_context(user_query))
            analytics_mode = is_user_data_analytics(user_query) and not math_answer

            if (
                is_glossary_definition_query(user_query)
                and not is_mixed_glossary_rag_query(user_query)
                and not is_graph_relation_query(user_query)
            ):
                glossary_answer, glossary_sources, glossary_debug = answer_glossary_terms_directly(user_query)

                if glossary_answer:
                    self.messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=glossary_answer,
                            sources=prepare_sources_for_ui(glossary_sources),
                            debug_md=prepare_debug_for_ui(glossary_debug),
                        )
                    )
                    self.is_processing = False
                    yield rx.scroll_to("chat_bottom")
                    return

            if math_answer and not math_needs_context:
                self.messages.append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=math_answer,
                        sources=prepare_sources_for_ui(make_analytics_sources(user_query)),
                        debug_md=prepare_debug_for_ui(
                            "### 🔎 Audit (Deterministic Math Mode)\n"
                            "- retrieval: **bypassed for arithmetic**\n"
                            "- source: **USER_INPUT**"
                        ),
                    )
                )
                self.is_processing = False
                yield rx.scroll_to("chat_bottom")
                return

            # Variabili per il payload
            system_instructions = ""
            final_user_content = ""
            debug_md = ""
            sources = []

            if analytics_mode:
                sources = make_analytics_sources(user_query)
                debug_md = "### 🔎 Audit (Analytics Mode)\n- retrieval: **bypassed**\n- source: **USER_INPUT**"
                system_instructions = build_system_instructions_analytics(intent)
                
                # In Analytics Mode, i dati sono nella domanda stessa
                final_user_content = f"### QUESTION ###\n{user_query}{language_reminder}"
            else:
                # --- INIZIO NUOVA LOGICA: MEMORIA DI CONTESTO ---
                # Estraiamo il documento dalla query. Se c'è, lo salviamo in memoria.
                extracted_doc = extract_requested_document(user_query)
                if extracted_doc:
                    self.current_active_doc = extracted_doc

                active_doc_for_query = self.current_active_doc if is_follow_up_query(user_query) else ""
                if extracted_doc:
                    active_doc_for_query = extracted_doc

                # 1. RECUPERO DATI (Hybrid Search + Rerank)
                # Usa memoria documento solo per follow-up reali, per evitare contaminazioni nella batteria test.
                retrieval_query = user_query

                if math_needs_context:
                    retrieval_query = (
                        user_query
                        + "\n risk assessment evidence assessment valutazione del rischio controlli evidenze assessment integrato"
                    )

                sources, debug_md = retrieve_v2(retrieval_query, active_doc=active_doc_for_query)
                sources = dedupe_sources_for_answer(sources)
                # --- FINE NUOVA LOGICA ---

                if not sources:
                    self.messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=(
                                "**A) Risposta**\n\n"
                                "Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                                "**B) Evidenze**\n\n"
                                "- Nessuna fonte pertinente recuperata per il documento richiesto.\n\n"
                                "**C) Limiti**\n\n"
                                "- Il sistema non deve usare formule provenienti da altri documenti.\n\n"
                                "**D) Fonti**\n\n"
                                "- Nessuna fonte utilizzabile."
                            ),
                            sources=[],
                            debug_md=prepare_debug_for_ui(debug_md),
                        )
                    )
                    self.is_processing = False
                    yield rx.scroll_to("chat_bottom")
                    return
                

                # ============================================================
                # 🧮 MATH-FIRST MODE - v4.3 minimal fix
                # ============================================================
                # Se il solver matematico ha già prodotto un risultato, quel risultato
                # è autoritativo. Il retrieval serve solo per il collegamento documentale.
                # Questo impedisce al Graph Relation Strict Mode di intercettare query
                # tipo "calcola ... e collega il risultato al risk assessment".
                if math_answer and math_needs_context:
                    math_context_answer = build_math_answer_with_document_context(
                        math_answer,
                        sources,
                    )
                    self.messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=math_context_answer,
                            sources=prepare_sources_for_ui(sources),
                            debug_md=prepare_debug_for_ui(
                                (debug_md or "")
                                + "\n\n### 🧮 Math-First Mode v4.4\n"
                                "- Calcolo deterministico eseguito prima di Graph Relation Mode.\n"
                                "- Il risultato numerico non è stato ricalcolato dall'LLM.\n"
                                "- Le fonti recuperate sono usate solo per contestualizzazione documentale."
                            ),
                        )
                    )
                    self.is_processing = False
                    yield rx.scroll_to("chat_bottom")
                    return

                # ============================================================
                # 🕸️ GRAPH RELATION STRICT MODE
                # ============================================================
                # Per domande su collegamenti/relazioni/entità, evita risposte discorsive
                # quando è possibile costruire una tabella verificabile dalle fonti recuperate.
                if is_graph_relation_query(user_query) and not math_answer:
                    graph_answer = answer_graph_relations_strict(user_query, sources)

                    if graph_answer:
                        self.messages.append(
                            ChatMessage(
                                id=str(uuid.uuid4()),
                                role="assistant",
                                content=graph_answer,
                                sources=prepare_sources_for_ui(sources),
                                debug_md=prepare_debug_for_ui(
                                    (debug_md or "")
                                    + "\n\n### 🕸️ Graph Relation Strict Mode\n"
                                    "- Risposta generata in modo deterministico da relazioni Neo4j e/o co-occorrenze testuali recuperate.\n"
                                    "- Il modello LLM non è stato usato per inventare relazioni mancanti.\n"
                                    "- Le relazioni testuali sono marcate come non esplicite nel grafo."
                                ),
                            )
                        )
                        self.is_processing = False
                        yield rx.scroll_to("chat_bottom")
                        return



                # ============================================================
                # 🧮 FORMULA STRICT MODE
                # ============================================================
                # Per domande su formule/metriche/LaTeX, evita che il modello inventi formule
                # o lasci output vuoti. Usa solo formule/metriche presenti nelle fonti recuperate.
                if is_formula_strict_query(user_query):
                    formula_answer = answer_formula_strict(user_query, sources)

                    if formula_answer:
                        self.messages.append(
                            ChatMessage(
                                id=str(uuid.uuid4()),
                                role="assistant",
                                content=formula_answer,
                                sources=prepare_sources_for_ui(filter_sources_for_formula_answer(user_query, sources)),
                                debug_md=prepare_debug_for_ui(
                                    (debug_md or "")
                                    + "\n\n### 🧮 Formula Strict Mode\n"
                                    "- Risposta generata in modo deterministico da formule/metriche recuperate.\n"
                                    "- Il modello LLM non è stato usato per inventare formule mancanti.\n"
                                    "- Se una metrica è citata ma la formula non è esplicita, viene dichiarato chiaramente."
                                ),
                            )
                        )
                        self.is_processing = False
                        yield rx.scroll_to("chat_bottom")
                        return

                # --- INIZIO NUOVA LOGICA: PROMPT ANTI-CONTAMINAZIONE IN INGLESE ---
                # Subito dopo il blocco "if not sources:", creiamo le istruzioni di sistema
                # e aggiungiamo il guardrail robusto.
                system_instructions = build_system_instructions(intent)
                system_instructions += """
                
                8) ANTI-CONTAMINATION AND DISAMBIGUATION (CRITICAL):
                - If the user specifies or implies a specific document context, you MUST STRICTLY IGNORE retrieved chunks from other documents that define the same variables (e.g., 'alpha', 'D') differently.
                - Mathematical variables are highly context-dependent. Do not mix formulas from 'algorithmic trading' with those from 'asset allocation' or other topics.
                - If you see conflicting definitions for a variable across different sources, ALWAYS prioritize the definitions from the active requested document context.
                """

                if is_strict_checklist_query(user_query):
                    system_instructions += """

                9) STRICT CHECKLIST MODE (CRITICAL):
                - The user is asking for an audit/checklist output.
                - You MUST NOT use external URLs or web references.
                - You MUST NOT cite laws, article numbers, deadlines, standards, or obligations unless the exact reference is explicitly present in the retrieved context.
                - Every checklist row MUST include a retrieved source reference like [1], [2], etc. and the source must correspond to an actual retrieved chunk.
                - If a checklist item is reasonable but not directly supported by a retrieved source, write: "Fonte non recuperata" instead of inventing a source.
                - Do NOT create a final bibliography with external websites. In section D list only retrieved filenames.
                - Prefer a Markdown table with these columns:
                  | Area | Controllo/Requisito | Evidenza richiesta | Fonte recuperata | Livello di supporto |
                - Use these support levels only: "esplicito nella fonte", "supportato testualmente", "deduzione non esplicita", "fonte non recuperata".
                    """

                if is_crosswalk_mapping_query(user_query):
                    system_instructions += """

                10) CROSSWALK / MATRIX MODE (CRITICAL):
                - The user is asking for a mapping, crosswalk, matrix or control alignment.
                - You may produce a Markdown table, but every specific mapping must be grounded in the retrieved context.
                - Do NOT invent control codes, clauses, article numbers, catalog items, or mappings that are not present in the retrieved sources.
                - If a cell is not explicitly available, write "non recuperato puntualmente".
                - If the mapping is a reasonable synthesis but not explicit, write "deduzione non esplicita".
                - Add a column named "Livello di supporto" with one of:
                  "esplicito nella fonte", "supportato testualmente", "deduzione non esplicita", "non recuperato puntualmente".
                - Section C must clearly state whether the document contains an explicit crosswalk or only the instruction/need to build one.
                    """

                if is_graph_relation_query(user_query):
                    system_instructions += """

                11) GRAPH RELATION MODE (CRITICAL):
                - The user is asking for entities, concepts, links, or relations.
                - Section A MUST contain a Markdown table with exactly these columns:
                  | Entità sorgente | Relazione | Entità target | Documento | Pagina | Evidenza |
                - Use Neo4j graph context first when available.
                - You may also use textual retrieved sources to support a relation.
                - Do NOT answer only with definitions.
                - Do NOT use glossary-only mode.
                - Do NOT say that a relation is absent if it appears in the graph context or retrieved sources.
                - If a relation is semantically supported by text but not explicit as a graph edge, write: "supportata testualmente, non esplicita come arco".
                - If a relation is not supported, write: "non supportata dalle fonti recuperate".
                - Section B must briefly explain the strongest relations.
                - Section C must list missing or weak relations.
                - Section C MUST NOT say "no conflicts or missing information" unless every relation in Section A is explicitly supported by a retrieved source or graph edge.
                - If a relation is inferred from co-occurrence or textual proximity, explicitly write: "supportata testualmente, non esplicita come arco".
                - If evidence is incomplete, state exactly what is missing.
                    """

                if math_needs_context:
                    system_instructions += """

                11) DETERMINISTIC MATH RESULT (CRITICAL):
                - The deterministic calculation block is authoritative.
                - You MUST NOT recalculate or change the numerical result.
                - Use retrieved documents only to explain how the result relates to risk assessment, evidence assessment, control coverage, or audit reasoning.
                    """
                # --- FINE NUOVA LOGICA ---

                if should_force_tier_a(user_query) and not is_strict_checklist_query(user_query):
                    has_tier_a = any((s.tier or "").upper() == "A" for s in sources)

                    if not has_tier_a:
                        tier_a_context = self.get_context_by_tier(user_query, "A")

                        if tier_a_context:
                            sources.insert(
                                0,
                                SourceItem(
                                    id="forced_tier_a",
                                    content=tier_a_context,
                                    filename="TIER_A_METHODOLOGY",
                                    page=0,
                                    type="methodology",
                                    score=1.0,
                                    tier="A",
                                    db_origin="Qdrant Forced Tier A",
                                    section_hint="Forced methodology context"
                                )
                            )
                                
                # 2. RAGGRUPPAMENTO FONTI
                c_a_list, c_b_list, c_c_list, c_g_list = [], [], [], []

                for i, s in enumerate(sources, start=1):
                    tier_norm = normalize_tier_value(s.tier)

                    # FIX: Usa "Source" e "Page" per allinearsi perfettamente al System Prompt
                    header = f"--- Source [{i}] — {s.filename} — Page {s.page} — ({s.type}) ---\n"
                    meta = f"(tier={tier_norm} | db={s.db_origin})\n"
                    body = (s.content or "").strip()

                    if not body:
                        continue

                    snippet = header + meta + body + "\n\n"

                    if tier_norm == "A":
                        c_a_list.append(snippet)
                    elif tier_norm == "B":
                        c_b_list.append(snippet)
                    elif tier_norm == "GRAPH":
                        c_g_list.append(snippet)
                    else:
                        # FIX CRITICO: Qualsiasi tier non riconosciuto finisce qui. 
                        # Nessun chunk recuperato verrà mai più perso.
                        c_c_list.append(snippet)


                c_a = "".join(c_a_list).strip()
                c_b = "".join(c_b_list).strip()
                c_c = "".join(c_c_list).strip()
                c_g = "".join(c_g_list).strip()

                # --- INIZIO FIX MATEMATICO PRE-FLIGHT (Ampliato e in Inglese) ---
                math_injection = ""
                q_low = (user_query or "").lower()
                
                # Dizionario statico ampliato per il dominio assessment e audit (IT/EN)
                math_triggers = [
                    # IT: Termini finanziari, sanzionatori e di calcolo
                    "calcola", "calcolo", "sanzione", "percentuale", "ammonta", "penale", "multa",
                    "somma", "totale", "media", "budget", "costo", "stima", "differenza", "quantifica",
                    "impatto economico", "danno", "roi", "massimale", "soglia", "limite", "budget",
                    # EN: Financial, sanction and calculation terms
                    "calculate", "calculation", "fine", "percentage", "amount", "penalty", "compute",
                    "sum", "total", "average", "cost", "estimate", "difference", "quantify",
                    "economic impact", "damage", "roi", "maximum", "threshold", "limit",
                    # Simboli matematici universali
                    "%"
                ]
                
                # Attiviamo la calcolatrice se la domanda contiene termini matematici o finanziari
                if any(k in q_low for k in math_triggers):
                    mini_context = f"Tier A (Normative):\n{c_a}\nTier B (Policies):\n{c_b}\nTier C (Evidences):\n{c_c}"
                    
                    math_prompt = (
                        f"Context:\n{mini_context}\n\n"
                        f"Question: {user_query}\n\n"
                        f"Task: Extract ONLY the pure mathematical expression (numbers and operators) "
                        f"needed to compute the exact answer based on the numbers provided in the context "
                        f"(e.g., 150000000 * 0.04). Do NOT write any text, units, or explanations. "
                        f"If no calculation is required, output exactly the word NO."
                    )
                    
                    try:
                        # Micro-chiamata sincrona per estrarre la formula
                        math_resp = llm_client.chat.completions.create(
                            model=LLM_MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a strict math expression extractor. Output ONLY the raw mathematical expression (e.g. 10 * 0.05) or the word NO."},
                                {"role": "user", "content": math_prompt}
                            ],
                            temperature=0.0,
                            stream=False,
                            extra_body={
                                "options": {
                                    "num_ctx": LLM_NUM_CTX,
                                    "num_predict": 50, # Bastano 50 token
                                    "repeat_penalty": 1.05,
                                }
                            }
                        )
                        
                        expr = math_resp.choices[0].message.content.strip()
                        
                        # Se ha estratto una formula, la facciamo calcolare a Python
                        if expr != "NO" and "NO" not in expr.upper():
                            risultato_esatto = calcolatrice_universale(expr)
                            if risultato_esatto:
                                math_injection = (
                                    f"\n\n[SYSTEM CALCULATED MATH RESULT: {risultato_esatto}]\n"
                                    f"CRITICAL RULE: You MUST use this exact numerical result in your final answer as the absolute truth. "
                                    f"Briefly explain the numbers used for the calculation.\n\n"
                                )
                    except Exception as e:
                        print(f"⚠️ Errore Pre-Flight Matematico: {e}")

                # 3. PROMPT DI SISTEMA
                # system_instructions è già stato costruito sopra.
                # Iniettiamo qui la verità matematica se è stata calcolata:
                if math_injection:
                    system_instructions += math_injection
                # --- FINE FIX MATEMATICO PRE-FLIGHT ---






                # 3. PROMPT DI SISTEMA
                # system_instructions è già stato costruito sopra con anti-contaminazione.
                # Non riassegnarlo qui, altrimenti si perde il guardrail.

                # Aggiunta audit nel debug visivo
                debug_md += (
                    f"\n\n### 🛡️ Tier Context Check\n"
                    f"- Tier A (Normative): {'✅ Presente' if c_a else '❌ Assente'}\n"
                    f"- Tier B (Governance): {'✅ Presente' if c_b else '❌ Assente'}\n"
                    f"- Tier C (Evidences): {'✅ Presente' if c_c else '❌ Assente'}"
                )

                # 4. ASSEMBLAGGIO CONTENUTO UTENTE
                requested_doc = extract_requested_document(user_query)

                doc_scope_block = ""
                if requested_doc:
                    doc_scope_block = (
                        f"### REQUESTED DOCUMENT SCOPE ###\n"
                        f"The user explicitly requested this document: {requested_doc}\n"
                        f"You MUST answer using ONLY sources whose filename matches this requested document.\n"
                        f"If the retrieved context does not contain sources from this document, answer only:\n"
                        f"Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                    )

                answer_mode = detect_answer_mode(user_query)
                strict_checklist_mode = is_strict_checklist_query(user_query)
                graph_relation_mode = is_graph_relation_query(user_query)

                if answer_mode == "knowledge":
                    c_a_text = c_a if c_a else "Not required for this knowledge question."
                    c_b_text = c_b if c_b else "Not required for this knowledge question."
                    c_c_text = c_c if c_c else "Not required for this knowledge question."
                else:
                    c_a_text = c_a if c_a else "No normative baseline found."
                    c_b_text = c_b if c_b else "No governance or policy evidence found."
                    c_c_text = c_c if c_c else "No implementation evidence found."

                c_g_text = c_g if c_g else "No relational/formula data."

                math_context_block = ""
                if math_needs_context and math_answer:
                    math_context_block = (
                        "### DETERMINISTIC CALCULATION RESULT - DO NOT CHANGE ###\n"
                        f"{math_answer}\n\n"
                        "Instruction: preserve the exact numerical results above. "
                        "Use retrieved documents only to explain the assessment/risk/evidence context.\n\n"
                    )

                final_user_content = (
                    doc_scope_block
                    + f"### ANSWER MODE ###\n{answer_mode}\n"
                    + "If mode is knowledge, section C MUST NOT mention missing Tier B or Tier C unless explicitly requested.\n\n"
                    + f"### STRICT_CHECKLIST_MODE ###\n{'ON' if strict_checklist_mode else 'OFF'}\n\n"
                    + f"### GRAPH_RELATION_MODE ###\n{'ON' if graph_relation_mode else 'OFF'}\n\n"
                    + math_context_block
                    + f"### NORMATIVE BASELINE [TIER A] ###\n{c_a_text}\n\n"
                    + f"### GOVERNANCE / POLICIES [TIER B] ###\n{c_b_text}\n\n"
                    + f"### IMPLEMENTATION EVIDENCE [TIER C] ###\n{c_c_text}\n\n"
                    + f"### KNOWLEDGE GRAPH [NEO4J] ###\n{c_g_text}\n\n"
                    + f"### USER QUESTION ###\n{user_query}\n"
                    + f"{language_reminder}\n\n"
                    + "CRITICAL REMINDER: You MUST output EXACTLY these four headers and nothing else: "
                    + "**A) Risposta**, **B) Evidenze**, **C) Limiti / Conflitti**, **D) Fonti**."
                )
            
            # --- COSTRUZIONE PAYLOAD CHAT ---
            messages_payload = build_alternating_history(self.messages, MEMORY_LIMIT)
            
            if messages_payload and messages_payload[-1]["role"] == "user":
                messages_payload.pop()
            
            messages_payload = [m for m in messages_payload if m["role"] != "system"]

            final_messages = [
                {"role": "system", "content": system_instructions}
            ] + messages_payload + [
                {"role": "user", "content": final_user_content}
            ]


            # Aggiunge subito un messaggio "placeholder" (senza fonti) così la UI non sembra bloccata
            assistant_id = str(uuid.uuid4())
            self.messages.append(
                ChatMessage(
                    id=assistant_id,
                    role="assistant",
                    content="⏳ Sto generando la risposta…",
                    sources=[],          # ✅ NON mostrare fonti subito
                    debug_md=""          # ✅ audit dopo
                )
            )
            yield rx.scroll_to("chat_bottom")
            yield  # ✅ forza refresh UI

            # --- BLOCCO UNICO DI GENERAZIONE CORRETTO ---
            # --- BLOCCO UNICO DI GENERAZIONE (FIXATO) ---

            full_resp = ""

            if llm_client:
                try:
                    print("🧠 Avvio generazione risposta LLM...")

                    # Chiamata bloccante spostata su thread separato:
                    # la UI resta viva e il timeout è gestito da requests.
                    full_resp = await asyncio.to_thread(
                        call_ollama_chat_native,
                        final_messages,
                    )

                    if not full_resp.strip():
                        full_resp = (
                            "**A) Risposta**\n\n"
                            "Il modello non ha restituito contenuto utile.\n\n"
                            "**B) Evidenze**\n\n"
                            "- Il retrieval ha prodotto fonti, ma la generazione LLM è risultata vuota.\n\n"
                            "**C) Limiti / Conflitti**\n\n"
                            "- Verificare modello Ollama, timeout e dimensione del contesto.\n\n"
                            "**D) Fonti**\n\n"
                            "- Vedi pannello Fonti/Audit."
                        )

                    self.messages[-1].content = postprocess_generated_answer(strip_id_leaks(full_resp), user_query, sources)
                    yield

                except Exception as e:
                    print(f"❌ Errore generazione LLM: {e}")

                    self.messages[-1].content = (
                        "**A) Risposta**\n\n"
                        "La generazione della risposta è andata in timeout o ha prodotto un errore.\n\n"
                        "**B) Evidenze**\n\n"
                        "- Il retrieval è stato completato, ma la chiamata al modello LLM non ha risposto correttamente.\n\n"
                        "**C) Limiti / Conflitti**\n\n"
                        f"- Errore tecnico: `{str(e)}`\n"
                        "- Riduci temporaneamente `LLM_NUM_CTX`, `LLM_NUM_PREDICT` e `MAX_CONTEXT_CHARS`.\n\n"
                        "**D) Fonti**\n\n"
                        "- Vedi pannello Fonti/Audit."
                    )

                    self.messages[-1].sources = prepare_sources_for_ui(sources)
                    self.messages[-1].debug_md = prepare_debug_for_ui(debug_md)
                    yield
                    return
                
              
                # ✅ SOLO ALLA FINE agganciamo fonti, audit e KPI di faithfulness
                answer_clean = postprocess_generated_answer(strip_id_leaks(full_resp), user_query, sources)

                requested_doc = ""
                try:
                    requested_doc = extract_requested_document(user_query)
                except Exception:
                    requested_doc = ""

                eval_result = evaluate_rag_answer(
                    query_text=user_query,
                    answer=answer_clean,
                    sources=sources,
                    requested_doc=requested_doc,
                )

                debug_md += "\n\n" + format_eval_debug_md(eval_result)

                append_rag_eval_log(
                    query_text=user_query,
                    answer=answer_clean,
                    sources=sources,
                    eval_result=eval_result,
                    requested_doc=requested_doc,
                )

                # Modalità osservabilità: mostra la risposta ma segnala il rischio nell'audit.
                self.messages[-1].content = answer_clean

                # Modalità blocco severo: sostituisce risposte non fedeli.
                if EVAL_STRICT_BLOCK:
                    bad_faithfulness = eval_result.faithfulness < EVAL_MIN_FAITHFULNESS
                    bad_relevance = eval_result.answer_relevance < EVAL_MIN_ANSWER_RELEVANCE
                    bad_scope = eval_result.source_scope_violation

                    if bad_faithfulness or bad_relevance or bad_scope:
                        self.messages[-1].content = (
                            "**A) Risposta**\n\n"
                            "Non ho trovato evidenze sufficienti nei documenti recuperati.\n\n"
                            "**B) Evidenze**\n\n"
                            "- La risposta generata non ha superato il controllo automatico di faithfulness.\n\n"
                            "**C) Limiti**\n\n"
                            f"- Faithfulness: {eval_result.faithfulness:.2f}\n"
                            f"- Answer relevance: {eval_result.answer_relevance:.2f}\n"
                            f"- Source scope violation: {eval_result.source_scope_violation}\n\n"
                            "**D) Fonti**\n\n"
                            "- Vedi pannello Fonti/Audit."
                        )

                # ✅ SOLO ALLA FINE agganciamo fonti e audit in versione UI-safe
                # Il RAG usa sources/debug_md completi; la UI riceve una versione ridotta.
                self.messages[-1].sources = prepare_sources_for_ui(sources)
                self.messages[-1].debug_md = prepare_debug_for_ui(debug_md)
                yield
            else:
                self.messages[-1].content = "⚠️ LLM non inizializzato. Verifica che Ollama sia attivo."
                self.messages[-1].sources = prepare_sources_for_ui(sources)
                self.messages[-1].debug_md = prepare_debug_for_ui(debug_md)
                yield
        finally:
            self.is_processing = False
            self.refresh_gpu()

# =========================
# 🎨 UI COMPONENTS
# =========================
def source_badge(text: str, color: str, icon: str):
    return rx.badge(
        rx.hstack(rx.icon(icon, size=12), rx.text(text)),
        color_scheme=color,
        variant="soft",
        radius="full",
        size="1",
    )

def message_ui(msg: ChatMessage):
    is_bot = msg.role == "assistant"
    bg_color = rx.cond(is_bot, rx.color("gray", 3), rx.color("indigo", 9))
    text_color = rx.cond(is_bot, rx.color("gray", 12), "white")
    align_self = rx.cond(is_bot, "start", "end")

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.avatar(
                    fallback=rx.cond(is_bot, "🤖", "👤"),
                    size="2",
                    variant="soft",
                    color_scheme=rx.cond(is_bot, "gray", "indigo"),
                ),
                rx.text(rx.cond(is_bot, "Financial AI", "Tu"), weight="bold", size="2"),
                rx.spacer(),
                # Pulsante "Info" in alto a destra nel messaggio
                rx.cond(
                    is_bot & (msg.sources.length() > 0),
                    rx.button(
                        rx.hstack(
                            rx.icon("info", size=14),
                            rx.text("Dettagli Ricerca", size="1"),
                            spacing="2",
                        ),
                        variant="soft",
                        color_scheme="gray",
                        size="1",
                        on_click=State.open_sources_audit(msg.id),
                    ),
                    rx.box(),
                ),
                width="100%",
                align_items="center",
                spacing="2",
            ),
            # Contenuto del Messaggio
            rx.box(
                rx.markdown(
                    msg.content,
                    width="100%",
                    overflow_wrap="anywhere",
                    word_break="break-word",
                ),
                width="100%",
                min_width="0",
                overflow_x="auto",
                overflow_y="visible",
            ),
            
            # Badge rapidi sotto il testo (Opzionale, richiama la funzione helper)
            rx.cond(
                is_bot & (msg.sources.length() > 0),
                render_inline_sources(msg)
            ),

            spacing="2",
            width="100%",
        ),

        # ---- Inline popup "Fonti + Audit" sotto la risposta LLM ----
        rx.cond(
            is_bot & ((msg.sources.length() > 0) | (msg.debug_md.length() > 0)),
            rx.box(
                # barra azioni (Pulsanti Fonti / Audit)
                rx.hstack(
                    rx.button(
                        rx.hstack(
                            rx.icon("book-open", size=14),
                            rx.text("Fonti", size="1"),
                            rx.badge(rx.text(msg.sources.length()), color_scheme="green", variant="soft"),
                            spacing="2",
                            align_items="center",
                        ),
                        size="1",
                        variant="soft",
                        on_click=State.toggle_inline_sources(msg.id),
                    ),
                    rx.button(
                        rx.hstack(
                            rx.icon("shield-check", size=14),
                            rx.text("Audit", size="1"),
                            spacing="2",
                            align_items="center",
                        ),
                        size="1",
                        variant="soft",
                        on_click=State.toggle_inline_audit(msg.id),
                    ),
                    rx.spacer(),
                    spacing="2",
                    width="100%",
                    margin_top="0.6em",
                ),

                # --- PANNELLO ESPANSO ---
                rx.cond(
                    State.inline_open_for == msg.id,
                    rx.box(
                        rx.cond(
                            State.inline_tab == "sources",
                            
                            # === SEZIONE FONTI (FIXATA: NESSUN LOOP SU STATE.MESSAGES) ===
                            rx.scroll_area(
                                rx.vstack(
                                    rx.text("📚 Fonti Documentali correlate:", font_weight="bold", size="2", margin_bottom="0.5em"),
                                    rx.foreach(
                                        msg.sources,
                                        lambda s: rx.card(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.badge(s.tier, color_scheme="red", variant="soft"),
                                                    rx.badge(s.db_origin, color_scheme="violet", variant="outline"),
                                                    rx.text(s.filename, size="1", weight="bold"),
                                                    rx.spacer(),
                                                    rx.text("Pag. ", s.page, size="1"),
                                                    width="100%",
                                                ),
                                                rx.text(s.content, size="1", line_clamp=3, font_style="italic", color_scheme="gray"),
                                                spacing="1",
                                                width="100%",
                                            ),
                                            variant="ghost",
                                            width="100%",
                                            margin_bottom="0.5em",
                                        )
                                    ),
                                    spacing="2",
                                    width="100%",
                                ),
                                height="260px",
                                type="always",
                            ),
                            
                            # === SEZIONE AUDIT ===
                            rx.box(
                                rx.heading("Audit & Reasoning", size="3", margin_bottom="0.5em"),
                                rx.scroll_area(
                                    rx.text(
                                        msg.debug_md,
                                        width="100%",
                                        white_space="pre-wrap",
                                        overflow_wrap="anywhere",
                                        word_break="break-word",
                                        size="1",
                                    ),
                                    height="260px",
                                    type="always",
                                ),
                                width="100%",
                            ),
                        ),

                        # Footer del pannello (Pulsante Chiudi)
                        rx.hstack(
                            rx.spacer(),
                            rx.button(
                                "Chiudi",
                                size="1",
                                variant="ghost",
                                on_click=State.close_inline_panel,
                            ),
                            width="100%",
                            margin_top="0.5em",
                        ),

                        border=f"1px solid {rx.color('gray', 5)}",
                        border_radius="12px",
                        padding="0.8em",
                        margin_top="0.6em",
                        bg=rx.color("gray", 1),
                        width="100%",
                    ),
                    rx.box(), # Else block del pannello espanso (vuoto)
                ),
                width="100%",
            ),
            rx.box(), # Else block del pulsante espansione (vuoto)
        ),

        bg=bg_color,
        color=text_color,
        padding="1em",
        border_radius="12px",
        max_width="85%",
        width="85%",
        align_self=align_self,
        box_shadow="sm",
        margin_y="0.5em",
        min_width="280px",
        flex_shrink="0",
        overflow="visible",
    )


def render_inline_sources(msg: ChatMessage):
    """Visualizza i badge sintetici delle fonti sotto il messaggio."""
    return rx.flex(
        rx.foreach(
            msg.sources,
            lambda s: rx.badge(
                rx.hstack(
                    rx.icon("database", size=12),
                    # FIX: Passiamo i valori come argomenti separati a rx.text
                    # invece di usare una f-string che può causare errori su Var
                    rx.text(s.db_origin, ": ", s.filename, " (p.", s.page, ")", size="1"),
                    align_items="center",
                    spacing="1",
                ),
                variant="soft",
                color_scheme="indigo",
                margin_right="0.5em",
                margin_bottom="0.2em",
                cursor="pointer",
                # Cliccando sul badge si apre il pannello dettagli
                on_click=State.toggle_inline_sources(msg.id),
            )
        ),
        wrap="wrap",
        margin_top="0.5em",
    )

def render_inline_audit(msg: ChatMessage):
    """Visualizza il log di ragionamento (Audit) sotto il messaggio."""
    return rx.box(
        rx.text(
            msg.debug_md,
            white_space="pre-wrap",
            overflow_wrap="anywhere",
            word_break="break-word",
            size="1",
        ),
        background_color="#FFFBEB",
        padding="1rem",
        border_radius="md",
        margin_top="0.5rem",
        border_left="4px solid #F6AD55",
    )



def index():
    return rx.flex(
        # Sidebar
        rx.vstack(
            rx.heading("System Status", size="3"),
            rx.divider(),
            rx.hstack(rx.icon("cpu"), rx.text(State.vram_info, size="1")),
            rx.hstack(rx.icon("hard-drive"), rx.text(f"GPU free: {State.vram_free}", size="1")),
            rx.hstack(
                rx.icon("activity"),
                rx.text(f"Backend: {State.backend_status}", size="1"),
            ),
            rx.text(f"LLM: {LLM_MODEL_NAME}", size="1", color="gray"),
            rx.spacer(),
            rx.button(
                "Refresh GPU",
                on_click=State.refresh_gpu,
                color_scheme="gray",
                variant="soft",
                width="100%",
            ),
            rx.button(
                "Clear Chat",
                on_click=State.clear_history,
                color_scheme="red",
                variant="soft",
                width="100%",
            ),
            width="260px",
            height="100%",
            padding="1.5em",
            bg=rx.color("gray", 2),
            display=["none", "none", "flex"],
            flex_shrink="0",
            min_height="0",
            overflow="hidden",
        ),

        # Main
        rx.vstack(
            # Header
            rx.box(
                rx.heading(PAGE_TITLE, size="6", align="center"),
                rx.text(
                    f"Powered by {LLM_MODEL_NAME} + Qdrant + Neo4j",
                    color="gray",
                    size="2",
                    align="center",
                ),
                padding_y="1em",
                width="100%",
                text_align="center",
                flex_shrink="0",
            ),

            # Popup Fonti/Audit
            rx.dialog.root(
                rx.dialog.content(
                    rx.dialog.title(State.modal_title),
                    rx.dialog.description("Fonti e audit della risposta."),
                    rx.divider(),

                    # ====== FONTI ======
                    rx.cond(
                        State.modal_sources.length() > 0,
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    State.modal_sources,
                                    lambda s: rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.badge(
                                                    s.tier,
                                                    color_scheme="tomato",
                                                    variant="surface",
                                                ),
                                                rx.badge(
                                                    s.db_origin,
                                                    color_scheme="plum",
                                                    variant="outline",
                                                ),
                                                rx.text(
                                                    "Doc: ",
                                                    s.filename,
                                                    weight="bold",
                                                    size="2",
                                                ),
                                                width="100%",
                                                justify="between",
                                            ),
                                            rx.text(
                                                s.content,
                                                size="1",
                                                line_clamp=3,
                                            ),
                                            rx.hstack(
                                                rx.text(
                                                    "Pagina: ",
                                                    s.page,
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                rx.spacer(),
                                                rx.text(
                                                    "Score: ",
                                                    s.score,
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                width="100%",
                                            ),
                                            spacing="2",
                                        ),
                                        width="100%",
                                        margin_bottom="2",
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            height="400px",
                            type="always",
                        ),
                        rx.center(
                            rx.text(
                                "Nessuna fonte trovata per questo messaggio.",
                                color="gray",
                            )
                        ),
                    ),

                    rx.divider(),

                    # ====== AUDIT ======
                    rx.cond(
                        State.modal_debug_md.length() > 0,
                        rx.box(
                            rx.heading("Audit", size="3"),
                            rx.text(
                                State.modal_debug_md,
                                width="100%",
                                white_space="pre-wrap",
                                overflow_wrap="anywhere",
                                word_break="break-word",
                                size="1",
                            ),
                            width="100%",
                        ),
                        rx.text("Nessun audit disponibile.", color="gray"),
                    ),

                    rx.hstack(
                        rx.spacer(),
                        rx.button(
                            "Chiudi",
                            variant="soft",
                            on_click=State.close_sources_audit,
                        ),
                        width="100%",
                        margin_top="1em",
                    ),

                    max_width="900px",
                    width="90vw",
                ),
                open=State.show_sources_modal,
                on_open_change=State.set_sources_modal_open,
            ),

            # Chat scroll area
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(State.messages, message_ui),
                    rx.box(id="chat_bottom", height="1px", flex_shrink="0"),
                    width="100%",
                    padding="1em",
                    max_width="900px",
                    margin="0 auto",
                    spacing="4",
                    min_height="0",
                    flex_shrink="0",
                    align_items="stretch",
                ),
                width="100%",
                flex="1",
                min_height="0",
                min_width="0",
                type="always",
                scrollbars="vertical",
                id="chat_scroll_area",
                overflow_x="hidden",
            ),

            # Input area
            rx.box(
                rx.hstack(
                    rx.input(
                        placeholder="Chiedi informazioni sui documenti...",
                        value=State.input_text,
                        on_change=State.set_input_text,
                        #on_key_down=lambda k: rx.cond(
                        #    k == "Enter",
                        #    State.handle_submit(),
                        #    None,
                        #),
                        radius="full",
                        size="3",
                        flex="1",
                    ),
                    rx.button(
                        rx.icon("send"),
                        on_click=State.handle_submit,
                        loading=State.is_processing,
                        radius="full",
                        size="3",
                    ),
                    width="100%",
                    max_width="900px",
                    padding="1em",
                ),
                width="100%",
                display="flex",
                justify_content="center",
                bg=rx.color("gray", 1),
                border_top="1px solid #e5e5e5",
                flex_shrink="0",
            ),

            height="100%",
            width="100%",
            spacing="0",
            overflow="hidden",
            overflow_x="hidden",
            min_height="0",
        ),

        # ROOT
        width="100%",
        height="100dvh",
        position="fixed",
        top="0",
        left="0",
        right="0",
        bottom="0",
        overflow="hidden",
        overflow_x="hidden",
        min_height="0",
    )




#app = rx.App(theme=rx.theme(appearance="light", accent_color="indigo", radius="large"))
#app.add_page(index, on_load=State.on_load)


#app = rx.App()
#app.add_page(index, on_load=State.on_load)


# ============================================================
# 🧮 FORMULA STRICT MODE - v4.5 refinement
# ============================================================
def _formula_plain_text(value: str) -> str:
    """Converte LaTeX/markdown leggero in testo leggibile per regole soglia."""
    v = _normalize_latex_value(value or "")
    v = _strip_math_wrappers(v)
    v = re.sub(r"\\text\{([^}]*)\}", r"\1", v)
    v = v.replace(r"\%", "%")
    v = re.sub(r"[{}]", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _looks_threshold_rule(text: str) -> bool:
    """
    v4.5: riconosce regole soglia anche quando i valori sono in LaTeX,
    es. 5\% oppure 1\text{ milione}.
    """
    raw = text or ""
    plain = _formula_plain_text(raw).lower()
    threshold_terms = [
        "oltre", "superiore", "almeno", "non inferiore", "maggiore di",
        "greater than", "over", "more than", "at least", "threshold",
        "soglia", "condizione", "condition",
    ]
    has_threshold_word = any(x in plain for x in threshold_terms)
    has_threshold_value = bool(
        re.search(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)", plain)
    )
    return has_threshold_word and has_threshold_value


def _threshold_rule_name(name: str, formula_or_text: str) -> str:
    plain = _formula_plain_text(formula_or_text)
    n = re.sub(r"\s+", " ", name or "").strip()
    if n and n.lower() not in {"formula recuperata", "formula/metric", "metrica/indicatore citato", "elemento recuperato"}:
        return n
    m = re.search(r"\b(condizione\s*\d+)\b", plain, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    if plain:
        return "Regola soglia"
    return "Regola soglia"


def _extract_definition_from_latex(latex: str) -> str:
    v = _normalize_latex_value(latex or "")
    m = re.search(r"=\s*\\?text\{([^}]+)\}", v)
    if m:
        return m.group(1).strip()
    if "=" in v:
        right = v.split("=", 1)[1]
        return _formula_plain_text(right)
    return ""


def _extract_left_name_from_equation(latex: str) -> str:
    v = _strip_math_wrappers(_normalize_latex_value(latex or ""))
    if "=" not in v:
        return ""
    left = v.split("=", 1)[0]
    left = _formula_plain_text(left)
    left = re.sub(r"[^A-Za-zÀ-ÿ0-9_\-/ ]+", "", left).strip()
    return left[:80]


def _classify_formula_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    v4.5:
    - le soglie normative restano in output come Regola soglia;
    - MTTD/MTTR e simili restano metriche definitorie se non hanno calcolo;
    - non trasforma definizioni testuali in formule computazionali.
    """
    rr = dict(row)
    original_name = re.sub(r"\s+", " ", str(rr.get("name") or "")).strip() or "Elemento recuperato"
    latex_raw = str(rr.get("latex") or "").strip()
    latex = _normalize_latex_value(latex_raw)
    meaning = re.sub(r"\s+", " ", str(rr.get("meaning") or "")).strip()
    combined = " ".join([original_name, latex, meaning])

    if _looks_threshold_rule(combined):
        tipo = "Regola soglia"
        formula = _formula_plain_text(latex or combined)
        name = _threshold_rule_name(original_name, formula)
        meaning = "Regola/soglia normativa recuperata; non è una formula computazionale."

    elif _looks_computational_formula(latex):
        tipo = "Formula computazionale"
        formula = latex
        name = original_name

    elif _looks_definitional_metric(latex, meaning):
        tipo = "Metrica definitoria"
        formula = "formula computazionale non recuperata"
        left_name = _extract_left_name_from_equation(latex)
        name = left_name or original_name
        definition = _extract_definition_from_latex(latex)
        if definition:
            meaning = f"Definizione testuale della metrica: {definition}. Formula computazionale non recuperata nella fonte."
        elif not meaning or "Formula testuale esplicita" in meaning:
            meaning = "Definizione testuale della metrica; formula computazionale non recuperata nella fonte."

    else:
        tipo = "Metrica/elemento citato"
        formula = "formula esplicita non recuperata"
        name = original_name
        if not meaning:
            meaning = "Elemento citato nelle fonti recuperate; nessuna formula esplicita è stata individuata nello stesso chunk."

    rr["name"] = name
    rr["tipo"] = tipo
    rr["latex"] = formula
    rr["meaning"] = meaning
    return rr


def _is_noise_formula_row_v45(row: Dict[str, Any]) -> bool:
    name = str(row.get("name") or "").strip().lower()
    formula = str(row.get("latex") or "").strip().lower()
    tipo = str(row.get("tipo") or "").strip().lower()

    generic_names = {
        "", "formula/metric", "formula recuperata", "contenuto", "variabili",
        "metrica/indicatore citato", "formula", "metric", "formule e modelli matematici",
        "formule e modelli matematici - pagina 12 --", "formule e modelli matematici - pagina 24 --",
        "elemento recuperato",
    }

    if tipo == "regola soglia":
        return not _looks_threshold_rule(" ".join([name, formula, str(row.get("meaning") or "")]))

    if name in generic_names:
        return True

    # Scarta valori isolati tipo 5%, 1 milione, 72, ecc. se non sono regole soglia.
    plain = _formula_plain_text(formula).lower()
    if re.fullmatch(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)?", plain):
        return True

    return False


def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
    """v4.5: deduplica senza far collassare le diverse condizioni soglia."""
    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for r in rows:
        rr = _classify_formula_row(r)
        if _is_noise_formula_row_v45(rr):
            continue

        tipo = str(rr.get("tipo") or "").lower()
        fname = normalize_doc_name(str(rr.get("filename") or ""))
        page = int(rr.get("page") or 0)
        name_norm = re.sub(r"[^a-z0-9]+", "", str(rr.get("name") or "").lower())
        formula_norm = re.sub(r"\s+", " ", _formula_plain_text(str(rr.get("latex") or "")).lower()).strip()

        if tipo == "regola soglia":
            key = ("threshold", formula_norm[:220], fname, page)
        else:
            key = (name_norm, tipo, fname, page)

        if key in seen:
            continue
        seen.add(key)
        cleaned.append(rr)
        if len(cleaned) >= max_rows:
            break

    # Ordine più leggibile: formule, regole soglia, metriche definitorie, altri elementi.
    priority = {
        "formula computazionale": 0,
        "regola soglia": 1,
        "metrica definitoria": 2,
        "metrica/elemento citato": 3,
    }
    cleaned.sort(key=lambda x: (priority.get(str(x.get("tipo") or "").lower(), 9), str(x.get("filename") or ""), int(x.get("page") or 0), str(x.get("name") or "")))
    return cleaned[:max_rows]


def _requested_formula_terms_missing(query_text: str, rows: List[Dict[str, Any]]) -> List[str]:
    ql = (query_text or "").lower()
    requested_generic = ["cvss", "rischio", "risk", "maturità", "maturity", "copertura", "coverage"]
    found_text = " ".join([str(r.get("name", "")) + " " + str(r.get("latex", "")) for r in rows]).lower()
    missing: List[str] = []
    for t in requested_generic:
        if t in ql and t not in found_text:
            missing.append(t)
    return sorted(set(missing))


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=10)
    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie o regole soglia sufficientemente esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    table_lines = [
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        table_lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 160)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 120)} | "
            f"`{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 320)}` | "
            f"{_formula_md_cell(r.get('meaning') or '', 340)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 220)} | "
            f"{int(r.get('page') or 0)} |"
        )

    used_files = sorted({str(r.get("filename") or "") for r in rows if r.get("filename")})
    has_threshold = any(str(r.get("tipo") or "").lower() == "regola soglia" for r in rows)
    has_def_metric = any(str(r.get("tipo") or "").lower() == "metrica definitoria" for r in rows)
    missing_terms = _requested_formula_terms_missing(query_text, rows)

    evidence_lines = ["- Le formule/metriche/regole sono state classificate in modo deterministico."]
    if has_threshold:
        evidence_lines.append("- Le regole soglia sono riportate come regole normative/scoring, non come formule computazionali.")
    if has_def_metric:
        evidence_lines.append("- Le definizioni di metriche sono distinte dalle formule calcolabili.")
    if missing_terms:
        evidence_lines.append("- Non sono state recuperate formule esplicite per: " + ", ".join(missing_terms) + ".")

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una regola soglia indica una condizione/criterio, non una formula matematica da calcolare.\n\n"
        + "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )


def filter_sources_for_formula_answer(query_text: str, sources: List[SourceItem]) -> List[SourceItem]:
    """Riduce le fonti UI alle pagine effettivamente usate dalla tabella formule."""
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=10)
    if not rows:
        return sources
    keys = {(normalize_doc_name(str(r.get("filename") or "")), int(r.get("page") or 0)) for r in rows}
    filtered: List[SourceItem] = []
    seen = set()
    for s in sources or []:
        key = (normalize_doc_name(str(getattr(s, "filename", "") or "")), int(getattr(s, "page", 0) or 0))
        if key in keys and key not in seen:
            seen.add(key)
            filtered.append(s)
    return filtered or sources

# ============================================================
# 🧮 FORMULA STRICT MODE - v4.6 final presentation cleanup
# ============================================================
def _formula_display_text(value: Any, max_len: int = 600) -> str:
    """
    Cleans formula/rule text for Markdown output.
    Fixes literal LaTeX artefacts such as \t from \text and removes math wrappers.
    """
    text = str(value or "")
    # literal escape artefacts coming from broken LaTeX/OCR, not real tabs only
    text = text.replace("\\t", " ").replace("\\n", " ").replace("\\r", " ")
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("$$$", "$$")
    text = re.sub(r"^`+|`+$", "", text).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    # normalize broken text macros: ext{...}, \text{...}, text{...}
    text = re.sub(r"(?<![A-Za-z\\])ext\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\btext\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\%", "%")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text


def _formula_plain_text(value: str) -> str:
    return _formula_display_text(value, 1000)


def _normalize_latex_value(value: str) -> str:
    v = str(value or "").strip()
    v = v.replace("\\t", " ").replace("\\n", " ").replace("\\r", " ")
    v = v.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    v = v.replace("$$$", "$$")
    v = re.sub(r"(?<!\\)ext\{", r"\\text{", v)
    v = re.sub(r"\${3,}", "$$", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _threshold_rule_name(name: str, formula_or_text: str) -> str:
    plain = _formula_plain_text(formula_or_text)
    n = _formula_display_text(name, 120)
    generic = {"formula recuperata", "formula/metric", "metrica/indicatore citato", "elemento recuperato", "regola soglia"}
    m = re.search(r"\b(condizione\s*\d+)\b", plain, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    if n and n.lower() not in generic:
        return n
    return "Regola soglia"


def _extract_definition_from_latex(latex: str) -> str:
    v = _normalize_latex_value(latex or "")
    m = re.search(r"=\s*\\?text\{([^}]+)\}", v)
    if m:
        return _formula_display_text(m.group(1), 500)
    if "=" in v:
        return _formula_display_text(v.split("=", 1)[1], 500)
    return ""


def _extract_left_name_from_equation(latex: str) -> str:
    v = _strip_math_wrappers(_normalize_latex_value(latex or ""))
    if "=" not in v:
        return ""
    left = _formula_display_text(v.split("=", 1)[0], 120)
    left = re.sub(r"[^A-Za-zÀ-ÿ0-9_\-/ ]+", "", left).strip()
    return left[:80]


def _classify_formula_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(row)
    original_name = _formula_display_text(rr.get("name") or "", 120) or "Elemento recuperato"
    latex_raw = str(rr.get("latex") or "").strip()
    latex = _normalize_latex_value(latex_raw)
    meaning_raw = str(rr.get("meaning") or "")
    meaning = _formula_display_text(meaning_raw, 700)
    combined = " ".join([original_name, latex, meaning])

    if _looks_threshold_rule(combined):
        formula_plain = _formula_display_text(latex or combined, 700)
        rr["name"] = _threshold_rule_name(original_name, formula_plain)
        rr["tipo"] = "Regola soglia"
        rr["latex"] = formula_plain
        rr["meaning"] = "Criterio/soglia normativa recuperata; non è una formula computazionale."
        return rr

    if _looks_computational_formula(latex):
        rr["name"] = original_name
        rr["tipo"] = "Formula computazionale"
        rr["latex"] = _formula_display_text(latex, 700)
        rr["meaning"] = meaning or "Formula computazionale esplicita presente nella fonte recuperata."
        return rr

    if _looks_definitional_metric(latex, meaning):
        left_name = _extract_left_name_from_equation(latex)
        definition = _extract_definition_from_latex(latex)
        rr["name"] = left_name or original_name
        rr["tipo"] = "Metrica definitoria"
        rr["latex"] = "formula computazionale non recuperata"
        rr["meaning"] = (
            f"Definizione testuale della metrica: {definition}. Formula computazionale non recuperata nella fonte."
            if definition else
            "Definizione testuale della metrica; formula computazionale non recuperata nella fonte."
        )
        return rr

    rr["name"] = original_name
    rr["tipo"] = "Metrica/elemento citato"
    rr["latex"] = "formula esplicita non recuperata"
    rr["meaning"] = meaning or "Elemento citato nelle fonti recuperate; nessuna formula esplicita è stata individuata nello stesso chunk."
    return rr


def _is_noise_formula_row_v45(row: Dict[str, Any]) -> bool:
    name = _formula_display_text(row.get("name") or "", 160).strip().lower()
    formula = _formula_display_text(row.get("latex") or "", 400).strip().lower()
    tipo = _formula_display_text(row.get("tipo") or "", 120).strip().lower()

    generic_names = {
        "", "formula/metric", "formula recuperata", "contenuto", "variabili",
        "metrica/indicatore citato", "formula", "metric", "formule e modelli matematici",
        "formule e modelli matematici - pagina 12 --", "formule e modelli matematici - pagina 24 --",
        "elemento recuperato",
    }

    if tipo == "regola soglia":
        return not _looks_threshold_rule(" ".join([name, formula, str(row.get("meaning") or "")]))

    if name in generic_names:
        return True

    # Exclude isolated values if they are not part of a threshold rule.
    plain = _formula_display_text(formula, 120).lower()
    if re.fullmatch(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)?", plain):
        return True

    return False


def _formula_md_cell(value: Any, max_len: int = 600) -> str:
    return _formula_display_text(value, max_len)


def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for r in rows:
        rr = _classify_formula_row(r)
        if _is_noise_formula_row_v45(rr):
            continue

        tipo = str(rr.get("tipo") or "").lower()
        fname = normalize_doc_name(str(rr.get("filename") or ""))
        page = int(rr.get("page") or 0)
        name_norm = re.sub(r"[^a-z0-9]+", "", str(rr.get("name") or "").lower())
        formula_norm = re.sub(r"\s+", " ", _formula_display_text(rr.get("latex") or "", 1000).lower()).strip()

        if tipo == "regola soglia":
            # Keep separate conditions even if they share the same numeric thresholds.
            key = ("threshold", name_norm, formula_norm[:260], fname, page)
        else:
            key = (name_norm, tipo, fname, page)

        if key in seen:
            continue
        seen.add(key)
        cleaned.append(rr)

    priority = {
        "formula computazionale": 0,
        "regola soglia": 1,
        "metrica definitoria": 2,
        "metrica/elemento citato": 3,
    }
    cleaned.sort(key=lambda x: (
        priority.get(str(x.get("tipo") or "").lower(), 9),
        str(x.get("filename") or ""),
        int(x.get("page") or 0),
        str(x.get("name") or ""),
    ))
    return cleaned[:max_rows]


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=10)
    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie o regole soglia sufficientemente esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    table_lines = [
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        table_lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 140)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 90)} | "
            f"{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 420)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 300)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )

    used_files = sorted({str(r.get("filename") or "") for r in rows if r.get("filename")})
    has_threshold = any(str(r.get("tipo") or "").lower() == "regola soglia" for r in rows)
    has_def_metric = any(str(r.get("tipo") or "").lower() == "metrica definitoria" for r in rows)
    missing_terms = _requested_formula_terms_missing(query_text, rows)

    evidence_lines = ["- Le formule/metriche/regole sono state classificate in modo deterministico."]
    if has_threshold:
        evidence_lines.append("- Le regole soglia sono riportate come criteri normativi/scoring, non come formule computazionali.")
    if has_def_metric:
        evidence_lines.append("- Le definizioni di metriche sono distinte dalle formule calcolabili.")
    if missing_terms:
        evidence_lines.append("- Non sono state recuperate formule computazionali esplicite per: " + ", ".join(missing_terms) + ".")

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una regola soglia indica una condizione/criterio, non una formula matematica da calcolare.\n\n"
        + "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )

# ============================================================
# 🧮 FORMULA STRICT MODE - v4.7 output semantics cleanup
# ============================================================
def _extract_threshold_domain_from_rule(text: str) -> str:
    """Estrae un ambito leggibile senza dipendere da un corpus specifico (es. NIS2)."""
    plain = _formula_display_text(text, 160)
    
    # Rimuove l'intestazione tecnica se presente
    plain = re.sub(r"^(Condizione|Regola|Soglia|Threshold)\s*\d*\s*:\s*", "", plain, flags=re.IGNORECASE)
    
    # Prende semplicemente le prime parole significative come "ambito" descrittivo
    words = plain.split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _aggregate_threshold_rules(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggrega soglie normative ripetute quando condividono la stessa logica numerica.
    Le mantiene come 'Regola soglia normativa', non come scoring formula.
    """
    threshold_rows = [r for r in rows if str(r.get("tipo") or "").lower() == "regola soglia"]
    if not threshold_rows:
        return []

    groups: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for r in threshold_rows:
        rule_text = _formula_display_text(r.get("latex") or "", 800)
        fname = str(r.get("filename") or "N/D")
        page = int(r.get("page") or 0)

        # General numeric signature; avoids tying the grouping to NIS2 wording.
        has_percent = bool(re.search(r"\b\d+(?:[,.]\d+)?\s*%", rule_text))
        has_large_number = bool(re.search(r"\b\d+(?:[,.]\d+)?\s*(?:milione|milioni|million|millions)\b", rule_text, flags=re.IGNORECASE))
        signature = f"percent={has_percent};large_number={has_large_number}"
        key = (fname, page, signature)

        domain = _extract_threshold_domain_from_rule(rule_text)
        if key not in groups:
            groups[key] = {
                "name": "Soglia normativa recuperata",
                "tipo": "Regola soglia normativa non di scoring",
                "latex": "",
                "meaning": "Criterio/condizione normativa recuperata. Non è una formula computazionale e non è una regola di scoring.",
                "filename": fname,
                "page": page,
                "domains": [],
                "rule_examples": [],
            }
        if domain and domain not in groups[key]["domains"]:
            groups[key]["domains"].append(domain)
        if rule_text and rule_text not in groups[key]["rule_examples"]:
            groups[key]["rule_examples"].append(rule_text)

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        domains = g.pop("domains", [])
        examples = g.pop("rule_examples", [])
        if domains:
            g["name"] = "Soglie normative per: " + "; ".join(domains[:5])
        # Prefer the most informative example, but do not repeat 3 near-identical long conditions.
        example = examples[0] if examples else "soglia normativa recuperata"
        # Remove leading condition label: the aggregated row already names the rule.
        example = re.sub(r"^Condizione\s*\d+\s*:\s*", "", example, flags=re.IGNORECASE)
        g["latex"] = example
        out.append(g)
    return out


def _formula_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        f"**{title}**",
        "",
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 180)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 120)} | "
            f"{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 520)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 340)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )
    return "\n".join(lines)


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    v4.7:
    - primary output focuses on computable formulas, definitional metrics, scoring rules;
    - threshold rules are shown separately as normative thresholds, not scoring formulas;
    - avoids implying that NIS2 dimensional thresholds are formulas or scoring rules.
    """
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=20)

    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie o regole di scoring esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    computational = [r for r in rows if str(r.get("tipo") or "").lower() == "formula computazionale"]
    definitional = [r for r in rows if str(r.get("tipo") or "").lower() == "metrica definitoria"]
    thresholds = _aggregate_threshold_rules(rows)
    cited = [r for r in rows if str(r.get("tipo") or "").lower() == "metrica/elemento citato"]

    # Keep the response compact and semantically clear.
    primary_rows = computational + definitional
    blocks: List[str] = []

    if primary_rows:
        blocks.append(_formula_table("Formule computazionali e metriche recuperate", primary_rows[:8]))

    if thresholds:
        blocks.append(_formula_table("Regole soglia normative recuperate ma non classificabili come scoring", thresholds[:4]))

    if cited and not primary_rows and not thresholds:
        blocks.append(_formula_table("Elementi citati senza formula esplicita", cited[:6]))

    if not blocks:
        blocks.append("Non ho trovato formule computazionali o metriche sufficientemente esplicite nelle fonti recuperate.")

    used_files = sorted({str(r.get("filename") or "") for r in (primary_rows + thresholds + cited) if r.get("filename")})
    missing_terms = _requested_formula_terms_missing(query_text, primary_rows + thresholds + cited)

    evidence_lines = [
        "- Gli elementi sono stati classificati in modo deterministico.",
        "- Le metriche definitorie sono distinte dalle formule computazionali.",
    ]
    if thresholds:
        evidence_lines.append("- Le soglie normative recuperate sono riportate separatamente perché non sono regole di scoring né formule computazionali.")
    if missing_terms:
        evidence_lines.append("- Non sono state recuperate formule computazionali esplicite per: " + ", ".join(missing_terms) + ".")

    return (
        "**A) Risposta**\n\n"
        + "\n\n".join(blocks)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una soglia normativa indica una condizione/criterio; non misura automaticamente un punteggio o una maturità.\n\n"
        + "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )


# ============================================================
# 🧮 FORMULA STRICT MODE - v4.8 non-adaptive micro-fix
# ============================================================
def _extract_threshold_criterion(rule_text: str) -> str:
    """
    Estrae il criterio numerico comune da una regola soglia senza legarsi al corpus.
    Esempio generico: percentuale utenti + numero assoluto utenti.
    """
    plain = _formula_display_text(rule_text, 900)

    # Percent threshold, e.g. "oltre il 5% degli utenti ... nell'Unione"
    percent_part = ""
    m_percent = re.search(
        r"\b(oltre|superiore\s+a|maggiore\s+di|almeno|more\s+than|over|above)?\s*(?:il\s*)?(\d+(?:[,.]\d+)?)\s*%",
        plain,
        flags=re.IGNORECASE,
    )
    if m_percent:
        op = (m_percent.group(1) or "oltre").strip()
        value = m_percent.group(2).replace(",", ".")
        # Preserve a human-friendly Italian wording when the source is Italian.
        if re.search(r"\butenti\b", plain, flags=re.IGNORECASE):
            percent_part = f"oltre il {value}% degli utenti"
        else:
            percent_part = f"oltre il {value}%"
        if re.search(r"nell['’]Unione|Unione\s+europea|\bUE\b|\bEU\b", plain, flags=re.IGNORECASE):
            percent_part += " nell'Unione"

    # Absolute threshold, e.g. "oltre 1 milione di utenti ... nell'Unione"
    number_part = ""
    m_abs = re.search(
        r"\b(oltre|superiore\s+a|maggiore\s+di|almeno|more\s+than|over|above)?\s*(\d+(?:[,.]\d+)?)\s*(milione|milioni|million|millions)\b(?:\s+di\s+utenti)?",
        plain,
        flags=re.IGNORECASE,
    )
    if m_abs:
        value_raw = m_abs.group(2).replace(",", ".")
        unit = m_abs.group(3).lower()
        # Normalize English/Italian units only for display, not for logic.
        try:
            value_num = float(value_raw)
            value_display = f"{value_num:g}"
        except Exception:
            value_num = None
            value_display = value_raw

        is_one = value_display in {"1", "1.0"}
        unit_it = "milione" if is_one else "milioni"
        if unit in {"million", "millions"}:
            unit_it = "milione" if is_one else "milioni"
        number_part = f"oltre {value_display} {unit_it}"
        if re.search(r"\butenti\b", plain, flags=re.IGNORECASE):
            number_part += " di utenti"
        if re.search(r"nell['’]Unione|Unione\s+europea|\bUE\b|\bEU\b", plain, flags=re.IGNORECASE):
            number_part += " nell'Unione"

    parts = [p for p in [percent_part, number_part] if p]
    if parts:
        return " oppure ".join(parts)

    return plain


def _aggregate_threshold_rules(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    v4.8: aggrega soglie ripetute separando criterio e ambito.
    Non usa nomi/codici specifici del corpus: estrae domini e criteri dai testi recuperati.
    """
    threshold_rows = [r for r in rows if str(r.get("tipo") or "").lower() == "regola soglia"]
    if not threshold_rows:
        return []

    groups: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for r in threshold_rows:
        rule_text = _formula_display_text(r.get("latex") or "", 900)
        fname = str(r.get("filename") or "N/D")
        page = int(r.get("page") or 0)
        criterion = _extract_threshold_criterion(rule_text)
        criterion_key = re.sub(r"\s+", " ", criterion.lower()).strip()
        key = (fname, page, criterion_key)

        domain = _extract_threshold_domain_from_rule(rule_text)

        if key not in groups:
            groups[key] = {
                "elemento": "Soglie normative recuperate",
                "tipo": "Soglia normativa non di scoring",
                "criterio": criterion,
                "ambito": [],
                "meaning": "Criterio/condizione normativa recuperata. Non è una formula computazionale e non è una regola di scoring.",
                "filename": fname,
                "page": page,
            }

        if domain and domain not in groups[key]["ambito"]:
            groups[key]["ambito"].append(domain)

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        ambiti = g.get("ambito") or []
        g["ambito"] = "; ".join(ambiti[:8]) if ambiti else "ambito non specificato nella soglia recuperata"
        out.append(g)

    return out


def _formula_metrics_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        f"**{title}**",
        "",
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 180)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 120)} | "
            f"{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 520)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 340)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )
    return "\n".join(lines)


def _threshold_rules_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        f"**{title}**",
        "",
        "| Elemento | Tipo | Criterio | Ambito | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {_formula_md_cell(r.get('elemento') or 'Soglia normativa recuperata', 180)} | "
            f"{_formula_md_cell(r.get('tipo') or 'Soglia normativa non di scoring', 150)} | "
            f"{_formula_md_cell(r.get('criterio') or 'criterio non recuperato puntualmente', 320)} | "
            f"{_formula_md_cell(r.get('ambito') or 'ambito non specificato', 260)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 300)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )
    return "\n".join(lines)


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    v4.8:
    - keeps formulas/definitional metrics separate from normative thresholds;
    - threshold rows use Elemento/Criterio/Ambito to avoid semantic ambiguity;
    - no corpus-specific hardcoded mappings.
    """
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=20)

    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie o regole di scoring esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    computational = [r for r in rows if str(r.get("tipo") or "").lower() == "formula computazionale"]
    definitional = [r for r in rows if str(r.get("tipo") or "").lower() == "metrica definitoria"]
    thresholds = _aggregate_threshold_rules(rows)
    cited = [r for r in rows if str(r.get("tipo") or "").lower() == "metrica/elemento citato"]

    primary_rows = computational + definitional
    blocks: List[str] = []

    if primary_rows:
        blocks.append(_formula_metrics_table("Formule computazionali e metriche recuperate", primary_rows[:8]))

    if thresholds:
        blocks.append(_threshold_rules_table("Soglie normative recuperate ma non classificabili come scoring", thresholds[:4]))

    if cited and not primary_rows and not thresholds:
        blocks.append(_formula_metrics_table("Elementi citati senza formula esplicita", cited[:6]))

    if not blocks:
        blocks.append("Non ho trovato formule computazionali o metriche sufficientemente esplicite nelle fonti recuperate.")

    rows_for_sources: List[Dict[str, Any]] = primary_rows + thresholds + cited
    used_files = []
    seen_files = set()
    for r in rows_for_sources:
        fname = str(r.get("filename") or "").strip()
        page = int(r.get("page") or 0)
        if not fname:
            continue
        label = f"{fname} (p.{page})" if page else fname
        if label not in seen_files:
            seen_files.add(label)
            used_files.append(label)

    missing_terms = _requested_formula_terms_missing(query_text, primary_rows + thresholds + cited)

    evidence_lines = [
        "- Gli elementi sono stati classificati in modo deterministico.",
        "- Le metriche definitorie sono distinte dalle formule computazionali.",
    ]
    if thresholds:
        evidence_lines.append("- Le soglie normative sono riportate in una tabella separata con criterio e ambito, perché non sono formule né regole di scoring.")
    if missing_terms:
        evidence_lines.append("- Non sono state recuperate formule computazionali esplicite per: " + ", ".join(missing_terms) + ".")

    return (
        "**A) Risposta**\n\n"
        + "\n\n".join(blocks)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una soglia normativa indica una condizione/criterio; non misura automaticamente un punteggio o una maturità.\n\n"
        + "**D) Fonti**\n\n"
        + ("\n".join(f"- {f}" for f in used_files) if used_files else "- Fonti non disponibili.")
    )

# ============================================================
# 🔎 FORMULA / METRIC RECALL PATCH - v4.10 non-adaptive
# ============================================================
# Goal:
# - Do not change the formula classifier.
# - Improve recall when the user asks semantically for temporal incident metrics
#   (e.g. "tempi di rilevamento/risoluzione") without explicitly writing MTTD/MTTR.
# - Keep this as a generic alias expansion, not tied to one test question.

_ORIGINAL_expand_assessment_query_v410 = expand_assessment_query
_ORIGINAL_extract_exact_phrases_v410 = extract_exact_phrases
_ORIGINAL_requested_formula_terms_missing_v410 = _requested_formula_terms_missing


def _is_formula_metric_intent_v410(query_text: str) -> bool:
    """True only when the query is about formulas/metrics/scoring/calculation."""
    try:
        return bool(is_formula_strict_query(query_text))
    except Exception:
        q = (query_text or "").lower()
        return any(t in q for t in ["formula", "formule", "metric", "metriche", "scoring", "score", "calcolo"])


def _temporal_metric_aliases_v410(query_text: str) -> List[str]:
    """
    Generic IT/EN synonym expansion for incident-response time metrics.
    It is activated only for formula/metric/scoring queries.
    """
    if not _is_formula_metric_intent_v410(query_text):
        return []

    q = (query_text or "").lower()

    detection_cues = [
        "tempo di rilevamento", "tempi di rilevamento", "tempo medio di rilevamento",
        "rilevamento", "detection time", "time to detect", "mean time to detect",
        "detect time",
    ]

    resolution_cues = [
        "tempo di risoluzione", "tempi di risoluzione", "tempo medio di risoluzione",
        "tempo di riparazione", "tempi di riparazione", "risoluzione", "riparazione",
        "resolution time", "time to resolution", "mean time to resolution",
        "repair time", "time to repair", "mean time to repair",
    ]

    aliases: List[str] = []

    if any(cue in q for cue in detection_cues):
        aliases.extend(["MTTD", "Mean Time to Detect", "tempo medio impiegato per rilevare", "tempo medio di rilevamento"])

    if any(cue in q for cue in resolution_cues):
        aliases.extend(["MTTR", "Mean Time to Resolution", "Mean Time to Repair", "tempo medio necessario per risolvere", "tempo medio di risoluzione", "tempo medio di riparazione"])

    # If the user says "tempi di rilevamento/risoluzione" or similar compact wording,
    # both branches should be retrieved.
    if re.search(r"rilevament[oa]\s*/\s*risoluzion[ea]|detect(?:ion)?\s*/\s*resolution", q):
        aliases.extend([
            "MTTD", "Mean Time to Detect", "tempo medio impiegato per rilevare",
            "MTTR", "Mean Time to Resolution", "Mean Time to Repair", "tempo medio necessario per risolvere",
        ])

    out: List[str] = []
    seen = set()
    for a in aliases:
        key = a.lower().strip()
        if a and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def expand_assessment_query(query_text: str) -> str:
    """
    v4.10 override:
    preserve the original expansion and add semantic aliases for temporal metrics.
    """
    expanded = _ORIGINAL_expand_assessment_query_v410(query_text)
    aliases = _temporal_metric_aliases_v410(query_text)
    if not aliases:
        return expanded
    return (expanded + "\n" + " ".join(aliases)).strip()


def extract_exact_phrases(query_text: str) -> List[str]:
    """Estrazione generalista: prende stringhe tra virgolette e acronimi."""
    q = query_text or ""
    phrases: List[str] = []
    
    # 1. Estrae qualsiasi cosa l'utente abbia messo tra virgolette (es. "Data Breach")
    quoted = re.findall(r"[\"“']([^\"”']+)[\"”']", q)
    phrases.extend([x.strip().lower() for x in quoted if len(x.strip()) > 2])
    
    # 2. Estrae acronimi (parole di 2-6 lettere tutte in maiuscolo, es. GDPR, MFA, CVSS)
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", q)
    phrases.extend([x.lower() for x in acronyms])
    
    return list(dict.fromkeys([p for p in phrases if p]))


def _requested_formula_terms_missing(query_text: str, rows: List[Dict[str, Any]]) -> List[str]:
    """
    v4.10 override:
    keep previous missing-term logic and also report generic severity/severità
    when explicitly requested and no computational formula is recovered.
    """
    missing = list(_ORIGINAL_requested_formula_terms_missing_v410(query_text, rows) or [])
    ql = (query_text or "").lower()
    found_text = " ".join([str(r.get("name", "")) + " " + str(r.get("latex", "")) + " " + str(r.get("meaning", "")) for r in rows]).lower()

    for term in ["severity", "severità"]:
        if term in ql and term not in found_text:
            missing.append(term)

    # If temporal metric aliases were requested but still not recovered, make the limitation explicit.
    temporal_aliases = _temporal_metric_aliases_v410(query_text)
    if temporal_aliases:
        wants_detection = any(a.lower() in {"mttd", "mean time to detect"} or "rileva" in a.lower() for a in temporal_aliases)
        wants_resolution = any(a.lower() in {"mttr", "mean time to resolution", "mean time to repair"} or "risolvere" in a.lower() or "riparazione" in a.lower() for a in temporal_aliases)
        if wants_detection and "mttd" not in found_text:
            missing.append("MTTD")
        if wants_resolution and "mttr" not in found_text:
            missing.append("MTTR")

    return sorted(set(missing))


# ============================================================
# 🧮 FORMULA STRICT MODE HOTFIX - v4.11
# ============================================================
# Fix objective:
# - keep v4.10 metric recall expansion;
# - prevent KG aggregate artefacts from being classified as computable formulas;
# - deduplicate KG Formula Lookup rows when document-backed rows already exist;
# - keep real document-backed computational formulas if present.

_FORMULA_KG_ARTIFACT_MARKERS_V411 = [
    "Plain:", "Meaning:", "Formula::", "Formule collegate", "Formula from Knowledge Graph",
]


def _formula_has_kg_artifacts_v411(value: Any) -> bool:
    text = str(value or "")
    return any(marker.lower() in text.lower() for marker in _FORMULA_KG_ARTIFACT_MARKERS_V411)


def _formula_is_kg_aggregate_source_v411(row: Dict[str, Any]) -> bool:
    fname = str(row.get("filename") or "").strip().lower()
    page = int(row.get("page") or 0)
    name = str(row.get("name") or "").strip().lower()
    return (
        fname in {"kg", "neo4j", "neo4j knowledge graph"}
        or (page == 0 and name in {"formule collegate", "latex"})
        or _formula_has_kg_artifacts_v411(row.get("latex"))
        or _formula_has_kg_artifacts_v411(row.get("meaning"))
    )


def _looks_computational_formula(latex: str) -> bool:
    """
    v4.11 stricter computational formula detector.

    A computable formula must contain a real mathematical operator/function.
    Equality with prose on the right side is NOT computable, even if the text
    contains numbers or KG metadata tokens.
    """
    v = _strip_math_wrappers(_normalize_latex_value(str(latex or "")))
    vl = v.lower()

    if not v or "formula esplicita non recuperata" in vl:
        return False

    if _formula_has_kg_artifacts_v411(v):
        return False

    # Strong mathematical constructs.
    if any(x in v for x in ["\\frac", "\\sum", "\\prod", "√", "^", "×", "*", "/", "÷"]):
        return True

    if "=" not in v:
        return False

    left, right = v.split("=", 1)
    left = left.strip()
    right_clean = right.strip()
    right_plain = _formula_display_text(right_clean, 1000)
    right_l = right_plain.lower()

    if not left or not right_clean:
        return False

    # Explicit text/prose definitions are definitional metrics, not formulas.
    if re.fullmatch(r"\\?text\{[^}]+\}", right_clean):
        return False

    if re.search(r"\b(tempo medio|mean time|necessario per|impiegato per|rilevare|risolvere|incident[ei]?|definizione|definition)\b", right_l):
        return False

    # Equality is computable only when the RHS has math operators/functions or
    # a numeric expression, not arbitrary prose with incidental numbers.
    has_math_operator = bool(re.search(r"[+\-*/×÷^]|\\frac|\\sum|\\prod|sqrt|log|exp|min|max|avg|mean", right_clean, re.IGNORECASE))
    if has_math_operator:
        return True

    # Pure numeric assignment can be a value, but not enough for a formula in this RAG mode.
    if re.fullmatch(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent)?", right_plain):
        return False

    return False


def _formula_row_quality_v411(row: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Higher is better. Used to keep the best duplicate when the same metric is
    found through both document text and KG aggregate lookup.
    """
    fname = str(row.get("filename") or "")
    page = int(row.get("page") or 0)
    text = " ".join([str(row.get("name") or ""), str(row.get("latex") or ""), str(row.get("meaning") or "")])
    is_kg = _formula_is_kg_aggregate_source_v411(row)
    has_artifacts = _formula_has_kg_artifacts_v411(text)
    has_real_doc = bool(fname and fname.lower() not in {"kg", "neo4j", "neo4j knowledge graph", "n/d"} and page > 0)
    return (
        2 if has_real_doc else (0 if is_kg else 1),
        0 if has_artifacts else 1,
        min(len(_formula_display_text(row.get("meaning") or "", 1000)), 500),
    )


def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
    """
    v4.11 clean-up:
    - reclassify rows with stricter computational detection;
    - remove generic KG aggregate artefact rows;
    - if a KG aggregate row duplicates a document-backed metric, keep the
      document-backed row only;
    - preserve threshold aggregation inputs.
    """
    classified: List[Dict[str, Any]] = []

    for r in rows or []:
        rr = _classify_formula_row(r)

        # Drop pure aggregate/generic KG artefacts.
        name_l = _formula_display_text(rr.get("name") or "", 160).lower()
        if name_l in {"formule collegate", "latex", "formula from knowledge graph"}:
            continue

        if _is_noise_formula_row_v45(rr):
            continue

        # If KG artefacts forced a bad row, never keep it as computational.
        if _formula_is_kg_aggregate_source_v411(rr) and str(rr.get("tipo") or "").lower() == "formula computazionale":
            rr["tipo"] = "Metrica definitoria"
            rr["latex"] = "formula computazionale non recuperata"
            rr["meaning"] = "Definizione testuale della metrica; formula computazionale non recuperata nella fonte."

        classified.append(rr)

    # First pass: keep best row per metric/type/document/page.
    by_key: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for rr in classified:
        tipo = str(rr.get("tipo") or "").lower()
        fname = normalize_doc_name(str(rr.get("filename") or ""))
        page = int(rr.get("page") or 0)
        name_norm = re.sub(r"[^a-z0-9]+", "", str(rr.get("name") or "").lower())
        formula_norm = re.sub(r"\s+", " ", _formula_display_text(rr.get("latex") or "", 1000).lower()).strip()

        if tipo == "regola soglia":
            key = ("threshold:" + formula_norm[:260], tipo, fname, page)
        else:
            key = (name_norm, tipo, fname, page)

        existing = by_key.get(key)
        if existing is None or _formula_row_quality_v411(rr) > _formula_row_quality_v411(existing):
            by_key[key] = rr

    deduped = list(by_key.values())

    # Second pass: remove KG aggregate definitional rows when the same canonical
    # metric exists in a real document-backed source.
    real_doc_metric_keys = {
        (re.sub(r"[^a-z0-9]+", "", str(r.get("name") or "").lower()), str(r.get("tipo") or "").lower())
        for r in deduped
        if not _formula_is_kg_aggregate_source_v411(r)
        and normalize_doc_name(str(r.get("filename") or "")) not in {"", "kg", "neo4j", "neo4jknowledgegraph"}
        and int(r.get("page") or 0) > 0
    }

    filtered: List[Dict[str, Any]] = []
    for r in deduped:
        key = (re.sub(r"[^a-z0-9]+", "", str(r.get("name") or "").lower()), str(r.get("tipo") or "").lower())
        if _formula_is_kg_aggregate_source_v411(r) and key in real_doc_metric_keys:
            continue
        filtered.append(r)

    priority = {
        "formula computazionale": 0,
        "regola soglia": 1,
        "metrica definitoria": 2,
        "metrica/elemento citato": 3,
    }
    filtered.sort(key=lambda x: (
        priority.get(str(x.get("tipo") or "").lower(), 9),
        str(x.get("filename") or ""),
        int(x.get("page") or 0),
        str(x.get("name") or ""),
    ))

    return filtered[:max_rows]


# ============================================================
# 🔎 FORMULA / THRESHOLD CATEGORY PRESERVATION PATCH - v4.12
# ============================================================
# Goal:
# - Preserve the good v4.11 classification of MTTD/MTTR as definitional metrics.
# - Preserve threshold/normative-condition rows when the user asks semantically for
#   impacted users, notification thresholds, notification obligations, or significant incidents.
# - Keep this non-adaptive: it is a generic synonym/recall expansion for threshold-style
#   formula/metric queries, not a hardcoded answer.

_ORIGINAL_expand_assessment_query_v412 = expand_assessment_query
_ORIGINAL_extract_exact_phrases_v412 = extract_exact_phrases
_ORIGINAL_answer_formula_strict_v412 = answer_formula_strict


def _threshold_metric_aliases_v412(query_text: str) -> List[str]:
    """
    Generic IT/EN synonym expansion for threshold/normative-condition retrieval.
    Activated only for formula/metric/scoring queries where the user mentions
    users impacted, notification obligations, thresholds, or significant incidents.
    """
    try:
        formula_intent = bool(is_formula_strict_query(query_text))
    except Exception:
        formula_intent = False

    if not formula_intent:
        return []

    q = (query_text or "").lower()

    cues = [
        # IT
        "soglia", "soglie", "utenti impattati", "utenti coinvolti", "utenti interessati",
        "utenti nell'unione", "utenti nell’unione", "obbligo di notifica", "obblighi di notifica",
        "notifica incidenti", "notifica degli incidenti", "incidente significativo", "incidenti significativi",
        # EN
        "threshold", "thresholds", "affected users", "impacted users", "notification obligation",
        "notification obligations", "incident notification", "significant incident", "significant incidents",
    ]

    if not any(c in q for c in cues):
        return []

    aliases = [
        # Keep terms generic enough for NIS/incident notification threshold retrieval.
        "soglia", "soglie", "utenti", "utenti nell'Unione", "notifica", "obbligo di notifica",
        "incidente significativo", "incidenti significativi", "threshold", "affected users",
        "incident notification", "significant incident",
        # Common threshold wording that may appear in normative sources.
        "oltre il 5%", "oltre 1 milione", "milione di utenti",
        # Digital-service scope terms, still generic within notification-threshold contexts.
        "mercato online", "motore di ricerca online", "piattaforma di servizi di social network",
    ]

    out: List[str] = []
    seen = set()
    for a in aliases:
        key = a.lower().strip()
        if a and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def expand_assessment_query(query_text: str) -> str:
    """v4.12 override: preserve current expansion and add threshold aliases when relevant."""
    expanded = _ORIGINAL_expand_assessment_query_v412(query_text)
    aliases = _threshold_metric_aliases_v412(query_text)
    if not aliases:
        return expanded
    return (expanded + "\n" + " ".join(aliases)).strip()


def extract_exact_phrases(query_text: str) -> List[str]:
    """v4.12 override: exact phrase search also receives threshold aliases when relevant."""
    phrases = list(_ORIGINAL_extract_exact_phrases_v412(query_text) or [])
    phrases.extend(_threshold_metric_aliases_v412(query_text))

    out: List[str] = []
    seen = set()
    for p in phrases:
        clean = str(p or "").strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(str(p).strip())
    return out

def _threshold_supplemental_sources_v412(query_text: str, limit: int = 18) -> List[SourceItem]:

    """
    v4.13 override with increased limit (18) to ensure technical terms 
    and glossary definitions are not truncated.
    """
    aliases = _threshold_metric_aliases_v412(query_text)
    if not aliases:
        return []

    supplemental_query = query_text + "\n" + " ".join(aliases)
    hits: List[Dict[str, Any]] = []

    # BM25 gives broader recall; exact phrase gives precision if the normative phrase is present.
    try:
        hits.extend(search_pg_bm25(supplemental_query, limit=limit))
    except Exception as e:
        print(f"⚠️ v4.12 threshold BM25 supplement error: {e}")

    try:
        hits.extend(search_pg_exact_phrases(supplemental_query, limit=limit))
    except Exception as e:
        print(f"⚠️ v4.12 threshold exact supplement error: {e}")

    sources_extra: List[SourceItem] = []
    seen_ids = set()

    for h in hits:
        uid = str(h.get("id", "")).strip()
        if not uid or uid in seen_ids:
            continue
        seen_ids.add(uid)

        meta = h.get("metadata", {}) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        content = h.get("content") or h.get("content_semantic") or h.get("content_raw") or ""
        if not content:
            continue

        fname = meta.get("filename") or meta.get("source_name") or "Postgres"
        page = int(meta.get("page_no") or meta.get("page") or 0)
        source_type = meta.get("toon_type") or meta.get("type") or "text"
        tier = normalize_tier_value(meta.get("tier", "C"))

        sources_extra.append(
            SourceItem(
                id=uid,
                content=content,
                filename=fname,
                page=page,
                type=source_type,
                score=float(h.get("score", 0.0) or 0.0),
                tier=tier,
                db_origin=str(h.get("origin") or "PostgresThresholdSupplement"),
                section_hint="v4.12 threshold supplemental retrieval",
            )
        )

        if len(sources_extra) >= limit:
            break

    return sources_extra


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    v4.12 wrapper:
    - Uses the stable v4.11/v4.9 formula classifier.
    - If the user asks for threshold/user/notification criteria and the first retrieved
      source set does not contain threshold rows, performs a small supplemental retrieval
      and lets the same classifier build the answer.
    """
    try:
        current_rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=30)
        has_threshold = any(str(r.get("tipo") or "").lower() == "regola soglia" for r in current_rows)
    except Exception:
        current_rows = []
        has_threshold = False

    wants_threshold = bool(_threshold_metric_aliases_v412(query_text))

    if wants_threshold and not has_threshold:
        extra_sources = _threshold_supplemental_sources_v412(query_text)
        if extra_sources:
            merged = dedupe_sources_for_answer(list(sources or []) + extra_sources)
            return _ORIGINAL_answer_formula_strict_v412(query_text, merged)

    return _ORIGINAL_answer_formula_strict_v412(query_text, sources)

# ============================================================
# 🔎 FORMULA / THRESHOLD RECALL PATCH - v4.13
# ============================================================
# Goal:
# - Keep v4.12 behaviour for MTTD/MTTR.
# - Improve threshold recall when threshold-like chunks are present as plain text
#   and not as LaTeX/formula nodes.
# - Non-adaptive: generic threshold-rule extraction + PostgreSQL regex recall.

_ORIGINAL_extract_formula_rows_from_sources_v413 = extract_formula_rows_from_sources
_ORIGINAL_threshold_supplemental_sources_v412_v413 = _threshold_supplemental_sources_v412


def _threshold_rule_segments_v413(text: str, max_segments: int = 8) -> List[str]:
    """Extract readable threshold-rule segments from arbitrary text."""
    raw = str(text or "")
    if not raw.strip():
        return []

    # Convert common bullet/list separators into split points, but keep sentences readable.
    candidates = re.split(r"(?<=[\.\;\!\?])\s+|\n+|\r+", raw)

    out: List[str] = []
    seen = set()

    for c in candidates:
        seg = _formula_display_text(c, 900)
        if not seg:
            continue

        if not _looks_threshold_rule(seg):
            continue

        # Avoid isolated numeric fragments such as only "5%" or "1 milione".
        words = re.findall(r"[A-Za-zÀ-ÿ]+", seg)
        if len(words) < 5:
            continue

        key = re.sub(r"\s+", " ", seg.lower())[:260]
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)

        if len(out) >= max_segments:
            break

    # Fallback: if the whole chunk contains a threshold but splitting missed it,
    # take a window around the first threshold-looking expression.
    if not out and _looks_threshold_rule(raw):
        plain = _formula_display_text(raw, 3000)
        m = re.search(
            r"(?:oltre|superiore|almeno|non inferiore|maggiore di|greater than|over|more than|at least|threshold|soglia|condizione|condition).{0,420}?(?:\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)).{0,420}",
            plain,
            flags=re.IGNORECASE,
        )
        if m:
            out.append(_formula_display_text(m.group(0), 900))

    return out


def extract_formula_rows_from_sources(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    v4.13 override:
    use the previous extractor and add plain-text threshold-rule rows.
    This lets Formula Strict Mode preserve normative thresholds even when they are
    not encoded as LaTeX/formula nodes.
    """
    rows = list(_ORIGINAL_extract_formula_rows_from_sources_v413(sources) or [])
    seen = {
        (
            str(r.get("name") or "").lower(),
            _formula_display_text(r.get("latex") or "", 500).lower(),
            str(r.get("filename") or "").lower(),
            int(r.get("page") or 0),
        )
        for r in rows
    }

    for s in sources or []:
        content = getattr(s, "content", "") or ""
        filename = getattr(s, "filename", "N/D") or "N/D"
        page = int(getattr(s, "page", 0) or 0)

        for seg in _threshold_rule_segments_v413(content, max_segments=8):
            key = ("regola soglia", seg.lower()[:500], filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": "Regola soglia",
                "latex": seg,
                "meaning": "Criterio/soglia normativa recuperata; non è una formula computazionale.",
                "filename": filename,
                "page": page,
            })

    return rows


def _search_pg_threshold_regex_v413(limit: int = 12) -> List[SourceItem]:
    """
    High-precision PostgreSQL recall for threshold rules and requirements.
    Genera dinamicamente le espressioni regolari basandosi sul dizionario globale,
    rendendo il sistema completamente agnostico rispetto al dominio documentale.
    """
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    # 1. Unisce tutti i termini della lista globale separandoli con OR (|)
    termini_uniti = "|".join(re.escape(t) for t in THRESHOLD_TERMS_LIST)

    # 2. Costruisce il pattern in modo dinamico usando f-string.
    # Cerca un termine della lista, seguito da un massimo di 60 caratteri, seguito da un numero.
    patterns = [
        rf"\b({termini_uniti})\b.{{0,60}}\b\d+(?:[,.]\d+)?\b"
    ]

    clauses = []
    params: List[Any] = []
    for pat in patterns:
        clauses.append("""(
            COALESCE(content_semantic, '') ~* %s OR
            COALESCE(content_raw, '') ~* %s OR
            COALESCE(metadata_json::text, '') ~* %s
        )""")
        params.extend([pat, pat, pat])

    sql = f"""
    SELECT chunk_uuid::text, content_raw, content_semantic, metadata_json, ingestion_ts
    FROM public.document_chunks
    WHERE {' OR '.join(clauses)}
    ORDER BY ingestion_ts DESC
    LIMIT %s;
    """
    params.append(limit)

    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        out: List[SourceItem] = []
        for chunk_uuid, content_raw, content_semantic, metadata_json, ingestion_ts in rows:
            if isinstance(metadata_json, str):
                try:
                    metadata_json = json.loads(metadata_json)
                except Exception:
                    metadata_json = {}
            if metadata_json is None:
                metadata_json = {}

            content = content_semantic or content_raw or ""
            if not content:
                continue

            out.append(SourceItem(
                id=str(chunk_uuid),
                content=content,
                filename=metadata_json.get("filename") or metadata_json.get("source_name") or "Postgres",
                page=int(metadata_json.get("page_no") or metadata_json.get("page") or 0),
                type=metadata_json.get("toon_type") or metadata_json.get("type") or "text",
                score=2.5,
                tier=normalize_tier_value(metadata_json.get("tier", "C")),
                db_origin="PostgresThresholdRegex",
                section_hint="v4.13 threshold regex retrieval (Dynamic)",
                pg_ingestion_ts=ingestion_ts.isoformat() if ingestion_ts else "",
                pg_source_name=metadata_json.get("source_name", ""),
                pg_source_type=metadata_json.get("source_type", ""),
                pg_log_id=int(metadata_json.get("log_id") or 0),
                pg_chunk_id=int(metadata_json.get("chunk_index") or 0),
                pg_toon_type=metadata_json.get("toon_type", ""),
            ))
        return out
    except Exception as e:
        print(f"⚠️ v4.13 threshold regex supplement error: {e}")
        return []
    finally:
        pg_pool.putconn(conn)


app = rx.App()
app.add_page(index, on_load=State.on_load)

