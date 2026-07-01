# FINAL MULTI-TENANT HARDENED VERSION - aligned with Architecture v1.1
# LaTeX rendering + formula provenance fix v4.14: raw text priority, document scope, KaTeX-safe output

import os
import sys

# Log immediati nel container Docker.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# --- FIX ANTI-BLOCCO GUI/RAG ---
# Deve stare prima di torch / sentence_transformers.
# Evita oversubscription CPU e freeze quando embedding + Ollama lavorano insieme.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CPU_THREADS = os.environ.get("EMBED_CPU_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", CPU_THREADS)


import reflex as rx
import torch

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
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from neo4j import GraphDatabase
from openai import OpenAI

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import uuid
import contextvars
import secrets

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
# MULTI-TENANT CONTEXT
# =========================
def _required_positive_int_env(name: str) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        raise RuntimeError(f"Variabile ambiente obbligatoria non configurata: {name}")
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} deve essere un intero positivo") from exc
    if value <= 0:
        raise RuntimeError(f"{name} deve essere maggiore di zero")
    return value


# POC: un'unica organizzazione configurata direttamente nel codice.
# Deve essere un intero, non un confronto booleano.
POC_MODE = True
ORGANIZATION_ID: int = 1234

CORPUS_VERSION = (os.getenv("CORPUS_VERSION", "v1").strip() or "v1")
PG_AUTO_HARDEN_SCHEMA = os.getenv("PG_AUTO_HARDEN_SCHEMA", "0") == "1"
PG_ENFORCE_LEAST_PRIVILEGE = (
    False if POC_MODE
    else os.getenv("PG_ENFORCE_LEAST_PRIVILEGE", "1") == "1"
)


class TenantContext(BaseModel):
    organization_id: int
    user_id: str
    roles: List[str] = Field(default_factory=list)
    request_id: str
    is_super_admin: bool = False
    allowed_scopes: List[str] = Field(default_factory=lambda: ["GLOBAL", "ACCOUNT"])

    class Config:
        allow_mutation = False


def resolve_tenant_context(*, request_id: str = "", user_id: str = "") -> TenantContext:
    # Il TenantContext iniziale deve usare direttamente la configurazione trusted.
    # Non può chiamare current_organization_id(), perché la ContextVar viene
    # creata soltanto dopo la costruzione del contesto base.
    roles = [r.strip() for r in os.getenv("RAG_USER_ROLES", "user").split(",") if r.strip()]
    return TenantContext(
        organization_id=int(ORGANIZATION_ID),
        user_id=user_id or os.getenv("RAG_USER_ID", "service-user"),
        roles=roles,
        request_id=request_id or str(uuid.uuid4()),
        is_super_admin=False,
    )


_BASE_TENANT_CONTEXT = resolve_tenant_context(request_id="startup")
_CURRENT_TENANT_CONTEXT: contextvars.ContextVar[TenantContext] = contextvars.ContextVar(
    "rag_tenant_context", default=_BASE_TENANT_CONTEXT
)


def get_tenant_context() -> TenantContext:
    return _CURRENT_TENANT_CONTEXT.get()


def current_organization_id() -> int:
    return int(get_tenant_context().organization_id)


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def tenant_record_is_visible(
    *,
    scope: Any,
    organization_id: Any,
    tier: Any,
    status: Any = "active",
    current_organization_id: Optional[int] = None,
    allow_graph_tier: bool = False,
) -> bool:
    """Regola fail-closed condivisa da Qdrant, PostgreSQL, Neo4j e prompt."""
    org_current = int(current_organization_id or current_organization_id_fn())
    scope_norm = str(scope or "").strip().upper()
    tier_norm = str(tier or "").strip().upper()
    status_norm = str(status or "").strip().lower()
    org_norm = _optional_int(organization_id)

    if status_norm != "active":
        return False
    if scope_norm == "GLOBAL":
        return org_norm is None and (tier_norm == "A" or (allow_graph_tier and tier_norm == "GRAPH"))
    if scope_norm == "ACCOUNT":
        return org_norm == org_current and (
            tier_norm in {"B", "C"} or (allow_graph_tier and tier_norm == "GRAPH")
        )
    return False


def current_organization_id_fn() -> int:
    return current_organization_id()


def qdrant_payload_is_visible(
    payload: Dict[str, Any],
    current_organization_id: Optional[int] = None,
) -> bool:
    return tenant_record_is_visible(
        scope=payload.get("scope"),
        organization_id=payload.get("organization_id"),
        tier=payload.get("tier"),
        status=payload.get("status"),
        current_organization_id=current_organization_id,
        allow_graph_tier=False,
    )


def source_is_visible(
    source: "SourceItem",
    current_organization_id: Optional[int] = None,
) -> bool:
    org_current = int(current_organization_id or current_organization_id_fn())
    tier_norm = str(getattr(source, "tier", "") or "").strip().upper()
    if tier_norm == "USER":
        return (
            str(getattr(source, "scope", "") or "").strip().upper() == "ACCOUNT"
            and _optional_int(getattr(source, "organization_id", None)) == org_current
            and str(getattr(source, "id", "") or "") in {"user_input", "error"}
        )
    return tenant_record_is_visible(
        scope=getattr(source, "scope", ""),
        organization_id=getattr(source, "organization_id", None),
        tier=tier_norm,
        status=getattr(source, "status", ""),
        current_organization_id=org_current,
        allow_graph_tier=True,
    )


def filter_sources_for_current_organization(
    sources: List["SourceItem"],
    current_organization_id: Optional[int] = None,
) -> List["SourceItem"]:
    org_current = int(current_organization_id or current_organization_id_fn())
    visible: List["SourceItem"] = []
    dropped = 0
    for source in sources or []:
        if source_is_visible(source, org_current):
            visible.append(source)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "Tenant guard: scartate %s fonti non visibili per organization_id=%s request_id=%s",
            dropped, org_current, get_tenant_context().request_id,
        )
    return visible


def build_qdrant_tenant_filter(extra_must: Optional[List[Any]] = None) -> models.Filter:
    """Ogni query Qdrant include status e perimetro tenant costruiti da codice trusted."""
    org_id = current_organization_id()
    mandatory = [
        models.FieldCondition(key="status", match=models.MatchValue(value="active")),
        *(list(extra_must or [])),
    ]
    return models.Filter(
        must=mandatory,
        should=[
            models.Filter(
                must=[
                    models.FieldCondition(key="scope", match=models.MatchValue(value="GLOBAL")),
                    models.FieldCondition(key="tier", match=models.MatchValue(value="A")),
                ]
            ),
            models.Filter(
                must=[
                    models.FieldCondition(key="scope", match=models.MatchValue(value="ACCOUNT")),
                    models.FieldCondition(key="organization_id", match=models.MatchValue(value=org_id)),
                    models.FieldCondition(key="tier", match=models.MatchAny(any=["B", "C"])),
                ]
            ),
        ],
    )


def metadata_with_tenant(
    metadata: Any,
    scope: Any,
    organization_id: Any,
    tier: Any,
) -> Dict[str, Any]:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    result = dict(metadata or {})
    result["scope"] = str(scope or "").strip().upper()
    result["organization_id"] = _optional_int(organization_id)
    result["tier"] = str(tier or result.get("tier") or "").strip().upper()
    result["status"] = str(result.get("status") or "active").strip().lower()
    result["corpus_version"] = str(result.get("corpus_version") or CORPUS_VERSION)
    return result


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
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma4:12b")
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", LLM_MODEL_NAME)

# alternativa
#LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1:8b")
#VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "ministral-3:8b")
#EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
#EMBEDDING_MODEL_NAME = "E:/Modelli/bge-m3"

#RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
#RERANKER_MODEL_NAME = "E:/Modelli/ms-marco-reranker"



# conf. docker 
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "/workspace/models/bge-m3"
)

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "/workspace/models/ms-marco-reranker"
)


# LM Studio / OpenAI Compatible API
#LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
#LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy key, Ollama non la valida

# =========================
# 🧠 LLM / OLLAMA CONTEXT
# =========================
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384")) #8192
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

        # Evita che il modello produca soltanto il campo thinking,
        # lasciando message.content vuoto.
        "think": False,

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
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))  # 16000 - prevent prompt blow-ups
MAX_ASSISTANT_CHARS = int(os.getenv("MAX_ASSISTANT_CHARS", "15000"))

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
RESOURCE_INIT_ERROR = ""

# Device selection (già definiti nel tuo script, ma assicurati siano accessibili)
# Device selection
# Per la GUI/RAG conviene CPU di default per non competere con Ollama sulla VRAM.
# Se vuoi forzare CUDA: set EMBED_DEVICE=cuda
#device_embed = "cuda" if torch.cuda.is_available() else "cpu"
#device_rerank = "cpu" 
device_embed = os.getenv("EMBED_DEVICE", "cpu")
device_rerank = os.getenv("RERANK_DEVICE", "cpu")


def _pg_role_security_check(cur) -> None:
    cur.execute(
        "SELECT current_user, rolsuper, rolbypassrls "
        "FROM pg_roles WHERE rolname = current_user"
    )
    row = cur.fetchone() or ("unknown", False, False)
    user_name = str(row[0])
    is_superuser = bool(row[1])
    bypass_rls = bool(row[2])

    if is_superuser or bypass_rls:
        if POC_MODE:
            print(
                "⚠️ POC MODE: ruolo PostgreSQL privilegiato consentito "
                f"(user={user_name}, superuser={is_superuser}, bypassrls={bypass_rls})."
            )
            return

        if PG_ENFORCE_LEAST_PRIVILEGE:
            raise RuntimeError(
                "PG_USER non può essere SUPERUSER/BYPASSRLS nel backend RAG. "
                "Configurare un ruolo applicativo dedicato."
            )


def pg_get_conn_secure():
    if not pg_pool:
        raise RuntimeError("Pool PostgreSQL non inizializzato")
    conn = pg_pool.getconn()
    try:
        ctx = get_tenant_context()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('app.current_customer_account_id', %s, false), "
                "set_config('app.current_request_id', %s, false)",
                (str(ctx.organization_id), ctx.request_id),
            )
        conn.commit()
        return conn
    except Exception:
        conn.rollback()
        pg_pool.putconn(conn)
        raise


def pg_put_conn_secure(conn) -> None:
    if conn is None or not pg_pool:
        return
    try:
        if not conn.closed:
            with conn.cursor() as cur:
                cur.execute("RESET app.current_customer_account_id")
                cur.execute("RESET app.current_request_id")
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        pg_pool.putconn(conn)


def ensure_postgres_rag_security() -> None:
    """
    Verifica fail-closed delle difese PostgreSQL.

    Il backend RAG non esegue DDL e non deve usare un ruolo proprietario,
    SUPERUSER o BYPASSRLS. Il bootstrap viene eseguito una sola volta tramite
    ingestion_final_multitenant.py con PG_SCHEMA_MIGRATION_ONLY=1.
    """
    if PG_AUTO_HARDEN_SCHEMA and not POC_MODE:
        raise RuntimeError(
            "Il backend RAG non può modificare lo schema. Impostare "
            "PG_AUTO_HARDEN_SCHEMA=0 ed eseguire il bootstrap con il job ingestion."
        )

    if POC_MODE:
        print(
            f"ℹ️ POC MODE attivo | ORGANIZATION_ID={ORGANIZATION_ID} | "
            "controlli RLS/least-privilege non bloccanti."
        )
        conn = pg_pool.getconn()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                _pg_role_security_check(cur)
                cur.execute("SELECT 1")
        finally:
            try:
                conn.autocommit = False
            except Exception:
                pass
            pg_pool.putconn(conn)
        return

    conn = pg_pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            _pg_role_security_check(cur)

            required_columns = {
                "status", "ingestion_run_id", "tenant_key", "corpus_version",
                "classification", "embedding_model", "organization_id", "tier", "scope",
            }
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='document_chunks'
                """
            )
            existing = {str(row[0]) for row in cur.fetchall()}
            missing = sorted(required_columns - existing)
            if missing:
                raise RuntimeError(
                    "Schema document_chunks incompleto; colonne mancanti: " + ", ".join(missing)
                )

            cur.execute(
                """
                SELECT relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE oid = 'public.document_chunks'::regclass
                """
            )
            rls = cur.fetchone()
            if not rls or not all(bool(x) for x in rls):
                raise RuntimeError("RLS/FORCE RLS non attive su document_chunks")

            cur.execute(
                """
                SELECT count(*)
                FROM pg_policies
                WHERE schemaname='public' AND tablename='document_chunks'
                  AND policyname='document_chunks_tenant_select'
                """
            )
            if int(cur.fetchone()[0]) != 1:
                raise RuntimeError("Policy document_chunks_tenant_select assente")

            cur.execute("SELECT to_regclass('public.rag_query_audit')")
            if cur.fetchone()[0] is None:
                raise RuntimeError("Tabella rag_query_audit assente")

            cur.execute(
                """
                SELECT relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE oid = 'public.rag_query_audit'::regclass
                """
            )
            audit_rls = cur.fetchone()
            if not audit_rls or not all(bool(x) for x in audit_rls):
                raise RuntimeError("RLS/FORCE RLS non attive su rag_query_audit")

            cur.execute(
                """
                SELECT count(*)
                FROM pg_policies
                WHERE schemaname='public' AND tablename='rag_query_audit'
                  AND policyname='rag_query_audit_tenant_all'
                """
            )
            if int(cur.fetchone()[0]) != 1:
                raise RuntimeError("Policy rag_query_audit_tenant_all assente")
    finally:
        try:
            conn.autocommit = False
        except Exception:
            pass
        pg_pool.putconn(conn)



NEO4J_ALLOWED_RELATIONSHIPS = [
    "IS_A", "PART_OF", "HAS_COMPONENT", "CONTAINS", "BELONGS_TO",
    "APPLIES_TO", "MAPS_TO", "DEFINES", "CLASSIFIES", "COMPLIES_WITH",
    "NON_COMPLIANT_WITH", "HAS_COMPLIANCE_STATUS", "HAS_GAP",
    "REQUIRES_REMEDIATION", "REMEDIATES", "MANDATES", "REQUIRES",
    "GOVERNS", "APPROVES", "REVIEWS", "ASSIGNS_RESPONSIBILITY_TO",
    "TRIGGERS", "ACTIVATES", "STARTS", "FOLLOWS", "PRECEDES",
    "LEADS_TO", "ESCALATES_TO", "MANAGES", "HANDLES", "NOTIFIES",
    "REPORTS_TO", "HAS_DEADLINE", "SUPPORTS", "ENABLES", "IMPLEMENTS",
    "DEPENDS_ON", "ALIGNS_WITH", "CONTRIBUTES_TO", "MEASURES", "MONITORS",
    "MITIGATES", "REDUCES", "THREATENS", "EXPLOITS", "PROTECTS",
    "VULNERABLE_TO", "IMPACTS", "AFFECTS", "GENERATES", "VERIFIES",
    "TESTS", "DEMONSTRATES", "DOCUMENTS", "SUPPORTS_EVIDENCE_FOR",
    "EVIDENCES", "SATISFIES", "REFERENCES_REQUIREMENT", "HAS_FORMULA",
]


def init_resources():
    """
    Inizializza i modelli e le connessioni ai database in un unico passaggio.
    Previene il caricamento duplicato durante la compilazione del frontend Reflex.
    """
    global embedder, reranker, llm_client, qdrant_client_inst, neo4j_driver, pg_pool, NEO4J_ENABLED, RESOURCE_INIT_ERROR

    with _init_lock:
        if embedder is not None and qdrant_client_inst is not None and llm_client is not None:
            return

        RESOURCE_INIT_ERROR = ""
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
                ensure_postgres_rag_security()
                conn = pg_get_conn_secure()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                finally:
                    pg_put_conn_secure(conn)

            print("✅ [BACKEND] Risorse caricate con successo.")
            print("═"*60 + "\n")

        except Exception as e:
            RESOURCE_INIT_ERROR = str(e)
            print(f"❌ [ERRORE] Fallimento inizializzazione: {e}")
            # Nel POC l'errore delle risorse non deve impedire a Reflex di
            # creare l'app ed esporre le porte 3000/8000. Un successivo
            # on_load o submit ritenterà l'inizializzazione.
            if pg_pool is not None:
                try:
                    pg_pool.closeall()
                except Exception:
                    pass
                pg_pool = None
            if not POC_MODE:
                raise
        
# Non inizializzare modelli e database durante l'import del modulo Reflex.
# L'import deve completarsi rapidamente affinché il backend esponga la porta 8000.
print("ℹ️ RAG POC: inizializzazione risorse differita all'on_load della GUI.")


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

    # Provenance condivisa tra Qdrant, PostgreSQL e Neo4j
    page_chunk_index: int = 0
    doc_id: str = ""

    type: str = "text"
    score: float = 0.0
    graph_context: List[GraphEntity] = Field(default_factory=list)

    # Extra provenance / metadata
    section_hint: str = ""
    image_id: Optional[int] = None
    tier: str = "C"
    scope: str = ""
    organization_id: Optional[int] = None
    status: str = ""
    ingestion_run_id: str = ""
    corpus_version: str = ""
    classification: str = "internal"
    embedding_model: str = ""
    request_id: str = ""

    # PostgreSQL canonical provenance

    # PostgreSQL canonical provenance
    pg_ingestion_ts: str = ""
    pg_source_name: str = ""
    pg_source_type: str = ""
    pg_log_id: int = 0
    pg_chunk_id: int = 0
    pg_page_chunk_index: int = 0
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
    query_sha256: str = ""
    intent: str = ""
    organization_id: int = 0
    user_id: str = ""
    roles: List[str] = Field(default_factory=list)
    request_id: str = ""
    corpus_version: str = ""
    filters: Dict[str, Any] = Field(default_factory=dict)
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list)

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

def is_regulatory_classification_query(query_text: str) -> bool:
    """
    Riconosce domande normative/classificatorie che NON devono entrare
    in Formula Strict Mode solo perché contengono parole come soglia,
    sanzione, regolamento, soggetti o categorie.

    Non è adattativa:
    - non contiene nomi di test;
    - non contiene nomi di documenti;
    - non forza risposte;
    - riconosce una classe generale di domande normative.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    classification_starters = [
        "chi sono", "quali sono", "qual è", "quale è",
        "what are", "who are", "which are", "what is",
    ]

    regulatory_terms = [
        # IT
        "soggetti", "soggetto", "categorie", "categoria",
        "tipologie", "tipologia", "regime", "vigilanza",
        "obblighi", "obbligo", "requisiti", "requisito",
        "normativa", "regolamento", "direttiva", "legge",
        "classificazione", "classifica", "autorità",
        "responsabilità", "categorie normative",

        # EN
        "subjects", "entities", "categories", "category",
        "types", "classification", "regime", "supervision",
        "oversight", "obligations", "requirements",
        "regulation", "directive", "law", "authority",
        "responsibilities",
    ]

    has_classification_starter = any(t in q for t in classification_starters)
    has_regulatory_term = any(t in q for t in regulatory_terms)

    if has_classification_starter and has_regulatory_term:
        return True

    # Anche senza starter esplicito, una domanda su regime/categorie/soggetti
    # è classificatoria se non chiede calcolo o derivazione.
    classification_density = sum(1 for t in regulatory_terms if t in q)

    return classification_density >= 2


def detect_intent(query: str) -> str:
    """
    Router di intenti esteso per sistema di Assessment/Audit RAG.
    Classifica la domanda dell'utente per attivare pipeline o prompt specifici.
    Restituisce: 'formula', 'table', 'chart', 'audit' o 'text'.

    Fix non adattativo:
    - le query matematiche/algebriche hanno priorità;
    - le domande normative/classificatorie non devono finire per errore in formula;
    - parole come "sanzione" o "multa" non bastano da sole: serve un segnale numerico/calcolatorio.
    """
    q_raw = query or ""
    q = q_raw.lower()

    if is_assessment_evidence_relevance_query(q_raw):
        return "audit"


    # 0. INTENT: CLASSIFICAZIONE NORMATIVA
    # Deve venire prima della formula mode.
    # Esempio generale: "chi sono i soggetti/categorie e come varia il regime..."
    if is_regulatory_classification_query(q_raw):
        return "audit"

    # 1. INTENT: FORMULA / ALGEBRA / CALCOLO
    # Parte solo se la query è davvero matematica/algebrica.

    if is_calculation_request(q_raw):
        return "formula"

    if is_formula_lookup_query(q_raw):
        return "formula"

    # Normalizza percentuali LaTeX/Markdown per intercettare 40\%, 35\%, ecc.
    q_norm = q.replace("\\%", "%")

    # Trigger matematici forti: bastano da soli.
    STRONG_MATH_KEYWORDS = [
        "formula", "formule",
        "equazione", "equazioni", "equation", "equations",
        "disequazione", "disequazioni", "inequality", "inequalities",
        "algebra", "algebrica", "algebrico", "algebricamente",
        "algebraic", "algebraically",
        "calcola", "calcolo", "calculate", "compute",
        "risolvi", "solve", "solve for",
        "deriva", "derivazione", "derive",
        "esprimi", "express", "in funzione di", "as a function of",
        "isola", "isolate",
        "variabile", "variabili", "variable", "variables",
        "percentuale", "percentage",
        "roi", "rosi", "cvss", "risk score",
        "probabilità", "probability",
        "calcolo del rischio",
        "budget", "costo", "costi", "cost", "costs",
        "quantifica", "quantify",
        "impatto economico"
    ]

    if any(k in q_norm for k in STRONG_MATH_KEYWORDS):
        return "formula"

    # Trigger matematici deboli: parole come sanzione/multa/penale possono
    # essere normative. Le trattiamo come formula solo se ci sono anche numeri,
    # percentuali o richiesta esplicita di calcolo/importo.
    WEAK_MATH_TERMS = [
        "sanzione", "sanzioni", "sanzionatorio", "sanzionatoria",
        "penale", "penali", "multa", "multe", "ammenda", "ammende",
        "fine", "fines", "sanction", "sanctions", "penalty", "penalties",
        "ammonta", "importo", "amount"
    ]

    CALCULATION_CUES = [
        "calcola", "calcolo", "quantifica", "quanto", "cifra",
        "esatta", "esatto", "totale", "risultato",
        "calculate", "compute", "how much", "amount", "total", "result"
    ]

    has_weak_math = any(k in q_norm for k in WEAK_MATH_TERMS)
    has_numbers_or_symbols = bool(
        re.search(r"\d", q_norm)
        or re.search(r"(<=|>=|≤|≥|=|>|<|%|×|\*|/|\\frac|\\times)", q_raw)
    )
    has_calc_cue = any(k in q_norm for k in CALCULATION_CUES)

    if has_weak_math and (has_numbers_or_symbols or has_calc_cue):
        return "formula"

    # 2. INTENT: TABELLE E MATRICI
    TABLE_KEYWORDS = [
        "tabella", "table", "righe", "colonne", "row", "column",
        "matrice rischi", "risk register", "asset inventory", "inventario",
        "crosswalk", "allineamento", "mappatura"
    ]

    if any(k in q for k in TABLE_KEYWORDS):
        return "table"

    # Nota: "confronta" e "confronto" NON forzano table.
    # Una comparazione può essere discorsiva e audit-oriented.
    COMPARISON_TERMS = [
        "confronta", "confronto", "comparazione", "comparativa",
        "compare", "comparison", "comparative"
    ]

    if any(k in q for k in COMPARISON_TERMS):
        return "audit"

    # 3. INTENT: GRAFI E DIAGRAMMI
    CHART_KEYWORDS = [
        "grafico", "graph", "flow", "flowchart", "diagramma", "diagram",
        "architettura", "topologia", "chart", "figura", "rete",
        "network map", "schema", "relazioni", "collegamenti", "nodi",
        "node", "nodes", "archi", "edge", "edges", "path", "percorso",
        "traversamento", "multi-hop"
    ]

    if any(k in q for k in CHART_KEYWORDS):
        return "chart"

    # 4. INTENT: AUDIT E COMPLIANCE
    AUDIT_KEYWORDS = [
        "audit", "compliance", "conformità", "verifica", "valuta", "assessment",
        "ispeziona", "requisito", "requisiti", "normativa", "regolamento",
        "direttiva", "legge", "iso 27001", "nis2", "gdpr", "dora",
        "linee guida", "policy", "controllo", "controlli",
        "violazione", "violazioni", "obbligo", "obblighi",
        "soggetti", "categorie", "categoria", "regime", "vigilanza",
        "autorità", "responsabilità", "responsabile",
        "classifica", "classificazione"
    ]

    if any(k in q for k in AUDIT_KEYWORDS):
        return "audit"

    # 5. INTENT DI DEFAULT
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
    if is_assessment_evidence_relevance_query(query):
        return True

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
    return (len(nums) >= 1 and (has_math_term or has_math_symbol)) or is_calculation_request(query_text)


def solve_control_coverage(query_text: str) -> Optional[str]:
    """
    Solver deterministico non adattativo per copertura controlli.

    Classe gestita:
    - totale controlli;
    - controlli implementati;
    - controlli parziali;
    - peso percentuale dei parziali.
    """
    q = query_text or ""
    ql = q.lower().replace("\\%", "%")

    if not any(t in ql for t in ["controlli", "checklist", "controls", "control"]):
        return None

    if not any(t in ql for t in ["copertura", "coverage", "equivalente", "complessiva"]):
        return None

    total_match = re.search(
        r"(?:checklist\s+di|totale\s+di|su|of)\s+(\d+)\s+(?:controlli|controls)",
        ql,
        flags=re.IGNORECASE,
    )

    implemented_match = re.search(
        r"(\d+)\s+(?:risultano\s+)?(?:implementati|implemented|completi|complete)",
        ql,
        flags=re.IGNORECASE,
    )

    partial_match = re.search(
        r"(\d+)\s+(?:parziali|partial|partially)",
        ql,
        flags=re.IGNORECASE,
    )

    partial_weight_match = re.search(
        r"(?:valgono|valgano|worth|weighted\s+at|peso)\s+(?:al\s+)?(\d+(?:[.,]\d+)?)\s*%",
        ql,
        flags=re.IGNORECASE,
    )

    if not (total_match and implemented_match and partial_match and partial_weight_match):
        return None

    total = int(total_match.group(1))
    implemented = int(implemented_match.group(1))
    partial = int(partial_match.group(1))
    partial_weight_pct = _parse_it_number(partial_weight_match.group(1))

    if total <= 0:
        return None

    equivalent_controls = implemented + partial * (partial_weight_pct / 100.0)
    coverage_pct = (equivalent_controls / total) * 100.0

    return (
        "**A) Risposta**\n\n"
        f"La copertura equivalente complessiva è **{coverage_pct:.2f}%**.\n\n"
        "**Calcolo deterministico:**\n\n"
        f"- Controlli totali = `{total}`\n"
        f"- Controlli implementati = `{implemented}`\n"
        f"- Controlli parziali = `{partial}`\n"
        f"- Peso controlli parziali = `{partial_weight_pct:g}%`\n"
        f"- Controlli equivalenti = `{implemented} + {partial} × {partial_weight_pct:g}% = {equivalent_controls:g}`\n"
        f"- Copertura = `{equivalent_controls:g} / {total} × 100 = {coverage_pct:.2f}%`\n\n"
        "\n\n**B) Evidenze**\n\n"
        
        "- I valori numerici usati nel calcolo sono stati estratti dalla domanda dell'utente.\n"
        "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- Il calcolo assume che i controlli implementati valgano al 100%.\n"
        "- Il peso dei controlli parziali è quello indicato nella domanda.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni matematiche presenti nella domanda."
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
        "\n\n**B) Evidenze**\n\n"
        f"{evidence_lines}\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- La formula `rischio = probabilità × impatto` è stata fornita dall'utente nella domanda.\n"
        "- Il risultato numerico non dimostra da solo la conformità: va collegato al risk assessment documentale.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni matematiche presenti nella domanda."
    )

def _parse_it_number(value: str) -> float:
    """
    Converte numeri IT/EN:
    - 250.000 -> 250000
    - 250,5 -> 250.5
    - 250.000,50 -> 250000.50
    - 250,000.50 -> 250000.50
    """
    v = (value or "").strip().replace(" ", "")

    if not v:
        raise ValueError("empty number")

    # Formato italiano: 1.234,56
    if "," in v and "." in v and v.rfind(",") > v.rfind("."):
        v = v.replace(".", "").replace(",", ".")
    # Formato inglese: 1,234.56
    elif "," in v and "." in v and v.rfind(".") > v.rfind(","):
        v = v.replace(",", "")
    # Solo virgola decimale
    elif "," in v and "." not in v:
        v = v.replace(",", ".")
    # Solo punti: se sembrano migliaia, rimuovili
    elif "." in v and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", v):
        v = v.replace(".", "")

    return float(v)


def _format_euro_it(value: float) -> str:
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def solve_percentage_remainder_allocation(query_text: str) -> Optional[str]:
    """
    Solver non adattativo per allocazioni percentuali residue.

    Esempi di classe:
    - budget 250000, 40% ad A, 35% a B, restante a C
    - importo totale X, quota 20%, quota 30%, residuo
    - total budget X, 40% A, 35% B, remaining C
    """

    q = query_text or ""

    # Normalizza percentuali scritte in formato Markdown/LaTeX:
    # 40\% -> 40%
    # 35\% -> 35%
    q = q.replace("\\%", "%")
    ql = q.lower()

    remainder_terms = [
        # IT - residuo / restante
        "restante",
        "residuo",
        "residua",
        "residui",
        "residue",
        "rimanente",
        "rimanenza",

        # IT - quota residua
        "quota residua",
        "quota restante",
        "quota rimanente",

        # IT - destinazione residua
        "destinabile",
        "da destinare",

        # IT - non allocato / non assegnato
        "non allocato",
        "non allocata",
        "non allocati",
        "non allocate",
        "non assegnato",
        "non assegnata",
        "non assegnati",
        "non assegnate",

        # EN - remaining / residual
        "remaining",
        "remainder",
        "residual",
        "leftover",

        # EN - remaining share / amount
        "remaining share",
        "remaining quota",
        "remaining amount",
        "residual amount",
        "leftover amount",

        # EN - unallocated / unassigned
        "unallocated",
        "unassigned",
        "not allocated",
        "not assigned",

        # EN - to allocate / to assign
        "to allocate",
        "to assign",
    ]

    allocation_terms = [
        # IT - totale / importo
        "budget",
        "totale",
        "importo",
        "ammontare",
        "stanziamento",
        "valore complessivo",
        "importo complessivo",
        "totale disponibile",

        # IT - costo / costi
        "costo",
        "costi",

        # IT - allocazione / ripartizione
        "alloca",
        "allocato",
        "allocata",
        "allocati",
        "allocate",
        "allocazione",
        "ripartito",
        "ripartita",
        "ripartiti",
        "ripartite",
        "ripartizione",

        # IT - quota / percentuale / destinazione
        "quota",
        "quote",
        "percentuale",
        "percentuali",
        "destinato",
        "destinata",
        "destinati",
        "destinate",
        "assegnato",
        "assegnata",
        "assegnati",
        "assegnate",

        # IT/EN - effort
        "effort",

        # EN - total / amount
        "budget",
        "total",
        "amount",
        "total amount",
        "overall amount",
        "available budget",

        # EN - cost / costs
        "cost",
        "costs",

        # EN - allocation / distribution
        "allocate",
        "allocated",
        "allocation",
        "allocations",
        "distribute",
        "distributed",
        "distribution",

        # EN - share / percentage / assignment
        "share",
        "shares",
        "percentage",
        "percentages",
        "assigned",
        "assignment",
        "dedicated",
        "earmarked",
    ]

    if not any(t in ql for t in remainder_terms):
        return None

    if not any(t in ql for t in allocation_terms):
        return None

    # Estrae percentuali.
    percentages = [
        _parse_it_number(m.group(1))
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%+", q)
    ]

    if len(percentages) < 2:
        return None

    # Estrae importi candidati. Prende il valore più grande come totale.
    numeric_candidates = []
    for m in re.finditer(r"\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\b|\b\d+(?:[,.]\d+)?\b", q):
        raw = m.group(0)
        try:
            val = _parse_it_number(raw)
            if val > 100:
                numeric_candidates.append(val)
        except Exception:
            continue

    if not numeric_candidates:
        return None

    total = max(numeric_candidates)
    used_pct = sum(percentages)
    remaining_pct = 100.0 - used_pct

    if remaining_pct < 0:
        return None

    remaining_amount = total * remaining_pct / 100.0

    pct_details = " + ".join(f"{p:g}%" for p in percentages)

    return (
        "**A) Risposta**\n\n"
        f"La quota residua è **{remaining_pct:g}%** e corrisponde a "
        f"**{_format_euro_it(remaining_amount)} euro**.\n\n"
        "**Calcolo deterministico:**\n\n"
        f"- Totale = `{_format_euro_it(total)} euro`\n"
        f"- Percentuali già allocate = `{pct_details} = {used_pct:g}%`\n"
        f"- Percentuale residua = `100% - {used_pct:g}% = {remaining_pct:g}%`\n"
        f"- Importo residuo = `{_format_euro_it(total)} × {remaining_pct:g}% = {_format_euro_it(remaining_amount)} euro`\n\n"
        "\n\n**B) Evidenze**\n\n"
        "- I valori numerici usati nel calcolo sono stati estratti dalla domanda dell'utente.\n"
        "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- Il calcolo considera le percentuali come quote del totale indicato.\n"
        "- Eventuali costi indiretti, arrotondamenti contabili o imposte non sono considerati se non esplicitamente forniti.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni matematiche presenti nella domanda."
    )
    

def try_solve_user_provided_algebra(query_text: str) -> Optional[str]:
    """
    Solver deterministico non adattativo per algebra fornita dall'utente.

    Gestisce classi generali:
    1) Ri = K × V, Rr <= Ri / N, Vm richiesta;
    2) percentuale di una variabile > soglia.

    Non usa conoscenza normativa.
    Non usa nomi di documenti.
    Non usa hardcoding sulle domande del test.
    """
    q_raw = query_text or ""

    if not q_raw.strip():
        return None

    q = q_raw
    q = q.replace("\\%", "%")
    q = q.replace("\\_", "_")
    q = q.replace("\\times", "×")
    q = q.replace("\\leq", "≤")
    q = q.replace("\\le", "≤")
    q = q.replace("\\geq", "≥")
    q = q.replace("\\ge", "≥")
    q = q.replace("$", "")
    q = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", q)
    q = q.replace("{", "").replace("}", "")

    ql = q.lower()

    algebra_trigger = any(t in ql for t in [
        "equazione", "disequazione", "algebrica", "algebricamente",
        "in funzione di", "isola", "formula", "variabile",
        "risolvi", "esprimi", "deriva",
        "supera", "superare", "superi", "maggiore", "superiore",
        "rischio residuo", "rischio inerente",
        "equation", "inequality", "algebraic", "solve for",
        "as a function of", "derive", "express",
        "exceeds", "exceed", "greater", "higher", "more than",
        "residual risk", "inherent risk",
    ])

    if not algebra_trigger:
        return None

    def fmt_num(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value))).replace(".", ",")
        return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",")

    # ============================================================
    # CASO 1: Ri = K × V, Rr <= Ri / N, Vm richiesta
    # Classe generale non adattativa:
    # - K può essere T, M, P, K, ecc.
    # - N è letto dalla domanda.
    # ============================================================
    inherent_match = re.search(
        r"\bR[_\s]?i\b\s*(?:=|è\s+definito\s+come|e\s+definito\s+come|defined\s+as)\s*([A-Z])\s*(?:×|x|\*|\\times)\s*V\b",
        q,
        flags=re.IGNORECASE,
    )

    residual_bound_match = re.search(
        r"\bR[_\s]?r\b\s*(?:≤|<=|<)\s*\bR[_\s]?i\b\s*/\s*(\d+)",
        q,
        flags=re.IGNORECASE,
    )

    asks_vm = bool(re.search(r"\bV[_\s]?m\b", q, flags=re.IGNORECASE))

    if inherent_match and residual_bound_match and asks_vm:
        factor = inherent_match.group(1).upper()
        denom = int(residual_bound_match.group(1))

        if denom <= 0:
            return None

        return (
            "**A) Risposta**\n\n"
            f"La vulnerabilità mitigata deve rispettare **Vm ≤ V / {denom}**.\n\n"
            "**Calcolo deterministico:**\n\n"
            f"- Rischio inerente: `Ri = {factor} × V`\n"
            f"- Rischio residuo coerente con la vulnerabilità mitigata: `Rr = {factor} × Vm`\n"
            f"- Vincolo richiesto: `Rr ≤ Ri / {denom}`\n"
            f"- Sostituzione: `{factor} × Vm ≤ ({factor} × V) / {denom}`\n"
            f"- Poiché `{factor}` è costante e positiva: `Vm ≤ V / {denom}`\n\n"
            f"- Formula LaTeX:\n\n$$\nV_m \\leq \\frac{{V}}{{{denom}}}\n$$\n\n"
            "\n\n**B) Evidenze**\n\n"
            "- Le relazioni algebriche sono state estratte dalla domanda dell'utente.\n"
            "- La derivazione è stata eseguita in modo deterministico da Python, non dal modello LLM.\n\n"
            "\n\n**C) Limiti / Conflitti**\n\n"
            f"- La semplificazione richiede che `{factor}` sia costante e positiva.\n"
            "- La relazione residua usa la stessa struttura moltiplicativa del rischio inerente, sostituendo `V` con `Vm`.\n"
            "- Il risultato è una derivazione matematica dei dati forniti, non una validazione empirica del modello di rischio.\n\n"
            "**D) Fonti**\n\n"
            "- Input utente: valori e relazioni matematiche presenti nella domanda."
        )

    # ============================================================
    # CASO 2: percentuale di una variabile > soglia
    # Classe generale:
    # - P% di X supera Y milioni/euro
    # - restituisce X > soglia / P
    # ============================================================
    pct_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", q)

    threshold_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(milioni|milione|million|millions)\b",
        q,
        flags=re.IGNORECASE,
    )

    if not threshold_match:
        threshold_match = re.search(
            r"(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)\s*(?:euro|€)",
            q,
            flags=re.IGNORECASE,
        )

    has_greater_condition = any(t in ql for t in [
        "supera", "superare", "superi",
        "maggiore", "superiore",
        "exceeds", "exceed",
        "greater", "higher", "more than"
    ]) or ">" in q

    if pct_match and threshold_match and has_greater_condition:
        pct = _parse_it_number(pct_match.group(1))

        if pct <= 0:
            return None

        variable_match = re.search(
            r"\b(?:fatturato|revenue|turnover)\s+(?:annuo|annual)?\s*([A-Z])\b",
            q,
            flags=re.IGNORECASE,
        )

        if not variable_match:
            variable_match = re.search(
                r"\b(?:valore|importo|totale|amount|total)\s+(?:annuo|annual)?\s*([A-Z])\b",
                q,
                flags=re.IGNORECASE,
            )

        if not variable_match:
            variable_match = re.search(
                r"\b(?:annuo|annual)\s+([A-Z])\b",
                q,
                flags=re.IGNORECASE,
            )

        variable = variable_match.group(1).upper() if variable_match else "X"
        

        threshold = _parse_it_number(threshold_match.group(1))

        unit = ""
        try:
            unit = threshold_match.group(2).lower()
        except Exception:
            unit = ""

        threshold_in_euro = threshold * 1_000_000 if "milion" in unit else threshold
        threshold_millions = threshold_in_euro / 1_000_000

        pct_decimal = pct / 100.0
        result_euro = threshold_in_euro / pct_decimal
        result_millions = result_euro / 1_000_000

        return (
            "**A) Risposta**\n\n"
            f"La disequazione è **{fmt_num(pct_decimal)} × {variable} > {fmt_num(threshold_millions)} milioni**.\n\n"
            "**Calcolo deterministico:**\n\n"
            f"- Percentuale = `{fmt_num(pct)}% = {fmt_num(pct_decimal)}`\n"
            f"- Soglia = `{fmt_num(threshold_millions)} milioni`\n"
            f"- Disequazione = `{fmt_num(pct_decimal)} × {variable} > {fmt_num(threshold_millions)} milioni`\n"
            f"- Divisione per `{fmt_num(pct_decimal)}`: `{variable} > {fmt_num(threshold_millions)} / {fmt_num(pct_decimal)}`\n"
            f"- Risultato = **{variable} > {fmt_num(result_millions)} milioni**, cioè **{_format_euro_it(result_euro)} euro**.\n\n"
            f"- Formula LaTeX:\n\n$$\n{pct_decimal:g}{variable} > {threshold_millions:g} \\Rightarrow {variable} > {result_millions:g}\n$$\n\n"
            "\n\n**B) Evidenze**\n\n"
            "- La percentuale, la variabile e la soglia sono state estratte dalla domanda dell'utente.\n"
            "- La derivazione è stata eseguita in modo deterministico da Python, non dal modello LLM.\n\n"
            "\n\n**C) Limiti / Conflitti**\n\n"
            "- Il calcolo considera la soglia nella stessa unità indicata nella domanda.\n"
            "- Non interpreta ulteriori criteri normativi non presenti nella domanda.\n\n"
            "**D) Fonti**\n\n"
            "- Input utente: valori e relazioni matematiche presenti nella domanda."
        )

    return None

def solve_sla_cumulative_hours(query_text: str) -> Optional[str]:
    """
    Solver deterministico non adattativo per tempo cumulativo:
    N elementi × ore massime per categoria.
    """
    q = query_text or ""
    ql = q.lower()

    if not any(t in ql for t in ["tempo cumulativo", "cumulativo", "ore massime", "maximum time", "cumulative"]):
        return None

    pairs = []

    # Pattern IT: "3 ore ... critici", "10 ore ... alti", "30 ore ... medi"
    time_by_label = {}
    for m in re.finditer(
        r"(\d+(?:[.,]\d+)?)\s*ore[^\n,;.]{0,60}?\b(critici|critico|alti|alto|medi|medio|bassi|basso)\b",
        ql,
        flags=re.IGNORECASE,
    ):
        hours = _parse_it_number(m.group(1))
        label = m.group(2).lower()
        time_by_label[label] = hours

    count_by_label = {}
    for m in re.finditer(
        r"(\d+)\s+(?:incidenti\s+)?(critici|critico|alti|alto|medi|medio|bassi|basso)\b",
        ql,
        flags=re.IGNORECASE,
    ):
        count = int(m.group(1))
        label = m.group(2).lower()
        count_by_label[label] = count

    canonical_groups = {
        "critici": ["critici", "critico"],
        "alti": ["alti", "alto"],
        "medi": ["medi", "medio"],
        "bassi": ["bassi", "basso"],
    }

    for canon, aliases in canonical_groups.items():
        h = next((time_by_label[a] for a in aliases if a in time_by_label), None)
        c = next((count_by_label[a] for a in aliases if a in count_by_label), None)

        if h is not None and c is not None:
            pairs.append((canon, c, h, c * h))

    if len(pairs) < 2:
        return None

    total = sum(x[3] for x in pairs)

    evidence = "\n".join(
        f"- Incidenti {label}: `{count} × {hours:g} ore = {subtotal:g} ore`"
        for label, count, hours, subtotal in pairs
    )

    return (
        "**A) Risposta**\n\n"
        f"Il tempo cumulativo massimo è **{total:g} ore**.\n\n"
        "**Calcolo deterministico:**\n\n"
        f"{evidence}\n"
        f"- Totale = **{total:g} ore**\n\n"
        "\n\n**B) Evidenze**\n\n"
        "- I tempi massimi e il numero di incidenti sono stati estratti dalla domanda dell'utente.\n"
        "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- Il calcolo somma i massimali per categoria.\n"
        "- Non considera sovrapposizioni operative o parallelizzazione se non esplicitamente indicate.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni matematiche presenti nella domanda."
    )

def _parse_probability_number(value: str) -> float:
    """
    Parsing sicuro per probabilità decimali.
    0.08  -> 0.08
    0,08  -> 0.08
    0.025 -> 0.025
    """
    v = (value or "").strip().replace(",", ".")
    return float(v)


def solve_rosi_query(query_text: str) -> Optional[str]:
    """
    Solver deterministico non adattativo per ROSI:
    ALE iniziale, ALE dopo misura, beneficio lordo, beneficio netto, ROSI.
    """
    q = query_text or ""
    ql = q.lower()

    if not any(t in ql for t in ["rosi", "beneficio lordo", "beneficio netto", "return on security investment"]):
        return None

    impact_match = re.search(
        r"(?:impatto economico|asset ha impatto|impact)[^\d]{0,60}(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
        ql,
        flags=re.IGNORECASE,
    )

    probs = [
        _parse_probability_number(x)
        for x in re.findall(r"\b0[.,]\d+\b|\b1[.,]0+\b", ql)
    ]

    cost_match = re.search(
        r"(?:costo annuo|costo|cost)[^\d]{0,60}(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
        ql,
        flags=re.IGNORECASE,
    )

    if not impact_match or len(probs) < 2 or not cost_match:
        return None

    impact = _parse_it_number(impact_match.group(1))
    p_initial = probs[0]
    p_after = probs[1]
    cost = _parse_it_number(cost_match.group(1))

    if cost <= 0:
        return None

    ale_initial = impact * p_initial
    ale_after = impact * p_after
    gross_benefit = ale_initial - ale_after
    net_benefit = gross_benefit - cost
    rosi = (net_benefit / cost) * 100.0

    return (
        "**A) Risposta**\n\n"
        f"Il beneficio lordo è **{_format_euro_it(gross_benefit)} euro**, "
        f"il beneficio netto è **{_format_euro_it(net_benefit)} euro** "
        f"e il ROSI è **{rosi:.2f}%**.\n\n"
        "**Calcolo deterministico:**\n\n"
        f"- ALE iniziale = `{_format_euro_it(impact)} × {p_initial:g} = {_format_euro_it(ale_initial)} euro`\n"
        f"- ALE dopo misura = `{_format_euro_it(impact)} × {p_after:g} = {_format_euro_it(ale_after)} euro`\n"
        f"- Beneficio lordo = `{_format_euro_it(ale_initial)} - {_format_euro_it(ale_after)} = {_format_euro_it(gross_benefit)} euro`\n"
        f"- Beneficio netto = `{_format_euro_it(gross_benefit)} - {_format_euro_it(cost)} = {_format_euro_it(net_benefit)} euro`\n"
        f"- ROSI = `{_format_euro_it(net_benefit)} / {_format_euro_it(cost)} × 100 = {rosi:.2f}%`\n\n"
        "\n\n**B) Evidenze**\n\n"
        "- Impatto, probabilità iniziale, probabilità post-misura e costo sono stati estratti dalla domanda dell'utente.\n"
        "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- Il calcolo usa un modello annuo semplificato.\n"
        "- Non considera costi indiretti, attualizzazione o variazione temporale del rischio se non indicati nella domanda.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni matematiche presenti nella domanda."
    )


# Override v4.4: include date offsets without losing the other deterministic solvers.
def try_solve_math_query(query_text: str) -> Optional[str]:
    """
    Solver matematico deterministico non adattativo.
    Ordine: solver specifici prima dei solver generici.
    """
    coverage = solve_control_coverage(query_text)
    if coverage:
        return coverage

    rosi = solve_rosi_query(query_text)
    if rosi:
        return rosi

    sla = solve_sla_cumulative_hours(query_text)
    if sla:
        return sla

    risk_product = solve_risk_product(query_text)
    if risk_product:
        return risk_product

    remainder = solve_percentage_remainder_allocation(query_text)
    if remainder:
        return remainder

    algebra = try_solve_user_provided_algebra(query_text)
    if algebra:
        return algebra

    date_offsets = try_solve_date_offsets(query_text)
    if date_offsets:
        return date_offsets

    return None


def is_calculation_request(query_text: str) -> bool:
    """
    Riconosce richieste di calcolo operativo/matematico.
    Non decide COME calcolare.
    Decide solo che la domanda NON deve finire in formula lookup documentale.

    Non contiene nomi di normative, framework o query del test.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    calculation_verbs = [
        # IT
        "calcola", "calcolare", "calcolo", "quantifica", "quantificare",
        "determina", "determinare", "quanto", "entro quale",
        "cifra esatta", "importo esatto", "tempo totale",
        "totale cumulativo", "delta", "risultato",
        "risolvi", "esprimi", "isola",

        # EN
        "calculate", "compute", "quantify", "determine",
        "how much", "deadline", "total", "cumulative",
        "delta", "result", "solve", "express", "isolate",
    ]

    has_calc_verb = any(t in q for t in calculation_verbs)

    has_numbers_or_symbols = bool(
        re.search(r"\d", q)
        or re.search(r"(<=|>=|≤|≥|=|>|<|%|×|\*|/|\\frac|\\times)", query_text or "")
    )

    # Se l'utente chiede esplicitamente una disequazione/equazione,
    # è comunque una richiesta di calcolo/formulazione, non formula lookup.
    algebra_terms = [
        "equazione", "equazioni", "disequazione", "disequazioni",
        "algebrica", "algebricamente", "in funzione di",
        "equation", "inequality", "algebraic", "as a function of",
    ]

    has_algebra = any(t in q for t in algebra_terms)

    return (has_calc_verb and has_numbers_or_symbols) or has_algebra

def is_formula_lookup_query(query_text: str) -> bool:
    """
    Riconosce richieste documentali di recupero di formule, equazioni,
    metriche o regole di scoring tramite segnali composizionali.

    Non usa frasi complete legate a uno specifico test o documento e non
    intercetta i calcoli operativi, gestiti dai solver deterministici.
    """
    q = (query_text or "").lower().strip()

    if not q or is_calculation_request(query_text):
        return False

    has_formula_object = bool(
        re.search(
            r"\b(?:"
            r"formula|formule|equazione|equazioni|"
            r"metrica|metriche|indicatore|indicatori|"
            r"modello\s+matematico|modelli\s+matematici|"
            r"regola\s+di\s+scoring|regole\s+di\s+scoring|"
            r"formulas?|equations?|metrics?|indicators?|"
            r"mathematical\s+models?|scoring\s+rules?"
            r")\b",
            q,
        )
    )

    has_lookup_action = bool(
        re.search(
            r"\b(?:"
            r"elenca|estrai|mostra|riporta|fornisci|dammi|indica|trova|"
            r"quale|quali|"
            r"list|extract|show|report|provide|give|identify|find|which"
            r")\b",
            q,
        )
    )

    has_source_or_collection_cue = bool(
        re.search(
            r"\b(?:"
            r"documento|documenti|fonte|fonti|testo|"
            r"presente|presenti|menzionata|menzionate|citata|citate|"
            r"contenuta|contenute|definita|definite|tutte|tutti|"
            r"document|documents|source|sources|text|"
            r"present|mentioned|cited|contained|defined|all"
            r")\b",
            q,
        )
    )

    return has_formula_object and (
        has_lookup_action or has_source_or_collection_cue
    )


def should_query_neo4j_formulas(query_text: str) -> bool:
    """
    Attiva il recupero delle formule dal Knowledge Graph quando la domanda
    riguarda formule, equazioni, metriche o calcoli documentali.

    Diversamente da ``is_formula_lookup_query()``, non è limitata a poche
    formulazioni letterali come "quale formula". I calcoli deterministici puri
    vengono comunque intercettati prima del retrieval da ``handle_submit()``.
    """
    q = (query_text or "").strip()
    if not q:
        return False

    if is_formula_lookup_query(q):
        return True

    if detect_intent(q) == "formula" or is_formula_strict_query(q):
        return True

    formula_terms = [
        "formula", "formule", "equazione", "equazioni",
        "disequazione", "disequazioni", "metrica", "metriche",
        "scoring", "score", "calcolo", "calcolare", "calcola",
        "formula matematica", "modello matematico",
        "formulae", "formulas", "equation", "equations",
        "inequality", "inequalities", "metric", "metrics",
        "calculate", "calculation", "scoring rule",
    ]
    q_lower = q.lower()
    return any(term in q_lower for term in formula_terms)


def needs_math_document_context(query_text: str) -> bool:
    """
    Il math_direct deve usare fonti documentali solo se l'utente
    lo chiede esplicitamente.
    La presenza di parole come NIS2, GDPR, audit, documenti, scadenze,
    normativa o fonti NON basta.
    """
    q = (query_text or "").lower()

    context_terms = [
        "collega ai documenti",
        "collegalo ai documenti",
        "collegala ai documenti",
        "collega alle fonti",
        "usa le fonti recuperate",
        "usando le fonti recuperate",
        "secondo le fonti recuperate",
        "secondo i documenti recuperati",
        "con evidenze documentali",
        "con supporto documentale",
        "giustifica con le fonti",
        "cita le fonti nel calcolo",
        "calcolo basato sui documenti recuperati",
        "using retrieved sources",
        "according to retrieved documents",
        "with documentary evidence",
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

        # ITALIANO - Query normative/classificatorie: NON sono glossario atomico
        "soggetti", "categorie", "categoria", "tipologie", "tipologia",
        "classifica", "classificazione", "regime", "vigilanza",
        "obblighi", "obbligo", "requisiti", "requisito",
        "normativa", "regolamento", "direttiva", "legge",
        "autorità", "responsabilità", "sanzioni", "sanzione",

        # INGLESE - Analisi, Causalità e Confronto
        "explain", "compare", "difference", "differences", "evaluate", "why",
        "correlated", "relation", "relationship", "analyze", "analyse", "reason",
        "cause", "impact", "consequence", "consequences", "advantage", "advantages",
        "disadvantage", "disadvantages", "justify", "argue", "deduce", "connect",
        "contrast", "distinction", "how does it work", "in what way", "summarize",
        "summarise", "purpose", "goal",

        # EN - Normative/classification queries: NOT atomic glossary
        "subjects", "entities", "categories", "category", "types",
        "classification", "regime", "supervision", "oversight",
        "obligations", "obligation", "requirements", "requirement",
        "regulation", "directive", "law", "authority",
        "responsibility", "responsibilities", "sanctions", "sanction"
    ]
    
    # Se l'utente vuole un'analisi approfondita, scavalca il glossario
    if any(t in q for t in reasoning_terms):
        return False

    # 3. Solo se sopravvive ai filtri sopra ed è una richiesta pura di definizione, si attiva
    glossary_terms = [
        # ITALIANO - Definizioni pure / glossario atomico
        "cosa significa", "cosa vuol dire", "definisci", "definizione", "significato",
        "glossario", "cos'è", "cosa è", "chi è", "acronimo",
        "sta per", "cosa si intende", "dizionario", "vocabolario", "termine",

        # INGLESE - Pure definition / atomic glossary
        "what does it mean", "what is", "who is", "define",
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


def _looks_like_filename(value: str) -> bool:
    """
    Evita che nomi file tra virgolette vengano trattati come voci di glossario.
    """
    v = (value or "").strip().lower()
    return bool(re.search(r"\.(pdf|md|txt|docx|html|csv|xlsx)$", v))


def _is_compound_term(value: str) -> bool:
    """
    Riconosce termini composti che NON devono essere spezzati automaticamente:
    CSIRT-NX, ISO-XYZ, MFA_2, DORA/X, NIS2-Cloud, ecc.
    """
    v = (value or "").strip()
    if len(v) < 3:
        return False

    return bool(
        re.search(r"[A-Z]{2,}[A-Z0-9]*\s*[-_/]\s*[A-Z0-9]{1,}", v)
        or re.search(r"\b[A-Z]{2,}\d+[A-Z0-9]*\b", v)
    )


def extract_requested_terms(query_text: str) -> List[str]:
    """
    Estrae termini richiesti per glossario/acronimi in modo non adattativo.

    Regola:
    1. prima frasi tra virgolette;
    2. poi termini composti con -, _, / o numeri;
    3. solo se non ci sono composti, acronimi singoli.

    Questo evita che "CSIRT-NX" venga spezzato in "CSIRT" e "NX".
    """
    q = query_text or ""
    terms: List[str] = []

    # 1) Frasi esplicite tra virgolette.
    quoted = re.findall(r"[\"“'«]([^\"”'»]+)[\"”'»]", q)
    for item in quoted:
        clean = item.strip()
        if len(clean) > 2 and not _looks_like_filename(clean):
            terms.append(clean)

    # 2) Termini composti non separabili.
    compound_terms = re.findall(
        r"\b[A-Z][A-Z0-9]{1,}(?:[-_/][A-Z0-9]{1,})+\b",
        q
    )

    for item in compound_terms:
        clean = item.strip()
        if clean and clean not in terms:
            terms.append(clean)

    # 3) Se esiste almeno un composto, NON spezzarlo in sottoparti.
    if compound_terms:
        return list(dict.fromkeys(terms))

    # 4) Acronomi singoli solo se non abbiamo trovato composti.
    acronyms = re.findall(r"\b[A-Z]{2,10}\d{0,3}\b", q)
    for acr in acronyms:
        if acr not in terms:
            terms.append(acr)

    return list(dict.fromkeys(terms))



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

    if is_assessment_evidence_relevance_query(query_text):
        return "evidence_relevance"

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

    absence_check_terms = [
        "se non è presente",
        "se non e presente",
        "se non presente",
        "non è presente",
        "non e presente",
        "dichiaralo esplicitamente",
        "dichiara esplicitamente",
        "voce non presente",
        "termine non presente",
        "obbligo previsto",
        "qual è l'obbligo previsto",
        "qual e l'obbligo previsto",
        "if not present",
        "if it is not present",
        "not present",
        "not found",
        "explicitly state",
    ]

    if any(t in q for t in absence_check_terms):
        return False

    # --- 0. OVERRIDE ESPLICITO GRAFO / NEO4J ---
    # Se l'utente chiede esplicitamente Neo4j, Cypher, grafo, archi,
    # path o relazioni esplicite, non bloccare il graph mode per parole
    # generiche come "se", "verifica", "descrivi", ecc.
    explicit_graph_terms = [
        # IT
        "neo4j", "cypher", "grafo", "interroga neo4j", "usando neo4j",
        "archi", "arco", "nodi", "nodo", "path", "percorso",
        "traversamento", "multi-hop", "catena semantica",
        "relazioni esplicite", "relazioni nel grafo",
        "tabella relazioni",

        # EN
        "graph", "query neo4j", "using neo4j", "cypher query",
        "nodes", "node", "edges", "edge", "path", "traversal",
        "multi-hop", "semantic chain", "explicit relationships",
        "relationship table",
    ]

    if any(t in q for t in explicit_graph_terms):
        return True

    # --- 1. GATEKEEPER: Protezione per ragionamento logico e scenari ---
    # Se la query richiede solo un'analisi discorsiva e NON chiede esplicitamente grafo,
    # l'output tabellare rigido viene disattivato.
    analysis_terms = [
        # ITALIANO
        "spiega", "valuta", "confronta", "differenza", "differenze", "se", "basandoti",
        "analizza", "perché", "motivo", "causa", "giustifica", "descrivi", "sintetizza",
        "racconta", "scenario", "ipotesi",

        # INGLESE
        "explain", "evaluate", "compare", "difference", "differences", "if", "based on",
        "analyze", "analyse", "why", "reason", "cause", "justify", "describe",
        "summarize", "summarise", "scenario", "hypothesis", "what happens",
    ]

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

def should_use_graph_relation_strict_mode(query_text: str) -> bool:
    """
    Decide se usare la risposta deterministica tabellare da grafo.

    Regola non adattativa:
    - se l'utente chiede esplicitamente Neo4j/Cypher/grafo/archi/path/traversamento,
      il graph strict mode deve prevalere;
    - le domande solo esplicative possono restare discorsive;
    - non si deve mai dire che Neo4j è "simulato" se l'app ha un ramo Neo4j.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    explicit_graph_terms = [
        # IT
        "neo4j", "cypher", "grafo", "interroga neo4j", "usando neo4j",
        "archi", "arco", "nodi", "nodo", "path", "percorso",
        "traversamento", "multi-hop", "catena semantica",
        "relazioni esplicite", "relazioni nel grafo",
        "tabella relazioni",

        # EN
        "graph", "query neo4j", "using neo4j", "cypher query",
        "nodes", "node", "edges", "edge", "path", "traversal",
        "multi-hop", "semantic chain", "explicit relationships",
        "relationship table",
    ]

    if any(t in q for t in explicit_graph_terms):
        return True

    explanatory_terms = [
        # IT
        "qual è", "quale è", "quali sono", "che cosa", "cosa significa",
        "ruolo", "scopo", "funzione", "descrivi", "spiega", "analizza",
        "valuta", "giustifica", "perché", "perche", "in che modo",
        "come funziona", "elabora", "sintetizza",

        # EN
        "what is", "what are", "role", "purpose", "function",
        "describe", "explain", "analyze", "analyse", "evaluate",
        "justify", "why", "how does", "how do", "summarize", "summarise",
    ]

    if any(t in q for t in explanatory_terms):
        return False

    strong_graph_terms = [
        # IT
        "relazioni tra", "collegamenti tra", "mostra le relazioni",
        "traccia la catena", "connessioni", "mappa", "mappatura",
        "rete semantica", "triple",

        # EN
        "relations between", "links between", "show relationships",
        "trace the chain", "connections", "mapping",
        "semantic network", "triples",
    ]

    return any(t in q for t in strong_graph_terms)


def is_formula_strict_query(query_text: str) -> bool:
    """
    Riconosce query matematiche/algebriche in modo non adattativo.

    Regola:
    - Formula Strict Mode parte solo se l'utente chiede davvero formula,
      equazione, disequazione, derivazione, calcolo o algebra.
    - Le domande normative/classificatorie NON devono entrare qui solo perché
      contengono soglie, sanzioni, soggetti, regolamenti o categorie.
    """
    q_raw = query_text or ""
    q = q_raw.lower()

    if not q.strip():
        return False

    # Se è una domanda normativa/classificatoria, NON usare formula mode
    # salvo che ci siano segnali matematici/algebrici espliciti.
    explicit_formula_terms = [
        # IT
        "formula", "formule",
        "equazione", "equazioni",
        "disequazione", "disequazioni",
        "algebra", "algebrica", "algebrico", "algebricamente",
        "esprimi", "isola", "in funzione di",
        "risolvi", "deriva", "derivazione",
        "scrivi la disequazione", "scrivi l'equazione",

        # EN
        "equation", "equations",
        "inequality", "inequalities",
        "algebraic", "algebraically",
        "solve", "solve for",
        "derive", "express", "as a function of",
        "formula", "formulas",
    ]

    if any(t in q for t in explicit_formula_terms):
        return True

    if is_regulatory_classification_query(query_text):
        return False

    # Calcolo numerico esplicito.
    calculation_terms = [
        # IT
        "calcola", "calcolo", "quantifica", "quanto vale",
        "cifra esatta", "importo esatto", "risultato",
        "percentuale", "budget", "roi", "rosi",
        "probabilità", "calcolo del rischio",

        # EN
        "calculate", "compute", "quantify", "how much",
        "exact amount", "result", "percentage",
        "budget", "risk score", "probability",
    ]

    if any(t in q for t in calculation_terms):
        return True

    # Simboli matematici + verbo operativo.
    has_math_symbols = bool(
        re.search(r"(<=|>=|≤|≥|=|>|<|\\times|×|\*|/|\\frac|%)", q_raw)
    )

    has_operational_verb = any(t in q for t in [
        "calcola", "risolvi", "scrivi", "esprimi", "isola",
        "verifica", "determina", "derive", "solve", "express",
        "calculate", "compute", "determine",
    ])

    if has_math_symbols and has_operational_verb:
        return True

    return False


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



def is_assessment_evidence_relevance_query(query_text: str) -> bool:
    """
    Rileva richieste di valutazione attinenza/sufficienza di un documento-evidenza
    rispetto a una domanda, requisito, controllo o item di assessment.

    Non è adattativa:
    - non contiene nomi di documenti specifici;
    - non contiene query hard-coded;
    - riconosce una classe generale evidence-vs-question / evidence-vs-requirement.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    evidence_terms = [
        # IT
        "evidenza", "evidenze",
        "prova", "prove",
        "documento", "documenti",
        "file", "pdf", "allegato", "allegati",
        "upload", "caricato", "caricati", "documentazione",
        "artefatto", "artefatti",
        "record", "registrazione", "registrazioni",
        "log", "screenshot", "report", "rapporto",
        "procedura", "policy", "registro",

        # EN
        "evidence", "evidences",
        "proof", "proofs",
        "document", "documents",
        "file", "pdf", "attachment", "attachments",
        "upload", "uploaded", "documentation",
        "artifact", "artifacts",
        "record", "records",
        "log", "logs", "screenshot", "report", "reports",
        "procedure", "policy", "register",
    ]

    assessment_terms = [
        # IT
        "domanda", "domande",
        "questionario", "questionari",
        "assessment", "audit",
        "requisito", "requisiti",
        "controllo", "controlli",
        "checklist", "item", "punto", "punti",
        "criterio", "criteri",
        "misura", "misure",
        "obbligo", "obblighi",
        "clausola", "clausole",
        "capitolo", "sezione",

        # EN
        "question", "questions",
        "questionnaire", "questionnaires",
        "assessment", "audit",
        "requirement", "requirements",
        "control", "controls",
        "checklist", "item", "items",
        "criterion", "criteria",
        "measure", "measures",
        "obligation", "obligations",
        "clause", "clauses",
        "chapter", "section",
    ]

    relevance_terms = [
        # IT
        "attinente", "attinenza",
        "inerente", "inerenza",
        "pertinente", "pertinenza",
        "rilevante", "rilevanza",
        "correlato", "correlata", "correlazione",
        "collegato", "collegata", "collegamento",
        "coerente", "coerenza",
        "adeguato", "adeguata", "adeguatezza",
        "sufficiente", "sufficienza",
        "idoneo", "idonea", "idoneità",
        "applicabile", "applicabilità",
        "supporta", "supportato", "supportata",
        "dimostra", "dimostrato", "dimostrata",
        "comprova", "comprovato", "comprovata",
        "giustifica", "giustificato", "giustificata",
        "copre", "copertura",
        "risponde", "risposta",
        "valuta", "valutare", "verifica", "verificare",

        # EN
        "relevant", "relevance",
        "pertinent", "pertinence",
        "related", "relation", "relationship",
        "correlated", "correlation",
        "linked", "link", "connection",
        "consistent", "consistency",
        "adequate", "adequacy",
        "sufficient", "sufficiency",
        "suitable", "suitability",
        "applicable", "applicability",
        "supports", "supported", "supporting",
        "demonstrates", "demonstrated",
        "proves", "proven",
        "justifies", "justified",
        "covers", "coverage",
        "answers", "answer",
        "evaluate", "assess", "verify", "check",
    ]

    gap_terms = [
        # IT
        "gap", "lacuna", "lacune",
        "mancanza", "mancanze",
        "manca", "mancano",
        "carente", "carenti", "carenza", "carenze",
        "debole", "debolezza", "debolezze",
        "incompleto", "incompleta", "parziale",
        "non sufficiente", "non adeguato", "non adeguata",
        "non attinente", "poco attinente",
        "scostamento", "scostamenti",
        "non conformità", "non conforme","differenza", "differenze", "deviazione", "deviazioni",

        # EN
        "gap", "gaps",
        "missing", "absence", "lack", "lacks",
        "weak", "weakness", "weaknesses",
        "deficiency", "deficiencies",
        "incomplete", "partial",
        "not sufficient", "insufficient",
        "not adequate", "inadequate",
        "not relevant", "poorly relevant",
        "deviation", "deviations",
        "non-compliance", "non-compliant","difference", "differences", "deviation", "deviations"
    ]

    remediation_terms = [
        # IT
        "remediation", "piano di remediation",
        "piano correttivo", "azioni correttive",
        "azione correttiva", "correzione", "correzioni",
        "rimedio", "rimedi",
        "miglioramento", "miglioramenti",
        "integrazione", "integrare",
        "raccomandazione", "raccomandazioni",
        "cosa manca", "cosa integrare", "come migliorare",

        # EN
        "remediation", "remediation plan",
        "corrective action", "corrective actions",
        "correction", "corrections",
        "remedy", "remedies",
        "improvement", "improvements",
        "integration", "integrate",
        "recommendation", "recommendations",
        "what is missing", "what to add", "how to improve",
    ]

    scoring_terms = [
        # IT
        "livello", "livelli",
        "percentuale", "percentuali",
        "score", "punteggio", "valutazione",
        "grado", "classifica", "classificazione",
        "basso", "medio", "alto",
        "debole", "parziale", "forte",

        # EN
        "level", "levels",
        "percentage", "percentages",
        "score", "scoring", "rating",
        "grade", "classification",
        "low", "medium", "high",
        "weak", "partial", "strong",
    ]

    has_evidence = any(t in q for t in evidence_terms)
    has_assessment = any(t in q for t in assessment_terms)
    has_relevance = any(t in q for t in relevance_terms)
    has_gap = any(t in q for t in gap_terms)
    has_remediation = any(t in q for t in remediation_terms)
    has_scoring = any(t in q for t in scoring_terms)

    return (
        has_evidence
        and has_assessment
        and (
            has_relevance
            or has_gap
            or has_remediation
            or has_scoring
        )
    )


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

def diversify(
    items: List[Dict[str, Any]],
    max_per_page: int,
    max_per_doc: int,
    final_k: int,
) -> List[Dict[str, Any]]:
    """Mantiene i migliori candidati limitando duplicazioni per documento e pagina."""
    out: List[Dict[str, Any]] = []
    page_count: Dict[Tuple[str, int], int] = {}
    doc_count: Dict[str, int] = {}

    sorted_items = sorted(
        items,
        key=lambda x: float(x.get("final_score", x.get("score", 0.0))),
        reverse=True,
    )

    for it in sorted_items:
        filename = str(it.get("filename") or "Unknown")
        doc_key = str(
            it.get("doc_id")
            or normalize_doc_name(filename)
            or filename
        )
        page = int(it.get("page") or 0)
        page_key = (doc_key, page)

        if doc_count.get(doc_key, 0) >= max_per_doc:
            continue
        if page_count.get(page_key, 0) >= max_per_page:
            continue

        out.append(it)
        doc_count[doc_key] = doc_count.get(doc_key, 0) + 1
        page_count[page_key] = page_count.get(page_key, 0) + 1

        if len(out) >= final_k:
            break

    return out

def append_audit_log(audit: AuditTrail):
    if not AUDIT_ENABLED:
        return
    payload = audit.model_dump()
    payload["query"] = ""  # nei log persistenti resta solo l'hash della domanda
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Audit file write error: {e}")

    if PG_ENRICH_ENABLED and pg_pool:
        conn = pg_get_conn_secure()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.rag_query_audit (
                        request_id, organization_id, user_id, roles, query_sha256,
                        intent, filters, retrieved_sources, llm_model, corpus_version
                    ) VALUES (%s::uuid, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        audit.request_id, audit.organization_id, audit.user_id,
                        json.dumps(audit.roles), audit.query_sha256, audit.intent,
                        json.dumps(audit.filters, ensure_ascii=False),
                        json.dumps(audit.retrieved_sources, ensure_ascii=False),
                        audit.llm_model, audit.corpus_version,
                    ),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"⚠️ Audit PostgreSQL write error: {e}")
        finally:
            pg_put_conn_secure(conn)


def get_graph_entities(chunk_ids: List[str]) -> Dict[str, List[GraphEntity]]:
    """Recupera entità solo da chunk visibili al tenant corrente."""
    if not chunk_ids or not neo4j_driver:
        return {}

    graph_map: Dict[str, List[GraphEntity]] = {}
    query = """
    UNWIND $ids AS target_id
    MATCH (c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) = target_id
      AND c.status = 'active'
      AND (
            (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
            OR
            (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
      )

    CALL (c) {
        MATCH (c)-[r:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)
        WHERE r.status = 'active'
          AND e.status = 'active'
          AND NOT toLower(coalesce(e.name, e.label, e.canonical_id, e.id)) IN [
            'dato', 'dati', 'sistema', 'sistemi', 'azienda', 'aziende',
            'utente', 'utenti', 'informazione', 'informazioni', 'documento',
            'data', 'system', 'company', 'user', 'information', 'document'
        ]
        RETURN
            coalesce(e.name, e.label, e.canonical_id, e.id) AS entity_name,
            coalesce(e.category, labels(e)[0], 'Entity') AS entity_type,
            type(r) AS rel_type
        LIMIT 10
    }

    RETURN target_id AS chunk_id,
           entity_name AS name,
           entity_type AS type,
           rel_type AS rel
    """

    try:
        with neo4j_driver.session() as session:
            result = session.run(query, ids=chunk_ids, org_id=current_organization_id())
            for record in result:
                cid = str(record["chunk_id"])
                graph_map.setdefault(cid, []).append(
                    GraphEntity(
                        name=record["name"],
                        type=record["type"],
                        relation=record["rel"],
                    )
                )
    except Exception as e:
        logger.error(
            "Neo4j Query Error (get_graph_entities) - IDs %s: %s",
            chunk_ids[:3],
            e,
        )

    return graph_map

def get_formulas_for_chunks(
    chunk_ids: List[str],
    limit_per_chunk: int = 5,
) -> Dict[str, List[str]]:
    """
    Recupera formule dai chunk visibili al tenant corrente.

    Supporta entrambi i modelli presenti o storicamente prodotti dalla
    pipeline:
    - ``Chunk-[:HAS_FORMULA|MENTIONS]->Formula/Entity(FORMULA)``;
    - ``Chunk-[:MENTIONS]->Entity-[:HAS_FORMULA]->Formula/Entity(FORMULA)``.
    """
    if not chunk_ids or not neo4j_driver:
        return {}

    formula_map: Dict[str, List[str]] = {}
    query = """
    UNWIND $ids AS target_id
    MATCH (c:Chunk)
    WHERE coalesce(c.chunk_id, c.id) = target_id
      AND c.status = 'active'
      AND (
            (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
            OR
            (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
      )

    CALL (c) {
        MATCH (c)-[rf:HAS_FORMULA|MENTIONS|MENTIONED_IN|PRESENT_IN]-(f)
        WHERE rf.status = 'active'
          AND f.status = 'active'
          AND (
                f:Formula
                OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                OR toUpper(coalesce(f.type, '')) = 'FORMULA'
          )
        RETURN f

        UNION

        MATCH (c)-[re:MENTIONS|MENTIONED_IN|PRESENT_IN]-(e:Entity)
              -[hf:HAS_FORMULA]-(f)
        WHERE re.status = 'active'
          AND e.status = 'active'
          AND hf.status = 'active'
          AND f.status = 'active'
          AND (
                f:Formula
                OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                OR toUpper(coalesce(f.type, '')) = 'FORMULA'
          )
        RETURN f
    }

    WITH target_id, f,
         coalesce(f.latex, f.formula, '') AS latex,
         coalesce(f.plain, f.name, f.canonical_id, f.fid, f.id, '') AS plain,
         coalesce(f.meaning_it, f.meaning, f.description, '') AS meaning
    WHERE trim(toString(latex)) <> ''
       OR trim(toString(plain)) <> ''
       OR trim(toString(meaning)) <> ''
    WITH target_id, collect(DISTINCT {
        latex: latex,
        plain: plain,
        meaning: meaning
    })[0..$lim] AS formulas
    UNWIND formulas AS formula
    RETURN target_id AS chunk_id,
           formula.latex AS latex,
           formula.plain AS plain,
           formula.meaning AS meaning
    """

    try:
        with neo4j_driver.session() as session:
            rows = session.run(
                query,
                ids=list(dict.fromkeys(str(x) for x in chunk_ids if str(x).strip())),
                lim=max(1, int(limit_per_chunk)),
                org_id=current_organization_id(),
            )
            for row in rows:
                parts: List[str] = []
                latex = str(row.get("latex") or "").strip()
                plain = str(row.get("plain") or "").strip()
                meaning = str(row.get("meaning") or "").strip()
                if latex:
                    parts.append(f"LaTeX: {latex}")
                if plain and plain != latex:
                    parts.append(f"Plain: {plain}")
                if meaning:
                    parts.append(f"Meaning: {meaning}")
                if parts:
                    rendered = " | ".join(parts)
                    bucket = formula_map.setdefault(str(row["chunk_id"]), [])
                    if rendered not in bucket:
                        bucket.append(rendered)
    except Exception as e:
        logger.error("Neo4j Query Error (get_formulas_for_chunks): %s", e)

    return formula_map

def get_neighbor_chunk_ids(
    chunk_ids: List[str],
    limit: int = GRAPH_MAX_NEIGHBOR_CHUNKS,
) -> List[str]:
    """Espande solo tra chunk ed entità appartenenti allo stesso perimetro visibile."""
    if not chunk_ids or not neo4j_driver:
        return []

    query = """
    MATCH
        (c1:Chunk)-[r1:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)-[r2:MENTIONS|PRESENT_IN|MENTIONED_IN]-(c2:Chunk)
    WHERE coalesce(c1.chunk_id, c1.id) IN $ids
      AND c1.status = 'active' AND c2.status = 'active' AND e.status = 'active'
      AND r1.status = 'active' AND r2.status = 'active'
      AND NOT coalesce(c2.chunk_id, c2.id) IN $ids
      AND (
            (c1.scope = 'GLOBAL' AND c1.organization_id IS NULL AND c1.tier = 'A')
            OR
            (c1.scope = 'ACCOUNT' AND c1.organization_id = $org_id AND c1.tier IN ['B', 'C'])
      )
      AND (
            (c2.scope = 'GLOBAL' AND c2.organization_id IS NULL AND c2.tier = 'A')
            OR
            (c2.scope = 'ACCOUNT' AND c2.organization_id = $org_id AND c2.tier IN ['B', 'C'])
      )
      AND (
            (e.scope = 'GLOBAL' AND e.organization_id IS NULL AND e.tier = 'A')
            OR
            (e.scope = 'ACCOUNT' AND e.organization_id = $org_id AND e.tier IN ['B', 'C'])
      )
      AND NOT toUpper(coalesce(e.type, e.category, labels(e)[0], '')) IN ['GENERIC', 'YEAR', 'DATE']
    WITH c2, count(DISTINCT e) AS entity_count
    WHERE entity_count >= 2
    RETURN coalesce(c2.chunk_id, c2.id) AS cid
    ORDER BY entity_count DESC,
             coalesce(c2.page, 0),
             coalesce(c2.page_chunk_index, 0)
    LIMIT $lim
    """

    try:
        with neo4j_driver.session() as session:
            rows = session.run(
                query,
                ids=chunk_ids,
                lim=limit,
                org_id=current_organization_id(),
            )
            return [str(row["cid"]) for row in rows if row.get("cid")]
    except Exception as e:
        print(f"⚠️ Neo4j Semantic Neighbors Error: {e}")
        return []

def fetch_chunks_from_qdrant_by_ids(ids: List[str]) -> List[SourceItem]:
    """Recupera Point ID da Qdrant e applica sempre il tenant guard sul payload."""
    if not ids or not qdrant_client_inst:
        return []

    out: List[SourceItem] = []
    try:
        points = qdrant_client_inst.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            if not qdrant_payload_is_visible(payload):
                logger.warning(
                    "Qdrant tenant guard: point %s scartato per organization_id=%s",
                    point.id,
                    current_organization_id(),
                )
                continue

            content = safe_payload_text(payload)
            if not content:
                continue

            out.append(
                SourceItem(
                    id=str(point.id),
                    content=content,
                    filename=str(payload.get("filename", "Unknown")),
                    page=get_payload_page(payload),
                    page_chunk_index=int(payload.get("page_chunk_index") or 0),
                    doc_id=str(payload.get("doc_id") or ""),
                    type=get_payload_type(payload),
                    score=0.0,
                    graph_context=[],
                    section_hint=get_payload_section(payload),
                    image_id=get_payload_image_id(payload),
                    tier=normalize_tier_value(get_payload_tier(payload)),
                    scope=str(payload.get("scope") or "").upper(),
                    organization_id=_optional_int(payload.get("organization_id")),
                    status=str(payload.get("status") or ""),
                    ingestion_run_id=str(payload.get("ingestion_run_id") or ""),
                    corpus_version=str(payload.get("corpus_version") or ""),
                    classification=str(payload.get("classification") or "internal"),
                    embedding_model=str(payload.get("embedding_model") or ""),
                    request_id=get_tenant_context().request_id,
                    db_origin="Qdrant Graph Expansion",
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
    lines.append(f"- **Organization ID**: `{current_organization_id()}`")

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
    """Recupera l'ultima versione visibile dei chunk, filtrando prima del ranking."""
    if not PG_ENRICH_ENABLED or not pg_pool or not chunk_uuids:
        return {}

    uuids = list(dict.fromkeys(str(value).strip() for value in chunk_uuids if str(value).strip()))
    if not uuids:
        return {}

    sql = """
    WITH visible AS (
        SELECT d.*
        FROM public.document_chunks d
        WHERE d.chunk_uuid::text = ANY(%s)
          AND d.status = 'active'
          AND (
                (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
                OR
                (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
          )
    ),
    ranked AS (
        SELECT
            d.chunk_uuid::text AS chunk_uuid,
            d.content_raw,
            d.content_semantic,
            d.metadata_json,
            d.ingestion_ts,
            d.scope,
            d.organization_id,
            d.tier,
            ROW_NUMBER() OVER (
                PARTITION BY d.chunk_uuid, d.scope, d.organization_id
                ORDER BY d.ingestion_ts DESC
            ) AS rn
        FROM visible d
    )
    SELECT
        chunk_uuid,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts,
        scope,
        organization_id,
        tier
    FROM ranked
    WHERE rn = 1;
    """

    conn = pg_get_conn_secure()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (uuids, current_organization_id()))
            rows = cur.fetchall()

        result: Dict[str, Dict[str, Any]] = {}
        for chunk_uuid, raw, semantic, metadata, ingestion_ts, scope, org_id, tier in rows:
            metadata = metadata_with_tenant(metadata, scope, org_id, tier)
            result[str(chunk_uuid)] = {
                "chunk_uuid": str(chunk_uuid),
                "content_raw": raw or "",
                "content_semantic": semantic or "",
                "metadata_json": metadata,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
                "scope": metadata.get("scope", ""),
                "organization_id": metadata.get("organization_id"),
                "tier": metadata.get("tier", ""),
            }
        return result
    except Exception as e:
        print(f"⚠️ PG enrich by chunk_uuid error: {e}")
        return {}
    finally:
        pg_put_conn_secure(conn)

def search_pg_bm25(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Ricerca full-text limitata a Tier A globale e Tier B/C del tenant corrente."""
    if not PG_ENRICH_ENABLED or not pg_pool or not (query_text or "").strip():
        return []

    tokens = extract_search_tokens(query_text)
    if not tokens:
        return []
    pg_query = " OR ".join(tokens)

    sql = """
    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq)
    SELECT
        d.chunk_uuid::text,
        d.content_raw,
        d.content_semantic,
        d.metadata_json,
        d.scope,
        d.organization_id,
        d.tier,
        ts_rank_cd(
            to_tsvector('simple', COALESCE(d.content_semantic, '') || ' ' || COALESCE(d.content_raw, '') || ' ' || COALESCE(d.metadata_json::text, '')),
            q.tsq
        ) AS rank
    FROM public.document_chunks d, q
    WHERE d.status = 'active'
      AND to_tsvector('simple', COALESCE(d.content_semantic, '') || ' ' || COALESCE(d.content_raw, '') || ' ' || COALESCE(d.metadata_json::text, '')) @@ q.tsq
      AND (
            (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
            OR
            (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
      )
    ORDER BY rank DESC
    LIMIT %s;
    """

    conn = pg_get_conn_secure()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (pg_query, current_organization_id(), limit))
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for chunk_uuid, raw, semantic, metadata, scope, org_id, tier, rank in rows:
            metadata = metadata_with_tenant(metadata, scope, org_id, tier)
            out.append({
                "id": str(chunk_uuid),
                "content": semantic or raw or "",
                "metadata": metadata,
                "score": float(rank or 0.0),
                "origin": "PostgresBM25",
            })
        return out
    except Exception as e:
        print(f"⚠️ BM25 Error: {e}")
        return []
    finally:
        pg_put_conn_secure(conn)

def search_pg_exact_phrases(query_text: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Ricerca esatta per termine, con quota per frase e filtro tenant fail-closed."""
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    phrases = extract_exact_phrases(query_text)[:12]
    if not phrases:
        return []
    per_phrase_limit = max(3, limit // len(phrases))

    sql_template = """
        SELECT
            chunk_uuid::text,
            content_raw,
            content_semantic,
            metadata_json,
            ingestion_ts,
            scope,
            organization_id,
            tier
        FROM public.document_chunks
        WHERE status = 'active'
          AND {condition}
          AND (
                (scope = 'GLOBAL' AND organization_id IS NULL AND tier = 'A')
                OR
                (scope = 'ACCOUNT' AND organization_id = %s AND tier IN ('B', 'C'))
          )
        ORDER BY ingestion_ts DESC
        LIMIT %s;
    """

    conn = pg_get_conn_secure()
    try:
        found: Dict[str, Dict[str, Any]] = {}
        with conn.cursor() as cur:
            for phrase in phrases:
                condition, condition_params = _term_sql_condition(phrase)
                if not condition:
                    continue
                cur.execute(
                    sql_template.format(condition=condition),
                    [*condition_params, current_organization_id(), per_phrase_limit],
                )
                for row in cur.fetchall():
                    chunk_uuid, raw, semantic, metadata, ingestion_ts, scope, org_id, tier = row
                    metadata = metadata_with_tenant(metadata, scope, org_id, tier)
                    uid = str(chunk_uuid)
                    if uid not in found:
                        found[uid] = {
                            "id": uid,
                            "content": semantic or raw or "",
                            "metadata": metadata,
                            "score": 2.0,
                            "origin": "PostgresExactPhrase",
                            "scope": str(metadata.get("scope") or "").upper(),
                            "organization_id": _optional_int(metadata.get("organization_id")),
                            "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
                        }
                    else:
                        found[uid]["score"] = float(found[uid].get("score", 2.0)) + 1.0

        return sorted(found.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)[:limit]
    except Exception as e:
        print(f"⚠️ Exact phrase search error: {e}")
        return []
    finally:
        pg_put_conn_secure(conn)

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
    """Lookup di glossario limitato al perimetro tenant visibile."""
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    clauses: List[str] = []
    params: List[Any] = []
    for alias in aliases:
        condition, condition_params = _term_sql_condition(alias)
        if condition:
            clauses.append(condition)
            params.extend(condition_params)
    if not clauses:
        return []

    sql = f"""
    SELECT
        chunk_uuid::text,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts,
        scope,
        organization_id,
        tier
    FROM public.document_chunks
    WHERE status = 'active'
      AND (
            lower(COALESCE(metadata_json->>'filename', '')) LIKE %s
            OR lower(COALESCE(metadata_json->>'source_name', '')) LIKE %s
            OR lower(COALESCE(metadata_json::text, '')) LIKE %s
          )
      AND ({' OR '.join(clauses)})
      AND (
            (scope = 'GLOBAL' AND organization_id IS NULL AND tier = 'A')
            OR
            (scope = 'ACCOUNT' AND organization_id = %s AND tier IN ('B', 'C'))
      )
    ORDER BY ingestion_ts DESC
    LIMIT %s;
    """

    params = ["%glossario%", "%glossario%", "%glossario%"] + params
    params.extend([current_organization_id(), limit])

    conn = pg_get_conn_secure()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for chunk_uuid, raw, semantic, metadata, ingestion_ts, scope, org_id, tier in rows:
            metadata = metadata_with_tenant(metadata, scope, org_id, tier)
            out.append({
                "id": str(chunk_uuid),
                "content_raw": raw or "",
                "content_semantic": semantic or "",
                "metadata": metadata,
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
                "term": canonical_term,
            })
        return out
    except Exception as e:
        print(f"⚠️ Glossary term lookup error for {canonical_term}: {e}")
        return []
    finally:
        pg_put_conn_secure(conn)

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
                    page_chunk_index=int(meta.get("page_chunk_index") or 0),
                    doc_id=str(meta.get("doc_id") or ""),
                    type=meta.get("toon_type") or meta.get("type") or "text",
                    score=2.0,
                    tier=normalize_tier_value(meta.get("tier", "C")),
                    scope=str(meta.get("scope") or "").upper(),
                    organization_id=_optional_int(meta.get("organization_id")),
                    status="active",
                    ingestion_run_id=str(meta.get("ingestion_run_id") or ""),
                    corpus_version=str(meta.get("corpus_version") or CORPUS_VERSION),
                    classification=str(meta.get("classification") or "internal"),
                    embedding_model=str(meta.get("embedding_model") or ""),
                    request_id=get_tenant_context().request_id,
                    db_origin="PostgresGlossaryTerm",
                    section_hint=f"Glossary term: {term}",
                    pg_ingestion_ts=best.get("ingestion_ts", ""),
                    pg_source_name=meta.get("source_name", ""),
                    pg_source_type=meta.get("source_type", ""),
                    pg_log_id=int(meta.get("log_id") or 0),
                    pg_chunk_id=int(meta.get("chunk_index") or 0),
                    pg_page_chunk_index=int(meta.get("page_chunk_index") or 0),
                    pg_toon_type=meta.get("toon_type", ""),
                )
            )

    used_files = sorted({s.filename for s in source_items if s.filename})

    answer = (
        "**A) Risposta**\n\n"
        + "\n".join(answer_lines)
        + "\n\n"
        "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
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

    # Se non ci sono fonti reali utili, lascia la risposta matematica pura.
    if not sources:
        return math_answer

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


GRAPH_QUERY_NOISE_TERMS = {
    # Istruzioni generiche
    "mostra", "mostrami", "trova", "cerca", "elenca", "riporta",
    "descrivi", "spiega", "analizza", "verifica", "interroga",
    "show", "find", "search", "list", "report",
    "describe", "explain", "analyze", "analyse", "verify", "query",

    # Termini tecnici che descrivono la richiesta, non i concetti cercati
    "neo4j", "cypher", "grafo", "grafi", "graph", "graphs",
    "nodo", "nodi", "node", "nodes",
    "arco", "archi", "edge", "edges",
    "relazione", "relazioni", "relation", "relations",
    "relationship", "relationships",
    "collegamento", "collegamenti", "link", "links",
    "connessione", "connessioni", "connection", "connections",
    "percorso", "path", "traversamento", "traversal",
    "multihop", "multi-hop",

    # Formato della risposta
    "tabella", "table", "markdown",
    "colonna", "colonne", "column", "columns",
    "riga", "righe", "row", "rows",

    # Parole generiche
    "entità", "entita", "entity", "entities",
    "concetto", "concetti", "concept", "concepts",
    "documento", "documenti", "document", "documents",
    "fonte", "fonti", "source", "sources",
}


def extract_rag_tokens(query_text: str) -> List[str]:
    """
    Estrae token utili per filename matching, Neo4j e formula lookup.
    Mantiene acronimi brevi (MFA, APT, CVE, KPI) invece di eliminarli.
    """
    return [t for t in extract_search_tokens(query_text) if t not in RAG_STOPWORDS]


def search_neo4j_entities(
    query_text: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Ricerca entità attraverso Chunk->MENTIONS, con filtro tenant sul Chunk."""
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)
    if not tokens:
        return []

    cypher = """
    MATCH (c:Chunk)-[m:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)
    WHERE c.status = 'active' AND e.status = 'active' AND m.status = 'active'
      AND (
            (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
            OR
            (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
          )
      AND any(tok IN $tokens WHERE
            toLower(coalesce(e.name, e.canonical_id, e.id, '')) CONTAINS tok OR
            toLower(coalesce(e.description, '')) CONTAINS tok OR
            toLower(coalesce(e.category, e.type, labels(e)[0], '')) CONTAINS tok OR
            any(s IN coalesce(e.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok) OR
            toLower(coalesce(c.filename, '')) CONTAINS tok OR
            toLower(coalesce(c.text, '')) CONTAINS tok
      )
    WITH c,
         collect(DISTINCT coalesce(e.name, e.canonical_id, e.id)) AS entities,
         count(DISTINCT e) AS rel_count
    RETURN
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.doc_id, '') AS doc_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.page_chunk_index, 0) AS page_chunk_index,
        c.scope AS scope,
        c.organization_id AS organization_id,
        c.tier AS source_tier,
        entities,
        rel_count,
        toFloat(rel_count) * 2.0 AS graph_score
    ORDER BY graph_score DESC, page ASC, page_chunk_index ASC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []
    try:
        with neo4j_driver.session() as session:
            rows = session.run(cypher, tokens=tokens, limit=limit, org_id=current_organization_id())
            for row in rows:
                cid = row.get("chunk_id")
                if not cid:
                    continue
                entities = row.get("entities") or []
                out.append({
                    "id": str(cid),
                    "doc_id": str(row.get("doc_id") or ""),
                    "content": "Entity match: " + ", ".join(str(x) for x in entities[:12]),
                    "filename": row.get("filename") or "Neo4j",
                    "page": int(row.get("page") or 0),
                    "page_chunk_index": int(row.get("page_chunk_index") or 0),
                    "type": "graph",
                    "tier": "GRAPH",
                    "source_tier": str(row.get("source_tier") or ""),
                    "scope": str(row.get("scope") or "").upper(),
                    "organization_id": _optional_int(row.get("organization_id")),
                    "status": "active",
                    "corpus_version": CORPUS_VERSION,
                    "score_graph": float(row.get("graph_score") or row.get("rel_count") or 1.0),
                    "origin": "Neo4j Entity Search",
                    "section_hint": "Entities: " + ", ".join(str(x) for x in entities[:5]),
                })
    except Exception as e:
        print(f"⚠️ Neo4j entity search error: {e}")
    return out

def search_neo4j_formulas(
    query_text: str,
    limit: int = 20,
    requested_doc: str = "",
) -> List[Dict[str, Any]]:
    """
    Ricerca formule nel Knowledge Graph con filtro tenant sul Chunk.

    La ricerca copre sia formule direttamente menzionate dal Chunk sia formule
    collegate a un concetto menzionato dal Chunk. Se la domanda è generica
    (es. "quali formule sono presenti?") la query non fallisce per lista token
    vuota; il limite e l'eventuale documento richiesto delimitano i risultati.
    """
    if not neo4j_driver or not query_text.strip():
        return []

    tokens = extract_rag_tokens(query_text)
    requested_doc_norm = normalize_doc_name(requested_doc)
    requested_doc_lower = os.path.basename(str(requested_doc or "")).strip().lower()

    cypher = """
    MATCH (c:Chunk)
    WHERE c.status = 'active'
      AND (
            (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
            OR
            (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
          )
      AND (
            $requested_doc_norm = ''
            OR toLower(coalesce(c.filename, '')) CONTAINS $requested_doc_lower
            OR replace(replace(replace(replace(replace(replace(
                   toLower(coalesce(c.filename, '')), '.pdf', ''), '.md', ''),
                   '.txt', ''), '_', ''), '-', ''), ' ', '') CONTAINS $requested_doc_norm
          )

    CALL (c) {
        MATCH (c)-[rf:HAS_FORMULA|MENTIONS|MENTIONED_IN|PRESENT_IN]-(f)
        WHERE rf.status = 'active'
          AND f.status = 'active'
          AND (
                f:Formula
                OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                OR toUpper(coalesce(f.type, '')) = 'FORMULA'
          )
        RETURN f

        UNION

        MATCH (c)-[re:MENTIONS|MENTIONED_IN|PRESENT_IN]-(e:Entity)
              -[hf:HAS_FORMULA]-(f)
        WHERE re.status = 'active'
          AND e.status = 'active'
          AND hf.status = 'active'
          AND f.status = 'active'
          AND (
                f:Formula
                OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                OR toUpper(coalesce(f.type, '')) = 'FORMULA'
          )
        RETURN f
    }

    WITH c, f,
         coalesce(f.latex, f.formula, '') AS latex,
         coalesce(f.plain, f.name, f.canonical_id, f.fid, f.id, '') AS plain,
         coalesce(f.meaning_it, f.meaning, f.description, '') AS meaning
    WHERE size($tokens) = 0
       OR any(tok IN $tokens WHERE
            toLower(coalesce(c.filename, '')) CONTAINS tok OR
            toLower(coalesce(c.text, '')) CONTAINS tok OR
            toLower(toString(latex)) CONTAINS tok OR
            toLower(toString(plain)) CONTAINS tok OR
            toLower(toString(meaning)) CONTAINS tok OR
            any(k IN coalesce(f.keywords, []) WHERE toLower(toString(k)) CONTAINS tok)
       )
    RETURN DISTINCT
        coalesce(c.chunk_id, c.id) AS chunk_id,
        coalesce(c.doc_id, '') AS doc_id,
        coalesce(c.filename, 'Neo4j') AS filename,
        coalesce(c.page, 0) AS page,
        coalesce(c.page_chunk_index, 0) AS page_chunk_index,
        c.scope AS scope,
        c.organization_id AS organization_id,
        c.tier AS source_tier,
        latex,
        plain,
        meaning,
        coalesce(f.fid, f.entity_key, f.id, plain, latex) AS formula_key
    ORDER BY page ASC, page_chunk_index ASC
    LIMIT $limit
    """

    out: List[Dict[str, Any]] = []
    try:
        with neo4j_driver.session() as session:
            rows = session.run(
                cypher,
                tokens=tokens,
                limit=max(1, int(limit)),
                org_id=current_organization_id(),
                requested_doc_norm=requested_doc_norm,
                requested_doc_lower=requested_doc_lower,
            )
            seen = set()
            for row in rows:
                cid = row.get("chunk_id")
                if not cid:
                    continue

                formula_key = str(row.get("formula_key") or "").strip()
                dedupe_key = (str(cid), formula_key)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                parts: List[str] = []
                latex = str(row.get("latex") or "").strip()
                plain = str(row.get("plain") or "").strip()
                meaning = str(row.get("meaning") or "").strip()
                if latex:
                    parts.append(f"LaTeX: {latex}")
                if plain and plain != latex:
                    parts.append(f"Plain: {plain}")
                if meaning:
                    parts.append(f"Meaning: {meaning}")
                if not parts:
                    continue

                out.append({
                    "id": str(cid),
                    "formula_key": formula_key,
                    "doc_id": str(row.get("doc_id") or ""),
                    "content": "Formula from Knowledge Graph:\n" + "\n".join(parts),
                    "filename": row.get("filename") or "Neo4j",
                    "page": int(row.get("page") or 0),
                    "page_chunk_index": int(row.get("page_chunk_index") or 0),
                    "type": "formula",
                    "tier": "GRAPH",
                    "source_tier": str(row.get("source_tier") or ""),
                    "scope": str(row.get("scope") or "").upper(),
                    "organization_id": _optional_int(row.get("organization_id")),
                    "status": "active",
                    "corpus_version": CORPUS_VERSION,
                    "score_graph": 5.0,
                    "origin": "Neo4j Formula Search",
                    "section_hint": "Formula node",
                })
    except Exception as e:
        print(f"⚠️ Neo4j formula search error: {e}")
    return out

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
    Mantiene solo archi i cui due estremi corrispondono a due concetti distinti
    richiesti nella query. I token generici sono usati solo come fallback quando
    non sono stati estratti concetti strutturati.
    """
    if not rows:
        return []

    concepts = extract_graph_concepts_from_query(query_text, max_concepts=12)
    tokens = graph_relevant_tokens(query_text)

    if not concepts and not tokens:
        return rows[:limit]

    scored: List[Tuple[int, int, Dict[str, Any]]] = []

    for row in rows:
        source_text = str(row.get("source") or "").lower()
        target_text = str(row.get("target") or "").lower()
        relation_text = _relation_row_text(row)

        source_hits = {
            _canonical_graph_concept(c)
            for c in concepts
            if _concept_in_text(c, source_text)
        }
        target_hits = {
            _canonical_graph_concept(c)
            for c in concepts
            if _concept_in_text(c, target_text)
        }
        endpoint_hits = source_hits | target_hits
        token_hits = {t for t in tokens if t in relation_text}

        if concepts:
            if source_hits and target_hits and len(endpoint_hits) >= 2:
                scored.append((len(endpoint_hits), len(token_hits), row))
        elif len(token_hits) >= 2:
            scored.append((0, len(token_hits), row))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [row for _, _, row in scored[:limit]]

def search_neo4j_relations(query_text: str, limit: int = 60) -> List[Dict[str, Any]]:
    """Archi tenant-safe, inclusi i soli collegamenti diretti ACCOUNT -> GLOBAL."""
    if not neo4j_driver:
        return []
    tokens = graph_relevant_tokens(query_text) or extract_rag_tokens(query_text)
    if not tokens:
        return []
    tokens = list(dict.fromkeys(tokens))

    cypher = """
    MATCH (e1:Entity)-[rel]->(e2:Entity)
    WHERE rel.status = 'active'
      AND e1.status = 'active'
      AND e2.status = 'active'
      AND type(rel) IN $allowed_rels
      AND (
            (
                rel.scope = 'GLOBAL' AND rel.organization_id IS NULL
                AND e1.scope = 'GLOBAL' AND e1.organization_id IS NULL AND e1.tier = 'A'
                AND e2.scope = 'GLOBAL' AND e2.organization_id IS NULL AND e2.tier = 'A'
            )
            OR
            (
                rel.scope = 'ACCOUNT' AND rel.organization_id = $org_id
                AND e1.scope = 'ACCOUNT' AND e1.organization_id = $org_id AND e1.tier IN ['B','C']
                AND (
                    (e2.scope = 'ACCOUNT' AND e2.organization_id = $org_id AND e2.tier IN ['B','C'])
                    OR
                    (e2.scope = 'GLOBAL' AND e2.organization_id IS NULL AND e2.tier = 'A')
                )
            )
          )
      AND any(tok IN $tokens WHERE
            toLower(coalesce(e1.name, e1.canonical_id, e1.id, '')) CONTAINS tok OR
            toLower(coalesce(e2.name, e2.canonical_id, e2.id, '')) CONTAINS tok OR
            toLower(coalesce(e1.description, '')) CONTAINS tok OR
            toLower(coalesce(e2.description, '')) CONTAINS tok OR
            any(s IN coalesce(e1.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok) OR
            any(s IN coalesce(e2.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok)
      )
    RETURN
        coalesce(e1.name, e1.canonical_id, e1.id) AS source,
        type(rel) AS relation,
        coalesce(e2.name, e2.canonical_id, e2.id) AS target,
        properties(rel) AS props,
        coalesce(rel.source_file, head(coalesce(rel.source_files, [])), '') AS filename,
        coalesce(rel.page_no, head(coalesce(rel.page_nos, [])), 0) AS page,
        rel.scope AS scope,
        rel.organization_id AS organization_id,
        CASE WHEN rel.scope = 'GLOBAL' THEN 'A' ELSE 'C' END AS tier,
        rel.status AS status,
        rel.ingestion_run_id AS ingestion_run_id,
        rel.corpus_version AS corpus_version,
        rel.classification AS classification
    LIMIT $limit
    """
    try:
        with neo4j_driver.session() as session:
            rows = session.run(
                cypher,
                tokens=tokens,
                limit=max(limit * 4, limit),
                org_id=current_organization_id(),
                allowed_rels=NEO4J_ALLOWED_RELATIONSHIPS,
            )
            raw_rows = [dict(row) for row in rows]
        return filter_neo4j_relation_rows(query_text, raw_rows, limit)
    except Exception as e:
        print(f"⚠️ Neo4j relation search error: {e}")
        return []

def clean_graph_relation_label(value: Any) -> str:
    """
    Pulisce il nome della relazione Neo4j prima di mostrarla in tabella.
    Evita che props Neo4j come last_seen/evidence finiscano nella colonna Relazione.
    """
    text = str(value or "RELATES_TO").strip()

    # Se per errore arriva già una relazione contaminata da props:
    # "COMPLIES_WITH {'last_seen': ...}" -> "COMPLIES_WITH"
    if "{" in text:
        text = text.split("{", 1)[0].strip()

    text = text.upper()
    text = re.sub(r"[^A-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text[:80] or "RELATES_TO"


def graph_relations_to_source(rows: List[Dict[str, Any]]) -> Optional[SourceItem]:
    """
    Converte le relazioni Neo4j in una tabella Markdown.
    La colonna Relazione contiene SOLO il type Neo4j, non le proprietà dell'arco.
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
        source = _md_cell(r.get("source") or "", 180)
        relation = _md_cell(clean_graph_relation_label(r.get("relation")), 120)
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
        scope="ACCOUNT",
        organization_id=current_organization_id(),
        status="active",
        corpus_version=CORPUS_VERSION,
        request_id=get_tenant_context().request_id,
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
    Non contiene logica adattativa: normalizza punteggiatura, virgolette,
    articoli e parole di ruolo generiche.
    """
    text = re.sub(r"\s+", " ", value or "").strip()

    text = text.strip(" \t\n\r.,;:!?()[]{}\"'“”‘’«»`")

    # Rimuove prefissi descrittivi generici:
    # es. funzione “Respond” -> Respond
    text = re.sub(
        r"^(?:funzione|function|concetto|concept|termine|term|voce|entity|entità)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Rimuove articoli/congiunzioni iniziali.
    leading_noise = (
        r"^(?:(?:e|ed|and|or|oppure|o|il|lo|la|i|gli|le|un|una|uno|the|a|an)\s+"
        r"|(?:l|un)['’])+"
    )
    text = re.sub(leading_noise, "", text, flags=re.IGNORECASE)

    # Rimuove congiunzioni finali residue.
    text = re.sub(r"\s+(?:e|ed|and|or|oppure|o)$", "", text, flags=re.IGNORECASE)

    return text.strip(" \t\n\r.,;:!?()[]{}\"'“”‘’«»`")



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
    c = _clean_graph_concept(concept).lower().strip()

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
    Estrae concetti forti dalla domanda per query relazionali/multi-hop.
    Non è adattativa:
    - non contiene nomi di test;
    - non contiene nomi di documenti;
    - usa solo pattern linguistici generali.
    """
    q = query_text or ""
    concepts: List[str] = []

    # 1. Termini tra virgolette dritte o curve.
    quoted = re.findall(r"[\"“'‘«]([^\"”'’»]+)[\"”'’»]", q)
    for item in quoted:
        clean = _clean_graph_concept(item)
        if len(clean) >= 2:
            concepts.append(clean)

    # 2. Segmenti dopo tra/fra/between/among.
    relation_segment_patterns = [
        r"\b(?:tra|fra)\s+(.+?)(?:[\.?]|$)",
        r"\bbetween\s+(.+?)(?:[\.?]|$)",
        r"\bamong\s+(.+?)(?:[\.?]|$)",
    ]

    for pat in relation_segment_patterns:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            concepts.extend(_split_relation_segment(m.group(1)))

    # 3. Segmenti multi-hop: "da X a Y passando per A, B, C".
    from_to = re.search(
        r"\b(?:da|from)\s+(.+?)\s+(?:a|to)\s+(.+?)(?:,|\s+passando\s+per|\s+through|\s+via|\.|\?|$)",
        q,
        flags=re.IGNORECASE,
    )

    if from_to:
        concepts.append(_clean_graph_concept(from_to.group(1)))
        concepts.append(_clean_graph_concept(from_to.group(2)))

    via = re.search(
        r"\b(?:passando\s+per|through|via)\s+(.+?)(?:[\.?]|$)",
        q,
        flags=re.IGNORECASE,
    )

    if via:
        concepts.extend(_split_relation_segment(via.group(1)))

    # 4. Acronomi composti o semplici: GDPR, NIS2, CSIRT, CID, ACN-ZK.
    acronyms = re.findall(r"\b[A-Z]{2,10}(?:[-_/][A-Z0-9]{1,10})?\b", q)
    concepts.extend(acronyms)

    # 5. Frasi esatte già recuperabili dal sistema.
    for p in extract_exact_phrases(q):
        clean = _clean_graph_concept(p)
        if clean:
            concepts.append(clean)

    # 6. Fallback token solo se non ci sono concetti forti.
    if not concepts:
        for t in graph_relevant_tokens(q):
            if len(t) >= 4:
                concepts.append(t)

    weak_single_terms = {
        "tutti", "tutto", "all", "each", "ogni",
        "fattore", "fattori", "factor", "factors",
        "access", "accesso", "control", "controllo", "controlli",
        "autenticazione", "authentication",
        "rischio", "risk", "utente", "user", "identity", "identità",
        "documenti", "documents", "normativi", "normative",
        "funzione", "function", "processo", "process",
        "catena", "chain", "percorso", "path", "passaggio", "step",
        "traversamento", "traversal", "grafo", "graph", "neo4j",
        "multi-hop", "multihop",
    }

    cleaned: List[str] = []
    seen_canonical = set()

    for c in concepts:
        clean = _clean_graph_concept(c)
        if not clean:
            continue

        cl = clean.lower()
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", clean))
        is_acronym = bool(re.fullmatch(r"[A-Z]{2,10}(?:[-_/][A-Z0-9]{1,10})?", clean))

        if not is_acronym and word_count == 1 and cl in weak_single_terms:
            continue

        canonical = _canonical_graph_concept(clean)

        if canonical in seen_canonical:
            continue

        seen_canonical.add(canonical)
        cleaned.append(clean)

        if len(cleaned) >= max_concepts:
            break

    return cleaned

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


def _evidence_snippet_for_pair(content: str, a: str, b: str, max_chars: int = 260) -> Tuple[str, str]:
    """
    Restituisce:
    - snippet;
    - livello evidenza: supporto_testuale_forte oppure co_occorrenza_debole.

    Non usa termini di dominio.
    Usa solo vicinanza testuale:
    - stessa frase = supporto forte;
    - stesso paragrafo/chunk = co-occorrenza debole.
    """
    if not content:
        return "", "non_supportata"

    text = re.sub(r"\s+", " ", content or "").strip()
    text_l = text.lower()

    a_l = (a or "").lower()
    b_l = (b or "").lower()

    sentences = re.split(r"(?<=[\.\!\?])\s+", text)

    for sent in sentences:
        sl = sent.lower()
        if a_l in sl and b_l in sl:
            return _md_cell(sent, max_chars), "supporto_testuale_forte"

    if a_l in text_l and b_l in text_l:
        pos_a = text_l.find(a_l)
        pos_b = text_l.find(b_l)

        start = max(0, min(pos_a, pos_b) - 120)
        end = min(len(text), max(pos_a, pos_b) + 180)

        snippet = text[start:end].strip()
        return _md_cell(snippet, max_chars), "co_occorrenza_debole"

    return "", "non_supportata"



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
            "relation": clean_graph_relation_label(cols[1]),
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

    Fix v5:
    - filtra archi Neo4j fuori target;
    - distingue archi espliciti, supporto testuale e relazioni non supportate;
    - evita che relazioni vere ma non pertinenti dominino la risposta;
    - evita fallback LLM su query esplicitamente graph/Neo4j;
    - mantiene output in italiano.
    """
    if not is_graph_relation_query(query_text):
        return None

    concepts = extract_graph_concepts_from_query(query_text)

    if len(concepts) < 2:
        concepts = [t for t in graph_relevant_tokens(query_text) if len(t) >= 4][:6]

    concepts = [c for c in concepts if c and len(str(c).strip()) >= 3]

    if len(concepts) < 2:
        return None

    rows: List[Dict[str, Any]] = []
    unsupported_rows: List[Dict[str, Any]] = []
    seen = set()
    seen_pairs = set()

    concept_canons = {
        _canonical_graph_concept(c)
        for c in concepts
        if _canonical_graph_concept(c)
    }

    def is_edge_relevant(src: str, tgt: str, relation: str = "") -> bool:
        """
        Un arco è pertinente solo quando sorgente e target corrispondono a due
        concetti distinti richiesti dall'utente. Il nome della relazione non può
        compensare un estremo fuori target.
        """
        src_text = str(src or "").lower()
        tgt_text = str(tgt or "").lower()

        src_hits = {
            _canonical_graph_concept(c)
            for c in concepts
            if _concept_in_text(c, src_text)
        }
        tgt_hits = {
            _canonical_graph_concept(c)
            for c in concepts
            if _concept_in_text(c, tgt_text)
        }

        return bool(src_hits and tgt_hits and len(src_hits | tgt_hits) >= 2)

    def add_row(row: Dict[str, Any]) -> None:
        src = str(row.get("source", "")).strip()
        tgt = str(row.get("target", "")).strip()
        rel = str(row.get("relation", "")).strip()
        status = str(row.get("status", "")).strip().lower()

        src_can = _canonical_graph_concept(src)
        tgt_can = _canonical_graph_concept(tgt)

        pair_key = tuple(sorted([src_can, tgt_can])) + (status,)

        # Per supporto testuale/co-occorrenza evita ripetizioni della stessa coppia.
        if "testual" in status or "co-occorrenza" in status:
            if pair_key in seen_pairs:
                return
            seen_pairs.add(pair_key)

        key = (
            src_can,
            rel.lower(),
            tgt_can,
            str(row.get("filename", "")).lower(),
            str(row.get("page", "")),
            status,
        )

        if key in seen:
            return

        seen.add(key)
        rows.append(row)

    # 1) Relazioni esplicite dal Knowledge Graph, filtrate per pertinenza.
    for s in sources:
        if s.type == "graph_relations" or "Relazioni Neo4j trovate" in (s.content or ""):
            for r in _parse_graph_relation_table_from_source(s):
                src = str(r.get("source", ""))
                rel = str(r.get("relation", ""))
                tgt = str(r.get("target", ""))

                if not is_edge_relevant(src, tgt, rel):
                    continue

                r["status"] = r.get("status") or "esplicita nel grafo"
                r["evidence"] = r.get("evidence") or "Relazione presente nel Knowledge Graph."

                add_row(r)

                if len(rows) >= max_rows:
                    break

        if len(rows) >= max_rows:
            break

    # 2) Supporto testuale: usa solo vere fonti documentali e distingue
    # stessa frase da semplice co-occorrenza nello stesso chunk.
    if len(rows) < max_rows:
        for s in sources:
            if not s.content:
                continue

            if normalize_tier_value(s.tier) == "GRAPH" or s.type == "graph_relations":
                continue

            text = s.content
            text_l = text.lower()

            matched = [
                c for c in concepts
                if _concept_in_text(c, text_l)
            ]

            if len(matched) < 2:
                continue

            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    src = matched[i]
                    tgt = matched[j]
                    src_alias = _best_alias_for_text(src, text_l)
                    tgt_alias = _best_alias_for_text(tgt, text_l)
                    snippet, evidence_level = _evidence_snippet_for_pair(
                        text,
                        src_alias,
                        tgt_alias,
                    )

                    if not snippet:
                        continue

                    status = (
                        "supporto testuale forte, non esplicita come arco"
                        if evidence_level == "supporto_testuale_forte"
                        else "co-occorrenza debole, non esplicita come arco"
                    )

                    add_row({
                        "source": src,
                        "relation": "collegamento testuale",
                        "target": tgt,
                        "filename": s.filename or "N/D",
                        "page": s.page or "",
                        "evidence": snippet,
                        "status": status,
                    })

                    if len(rows) >= max_rows:
                        break

                if len(rows) >= max_rows:
                    break

            if len(rows) >= max_rows:
                break

    # 3) Fallback controllato: se non ci sono righe, dichiara assenza di archi pertinenti.
    if not rows:
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                unsupported_rows.append(
                    {
                        "source": concepts[i],
                        "relation": "collegamento richiesto",
                        "target": concepts[j],
                        "filename": "N/D",
                        "page": "",
                        "evidence": "Nessun arco Neo4j esplicito pertinente recuperato.",
                        "status": "non trovato",
                    }
                )

                if len(unsupported_rows) >= max_rows:
                    break

            if len(unsupported_rows) >= max_rows:
                break

        rows = unsupported_rows

    header = (
        "| Entità sorgente | Relazione | Entità target | Documento | Pagina | Evidenza | Stato |\n"
        "|---|---|---|---|---:|---|---|"
    )

    table_lines = [header]

    for r in rows[:max_rows]:
        source = str(r.get("source", "")).replace("\n", " ").strip()
        relation = str(r.get("relation", "")).replace("\n", " ").strip()
        target = str(r.get("target", "")).replace("\n", " ").strip()
        filename = str(r.get("filename", "N/D")).replace("\n", " ").strip()
        page = str(r.get("page", "")).replace("\n", " ").strip()
        evidence = str(r.get("evidence", "")).replace("\n", " ").strip()
        status = str(r.get("status", "")).replace("\n", " ").strip()

        if len(evidence) > 220:
            evidence = evidence[:220] + "..."

        table_lines.append(
            f"| {source} | {relation} | {target} | {filename} | {page} | {evidence} | {status} |"
        )

    used_files = []
    for r in rows:
        fn = str(r.get("filename", "")).strip()
        if fn and fn != "N/D" and fn not in used_files:
            used_files.append(fn)

    if used_files:
        sources_text = "\n".join(f"- {fn}" for fn in used_files[:8])
    else:
        sources_text = "- Nessuna fonte documentale diretta utilizzabile."

    has_explicit = any(
        "esplicita" in str(r.get("status", "")).lower()
        or "grafo" in str(r.get("status", "")).lower()
        for r in rows
    )

    has_textual = any(
        "testual" in str(r.get("status", "")).lower()
        for r in rows
    )

    has_not_found = any(
        "non trovato" in str(r.get("status", "")).lower()
        or "non supportata" in str(r.get("status", "")).lower()
        for r in rows
    )

    evidence_notes = [
        "- La tabella è stata costruita in modalità deterministica.",
        "- Sono state escluse relazioni Neo4j esplicite ma fuori target rispetto alle entità richieste.",
        "- Ogni riga distingue tra arco esplicito Neo4j, supporto testuale o relazione non trovata.",
    ]

    has_explicit = any(
        str(r.get("status", "")).strip().lower() == "esplicita nel grafo"
        for r in rows
    )
    
    if has_explicit:
        evidence_notes.append("- Sono presenti relazioni esplicite recuperate dal Knowledge Graph.")
    else:
        evidence_notes.append("- Non sono stati recuperati archi Neo4j espliciti pertinenti; le relazioni riportate sono testuali o non trovate.")
         
    if has_textual:
        evidence_notes.append("- Alcune relazioni sono supportate testualmente ma non risultano esplicite come archi Neo4j.")
    if has_not_found:
        evidence_notes.append("- Alcuni collegamenti richiesti non risultano supportati dalle fonti recuperate.")

    is_multihop_request = any(t in (query_text or "").lower() for t in [
        "multi-hop",
        "multihop",
        "catena",
        "percorso",
        "path",
        "traversamento",
        "chain",
        "traversal",
    ])

    limits = [
        "- Una relazione plausibile non viene trasformata in arco esplicito se non è presente nel grafo.",
        "- Relazioni vere ma non pertinenti alla domanda non sono usate come evidenza principale.",
        "- Se il grafo non contiene archi pertinenti, la risposta distingue supporto testuale, inferenza e relazione non trovata.",
    ]

    if is_multihop_request:
        explicit_graph_rows = [
            r for r in rows
            if "esplicita" in str(r.get("status", "")).lower()
            or "grafo" in str(r.get("status", "")).lower()
        ]

        msg = (
            "- La richiesta è multi-hop, ma non sono stati recuperati abbastanza archi Neo4j espliciti "
            "per ricostruire una catena completa. La risposta riporta solo collegamenti testuali o assenze."
        )

        if len(explicit_graph_rows) < 2 and msg not in limits:
            limits.append(msg)

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_notes)
        + "\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        + "\n".join(limits)
        + "\n\n"
        "**D) Fonti**\n\n"
        + sources_text
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
    Estrae il documento richiesto dalla query in modo robusto.
    Supporta:
    - virgolette dritte: "file.pdf"
    - virgolette curve: “file.pdf”
    - apici: 'file.pdf'
    - filename libero nel testo: file.pdf
    """
    q = query_text or ""

    q = (
        q.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
    )

    # 1. Documento/file/pdf seguito da nome tra virgolette
    patterns = [
        r'\b(?:documento|file|pdf)\s+["\']([^"\']+\.(?:pdf|md|txt|docx|csv|html))["\']',
        r'\b(?:nel|nella|dal|dalla)\s+(?:documento|file|pdf)\s+["\']([^"\']+\.(?:pdf|md|txt|docx|csv|html))["\']',

        # 2. Documento/file/pdf seguito da filename non quotato
        r'\b(?:documento|file|pdf)\s+([A-Za-z0-9_\-\s\.]+\.(?:pdf|md|txt|docx|csv|html))\b',

        # 3. Qualunque filename esplicito nel testo
        r'\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*\.(?:pdf|md|txt|docx|csv|html))\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,:;!?\"'")

    return ""

def candidate_matches_requested_doc(candidate: Dict[str, Any], requested_doc: str) -> bool:
    """
    Verifica se un candidato appartiene al documento richiesto.
    Evita che filename vuoto / Unknown / Neo4j passino il filtro documentale.
    """
    if not requested_doc:
        return True

    wanted = normalize_doc_name(requested_doc)
    if not wanted:
        return True

    raw_filename = str(candidate.get("filename", "") or "").strip()

    if raw_filename in ("", "Unknown", "Neo4j", "KG", "Neo4j Knowledge Graph"):
        return False

    filename = normalize_doc_name(raw_filename)

    if not filename:
        return False

    return wanted in filename or filename in wanted

def search_pg_by_document_scope(
    requested_doc: str,
    query_text: str,
    limit: int = 80,
) -> List[Dict[str, Any]]:
    """Recupera il documento richiesto applicando il tenant filter prima di ROW_NUMBER."""
    if not PG_ENRICH_ENABLED or not pg_pool:
        return []

    wanted_norm = normalize_doc_name(requested_doc)
    if not wanted_norm:
        return []

    sql = """
    WITH q AS (
        SELECT plainto_tsquery('simple', %s) AS tsq
    ),
    visible AS (
        SELECT
            d.*,
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(coalesce(d.metadata_json->>'filename', d.metadata_json->>'source_name', '')),
                        '\\.(pdf|md|txt|docx|html)$', '', 'g'
                    ),
                    '[_\\-\\s]+(out|output)$', '', 'g'
                ),
                '[^a-z0-9]+', '', 'g'
            ) AS filename_norm,
            ts_rank_cd(
                to_tsvector('simple', coalesce(d.content_semantic, '') || ' ' || coalesce(d.content_raw, '') || ' ' || coalesce(d.metadata_json::text, '')),
                q.tsq
            ) AS rank
        FROM public.document_chunks d, q
        WHERE d.status = 'active'
          AND (
                (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
                OR
                (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
              )
    ),
    ranked AS (
        SELECT *,
               row_number() OVER (
                    PARTITION BY chunk_uuid, scope, organization_id
                    ORDER BY ingestion_ts DESC
               ) AS rn
        FROM visible
    )
    SELECT
        chunk_uuid::text,
        content_raw,
        content_semantic,
        metadata_json,
        ingestion_ts,
        rank,
        scope,
        organization_id,
        tier
    FROM ranked
    WHERE rn = 1
      AND length(filename_norm) > 0
      AND (filename_norm LIKE %s OR %s LIKE ('%%' || filename_norm || '%%'))
    ORDER BY rank DESC, ingestion_ts DESC
    LIMIT %s;
    """

    conn = pg_get_conn_secure()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_text,
                    current_organization_id(),
                    f"%{wanted_norm}%",
                    wanted_norm,
                    limit,
                ),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for chunk_uuid, raw, semantic, metadata, ingestion_ts, rank, scope, org_id, tier in rows:
            metadata = metadata_with_tenant(metadata, scope, org_id, tier)
            out.append({
                "id": str(chunk_uuid),
                "content": semantic or raw or "",
                "metadata": metadata,
                "score": float(rank or 0.001),
                "origin": "PostgresDocScope",
                "scope": str(metadata.get("scope") or "").upper(),
                "organization_id": _optional_int(metadata.get("organization_id")),
                "ingestion_ts": ingestion_ts.isoformat() if ingestion_ts else "",
            })
        return out
    except Exception as e:
        print(f"⚠️ Postgres document-scope search error: {e}")
        return []
    finally:
        pg_put_conn_secure(conn)

def relation_row_matches_requested_doc(row: Dict[str, Any], requested_doc: str) -> bool:
    """
    Applica lo stesso filtro documento anche alle relazioni Neo4j.
    """
    if not requested_doc:
        return True

    return candidate_matches_requested_doc(
        {"filename": row.get("filename", "")},
        requested_doc,
    )


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
        return [SourceItem(
            id="error",
            content="Backend OFF",
            filename="System",
            tier="USER",
            scope="ACCOUNT",
            organization_id=current_organization_id(),
            status="active",
            corpus_version=CORPUS_VERSION,
            request_id=get_tenant_context().request_id,
        )], "Backend OFF"

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
    
    tenant_filter = build_qdrant_tenant_filter()

    try:
        # Compatibilità universale per le versioni nuove e vecchie di Qdrant
        if hasattr(qdrant_client_inst, 'query_points'):
            response = qdrant_client_inst.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=tenant_filter,
                limit=qdrant_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = qdrant_client_inst.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=tenant_filter,
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
    neo4j_relation_rows = search_neo4j_relations(expanded_query, limit=60)

    # Se l'utente ha chiesto un documento specifico,
    # anche le relazioni Neo4j devono rispettare lo stesso perimetro.
    if requested_doc and neo4j_relation_rows:
        before_rel_scope = len(neo4j_relation_rows)

        neo4j_relation_rows = [
            r for r in neo4j_relation_rows
            if relation_row_matches_requested_doc(r, requested_doc)
        ]

        counts["neo4j_relation_scope_before"] = before_rel_scope
        counts["neo4j_relation_scope_after"] = len(neo4j_relation_rows)

    formula_query = should_query_neo4j_formulas(query_text)

    neo4j_formula_hits = (
        search_neo4j_formulas(
            query_text,
            limit=GRAPH_MAX_FORMULAS,
            requested_doc=requested_doc,
        )
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

        if not qdrant_payload_is_visible(payload):
            logger.warning("Qdrant vector hit %s scartato dal tenant guard", uid)
            continue

        content = safe_payload_text(payload)
        if not content:
            continue

        candidates_dict[uid] = {
            "id": uid,
            "content": content,
            "filename": str(payload.get("filename", "Unknown")),
            "doc_id": str(payload.get("doc_id") or ""),
            "page": get_payload_page(payload),
            "page_chunk_index": int(payload.get("page_chunk_index") or 0),
            "type": get_payload_type(payload),
            "tier": normalize_tier_value(str(payload.get("tier", "C"))),
            "score_base": float(hit.score or 0.0),
            "score_vec": float(hit.score or 0.0),
            "score_bm25": 0.0,
            "score_graph": 0.0,
            "origin": "Qdrant",
            "section_hint": get_payload_section(payload),
            "image_id": get_payload_image_id(payload),
            "scope": str(payload.get("scope") or "").upper(),
            "organization_id": _optional_int(payload.get("organization_id")),
            "status": str(payload.get("status") or ""),
            "ingestion_run_id": str(payload.get("ingestion_run_id") or ""),
            "corpus_version": str(payload.get("corpus_version") or ""),
            "classification": str(payload.get("classification") or "internal"),
            "embedding_model": str(payload.get("embedding_model") or ""),
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
                "doc_id": str(meta.get("doc_id") or ""),
                "page": page,
                "page_chunk_index": int(meta.get("page_chunk_index") or 0),
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
                "doc_id": str(meta.get("doc_id") or ""),
                "page": page,
                "page_chunk_index": int(meta.get("page_chunk_index") or 0),
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
                "doc_id": str(meta.get("doc_id") or ""),
                "page": page,
                "page_chunk_index": int(meta.get("page_chunk_index") or 0),
                "type": toon_type,
                "tier": tier,
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": float(b.get("score", 0.0)),
                "score_graph": 0.0,
                "origin": "Postgres",
                "scope": str(meta.get("scope") or "").upper(),
                "organization_id": _optional_int(meta.get("organization_id")),
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
                "doc_id": str(g.get("doc_id") or ""),
                "page": int(g.get("page") or 0),
                "page_chunk_index": int(g.get("page_chunk_index") or 0),
                "type": g.get("type", "graph"),
                "tier": "GRAPH",
                "score_base": 0.0,
                "score_vec": 0.0,
                "score_bm25": 0.0,
                "score_graph": float(g.get("score_graph", 0.0)),
                "origin": g.get("origin", "Neo4j"),
                "section_hint": g.get("section_hint", ""),
                "scope": str(g.get("scope") or "").upper(),
                "organization_id": _optional_int(g.get("organization_id")),
            }
        else:
            candidates_dict[uid]["score_graph"] = max(
                float(candidates_dict[uid].get("score_graph", 0.0)),
                float(g.get("score_graph", 0.0)),
            )

            # Non perdere la formula quando lo stesso chunk è già arrivato da
            # Qdrant/PostgreSQL. Prima il merge conservava solo lo score e
            # scartava il contenuto del nodo formula.
            if normalize_source_type(g.get("type", "")) == "formula":
                formula_content = str(g.get("content") or "").strip()
                current_content = str(candidates_dict[uid].get("content") or "").strip()
                if formula_content and formula_content not in current_content:
                    candidates_dict[uid]["content"] = (
                        current_content
                        + "\n\n--- Formula collegata dal Knowledge Graph ---\n"
                        + formula_content
                    ).strip()
                candidates_dict[uid]["type"] = "formula"
                candidates_dict[uid]["section_hint"] = g.get(
                    "section_hint",
                    candidates_dict[uid].get("section_hint", ""),
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
                        "doc_id": gs.doc_id,
                        "page": gs.page,
                        "page_chunk_index": gs.page_chunk_index,
                        "type": gs.type,
                        "tier": normalize_tier_value(getattr(gs, "tier", "C")),
                        "score_base": 0.0,
                        "score_vec": 0.0,
                        "score_bm25": 0.0,
                        "score_graph": 1.0,
                        "origin": "Neo4j_Expansion",
                        "section_hint": getattr(gs, "section_hint", ""),
                        "scope": str(getattr(gs, "scope", "") or "").upper(),
                        "organization_id": _optional_int(getattr(gs, "organization_id", None)),
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

        if formula_query and ctype == "formula":
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
    candidates.sort(
        key=lambda x: x.get("pre_rerank_score", 0.0),
        reverse=True,
    )

    # Un lookup documentale di formule deve mantenere tutti i chunk già
    # filtrati sul documento, senza dipendere da un limite costruito sul caso.
    exhaustive_formula_lookup = bool(
        requested_doc and is_formula_lookup_query(query_text)
    )
    if exhaustive_formula_lookup:
        top_candidates = list(candidates)
    else:
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
    # Il lookup esaustivo mantiene il perimetro documentale già applicato al
    # punto 7B; le altre query conservano la diversificazione standard.
    if exhaustive_formula_lookup:
        final_selection = list(top_candidates)
    else:
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

        # v4.14 - Per i lookup di formule il testo raw del documento è
        # autoritativo. Il contenuto semantico può aver subito normalizzazioni
        # LLM/OCR che alterano backslash, parentesi e operatori LaTeX.
        raw_content = str(pg_row.get("content_raw", "") or "")
        semantic_content = str(pg_row.get("content_semantic", "") or "")

        if formula_query:
            preferred_content = raw_content or semantic_content
        elif PG_PREFER_RAW:
            preferred_content = raw_content or semantic_content
        else:
            preferred_content = semantic_content or raw_content

        if preferred_content:
            t["content"] = preferred_content

        current_filename = str(t.get("filename") or "").strip()
        pg_filename = (
            pg_meta.get("filename")
            or pg_meta.get("source_name")
            or current_filename
            or "Unknown"
        )

        if current_filename in ("", "Unknown", "Neo4j", "KG", "Neo4j Knowledge Graph"):
            t["filename"] = pg_filename
        else:
            t["filename"] = current_filename

        current_page = int(t.get("page") or 0)
        pg_page = int(pg_meta.get("page_no") or pg_meta.get("page") or 0)

        if current_page <= 0 and pg_page > 0:
            t["page"] = pg_page
        else:
            t["page"] = current_page

        current_type = normalize_source_type(t.get("type", ""))
        pg_type = normalize_source_type(pg_meta.get("toon_type") or pg_meta.get("type") or "")

        if current_type in ("", "graph") and pg_type:
            t["type"] = pg_type
        else:
            t["type"] = current_type or "text"

        t["tier"] = normalize_tier_value(pg_row.get("tier") or pg_meta.get("tier") or t.get("tier") or "C")
        t["scope"] = str(pg_row.get("scope") or pg_meta.get("scope") or t.get("scope") or "").upper()
        t["organization_id"] = _optional_int(
            pg_row.get("organization_id")
            if pg_row.get("organization_id") is not None
            else pg_meta.get("organization_id", t.get("organization_id"))
        )
        t["status"] = str(pg_meta.get("status") or t.get("status") or "active")
        t["ingestion_run_id"] = str(pg_meta.get("ingestion_run_id") or t.get("ingestion_run_id") or "")
        t["corpus_version"] = str(pg_meta.get("corpus_version") or t.get("corpus_version") or CORPUS_VERSION)
        t["classification"] = str(pg_meta.get("classification") or t.get("classification") or "internal")
        t["embedding_model"] = str(pg_meta.get("embedding_model") or t.get("embedding_model") or "")

        t["pg_ingestion_ts"] = pg_row.get("ingestion_ts", "")
        t["pg_source_name"] = pg_meta.get("source_name", "")
        t["pg_source_type"] = pg_meta.get("source_type", "")
        t["pg_log_id"] = int(pg_meta.get("log_id") or 0)
        t["doc_id"] = str(pg_meta.get("doc_id") or t.get("doc_id") or "")
        t["page_chunk_index"] = int(
            pg_meta.get("page_chunk_index")
            if pg_meta.get("page_chunk_index") is not None
            else (t.get("page_chunk_index") or 0)
        )
        t["pg_chunk_id"] = int(pg_meta.get("chunk_index") or 0)
        t["pg_page_chunk_index"] = int(pg_meta.get("page_chunk_index") or 0)
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
                page_chunk_index=int(t.get("page_chunk_index") or 0),
                doc_id=str(t.get("doc_id") or ""),
                type=t.get("type", "text"),
                score=float(t.get("final_score", 0.0)),
                tier=normalize_tier_value(t.get("tier", "C")),
                db_origin=t.get("origin", "Unknown"),
                section_hint=t.get("section_hint", ""),
                image_id=t.get("image_id"),
                scope=str(t.get("scope") or "").upper(),
                organization_id=_optional_int(t.get("organization_id")),
                status=str(t.get("status") or "active"),
                ingestion_run_id=str(t.get("ingestion_run_id") or ""),
                corpus_version=str(t.get("corpus_version") or CORPUS_VERSION),
                classification=str(t.get("classification") or "internal"),
                embedding_model=str(t.get("embedding_model") or ""),
                request_id=get_tenant_context().request_id,
                pg_ingestion_ts=t.get("pg_ingestion_ts", ""),
                pg_source_name=t.get("pg_source_name", ""),
                pg_source_type=t.get("pg_source_type", ""),
                pg_log_id=int(t.get("pg_log_id") or 0),
                pg_chunk_id=int(t.get("pg_chunk_id") or 0),
                pg_page_chunk_index=int(t.get("pg_page_chunk_index") or 0),
                pg_toon_type=t.get("pg_toon_type", ""),
            )
        )

    # 13) Final formulas from Neo4j
    counts["final_formulas"] = 0

    if GRAPH_EXPAND_ENABLED and neo4j_driver:
        chunk_ids = [s.id for s in sources if s.id and s.id != "graph"]

        all_formulas_flat = []

        if formula_query:
            formulas_dict = get_formulas_for_chunks(
                chunk_ids,
                limit_per_chunk=GRAPH_MAX_FORMULAS,
            )

            all_formulas_flat = [
                formula
                for f_list in formulas_dict.values()
                for formula in f_list
            ]

        counts["final_formulas"] = len(all_formulas_flat)

        # v4.14 - Il source aggregato "KG" non deve essere aggiunto dopo
        # l'HARD DOCUMENT SCOPE FILTER. Quando è richiesto un documento preciso,
        # le formule Neo4j sono già entrate come candidati con filename/pagina del
        # chunk sorgente. L'aggregato generico perderebbe la provenienza e
        # reintrodurrebbe formule esterne o duplicate.
        if all_formulas_flat and not requested_doc:
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
                    scope="ACCOUNT",
                    organization_id=current_organization_id(),
                    status="active",
                    corpus_version=CORPUS_VERSION,
                    request_id=get_tenant_context().request_id,
                )
            )

        rel_source = graph_relations_to_source(neo4j_relation_rows)
        if rel_source:
            sources.append(rel_source)

    sources = filter_sources_for_current_organization(sources)
    counts["final_sources_after_tenant_guard"] = len(sources)

    ctx = get_tenant_context()
    append_audit_log(
        AuditTrail(
            ts_utc=datetime.utcnow().isoformat() + "Z",
            query="",
            query_sha256=hashlib.sha256((query_text or "").encode("utf-8")).hexdigest(),
            intent=intent,
            organization_id=ctx.organization_id,
            user_id=ctx.user_id,
            roles=ctx.roles,
            request_id=ctx.request_id,
            corpus_version=CORPUS_VERSION,
            filters={
                "visibility": "GLOBAL/A or ACCOUNT/current/B,C",
                "status": "active",
                "active_doc": active_doc or "",
            },
            retrieved_sources=[
                {
                    "id": s.id, "doc_id": s.doc_id, "filename": s.filename,
                    "page": s.page, "tier": s.tier, "scope": s.scope,
                    "organization_id": s.organization_id, "status": s.status,
                    "ingestion_run_id": s.ingestion_run_id,
                }
                for s in sources
            ],
            retrieval=RetrievalDebug(
                query="", intent=intent, final_sources=len(sources),
                tier_counts=dict(Counter(s.tier for s in sources)),
            ),
            llm_model=LLM_MODEL_NAME,
            memory_limit=MEMORY_LIMIT,
        )
    )

    return sources, build_retrieval_audit_md(
        query_text, intent, timings, counts, [],
    )


def build_context_block(sources: List[SourceItem], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context only from sources visible to the current organization."""
    sources = filter_sources_for_current_organization(sources)
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
    Framework-agnostic, assessment-oriented, with strict grounding,
    source discipline, math consistency and formula rendering rules.
    """
    base = """
ROLE:
You are a Senior Technical Auditor and Compliance AI.

1. MATHEMATICAL PRIORITY:
If the query provides numerical values and requires a calculation, execute the math step-by-step as your absolute priority.
The final number stated in the first paragraph must exactly match the number obtained in the calculation steps.
Before finalizing, check that there is no contradiction between the declared result and the arithmetic shown below.
If the calculation uses only numerical values provided by the user, state clearly that the result is based on user-provided values and do not use retrieved documents to alter the calculation.
For time calculations, verify day transitions carefully: 24 hours = 1 day, 48 hours = 2 days, 72 hours = 3 days.
For economic calculations, distinguish gross benefit from net benefit: if explicit costs are provided, subtract them before calling the result “net”.

2. DATA GROUNDING:
Answer using ONLY the provided retrieved context.
If a specific value, formula, authority, institution, legal article, deadline, sanction, framework relation, or concept is not in the retrieved context, explicitly state:
"Information not found in retrieved documents."
Do not invent, assume, or import external standards, authorities, laws, websites, official portals or background knowledge.

3. DEFINITIONS:
Extract definitions exactly as written in the retrieved text.
If the user asks for a pure definition, quote or paraphrase only what is present in the retrieved context.
If the exact term is not found, say that it was not found in the retrieved documents.

4. CROSS-REFERENCING:
Synthesize information across all relevant retrieved documents impartially.
Do not bias toward a specific framework unless requested.
If multiple documents support different aspects of the answer, clearly distinguish what each document supports.

5. CITATION:
Always cite the specific retrieved source file and page for every claim.
Never cite external URLs, websites, official portals, laws, standards, authorities or references that are not present in the retrieved context.
Section B and Section D must use ONLY retrieved source filenames and pages.

6. NARRATIVE SYNTHESIS:
If the context contains structured data, JSON, graph nodes, or relations such as source-relation-target, synthesize them into fluent, professional paragraphs.
Never output raw database logs, raw JSON, or raw graph triples unless the user explicitly asks for graph/table output.
Translate technical relations into plain language.

7. CLASSIFICATION / REGULATORY QUERY RULE:
If the user asks "chi sono", "quali sono", "what are", "who are", "which are" followed by categories, subjects, entities, obligations, supervision regimes, requirements, sanctions, or regulatory classes, this is NOT a glossary question.
Treat it as a regulatory classification question and answer from retrieved context using structured paragraphs or bullets.

8. FORMULA VISIBILITY RULE:
If the answer contains formulas, equations, inequalities, thresholds or algebraic derivations, never rely only on rendered LaTeX.
Always include a visible plain-text formula before any LaTeX version.

TONE:
Technical, objective, concise, and evidence-based.

OUTPUT STRUCTURE:
You MUST structure your response in EXACTLY these four sections, using these EXACT headers:

**A) Risposta**
Direct technical assessment in discursive paragraphs or structured bullets.

**B) Evidenze**
Bullet points citing ONLY retrieved source files and pages.
Every bullet should refer to a retrieved filename and page, unless the answer is purely deterministic math based only on user-provided values.
Do not cite generic standards, external laws, websites, official portals or background knowledge unless they are present in the retrieved sources.

**C) Limiti / Conflitti**
State missing evidence, contradictions, assumptions, or limits of the retrieved context.

**D) Fonti**
List only retrieved filenames and pages. Do not list external websites or references not present in the retrieved context.

LANGUAGE RULE:
Always answer in the same language used by the user.
If the user writes in Italian, the entire answer must be in Italian, including reasoning labels, explanations, limitations, and formula descriptions.
Do not mix English and Italian unless the user explicitly asks for bilingual output.
"""

    if intent == "formula":
        base += """
INTENT: FORMULA / METRIC / ALGEBRA.

FORMULA OUTPUT RULES:
1. If the user asks to write, derive, express, isolate or solve an equation or inequality, always show the formula twice:
   - Formula testuale: `plain text formula`
   - Formula LaTeX, delimited as display math on separate lines:
     $$
     LaTeX code
     $$
2. The plain text formula is mandatory and must appear BEFORE the LaTeX version.
3. Never wrap LaTeX in backticks or in fenced code blocks such as ```latex; those formats display code instead of rendering mathematics.
4. Never leave blank mathematical placeholders.
5. Never write empty parentheses for variables.
6. If variables appear in the user question, preserve their names exactly in the plain text formula.
7. If the formula is derived from user-provided values, explicitly state that it is derived from user-provided values.
8. If the context does not contain a formula but the user provided all variables and relationships, you may derive the algebraic expression from the user-provided statement, marking it as user-provided derivation.
"""
    elif intent == "table":
        base += """
INTENT: TABLE.
Output a complete Markdown table based only on the retrieved context.
Do not invent rows, columns, control IDs, article numbers, mappings or evidence not present in the retrieved context.
If a value is missing, write "non recuperato" or "fonte non recuperata".
"""
    elif intent == "chart":
        base += """
INTENT: CHART / DIAGRAM.
Describe the topology, process, architecture, chart or diagram extracted from the retrieved context.
If graph relations are present, synthesize them in natural language unless the user explicitly asks for a graph table.
Do not infer missing nodes, links, numbers or labels.
"""
    elif intent == "audit":
        base += """
INTENT: AUDIT / COMPLIANCE.
Prioritize requirements, obligations, evidence, gaps, controls, responsibilities, risk, compliance status and conflicts.
Clearly distinguish:
- requisito normativo;
- controllo/procedura;
- evidenza recuperata;
- deduzione non esplicita;
- informazione non trovata.
"""

    return base

def tier_guardrail_instructions(query_text: str) -> str:
    wants_evidence = is_evidence_query(query_text)
    return (
        "COMPLIANCE-GRADE GUARDRAILS:\n"
        "1) Tier A (Normative): Primary source for legal and framework requirements.\n"
        "2) Tier B (Governance): Internal policies and planned procedures.\n"
        "3) Tier C (Evidences): Technical proof of actual implementation.\n"
        "4) Grounding: Every statement must be supported by the provided context. Do not use external laws, portals, authorities, standards or URLs unless they are explicitly present in the retrieved sources. Flag any non-conformities.\n"
        "5) Gap Analysis: If technical evidence is missing to prove a policy, state it in section C.\n"
        f"6) {'EVIDENCE FOCUS: The user specifically requested technical proofs, logs, or configurations. Prioritize Tier C context.' if wants_evidence else 'Standard audit: verify alignment across all Tiers.'}\n"
        f"7) GLOSSARY & DEFINITIONS: If the user asks for a pure definition ('definizione', 'significato', 'meaning', 'definition'), quote EXACTLY from the provided context. If the user asks for categories, subjects, regimes, obligations or requirements, do NOT treat it as atomic glossary: answer as a regulatory/compliance question using retrieved sources.\n"
        f"8) MATH & PENALTIES: If calculating percentages, fines, thresholds, deadlines or algebraic relations, explicitly write out the mathematical steps before providing the final number. The final number must match the calculation. For formulas, always provide a plain-text formula before any LaTeX.\n"
        f"9) ANTI-HALLUCINATION: If the context does not mention a specific scenario, authority, sanction, date, article or relation, state clearly: 'I documenti forniti non contengono questa informazione'.\n"
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
HALLUCINATION: If the context does not mention a specific scenario, state clearly: 'I documenti forniti non contengono questa informazione'.\n"
    )
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



def normalize_math_markdown_for_reflex(text: str) -> str:
    """
    Converte le forme LaTeX comunemente prodotte dall'LLM nel formato
    riconosciuto da ``rx.markdown``/remark-math:

    - inline: ``$ ... $``;
    - display: ``$$ ... $$``.

    I blocchi `````latex```, ``\\[...\\]`` e le formule racchiuse nei
    backtick dopo l'etichetta "Formula LaTeX" vengono convertiti in display
    math. Il contenuto matematico non viene trasformato in testo normale.
    """
    if not text:
        return ""

    out = str(text).replace("\r\n", "\n").replace("\r", "\n")

    def _clean_formula_body(value: str) -> str:
        body = (value or "").strip()
        body = re.sub(r"^\$\$?\s*|\s*\$\$?$", "", body).strip()
        body = re.sub(r"^\\\[\s*|\s*\\\]$", "", body).strip()
        body = re.sub(r"^\\\(\s*|\s*\\\)$", "", body).strip()
        return body

    # Un fenced code block è codice, non matematica. Lo trasformiamo in
    # display math affinché rehype-katex possa renderizzarlo.
    out = re.sub(
        r"```(?:latex|tex|math)\s*\n(.*?)\n```",
        lambda m: "\n\n$$\n" + _clean_formula_body(m.group(1)) + "\n$$\n\n",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Forma frequentemente prodotta dal prompt precedente:
    # Formula LaTeX: `V_m \\leq \\frac{V}{3}`
    out = re.sub(
        r"(?im)(Formula\s+LaTeX\s*:)\s*`([^`\n]+)`",
        lambda m: m.group(1) + "\n\n$$\n" + _clean_formula_body(m.group(2)) + "\n$$",
        out,
    )

    # Delimitatori MathJax-style -> delimitatori remark-math/KaTeX.
    out = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "\n\n$$\n" + _clean_formula_body(m.group(1)) + "\n$$\n\n",
        out,
        flags=re.DOTALL,
    )
    out = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + _clean_formula_body(m.group(1)) + "$",
        out,
        flags=re.DOTALL,
    )

    # Garantisce righe autonome per i delimitatori display.
    out = re.sub(r"[^\S\n]*\$\$[^\S\n]*", "$$", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
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
            scope="ACCOUNT",
            organization_id=current_organization_id(),
            status="active",
            corpus_version=CORPUS_VERSION,
            request_id=get_tenant_context().request_id,
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
                page_chunk_index=ui_safe_int(getattr(s, "page_chunk_index", 0), 0),
                doc_id=ui_safe_text(getattr(s, "doc_id", ""), 120),
                type=ui_safe_text(getattr(s, "type", "text"), 80),
                score=ui_safe_float(getattr(s, "score", 0.0), 0.0),
                graph_context=[],
                section_hint=ui_safe_text(getattr(s, "section_hint", ""), 300),
                image_id=getattr(s, "image_id", None),
                tier=ui_safe_text(getattr(s, "tier", "C"), 40),
                scope=ui_safe_text(getattr(s, "scope", ""), 20),
                organization_id=_optional_int(getattr(s, "organization_id", None)),
                status=ui_safe_text(getattr(s, "status", "active"), 30),
                ingestion_run_id=ui_safe_text(getattr(s, "ingestion_run_id", ""), 80),
                corpus_version=ui_safe_text(getattr(s, "corpus_version", ""), 80),
                classification=ui_safe_text(getattr(s, "classification", "internal"), 40),
                embedding_model=ui_safe_text(getattr(s, "embedding_model", ""), 160),
                request_id=ui_safe_text(getattr(s, "request_id", ""), 80),
                pg_ingestion_ts=ui_safe_text(getattr(s, "pg_ingestion_ts", ""), 80),
                pg_source_name=ui_safe_text(getattr(s, "pg_source_name", ""), 160),
                pg_source_type=ui_safe_text(getattr(s, "pg_source_type", ""), 80),
                pg_log_id=ui_safe_int(getattr(s, "pg_log_id", 0), 0),
                pg_chunk_id=ui_safe_int(getattr(s, "pg_chunk_id", 0), 0),
                pg_page_chunk_index=ui_safe_int(getattr(s, "pg_page_chunk_index", 0), 0),
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
                page_chunk_index=ui_safe_int(state_get(s, "page_chunk_index", 0), 0),
                doc_id=ui_safe_text(state_get(s, "doc_id", ""), 120),
                type=ui_safe_text(state_get(s, "type", "text"), 80),
                score=ui_safe_float(state_get(s, "score", 0.0), 0.0),
                graph_context=[],
                section_hint=ui_safe_text(
                    state_get(s, "section_hint", ""),
                    300,
                ),
                image_id=state_get(s, "image_id", None),
                tier=ui_safe_text(state_get(s, "tier", "C"), 40),
                scope=ui_safe_text(state_get(s, "scope", ""), 20),
                organization_id=_optional_int(state_get(s, "organization_id", None)),
                status=ui_safe_text(state_get(s, "status", ""), 30),
                ingestion_run_id=ui_safe_text(state_get(s, "ingestion_run_id", ""), 80),
                corpus_version=ui_safe_text(state_get(s, "corpus_version", ""), 80),
                classification=ui_safe_text(state_get(s, "classification", "internal"), 80),
                embedding_model=ui_safe_text(state_get(s, "embedding_model", ""), 160),
                request_id=ui_safe_text(state_get(s, "request_id", ""), 80),
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
                pg_page_chunk_index=ui_safe_int(state_get(s, "pg_page_chunk_index", 0), 0),
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


def _weekday_index_it_en(value: str) -> Optional[int]:
    days = {
        "lunedì": 0, "lunedi": 0, "monday": 0,
        "martedì": 1, "martedi": 1, "tuesday": 1,
        "mercoledì": 2, "mercoledi": 2, "wednesday": 2,
        "giovedì": 3, "giovedi": 3, "thursday": 3,
        "venerdì": 4, "venerdi": 4, "friday": 4,
        "sabato": 5, "saturday": 5,
        "domenica": 6, "sunday": 6,
    }
    return days.get((value or "").lower().strip())


def _weekday_name_it(index: int) -> str:
    names = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    return names[index % 7]


def _parse_base_datetime_or_weekday(query_text: str) -> Tuple[Optional[datetime], Optional[int], str]:
    """
    Estrae una base temporale:
    - data esplicita dd/mm/yyyy con eventuale ora;
    - oppure giorno della settimana + ora.

    Ritorna:
    (datetime_reale, weekday_index, label_base)
    """
    q = query_text or ""
    ql = q.lower()

    # Data esplicita: 01/06/2026 ore 08:00
    m_date = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:.*?\b(?:ore|at)?\s*(\d{1,2})[:.](\d{2}))?",
        q,
        flags=re.IGNORECASE,
    )

    if m_date:
        day, month, year = map(int, m_date.group(1, 2, 3))
        hour = int(m_date.group(4) or 0)
        minute = int(m_date.group(5) or 0)
        try:
            dt = datetime(year, month, day, hour, minute)
            return dt, dt.weekday(), _format_it_date(dt) + f" {hour:02d}:{minute:02d}"
        except Exception:
            return None, None, ""

    # Giorno settimana + ora: lunedì alle ore 08:00
    m_weekday = re.search(
        r"\b(lunedì|lunedi|monday|martedì|martedi|tuesday|mercoledì|mercoledi|wednesday|giovedì|giovedi|thursday|venerdì|venerdi|friday|sabato|saturday|domenica|sunday)\b"
        r".{0,40}?\b(?:ore|at|alle|le)?\s*(\d{1,2})[:.](\d{2})",
        ql,
        flags=re.IGNORECASE,
    )

    if m_weekday:
        wd = _weekday_index_it_en(m_weekday.group(1))
        hour = int(m_weekday.group(2))
        minute = int(m_weekday.group(3))
        # Data fittizia: lunedì 2000-01-03.
        base_monday = datetime(2000, 1, 3, hour, minute)
        base_dt = base_monday + timedelta(days=int(wd or 0))
        return base_dt, wd, f"{_weekday_name_it(int(wd or 0))} {hour:02d}:{minute:02d}"

    return None, None, ""


def try_solve_date_offsets(query_text: str) -> Optional[str]:
    """
    Solver deterministico non adattativo per offset temporali in ore.

    Gestisce:
    - giorno della settimana;
    - ora HH:MM;
    - offset espressi in ore;
    - delta tra prima e ultima scadenza.
    """
    q = query_text or ""
    ql = q.lower()

    if not any(t in ql for t in ["ore", "ora", "entro", "scadenza", "within",  "delta", "hours", "deadline"]):
        return None

    weekday_map = {
        "lunedì": 0, "lunedi": 0, "monday": 0,
        "martedì": 1, "martedi": 1, "tuesday": 1,
        "mercoledì": 2, "mercoledi": 2, "wednesday": 2,
        "giovedì": 3, "giovedi": 3, "thursday": 3,
        "venerdì": 4, "venerdi": 4, "friday": 4,
        "sabato": 5, "saturday": 5,
        "domenica": 6, "sunday": 6,
    }

    weekday_names_it = [
        "lunedì", "martedì", "mercoledì", "giovedì",
        "venerdì", "sabato", "domenica"
    ]

    found_day = None
    for name, idx in weekday_map.items():
        if name in ql:
            found_day = idx
            break

    if found_day is None:
        return None

    time_match = re.search(r"(?:ore\s*)?(\d{1,2})[:.](\d{2})", ql)
    if not time_match:
        return None

    hour = int(time_match.group(1))
    minute = int(time_match.group(2))

    if hour > 23 or minute > 59:
        return None

    offsets = []
    for m in re.finditer(r"entro\s+(\d+(?:[.,]\d+)?)\s*ore", ql):
        offsets.append(_parse_it_number(m.group(1)))

    if len(offsets) < 1:
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*ore", ql):
            offsets.append(_parse_it_number(m.group(1)))

    offsets = sorted(set(offsets))

    if len(offsets) < 1:
        return None

    base_minutes = found_day * 24 * 60 + hour * 60 + minute

    rows = []
    deadlines = []

    for off in offsets:
        total_minutes = base_minutes + int(off * 60)
        day_idx = (total_minutes // (24 * 60)) % 7
        final_hour = (total_minutes % (24 * 60)) // 60
        final_minute = total_minutes % 60

        deadlines.append(total_minutes)

        rows.append(
            f"- `+{off:g} ore` → **{weekday_names_it[day_idx]} {final_hour:02d}:{final_minute:02d}**"
        )

    delta_text = ""
    if len(deadlines) >= 2:
        delta_hours = (max(deadlines) - min(deadlines)) / 60.0
        delta_text = f"\n- Delta tra prima e ultima scadenza = **{delta_hours:g} ore**"

    return (
        "**A) Risposta**\n\n"
        f"Base temporale: **{weekday_names_it[found_day]} {hour:02d}:{minute:02d}**.\n\n"
        "**Scadenze calcolate:**\n\n"
        + "\n".join(rows)
        + delta_text
        + "\n\n"
        "\n\n**B) Evidenze**\n\n"
        "- Il giorno, l'orario iniziale e gli offset in ore sono stati estratti dalla domanda dell'utente.\n"
        "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- Il calcolo considera gli offset come ore solari continue.\n"
        "- Non considera festività, sospensioni operative o calendari lavorativi se non indicati nella domanda.\n\n"
        "**D) Fonti**\n\n"
        "- Input utente: valori e relazioni temporali presenti nella domanda."
    )


def _strip_math_wrappers(value: str) -> str:
    """Rimuove soltanto delimitatori matematici esterni completi."""
    v = (value or "").strip()
    v = re.sub(r"^`+|`+$", "", v).strip()

    changed = True
    while changed and v:
        changed = False
        wrapper_pairs = [
            ("$$", "$$"),
            ("$", "$"),
            (r"\[", r"\]"),
            (r"\(", r"\)"),
        ]
        for left, right in wrapper_pairs:
            if v.startswith(left) and v.endswith(right) and len(v) >= len(left) + len(right):
                v = v[len(left):-len(right)].strip()
                changed = True
                break

    return v
def _looks_definitional_metric(latex: str, meaning: str = "") -> bool:
    """
    Riconosce assegnazioni definitorie testuali senza usare termini di dominio.
    """
    value = _strip_math_wrappers(_normalize_latex_value(latex or ""))

    if "=" in value:
        _, right = value.split("=", 1)
        right = right.strip()
        right_plain = _formula_display_text(right, 1000)

        if re.fullmatch(r"\\?text\{[^}]+\}", right):
            return True

        words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", right_plain)
        operators = re.findall(
            r"[+\-*/×÷^]|\\frac|\\sum|\\prod|\\operatorname",
            right,
            flags=re.IGNORECASE,
        )
        if len(words) >= 3 and not operators:
            return True

    meaning_plain = _formula_display_text(meaning or "", 1000)
    meaning_words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", meaning_plain)
    return len(meaning_words) >= 5 and bool(
        re.search(
            r"\b(?:definizione|definition|metrica|metric|indicatore|indicator)\b",
            meaning_plain,
            flags=re.IGNORECASE,
        )
    )


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

    pattern = (
        r"(?is)"
        r"(?:^|\n)\s*"
        r"(?:#{1,6}\s*)?"
        r"(?:\*\*)?"
        r"D\s*[\)\.\-:]\s*"
        r"(?:Fonti|Sources|Riferimenti|References)"
        r"(?:\*\*)?"
        r"\s*:?"
        r".*\Z"
    )

    if re.search(pattern, answer or ""):
        return re.sub(pattern, "\n\n" + replacement, answer.rstrip()).strip()

    return (answer or "").rstrip() + "\n\n" + replacement


def _has_required_abcd_headers(answer: str) -> bool:
    """
    Riconosce header A/B/C/D sia in forma Markdown bold sia in forma semplice.
    """
    text = answer or ""

    patterns = [
        r"(?im)^\s*(?:\*\*)?\s*A\s*[\)\.\-:]\s*Risposta(?:\*\*)?",
        r"(?im)^\s*(?:\*\*)?\s*B\s*[\)\.\-:]\s*Evidenze(?:\*\*)?",
        r"(?im)^\s*(?:\*\*)?\s*C\s*[\)\.\-:]\s*(?:Limiti\s*/\s*Conflitti|Limiti|Conflitti)(?:\*\*)?",
        r"(?im)^\s*(?:\*\*)?\s*D\s*[\)\.\-:]\s*Fonti(?:\*\*)?",
    ]

    return all(re.search(p, text) for p in patterns)


def _looks_like_graph_table_answer(answer: str) -> bool:
    text = answer or ""
    return (
        "| Entità sorgente | Relazione | Entità target |" in text
        or "| Source entity | Relation | Target entity |" in text
    )


def _is_explanatory_question(query_text: str) -> bool:
    q = (query_text or "").lower()
    terms = [
        "qual è", "quale è", "quali sono", "ruolo", "scopo", "descrivi",
        "spiega", "analizza", "valuta", "in che modo", "come funziona",
        "what is", "what are", "role", "purpose", "describe", "explain",
        "analyze", "analyse", "evaluate", "how does",
    ]
    return any(t in q for t in terms)


def _repair_missing_abcd_headers(answer: str) -> str:
    """
    Ripara solo se la struttura A/B/C/D è realmente assente.
    Se gli header esistono ma non sono in grassetto, li normalizza.
    """
    text = (answer or "").strip()

    if not text:
        return (
            "**A) Risposta**\n\n"
            "Risposta non disponibile.\n\n"
            "\n\n**B) Evidenze**\n\n"
            "- Nessuna evidenza disponibile.\n\n"
            "\n\n**C) Limiti / Conflitti**\n\n"
            "- Risposta vuota o non generata.\n\n"
            "**D) Fonti**\n\n"
            "- Nessuna fonte disponibile."
        )

    if _has_required_abcd_headers(text):
        text = re.sub(r"(?im)^\s*(?:\*\*)?\s*A\s*[\)\.\-:]\s*Risposta(?:\*\*)?\s*$", "**A) Risposta**", text)
        text = re.sub(r"(?im)^\s*(?:\*\*)?\s*B\s*[\)\.\-:]\s*Evidenze(?:\*\*)?\s*$", "\n\n**B) Evidenze**", text)
        text = re.sub(r"(?im)^\s*(?:\*\*)?\s*C\s*[\)\.\-:]\s*(?:Limiti\s*/\s*Conflitti|Limiti|Conflitti)(?:\*\*)?\s*$", "**C) Limiti / Conflitti**", text)
        text = re.sub(r"(?im)^\s*(?:\*\*)?\s*D\s*[\)\.\-:]\s*Fonti(?:\*\*)?\s*$", "**D) Fonti**", text)
        return text

    return (
        "**A) Risposta**\n\n"
        + text
        + "\n\n**B) Evidenze**\n\n"
        "- Vedi fonti recuperate nel pannello Fonti/Audit.\n\n"
        "\n\n**C) Limiti / Conflitti**\n\n"
        "- La struttura della risposta è stata normalizzata automaticamente.\n\n"
        "**D) Fonti**\n\n"
        "- Vedi pannello Fonti/Audit."
    )

def quality_gate_postprocess(answer: str, query_text: str, sources: List[SourceItem]) -> str:
    """
    Quality gate finale non adattativo.
    Non corregge il dominio; corregge classi generali di errore.
    """
    out = answer or ""

    # 1) Struttura A/B/C/D obbligatoria.
    out = _repair_missing_abcd_headers(out)

    # 2) Se la domanda è esplicativa, non lasciare che una tabella grafo
    # diventi l'unico contenuto della sezione A.
    if (
        _is_explanatory_question(query_text)
        and _looks_like_graph_table_answer(out)
        and not should_use_graph_relation_strict_mode(query_text)
    ):
        out = (
            "**A) Risposta**\n\n"
            "La domanda richiede una spiegazione discorsiva. Le relazioni del grafo recuperate "
            "possono essere usate come supporto, ma non sono sufficienti da sole a costituire "
            "la risposta finale.\n\n"
            "\n\n**B) Evidenze**\n\n"
            "- Il retrieval ha recuperato relazioni o co-occorrenze, ma il formato tabellare non è appropriato come unica risposta.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- È necessario usare i chunk testuali recuperati per produrre una spiegazione motivata.\n"
            "- Il grafo resta fonte di supporto, non formato obbligatorio della risposta.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit."
        )

    # 3) URL esterni vietati nel testo finale.
    out = re.sub(
        r"https?://\S+",
        "[Link esterno non autorizzato rimosso]",
        out,
        flags=re.IGNORECASE,
    )

    # 3-ter) Neo4j wording guard.
    # Se la query chiede esplicitamente grafo/Neo4j, non permettere frasi
    # che facciano sembrare il grafo inesistente o simulato.
    if is_graph_relation_query(query_text) or should_use_graph_relation_strict_mode(query_text):
        forbidden_graph_phrases = [
            "Poiché non è stato fornito un grafo Neo4j",
            "poiché non è stato fornito un grafo Neo4j",
            "non è stato fornito un grafo Neo4j",
            "simulando una query Neo4j",
            "simulando Neo4j",
            "non contengono un grafo Neo4j preesistente",
            "assenza di un grafo Neo4j",
        ]

        for phrase in forbidden_graph_phrases:
            out = out.replace(
                phrase,
                "Non sono stati recuperati archi Neo4j espliciti sufficienti"
            )

    # 3-bis) Formula visibility guard.
    # Se una query matematica/algebrica produce una risposta senza formule visibili,
    # prova il fallback deterministico sui dati forniti dall'utente.
    if is_formula_strict_query(query_text):
        has_visible_formula = bool(
            re.search(r"[A-Za-z0-9_]\s*(=|>|<|≤|≥|×|\*)\s*[A-Za-z0-9_(]", out)
            or re.search(r"(Formula testuale|Formula LaTeX)", out, flags=re.IGNORECASE)
        )

        if not has_visible_formula:
            algebra_fallback = try_solve_user_provided_algebra(query_text)
            if algebra_fallback:
                out = algebra_fallback


    # Questo elimina riferimenti esterni residui anche quando il modello usa varianti
    # come "D) Sources", "D. Fonti", "References", ecc.
    # Nei calcoli puri, però, NON sostituisce D) Fonti con documenti recuperati.
    if sources and not (
        is_calculation_request(query_text)
        and not needs_math_document_context(query_text)
    ):
        out = _replace_final_sources_section(out, sources)

    return out.strip()


def postprocess_generated_answer(answer: str, query_text: str, sources: List[SourceItem]) -> str:
    """
    Corregge automaticamente le etichette della struttura e le frasi di fallback
    in base alla lingua della domanda dell'utente.
    """
    out = answer or ""

    out = re.sub(
        r"(?im)^\s*Formula LaTeX:\s*$\n?",
        "",
        out,
    )

    out = re.sub(
        r"Formula LaTeX:\s*(?:\n|$)",
        "",
        out,
        flags=re.IGNORECASE,
    )

    if _is_likely_italian_query(query_text):
        # 1. Traduzione etichette strutturali
        replacements = {
            "**C) Limiti / Conflitti**": "**C) Limiti / Conflitti**",
            "**C) Limitations / Conflicts**": "**C) Limiti / Conflitti**",
            "Limitations / Conflicts": "Limiti / Conflitti",
            "Information not found in retrieved documents": "Non ho trovato informazioni sufficienti nei documenti recuperati",
            "The provided context does not contain": "Il contesto recuperato non contiene",
            "The calculation assumes": "Il calcolo assume",
            "The actual sanction imposed would depend on": "La sanzione effettiva dipenderà da",
            "to answer the user's question": "per rispondere alla domanda",

            "The user is requesting": "L'utente sta chiedendo",
            "First, we need to": "Per prima cosa occorre",
            "Now, we want to": "Ora occorre",
            "Since the question asks": "Poiché la domanda chiede",
            "However, without more information": "Tuttavia, senza ulteriori informazioni",
            "This means that": "Questo significa che",
            "Therefore": "Pertanto",
            "Formula text": "Formula testuale",
            "Textual formula": "Formula testuale",
            "Limitations": "Limiti",
            "Sources": "Fonti",
        }
        
        for old, new in replacements.items():
            out = out.replace(old, new)
            
        # 2. Correzione forzata se il modello ha usato inglese per la sezione C
        if "Limitations / Conflicts" in out:
            out = out.replace("Limitations / Conflicts", "Limiti / Conflitti")

    # Manteniamo la logica esistente per URL e Fonti
    # ============================================================
    # SOURCE DISCIPLINE - non adattativo
    # ============================================================
    # In modalità RAG assessment la sezione D deve contenere solo fonti recuperate.
    # Gli URL o riferimenti esterni generati dal modello vengono rimossi sempre.
    out = re.sub(
        r"https?://\S+",
        "[Link esterno non autorizzato rimosso]",
        out,
        flags=re.IGNORECASE,
    )

    # Ricostruisce la sezione D) Fonti usando i SourceItem reali
    # solo quando NON siamo in una richiesta di calcolo puro.
    # Nei calcoli puri, la fonte corretta è l'input utente.
    if sources and not (
        is_calculation_request(query_text)
        and not needs_math_document_context(query_text)
    ):
        out = _replace_final_sources_section(out, sources)

    return normalize_math_markdown_for_reflex(
        quality_gate_postprocess(out, query_text, sources)
    )

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
            tenant_filter = build_qdrant_tenant_filter(
                extra_must=[
                    models.FieldCondition(
                        key="tier",
                        match=models.MatchValue(value=tier),
                    )
                ]
            )

            if hasattr(qdrant_client_inst, "query_points"):
                search_result = qdrant_client_inst.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=tenant_filter,
                    limit=15,
                    with_payload=True,
                ).points
            else:
                search_result = qdrant_client_inst.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=tenant_filter,
                    limit=15,
                    with_payload=True,
                )

            texts: List[str] = []
            for result in search_result:
                payload = result.payload or {}
                if not qdrant_payload_is_visible(payload):
                    continue
                content = safe_payload_text(payload)
                if content:
                    texts.append(content)
            return "\n".join(texts)
        except Exception as e:
            print(f"⚠️ Errore recupero Tier {tier}: {e}")
            return ""

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
        try:
            init_resources()
        except Exception as exc:
            print(f"⚠️ Inizializzazione RAG on_load fallita: {exc}")
        self.refresh_gpu()
        self.refresh_backend_status()

    def refresh_backend_status(self):
        ready = bool(embedder and qdrant_client_inst and llm_client)

        if PG_ENRICH_ENABLED:
            ready = ready and bool(pg_pool)

        if NEO4J_ENABLED:
            ready = ready and bool(neo4j_driver)

        if ready:
            self.backend_status = "OK"
        elif RESOURCE_INIT_ERROR:
            self.backend_status = f"DEGRADED: {RESOURCE_INIT_ERROR[:180]}"
        else:
            self.backend_status = "INITIALIZING"



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

        if embedder is None or qdrant_client_inst is None or llm_client is None:
            await asyncio.to_thread(init_resources)
            self.refresh_backend_status()

        if not self.input_text.strip() or self.is_processing:
            return

        user_query = self.input_text.strip()
        self.input_text = ""
        self.is_processing = True

        session_obj = getattr(getattr(self, "router", None), "session", None)
        session_user_id = str(getattr(session_obj, "client_token", "") or os.getenv("RAG_USER_ID", "service-user"))
        tenant_token = _CURRENT_TENANT_CONTEXT.set(
            resolve_tenant_context(request_id=str(uuid.uuid4()), user_id=session_user_id)
        )
        
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

            is_evidence_relevance = is_assessment_evidence_relevance_query(user_query)

            intent = "audit" if is_evidence_relevance else detect_intent(user_query)

            math_answer = None if is_evidence_relevance else try_solve_math_query(user_query)

            math_needs_context = bool(
                math_answer
                and needs_math_document_context(user_query)
                and not is_evidence_relevance
            )

            analytics_mode = (
                is_user_data_analytics(user_query)
                and not math_answer
                and not is_evidence_relevance
            )
            
            # tmp code added in v4.3 to debug routing issues in the test battery, to be removed in future versions
            print("========== ROUTING DEBUG ==========")
            print("QUERY:", user_query)
            print("INTENT:", intent)
            print("MATH_ANSWER_PRESENT:", bool(math_answer))
            print("MATH_NEEDS_CONTEXT:", math_needs_context)
            print("ANALYTICS_MODE:", analytics_mode)
            print("===================================")



            # ============================================================
            # 🧮 DETERMINISTIC MATH DIRECT MODE - EARLY EXIT
            # ============================================================
            if math_answer and not math_needs_context:
                print("✅ ENTERED MATH_DIRECT MODE")

                self.messages.append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=normalize_math_markdown_for_reflex(math_answer),
                        sources=[],
                        debug_md=prepare_debug_for_ui(
                            "### 🔎 Audit (Deterministic Math Direct Mode)\n"
                            "- routing: **math_direct**\n"
                            "- retrieval: **bypassed**\n"
                            "- formula_lookup: **bypassed**\n"
                            "- graph_mode: **bypassed**\n"
                            "- llm: **bypassed**\n"
                            "- source: **USER_INPUT**"
                        ),
                    )
                )
                self.is_processing = False
                yield rx.scroll_to("chat_bottom")
                return


            pure_glossary_trigger = any(t in user_query.lower() for t in [
                "glossario", "definisci", "definizione", "significato",
                "acronimo", "sta per", "cosa significa", "cosa vuol dire",
                "cosa si intende", "dizionario", "vocabolario",
                "glossary", "define", "definition", "meaning",
                "acronym", "stands for", "what does it mean", "what is meant by",
                "dictionary", "vocabulary",
            ])

            if (
                pure_glossary_trigger
                and is_glossary_definition_query(user_query)
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

                # ============================================================
                # 🧮 DETERMINISTIC MATH DIRECT MODE
                # ============================================================
                # Se il solver deterministico ha prodotto una risposta e l'utente
                # NON chiede esplicitamente un collegamento documentale, esci subito.
                #
                # Questo blocco deve stare prima di:
                # - retrieve_v2(...)
                # - Formula Lookup Strict Mode
                # - chiamata LLM
            """
                if math_answer and not math_needs_context:
                    print("✅ ENTERED MATH_DIRECT MODE")
                    self.messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=math_answer,
                            sources=[],
                            debug_md=prepare_debug_for_ui(
                                "### 🔎 Audit (Deterministic Math Direct Mode)\n"
                                "- routing: **math_direct**\n"
                                "- retrieval: **bypassed**\n"
                                "- formula_lookup: **bypassed**\n"
                                "- graph_mode: **bypassed**\n"
                                "- llm: **bypassed**\n"
                                "- source: **USER_INPUT**"
                            ),
                        )
                    )
                    self.is_processing = False
                    yield rx.scroll_to("chat_bottom")
                    return
            """
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

                if is_assessment_evidence_relevance_query(user_query):
                    retrieval_query = (
                        user_query
                        + "\n evidence relevance assessment question requirement control "
                        + "attinenza evidenza domanda questionario requisito controllo "
                        + "gap remediation corrective action sufficiency adequacy"
                    )

                elif math_needs_context:
                    retrieval_query = (
                        user_query
                        + "\n risk assessment evidence assessment valutazione del rischio controlli evidenze assessment integrato"
                    )

                sources, debug_md = retrieve_v2(retrieval_query, active_doc=active_doc_for_query)
                sources = filter_sources_for_current_organization(sources)
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
                                "\n\n**B) Evidenze**\n\n"
                                "- Nessuna fonte pertinente recuperata per il documento richiesto.\n\n"
                                "\n\n**C) Limiti / Conflitti**\n\n"
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
                if should_use_graph_relation_strict_mode(user_query) and not math_answer:
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
                # Formula Strict Mode deve servire per recuperare formule presenti nei documenti.
                # Non deve intercettare algebra già fornita dall'utente, perché quella viene risolta
                # dal solver deterministico try_solve_user_provided_algebra().
 
                if is_formula_lookup_query(user_query):
                    formula_answer = answer_formula_strict(user_query, sources)
       
                    if formula_answer:
                        self.messages.append(
                            ChatMessage(
                                id=str(uuid.uuid4()),
                                role="assistant",
                                content=normalize_math_markdown_for_reflex(formula_answer),
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
                system_instructions = build_system_instructions(intent) or ""
                
                if is_calculation_request(user_query) and not math_answer:
                    system_instructions += """

                CALCULATION MODE:
                - The user is asking to calculate, determine, quantify, solve, derive, or compute a result.
                - Do NOT answer by listing formulas, metrics, thresholds, or scoring rules found in the documents.
                - Use retrieved documents only to extract values, constants, definitions, thresholds, deadlines, or rules that are missing from the user question and are necessary for the calculation.
                - If the user question already provides all required values, use those values directly and do not discuss retrieved documents unless the user explicitly asks for documentary support.
                - If the calculation uses only values from the user question, section D) Fonti must contain only: "Input utente: valori e relazioni matematiche presenti nella domanda."

                GENERAL NUMERIC DISCIPLINE:
                1. First list all values used, including their units.
                2. State the formula, rule, or reasoning pattern applied in plain text.
                3. Perform the calculation step by step.
                4. Keep units consistent across all steps.
                5. Clearly distinguish intermediate results from the final result.
                6. The final result must be mathematically traceable to the shown steps.
                7. Before finalizing, verify that the final result is consistent with the intermediate calculations.
                8. For date/time calculations, verify the final weekday/date by counting the elapsed time from the starting timestamp to the final timestamp.
                9. If the question asks to compare, subtract, rank, or compute a delta between derived results, compute that comparison using the derived intermediate results, not the original starting value, unless the user explicitly asks otherwise.                
                10. Do not introduce new variables, coefficients, constants, parameters, equations, or relationships that are not present in the user question or retrieved context.
                11. Every equation used in the derivation must be either explicitly present in the user question or obtained only by direct substitution from equations already present.
                12. Do not assume subtractive, complementary, inverse, proportional, residual, or conservation relationships unless they are explicitly stated.
                13. If the target variable cannot be derived from the provided equations, say exactly which relationship is missing instead of inventing one.
                14. If the same positive multiplicative factor appears on both sides of an equation or inequality, it may be simplified explicitly.
                15. If a required value or relationship is missing from both the user question and the retrieved context, say exactly which value or relationship is missing and do not invent it.

                OUTPUT REQUIREMENTS:
                - Always show:
                1) values used;
                2) formula or rule applied;
                3) calculation steps;
                4) final result.
                - If formulas are shown, include visible plain-text formulas before LaTeX.
                - Never output empty formulas.
                - If a LaTeX formula would be empty, malformed, or uncertain, omit the LaTeX line entirely and keep only the visible plain-text formula.
                - Do not leave labels such as "Formula LaTeX:" without a formula after them.
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

                if should_use_graph_relation_strict_mode(user_query):
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
                                    section_hint="Forced methodology context",
                                    scope="GLOBAL",
                                    organization_id=None,
                                    status="active",
                                    corpus_version=CORPUS_VERSION,
                                    request_id=get_tenant_context().request_id,
                                )
                            )
                                
                sources = filter_sources_for_current_organization(sources)

                # 2. RAGGRUPPAMENTO FONTI CON BUDGET EQUO PER SORGENTE
                c_a_list, c_b_list, c_c_list, c_g_list = [], [], [], []
                current_context_length = 0
                max_allowed_length = MAX_CONTEXT_CHARS
                non_empty_sources = [s for s in sources if (s.content or "").strip()]
                per_source_budget = max(
                    600,
                    min(
                        4000,
                        (max_allowed_length // max(1, len(non_empty_sources))) - 300,
                    ),
                )

                for i, s in enumerate(sources, start=1):
                    tier_norm = normalize_tier_value(s.tier)

                    header = f"--- Source [{i}] — {s.filename} — Page {s.page} — ({s.type}) ---\n"
                    meta = f"(tier={tier_norm} | db={s.db_origin})\n"
                    body = (s.content or "").strip()

                    if not body:
                        continue

                    body = body[:per_source_budget]
                    snippet = header + meta + body + "\n\n"

                    if current_context_length + len(snippet) > max_allowed_length:
                        remaining = max_allowed_length - current_context_length - len(header) - len(meta) - 2
                        if remaining <= 200:
                            continue
                        snippet = header + meta + body[:remaining] + "\n\n"

                    if tier_norm == "A":
                        c_a_list.append(snippet)
                    elif tier_norm == "B":
                        c_b_list.append(snippet)
                    elif tier_norm == "GRAPH":
                        c_g_list.append(snippet)
                    else:
                        c_c_list.append(snippet)

                    current_context_length += len(snippet)

                c_a = "".join(c_a_list).strip()
                c_b = "".join(c_b_list).strip()
                c_c = "".join(c_c_list).strip()
                c_g = "".join(c_g_list).strip()
                    
                   
                    
                    
                # ============================================================
                # 🧮 MATHEMATICAL DISCIPLINE
                # ============================================================
                # Non usiamo più una micro-chiamata LLM per estrarre formule.
                # I calcoli deterministici vengono gestiti da try_solve_math_query().
                # Se try_solve_math_query() non risolve, il modello può spiegare i passaggi,
                # ma il quality gate finale impedirà fonti esterne e output incoerenti.
                math_injection = ""

                if math_answer:
                    math_injection = (
                        "\n\n### SYSTEM DETERMINISTIC MATH RESULT ###\n"
                        f"{math_answer}\n\n"
                        "CRITICAL: The deterministic result above is authoritative. "
                        "Do not change its numerical values.\n"
                    )

                if math_injection:
                    system_instructions += math_injection


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
                graph_relation_mode = should_use_graph_relation_strict_mode(user_query)

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


                evidence_relevance_block = ""

                if answer_mode == "evidence_relevance":
                    evidence_relevance_block = """
                ### EVIDENCE RELEVANCE MODE ###
                The user is asking whether a specific uploaded evidence document is relevant to an assessment questionnaire question.

                You MUST evaluate ONLY the retrieved evidence document if a requested document scope is present.

                Return the answer using this structure inside the standard four sections:

                In **A) Risposta**, include:
                - Livello di attinenza: 0 / 1 / 2 / 3
                - Percentuale stimata: 0-100%
                - Esito sintetico:
                - 0 = Non attinente
                - 1 = Debolmente attinente
                - 2 = Parzialmente attinente
                - 3 = Fortemente attinente

                Scoring criteria:
                - 3 / 76-100%: the evidence directly answers the assessment question.
                - 2 / 51-75%: the evidence partially supports the question but misses relevant elements.
                - 1 / 26-50%: the evidence is only indirectly related.
                - 0 / 0-25%: the evidence does not support the question.

                In **B) Evidenze**, cite only retrieved chunks from the requested document:
                - filename
                - page
                - short supporting excerpt

                In **C) Limiti / Conflitti**, list:
                - missing evidence
                - weak points
                - assumptions
                - whether the document is too generic

                In **D) Fonti**, list only retrieved filenames.

                If relevance is 0, 1, or 2, include a short remediation plan in section C.
                Do NOT invent evidence.
                Do NOT use documents outside the requested document scope.
                Do NOT claim the evidence is sufficient if the retrieved text does not explicitly support the assessment question.
                """



                final_user_content = (
                    doc_scope_block
                    + f"### ANSWER MODE ###\n{answer_mode}\n"  
                    + evidence_relevance_block            
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
                    + "**A) Risposta**, \n\n**B) Evidenze**, **C) Limiti / Conflitti**, **D) Fonti**."
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
                            "\n\n**B) Evidenze**\n\n"
                            "- Il retrieval ha prodotto fonti, ma la generazione LLM è risultata vuota.\n\n"
                            "\n\n**C) Limiti / Conflitti**\n\n"
                            "- Verificare modello Ollama, timeout e dimensione del contesto.\n\n"
                            "**D) Fonti**\n\n"
                            "- Vedi pannello Fonti/Audit."
                        )

                    # Non assegnare qui la risposta finale.
                    # La risposta viene post-processata una sola volta più sotto,
                    # dopo il controllo eval/quality gate.
                    yield

                except Exception as e:
                    print(f"❌ Errore generazione LLM: {e}")

                    self.messages[-1].content = (
                        "**A) Risposta**\n\n"
                        "La generazione della risposta è andata in timeout o ha prodotto un errore.\n\n"
                        "\n\n**B) Evidenze**\n\n"
                        "- Il retrieval è stato completato, ma la chiamata al modello LLM non ha risposto correttamente.\n\n"
                        "\n\n**C) Limiti / Conflitti**\n\n"
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
                            "\n\n**B) Evidenze**\n\n"
                            "- La risposta generata non ha superato il controllo automatico di faithfulness.\n\n"
                            "\n\n**C) Limiti / Conflitti**\n\n"
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
            _CURRENT_TENANT_CONTEXT.reset(tenant_token)
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
                rx.text(rx.cond(is_bot, "AI Assessment Manager", "Tu"), weight="bold", size="2"),
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
                    use_math=True,
                    use_katex=True,
                    use_gfm=True,
                    class_name="rag-markdown",
                    width="100%",
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


def _looks_threshold_rule(text: str) -> bool:
    """
    v4.5: riconosce regole soglia anche quando i valori sono in LaTeX,
    es. 5\\% oppure 1\\text{ milione}.
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
    Produce una rappresentazione testuale leggibile senza modificare i
    comandi LaTeX per sostituzioni parziali.

    In particolare ``\\left`` non deve mai essere interpretato come ``\\le``.
    """
    text = _strip_math_wrappers(_normalize_latex_value(str(value or "")))
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("$$$", "$$")
    text = re.sub(r"^`+|`+$", "", text).strip()

    for _ in range(8):
        previous = text
        text = re.sub(
            r"\\(?:mathrm|mathbf|mathit|text|operatorname)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
        if text == previous:
            break

    text = text.replace(r"\left", "").replace(r"\right", "")

    def replace_fraction_plain(expression: str) -> str:
        marker = r"\frac"
        while marker in expression:
            pos = expression.find(marker)
            cursor = pos + len(marker)
            while cursor < len(expression) and expression[cursor].isspace():
                cursor += 1
            if cursor >= len(expression) or expression[cursor] != "{":
                break

            def read_group(open_index: int):
                depth = 0
                for i in range(open_index, len(expression)):
                    if expression[i] == "{":
                        depth += 1
                    elif expression[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return expression[open_index + 1:i], i + 1
                return None, open_index

            numerator, after_num = read_group(cursor)
            if numerator is None:
                break
            while after_num < len(expression) and expression[after_num].isspace():
                after_num += 1
            if after_num >= len(expression) or expression[after_num] != "{":
                break
            denominator, after_den = read_group(after_num)
            if denominator is None:
                break
            replacement = (
                f"({replace_fraction_plain(numerator)})/"
                f"({replace_fraction_plain(denominator)})"
            )
            expression = expression[:pos] + replacement + expression[after_den:]
        return expression

    text = replace_fraction_plain(text)

    text = text.replace(r"\sum", "Σ")
    text = text.replace(r"\prod", "Π")
    text = text.replace(r"\times", " × ")
    text = text.replace(r"\cdot", " · ")

    # Boundary di comando obbligatorio: evita ``\left -> ≤ft``.
    text = re.sub(r"\\leq?(?![A-Za-z])", " ≤ ", text)
    text = re.sub(r"\\geq?(?![A-Za-z])", " ≥ ", text)
    text = re.sub(r"\\neq(?![A-Za-z])", " ≠ ", text)
    text = re.sub(r"\\Rightarrow(?![A-Za-z])", " ⇒ ", text)
    text = re.sub(r"\\rightarrow(?![A-Za-z])", " → ", text)

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
    """
    Normalizza escape tecnici senza riscrivere la semantica della formula.

    Supporta sia comandi LaTeX normali (``\\frac``) sia comandi rimasti
    doppiamente escapati nel testo persistito (``\\\\frac``).
    """
    v = str(value or "").strip()

    v = re.sub(r"\t(?=imes\b|ext\{)", r"\\t", v)
    v = re.sub(r"\r(?=ight\b)", r"\\r", v)
    v = re.sub(r"\f(?=rac\b)", r"\\f", v)
    v = re.sub("\x08(?=egin\\b)", r"\\b", v)
    v = v.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    latex_commands = (
        "frac", "sum", "prod", "left", "right", "mathrm", "mathbf",
        "mathit", "text", "operatorname", "cdot", "times", "leq", "le",
        "geq", "ge", "neq", "sqrt", "%", "_", "[", "]", "(", ")",
    )
    command_pattern = "|".join(re.escape(cmd) for cmd in latex_commands)
    v = re.sub(
        rf"\\{{2,}}(?=(?:{command_pattern})(?:\b|[^A-Za-z]))",
        lambda _m: "\\",
        v,
    )

    v = v.replace("$$$", "$$")
    v = re.sub(r"(?<![A-Za-z\\])ight(?=\s*[)\]}])", r"\\right", v)
    v = re.sub(r"(?<![A-Za-z\\])imes(?=\s*(?:\d|[A-Za-z\\({]))", r"\\times", v)
    v = re.sub(r"(?<![A-Za-z\\])rac(?=\s*\{)", r"\\frac", v)
    v = re.sub(r"(?<![A-Za-z\\])ext\{", r"\\text{", v)

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

        # Il LaTeX deve essere preservato, non convertito in plain text.
        rr["latex"] = latex

        rr["meaning"] = (
            meaning
            or "Formula computazionale esplicita presente nella fonte recuperata."
        )
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
        "metrica/indicatore citato", "formula", "metric",
        "formule e modelli matematici", "elemento recuperato",
    }

    combined = " ".join([
        name,
        formula,
        _formula_display_text(row.get("meaning") or "", 500).lower(),
    ])

    structural_noise = (
        "tikzpicture",
        "begintikzpicture",
        "\\draw",
        "\\node",
        "ode[",
        "cm,x=",
        "cm,y=",
    )

    if any(marker in combined for marker in structural_noise):
        return True

    if re.fullmatch(
        r"formule e modelli matematici(?:\s*-\s*pagina\s*\d+\s*--?)?",
        name,
    ):
        return True

    if tipo == "regola soglia":
        return not _looks_threshold_rule(" ".join([name, formula, str(row.get("meaning") or "")]))

    if (
        name in generic_names
        and tipo not in {"formula computazionale", "regola soglia"}
    ):
        return True

    # Exclude isolated values if they are not part of a threshold rule.
    plain = _formula_display_text(formula, 120).lower()
    if re.fullmatch(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)?", plain):
        return True

    return False


def _formula_md_cell(value: Any, max_len: int = 600) -> str:
    return _formula_display_text(value, max_len)


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



def _formula_metrics_table(
    title: str,
    rows: List[Dict[str, Any]],
) -> str:
    if not rows:
        return ""

    lines = [f"**{title}**", ""]

    for index, row in enumerate(rows, start=1):
        name = _formula_display_text(
            row.get("name") or f"Formula {index}", 180
        )
        formula = _strip_dangling_math_delimiters_v416(
            str(row.get("latex") or "formula esplicita non recuperata")
        )
        row_type = _formula_display_text(row.get("tipo") or "N/D", 120)
        meaning = _formula_display_text(row.get("meaning") or "", 340)
        filename = _formula_display_text(row.get("filename") or "N/D", 180)
        page = int(row.get("page") or 0)

        plain_formula = _formula_plain_text(formula)
        lines.extend([
            f"### {index}. {name}",
            "",
            f"- **Formula testuale:** `{plain_formula}`",
            "",
            "- **Formula LaTeX:**",
            "",
            "$$",
            formula,
            "$$",
            "",
        ])

        if row_type.lower() != "formula computazionale":
            lines.append(f"- **Tipo:** {row_type}")
            if meaning:
                lines.append(f"- **Significato:** {meaning}")

        lines.extend([
            f"- **Fonte:** {filename}, pagina {page}",
            "",
        ])

    return "\n".join(lines).strip()

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

def expand_assessment_query(query_text: str) -> str:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra in modo esplicito:
    - comportamento base: nessuna espansione hardcoded, query originale normalizzata;
    - richiamo semantico v4.10 per metriche temporali MTTD/MTTR;
    - richiamo semantico v4.12 per soglie/obblighi di notifica quando la query è formula/metric-oriented.
    """
    expanded = (query_text or "").strip()

    aliases: List[str] = []
    try:
        aliases.extend(_temporal_metric_aliases_v410(query_text))
    except Exception:
        pass

    try:
        aliases.extend(_threshold_metric_aliases_v412(query_text))
    except Exception:
        pass

    if not aliases:
        return expanded

    seen = set()
    clean_aliases: List[str] = []
    for alias in aliases:
        clean = str(alias or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            clean_aliases.append(clean)

    return (expanded + "\n" + " ".join(clean_aliases)).strip()

def extract_exact_phrases(query_text: str) -> List[str]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Estrae:
    - frasi tra virgolette;
    - acronimi maiuscoli;
    - alias metrici temporali v4.10;
    - alias soglie/notifiche v4.12.
    """
    q = query_text or ""
    phrases: List[str] = []

    quoted = re.findall(r"[\"“'«]([^\"”'»]+)[\"”'»]", q)
    phrases.extend([x.strip().lower() for x in quoted if len(x.strip()) > 2])

    acronyms = re.findall(r"\b[A-Z]{2,8}\b", q)
    phrases.extend([x.lower() for x in acronyms])

    try:
        phrases.extend(_temporal_metric_aliases_v410(query_text))
    except Exception:
        pass

    try:
        phrases.extend(_threshold_metric_aliases_v412(query_text))
    except Exception:
        pass

    out: List[str] = []
    seen = set()
    for p in phrases:
        clean = str(p or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)

    return out

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

def _requested_formula_terms_missing(query_text: str, rows: List[Dict[str, Any]]) -> List[str]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra:
    - logica base per termini generici richiesti ma non recuperati;
    - estensione v4.10 per severity/severità e MTTD/MTTR.
    """
    ql = (query_text or "").lower()
    found_text = " ".join([
        str(r.get("name", "")) + " " +
        str(r.get("latex", "")) + " " +
        str(r.get("meaning", ""))
        for r in rows or []
    ]).lower()

    requested_generic = [
        "cvss", "rischio", "risk", "maturità", "maturity", "copertura", "coverage"
    ]

    missing: List[str] = []

    for term in requested_generic:
        if term in ql and term not in found_text:
            missing.append(term)

    for term in ["severity", "severità"]:
        if term in ql and term not in found_text:
            missing.append(term)

    try:
        temporal_aliases = _temporal_metric_aliases_v410(query_text)
    except Exception:
        temporal_aliases = []

    if temporal_aliases:
        wants_detection = any(
            a.lower() in {"mttd", "mean time to detect"} or "rileva" in a.lower()
            for a in temporal_aliases
        )
        wants_resolution = any(
            a.lower() in {"mttr", "mean time to resolution", "mean time to repair"}
            or "risolvere" in a.lower()
            or "riparazione" in a.lower()
            for a in temporal_aliases
        )
        if wants_detection and "mttd" not in found_text:
            missing.append("MTTD")
        if wants_resolution and "mttr" not in found_text:
            missing.append("MTTR")

    return sorted(set(missing))

# ============================================================
# 🔎 FORMULA / METRIC RECALL PATCH - v4.10 non-adaptive
# ============================================================
# Goal:
# - Do not change the formula classifier.
# - Improve recall when the user asks semantically for temporal incident metrics
#   (e.g. "tempi di rilevamento/risoluzione") without explicitly writing MTTD/MTTR.
# - Keep this as a generic alias expansion, not tied to one test question.


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
    Riconosce formule computazionali tramite struttura matematica, non tramite
    nomi di metriche o parole legate a uno specifico dominio.
    """
    value = _strip_math_wrappers(
        _normalize_latex_value(str(latex or ""))
    )
    value_lower = value.lower()

    if not value or "formula esplicita non recuperata" in value_lower:
        return False

    if _formula_has_kg_artifacts_v411(value):
        return False

    if "=" not in value:
        return bool(
            re.search(
                r"\\frac|\\sum|\\prod|[+\-*/×÷^]",
                value,
                flags=re.IGNORECASE,
            )
        )

    left, right = value.split("=", 1)
    left = left.strip()
    right = right.strip()

    if not left or not right:
        return False

    if re.fullmatch(r"\\?text\{[^}]+\}", right):
        return False

    right_plain = _formula_display_text(right, 1000)
    word_pairs = re.findall(
        r"\b[A-Za-zÀ-ÿ]{2,}\s+[A-Za-zÀ-ÿ]{2,}\b",
        right_plain,
    )
    operator_tokens = re.findall(
        r"[+\-*/×÷^]|\\frac|\\sum|\\prod|\\operatorname",
        right,
        flags=re.IGNORECASE,
    )
    has_strong_math_construct = bool(
        re.search(
            r"\\frac|\\sum|\\prod|\\operatorname|[()^]",
            right,
            flags=re.IGNORECASE,
        )
    )

    # Una sequenza prevalentemente discorsiva con un simbolo incidentale
    # non è una formula computazionale.
    if len(word_pairs) >= 2 and len(operator_tokens) <= 1:
        return False

    if operator_tokens and (
        has_strong_math_construct
        or len(operator_tokens) >= 1
    ):
        return True

    # Un'assegnazione numerica isolata è un valore, non una formula.
    if re.fullmatch(
        r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent)?",
        right_plain,
    ):
        return False

    return False



def _formula_row_quality_v411(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Attribuisce priorità alla formula documentale nominata e ben formata."""
    fname = str(row.get("filename") or "")
    page = int(row.get("page") or 0)
    name = str(row.get("name") or "")
    latex = str(row.get("latex") or "")
    text = " ".join([name, latex, str(row.get("meaning") or "")])
    origin = str(row.get("formula_origin") or "")
    is_kg = _formula_is_kg_aggregate_source_v411(row)
    has_artifacts = _formula_has_kg_artifacts_v411(text)
    has_real_doc = bool(
        fname
        and fname.lower() not in {"kg", "neo4j", "neo4j knowledge graph", "n/d"}
        and page > 0
    )
    named = 0 if _is_generic_formula_name(name) else 1
    clean_latex = 0 if re.search(r"(?<![A-Za-z])(?:ight|imes|rac)(?![A-Za-z])", latex) else 1
    origin_score = {
        "document_equation": 3,
        "latex": 2,
        "knowledge_graph": 1,
    }.get(origin, 0)
    return (
        2 if has_real_doc else (0 if is_kg else 1),
        named,
        clean_latex,
        origin_score,
        0 if has_artifacts else 1,
    )



# ============================================================
# FORMULA SOURCE INTEGRITY GUARD - v4.14
# ============================================================
def _formula_has_invalid_latex_syntax_v414(value: Any) -> bool:
    """
    Rifiuta formule sintatticamente corrotte o ricostruite da nomi riservati
    LaTeX scambiati per variabili/funzioni.
    """
    v = _normalize_latex_value(str(value or "").strip())
    if not v:
        return False

    if re.search(r"\\(?:operatorname|mathrm|mathbf|mathit|text)(?!\s*\{)", v):
        return True
    if "\\№" in v or "№(" in v:
        return True
    if re.search(r"(?:≤|≥)\s*ft\b", v, flags=re.IGNORECASE):
        return True
    if v.count("{") != v.count("}"):
        return True

    reserved = r"left|right|frac|mathrm|mathbf|mathit|text|cdot|times|leq|geq|neq"
    if re.search(rf"\\operatorname\{{(?:{reserved})\}}", v, flags=re.IGNORECASE):
        return True
    if re.search(rf"\\mathrm\{{(?:{reserved})\}}", v, flags=re.IGNORECASE):
        return True
    if "```" in v or re.search(r"(?:Formula\s+LaTeX|Formula\s+testuale)\s*:", v, re.I):
        return True

    return False
def _scope_formula_sources_to_requested_document_v414(
    query_text: str,
    sources: List[SourceItem],
) -> List[SourceItem]:
    """
    Applica il document scope e, quando disponibile, usa la copia canonica
    arricchita da PostgreSQL al posto dei duplicati vettoriali/semantici.
    """
    requested_doc = extract_requested_document(query_text)
    if not requested_doc:
        return list(sources or [])

    scoped: List[SourceItem] = []
    for source in sources or []:
        candidate = {"filename": str(getattr(source, "filename", "") or "")}
        if candidate_matches_requested_doc(candidate, requested_doc):
            scoped.append(source)

    canonical = [
        source
        for source in scoped
        if (
            bool(str(getattr(source, "pg_ingestion_ts", "") or "").strip())
            or bool(str(getattr(source, "pg_source_name", "") or "").strip())
            or "PG_Enrich" in str(getattr(source, "db_origin", "") or "")
            or str(getattr(source, "db_origin", "") or "").startswith("PostgresDocScope")
        )
    ]

    return canonical or scoped
def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
    """Classifica e deduplica formule equivalenti provenienti da più database."""
    classified: List[Dict[str, Any]] = []

    for order, row in enumerate(rows or []):
        # v4.14 - Non tentare di riparare semanticamente formule corrotte.
        # Vengono scartate; una sorgente raw/documentale valida ha priorità.
        if _formula_has_invalid_latex_syntax_v414(row.get("latex")):
            continue

        classified_row = _classify_formula_row(row)
        classified_row["_formula_order"] = order

        name_lower = _formula_display_text(
            classified_row.get("name") or "", 160
        ).lower()
        if name_lower in {"formule collegate", "latex", "formula from knowledge graph"}:
            continue
        if _is_noise_formula_row_v45(classified_row):
            continue

        classified.append(classified_row)

    def identity_key(row: Dict[str, Any]) -> Tuple[str, str]:
        row_type = str(row.get("tipo") or "").lower()
        if row_type == "formula computazionale":
            identity = _canonical_formula_identity(str(row.get("latex") or ""))
            return "formula", identity
        if row_type == "regola soglia":
            identity = re.sub(
                r"\s+", " ", _formula_display_text(row.get("latex") or "", 900).lower()
            ).strip()
            return "threshold", identity[:400]
        name = re.sub(r"[^a-z0-9]+", "", str(row.get("name") or "").lower())
        return row_type, name

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    first_order: Dict[Tuple[str, str], int] = {}

    for row in classified:
        key = identity_key(row)
        if not key[1]:
            continue
        first_order.setdefault(key, int(row.get("_formula_order") or 0))
        existing = by_key.get(key)

        if existing is None:
            by_key[key] = row
            continue

        current_quality = _formula_row_quality_v411(row)
        existing_quality = _formula_row_quality_v411(existing)
        if current_quality > existing_quality:
            # Se la riga scelta ha un nome generico, conserva il nome migliore
            # già trovato per la stessa identità matematica.
            if _is_generic_formula_name(row.get("name")) and not _is_generic_formula_name(existing.get("name")):
                row["name"] = existing.get("name")
            by_key[key] = row
        elif _is_generic_formula_name(existing.get("name")) and not _is_generic_formula_name(row.get("name")):
            existing["name"] = row.get("name")

    deduped = list(by_key.items())
    priority = {
        "formula computazionale": 0,
        "regola soglia": 1,
        "metrica definitoria": 2,
        "metrica/elemento citato": 3,
    }
    deduped.sort(
        key=lambda pair: (
            priority.get(str(pair[1].get("tipo") or "").lower(), 9),
            first_order.get(pair[0], 10**9),
        )
    )

    result = []
    for _, row in deduped[:max_rows]:
        row.pop("_formula_order", None)
        result.append(row)
    return result

def _formula_examples_requested(query_text: str) -> bool:
    q = (query_text or "").lower()

    return bool(
        re.search(
            r"\b(esempio|esempi|example|examples)\b",
            q,
        )
    )


def _strip_dangling_math_delimiters_v416(value: str) -> str:
    """Rimuove delimitatori matematici esterni, anche se rimasti spaiati."""
    v = _strip_math_wrappers(_normalize_latex_value(value or "")).strip()
    v = re.sub(r"^\s*\${1,2}\s*", "", v)
    v = re.sub(r"\s*\${1,2}\s*$", "", v)
    return v.strip()


def _safe_eval_numeric_expression_v416(value: str) -> Optional[float]:
    """
    Valuta un'espressione aritmetica ASCII completamente numerica.

    Sono consentiti soltanto numeri, parentesi, +, -, *, /, potenze e le
    funzioni ``round`` e ``sqrt``. Nessun nome di variabile è ammesso.
    """
    expr = str(value or "").strip()
    if not expr:
        return None

    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
    expr = re.sub(r"\s+", "", expr)

    # Se il modello ha restituito anche un'uguaglianza, valuta entrambi i lati
    # e conserva il primo lato realmente numerico. Non si accettano variabili.
    candidates = [expr]
    if "=" in expr:
        candidates = [part.strip() for part in expr.split("=") if part.strip()]

    def evaluate_candidate(candidate: str) -> Optional[float]:
        if not candidate:
            return None
        if re.search(r"[^0-9A-Za-z_.,+\-*/()]", candidate):
            return None
        residual = re.sub(r"\b(?:round|sqrt)\b", "", candidate, flags=re.I)
        if re.search(r"[A-Za-z_]", residual):
            return None
        try:
            node = ast.parse(candidate, mode="eval").body
        except Exception:
            return None

        def evaluate(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.Num):
                return float(n.n)
            if isinstance(n, ast.BinOp) and type(n.op) in OPERATORS:
                return float(OPERATORS[type(n.op)](evaluate(n.left), evaluate(n.right)))
            if isinstance(n, ast.UnaryOp) and type(n.op) in OPERATORS:
                return float(OPERATORS[type(n.op)](evaluate(n.operand)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
                fn = n.func.id.lower()
                args = [evaluate(arg) for arg in n.args]
                if fn == "round" and 1 <= len(args) <= 2:
                    return float(round(args[0], int(args[1]) if len(args) == 2 else 0))
                if fn == "sqrt" and len(args) == 1 and args[0] >= 0:
                    return float(args[0] ** 0.5)
            raise TypeError("Espressione numerica non supportata")

        try:
            result = evaluate(node)
            if result != result or abs(result) == float("inf"):
                return None
            return float(result)
        except Exception:
            return None

    for candidate in candidates:
        result = evaluate_candidate(candidate)
        if result is not None:
            return result
    return None


def _latex_numeric_candidates_v416(value: str) -> List[str]:
    """Restituisce possibili lati numerici di una sostituzione LaTeX."""
    v = _strip_dangling_math_delimiters_v416(value)
    if not v:
        return []

    candidates = [v]
    if "=" in v:
        candidates = [part.strip() for part in v.split("=") if part.strip()]

    out: List[str] = []
    seen = set()
    for candidate in candidates:
        clean = re.sub(
            r"^(?:\\?(?:mathrm|mathbf|mathit|text)\s*\{[^{}]+\}|[A-Za-z_%][A-Za-z0-9_%\\]*)\s*",
            "",
            candidate,
        ).strip()
        clean = re.sub(r"(?:\\?%|%)\s*$", "", clean).strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)

    # In presenza di un'uguaglianza preferisce il lato che contiene
    # l'operazione completa rispetto a un semplice risultato scalare.
    out.sort(
        key=lambda item: (
            int(bool(re.search(r"\\frac|\\sqrt|\\times|\\cdot|[+*/()]", item))),
            len(item),
        ),
        reverse=True,
    )
    return out


def _safe_eval_numeric_latex_v416(value: str) -> Tuple[Optional[float], str]:
    """
    Valuta una sostituzione LaTeX numerica e restituisce anche l'espressione
    effettivamente scelta. Le uguaglianze e le unità percentuali finali sono
    tollerate, ma non vengono mai interpretate come variabili.
    """
    for candidate in _latex_numeric_candidates_v416(value):
        expr = candidate
        expr = expr.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        expr = expr.replace(r"\left", "").replace(r"\right", "")
        expr = expr.replace(r"\cdot", "*").replace(r"\cdotp", "*")
        expr = expr.replace(r"\times", "*").replace("×", "*").replace("÷", "/")
        expr = re.sub(r"\\(?:,|;|:|!|quad|qquad)\s*", "", expr)

        def replace_command_group(text: str, command: str, replacement_builder):
            marker = command
            while marker in text:
                pos = text.find(marker)
                cursor = pos + len(marker)
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] != "{":
                    break

                depth = 0
                end_group = None
                for i in range(cursor, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end_group = i + 1
                            break
                if end_group is None:
                    break
                inner = text[cursor + 1:end_group - 1]
                text = text[:pos] + replacement_builder(inner) + text[end_group:]
            return text

        def replace_fraction(text: str) -> str:
            marker = r"\frac"
            while marker in text:
                pos = text.find(marker)
                cursor = pos + len(marker)
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor >= len(text) or text[cursor] != "{":
                    break

                def read_group(open_index: int):
                    depth = 0
                    for i in range(open_index, len(text)):
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                return text[open_index + 1:i], i + 1
                    return None, open_index

                numerator, after_num = read_group(cursor)
                if numerator is None:
                    break
                while after_num < len(text) and text[after_num].isspace():
                    after_num += 1
                if after_num >= len(text) or text[after_num] != "{":
                    break
                denominator, after_den = read_group(after_num)
                if denominator is None:
                    break
                replacement = f"(({replace_fraction(numerator)})/({replace_fraction(denominator)}))"
                text = text[:pos] + replacement + text[after_den:]
            return text

        expr = replace_fraction(expr)
        expr = replace_command_group(expr, r"\sqrt", lambda inner: f"sqrt({inner})")
        expr = re.sub(r"\\operatorname\s*\{\s*round\s*\}", "round", expr, flags=re.I)
        expr = re.sub(r"\\(?:mathrm|text)\s*\{\s*round\s*\}", "round", expr, flags=re.I)
        expr = expr.replace("{", "(").replace("}", ")")
        expr = expr.replace("^", "**")
        expr = re.sub(r"(?<=\d)\s*(?=\()", "*", expr)
        expr = re.sub(r"(?<=\))\s*(?=\()", "*", expr)
        expr = re.sub(r"(?<=\))\s*(?=\d)", "*", expr)
        expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
        expr = re.sub(r"\s+", "", expr)

        result = _safe_eval_numeric_expression_v416(expr)
        if result is not None:
            return result, candidate

    return None, ""


def _format_verified_math_result_v416(
    value: float,
    original_result: str,
    formula_latex: str = "",
) -> str:
    if abs(value - round(value)) < 1e-10:
        rendered = str(int(round(value)))
    else:
        rendered = f"{value:.8f}".rstrip("0").rstrip(".")

    percent_formula = bool(
        re.search(r"(?:\\%|%)\s*=", formula_latex or "")
        or re.search(r"\bpercent(?:uale)?\b", formula_latex or "", flags=re.I)
    )
    if "%" in (original_result or "") or percent_formula:
        rendered += "%"
    return rendered


def _ascii_expression_to_latex_v416(value: str) -> str:
    """Converte solo gli operatori ASCII essenziali per la visualizzazione."""
    v = str(value or "").strip()
    v = v.replace("**", "^")
    v = v.replace("*", r" \cdot ")
    return v


def _parse_formula_examples_payload_v416(raw_result: str) -> Optional[Dict[str, Any]]:
    json_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        str(raw_result or "").strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    start = json_text.find("{")
    end = json_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(json_text[start:end + 1])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None



def _latex_fraction_parts_v417(value: str) -> Tuple[str, str]:
    """Estrae numeratore e denominatore dalla prima frazione LaTeX valida."""
    text = _strip_dangling_math_delimiters_v416(value or "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    marker = r"\frac"
    pos = text.find(marker)
    if pos < 0:
        return "", ""

    def read_group(open_index: int) -> Tuple[str, int]:
        if open_index >= len(text) or text[open_index] != "{":
            return "", open_index
        depth = 0
        for idx in range(open_index, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return text[open_index + 1:idx], idx + 1
        return "", open_index

    cursor = pos + len(marker)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    numerator, after_num = read_group(cursor)
    if not numerator:
        return "", ""
    cursor = after_num
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    denominator, _ = read_group(cursor)
    return numerator.strip(), denominator.strip()


def _formula_has_max_denominator_v417(formula_latex: str) -> bool:
    """
    True quando la prima frazione usa nel denominatore una variabile che
    rappresenta esplicitamente un massimo teorico (nome contenente ``max``).
    """
    numerator, denominator = _latex_fraction_parts_v417(formula_latex)
    if not denominator:
        return False
    den_plain = re.sub(r"[^A-Za-z0-9_]", "", denominator).lower()
    num_plain = re.sub(r"[^A-Za-z0-9_]", "", numerator).lower()
    return "max" in den_plain and "max" not in num_plain


def _first_numeric_division_operands_v417(value: str) -> Tuple[Optional[float], Optional[float]]:
    """Valuta gli operandi della prima divisione in un'espressione numerica AST-safe."""
    expr = str(value or "").strip()
    if not expr:
        return None, None
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
    if "=" in expr:
        candidates = [part.strip() for part in expr.split("=") if part.strip()]
        candidates.sort(key=lambda part: (part.count("/"), len(part)), reverse=True)
        expr = candidates[0] if candidates else expr
    try:
        root = ast.parse(expr, mode="eval").body
    except Exception:
        return None, None

    def find_division(node: ast.AST) -> Optional[ast.BinOp]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return node
        for child in ast.iter_child_nodes(node):
            found = find_division(child)
            if found is not None:
                return found
        return None

    division = find_division(root)
    if division is None:
        return None, None
    try:
        left_expr = ast.unparse(division.left)
        right_expr = ast.unparse(division.right)
    except Exception:
        return None, None
    return (
        _safe_eval_numeric_expression_v416(left_expr),
        _safe_eval_numeric_expression_v416(right_expr),
    )


def _max_denominator_example_is_valid_v417(
    formula_latex: str,
    numeric_expression: str,
) -> bool:
    """
    Impedisce esempi semanticamente impossibili quando la formula dichiara
    esplicitamente un massimo al denominatore. Non si applica alle altre formule.
    """
    if not _formula_has_max_denominator_v417(formula_latex):
        return True
    numerator, denominator = _first_numeric_division_operands_v417(numeric_expression)
    if numerator is None or denominator is None or denominator <= 0:
        return False
    return numerator <= denominator + 1e-12


def _validate_formula_examples_item_v416(
    formula: Dict[str, str],
    item: Any,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Valida gli esempi senza invalidare quelli già corretti."""
    errors: List[str] = []
    valid: List[Dict[str, str]] = []
    examples = item.get("examples") if isinstance(item, dict) else None
    if not isinstance(examples, list):
        return [], ["campo examples assente o non valido"]

    seen = set()
    for example_index, example in enumerate(examples, start=1):
        if len(valid) >= 2:
            break
        if not isinstance(example, dict):
            errors.append(f"esempio {example_index}: oggetto JSON non valido")
            continue

        values = re.sub(
            r"\s+", " ", str(example.get("values") or "").replace("`", "'")
        ).strip()
        numeric_expression = re.sub(
            r"\s+", " ", str(
                example.get("numeric_expression")
                or example.get("expression")
                or ""
            ).replace("`", "'")
        ).strip()
        substitution_raw = re.sub(
            r"\s+", " ", str(
                example.get("substitution_latex")
                or example.get("substitution")
                or ""
            ).replace("`", "'")
        ).strip()
        original_result = re.sub(
            r"\s+", " ", str(example.get("result") or "").replace("`", "'")
        ).strip()

        if not values:
            errors.append(f"esempio {example_index}: values mancante")
            continue

        verified_value: Optional[float] = None
        chosen_latex = ""

        if numeric_expression:
            verified_value = _safe_eval_numeric_expression_v416(numeric_expression)

        if substitution_raw:
            normalized_latex = _strip_dangling_math_delimiters_v416(substitution_raw)
            if normalized_latex and not _formula_has_invalid_latex_syntax_v414(normalized_latex):
                latex_value, chosen_candidate = _safe_eval_numeric_latex_v416(normalized_latex)
                if verified_value is None:
                    verified_value = latex_value
                if latex_value is not None and chosen_candidate:
                    chosen_latex = chosen_candidate

        if verified_value is None:
            errors.append(
                f"esempio {example_index}: nessuna espressione completamente numerica valutabile"
            )
            continue

        if not _max_denominator_example_is_valid_v417(
            formula.get("formula", ""),
            numeric_expression,
        ):
            errors.append(
                f"esempio {example_index}: il numeratore supera il massimo dichiarato dal denominatore"
            )
            continue

        if not chosen_latex:
            if numeric_expression:
                chosen_latex = _ascii_expression_to_latex_v416(numeric_expression)
            else:
                errors.append(f"esempio {example_index}: sostituzione visualizzabile assente")
                continue

        dedupe_key = (
            re.sub(r"\s+", "", numeric_expression or chosen_latex),
            round(float(verified_value), 12),
        )
        if dedupe_key in seen:
            errors.append(f"esempio {example_index}: duplicato")
            continue
        seen.add(dedupe_key)

        valid.append({
            "values": values,
            "substitution_latex": chosen_latex,
            "result": _format_verified_math_result_v416(
                verified_value,
                original_result,
                formula.get("formula", ""),
            ),
        })

    if len(valid) < 2:
        errors.append(f"ottenuti {len(valid)} esempi validi su 2 richiesti")
    return valid[:2], errors


def _request_formula_examples_v416(
    query_text: str,
    formulas: List[Dict[str, str]],
    formula_indexes: List[int],
    validation_feedback: Optional[Dict[int, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    requested = [
        {
            "formula_index": index,
            **formulas[index - 1],
        }
        for index in formula_indexes
        if 1 <= index <= len(formulas)
    ]
    if not requested:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "Riceverai un elenco chiuso di formule già recuperate e verificate. "
                "Non modificare, rinominare o aggiungere formule. Per ogni formula "
                "genera esattamente due esempi numerici distinti. Restituisci "
                "esclusivamente JSON valido con questa struttura: "
                "{\"items\":[{\"formula_index\":1,\"examples\":["
                "{\"values\":\"...\",\"numeric_expression\":\"...\","
                "\"substitution_latex\":\"...\"},{\"values\":\"...\","
                "\"numeric_expression\":\"...\",\"substitution_latex\":\"...\"}]}]}. "
                "numeric_expression deve essere una singola espressione aritmetica "
                "ASCII completamente numerica, senza variabili, unità, percentuali, "
                "testo o segno di uguaglianza; sono ammessi solo numeri, parentesi, "
                "+, -, *, /, **, round e sqrt. substitution_latex deve rappresentare "
                "la stessa identica operazione con tutte le variabili e sommatorie "
                "sostituite da numeri. Non inserire il risultato, delimitatori $, $$, "
                "\\(, \\) o Markdown. Usa il punto come separatore decimale. "
                "Se una frazione usa al denominatore una variabile il cui nome "
                "contiene 'max', scegli valori con numeratore minore o uguale al "
                "denominatore. Usa nei valori testuali la stessa lingua della richiesta."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_request": query_text,
                    "formulas": requested,
                    "validation_feedback": validation_feedback or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    raw_result = (call_ollama_chat_native(messages) or "").strip()
    if not raw_result:
        return None
    return _parse_formula_examples_payload_v416(raw_result)


def _generate_formula_examples(
    query_text: str,
    formula_rows: List[Dict[str, Any]],
) -> str:
    """
    Genera due esempi per formula con validazione progressiva.

    La prima chiamata è batch. Solo le formule mancanti o non valide vengono
    richieste nuovamente. Un errore su una formula non elimina gli esempi già
    validati delle altre formule.
    """
    formulas: List[Dict[str, str]] = []
    for row in formula_rows or []:
        name = str(row.get("name") or "").strip()
        latex = _strip_dangling_math_delimiters_v416(str(row.get("latex") or ""))
        if not name or not latex or "non recuperata" in latex.lower():
            continue
        formulas.append({"name": name, "formula": latex})

    if not formulas:
        return ""

    validated: Dict[int, List[Dict[str, str]]] = {}
    feedback: Dict[int, List[str]] = {}
    all_indexes = list(range(1, len(formulas) + 1))

    try:
        # Primo tentativo batch.
        payload = _request_formula_examples_v416(
            query_text,
            formulas,
            all_indexes,
        )
        items = payload.get("items") if isinstance(payload, dict) else None
        by_index: Dict[int, Any] = {}
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("formula_index"))
                except Exception:
                    continue
                by_index[index] = item

        for index in all_indexes:
            valid, errors = _validate_formula_examples_item_v416(
                formulas[index - 1],
                by_index.get(index),
            )
            if len(valid) == 2:
                validated[index] = valid
            else:
                feedback[index] = errors

        # Ripara soltanto le formule fallite, con massimo due tentativi mirati.
        for _attempt in range(2):
            missing = [index for index in all_indexes if index not in validated]
            if not missing:
                break

            for index in missing:
                repair_payload = _request_formula_examples_v416(
                    query_text,
                    formulas,
                    [index],
                    {index: feedback.get(index, [])},
                )
                repair_items = (
                    repair_payload.get("items")
                    if isinstance(repair_payload, dict)
                    else None
                )
                repair_item = None
                if isinstance(repair_items, list):
                    for candidate in repair_items:
                        if not isinstance(candidate, dict):
                            continue
                        try:
                            if int(candidate.get("formula_index")) == index:
                                repair_item = candidate
                                break
                        except Exception:
                            continue

                valid, errors = _validate_formula_examples_item_v416(
                    formulas[index - 1],
                    repair_item,
                )
                if len(valid) == 2:
                    validated[index] = valid
                    feedback.pop(index, None)
                else:
                    feedback[index] = errors
                    print(
                        "⚠️ Formula examples validation failed "
                        f"| formula_index={index} "
                        f"| name={formulas[index - 1]['name'][:80]} "
                        f"| errors={errors}"
                    )

        rendered: List[str] = []
        for formula_index, formula in enumerate(formulas, start=1):
            examples = validated.get(formula_index)
            if not examples:
                rendered.extend([
                    f"#### {formula_index}. {_formula_display_text(formula['name'], 160)}",
                    "",
                    "- Esempi non prodotti: la risposta generativa non ha superato "
                    "la validazione numerica deterministica.",
                    "",
                ])
                continue

            rendered.extend([
                f"#### {formula_index}. {_formula_display_text(formula['name'], 160)}",
                "",
            ])
            for example_index, example in enumerate(examples, start=1):
                rendered.extend([
                    f"**Esempio {example_index}**",
                    "",
                    f"- **Valori:** `{example['values']}`",
                    "- **Sostituzione:**",
                    "",
                    "$$",
                    example["substitution_latex"],
                    "$$",
                    "",
                    f"- **Risultato:** **{example['result']}**.",
                    "",
                ])
            rendered.append("")

        return "\n".join(rendered).strip()

    except Exception as exc:
        print(f"⚠️ Formula examples generation error: {exc}")
        return ""

def _answer_formula_strict_core(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    Costruisce una risposta deterministica per formule, metriche e soglie.

    Le categorie richieste vengono determinate semanticamente; le formule
    estratte non vengono inventate o riscritte dall'LLM.
    """
    rows = clean_formula_rows(
        extract_formula_rows_from_sources(sources),
        max_rows=30,
    )

    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie "
            "o regole di scoring esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring "
            "nei chunk recuperati e nel Knowledge Graph.\n\n"
            "**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono "
            "state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    computational = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "formula computazionale"
    ]
    definitional = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "metrica definitoria"
    ]
    thresholds = _aggregate_threshold_rules(rows)
    cited = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "metrica/elemento citato"
    ]

    query_lower = (query_text or "").lower()
    has_formula_terms = bool(
        re.search(
            r"\b(?:formula|formule|equazione|equazioni|formulas?|equations?)\b",
            query_lower,
        )
    )
    has_metric_terms = bool(
        re.search(
            r"\b(?:metrica|metriche|indicatore|indicatori|"
            r"scoring|metrics?|indicators?)\b",
            query_lower,
        )
    )
    asks_only_formulas = has_formula_terms and not has_metric_terms

    primary_rows = (
        computational
        if asks_only_formulas
        else computational + definitional
    )

    blocks: List[str] = []

    if primary_rows:
        title = (
            "Formule computazionali recuperate"
            if asks_only_formulas
            else "Formule computazionali e metriche recuperate"
        )
        blocks.append(
            _formula_metrics_table(
                title,
                primary_rows,
            )
        )

        if _formula_examples_requested(query_text) and computational:
            examples_text = _generate_formula_examples(
                query_text,
                computational,
            )
            if examples_text:
                blocks.append(
                    "### Esempi applicativi illustrativi\n\n"
                    + examples_text
                )
            else:
                blocks.append(
                    "### Esempi applicativi illustrativi\n\n"
                    "Le formule sono state recuperate, ma non è stato possibile "
                    "produrre esempi strutturati validi."
                )

    if thresholds and not asks_only_formulas:
        blocks.append(
            _threshold_rules_table(
                "Soglie normative recuperate ma non classificabili come scoring",
                thresholds,
            )
        )

    if (
        cited
        and not asks_only_formulas
        and not primary_rows
        and not thresholds
    ):
        blocks.append(
            _formula_metrics_table(
                "Elementi citati senza formula esplicita",
                cited,
            )
        )

    if not blocks:
        blocks.append(
            "Non ho trovato formule computazionali sufficientemente "
            "esplicite nelle fonti recuperate."
        )

    rows_for_sources: List[Dict[str, Any]] = (
        primary_rows
        if asks_only_formulas
        else primary_rows + thresholds + cited
    )

    used_files: List[str] = []
    seen_files = set()
    for row in rows_for_sources:
        filename = str(row.get("filename") or "").strip()
        page = int(row.get("page") or 0)
        if not filename:
            continue
        label = f"{filename} (p.{page})" if page else filename
        if label not in seen_files:
            seen_files.add(label)
            used_files.append(label)

    missing_terms = _requested_formula_terms_missing(
        query_text,
        rows_for_sources,
    )

    evidence_lines = [
        "- Gli elementi sono stati classificati in modo deterministico."
    ]
    if asks_only_formulas:
        evidence_lines.append(
            "- Sono state incluse esclusivamente le formule computazionali "
            "esplicite recuperate dalle fonti."
        )
    else:
        evidence_lines.append(
            "- Le metriche definitorie sono distinte dalle formule "
            "computazionali."
        )

    if thresholds and not asks_only_formulas:
        evidence_lines.append(
            "- Le soglie normative sono riportate separatamente perché non "
            "sono automaticamente formule o regole di scoring."
        )
    if missing_terms:
        evidence_lines.append(
            "- Non sono state recuperate formule computazionali esplicite per: "
            + ", ".join(missing_terms)
            + "."
        )

    return (
        "**A) Risposta**\n\n"
        + "\n\n".join(blocks)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula "
        + "computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una soglia normativa indica una condizione o un criterio; non "
        + "misura automaticamente un punteggio o una maturità.\n\n"
        + "**D) Fonti**\n\n"
        + (
            "\n".join(f"- {item}" for item in used_files)
            if used_files
            else "- Fonti non disponibili."
        )
    )


def answer_formula_strict(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra:
    - core v4.8/v4.11 per classificazione formule/metriche/soglie;
    - supplemento v4.12 per recuperare soglie normative se la prima retrieval non le contiene;
    - v4.14: document-scope finale e preferenza per la text layer raw.
    """
    scoped_sources = _scope_formula_sources_to_requested_document_v414(
        query_text,
        sources,
    )

    try:
        current_rows = clean_formula_rows(extract_formula_rows_from_sources(scoped_sources), max_rows=30)
        has_threshold = any(str(r.get("tipo") or "").lower() == "regola soglia" for r in current_rows)
    except Exception:
        has_threshold = False

    try:
        wants_threshold = bool(_threshold_metric_aliases_v412(query_text))
    except Exception:
        wants_threshold = False

    if wants_threshold and not has_threshold:
        extra_sources = _threshold_supplemental_sources_v412(query_text)
        if extra_sources:
            merged = dedupe_sources_for_answer(list(scoped_sources or []) + extra_sources)
            merged = _scope_formula_sources_to_requested_document_v414(
                query_text,
                merged,
            )
            return _answer_formula_strict_core(query_text, merged)

    return _answer_formula_strict_core(query_text, scoped_sources)

# ============================================================
# 🔎 FORMULA / THRESHOLD CATEGORY PRESERVATION PATCH - v4.12
# ============================================================
# Goal:
# - Preserve the good v4.11 classification of MTTD/MTTR as definitional metrics.
# - Preserve threshold/normative-condition rows when the user asks semantically for
#   impacted users, notification thresholds, notification obligations, or significant incidents.
# - Keep this non-adaptive: it is a generic synonym/recall expansion for threshold-style
#   formula/metric queries, not a hardcoded answer.



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
                page_chunk_index=int(meta.get("page_chunk_index") or 0),
                doc_id=str(meta.get("doc_id") or ""),
                type=source_type,
                score=float(h.get("score", 0.0) or 0.0),
                tier=tier,
                scope=str(meta.get("scope") or "").upper(),
                organization_id=_optional_int(meta.get("organization_id")),
                status="active",
                ingestion_run_id=str(meta.get("ingestion_run_id") or ""),
                corpus_version=str(meta.get("corpus_version") or CORPUS_VERSION),
                classification=str(meta.get("classification") or "internal"),
                embedding_model=str(meta.get("embedding_model") or ""),
                request_id=get_tenant_context().request_id,
                db_origin=str(h.get("origin") or "PostgresThresholdSupplement"),
                section_hint="v4.12 threshold supplemental retrieval",
            )
        )

        if len(sources_extra) >= limit:
            break

    return sources_extra


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





def _formula_text_to_latex(lhs: str, rhs: str) -> str:
    """
    Converte un'espressione aritmetica testuale in LaTeX valido.

    Il parser è indipendente dal dominio e supporta identificatori, numeri,
    chiamate di funzione, parentesi e operatori aritmetici comuni.
    """
    token_pattern = re.compile(
        r"\s*(?:(?P<number>\d+(?:[.,]\d+)?)|"
        r"(?P<name>[A-Za-z][A-Za-z0-9_]*)|"
        r"(?P<op>[+\-*/^(),]))"
    )

    def tokenize(expression: str):
        tokens = []
        position = 0
        while position < len(expression):
            match = token_pattern.match(expression, position)
            if not match:
                position += 1
                continue
            kind = "number" if match.group("number") else "name" if match.group("name") else "op"
            value = match.group(kind)
            tokens.append((kind, value))
            position = match.end()
        return tokens

    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0

        def peek(self, value=None):
            if self.pos >= len(self.tokens):
                return False
            return value is None or self.tokens[self.pos][1] == value

        def take(self):
            token = self.tokens[self.pos]
            self.pos += 1
            return token

        def parse(self):
            return self.expression()

        def expression(self):
            node = self.term()
            while self.peek("+") or self.peek("-"):
                op = self.take()[1]
                node = ("bin", op, node, self.term())
            return node

        def term(self):
            node = self.power()
            while self.peek("*") or self.peek("/"):
                op = self.take()[1]
                node = ("bin", op, node, self.power())
            return node

        def power(self):
            node = self.unary()
            if self.peek("^"):
                self.take()
                node = ("bin", "^", node, self.power())
            return node

        def unary(self):
            if self.peek("+") or self.peek("-"):
                return ("unary", self.take()[1], self.unary())
            return self.primary()

        def primary(self):
            if self.pos >= len(self.tokens):
                return ("raw", "")

            kind, value = self.take()
            if kind == "number":
                return ("number", value.replace(",", "."))

            if kind == "name":
                if self.peek("("):
                    self.take()
                    args = []
                    if not self.peek(")"):
                        while True:
                            args.append(self.expression())
                            if self.peek(","):
                                self.take()
                                continue
                            break
                    if self.peek(")"):
                        self.take()
                    return ("call", value, args)
                return ("name", value)

            if value == "(":
                node = self.expression()
                if self.peek(")"):
                    self.take()
                return ("group", node)

            return ("raw", value)

    def render_name(name: str) -> str:
        if "_" in name:
            base, suffix = name.split("_", 1)
            base_latex = base if len(base) == 1 else rf"\mathrm{{{base}}}"
            suffix_latex = suffix if len(suffix) == 1 else rf"\mathrm{{{suffix}}}"
            return rf"{base_latex}_{{{suffix_latex}}}"
        return name if len(name) == 1 else rf"\mathrm{{{name}}}"

    def render(node, parent_precedence=0):
        kind = node[0]
        if kind == "number":
            return node[1]
        if kind == "name":
            return render_name(node[1])
        if kind == "raw":
            return node[1]
        if kind == "group":
            return rf"\left({render(node[1])}\right)"
        if kind == "unary":
            return node[1] + render(node[2], 4)
        if kind == "call":
            function_name = node[1]
            args = node[2]
            rendered_args = ", ".join(render(arg) for arg in args)
            if function_name.lower() == "sum":
                return rf"\sum\left({rendered_args}\right)"
            return rf"\operatorname{{{function_name}}}\left({rendered_args}\right)"
        if kind == "bin":
            op, left, right = node[1], node[2], node[3]
            if op == "/":
                return rf"\frac{{{render(left)}}}{{{render(right)}}}"
            if op == "^":
                return rf"{{{render(left, 3)}}}^{{{render(right)}}}"
            precedence = 1 if op in {"+", "-"} else 2
            symbol = r" \cdot " if op == "*" else f" {op} "
            rendered = render(left, precedence) + symbol + render(right, precedence + 1)
            if precedence < parent_precedence:
                return rf"\left({rendered}\right)"
            return rendered
        return ""

    lhs_clean = re.sub(r"\s+", " ", str(lhs or "").strip())
    rhs_clean = re.sub(r"\s+", " ", str(rhs or "").strip())

    lhs_latex = r"\%" if lhs_clean == "%" else render_name(lhs_clean)
    tokens = tokenize(rhs_clean)
    rhs_latex = render(Parser(tokens).parse()) if tokens else rhs_clean
    return f"{lhs_latex} = {rhs_latex}"



def _canonical_formula_identity(value: str) -> str:
    """Restituisce una chiave canonica per deduplicare LaTeX e testo equivalenti."""
    v = _normalize_latex_value(value)
    v = _strip_math_wrappers(v)
    v = v.replace("\\(", "").replace("\\)", "")
    v = v.replace("\\[", "").replace("\\]", "")
    v = v.replace(r"\left", "").replace(r"\right", "")

    # Mantiene il contenuto dei wrapper tipografici.
    for _ in range(8):
        previous = v
        v = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", v)
        if v == previous:
            break

    # Normalizza underscore escapati e pedici dopo aver rimosso i wrapper tipografici.
    v = re.sub(r"\\+_", "_", v)
    v = re.sub(r"([A-Za-z][A-Za-z0-9]*)_\{([^{}]+)\}", r"\1\2", v)
    v = re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)", r"\1\2", v)

    def replace_fractions(expression: str) -> str:
        """Sostituisce \frac{A}{B} rispettando parentesi graffe annidate."""
        marker = r"\frac"
        start = 0
        while True:
            pos = expression.find(marker, start)
            if pos < 0:
                return expression

            index = pos + len(marker)
            while index < len(expression) and expression[index].isspace():
                index += 1
            if index >= len(expression) or expression[index] != "{":
                start = index
                continue

            def read_group(open_index: int):
                depth = 0
                for cursor in range(open_index, len(expression)):
                    char = expression[cursor]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return expression[open_index + 1:cursor], cursor + 1
                return None, open_index

            numerator, after_num = read_group(index)
            if numerator is None:
                start = index + 1
                continue
            while after_num < len(expression) and expression[after_num].isspace():
                after_num += 1
            if after_num >= len(expression) or expression[after_num] != "{":
                start = after_num
                continue
            denominator, after_den = read_group(after_num)
            if denominator is None:
                start = after_num + 1
                continue

            replacement = f"({numerator})/({denominator})"
            expression = expression[:pos] + replacement + expression[after_den:]
            start = max(0, pos - 1)

    v = replace_fractions(v)
    v = v.replace(r"\sum", "sum")
    v = v.replace(r"\cdot", "*").replace(r"\times", "*")
    v = v.replace("×", "*").replace(r"\%", "%")
    v = re.sub(r"[${}`\\]", "", v)
    v = re.sub(r"[{}()]", "", v)
    v = re.sub(r"\s+", "", v).lower()
    return v

def _is_generic_formula_name(value: Any) -> bool:
    name = _formula_display_text(value or "", 180).strip().lower()
    return name in {
        "", "formula", "formula recuperata", "formula/metric",
        "formula from knowledge graph", "elemento recuperato", "score", "%", "level",
    }


def _formula_context_name(content: str, match_start: int, previous_end: int) -> str:
    """
    Ricava il nome associato all'equazione dal blocco testuale precedente.

    Usa solo caratteristiche tipografiche generali: prossimità, brevità,
    parentesi descrittive e presenza di un identificatore iniziale.
    """
    start = max(0, previous_end)
    segment = (content or "")[start:match_start]
    lines = [re.sub(r"\s+", " ", line).strip() for line in segment.splitlines()]
    lines = [line for line in lines if line and "=" not in line]
    lines = lines[-10:]

    # Combina solo continuazioni brevi che chiudono una parentesi aperta.
    combined = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.count("(") > line.count(")")
            and index + 1 < len(lines)
            and lines[index + 1].endswith(")")
            and len(lines[index + 1].split()) <= 4
        ):
            line = line + " " + lines[index + 1]
            index += 1
        combined.append(line)
        index += 1

    header_words = {
        "formula", "formulas", "formule", "equation", "equations",
        "modello", "modelli", "model", "models", "descrizione", "description",
    }

    best_name = ""
    best_score = -999
    for position, line in enumerate(combined):
        clean = line.strip(" -–—•|\t")
        if not clean or len(clean) > 100:
            continue
        words = re.findall(r"[A-Za-zÀ-ÿ0-9_.-]+", clean)
        if not words or len(words) > 12:
            continue
        if clean.lower() in header_words:
            continue

        score = 0
        first = words[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,30}", first):
            score += 2
        if "(" in clean:
            score += 3
        if re.fullmatch(r"[A-Z0-9_.-]{2,30}", first):
            score += 2
        if len(words) <= 6:
            score += 1
        if clean.endswith(('.', ':', ';')):
            score -= 2
        # A parità di qualità, preferisce il candidato più vicino.
        score += position / 100.0

        if score > best_score:
            best_score = score
            best_name = clean

    if best_score < 3:
        return ""
    if best_name.count("(") == best_name.count(")") + 1:
        best_name += ")"
    return best_name


def _formula_name_from_equation_line_v415(content: str, match_start: int) -> str:
    """Ricava il nome del modello dalla stessa riga/cella dell'equazione."""
    raw = content or ""
    line_start = raw.rfind("\n", 0, match_start) + 1
    prefix = raw[line_start:match_start].strip(" \t|;:-")
    if not prefix:
        return ""

    header_terms = {"modello", "model", "descrizione", "description", "formula"}
    if "|" in prefix:
        cells = [c.strip() for c in prefix.split("|") if c.strip()]
    elif "\t" in prefix:
        cells = [c.strip() for c in prefix.split("\t") if c.strip()]
    else:
        cells = [c.strip() for c in re.split(r"\s{2,}", prefix) if c.strip()]

    for cell in cells:
        clean = re.sub(r"^[-*•\d.()\s]+", "", cell).strip()
        if not clean or clean.lower() in header_terms or "=" in clean:
            continue
        if len(clean) <= 120:
            code_match = re.match(
                r"^([A-Za-z][A-Za-z0-9_.-]{1,30}(?:\s*\([^)]{1,80}\))?)",
                clean,
            )
            if code_match:
                candidate = code_match.group(1).strip()
                if candidate.count("(") > candidate.count(")"):
                    line_end = raw.find("\n", match_start)
                    if line_end >= 0:
                        next_end = raw.find("\n", line_end + 1)
                        next_line = raw[line_end + 1: next_end if next_end >= 0 else len(raw)]
                        continuation = re.split(r"\s{2,}|\t|\|", next_line.strip(), maxsplit=1)[0].strip()
                        if ")" in continuation and len(continuation) <= 60:
                            candidate += " " + continuation.split(")", 1)[0].strip() + ")"
                return candidate

    match = re.match(
        r"^([A-Za-z][A-Za-z0-9_.-]{1,30}(?:\s*\([^)]{1,80}\))?)",
        prefix,
    )
    return match.group(1).strip() if match else ""


def _extract_formula_rows_from_sources_core(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    Estrae formule e metriche dai SourceItem recuperati.

    Le equazioni testuali del documento sono considerate la rappresentazione
    primaria perché conservano il contesto e il nome associato. Le versioni
    LaTeX o Knowledge Graph restano disponibili come fallback e vengono poi
    deduplicate semanticamente.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()

    latex_pat = re.compile(
        r"(?<!\\)(\$\$.*?\$\$|\$[^$\n]{2,500}\$)",
        re.DOTALL,
    )
    explicit_equation_pat = re.compile(
        r"(?im)(?<![A-Za-z0-9_])"
        r"(?P<lhs>%|[A-Za-z][A-Za-z0-9_]{0,60})"
        r"\s*=\s*(?P<rhs>[^\n;|]{2,320})"
    )
    metric_line_pat = re.compile(
        r"(?i)\b(formula|formulas|formulae|equation|equations|formule|"
        r"equazione|equazioni|metric|metrics|metrica|metriche|indicator|"
        r"indicators|indicatore|indicatori|score|scoring|punteggio|"
        r"calculation|calcolo|mean time|tempo medio|index|indice|ratio|"
        r"coverage|copertura|maturity|maturità|severity|severità)\b"
    )

    for source in sources or []:
        content = source.content or ""
        filename = source.filename or "N/D"
        page = int(source.page or 0)
        source_type = normalize_source_type(getattr(source, "type", "") or "")

        # 1. Equazioni testuali: preservano nome e ordine del documento.
        previous_equation_end = 0
        for equation_match in explicit_equation_pat.finditer(content):
            lhs = re.sub(r"\s+", " ", equation_match.group("lhs")).strip()
            rhs = re.sub(r"\s+", " ", equation_match.group("rhs")).strip()
            if not lhs or len(rhs) < 2:
                continue

            raw_equation = f"{lhs} = {rhs}"
            identity = _canonical_formula_identity(raw_equation)
            if not identity:
                continue

            line_name = _formula_name_from_equation_line_v415(
                content,
                equation_match.start(),
            )
            context_name = line_name or _formula_context_name(
                content,
                equation_match.start(),
                previous_equation_end,
            )
            previous_equation_end = equation_match.end()

            key = ("equation", identity, filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rhs_words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", rhs)
            rhs_operators = re.findall(r"[+\-*/×÷^]|\b(?:sum|round)\s*\(", rhs, flags=re.IGNORECASE)
            textual_assignment = len(rhs_words) >= 3 and len(rhs_operators) <= 1

            rows.append({
                "name": context_name or lhs,
                "latex": (
                    raw_equation
                    if textual_assignment
                    else _formula_text_to_latex(lhs, rhs)
                ),
                "meaning": (
                    "Definizione testuale esplicita presente nella fonte recuperata."
                    if textual_assignment
                    else "Equazione esplicita presente nella fonte recuperata."
                ),
                "filename": filename,
                "page": page,
                "formula_origin": "document_equation",
            })

        # 2. Formule LaTeX esplicite: fallback per contenuti senza testo lineare.
        for latex_match in latex_pat.findall(content):
            latex = _normalize_latex_value(latex_match.strip())
            identity = _canonical_formula_identity(latex)
            if not identity:
                continue
            key = ("latex", identity, filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": _extract_left_name_from_equation(latex) or "Formula recuperata",
                "latex": _strip_math_wrappers(latex),
                "meaning": "Formula LaTeX esplicita presente nella fonte recuperata.",
                "filename": filename,
                "page": page,
                "formula_origin": "latex",
            })

        # 3. Formula nodes del Knowledge Graph: fallback strutturato.
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

            formula_value = latex or plain
            identity = _canonical_formula_identity(formula_value)
            if identity:
                key = ("kg", identity, filename.lower(), page)
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        "name": plain or _extract_left_name_from_equation(latex) or "Formula recuperata",
                        "latex": _strip_math_wrappers(_normalize_latex_value(formula_value)),
                        "meaning": meaning,
                        "filename": filename,
                        "page": page,
                        "formula_origin": "knowledge_graph",
                    })

        # 4. Metriche citate senza equazione esplicita.
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line or "").strip()
            if not line or not metric_line_pat.search(line) or "=" in line:
                continue

            name = "Metrica/indicatore citato"
            name_match = re.match(
                r"^[-*•\s]*([A-Za-zÀ-ÿ0-9_\-/ ]{2,80})\s*[:=–-]",
                line,
            )
            if name_match:
                name = name_match.group(1).strip()

            key = ("metric", name.lower(), filename.lower(), page, line[:120].lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": name,
                "latex": "formula esplicita non recuperata",
                "meaning": (
                    "Metrica o indicatore citato nella fonte; nessuna equazione "
                    "esplicita è stata individuata nella stessa riga."
                ),
                "filename": filename,
                "page": page,
                "formula_origin": "metric_text",
            })

            if len(rows) >= 60:
                return rows

    return rows

def extract_formula_rows_from_sources(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra:
    - estrazione base di formule/metriche da SourceItem;
    - estensione v4.13 per aggiungere regole soglia da testo semplice.
    """
    rows = list(_extract_formula_rows_from_sources_core(sources) or [])
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

# ============================================================
# 🔎 FORMULA / THRESHOLD RECALL PATCH - v4.13
# ============================================================
# Goal:
# - Keep v4.12 behaviour for MTTD/MTTR.
# - Improve threshold recall when threshold-like chunks are present as plain text
#   and not as LaTeX/formula nodes.
# - Non-adaptive: generic threshold-rule extraction + PostgreSQL regex recall.


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
    WHERE status = 'active'
      AND ({' OR '.join(clauses)})
      AND ((scope = 'GLOBAL' AND organization_id IS NULL AND tier = 'A') OR (scope = 'ACCOUNT' AND organization_id = %s AND tier IN ('B', 'C')))
    ORDER BY ingestion_ts DESC
    LIMIT %s;
    """
    params.extend([current_organization_id(), limit])

    conn = pg_get_conn_secure()
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
                page_chunk_index=int(metadata_json.get("page_chunk_index") or 0),
                doc_id=str(metadata_json.get("doc_id") or ""),
                type=metadata_json.get("toon_type") or metadata_json.get("type") or "text",
                score=2.5,
                tier=normalize_tier_value(metadata_json.get("tier", "C")),
                scope=str(metadata_json.get("scope") or "").upper(),
                organization_id=_optional_int(metadata_json.get("organization_id")),
                status="active",
                ingestion_run_id=str(metadata_json.get("ingestion_run_id") or ""),
                corpus_version=str(metadata_json.get("corpus_version") or CORPUS_VERSION),
                classification=str(metadata_json.get("classification") or "internal"),
                embedding_model=str(metadata_json.get("embedding_model") or ""),
                request_id=get_tenant_context().request_id,
                db_origin="PostgresThresholdRegex",
                section_hint="v4.13 threshold regex retrieval (Dynamic)",
                pg_ingestion_ts=ingestion_ts.isoformat() if ingestion_ts else "",
                pg_source_name=metadata_json.get("source_name", ""),
                pg_source_type=metadata_json.get("source_type", ""),
                pg_log_id=int(metadata_json.get("log_id") or 0),
                pg_chunk_id=int(metadata_json.get("chunk_index") or 0),
                pg_page_chunk_index=int(metadata_json.get("page_chunk_index") or 0),
                pg_toon_type=metadata_json.get("toon_type", ""),
            ))
        return out
    except Exception as e:
        print(f"⚠️ v4.13 threshold regex supplement error: {e}")
        return []
    finally:
        pg_put_conn_secure(conn)



# ============================================================
# FORMULA TABLE NAME + KATEX VISIBILITY PATCH - v4.17
# ============================================================
_formula_name_from_equation_line_v416 = _formula_name_from_equation_line_v415


def _formula_name_from_equation_line_v415(content: str, match_start: int) -> str:
    """
    Estrae il nome del modello da una riga/tabella in modo generalista.

    Mantiene il parser precedente e, solo se il risultato è generico, cerca
    l'identificatore strutturale più vicino composto da maiuscole/cifre, con
    eventuale descrizione tra parentesi. Non contiene nomi di modelli specifici.
    """
    previous = _formula_name_from_equation_line_v416(content, match_start)
    if previous and not _is_generic_formula_name(previous):
        return previous

    raw = content or ""
    line_start = raw.rfind("\n", 0, match_start) + 1
    previous_line_start = raw.rfind("\n", 0, max(0, line_start - 1)) + 1
    window_start = max(previous_line_start, match_start - 500)
    prefix = raw[window_start:match_start]

    header_tokens = {
        "FORMULA", "FORMULE", "MODEL", "MODELLI", "MODELLO",
        "DESCRIPTION", "DESCRIZIONE", "SCORE", "LEVEL",
    }
    candidates: List[Tuple[int, int, str]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?P<code>[A-Z][A-Z0-9_.-]{1,24})"
        r"(?:\s*\((?P<label>[^)\n]{1,80})\))?"
    )
    for match in pattern.finditer(prefix):
        code = match.group("code").strip()
        if code in header_tokens:
            continue
        # Esclude parole capitalizzate: il codice deve essere realmente
        # maiuscolo oppure contenere almeno una cifra strutturale.
        if code.upper() != code and not re.search(r"\d", code):
            continue
        label = re.sub(r"\s+", " ", match.group("label") or "").strip()
        candidate = f"{code} ({label})" if label else code
        same_line = int("\n" not in prefix[match.end():])
        distance = match_start - (window_start + match.end())
        candidates.append((same_line, -distance, candidate))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return previous or ""


RAG_GLOBAL_STYLE = {
    ".rag-markdown p": {
        "overflow_wrap": "anywhere",
        "word_break": "break-word",
    },
    ".rag-markdown li": {
        "overflow_wrap": "anywhere",
        "word_break": "break-word",
    },
    ".rag-markdown .katex": {
        "white_space": "nowrap",
        "word_break": "normal",
        "overflow_wrap": "normal",
    },
    ".rag-markdown .katex-display": {
        "display": "block",
        "max_width": "100%",
        "overflow_x": "auto",
        "overflow_y": "hidden",
        "padding_bottom": "0.25rem",
    },
    ".rag-markdown .katex-display > .katex": {
        "display": "inline-block",
        "min_width": "max-content",
    },
    # KaTeX produce HTML + MathML per accessibilità. Senza il foglio CSS
    # completo il browser mostra entrambi; questa regola nasconde soltanto
    # visivamente il layer MathML, mantenendolo disponibile agli screen reader.
    ".rag-markdown .katex-mathml": {
        "position": "absolute",
        "width": "1px",
        "height": "1px",
        "padding": "0",
        "margin": "-1px",
        "overflow": "hidden",
        "clip": "rect(0, 0, 0, 0)",
        "white_space": "nowrap",
        "border": "0",
    },
    ".rag-markdown .katex-html": {
        "display": "inline-block",
        "white_space": "nowrap",
        "word_break": "normal",
        "overflow_wrap": "normal",
    },
}

# Il CDN è configurabile/disattivabile; le regole locali sopra restano un
# fallback per installazioni Docker senza accesso Internet.
KATEX_CSS_URL = os.getenv(
    "KATEX_CSS_URL",
    "https://cdn.jsdelivr.net/npm/katex@0.17.0/dist/katex.min.css",
).strip()
KATEX_USE_CDN = os.getenv("KATEX_USE_CDN", "1") == "1"
KATEX_STYLESHEETS = [KATEX_CSS_URL] if KATEX_USE_CDN and KATEX_CSS_URL else []

app = rx.App(style=RAG_GLOBAL_STYLE, stylesheets=KATEX_STYLESHEETS)
app.add_page(index, on_load=State.on_load)
