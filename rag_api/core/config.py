"""Configurazione centralizzata del servizio RAG.

Questo modulo contiene esclusivamente configurazione statica e lettura delle
variabili ambiente. Non apre connessioni, non carica modelli e non contiene
logica di retrieval o dipendenze da Reflex.

Derivato dalla configurazione presente nell'ultimo ``gui_reflex.py`` del PoC.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------------
# Runtime process environment
# Deve essere applicato prima di importare torch / sentence_transformers.
# -----------------------------------------------------------------------------
def configure_process_environment() -> None:
    """Imposta valori conservativi per evitare oversubscription della CPU."""

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cpu_threads = os.getenv("EMBED_CPU_THREADS", "4").strip() or "4"
    os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
    os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpu_threads)


configure_process_environment()


# -----------------------------------------------------------------------------
# Helpers di lettura ambiente
# -----------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"{name} deve essere un booleano: 1/0, true/false, yes/no oppure on/off"
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un intero") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un numero") from exc


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


# Whitelist di sicurezza usata dalle query Neo4j.
DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS: Final[tuple[str, ...]] = (
    # Structure / classification
    "IS_A",
    "PART_OF",
    "HAS_COMPONENT",
    "CONTAINS",
    "BELONGS_TO",
    "APPLIES_TO",
    "IMPLEMENTS",
    "USES",
    "MAPS_TO",
    "DEFINES",
    "CLASSIFIES",


    # Governance / compliance
    "COMPLIES_WITH",
    "NON_COMPLIANT_WITH",
    "HAS_COMPLIANCE_STATUS",
    "HAS_GAP",
    "REQUIRES_REMEDIATION",
    "REMEDIATES",
    "MANDATES",
    "REQUIRES",
    "GOVERNS",
    "APPROVES",
    "REVIEWS",
    "ASSIGNS_RESPONSIBILITY_TO",
    

    # Incident / process
    "TRIGGERS",
    "ACTIVATES",
    "STARTS",
    "FOLLOWS",
    "PRECEDES",
    "LEADS_TO",
    "ESCALATES_TO",
    "MANAGES",
    "HANDLES",
    "CONTAINS_INCIDENT",
    "ERADICATES",
    "RECOVERS",
    "CLOSES",
    "IMPROVES",

    # Notification / authority
    "NOTIFIES",
    "REPORTS_TO",
    "RECEIVES_NOTIFICATION",
    "REQUIRES_NOTIFICATION_TO",
    "HAS_DEADLINE",
    "HAS_RECIPIENT",
    "COMMUNICATES_TO",

    # Framework / implementation
    "SUPPORTS",
    "ENABLES",
    "DEPENDS_ON",
    "ALIGNS_WITH",
    "CONTRIBUTES_TO",
    "MEASURES",
    "MONITORS",

    # Risk / security
    "MITIGATES",
    "REDUCES",
    "THREATENS",
    "EXPLOITS",
    "PROTECTS",
    "VULNERABLE_TO",
    "COMPROMISES",
    "IMPACTS",
    "AFFECTS",
    "EXPOSES",

    # Audit / evidence
    "GENERATES",
    "VERIFIES",
    "TESTS",
    "DEMONSTRATES",
    "DOCUMENTS",
    "SUPPORTS_EVIDENCE_FOR",
    "EVIDENCES",
    "SATISFIES",
    "REFERENCES_REQUIREMENT",

    # Formula graph
    "HAS_FORMULA",
    "HAS_VARIABLE",
)


# Alias deterministici condivisi con l'ingestion. Sono ammessi soltanto mapping
# che non cambiano polarità o significato della relazione.
DEFAULT_NEO4J_RELATIONSHIP_ALIASES: Final[dict[str, str]] = {
    "IMPLEMENT": "IMPLEMENTS",
    "IMPLEMENTED": "IMPLEMENTS",
    "IMPLEMENTING": "IMPLEMENTS",
    "IMPLEMENTES": "IMPLEMENTS",
    "COMPLIANT": "COMPLIES_WITH",
    "COMPLIES": "COMPLIES_WITH",
    "COMPLY": "COMPLIES_WITH",
    "CONFORME_A": "COMPLIES_WITH",
    "CONFORMITY_WITH": "COMPLIES_WITH",
    "NON_COMPLIANT": "NON_COMPLIANT_WITH",
    "NON_COMPLIANCE": "NON_COMPLIANT_WITH",
    "NON_CONFORMITY": "NON_COMPLIANT_WITH",
    "NOT_COMPLIANT": "NON_COMPLIANT_WITH",
    "GAP_WITH": "NON_COMPLIANT_WITH",
    "VIOLATES": "NON_COMPLIANT_WITH",
    "STATUS": "HAS_COMPLIANCE_STATUS",
    "COMPLIANCE_STATUS": "HAS_COMPLIANCE_STATUS",
    "IMPLEMENTATION_STATUS": "HAS_COMPLIANCE_STATUS",
    "MATURITY_STATUS": "HAS_COMPLIANCE_STATUS",
    "GAP": "HAS_GAP",
    "HAS_FINDING": "HAS_GAP",
    "HAS_DEFICIENCY": "HAS_GAP",
    "HAS_WEAKNESS": "HAS_GAP",
    "REMEDIATION_REQUIRED": "REQUIRES_REMEDIATION",
    "NEEDS_REMEDIATION": "REQUIRES_REMEDIATION",
    "REQUIRES_CORRECTIVE_ACTION": "REQUIRES_REMEDIATION",
    "CORRECTS": "REMEDIATES",
    "CLOSES_GAP": "REMEDIATES",
    "NOTIFICATION_TO": "NOTIFIES",
    "SENDS_NOTIFICATION_TO": "NOTIFIES",
    "MUST_NOTIFY": "NOTIFIES",
    "REPORT_TO": "REPORTS_TO",
    "REPORTS": "REPORTS_TO",
    "ESCALATES_REPORT_TO": "REPORTS_TO",
    "DEADLINE": "HAS_DEADLINE",
    "TIME_LIMIT": "HAS_DEADLINE",
    "WITHIN_HOURS": "HAS_DEADLINE",
    "WITHIN_DAYS": "HAS_DEADLINE",
    "RESPONSIBLE": "ASSIGNS_RESPONSIBILITY_TO",
    "ACCOUNTABLE": "ASSIGNS_RESPONSIBILITY_TO",
    "OWNER": "ASSIGNS_RESPONSIBILITY_TO",
    "ASSIGNED_TO": "ASSIGNS_RESPONSIBILITY_TO",
}


class RagSettings(BaseModel):
    """Configurazione immutabile del backend RAG."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    # ------------------------------------------------------------------
    # PoC / tenant bootstrap
    # Il vero TenantContext verrà costruito nel futuro core/tenant.py.
    # ------------------------------------------------------------------
    poc_mode: bool = True
    poc_organization_id: int = 1234
    default_user_id: str = "service-user"
    default_user_roles: tuple[str, ...] = ("user",)
    allowed_scopes: tuple[str, ...] = ("GLOBAL", "ACCOUNT")
    corpus_version: str = "v1"
    rag_default_tiers: tuple[str, ...] = ("A", "B", "C")

    # ------------------------------------------------------------------
    # PostgreSQL / TimescaleDB
    # ------------------------------------------------------------------
    pg_enrich_enabled: bool = True
    pg_host: str = "127.0.0.1"
    pg_port: int = 5433
    pg_database: str = "assessment_ingestion"
    pg_user: str = "admin"
    pg_password: str = "admin_password"
    pg_min_connections: int = 1
    pg_max_connections: int = 8
    pg_prefer_raw: bool = False
    pg_auto_harden_schema: bool = False
    pg_enforce_least_privilege: bool = False

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6334
    qdrant_collection: str = "assessment_docs"

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_enabled: bool = True
    neo4j_uri: str = "bolt://127.0.0.1:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "admin_password"
    neo4j_allowed_relationships: tuple[str, ...] = Field(
        default=DEFAULT_NEO4J_ALLOWED_RELATIONSHIPS
    )

    # ------------------------------------------------------------------
    # Modelli e device
    # ------------------------------------------------------------------
    llm_model_name: str = "gemma4:12b"
    embedding_model_name: str = "/workspace/models/bge-m3"
    reranker_model_name: str = "/workspace/models/ms-marco-reranker"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"

    # ------------------------------------------------------------------
    # Ollama
    # ``ollama_openai_url`` serve solo agli eventuali endpoint compatibili
    # OpenAI; la generazione principale userà ``ollama_native_chat_url``.
    # ------------------------------------------------------------------
    ollama_openai_url: str = "http://127.0.0.1:11434/v1"
    ollama_api_key: str = "ollama"
    ollama_native_chat_url: str = "http://127.0.0.1:11434/api/chat"
    ollama_connect_timeout_seconds: int = 300
    llm_timeout_seconds: int = 300
    llm_num_ctx: int = 16384
    llm_num_predict: int = 4096
    llm_temperature: float = 0.15
    llm_repeat_penalty: float = 1.15

    # ------------------------------------------------------------------
    # Conversazione e retrieval
    # ------------------------------------------------------------------
    memory_limit: int = 3
    qdrant_candidates: int = 100
    rerank_candidates: int = 35
    final_sources: int = 8
    max_per_page: int = 2
    max_per_document: int = 5

    tier_boost_a: float = 0.08
    tier_boost_b: float = 0.04
    tier_penalty_c: float = 0.015
    tier_c_penalty_if_not_evidence: bool = True

    graph_expand_enabled: bool = True
    graph_max_formulas: int = 6
    graph_max_neighbor_chunks: int = 4

    max_context_chars: int = 24000
    max_assistant_chars: int = 15000

    # ------------------------------------------------------------------
    # Audit ed evaluation
    # ------------------------------------------------------------------
    log_dir: Path = Path.home() / "ai_rag_logs"
    audit_enabled: bool = True
    audit_log_path: Path = Path.home() / "ai_rag_logs" / "rag_audit.jsonl"

    evaluation_enabled: bool = False
    evaluation_model_name: str = "gemma4:12b"
    evaluation_log_path: Path = Path.home() / "ai_rag_logs" / "rag_eval_log.jsonl"
    evaluation_max_context_chars: int = 12000
    evaluation_min_faithfulness: float = 0.75
    evaluation_min_answer_relevance: float = 0.70
    evaluation_strict_block: bool = False
    evaluation_temperature: float = 0.0
    evaluation_repeat_penalty: float = 1.05

    @field_validator(
        "poc_organization_id",
        "pg_port",
        "pg_min_connections",
        "pg_max_connections",
        "qdrant_port",
        "ollama_connect_timeout_seconds",
        "llm_timeout_seconds",
        "llm_num_ctx",
        "llm_num_predict",
        "memory_limit",
        "qdrant_candidates",
        "rerank_candidates",
        "final_sources",
        "max_per_page",
        "max_per_document",
        "graph_max_formulas",
        "graph_max_neighbor_chunks",
        "max_context_chars",
        "max_assistant_chars",
        "evaluation_max_context_chars",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("il valore deve essere maggiore di zero")
        return value

    @field_validator(
        "evaluation_min_faithfulness",
        "evaluation_min_answer_relevance",
        "llm_temperature",
        "evaluation_temperature",
    )
    @classmethod
    def validate_probability_like_value(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("il valore deve essere compreso tra 0 e 1")
        return value

    @field_validator("llm_repeat_penalty", "evaluation_repeat_penalty")
    @classmethod
    def validate_repeat_penalty(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("repeat_penalty deve essere maggiore di zero")
        return value

    @field_validator("default_user_roles", "allowed_scopes", "rag_default_tiers")
    @classmethod
    def validate_non_empty_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("la lista non può essere vuota")
        return cleaned

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.upper() for item in value)
        unknown = sorted(set(normalized) - {"GLOBAL", "ACCOUNT"})
        if unknown:
            raise ValueError(f"scope non validi: {', '.join(unknown)}")
        return normalized

    @field_validator("rag_default_tiers")
    @classmethod
    def validate_default_tiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.upper() for item in value)
        unknown = sorted(set(normalized) - {"A", "B", "C"})
        if unknown:
            raise ValueError(f"tier non validi: {', '.join(unknown)}")
        return normalized

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "RagSettings":
        if self.pg_min_connections > self.pg_max_connections:
            raise ValueError(
                "PG_MIN_CONN non può essere maggiore di PG_MAX_CONN"
            )

        if self.final_sources > self.rerank_candidates:
            raise ValueError(
                "FINAL_SOURCES non può essere maggiore di RERANK_CANDIDATES"
            )

        if self.rerank_candidates > self.qdrant_candidates:
            raise ValueError(
                "RERANK_CANDIDATES non può essere maggiore di QDRANT_CANDIDATES"
            )

        if self.poc_mode and self.poc_organization_id <= 0:
            raise ValueError(
                "POC_ORGANIZATION_ID deve essere positivo quando POC_MODE è attivo"
            )

        return self

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        return self.neo4j_user, self.neo4j_password

    def ensure_runtime_directories(self) -> None:
        """Crea le directory necessarie ai log, fallendo allo startup se non scrivibili."""

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation_log_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> RagSettings:
    """Legge e valida una sola volta la configurazione del processo."""

    poc_mode = _env_bool("POC_MODE", True)
    llm_model_name = _env_str("LLM_MODEL_NAME", "gemma4:12b")

    log_dir = Path(
        _env_str("RAG_LOG_DIR", str(Path.home() / "ai_rag_logs"))
    ).expanduser()

    audit_log_path = Path(
        _env_str("AUDIT_LOG_PATH", str(log_dir / "rag_audit.jsonl"))
    ).expanduser()

    evaluation_log_path = Path(
        _env_str("EVAL_LOG_PATH", str(log_dir / "rag_eval_log.jsonl"))
    ).expanduser()

    settings = RagSettings(
        # PoC / tenant bootstrap
        poc_mode=poc_mode,
        poc_organization_id=_env_int("POC_ORGANIZATION_ID", _env_int("ORGANIZATION_ID", 1234)),
        default_user_id=_env_str("RAG_USER_ID", "service-user"),
        default_user_roles=_env_csv("RAG_USER_ROLES", ("user",)),
        allowed_scopes=_env_csv("RAG_ALLOWED_SCOPES", ("GLOBAL", "ACCOUNT")),
        corpus_version=_env_str("CORPUS_VERSION", "v1"),
        rag_default_tiers=_env_csv("RAG_DEFAULT_TIERS", ("A", "B", "C")),

        # PostgreSQL
        pg_enrich_enabled=_env_bool("PG_ENRICH_ENABLED", True),
        pg_host=_env_str("PG_HOST", "127.0.0.1"),
        pg_port=_env_int("PG_PORT", 5433),
        pg_database=_env_str("PG_DB", "assessment_ingestion"),
        pg_user=_env_str("PG_USER", "admin"),
        pg_password=_env_str("PG_PASS", "admin_password"),
        pg_min_connections=_env_int("PG_MIN_CONN", 1),
        pg_max_connections=_env_int("PG_MAX_CONN", 8),
        pg_prefer_raw=_env_bool("PG_PREFER_RAW", False),
        pg_auto_harden_schema=_env_bool("PG_AUTO_HARDEN_SCHEMA", False),
        pg_enforce_least_privilege=(
            False
            if poc_mode
            else _env_bool("PG_ENFORCE_LEAST_PRIVILEGE", True)
        ),

        # Qdrant
        qdrant_host=_env_str("QDRANT_HOST", "127.0.0.1"),
        qdrant_port=_env_int("QDRANT_PORT", 6334),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "assessment_docs"),

        # Neo4j
        neo4j_enabled=_env_bool("NEO4J_ENABLED", True),
        neo4j_uri=_env_str("NEO4J_URI", "bolt://127.0.0.1:7688"),
        neo4j_user=_env_str("NEO4J_USER", "neo4j"),
        neo4j_password=_env_str(
            "NEO4J_PASS",
            _env_str("NEO4J_PASSWORD", "admin_password"),
        ),

        # Modelli/device
        llm_model_name=llm_model_name,
        embedding_model_name=_env_str(
            "EMBEDDING_MODEL_NAME", "/workspace/models/bge-m3"
        ),
        reranker_model_name=_env_str(
            "RERANKER_MODEL_NAME", "/workspace/models/ms-marco-reranker"
        ),
        embedding_device=_env_str("EMBED_DEVICE", "cpu"),
        reranker_device=_env_str("RERANK_DEVICE", "cpu"),

        # Ollama/generazione
        ollama_openai_url=_env_str(
            "OLLAMA_URL", "http://127.0.0.1:11434/v1"
        ),
        ollama_api_key=_env_str("OLLAMA_API_KEY", "ollama"),
        ollama_native_chat_url=_env_str(
            "OLLAMA_NATIVE_CHAT_URL", "http://127.0.0.1:11434/api/chat"
        ),
        ollama_connect_timeout_seconds=_env_int(
            "OLLAMA_CONNECT_TIMEOUT_S", 300
        ),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_S", 300),
        llm_num_ctx=_env_int("LLM_NUM_CTX", 16384),
        llm_num_predict=_env_int("LLM_NUM_PREDICT", 4096),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.15),
        llm_repeat_penalty=_env_float("LLM_REPEAT_PENALTY", 1.15),

        # Retrieval
        memory_limit=_env_int("MEMORY_LIMIT", 3),
        qdrant_candidates=_env_int("QDRANT_CANDIDATES", 100),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 35),
        final_sources=_env_int("FINAL_SOURCES", 8),
        max_per_page=_env_int("MAX_PER_PAGE", 2),
        max_per_document=_env_int("MAX_PER_DOC", 5),
        tier_boost_a=_env_float("TIER_BOOST_A", 0.08),
        tier_boost_b=_env_float("TIER_BOOST_B", 0.04),
        tier_penalty_c=_env_float("TIER_PENALTY_C", 0.015),
        tier_c_penalty_if_not_evidence=_env_bool(
            "TIER_C_PENALTY_IF_NOT_EVIDENCE", True
        ),
        graph_expand_enabled=_env_bool("GRAPH_EXPAND_ENABLED", True),
        graph_max_formulas=_env_int("GRAPH_MAX_FORMULAS", 6),
        graph_max_neighbor_chunks=_env_int("GRAPH_MAX_NEIGHBOR_CHUNKS", 4),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 24000),
        max_assistant_chars=_env_int("MAX_ASSISTANT_CHARS", 15000),

        # Audit/evaluation
        log_dir=log_dir,
        audit_enabled=_env_bool("AUDIT_ENABLED", True),
        audit_log_path=audit_log_path,
        evaluation_enabled=_env_bool("EVAL_ENABLED", False),
        evaluation_model_name=_env_str("EVAL_MODEL_NAME", llm_model_name),
        evaluation_log_path=evaluation_log_path,
        evaluation_max_context_chars=_env_int("EVAL_MAX_CONTEXT_CHARS", 12000),
        evaluation_min_faithfulness=_env_float("EVAL_MIN_FAITHFULNESS", 0.75),
        evaluation_min_answer_relevance=_env_float(
            "EVAL_MIN_ANSWER_RELEVANCE", 0.70
        ),
        evaluation_strict_block=_env_bool("EVAL_STRICT_BLOCK", False),
        evaluation_temperature=_env_float("EVAL_TEMPERATURE", 0.0),
        evaluation_repeat_penalty=_env_float("EVAL_REPEAT_PENALTY", 1.05),
    )

    settings.ensure_runtime_directories()
    return settings


# Singleton immutabile da importare negli altri moduli:
#     from core.config import settings
settings: Final[RagSettings] = load_settings()
