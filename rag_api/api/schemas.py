"""Schemi pubblici dell'API RAG.

Il modulo definisce esclusivamente il contratto JSON esposto dalla futura API
FastAPI. Non contiene logica di retrieval, accesso ai database, risoluzione del
tenant o dipendenze da Reflex.

Principi di sicurezza:
- ``organization_id``, ruoli e privilegi non sono accettati nel body pubblico;
- il contesto tenant sarà ricavato da credenziali trusted nel layer API;
- campi non dichiarati vengono rifiutati con ``extra='forbid'``;
- la cronologia accetta soltanto messaggi ``user`` e ``assistant`` per impedire
  l'iniezione di messaggi ``system`` da parte del client.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------------
# Limiti del contratto HTTP
# I limiti runtime più restrittivi restano demandati a ``core/config.py`` e al
# servizio applicativo.
# -----------------------------------------------------------------------------
MAX_QUERY_CHARS = 20_000
MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_MESSAGE_CHARS = 30_000
MAX_TARGET_PAGES = 50
MAX_PUBLIC_SOURCE_EXCERPT_CHARS = 5_000
MAX_WARNING_CHARS = 2_000


class ApiSchema(BaseModel):
    """Base comune per tutti i modelli JSON pubblici."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
    )


def utc_now() -> datetime:
    """Restituisce un timestamp UTC timezone-aware."""

    return datetime.now(timezone.utc)


# =============================================================================
# ENUMERAZIONI PUBBLICHE
# =============================================================================
class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class RagIntent(StrEnum):
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    CHART = "chart"
    AUDIT = "audit"


class RagAnswerMode(StrEnum):
    KNOWLEDGE = "knowledge"
    AUDIT = "audit"
    EVIDENCE_RELEVANCE = "evidence_relevance"


class RagExecutionMode(StrEnum):
    """Ramo applicativo che ha materialmente prodotto la risposta."""

    RAG_GENERATION = "rag_generation"
    MATH_DIRECT = "math_direct"
    GLOSSARY_DIRECT = "glossary_direct"
    GRAPH_RELATION_STRICT = "graph_relation_strict"
    FORMULA_STRICT = "formula_strict"
    ANALYTICS = "analytics"


class AnswerFormat(StrEnum):
    MARKDOWN = "markdown"


class ApiStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class DependencyState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    DISABLED = "disabled"


class ServiceState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ApiErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    TENANT_CONTEXT_ERROR = "tenant_context_error"
    RESOURCE_NOT_READY = "resource_not_ready"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    EVALUATION_ERROR = "evaluation_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


# =============================================================================
# INPUT API
# =============================================================================
class ConversationMessage(ApiSchema):
    """Messaggio storico fornito dal client.

    Il ruolo ``system`` è intenzionalmente assente: le istruzioni di sistema
    sono costruite esclusivamente dal backend.
    """

    role: ChatRole
    content: str = Field(min_length=1, max_length=MAX_HISTORY_MESSAGE_CHARS)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content non può essere vuoto")
        return cleaned


class RagQueryOptions(ApiSchema):
    """Opzioni non privilegiate controllabili dal client.

    Non sono presenti filtri tenant, scope o ruoli. Anche ``target_document``
    restringe soltanto il corpus già visibile al tenant autenticato.
    """

    include_sources: bool = True
    include_debug: bool = False
    include_evaluation: bool = False

    target_document: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description=(
            "Nome del documento al quale restringere la ricerca. Non modifica "
            "il perimetro tenant e non accetta percorsi filesystem."
        ),
    )
    target_pages: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Pagine 1-based del documento da privilegiare o restringere.",
    )
    max_sources: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Numero massimo richiesto di fonti pubbliche. Il backend applica "
            "comunque il limite configurato e può restituirne meno."
        ),
    )

    @field_validator("target_document")
    @classmethod
    def validate_target_document(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        if not cleaned:
            return None

        # Il client può indicare un nome logico, non un percorso locale.
        if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
            raise ValueError("target_document deve essere un nome file, non un percorso")

        if "\x00" in cleaned:
            raise ValueError("target_document contiene caratteri non validi")

        return cleaned

    @field_validator("target_pages")
    @classmethod
    def validate_target_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))

        if len(normalized) > MAX_TARGET_PAGES:
            raise ValueError(
                f"target_pages può contenere al massimo {MAX_TARGET_PAGES} pagine"
            )

        if any(page <= 0 for page in normalized):
            raise ValueError("target_pages usa numerazione 1-based e accetta solo valori positivi")

        return normalized


class RagQueryRequest(ApiSchema):
    """Body del futuro ``POST /api/v1/rag/query``."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Identificativo applicativo della conversazione lato client.",
    )
    history: tuple[ConversationMessage, ...] = Field(default_factory=tuple)
    options: RagQueryOptions = Field(default_factory=RagQueryOptions)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query non può essere vuota")
        if "\x00" in cleaned:
            raise ValueError("query contiene caratteri null non validi")
        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        if not cleaned:
            return None

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}", cleaned):
            raise ValueError(
                "conversation_id può contenere lettere, numeri, punto, trattino, "
                "underscore, due punti e @"
            )
        return cleaned

    @field_validator("history")
    @classmethod
    def validate_history_length(
        cls,
        value: tuple[ConversationMessage, ...],
    ) -> tuple[ConversationMessage, ...]:
        if len(value) > MAX_HISTORY_MESSAGES:
            raise ValueError(
                f"history può contenere al massimo {MAX_HISTORY_MESSAGES} messaggi"
            )
        return value

    @model_validator(mode="after")
    def validate_history_sequence(self) -> "RagQueryRequest":
        # Non rendiamo obbligatoria l'alternanza perfetta: il RagService potrà
        # normalizzare messaggi consecutivi dello stesso ruolo come fa il PoC.
        # Vietiamo però che l'ultimo messaggio storico replichi la query corrente
        # come messaggio user, per evitare duplicazioni accidentali nel prompt.
        if self.history:
            last = self.history[-1]
            if last.role == ChatRole.USER and last.content.strip() == self.query.strip():
                raise ValueError(
                    "l'ultimo messaggio history non deve duplicare la query corrente"
                )
        return self

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "examples": [
                {
                    "query": (
                        "Valuta se il documento evidenza_12_policy_generica_"
                        "parziale_debole.pdf è attinente alla domanda di assessment: "
                        "Esiste una procedura formalizzata per la gestione degli incidenti?"
                    ),
                    "conversation_id": "assessment-001",
                    "history": [],
                    "options": {
                        "include_sources": True,
                        "include_debug": False,
                        "include_evaluation": False,
                        "target_document": (
                            "evidenza_12_policy_generica_parziale_debole.pdf"
                        ),
                        "target_pages": [1],
                    },
                }
            ]
        },
    )


# =============================================================================
# OUTPUT API: FONTI E PROVENANCE PUBBLICA
# =============================================================================
class GraphEntityResponse(ApiSchema):
    name: str = Field(min_length=1, max_length=300)
    type: str = Field(default="Entity", max_length=100)
    relation: str = Field(default="MENTIONED", max_length=100)


class SourceResponse(ApiSchema):
    """Fonte restituita al client dopo i controlli di visibilità tenant.

    Sono esclusi intenzionalmente password, identificativi PostgreSQL interni,
    embedding, query tecniche e ``organization_id``.
    """

    source_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(default="", max_length=256)
    filename: str = Field(min_length=1, max_length=512)
    page: int = Field(default=0, ge=0)
    page_chunk_index: int = Field(default=0, ge=0)

    source_type: str = Field(default="text", max_length=100)
    score: float = 0.0
    excerpt: str = Field(default="", max_length=MAX_PUBLIC_SOURCE_EXCERPT_CHARS)
    section_hint: str = Field(default="", max_length=500)
    graph_context: tuple[GraphEntityResponse, ...] = Field(default_factory=tuple)

    tier: Literal["A", "B", "C", "GRAPH", "USER"] | str = "C"
    scope: Literal["GLOBAL", "ACCOUNT"] | str = "ACCOUNT"
    classification: str = Field(default="internal", max_length=50)
    database_origin: str = Field(default="Unknown", max_length=150)


# =============================================================================
# OUTPUT API: RETRIEVAL, AUDIT ED EVALUATION
# =============================================================================
class ScoreSummary(ApiSchema):
    minimum: float = 0.0
    maximum: float = 0.0
    average: float = 0.0


class RetrievalMetricsResponse(ApiSchema):
    intent: RagIntent = RagIntent.TEXT
    wants_evidence: bool = False
    default_tiers: tuple[str, ...] = Field(default_factory=tuple)

    qdrant_candidates: int = Field(default=0, ge=0)
    qdrant_hits: int = Field(default=0, ge=0)
    postgres_bm25_hits: int = Field(default=0, ge=0)
    postgres_exact_phrase_hits: int = Field(default=0, ge=0)
    neo4j_direct_hits: int = Field(default=0, ge=0)
    neo4j_expanded_hits: int = Field(default=0, ge=0)

    kept_after_quality_filters: int = Field(default=0, ge=0)
    rerank_candidates: int = Field(default=0, ge=0)
    final_sources: int = Field(default=0, ge=0)

    tier_counts: dict[str, int] = Field(default_factory=dict)
    score: ScoreSummary = Field(default_factory=ScoreSummary)
    reranker_used: bool = False
    graph_expand_used: bool = False

    target_document: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)

    @field_validator("tier_counts")
    @classmethod
    def validate_tier_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("tier_counts non può contenere valori negativi")
        return dict(value)

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, int]) -> dict[str, int]:
        if any(duration < 0 for duration in value.values()):
            raise ValueError("timings_ms non può contenere valori negativi")
        return dict(value)


class RagEvaluationResponse(ApiSchema):
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    context_support: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    source_scope_violation: bool = False
    verdict: str = Field(default="UNKNOWN", max_length=80)
    unsupported_claims: tuple[str, ...] = Field(default_factory=tuple)
    supported_claims: tuple[str, ...] = Field(default_factory=tuple)
    reason: str = Field(default="", max_length=10_000)


class RagDebugResponse(ApiSchema):
    """Diagnostica opzionale, restituita solo con ``include_debug=true``."""

    retrieval: RetrievalMetricsResponse = Field(
        default_factory=RetrievalMetricsResponse
    )
    audit_markdown: str = Field(default="", max_length=30_000)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class RagQueryResponse(ApiSchema):
    """Risposta del futuro endpoint RAG."""

    status: ApiStatus = ApiStatus.SUCCESS
    request_id: UUID
    conversation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    answer: str = Field(min_length=1)
    answer_format: AnswerFormat = AnswerFormat.MARKDOWN
    intent: RagIntent = RagIntent.TEXT
    answer_mode: RagAnswerMode = RagAnswerMode.KNOWLEDGE
    execution_mode: RagExecutionMode = RagExecutionMode.RAG_GENERATION
    deterministic: bool = False

    sources: tuple[SourceResponse, ...] = Field(default_factory=tuple)
    debug: RagDebugResponse | None = None
    evaluation: RagEvaluationResponse | None = None
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    model: str = Field(default="", max_length=200)
    corpus_version: str = Field(default="", max_length=100)
    elapsed_ms: int = Field(default=0, ge=0)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for item in value:
            warning = item.strip()
            if not warning:
                continue
            if len(warning) > MAX_WARNING_CHARS:
                warning = warning[:MAX_WARNING_CHARS]
            if warning not in seen:
                seen.add(warning)
                cleaned.append(warning)

        return tuple(cleaned)


# =============================================================================
# ERRORI API
# =============================================================================
class ApiErrorDetail(ApiSchema):
    field: str | None = Field(default=None, max_length=300)
    message: str = Field(min_length=1, max_length=4_000)
    type: str | None = Field(default=None, max_length=200)
    context: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(ApiSchema):
    status: Literal["error"] = "error"
    request_id: UUID | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    code: ApiErrorCode
    message: str = Field(min_length=1, max_length=4_000)
    retryable: bool = False
    details: tuple[ApiErrorDetail, ...] = Field(default_factory=tuple)


# =============================================================================
# HEALTH API
# =============================================================================
class DependencyHealthResponse(ApiSchema):
    state: DependencyState
    latency_ms: int | None = Field(default=None, ge=0)
    detail: str = Field(default="", max_length=2_000)


class HealthResponse(ApiSchema):
    status: ServiceState
    service: str = "rag-api"
    version: str = Field(default="1.0.0", max_length=100)
    timestamp: datetime = Field(default_factory=utc_now)
    dependencies: dict[str, DependencyHealthResponse] = Field(default_factory=dict)
