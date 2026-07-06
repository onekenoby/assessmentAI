"""Modelli interni del motore RAG.

Questo modulo contiene gli oggetti di dominio condivisi tra retrieval,
reranking, prompting, generazione, validazione, audit e ``rag_service``.

Non contiene:
- schemi HTTP pubblici, definiti in ``api.schemas``;
- contesto o autorizzazione tenant, definiti in ``core.tenant``;
- connessioni e client runtime, definiti in ``core.resources``;
- dipendenze da Reflex o FastAPI.

I modelli derivano dagli oggetti ``GraphEntity``, ``SourceItem``,
``RetrievalDebug``, ``AuditTrail`` e ``RagEvalResult`` dell'ultimo
``gui_reflex.py``. Sono stati estesi soltanto con i dati necessari al backend
API e con validazioni coerenti con l'architettura multi-tenant.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# =============================================================================
# COSTANTI E TIPI
# =============================================================================
SourceTier = Literal["A", "B", "C", "GRAPH", "USER"]
SourceScope = Literal["GLOBAL", "ACCOUNT"]

_ALLOWED_SOURCE_TIERS = frozenset({"A", "B", "C", "GRAPH", "USER"})
_ALLOWED_SOURCE_SCOPES = frozenset({"GLOBAL", "ACCOUNT"})
_ALLOWED_CLASSIFICATIONS = frozenset(
    {"public", "internal", "confidential", "restricted"}
)


def utc_now() -> datetime:
    """Restituisce un timestamp UTC timezone-aware."""

    return datetime.now(timezone.utc)


def _normalize_tier(value: str) -> str:
    tier = str(value or "").strip().upper()

    if not tier:
        return "C"
    if tier.startswith("GRAPH"):
        return "GRAPH"
    if tier.startswith("USER"):
        return "USER"
    if tier == "A" or tier == "TIER_A_METHODOLOGY" or tier.endswith("_A_METHODOLOGY"):
        return "A"
    if tier == "B" or tier == "TIER_B_REFERENCE" or tier.endswith("_B_REFERENCE"):
        return "B"
    if (
        tier == "C"
        or tier == "TIER_C_EVIDENCE"
        or tier.endswith("_C_EVIDENCE")
        or "EVIDENCE" in tier
        or "EVIDENZA" in tier
    ):
        return "C"

    raise ValueError(f"tier sorgente non riconosciuto: {value!r}")


def _normalize_scope(value: str) -> str:
    scope = str(value or "").strip().upper()
    if scope not in _ALLOWED_SOURCE_SCOPES:
        raise ValueError("scope deve essere GLOBAL oppure ACCOUNT")
    return scope


def _ensure_finite(value: float, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} deve essere un numero finito")
    return parsed


def _unique_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()

    for item in values:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)

    return tuple(output)


class InternalModel(BaseModel):
    """Configurazione comune dei modelli interni mutabili."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
    )


class FrozenInternalModel(BaseModel):
    """Configurazione comune degli oggetti di dominio immutabili."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )


# =============================================================================
# ENUMERAZIONI DEL CORE
# Non importano ``api.schemas``: il core non dipende dal layer HTTP.
# =============================================================================
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
    RAG_GENERATION = "rag_generation"
    MATH_DIRECT = "math_direct"
    GLOSSARY_DIRECT = "glossary_direct"
    GRAPH_RELATION_STRICT = "graph_relation_strict"
    FORMULA_STRICT = "formula_strict"
    ANALYTICS = "analytics"


class EvaluationVerdict(StrEnum):
    UNKNOWN = "UNKNOWN"
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


# =============================================================================
# GRAFO E PROVENANCE
# =============================================================================
class GraphEntity(FrozenInternalModel):
    """Entità Neo4j associata a una fonte recuperata."""

    name: str = Field(min_length=1, max_length=500)
    type: str = Field(default="Entity", min_length=1, max_length=150)
    relation: str = Field(default="MENTIONED", min_length=1, max_length=150)

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, value: str) -> str:
        return value.strip().upper()


class SourceItem(InternalModel):
    """Fonte canonica utilizzata dal motore RAG.

    Il modello conserva la provenance completa necessaria ai controlli tenant,
    all'audit e al reranking. Il layer API selezionerà soltanto i campi pubblici
    definiti in ``api.schemas.SourceResponse``.
    """

    id: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=1_024)
    page: int = Field(default=0, ge=0)

    # Provenance condivisa tra Qdrant, PostgreSQL e Neo4j.
    page_chunk_index: int = Field(default=0, ge=0)
    doc_id: str = Field(default="", max_length=512)

    type: str = Field(default="text", min_length=1, max_length=100)
    score: float = 0.0
    graph_context: list[GraphEntity] = Field(default_factory=list)

    section_hint: str = Field(default="", max_length=2_000)
    image_id: int | None = Field(default=None, ge=0)

    tier: SourceTier = "C"
    scope: SourceScope = "ACCOUNT"
    organization_id: int | None = Field(default=None, gt=0)
    status: str = Field(default="active", min_length=1, max_length=50)
    ingestion_run_id: str = Field(default="", max_length=128)
    corpus_version: str = Field(default="", max_length=100)
    classification: str = Field(default="internal", min_length=1, max_length=50)
    embedding_model: str = Field(default="", max_length=500)
    request_id: str = Field(default="", max_length=128)

    # PostgreSQL canonical provenance.
    pg_ingestion_ts: str = Field(default="", max_length=100)
    pg_source_name: str = Field(default="", max_length=1_024)
    pg_source_type: str = Field(default="", max_length=150)
    pg_log_id: int = Field(default=0, ge=0)
    pg_chunk_id: int = Field(default=0, ge=0)
    pg_page_chunk_index: int = Field(default=0, ge=0)
    pg_toon_type: str = Field(default="", max_length=150)

    db_origin: str = Field(default="Unknown", min_length=1, max_length=200)

    @field_validator("tier", mode="before")
    @classmethod
    def normalize_tier(cls, value: Any) -> str:
        return _normalize_tier(str(value or ""))

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, value: Any) -> str:
        return _normalize_scope(str(value or ""))

    @field_validator("status")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("classification")
    @classmethod
    def normalize_classification(cls, value: str) -> str:
        classification = value.strip().lower()
        if classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                "classification deve essere public, internal, confidential o restricted"
            )
        return classification

    @field_validator("type")
    @classmethod
    def normalize_source_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "formula": "formula",
            "math": "formula",
            "equation": "formula",
            "image": "image",
            "immagine": "image",
            "imagine": "image",
            "visual": "image",
            "screenshot": "image",
            "chart": "chart",
            "grafico": "chart",
            "chart_analysis": "chart",
            "diagram": "chart",
            "diagramma": "chart",
            "table": "table",
            "tabella": "table",
            "graph": "graph",
            "graph_relations": "graph_relations",
            "text": "text",
            "testo": "text",
        }
        return aliases.get(normalized, normalized or "text")

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        return _ensure_finite(value, "score")

    @model_validator(mode="after")
    def validate_tenant_provenance(self) -> "SourceItem":
        """Verifica solo la coerenza strutturale della provenance.

        L'autorizzazione effettiva resta responsabilità di ``core.tenant``.
        """

        tier = str(self.tier)
        scope = str(self.scope)

        if tier == "A":
            if scope != "GLOBAL" or self.organization_id is not None:
                raise ValueError(
                    "Tier A richiede scope GLOBAL e organization_id nullo"
                )
        elif tier in {"B", "C", "USER"}:
            if scope != "ACCOUNT" or self.organization_id is None:
                raise ValueError(
                    f"Tier {tier} richiede scope ACCOUNT e organization_id"
                )
        elif tier == "GRAPH":
            if scope == "GLOBAL" and self.organization_id is not None:
                raise ValueError(
                    "Una fonte GRAPH globale richiede organization_id nullo"
                )
            if scope == "ACCOUNT" and self.organization_id is None:
                raise ValueError(
                    "Una fonte GRAPH account richiede organization_id"
                )
        else:  # pragma: no cover - protetto dal validator del tier
            raise ValueError(f"tier non supportato: {tier}")

        return self

    @computed_field
    @property
    def dedupe_key(self) -> str:
        """Chiave stabile per eliminare duplicati nel set finale di fonti."""

        return "::".join(
            (
                self.doc_id or self.filename.lower(),
                str(self.page),
                str(self.page_chunk_index),
                self.id,
            )
        )

    def excerpt(self, max_chars: int = 5_000) -> str:
        """Restituisce un estratto sicuro senza modificare la fonte originale."""

        if max_chars <= 0:
            return ""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars].rstrip() + "..."

    def audit_reference(self) -> dict[str, Any]:
        """Proiezione minima usata dall'audit tecnico."""

        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "page_chunk_index": self.page_chunk_index,
            "type": self.type,
            "tier": self.tier,
            "scope": self.scope,
            "organization_id": self.organization_id,
            "classification": self.classification,
            "db_origin": self.db_origin,
            "score": self.score,
        }


# =============================================================================
# CANDIDATI DEL RETRIEVAL
# =============================================================================
class RetrievalCandidate(InternalModel):
    """Candidato intermedio prodotto dai diversi motori di retrieval.

    Sostituisce progressivamente i dizionari eterogenei usati nel PoC. Ogni
    backend valorizza i propri score; RRF, reranker e tier policy completano i
    campi successivi.
    """

    id: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    filename: str = Field(default="Unknown", min_length=1, max_length=1_024)
    page: int = Field(default=0, ge=0)
    page_chunk_index: int = Field(default=0, ge=0)
    doc_id: str = Field(default="", max_length=512)

    type: str = Field(default="text", min_length=1, max_length=100)
    tier: SourceTier = "C"
    source_tier: str = Field(default="", max_length=100)
    scope: SourceScope = "ACCOUNT"
    organization_id: int | None = Field(default=None, gt=0)
    status: str = Field(default="active", min_length=1, max_length=50)
    ingestion_run_id: str = Field(default="", max_length=128)
    corpus_version: str = Field(default="", max_length=100)
    classification: str = Field(default="internal", min_length=1, max_length=50)
    embedding_model: str = Field(default="", max_length=500)

    section_hint: str = Field(default="", max_length=2_000)
    image_id: int | None = Field(default=None, ge=0)
    origin: str = Field(default="Unknown", min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)
    graph_context: list[GraphEntity] = Field(default_factory=list)

    score_base: float = 0.0
    score_vec: float = 0.0
    score_bm25: float = 0.0
    score_graph: float = 0.0
    score_rrf: float = 0.0
    score_rerank: float = 0.0
    score_tier_delta: float = 0.0
    final_score: float = 0.0

    @field_validator("tier", mode="before")
    @classmethod
    def normalize_tier(cls, value: Any) -> str:
        return _normalize_tier(str(value or ""))

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, value: Any) -> str:
        return _normalize_scope(str(value or ""))

    @field_validator("status")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("classification")
    @classmethod
    def normalize_classification(cls, value: str) -> str:
        classification = value.strip().lower()
        if classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                "classification deve essere public, internal, confidential o restricted"
            )
        return classification

    @field_validator(
        "score_base",
        "score_vec",
        "score_bm25",
        "score_graph",
        "score_rrf",
        "score_rerank",
        "score_tier_delta",
        "final_score",
    )
    @classmethod
    def validate_scores(cls, value: float, info: Any) -> float:
        return _ensure_finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_tenant_provenance(self) -> "RetrievalCandidate":
        # Riusa le stesse invarianti del SourceItem senza creare una dipendenza
        # da repository o TenantContext.
        tier = str(self.tier)
        scope = str(self.scope)

        if tier == "A" and (scope != "GLOBAL" or self.organization_id is not None):
            raise ValueError("Tier A richiede scope GLOBAL e organization_id nullo")
        if tier in {"B", "C", "USER"} and (
            scope != "ACCOUNT" or self.organization_id is None
        ):
            raise ValueError(
                f"Tier {tier} richiede scope ACCOUNT e organization_id"
            )
        if tier == "GRAPH":
            if scope == "GLOBAL" and self.organization_id is not None:
                raise ValueError("GRAPH globale richiede organization_id nullo")
            if scope == "ACCOUNT" and self.organization_id is None:
                raise ValueError("GRAPH account richiede organization_id")

        return self

    @computed_field
    @property
    def effective_score(self) -> float:
        """Score più avanzato disponibile nella pipeline."""

        for value in (
            self.final_score,
            self.score_rerank,
            self.score_rrf,
            self.score_vec,
            self.score_bm25,
            self.score_graph,
            self.score_base,
        ):
            if value != 0.0:
                return float(value)
        return 0.0

    def to_source_item(self, *, request_id: str = "") -> SourceItem:
        """Materializza il candidato come fonte finale canonica."""

        return SourceItem(
            id=self.id,
            content=self.content,
            filename=self.filename,
            page=self.page,
            page_chunk_index=self.page_chunk_index,
            doc_id=self.doc_id,
            type=self.type,
            score=self.effective_score,
            graph_context=list(self.graph_context),
            section_hint=self.section_hint,
            image_id=self.image_id,
            tier=self.tier,
            scope=self.scope,
            organization_id=self.organization_id,
            status=self.status,
            ingestion_run_id=self.ingestion_run_id,
            corpus_version=self.corpus_version,
            classification=self.classification,
            embedding_model=self.embedding_model,
            request_id=request_id,
            db_origin=self.origin,
        )


# =============================================================================
# METRICHE DI RETRIEVAL
# =============================================================================
class ScoreStats(InternalModel):
    minimum: float = 0.0
    maximum: float = 0.0
    average: float = 0.0

    @field_validator("minimum", "maximum", "average")
    @classmethod
    def validate_values(cls, value: float, info: Any) -> float:
        return _ensure_finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_order(self) -> "ScoreStats":
        if self.minimum > self.maximum:
            raise ValueError("minimum non può essere maggiore di maximum")
        return self

    @classmethod
    def from_values(cls, values: list[float] | tuple[float, ...]) -> "ScoreStats":
        finite_values = [
            _ensure_finite(value, "score") for value in values if value is not None
        ]
        if not finite_values:
            return cls()
        return cls(
            minimum=min(finite_values),
            maximum=max(finite_values),
            average=sum(finite_values) / len(finite_values),
        )


class RetrievalDebug(InternalModel):
    """Metriche e diagnostica prodotte dalla pipeline di retrieval."""

    query: str = Field(default="", repr=False)
    intent: RagIntent = RagIntent.TEXT
    answer_mode: RagAnswerMode = RagAnswerMode.KNOWLEDGE

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
    score: ScoreStats = Field(default_factory=ScoreStats)

    reranker_used: bool = False
    graph_expand_used: bool = False

    target_document: str | None = Field(default=None, max_length=1_024)
    target_pages: tuple[int, ...] = Field(default_factory=tuple)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("default_tiers", "warnings", mode="before")
    @classmethod
    def unique_string_sequences(cls, value: Any) -> tuple[str, ...]:
        return _unique_strings(tuple(value or ()))

    @field_validator("target_pages")
    @classmethod
    def validate_target_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(int(page) for page in value)))
        if any(page <= 0 for page in normalized):
            raise ValueError("target_pages accetta soltanto pagine 1-based positive")
        return normalized

    @field_validator("tier_counts")
    @classmethod
    def validate_tier_counts(cls, value: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for key, count in value.items():
            parsed = int(count)
            if parsed < 0:
                raise ValueError("tier_counts non può contenere valori negativi")
            normalized[str(key).upper()] = parsed
        return normalized

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for key, duration in value.items():
            parsed = int(duration)
            if parsed < 0:
                raise ValueError("timings_ms non può contenere valori negativi")
            normalized[str(key)] = parsed
        return normalized

    def record_timing(self, name: str, seconds: float) -> None:
        """Registra una durata espressa in secondi convertendola in millisecondi."""

        duration = _ensure_finite(seconds, "seconds")
        if duration < 0:
            raise ValueError("seconds non può essere negativo")
        self.timings_ms[str(name)] = int(round(duration * 1_000))

    def set_score_values(self, values: list[float] | tuple[float, ...]) -> None:
        self.score = ScoreStats.from_values(values)


# =============================================================================
# AUDIT
# =============================================================================
class AuditSourceRecord(FrozenInternalModel):
    id: str = Field(default="", max_length=512)
    doc_id: str = Field(default="", max_length=512)
    filename: str = Field(min_length=1, max_length=1_024)
    page: int = Field(default=0, ge=0)
    page_chunk_index: int = Field(default=0, ge=0)
    type: str = Field(default="text", max_length=100)
    tier: SourceTier = "C"
    scope: SourceScope = "ACCOUNT"
    organization_id: int | None = Field(default=None, gt=0)
    classification: str = Field(default="internal", max_length=50)
    db_origin: str = Field(default="Unknown", max_length=200)
    score: float = 0.0

    @field_validator("tier", mode="before")
    @classmethod
    def normalize_tier(cls, value: Any) -> str:
        return _normalize_tier(str(value or ""))

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, value: Any) -> str:
        return _normalize_scope(str(value or ""))

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        return _ensure_finite(value, "score")

    @classmethod
    def from_source(cls, source: SourceItem) -> "AuditSourceRecord":
        return cls(**source.audit_reference())


class AuditTrail(InternalModel):
    """Audit tecnico della singola interrogazione RAG.

    ``query`` è disponibile soltanto durante l'elaborazione. Il metodo
    ``persistent_payload`` la rimuove sempre e conserva soltanto l'hash.
    """

    ts_utc: datetime = Field(default_factory=utc_now)
    query: str = Field(default="", repr=False)
    query_sha256: str = Field(default="", min_length=0, max_length=64)
    intent: RagIntent = RagIntent.TEXT
    answer_mode: RagAnswerMode = RagAnswerMode.KNOWLEDGE
    execution_mode: RagExecutionMode = RagExecutionMode.RAG_GENERATION
    deterministic: bool = False

    organization_id: int = Field(gt=0)
    user_id: str = Field(min_length=1, max_length=512)
    roles: tuple[str, ...] = Field(default_factory=tuple)
    request_id: UUID = Field(default_factory=uuid4)
    corpus_version: str = Field(default="", max_length=100)

    filters: dict[str, Any] = Field(default_factory=dict)
    retrieved_sources: tuple[AuditSourceRecord, ...] = Field(default_factory=tuple)

    prompt_sha256: str = Field(default="", min_length=0, max_length=64)
    context_chars: int = Field(default=0, ge=0)
    retrieval: RetrievalDebug = Field(default_factory=RetrievalDebug)

    llm_model: str = Field(default="", max_length=500)
    temperature: float = 0.1
    memory_limit: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("ts_utc")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("roles", "warnings", mode="before")
    @classmethod
    def unique_sequences(cls, value: Any) -> tuple[str, ...]:
        return _unique_strings(tuple(value or ()))

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        parsed = _ensure_finite(value, "temperature")
        if parsed < 0:
            raise ValueError("temperature non può essere negativa")
        return parsed

    @field_validator("query_sha256", "prompt_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned and (
            len(cleaned) != 64
            or any(char not in "0123456789abcdef" for char in cleaned)
        ):
            raise ValueError("l'hash SHA-256 deve contenere 64 caratteri esadecimali")
        return cleaned

    @model_validator(mode="after")
    def derive_query_hash(self) -> "AuditTrail":
        if not self.query_sha256 and self.query:
            self.query_sha256 = sha256(self.query.encode("utf-8")).hexdigest()
        return self

    @classmethod
    def from_sources(
        cls,
        *,
        organization_id: int,
        user_id: str,
        roles: tuple[str, ...] | list[str],
        request_id: str | UUID,
        query: str,
        sources: list[SourceItem] | tuple[SourceItem, ...],
        **kwargs: Any,
    ) -> "AuditTrail":
        return cls(
            organization_id=organization_id,
            user_id=user_id,
            roles=tuple(roles),
            request_id=UUID(str(request_id)),
            query=query,
            retrieved_sources=tuple(
                AuditSourceRecord.from_source(source) for source in sources
            ),
            **kwargs,
        )

    def persistent_payload(self) -> dict[str, Any]:
        """Serializza l'audit senza persistere il testo della query."""

        payload = self.model_dump(mode="json")
        payload["query"] = ""
        return payload


# =============================================================================
# VALUTAZIONE RAG
# =============================================================================
class RagEvalResult(InternalModel):
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    context_support: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    source_scope_violation: bool = False
    verdict: EvaluationVerdict = EvaluationVerdict.UNKNOWN
    unsupported_claims: tuple[str, ...] = Field(default_factory=tuple)
    supported_claims: tuple[str, ...] = Field(default_factory=tuple)
    reason: str = Field(default="", max_length=20_000)

    @field_validator("unsupported_claims", "supported_claims", mode="before")
    @classmethod
    def unique_claims(cls, value: Any) -> tuple[str, ...]:
        return _unique_strings(tuple(value or ()))

    def resolve_verdict(
        self,
        *,
        minimum_faithfulness: float,
        minimum_answer_relevance: float,
    ) -> EvaluationVerdict:
        """Determina il verdetto quando il judge non ne restituisce uno valido."""

        if self.verdict in {
            EvaluationVerdict.PASS,
            EvaluationVerdict.WARN,
            EvaluationVerdict.FAIL,
            EvaluationVerdict.ERROR,
            EvaluationVerdict.DISABLED,
        }:
            return EvaluationVerdict(str(self.verdict))

        if (
            self.faithfulness >= minimum_faithfulness
            and self.answer_relevance >= minimum_answer_relevance
            and not self.source_scope_violation
        ):
            self.verdict = EvaluationVerdict.PASS
        elif self.faithfulness >= 0.55:
            self.verdict = EvaluationVerdict.WARN
        else:
            self.verdict = EvaluationVerdict.FAIL

        return EvaluationVerdict(str(self.verdict))

    @classmethod
    def disabled(cls) -> "RagEvalResult":
        return cls(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_support=1.0,
            hallucination_risk=0.0,
            verdict=EvaluationVerdict.DISABLED,
            reason="Evaluation disabled.",
        )

    @classmethod
    def error(cls, reason: str) -> "RagEvalResult":
        return cls(verdict=EvaluationVerdict.ERROR, reason=reason)


# =============================================================================
# RISULTATO APPLICATIVO DEL SERVIZIO
# =============================================================================
class RagServiceResult(InternalModel):
    """Risultato neutro restituito da ``core.rag_service``.

    Il futuro router FastAPI mapperà questo modello in
    ``api.schemas.RagQueryResponse`` senza far dipendere il core dal layer HTTP.
    """

    request_id: UUID = Field(default_factory=uuid4)
    conversation_id: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=utc_now)

    answer: str = Field(min_length=1)
    intent: RagIntent = RagIntent.TEXT
    answer_mode: RagAnswerMode = RagAnswerMode.KNOWLEDGE
    execution_mode: RagExecutionMode = RagExecutionMode.RAG_GENERATION
    deterministic: bool = False

    sources: tuple[SourceItem, ...] = Field(default_factory=tuple)
    retrieval: RetrievalDebug = Field(default_factory=RetrievalDebug)
    evaluation: RagEvalResult | None = None

    audit_markdown: str = Field(default="", max_length=100_000)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    model: str = Field(default="", max_length=500)
    corpus_version: str = Field(default="", max_length=100)
    elapsed_ms: int = Field(default=0, ge=0)

    @field_validator("created_at")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("warnings", mode="before")
    @classmethod
    def unique_warnings(cls, value: Any) -> tuple[str, ...]:
        return _unique_strings(tuple(value or ()))

    @model_validator(mode="after")
    def align_metrics(self) -> "RagServiceResult":
        self.retrieval.intent = self.intent
        self.retrieval.answer_mode = self.answer_mode
        self.retrieval.final_sources = len(self.sources)
        return self


__all__ = [
    "AuditSourceRecord",
    "AuditTrail",
    "EvaluationVerdict",
    "GraphEntity",
    "RagAnswerMode",
    "RagEvalResult",
    "RagExecutionMode",
    "RagIntent",
    "RagServiceResult",
    "RetrievalCandidate",
    "RetrievalDebug",
    "ScoreStats",
    "SourceItem",
    "SourceScope",
    "SourceTier",
]
