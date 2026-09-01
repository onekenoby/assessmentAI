"""Audit tecnico e persistenza osservabile del servizio RAG.

Il modulo sostituisce ``append_audit_log`` e ``append_rag_eval_log`` presenti
nel PoC Reflex con un servizio indipendente dal framework HTTP.

Responsabilità:
- costruzione dell'``AuditTrail`` dalla richiesta tenant corrente;
- persistenza JSONL senza query, prompt, risposta o contenuto documentale in chiaro;
- persistenza PostgreSQL tenant-safe nella tabella ``rag_query_audit``;
- persistenza separata delle metriche di evaluation;
- verifica dell'identità tenant prima di ogni scrittura;
- rendering Markdown delle metriche di retrieval/evaluation per il debug API.

Il modulo NON:
- contiene endpoint FastAPI;
- esegue retrieval, generazione o evaluation;
- modifica lo schema PostgreSQL;
- accetta ``organization_id`` dal body pubblico della richiesta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.config import RagSettings, settings
from core.models import (
    AuditSourceRecord,
    AuditTrail,
    RagAnswerMode,
    RagEvalResult,
    RagExecutionMode,
    RagIntent,
    RetrievalDebug,
    SourceItem,
)
from core.resources import postgres_connection
from core.tenant import TenantContext, get_tenant_context


logger = logging.getLogger(__name__)


# =============================================================================
# ECCEZIONI E RISULTATI
# =============================================================================
class AuditError(RuntimeError):
    """Errore base del sottosistema di audit."""


class AuditIdentityError(AuditError):
    """L'audit non coincide con il TenantContext della richiesta corrente."""


class AuditPersistenceError(AuditError):
    """Una o più destinazioni di persistenza non sono state scritte."""


class AuditSink(StrEnum):
    QUERY_JSONL = "query_jsonl"
    POSTGRES = "postgres"
    EVALUATION_JSONL = "evaluation_jsonl"


@dataclass(frozen=True, slots=True)
class AuditSinkOutcome:
    sink: AuditSink
    attempted: bool
    success: bool
    skipped: bool = False
    duplicate: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sink"] = self.sink.value
        return payload


@dataclass(frozen=True, slots=True)
class AuditWriteResult:
    request_id: str
    outcomes: tuple[AuditSinkOutcome, ...]

    @property
    def success(self) -> bool:
        attempted = [item for item in self.outcomes if item.attempted]
        return bool(attempted) and all(item.success for item in attempted)

    @property
    def degraded(self) -> bool:
        attempted = [item for item in self.outcomes if item.attempted]
        return bool(attempted) and any(item.success for item in attempted) and any(
            not item.success for item in attempted
        )

    @property
    def skipped(self) -> bool:
        return bool(self.outcomes) and all(item.skipped for item in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "degraded": self.degraded,
            "skipped": self.skipped,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }


# =============================================================================
# RECORD DI EVALUATION
# =============================================================================
class EvaluationAuditRecord(BaseModel):
    """Record persistente delle metriche del judge.

    Query e risposta non vengono mai memorizzate: vengono conservati soltanto
    gli hash SHA-256. Anche le fonti contengono esclusivamente provenance e
    score, mai il testo dei chunk.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: UUID
    organization_id: int = Field(gt=0)
    user_id: str = Field(min_length=1, max_length=512)

    query_sha256: str = Field(min_length=64, max_length=64)
    answer_sha256: str = Field(min_length=64, max_length=64)
    requested_document: str = Field(default="", max_length=1_024)

    sources: tuple[AuditSourceRecord, ...] = Field(default_factory=tuple)
    metrics: RagEvalResult

    llm_model: str = Field(default="", max_length=500)
    evaluation_model: str = Field(default="", max_length=500)
    corpus_version: str = Field(default="", max_length=100)
    strict_block_applied: bool = False
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("ts_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("query_sha256", "answer_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
            raise ValueError("hash SHA-256 non valido")
        return cleaned

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item).strip() for item in (value or ()) if str(item).strip()))

    @classmethod
    def from_evaluation(
        cls,
        *,
        query: str,
        answer: str,
        sources: Sequence[SourceItem],
        metrics: RagEvalResult,
        context: TenantContext,
        requested_document: str = "",
        llm_model: str = "",
        evaluation_model: str = "",
        corpus_version: str = "",
        strict_block_applied: bool = False,
        warnings: Sequence[str] = (),
    ) -> "EvaluationAuditRecord":
        return cls(
            request_id=UUID(context.request_id),
            organization_id=context.organization_id,
            user_id=context.user_id,
            query_sha256=_sha256_text(query),
            answer_sha256=_sha256_text(answer),
            requested_document=requested_document,
            sources=tuple(AuditSourceRecord.from_source(source) for source in sources),
            metrics=metrics,
            llm_model=llm_model,
            evaluation_model=evaluation_model,
            corpus_version=corpus_version,
            strict_block_applied=strict_block_applied,
            warnings=tuple(warnings),
        )

    def persistent_payload(self) -> dict[str, Any]:
        """Serializza le metriche senza testo del judge o claim in chiaro."""

        payload = self.model_dump(mode="json")
        metrics = dict(payload.get("metrics") or {})

        reason = str(metrics.get("reason") or "")
        unsupported = [str(item) for item in metrics.get("unsupported_claims") or []]
        supported = [str(item) for item in metrics.get("supported_claims") or []]

        metrics["reason"] = ""
        metrics["reason_sha256"] = _sha256_text(reason) if reason else ""
        metrics["unsupported_claims"] = []
        metrics["unsupported_claims_count"] = len(unsupported)
        metrics["unsupported_claims_sha256"] = [
            _sha256_text(item) for item in unsupported
        ]
        metrics["supported_claims"] = []
        metrics["supported_claims_count"] = len(supported)
        metrics["supported_claims_sha256"] = [
            _sha256_text(item) for item in supported
        ]

        payload["metrics"] = metrics
        return payload


# =============================================================================
# SANITIZZAZIONE
# =============================================================================
_SENSITIVE_AUDIT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "query",
        "prompt",
        "answer",
        "content",
        "context",
        "raw_text",
        "authorization",
        "access_token",
        "refresh_token",
        "api_key",
        "password",
        "secret",
    }
)

_FILE_LOCK: Final[threading.RLock] = threading.RLock()


def _sha256_text(value: str) -> str:
    return sha256((value or "").encode("utf-8")).hexdigest()


def _sanitize_payload(value: Any) -> Any:
    """Rimuove ricorsivamente eventuali campi sensibili inseriti nei metadati."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.strip().lower() in _SENSITIVE_AUDIT_KEYS:
                clean[key_text] = "[REDACTED]"
            else:
                clean[key_text] = _sanitize_payload(item)
        return clean

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_payload(item) for item in value]

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _sanitize_payload(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Appende una singola riga JSON con lock thread/process best-effort."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = _compact_json(payload) + "\n"

    with _FILE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            locked = False
            try:
                # Docker/Linux: impedisce l'interleaving tra più worker/processi.
                try:
                    import fcntl  # type: ignore

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    locked = True
                except (ImportError, OSError):
                    # Su Windows resta comunque attivo il lock di processo.
                    locked = False

                handle.write(line)
                handle.flush()
            finally:
                if locked:
                    try:
                        import fcntl  # type: ignore

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass


# =============================================================================
# COSTRUZIONE AUDIT
# =============================================================================
def create_query_audit(
    *,
    query: str,
    sources: Sequence[SourceItem],
    intent: RagIntent | str,
    answer_mode: RagAnswerMode | str,
    execution_mode: RagExecutionMode | str,
    retrieval: RetrievalDebug,
    filters: Mapping[str, Any] | None = None,
    prompt_sha256: str = "",
    context_chars: int = 0,
    deterministic: bool = False,
    llm_model: str = "",
    temperature: float | None = None,
    memory_limit: int | None = None,
    elapsed_ms: int = 0,
    warnings: Sequence[str] = (),
    context: TenantContext | None = None,
    config: RagSettings = settings,
) -> AuditTrail:
    """Costruisce l'audit usando esclusivamente il TenantContext trusted."""

    tenant = context or get_tenant_context()

    return AuditTrail.from_sources(
        organization_id=tenant.organization_id,
        user_id=tenant.user_id,
        roles=tenant.roles,
        request_id=tenant.request_id,
        query=query,
        sources=list(sources),
        intent=RagIntent(str(intent)),
        answer_mode=RagAnswerMode(str(answer_mode)),
        execution_mode=RagExecutionMode(str(execution_mode)),
        deterministic=deterministic,
        corpus_version=config.corpus_version,
        filters=dict(filters or {}),
        prompt_sha256=prompt_sha256,
        context_chars=context_chars,
        retrieval=retrieval,
        llm_model=llm_model or config.llm_model_name,
        temperature=(config.llm_temperature if temperature is None else temperature),
        memory_limit=(config.memory_limit if memory_limit is None else memory_limit),
        elapsed_ms=elapsed_ms,
        warnings=tuple(warnings),
    )


# =============================================================================
# SERVIZIO DI PERSISTENZA
# =============================================================================
PostgresConnectionFactory = Callable[..., AbstractContextManager[Any]]


class AuditService:
    """Persistenza multi-sink dell'audit RAG.

    La verifica tenant è sempre bloccante. Gli errori infrastrutturali vengono
    invece restituiti nel risultato; il chiamante può renderli bloccanti tramite
    ``raise_on_failure=True``.
    """

    def __init__(
        self,
        *,
        config: RagSettings = settings,
        postgres_connection_factory: PostgresConnectionFactory = postgres_connection,
    ) -> None:
        self._config = config
        self._postgres_connection = postgres_connection_factory

    def persist_query_audit(
        self,
        audit: AuditTrail,
        *,
        context: TenantContext | None = None,
        raise_on_failure: bool = False,
    ) -> AuditWriteResult:
        tenant = context or get_tenant_context()
        self._assert_query_audit_identity(audit, tenant)

        request_id = str(audit.request_id)
        outcomes: list[AuditSinkOutcome] = []

        if not self._config.audit_enabled:
            outcomes.extend(
                [
                    AuditSinkOutcome(
                        sink=AuditSink.QUERY_JSONL,
                        attempted=False,
                        success=True,
                        skipped=True,
                        detail="Audit disabilitato da configurazione.",
                    ),
                    AuditSinkOutcome(
                        sink=AuditSink.POSTGRES,
                        attempted=False,
                        success=True,
                        skipped=True,
                        detail="Audit disabilitato da configurazione.",
                    ),
                ]
            )
            return AuditWriteResult(request_id=request_id, outcomes=tuple(outcomes))

        payload = _sanitize_payload(audit.persistent_payload())

        # Sink JSONL: sempre previsto quando AUDIT_ENABLED=1.
        try:
            _append_jsonl(self._config.audit_log_path, payload)
            outcomes.append(
                AuditSinkOutcome(
                    sink=AuditSink.QUERY_JSONL,
                    attempted=True,
                    success=True,
                    detail=str(self._config.audit_log_path),
                )
            )
        except Exception as exc:  # noqa: BLE001 - il risultato deve descrivere il sink
            logger.exception("Errore scrittura audit JSONL request_id=%s", request_id)
            outcomes.append(
                AuditSinkOutcome(
                    sink=AuditSink.QUERY_JSONL,
                    attempted=True,
                    success=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

        # Sink PostgreSQL: previsto soltanto se il repository PG è abilitato.
        if not self._config.pg_enrich_enabled:
            outcomes.append(
                AuditSinkOutcome(
                    sink=AuditSink.POSTGRES,
                    attempted=False,
                    success=True,
                    skipped=True,
                    detail="PostgreSQL disabilitato.",
                )
            )
        else:
            outcomes.append(self._persist_query_postgres(audit, tenant))

        result = AuditWriteResult(request_id=request_id, outcomes=tuple(outcomes))
        self._raise_if_required(result, raise_on_failure)
        return result

    async def persist_query_audit_async(
        self,
        audit: AuditTrail,
        *,
        context: TenantContext | None = None,
        raise_on_failure: bool = False,
    ) -> AuditWriteResult:
        return await asyncio.to_thread(
            self.persist_query_audit,
            audit,
            context=context,
            raise_on_failure=raise_on_failure,
        )

    def persist_evaluation(
        self,
        record: EvaluationAuditRecord,
        *,
        context: TenantContext | None = None,
        raise_on_failure: bool = False,
    ) -> AuditWriteResult:
        tenant = context or get_tenant_context()
        self._assert_evaluation_identity(record, tenant)
        request_id = str(record.request_id)

        if not self._config.evaluation_enabled:
            return AuditWriteResult(
                request_id=request_id,
                outcomes=(
                    AuditSinkOutcome(
                        sink=AuditSink.EVALUATION_JSONL,
                        attempted=False,
                        success=True,
                        skipped=True,
                        detail="Evaluation disabilitata da configurazione.",
                    ),
                ),
            )

        try:
            _append_jsonl(
                self._config.evaluation_log_path,
                _sanitize_payload(record.persistent_payload()),
            )
            outcome = AuditSinkOutcome(
                sink=AuditSink.EVALUATION_JSONL,
                attempted=True,
                success=True,
                detail=str(self._config.evaluation_log_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Errore scrittura evaluation JSONL request_id=%s", request_id)
            outcome = AuditSinkOutcome(
                sink=AuditSink.EVALUATION_JSONL,
                attempted=True,
                success=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

        result = AuditWriteResult(request_id=request_id, outcomes=(outcome,))
        self._raise_if_required(result, raise_on_failure)
        return result

    async def persist_evaluation_async(
        self,
        record: EvaluationAuditRecord,
        *,
        context: TenantContext | None = None,
        raise_on_failure: bool = False,
    ) -> AuditWriteResult:
        return await asyncio.to_thread(
            self.persist_evaluation,
            record,
            context=context,
            raise_on_failure=raise_on_failure,
        )

    def _persist_query_postgres(
        self,
        audit: AuditTrail,
        tenant: TenantContext,
    ) -> AuditSinkOutcome:
        request_id = str(audit.request_id)
        payload = audit.persistent_payload()

        # La tabella corrente non contiene ancora colonne dedicate a tutte le
        # metriche del nuovo AuditTrail. I dati estesi restano nel JSONL; nel DB
        # vengono conservati i campi ufficialmente previsti dallo schema v1.1.
        filters_payload = _sanitize_payload(payload.get("filters") or {})
        sources_payload = _sanitize_payload(payload.get("retrieved_sources") or [])

        try:
            with self._postgres_connection(context=tenant) as conn:
                try:
                    with conn.cursor() as cur:
                        # Serializza check+insert anche tra worker distinti senza
                        # richiedere una modifica dello schema esistente.
                        cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (request_id,),
                        )
                        cur.execute(
                            """
                            SELECT audit_id
                            FROM public.rag_query_audit
                            WHERE request_id = %s::uuid
                              AND organization_id = %s
                            LIMIT 1
                            """,
                            (request_id, audit.organization_id),
                        )
                        existing = cur.fetchone()

                        if existing:
                            conn.rollback()
                            return AuditSinkOutcome(
                                sink=AuditSink.POSTGRES,
                                attempted=True,
                                success=True,
                                duplicate=True,
                                detail="Audit già presente per request_id e tenant.",
                            )

                        cur.execute(
                            """
                            INSERT INTO public.rag_query_audit (
                                request_id,
                                organization_id,
                                user_id,
                                roles,
                                query_sha256,
                                intent,
                                filters,
                                retrieved_sources,
                                llm_model,
                                corpus_version,
                                created_at
                            ) VALUES (
                                %s::uuid,
                                %s,
                                %s,
                                %s::jsonb,
                                %s,
                                %s,
                                %s::jsonb,
                                %s::jsonb,
                                %s,
                                %s,
                                %s
                            )
                            """,
                            (
                                request_id,
                                audit.organization_id,
                                audit.user_id,
                                json.dumps(
                                    list(audit.roles),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                audit.query_sha256,
                                str(audit.intent),
                                _compact_json(filters_payload),
                                json.dumps(
                                    sources_payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ),
                                audit.llm_model,
                                audit.corpus_version,
                                audit.ts_utc,
                            ),
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

            return AuditSinkOutcome(
                sink=AuditSink.POSTGRES,
                attempted=True,
                success=True,
                detail="public.rag_query_audit",
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Errore scrittura audit PostgreSQL request_id=%s", request_id)
            return AuditSinkOutcome(
                sink=AuditSink.POSTGRES,
                attempted=True,
                success=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _assert_query_audit_identity(audit: AuditTrail, tenant: TenantContext) -> None:
        mismatches: list[str] = []

        if audit.organization_id != tenant.organization_id:
            mismatches.append("organization_id")
        if str(audit.request_id) != tenant.request_id:
            mismatches.append("request_id")
        if audit.user_id != tenant.user_id:
            mismatches.append("user_id")
        if set(audit.roles) != set(tenant.roles):
            mismatches.append("roles")

        if mismatches:
            raise AuditIdentityError(
                "Audit non coerente con il TenantContext: " + ", ".join(mismatches)
            )

    @staticmethod
    def _assert_evaluation_identity(
        record: EvaluationAuditRecord,
        tenant: TenantContext,
    ) -> None:
        mismatches: list[str] = []

        if record.organization_id != tenant.organization_id:
            mismatches.append("organization_id")
        if str(record.request_id) != tenant.request_id:
            mismatches.append("request_id")
        if record.user_id != tenant.user_id:
            mismatches.append("user_id")

        if mismatches:
            raise AuditIdentityError(
                "Evaluation audit non coerente con il TenantContext: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _raise_if_required(result: AuditWriteResult, required: bool) -> None:
        if required and not result.success:
            failures = [
                f"{item.sink.value}: {item.detail}"
                for item in result.outcomes
                if item.attempted and not item.success
            ]
            raise AuditPersistenceError(
                "Persistenza audit incompleta: " + "; ".join(failures)
            )


# =============================================================================
# DEBUG MARKDOWN API-SAFE
# =============================================================================
def format_retrieval_audit_markdown(audit: AuditTrail) -> str:
    """Rende leggibili le metriche senza esporre query, tenant o ID chunk."""

    retrieval = audit.retrieval
    lines = [
        "### Audit retrieval",
        f"- Request ID: `{audit.request_id}`",
        f"- Intent: `{audit.intent}`",
        f"- Answer mode: `{audit.answer_mode}`",
        f"- Execution mode: `{audit.execution_mode}`",
        f"- Deterministico: **{'sì' if audit.deterministic else 'no'}**",
        f"- Modello: `{audit.llm_model or 'N/D'}`",
        f"- Fonti dopo reranking/diversificazione: **{retrieval.reranked_sources}**",
        f"- Fonti pubbliche finali: **{retrieval.final_sources}**",
        f"- Fonti effettivamente inserite nel prompt: **{retrieval.prompt_context_sources}**",
        f"- Fonti escluse dal prompt per filtri/budget: **{retrieval.prompt_dropped_sources}**",
        f"- Messaggi history inseriti nel prompt: **{retrieval.history_messages}**",
        f"- Caratteri history nel prompt: **{retrieval.history_chars}**",
        f"- Messaggi history esclusi/collassati: **{retrieval.history_dropped_messages}**",
        f"- Messaggi history troncati: **{retrieval.history_truncated_messages}**",
    ]

    if retrieval.target_document:
        lines.append(f"- Documento target: `{retrieval.target_document}`")
    if retrieval.target_pages:
        lines.append(
            "- Pagine target: `" + ", ".join(str(page) for page in retrieval.target_pages) + "`"
        )

    lines.extend(
        [
            "",
            "#### Retrieval per database",
            f"- Qdrant: **{retrieval.qdrant_hits}** hit su **{retrieval.qdrant_candidates}** candidati",
            f"- PostgreSQL BM25: **{retrieval.postgres_bm25_hits}** hit",
            f"- PostgreSQL exact phrase: **{retrieval.postgres_exact_phrase_hits}** hit",
            f"- Neo4j direct: **{retrieval.neo4j_direct_hits}** hit",
            f"- Neo4j expansion: **{retrieval.neo4j_expanded_hits}** hit",
            "",
            "#### Ranking",
            f"- Dopo quality filter: **{retrieval.kept_after_quality_filters}**",
            f"- Candidati rerank: **{retrieval.rerank_candidates}**",
            f"- Reranker usato: **{'sì' if retrieval.reranker_used else 'no'}**",
            f"- Graph expansion usata: **{'sì' if retrieval.graph_expand_used else 'no'}**",
            f"- Score min/max/media: **{retrieval.score.minimum:.4f} / {retrieval.score.maximum:.4f} / {retrieval.score.average:.4f}**",
        ]
    )

    if retrieval.tier_counts:
        lines.append("")
        lines.append("#### Distribuzione TIER")
        for tier, count in sorted(retrieval.tier_counts.items()):
            lines.append(f"- `{tier}`: **{count}**")

    if retrieval.timings_ms:
        lines.append("")
        lines.append("#### Tempi")
        for name, duration in sorted(retrieval.timings_ms.items()):
            lines.append(f"- `{name}`: **{duration} ms**")

    warnings = tuple(dict.fromkeys((*retrieval.warnings, *audit.warnings)))
    if warnings:
        lines.append("")
        lines.append("#### Warning")
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines).strip()


def format_evaluation_audit_markdown(result: RagEvalResult) -> str:
    lines = [
        "### Audit evaluation",
        f"- Verdict: `{result.verdict}`",
        f"- Faithfulness: **{result.faithfulness:.2f}**",
        f"- Answer relevance: **{result.answer_relevance:.2f}**",
        f"- Context support: **{result.context_support:.2f}**",
        f"- Hallucination risk: **{result.hallucination_risk:.2f}**",
        f"- Source scope violation: **{result.source_scope_violation}**",
    ]

    if result.reason:
        lines.append(f"- Motivo: {result.reason}")
    if result.unsupported_claims:
        lines.append("- Claim non supportati:")
        lines.extend(f"  - {claim}" for claim in result.unsupported_claims)

    return "\n".join(lines).strip()


# =============================================================================
# SINGLETON E WRAPPER COMPATIBILI
# =============================================================================
audit_service: Final[AuditService] = AuditService()


def append_audit_log(
    audit: AuditTrail,
    *,
    raise_on_failure: bool = False,
) -> AuditWriteResult:
    """Wrapper compatibile con il nome utilizzato nel PoC Reflex."""

    return audit_service.persist_query_audit(
        audit,
        raise_on_failure=raise_on_failure,
    )


async def append_audit_log_async(
    audit: AuditTrail,
    *,
    raise_on_failure: bool = False,
) -> AuditWriteResult:
    return await audit_service.persist_query_audit_async(
        audit,
        raise_on_failure=raise_on_failure,
    )


def append_rag_eval_log(
    *,
    query: str,
    answer: str,
    sources: Sequence[SourceItem],
    evaluation: RagEvalResult,
    requested_document: str = "",
    strict_block_applied: bool = False,
    warnings: Sequence[str] = (),
    context: TenantContext | None = None,
    raise_on_failure: bool = False,
    config: RagSettings = settings,
    llm_model: str | None = None,
    evaluation_model: str | None = None,
) -> AuditWriteResult:
    tenant = context or get_tenant_context()
    record = EvaluationAuditRecord.from_evaluation(
        query=query,
        answer=answer,
        sources=sources,
        metrics=evaluation,
        context=tenant,
        requested_document=requested_document,
        llm_model=(config.llm_model_name if llm_model is None else llm_model),
        evaluation_model=(
            config.evaluation_model_name
            if evaluation_model is None
            else evaluation_model
        ),
        corpus_version=config.corpus_version,
        strict_block_applied=strict_block_applied,
        warnings=warnings,
    )
    return audit_service.persist_evaluation(
        record,
        context=tenant,
        raise_on_failure=raise_on_failure,
    )


async def append_rag_eval_log_async(**kwargs: Any) -> AuditWriteResult:
    return await asyncio.to_thread(append_rag_eval_log, **kwargs)


__all__ = [
    "AuditError",
    "AuditIdentityError",
    "AuditPersistenceError",
    "AuditService",
    "AuditSink",
    "AuditSinkOutcome",
    "AuditWriteResult",
    "EvaluationAuditRecord",
    "append_audit_log",
    "append_audit_log_async",
    "append_rag_eval_log",
    "append_rag_eval_log_async",
    "audit_service",
    "create_query_audit",
    "format_evaluation_audit_markdown",
    "format_retrieval_audit_markdown",
]
