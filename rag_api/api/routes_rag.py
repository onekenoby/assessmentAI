"""Route HTTP del servizio Hybrid-RAG multi-tenant.

Il modulo collega il contratto pubblico definito in :mod:`api.schemas` al
servizio applicativo :mod:`core.rag_service`, senza spostare logica di business
nel layer FastAPI.

Responsabilità:
- risoluzione del ``TenantContext`` da un'identità trusted;
- mapping ``RagQueryRequest -> RagQueryCommand``;
- invocazione asincrona del ``RagService``;
- mapping ``RagServiceResult -> RagQueryResponse``;
- esposizione degli endpoint query, liveness e readiness;
- gestione uniforme degli errori API e del correlation ID.

Sicurezza:
- ``organization_id`` e ruoli non vengono mai letti da body o header client;
- in produzione l'identità deve essere inserita da middleware autenticato in
  ``request.state.tenant_identity``;
- il fallback tenant configurato è consentito soltanto con ``POC_MODE=1``;
- eventuali dettagli interni delle eccezioni non vengono restituiti al client;
- tutte le risposte RAG e health usano ``Cache-Control: no-store``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    ApiErrorCode,
    ApiErrorDetail,
    ApiErrorResponse,
    DependencyHealthResponse,
    DependencyState,
    GraphEntityResponse,
    HealthResponse,
    RagDebugResponse,
    RagEvaluationResponse,
    RagQueryRequest,
    RagQueryResponse,
    RetrievalMetricsResponse,
    ScoreSummary,
    ServiceState,
    SourceResponse,
)
from core.audit import AuditIdentityError
from core.config import settings
from core.models import RagServiceResult, RetrievalDebug, SourceItem
from core.rag_service import (
    RagQueryCommand,
    RagServiceConfigurationError,
    RagServiceError,
    RagServiceGenerationError,
    RagServiceRetrievalError,
    RagServiceValidationError,
    rag_service,
)
from core.resources import ResourceNotReadyError, resources
from core.tenant import (
    TenantAuthorizationError,
    TenantContext,
    TenantContextError,
    TrustedTenantIdentity,
    resolve_tenant_context,
)


logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
REQUEST_ID_HEADER = "X-Request-ID"

# ``include_debug`` espone metriche e audit Markdown. In produzione è limitato
# a ruoli esplicitamente autorizzati; nel PoC resta disponibile per i test.
DEBUG_ROLES = frozenset({"admin", "auditor", "rag_admin", "rag-debug"})


# =============================================================================
# ROUTER
# =============================================================================
router = APIRouter(
    prefix="/api/v1/rag",
    tags=["rag"],
)

health_router = APIRouter(tags=["health"])


# =============================================================================
# ECCEZIONE API INTERNA
# =============================================================================
class RagApiException(Exception):
    """Errore destinato al contratto pubblico dell'API.

    Il messaggio deve essere già sicuro per il client. I dettagli tecnici
    completi vengono registrati esclusivamente nei log server-side.
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: ApiErrorCode,
        message: str,
        request_id: UUID | None = None,
        retryable: bool = False,
        details: tuple[ApiErrorDetail, ...] = (),
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = str(message)
        self.request_id = request_id
        self.retryable = bool(retryable)
        self.details = tuple(details)
        self.headers = dict(headers or {})


# =============================================================================
# TENANT DEPENDENCY
# =============================================================================
def _parse_request_id(value: str | None) -> UUID:
    """Valida il correlation ID client oppure ne genera uno nuovo."""

    if value is None or not value.strip():
        return uuid4()

    try:
        return UUID(value.strip())
    except (TypeError, ValueError, AttributeError) as exc:
        raise RagApiException(
            status_code=status.HTTP_400_BAD_REQUEST,
            code=ApiErrorCode.VALIDATION_ERROR,
            message=f"L'header {REQUEST_ID_HEADER} deve contenere un UUID valido.",
            details=(
                ApiErrorDetail(
                    field=REQUEST_ID_HEADER,
                    message="UUID non valido",
                    type="uuid_parsing",
                ),
            ),
        ) from exc


def _coerce_trusted_identity(value: Any) -> TrustedTenantIdentity | None:
    """Converte soltanto identità provenienti da ``request.state``.

    ``request.state`` deve essere popolato da middleware autenticato. Nessun
    header tenant viene letto direttamente da questo router.
    """

    if value is None:
        return None
    if isinstance(value, TrustedTenantIdentity):
        return value
    if isinstance(value, Mapping):
        try:
            return TrustedTenantIdentity.model_validate(dict(value))
        except Exception as exc:
            raise TenantContextError(
                "L'identità trusted fornita dal middleware non è valida"
            ) from exc
    raise TenantContextError(
        "request.state.tenant_identity deve essere TrustedTenantIdentity o mapping"
    )


async def resolve_request_tenant(
    request: Request,
    x_request_id: Annotated[
        str | None,
        Header(alias=REQUEST_ID_HEADER, convert_underscores=False),
    ] = None,
) -> TenantContext:
    """Risoluzione fail-closed del tenant della richiesta.

    Ordine:
    1. usa un ``TenantContext`` già costruito dal middleware, se presente;
    2. usa ``request.state.tenant_identity`` come identità trusted;
    3. usa il tenant PoC solo quando ``settings.poc_mode`` è attivo.
    """

    prebuilt = getattr(request.state, "tenant_context", None)
    if prebuilt is not None:
        if not isinstance(prebuilt, TenantContext):
            raise RagApiException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code=ApiErrorCode.TENANT_CONTEXT_ERROR,
                message="Il contesto tenant interno non è valido.",
            )
        request.state.request_id = prebuilt.request_id
        return prebuilt

    request_id = _parse_request_id(x_request_id)
    request.state.request_id = str(request_id)

    try:
        identity = _coerce_trusted_identity(
            getattr(request.state, "tenant_identity", None)
        )
        context = resolve_tenant_context(
            identity=identity,
            request_id=request_id,
            config=settings,
        )
    except TenantAuthorizationError as exc:
        raise RagApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            code=ApiErrorCode.AUTHORIZATION_ERROR,
            message="L'identità autenticata non è autorizzata per questa richiesta.",
            request_id=request_id,
        ) from exc
    except TenantContextError as exc:
        status_code = (
            status.HTTP_401_UNAUTHORIZED
            if not settings.poc_mode
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        code = (
            ApiErrorCode.AUTHENTICATION_ERROR
            if status_code == status.HTTP_401_UNAUTHORIZED
            else ApiErrorCode.TENANT_CONTEXT_ERROR
        )
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else {}
        raise RagApiException(
            status_code=status_code,
            code=code,
            message=(
                "Identità tenant autenticata obbligatoria."
                if status_code == status.HTTP_401_UNAUTHORIZED
                else "Impossibile costruire il contesto tenant della richiesta."
            ),
            request_id=request_id,
            headers=headers,
        ) from exc

    request.state.tenant_context = context
    return context


# =============================================================================
# MAPPING REQUEST -> SERVICE
# =============================================================================
def _request_to_command(payload: RagQueryRequest) -> RagQueryCommand:
    options = payload.options

    # Passiamo dizionari minimali al core per non creare una dipendenza del
    # RagService dai modelli HTTP Pydantic.
    history = tuple(
        {
            "role": str(message.role),
            "content": message.content,
        }
        for message in payload.history
    )

    return RagQueryCommand(
        query=payload.query,
        conversation_id=payload.conversation_id,
        history=history,
        target_document=options.target_document,
        target_pages=options.target_pages,
        max_sources=options.max_sources,
        include_evaluation=options.include_evaluation,
    )


def _authorize_optional_features(
    payload: RagQueryRequest,
    tenant: TenantContext,
) -> None:
    if not payload.options.include_debug:
        return

    if settings.poc_mode:
        return

    roles = {str(role).strip().lower() for role in tenant.roles}
    if not roles.intersection(DEBUG_ROLES):
        raise RagApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            code=ApiErrorCode.AUTHORIZATION_ERROR,
            message="Il ruolo autenticato non è autorizzato a richiedere il debug RAG.",
            request_id=UUID(tenant.request_id),
        )


# =============================================================================
# MAPPING SERVICE -> RESPONSE
# =============================================================================
def _source_to_response(source: SourceItem) -> SourceResponse:
    return SourceResponse(
        source_id=source.id,
        document_id=source.doc_id,
        filename=source.filename,
        page=source.page,
        page_chunk_index=source.page_chunk_index,
        source_type=source.type,
        score=source.score,
        excerpt=source.excerpt(),
        section_hint=source.section_hint,
        graph_context=tuple(
            GraphEntityResponse(
                name=entity.name,
                type=entity.type,
                relation=entity.relation,
            )
            for entity in source.graph_context
        ),
        tier=str(source.tier),
        scope=str(source.scope),
        classification=source.classification,
        database_origin=source.db_origin,
    )


def _retrieval_to_response(debug: RetrievalDebug) -> RetrievalMetricsResponse:
    return RetrievalMetricsResponse(
        intent=debug.intent,
        wants_evidence=debug.wants_evidence,
        default_tiers=debug.default_tiers,
        qdrant_candidates=debug.qdrant_candidates,
        qdrant_hits=debug.qdrant_hits,
        postgres_bm25_hits=debug.postgres_bm25_hits,
        postgres_exact_phrase_hits=debug.postgres_exact_phrase_hits,
        neo4j_direct_hits=debug.neo4j_direct_hits,
        neo4j_expanded_hits=debug.neo4j_expanded_hits,
        kept_after_quality_filters=debug.kept_after_quality_filters,
        rerank_candidates=debug.rerank_candidates,
        final_sources=debug.final_sources,
        tier_counts=dict(debug.tier_counts),
        score=ScoreSummary(
            minimum=debug.score.minimum,
            maximum=debug.score.maximum,
            average=debug.score.average,
        ),
        reranker_used=debug.reranker_used,
        graph_expand_used=debug.graph_expand_used,
        target_document=debug.target_document,
        timings_ms=dict(debug.timings_ms),
    )


def _evaluation_to_response(result: RagServiceResult) -> RagEvaluationResponse | None:
    evaluation = result.evaluation
    if evaluation is None:
        return None

    return RagEvaluationResponse(
        faithfulness=evaluation.faithfulness,
        answer_relevance=evaluation.answer_relevance,
        context_support=evaluation.context_support,
        hallucination_risk=evaluation.hallucination_risk,
        source_scope_violation=evaluation.source_scope_violation,
        verdict=str(evaluation.verdict),
        unsupported_claims=tuple(evaluation.unsupported_claims),
        supported_claims=tuple(evaluation.supported_claims),
        reason=evaluation.reason,
    )


def _result_to_response(
    result: RagServiceResult,
    payload: RagQueryRequest,
) -> RagQueryResponse:
    include_sources = payload.options.include_sources
    include_debug = payload.options.include_debug
    include_evaluation = payload.options.include_evaluation

    sources = (
        tuple(_source_to_response(source) for source in result.sources)
        if include_sources
        else ()
    )

    debug = None
    if include_debug:
        debug = RagDebugResponse(
            retrieval=_retrieval_to_response(result.retrieval),
            audit_markdown=result.audit_markdown,
            warnings=tuple(result.retrieval.warnings),
        )

    evaluation = (
        _evaluation_to_response(result)
        if include_evaluation
        else None
    )

    return RagQueryResponse(
        request_id=result.request_id,
        conversation_id=result.conversation_id,
        created_at=result.created_at,
        answer=result.answer,
        intent=result.intent,
        answer_mode=result.answer_mode,
        execution_mode=result.execution_mode,
        deterministic=result.deterministic,
        sources=sources,
        debug=debug,
        evaluation=evaluation,
        warnings=result.warnings,
        model=result.model,
        corpus_version=result.corpus_version,
        elapsed_ms=result.elapsed_ms,
    )


# =============================================================================
# ERROR MAPPING
# =============================================================================
def _request_uuid(request: Request) -> UUID | None:
    raw = getattr(request.state, "request_id", None)
    if raw is None:
        return None
    try:
        return UUID(str(raw))
    except (TypeError, ValueError, AttributeError):
        return None


def _is_timeout_error(exc: BaseException) -> bool:
    visited: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (TimeoutError, asyncio.TimeoutError)):
            return True
        if "timeout" in type(current).__name__.lower():
            return True
        current = current.__cause__ or current.__context__

    return False


def _map_service_exception(
    exc: Exception,
    request_id: UUID,
) -> RagApiException:
    """Converte eccezioni core in errori pubblici senza data leakage."""

    if isinstance(exc, RagApiException):
        return exc

    if isinstance(exc, (TenantAuthorizationError,)):
        return RagApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            code=ApiErrorCode.AUTHORIZATION_ERROR,
            message="La richiesta non è autorizzata nel tenant corrente.",
            request_id=request_id,
        )

    if isinstance(exc, TenantContextError):
        return RagApiException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ApiErrorCode.TENANT_CONTEXT_ERROR,
            message="Il contesto tenant della richiesta non è disponibile.",
            request_id=request_id,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if isinstance(exc, (ResourceNotReadyError, RagServiceConfigurationError)):
        return RagApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code=ApiErrorCode.RESOURCE_NOT_READY,
            message="Il servizio RAG non è ancora pronto.",
            request_id=request_id,
            retryable=True,
            headers={"Retry-After": "5"},
        )

    if isinstance(exc, RagServiceRetrievalError):
        return RagApiException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code=ApiErrorCode.RETRIEVAL_ERROR,
            message="Il recupero delle fonti non è temporaneamente disponibile.",
            request_id=request_id,
            retryable=True,
            headers={"Retry-After": "5"},
        )

    if isinstance(exc, RagServiceGenerationError):
        if _is_timeout_error(exc):
            return RagApiException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                code=ApiErrorCode.TIMEOUT,
                message="Il modello non ha completato la risposta entro il timeout previsto.",
                request_id=request_id,
                retryable=True,
            )
        return RagApiException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code=ApiErrorCode.GENERATION_ERROR,
            message="Il modello di generazione non ha prodotto una risposta valida.",
            request_id=request_id,
            retryable=True,
        )

    if isinstance(exc, RagServiceValidationError):
        # Questa eccezione può indicare contaminazione cross-tenant o una
        # violazione degli invarianti interni. Non esponiamo il dettaglio.
        return RagApiException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ApiErrorCode.INTERNAL_ERROR,
            message="La risposta non ha superato i controlli di sicurezza interni.",
            request_id=request_id,
        )

    if isinstance(exc, AuditIdentityError):
        return RagApiException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ApiErrorCode.INTERNAL_ERROR,
            message="La richiesta non può essere completata per un errore di tracciabilità.",
            request_id=request_id,
        )

    if isinstance(exc, ValueError):
        return RagApiException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code=ApiErrorCode.VALIDATION_ERROR,
            message="La richiesta contiene parametri non validi.",
            request_id=request_id,
        )

    if _is_timeout_error(exc):
        return RagApiException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            code=ApiErrorCode.TIMEOUT,
            message="La richiesta non è stata completata entro il timeout previsto.",
            request_id=request_id,
            retryable=True,
        )

    if isinstance(exc, RagServiceError):
        return RagApiException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code=ApiErrorCode.INTERNAL_ERROR,
            message="Il servizio RAG non ha completato la richiesta.",
            request_id=request_id,
        )

    return RagApiException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=ApiErrorCode.INTERNAL_ERROR,
        message="Errore interno del servizio RAG.",
        request_id=request_id,
    )


def _error_json_response(exc: RagApiException) -> JSONResponse:
    payload = ApiErrorResponse(
        request_id=exc.request_id,
        code=exc.code,
        message=exc.message,
        retryable=exc.retryable,
        details=exc.details,
    )

    headers = {
        "Cache-Control": "no-store",
        **exc.headers,
    }
    if exc.request_id is not None:
        headers[REQUEST_ID_HEADER] = str(exc.request_id)

    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


# =============================================================================
# ENDPOINT QUERY
# =============================================================================
_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ApiErrorResponse, "description": "Richiesta non valida"},
    401: {"model": ApiErrorResponse, "description": "Autenticazione assente o non valida"},
    403: {"model": ApiErrorResponse, "description": "Operazione non autorizzata"},
    422: {"model": ApiErrorResponse, "description": "Payload semanticamente non valido"},
    500: {"model": ApiErrorResponse, "description": "Errore interno"},
    502: {"model": ApiErrorResponse, "description": "Errore del modello LLM"},
    503: {"model": ApiErrorResponse, "description": "Servizio non pronto o backend indisponibile"},
    504: {"model": ApiErrorResponse, "description": "Timeout"},
}


@router.post(
    "/query",
    response_model=RagQueryResponse,
    status_code=status.HTTP_200_OK,
    responses=_ERROR_RESPONSES,
    summary="Esegue una query Hybrid-RAG tenant-safe",
    description=(
        "Esegue routing, retrieval Hybrid-RAG, reranking, prompting, generazione, "
        "validation, evaluation opzionale e audit nel perimetro tenant autenticato."
    ),
)
async def query_rag(
    payload: RagQueryRequest,
    request: Request,
    response: Response,
    tenant: Annotated[TenantContext, Depends(resolve_request_tenant)],
) -> RagQueryResponse:
    request_id = UUID(tenant.request_id)

    try:
        _authorize_optional_features(payload, tenant)
        command = _request_to_command(payload)
        
        result = await rag_service.query(
            command,
            tenant_context=tenant,
        )


        api_response = _result_to_response(result, payload)
    except Exception as exc:
        mapped = _map_service_exception(exc, request_id)
        if mapped.status_code >= 500:
            logger.exception(
                "RAG request failed | request_id=%s | error=%s",
                request_id,
                type(exc).__name__,
            )
        else:
            logger.warning(
                "RAG request rejected | request_id=%s | error=%s",
                request_id,
                type(exc).__name__,
            )
        raise mapped from exc

    response.headers[REQUEST_ID_HEADER] = str(api_response.request_id)
    response.headers["Cache-Control"] = "no-store"
    return api_response


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================
def _dependency_state(*, enabled: bool, ready: bool, required: bool) -> DependencyState:
    if not enabled:
        return DependencyState.DISABLED
    if ready:
        return DependencyState.OK
    if required:
        return DependencyState.DOWN
    return DependencyState.DEGRADED


def _health_response_from_snapshot(snapshot: Any) -> HealthResponse:
    dependencies: dict[str, DependencyHealthResponse] = {}

    for dependency in snapshot.dependencies:
        dependencies[dependency.name] = DependencyHealthResponse(
            state=_dependency_state(
                enabled=dependency.enabled,
                ready=dependency.ready,
                required=dependency.required,
            ),
            detail=dependency.detail or "",
        )

    if not snapshot.ready:
        service_state = ServiceState.DOWN
    elif snapshot.degraded:
        service_state = ServiceState.DEGRADED
    else:
        service_state = ServiceState.OK

    return HealthResponse(
        status=service_state,
        service="rag-api",
        version=API_VERSION,
        dependencies=dependencies,
    )


@health_router.get(
    "/health/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
)
async def health_live(response: Response) -> HealthResponse:
    """Conferma che il processo HTTP è vivo, senza interrogare le dipendenze."""

    response.headers["Cache-Control"] = "no-store"
    return HealthResponse(
        status=ServiceState.OK,
        service="rag-api",
        version=API_VERSION,
        dependencies={},
    )


@health_router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={
        503: {
            "model": HealthResponse,
            "description": "Una dipendenza obbligatoria non è pronta",
        }
    },
    summary="Readiness probe",
)
async def health_ready(
    response: Response,
    deep: Annotated[
        bool,
        Query(
            description=(
                "Se true esegue probe reali verso Ollama, Qdrant, Neo4j e PostgreSQL."
            )
        ),
    ] = False,
) -> HealthResponse | JSONResponse:
    """Verifica lo stato delle risorse condivise del motore RAG."""

    snapshot = await asyncio.to_thread(resources.health_snapshot, deep=deep)
    payload = _health_response_from_snapshot(snapshot)

    headers = {"Cache-Control": "no-store"}
    if snapshot.ready:
        for key, value in headers.items():
            response.headers[key] = value
        return payload

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


# =============================================================================
# EXCEPTION HANDLERS DA REGISTRARE IN main.py
# =============================================================================
def _validation_details(exc: RequestValidationError) -> tuple[ApiErrorDetail, ...]:
    details: list[ApiErrorDetail] = []

    for item in exc.errors():
        location = item.get("loc") or ()
        field = ".".join(str(part) for part in location if part not in {"body"}) or None
        context = item.get("ctx") or {}

        # Evita oggetti non JSON-serializzabili e non propaga l'input ricevuto.
        safe_context = {
            str(key): str(value)[:500]
            for key, value in dict(context).items()
            if str(key).lower() not in {"input", "value", "given"}
        }

        details.append(
            ApiErrorDetail(
                field=field,
                message=str(item.get("msg") or "Valore non valido"),
                type=str(item.get("type") or "validation_error"),
                context=safe_context,
            )
        )

    return tuple(details[:50])


def install_rag_exception_handlers(app: FastAPI) -> None:
    """Registra gli handler necessari al contratto ``ApiErrorResponse``.

    Deve essere invocata una sola volta dal futuro ``main.py`` dopo la creazione
    dell'istanza ``FastAPI``.
    """

    if getattr(app.state, "rag_exception_handlers_installed", False):
        return

    @app.exception_handler(RagApiException)
    async def handle_rag_api_exception(
        request: Request,
        exc: RagApiException,
    ) -> JSONResponse:
        if exc.request_id is None:
            exc.request_id = _request_uuid(request)
        return _error_json_response(exc)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _request_uuid(request) or uuid4()
        request.state.request_id = str(request_id)
        api_exc = RagApiException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code=ApiErrorCode.VALIDATION_ERROR,
            message="Il payload della richiesta non è valido.",
            request_id=request_id,
            details=_validation_details(exc),
        )
        return _error_json_response(api_exc)

    app.state.rag_exception_handlers_installed = True


__all__ = [
    "API_VERSION",
    "REQUEST_ID_HEADER",
    "RagApiException",
    "health_router",
    "install_rag_exception_handlers",
    "resolve_request_tenant",
    "router",
]
