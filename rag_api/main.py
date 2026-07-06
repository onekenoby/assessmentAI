"""Entry point ASGI del servizio Multi-Tenant Hybrid-RAG.

Il modulo crea e configura l'applicazione FastAPI senza incorporare logica di
retrieval, generazione o gestione tenant.

Responsabilità:
- creazione dell'istanza FastAPI;
- gestione del ciclo di vita delle risorse condivise;
- registrazione dei router RAG e health;
- installazione degli exception handler pubblici;
- propagazione del correlation ID;
- applicazione di header HTTP difensivi.

Avvio locale/container::

    uvicorn main:app --host 0.0.0.0 --port 8000

L'import del modulo non carica modelli e non apre connessioni. Le risorse sono
inizializzate dal lifespan quando il server ASGI avvia l'applicazione.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response

from api.routes_rag import (
    API_VERSION,
    REQUEST_ID_HEADER,
    health_router,
    install_rag_exception_handlers,
    router as rag_router,
)
from core.config import settings
from core.resources import (
    close_resources,
    initialize_resources,
    resources,
)


logger = logging.getLogger(__name__)

SERVICE_NAME = "rag-api"
SERVICE_TITLE = "Multi-Tenant Hybrid-RAG API"
SERVICE_DESCRIPTION = (
    "API tenant-safe per retrieval ibrido PostgreSQL, Qdrant e Neo4j, "
    "reranking, generazione Ollama, validazione e audit."
)


# =============================================================================
# LIFESPAN
# =============================================================================
def _safe_error_text(exc: BaseException, max_length: int = 1000) -> str:
    """Restituisce un dettaglio limitato destinato soltanto allo stato interno."""

    text = str(exc).strip() or type(exc).__name__
    return text[:max_length]


def _build_lifespan(
    *,
    initialize_on_startup: bool,
    startup_strict: bool,
) -> Callable[[FastAPI], Any]:
    """Crea il lifespan dell'applicazione.

    ``startup_strict`` viene inoltrato al ``ResourceManager``. In produzione
    l'avvio fallisce se una dipendenza configurata come obbligatoria non è
    disponibile. Nel PoC gli errori di inizializzazione vengono registrati e
    l'HTTP server resta raggiungibile per esporre ``/health/ready``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service_name = SERVICE_NAME
        app.state.service_version = API_VERSION
        app.state.started_at = datetime.now(UTC).isoformat()
        app.state.startup_complete = False
        app.state.startup_error = ""
        app.state.startup_strict = startup_strict

        if initialize_on_startup:
            logger.info(
                "Avvio inizializzazione risorse RAG | strict=%s | poc_mode=%s",
                startup_strict,
                settings.poc_mode,
            )

            try:
                await asyncio.to_thread(
                    initialize_resources,
                    strict=startup_strict,
                )
            except Exception as exc:
                app.state.startup_error = _safe_error_text(exc)
                logger.exception(
                    "Inizializzazione risorse RAG fallita | state=%s",
                    resources.state.value,
                )

                # In produzione il processo deve fallire rapidamente. Nel PoC
                # resta disponibile l'endpoint di readiness per diagnosticare
                # le dipendenze non inizializzate.
                if not settings.poc_mode:
                    raise
            else:
                logger.info(
                    "Risorse RAG inizializzate | state=%s",
                    resources.state.value,
                )
        else:
            logger.info("Inizializzazione risorse disabilitata per questa app")

        app.state.startup_complete = True

        try:
            yield
        finally:
            logger.info("Chiusura risorse RAG avviata")
            try:
                await asyncio.to_thread(close_resources)
            except Exception:
                # Lo shutdown deve tentare di completarsi senza mascherare la
                # causa originaria dell'arresto del processo.
                logger.exception("Chiusura risorse RAG fallita")
            finally:
                app.state.startup_complete = False
                logger.info(
                    "Chiusura risorse RAG completata | state=%s",
                    resources.state.value,
                )

    return lifespan


# =============================================================================
# MIDDLEWARE
# =============================================================================
def _valid_or_new_request_id(raw_value: str | None) -> UUID:
    """Produce sempre un correlation ID valido per il livello HTTP.

    Un header client non valido non viene accettato silenziosamente dal router
    RAG: la dependency dedicata continuerà a restituire HTTP 400. Qui viene
    creato soltanto un ID valido per correlare anche quella risposta di errore.
    """

    if raw_value:
        try:
            return UUID(raw_value.strip())
        except (TypeError, ValueError, AttributeError):
            pass
    return uuid4()


async def _http_context_middleware(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    """Aggiunge correlation ID e header difensivi a tutte le risposte."""

    current = getattr(request.state, "request_id", None)
    if not current:
        request.state.request_id = str(
            _valid_or_new_request_id(request.headers.get(REQUEST_ID_HEADER))
        )

    response = await call_next(request)

    # Il router RAG può sostituire il request ID iniziale con quello proveniente
    # dal TenantContext trusted. Non sovrascriviamo un header già impostato.
    response.headers.setdefault(
        REQUEST_ID_HEADER,
        str(getattr(request.state, "request_id", "") or uuid4()),
    )
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")

    return response


# =============================================================================
# APP FACTORY
# =============================================================================
def create_app(
    *,
    initialize_on_startup: bool = True,
    startup_strict: bool | None = None,
) -> FastAPI:
    """Costruisce l'applicazione FastAPI.

    Parametri destinati soprattutto ai test:
    - ``initialize_on_startup=False`` evita di caricare modelli e database;
    - ``startup_strict`` permette di controllare l'obbligatorietà di Neo4j.

    Nel processo reale ``app`` viene creato con i valori predefiniti:
    produzione strict, PoC degradabile.
    """

    strict = (not settings.poc_mode) if startup_strict is None else startup_strict

    application = FastAPI(
        title=SERVICE_TITLE,
        description=SERVICE_DESCRIPTION,
        version=API_VERSION,
        lifespan=_build_lifespan(
            initialize_on_startup=initialize_on_startup,
            startup_strict=bool(strict),
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "rag",
                "description": (
                    "Query Hybrid-RAG tenant-safe con retrieval, generazione, "
                    "validazione ed audit."
                ),
            },
            {
                "name": "health",
                "description": "Liveness e readiness del servizio.",
            },
        ],
    )

    application.state.service_name = SERVICE_NAME
    application.state.service_version = API_VERSION
    application.state.startup_complete = False
    application.state.startup_error = ""

    application.middleware("http")(_http_context_middleware)

    application.include_router(rag_router)
    application.include_router(health_router)
    install_rag_exception_handlers(application)

    return application


# Istanza ASGI usata da Uvicorn/Gunicorn.
app = create_app()


__all__ = [
    "API_VERSION",
    "SERVICE_DESCRIPTION",
    "SERVICE_NAME",
    "SERVICE_TITLE",
    "app",
    "create_app",
]
