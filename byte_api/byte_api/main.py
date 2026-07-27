"""Entry point ASGI della Byte API.

Avvio::

    uvicorn main:app --host 0.0.0.0 --port 8020 --workers 1
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes_byte import REQUEST_ID_HEADER, health_router, router
from core.config import settings
from core.service import service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.startup_error = ""
    try:
        if settings.initialize_on_startup:
            try:
                await asyncio.to_thread(
                    service.initialize,
                    strict=settings.startup_strict,
                )
            except Exception as exc:
                app.state.startup_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                logger.exception("Startup Byte API fallito")
                if settings.startup_strict:
                    raise
        yield
    finally:
        await asyncio.to_thread(service.close)


def _valid_request_id(value: str | None) -> str:
    if value:
        try:
            return str(UUID(value.strip()))
        except (ValueError, TypeError, AttributeError):
            pass
    return str(uuid4())


async def http_context_middleware(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    request.state.request_id = _valid_request_id(request.headers.get(REQUEST_ID_HEADER))
    response = await call_next(request)
    response.headers.setdefault(REQUEST_ID_HEADER, request.state.request_id)
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Tenant BYTEA Upload API",
        version="1.0.0",
        description=(
            "Carica PDF e Markdown come BYTEA nel Database A, crea documento, "
            "contesto e job PENDING nello schema rag_ingestion. Non esegue "
            "chunking e non scrive in Qdrant, PostgreSQL B o Neo4j."
        ),
        lifespan=lifespan,
    )
    app.middleware("http")(http_context_middleware)
    app.include_router(router)
    app.include_router(health_router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid4()))
        if isinstance(exc.detail, dict):
            payload = {
                "code": str(exc.detail.get("code") or "http_error"),
                "message": str(exc.detail.get("message") or "Richiesta non completata."),
                "request_id": str(exc.detail.get("request_id") or request_id),
                "details": dict(exc.detail.get("details") or {}),
            }
        else:
            payload = {
                "code": "http_error",
                "message": str(exc.detail),
                "request_id": request_id,
                "details": {},
            }
        return JSONResponse(
            status_code=exc.status_code,
            content=payload,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Richiesta multipart non valida.",
                "request_id": getattr(request.state, "request_id", str(uuid4())),
                "details": {"errors": exc.errors()},
            },
        )

    return app


app = create_app()
