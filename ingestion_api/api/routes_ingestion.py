"""Endpoint HTTP della Ingestion API."""
from __future__ import annotations

from uuid import UUID, uuid4
import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from api.schemas import (
    ApiError,
    HealthResponse,
    IngestionRunRequest,
    IngestionRunResponse,
    RunListResponse,
)
from core.config import settings
from core.service import (
    RunNotFoundError,
    ServiceBusyError,
    ServiceClosedError,
    service,
)

REQUEST_ID_HEADER = "X-Request-ID"
router = APIRouter(prefix=f"{settings.api_prefix}/ingestion", tags=["ingestion"])
health_router = APIRouter(prefix="/health", tags=["health"])


def _request_id(request: Request) -> str:
    current = getattr(request.state, "request_id", None)
    if current:
        return str(current)
    value = str(uuid4())
    request.state.request_id = value
    return value


def require_api_key(
    request: Request,
    supplied_key: str | None = Header(default=None, alias=settings.api_key_header),
) -> None:
    if not settings.api_key:
        return
    if supplied_key is None or not hmac.compare_digest(supplied_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "authentication_error",
                "message": "API key non valida o assente.",
                "request_id": _request_id(request),
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )


@router.post(
    "/runs",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
def create_run(payload: IngestionRunRequest, request: Request) -> IngestionRunResponse:
    try:
        record = service.submit(max_jobs=payload.max_jobs)
    except ServiceBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "service_busy",
                "message": str(exc),
                "request_id": _request_id(request),
            },
        ) from exc
    except ServiceClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "service_closed",
                "message": "Servizio ingestion non disponibile.",
                "request_id": _request_id(request),
            },
        ) from exc
    return IngestionRunResponse.model_validate(record)


@router.get(
    "/runs/{run_id}",
    response_model=IngestionRunResponse,
    dependencies=[Depends(require_api_key)],
)
def get_run(run_id: UUID, request: Request) -> IngestionRunResponse:
    try:
        record = service.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "run_not_found",
                "message": "Esecuzione non trovata.",
                "request_id": _request_id(request),
            },
        ) from exc
    return IngestionRunResponse.model_validate(record)


@router.get(
    "/runs",
    response_model=RunListResponse,
    dependencies=[Depends(require_api_key)],
)
def list_runs() -> RunListResponse:
    return RunListResponse(
        items=[IngestionRunResponse.model_validate(item) for item in service.list_runs()]
    )


@health_router.get("/live", response_model=HealthResponse)
def live() -> HealthResponse:
    snapshot = service.health(deep=False)
    snapshot["ready"] = snapshot["state"] != "closed"
    return HealthResponse.model_validate(snapshot)


@health_router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": ApiError}},
)
def ready(response: Response, deep: bool = True) -> HealthResponse:
    snapshot = service.health(deep=deep)
    if not snapshot["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse.model_validate(snapshot)
