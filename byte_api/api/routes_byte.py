"""Endpoint HTTP della Byte API."""
from __future__ import annotations

import asyncio
import hmac
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from api.schemas import (
    CorpusUploadResponse,
    EvidenceUploadResponse,
    HealthResponse,
)
from core.config import settings
from core.service import ServiceBusyError, ServiceClosedError, service

REQUEST_ID_HEADER = "X-Request-ID"
router = APIRouter(prefix=f"{settings.api_prefix}/byte", tags=["byte-upload"])
health_router = APIRouter(prefix="/health", tags=["health"])


def _engine():
    # Import lazy: l'app può pubblicare liveness anche con dipendenze DB assenti.
    import byte_engine

    return byte_engine


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


async def _read_upload(file: UploadFile, request: Request) -> tuple[str, bytes]:
    filename = file.filename or ""
    max_bytes = settings.max_file_bytes
    data = await file.read(max_bytes + 1)
    await file.close()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "file_too_large",
                "message": f"File superiore al limite di {max_bytes} byte.",
                "request_id": _request_id(request),
            },
        )
    return filename, data


def _raise_upload_error(request: Request, exc: Exception) -> None:
    engine = _engine()
    if isinstance(exc, engine.UploadError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "upload_validation_error",
                "message": str(exc),
                "request_id": _request_id(request),
            },
        ) from exc
    if isinstance(exc, ServiceBusyError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "service_busy",
                "message": str(exc),
                "request_id": _request_id(request),
            },
        ) from exc
    if isinstance(exc, ServiceClosedError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "service_closed",
                "message": "Byte API non disponibile.",
                "request_id": _request_id(request),
            },
        ) from exc
    if isinstance(exc, (engine.DatabaseDependencyError, engine.DatabaseOperationError)):
        details = {"error": str(exc)} if settings.expose_error_details else {}
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "database_unavailable",
                "message": "Operazione sul Database A non completata.",
                "request_id": _request_id(request),
                "details": details,
            },
        ) from exc
    raise exc


@router.post(
    "/corpus",
    response_model=CorpusUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
async def upload_corpus(
    request: Request,
    file: UploadFile = File(..., description="PDF o Markdown da salvare come BYTEA"),
    tier: str = Form(...),
    organization_id: int | None = Form(default=None),
    user_id: int | None = Form(default=None),
    ontology_code: str | None = Form(default=None),
    ontology_label: str | None = Form(default=None),
    area: str | None = Form(default=None),
    subarea: str | None = Form(default=None),
    classification: str = Form(default="internal"),
    pipeline_version: str = Form(default="v1"),
    corpus_version: str = Form(default="v1"),
    embedding_model: str | None = Form(default=None),
    mime_type: str | None = Form(default=None),
) -> CorpusUploadResponse:
    filename, data = await _read_upload(file, request)
    engine = _engine()
    payload = engine.CorpusUpload(
        file=engine.UploadFileData(
            filename=filename,
            data=data,
            mime_type=mime_type,
        ),
        tier=tier,
        organization_id=organization_id,
        user_id=user_id,
        ontology_code=ontology_code,
        ontology_label=ontology_label,
        area=area,
        subarea=subarea,
        classification=classification,
        pipeline_version=pipeline_version,
        corpus_version=corpus_version,
        embedding_model=embedding_model,
    )
    try:
        result = await asyncio.to_thread(service.upload_corpus, payload)
    except Exception as exc:
        _raise_upload_error(request, exc)
    return CorpusUploadResponse.model_validate(result)


@router.post(
    "/evidence",
    response_model=EvidenceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
async def upload_evidence(
    request: Request,
    file: UploadFile = File(..., description="PDF o Markdown di evidenza"),
    organization_id: int = Form(...),
    user_id: int = Form(...),
    assessment_id: int = Form(...),
    response_id: int = Form(...),
    encryption_required: bool = Form(default=True),
    mime_type: str | None = Form(default=None),
) -> EvidenceUploadResponse:
    filename, data = await _read_upload(file, request)
    engine = _engine()
    payload = engine.EvidenceUpload(
        file=engine.UploadFileData(
            filename=filename,
            data=data,
            mime_type=mime_type,
        ),
        organization_id=organization_id,
        user_id=user_id,
        assessment_id=assessment_id,
        response_id=response_id,
        encryption_required=encryption_required,
    )
    try:
        result = await asyncio.to_thread(service.upload_evidence, payload)
    except Exception as exc:
        _raise_upload_error(request, exc)
    return EvidenceUploadResponse.model_validate(result)


@health_router.get("/live", response_model=HealthResponse)
def live() -> HealthResponse:
    snapshot = service.health_snapshot(deep=False)
    # Liveness valuta il processo, non la raggiungibilità del DB.
    snapshot["ready"] = snapshot["state"] != "closed"
    return HealthResponse.model_validate(snapshot)


@health_router.get("/ready", response_model=HealthResponse)
def ready(
    response: Response,
    deep: bool = Query(default=True),
) -> HealthResponse:
    snapshot = service.health_snapshot(deep=deep)
    if not snapshot["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse.model_validate(snapshot)
