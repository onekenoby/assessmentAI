"""Motore PostgreSQL della Byte API.

Deriva da ``INGESTION_carica_documento_bytea_rag.py`` e mantiene le due
modalità originali:

- ``corpus``: upload amministrativo/test con TIER e ontologia;
- ``evidence``: upload ufficiale collegato a una assessment response.

Il modulo non avvia FastAPI e non apre connessioni all'import.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Optional

ALLOWED_TIERS = {"A", "B", "C"}
ALLOWED_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted"}


class UploadError(RuntimeError):
    """Errore di validazione o contratto leggibile dal client."""


class DatabaseDependencyError(RuntimeError):
    """Driver PostgreSQL non disponibile."""


class DatabaseOperationError(RuntimeError):
    """Operazione PostgreSQL non completata."""


@dataclass(frozen=True, slots=True)
class UploadFileData:
    filename: str
    data: bytes
    mime_type: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CorpusUpload:
    file: UploadFileData
    tier: str
    organization_id: Optional[int] = None
    user_id: Optional[int] = None
    ontology_code: Optional[str] = None
    ontology_label: Optional[str] = None
    area: Optional[str] = None
    subarea: Optional[str] = None
    classification: str = "internal"
    pipeline_version: str = "v1"
    corpus_version: str = "v1"
    embedding_model: Optional[str] = None


@dataclass(frozen=True, slots=True)
class EvidenceUpload:
    file: UploadFileData
    organization_id: int
    user_id: int
    assessment_id: int
    response_id: int
    encryption_required: bool = True


@dataclass(frozen=True, slots=True)
class PreparedFile:
    filename: str
    suffix: str
    data: bytes
    mime_type: str
    sha256: str
    size: int


def database_config() -> dict[str, Any]:
    """Restituisce la stessa configurazione ambiente del programma CLI."""
    return {
        "host": os.getenv("PG_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_PORT", "5433")),
        "dbname": os.getenv("SOURCE_PG_DB", "assessment_gestio_tier"),
        "user": os.getenv("PG_USER", "admin"),
        "password": os.getenv("PG_PASS", "admin_password"),
    }


def _load_psycopg2():
    try:
        import psycopg2
        from psycopg2 import Binary
        from psycopg2.extras import Json, RealDictCursor
    except ImportError as exc:
        raise DatabaseDependencyError(
            "psycopg2 non installato: eseguire pip install -r requirements.txt"
        ) from exc
    return psycopg2, Binary, Json, RealDictCursor


def connect():
    psycopg2, _, _, _ = _load_psycopg2()
    try:
        return psycopg2.connect(**database_config())
    except Exception as exc:
        raise DatabaseOperationError("Connessione al Database A non riuscita") from exc


def sanitize_filename(value: str) -> str:
    """Riduce il nome inviato dal client al solo basename portabile."""
    raw = str(value or "").replace("\\", "/").strip()
    if not raw or "\x00" in raw:
        raise UploadError("Nome file non valido")
    name = PurePath(raw).name.strip()
    if not name or name in {".", ".."}:
        raise UploadError("Nome file non valido")
    return name


def detect_mime_type(filename: str, override: Optional[str]) -> str:
    """Determina il MIME type senza limitare i formati caricabili.

    Il valore dichiarato dal client ha precedenza; in sua assenza viene usato
    ``mimetypes`` con fallback neutro a ``application/octet-stream``.
    """
    if override and override.strip():
        return override.strip().lower()

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"

    guessed, _ = mimetypes.guess_type(filename)
    return (guessed or "application/octet-stream").lower()


def detect_source_format(filename: str, mime_type: str) -> str:
    """Restituisce un formato sorgente breve da salvare in rag_document.

    L'estensione, quando presente, resta la fonte primaria. Per file privi di
    estensione viene usato il sottotipo MIME; se non è significativo viene
    restituito ``binary``. La funzione non decide se il file sia ingestibile.
    """
    suffix = Path(filename).suffix.lower().lstrip(".").strip()
    if suffix:
        return suffix[:64]

    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    if not normalized_mime or normalized_mime == "application/octet-stream":
        return "binary"

    if "/" in normalized_mime:
        subtype = normalized_mime.split("/", 1)[1].strip()
        if subtype:
            return subtype[:64]

    return "binary"


def prepare_file(file: UploadFileData, *, max_file_bytes: int) -> PreparedFile:
    """Valida gli aspetti generali del file senza whitelist di formato."""
    filename = sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    data = bytes(file.data or b"")

    if not data:
        raise UploadError("Il file è vuoto")

    if len(data) > int(max_file_bytes):
        raise UploadError(
            f"File troppo grande: limite {int(max_file_bytes)} byte"
        )

    mime_type = detect_mime_type(filename, file.mime_type)

    return PreparedFile(
        filename=filename,
        suffix=suffix,
        data=data,
        mime_type=mime_type,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def set_tenant_context(cur, organization_id: int, user_id: int) -> None:
    cur.execute(
        "SELECT rag_ingestion.fn_set_tenant_context(%s, %s, %s::uuid)",
        (organization_id, user_id, str(uuid.uuid4())),
    )


def clear_tenant_context(cur) -> None:
    try:
        cur.execute("SELECT rag_ingestion.fn_clear_tenant_context()")
    except Exception:
        # La transazione verrà comunque chiusa; non mascherare l'errore principale.
        pass


def verify_database_contract(cur) -> dict[str, Any]:
    cur.execute(
        """
        SELECT current_database() AS database_name,
               session_user AS session_user,
               to_regclass('rag_ingestion.rag_file_blob') IS NOT NULL AS has_blob,
               to_regclass('rag_ingestion.rag_document') IS NOT NULL AS has_document,
               to_regclass('rag_ingestion.rag_document_context') IS NOT NULL AS has_context,
               to_regclass('rag_ingestion.rag_ingestion_job') IS NOT NULL AS has_job
        """
    )
    row = cur.fetchone()
    if not row or not all(row[k] for k in ("has_blob", "has_document", "has_context", "has_job")):
        raise UploadError("Schema rag_ingestion incompleto nel database sorgente")
    return dict(row)


def resolve_ontology_code(
    cur,
    *,
    ontology_code: Optional[str],
    ontology_label: Optional[str],
    area: Optional[str],
    subarea: Optional[str],
) -> tuple[str, str]:
    if ontology_code and ontology_code.strip():
        code = ontology_code.strip().lower()
    elif area and area.strip() and subarea and subarea.strip():
        cur.execute(
            "SELECT rag_ingestion.fn_build_ontology_code(%s, %s) AS ontology_code",
            (area, subarea),
        )
        row = cur.fetchone()
        if not row or not row.get("ontology_code"):
            raise UploadError("fn_build_ontology_code non ha restituito un codice")
        code = str(row["ontology_code"])
    else:
        raise UploadError("corpus richiede ontology_code oppure la coppia area e subarea")

    label = (ontology_label or "").strip()
    if not label:
        if area and area.strip() and subarea and subarea.strip():
            label = f"{area.strip()} / {subarea.strip()}"
        else:
            label = code.replace("__", " / ").replace("_", " ")
    return code, label


def get_or_create_manual_ontology(
    cur,
    *,
    tier: str,
    scope: str,
    organization_id: Optional[int],
    user_id: Optional[int],
    ontology_code: str,
    ontology_label: str,
    area: Optional[str],
    subarea: Optional[str],
) -> int:
    cur.execute(
        """
        SELECT ontology_id
        FROM rag_ingestion.rag_ontology
        WHERE tier_code = %s
          AND scope_code = %s
          AND organization_id IS NOT DISTINCT FROM %s
          AND ontology_code = %s
          AND ontology_source IN ('MANUAL', 'SYSTEM')
          AND is_active = TRUE
        LIMIT 1
        """,
        (tier, scope, organization_id, ontology_code),
    )
    existing = cur.fetchone()
    if existing:
        return int(existing["ontology_id"])

    cur.execute(
        """
        INSERT INTO rag_ingestion.rag_ontology (
            tier_code,
            scope_code,
            organization_id,
            ontology_code,
            ontology_label,
            ontology_source,
            area_label,
            subarea_label,
            created_by_user_id,
            is_active
        )
        VALUES (%s, %s, %s, %s, %s, 'MANUAL', %s, %s, %s, TRUE)
        RETURNING ontology_id
        """,
        (
            tier,
            scope,
            organization_id,
            ontology_code,
            ontology_label,
            area,
            subarea,
            user_id,
        ),
    )
    return int(cur.fetchone()["ontology_id"])


def get_or_create_blob(
    cur,
    *,
    Binary,
    Json,
    scope: str,
    organization_id: Optional[int],
    user_id: Optional[int],
    data: bytes,
    sha256: str,
    size: int,
    mime_type: str,
    original_filename: str,
) -> str:
    cur.execute(
        """
        SELECT file_blob_id
        FROM rag_ingestion.rag_file_blob
        WHERE scope_code = %s
          AND organization_id IS NOT DISTINCT FROM %s
          AND content_sha256 = %s
        LIMIT 1
        """,
        (scope, organization_id, sha256),
    )
    existing = cur.fetchone()
    if existing:
        return str(existing["file_blob_id"])

    details = {
        "manual_test_upload": True,
        "original_filename": original_filename,
        "source": "carica_documento_bytea_rag.py",
    }
    cur.execute(
        """
        INSERT INTO rag_ingestion.rag_file_blob (
            scope_code,
            organization_id,
            content_data,
            content_sha256,
            file_size_bytes,
            mime_type,
            is_encrypted,
            security_scan_status,
            security_scan_details,
            created_by_user_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, FALSE, 'NOT_SCANNED', %s, %s)
        RETURNING file_blob_id
        """,
        (
            scope,
            organization_id,
            Binary(data),
            sha256,
            size,
            mime_type,
            Json(details),
            user_id,
        ),
    )
    return str(cur.fetchone()["file_blob_id"])


def get_or_create_document(
    cur,
    *,
    file_blob_id: str,
    tier: str,
    scope: str,
    organization_id: Optional[int],
    classification: str,
    source_format: str,
    pipeline_version: str,
    corpus_version: str,
    embedding_model: Optional[str],
) -> tuple[str, bool, str]:
    cur.execute(
        """
        SELECT document_id, processing_status::text AS processing_status
        FROM rag_ingestion.rag_document
        WHERE file_blob_id = %s::uuid
          AND tier_code = %s
          AND scope_code = %s
          AND organization_id IS NOT DISTINCT FROM %s
          AND pipeline_version = %s
        LIMIT 1
        """,
        (file_blob_id, tier, scope, organization_id, pipeline_version),
    )
    existing = cur.fetchone()
    if existing:
        return str(existing["document_id"]), False, str(existing["processing_status"])

    cur.execute(
        """
        INSERT INTO rag_ingestion.rag_document (
            file_blob_id,
            tier_code,
            scope_code,
            organization_id,
            processing_status,
            pipeline_version,
            corpus_version,
            classification,
            embedding_model,
            source_format
        )
        VALUES (
            %s::uuid, %s, %s, %s, 'PENDING', %s, %s, %s, %s, %s
        )
        RETURNING document_id, processing_status::text AS processing_status
        """,
        (
            file_blob_id,
            tier,
            scope,
            organization_id,
            pipeline_version,
            corpus_version,
            classification,
            embedding_model,
            source_format,
        ),
    )
    row = cur.fetchone()
    return str(row["document_id"]), True, str(row["processing_status"])


def get_or_create_corpus_context(
    cur,
    *,
    document_id: str,
    ontology_id: int,
    organization_id: Optional[int],
) -> tuple[str, bool]:
    cur.execute(
        """
        SELECT document_context_id
        FROM rag_ingestion.rag_document_context
        WHERE document_id = %s::uuid
          AND ontology_id = %s
          AND context_type = 'CORPUS'
          AND is_active = TRUE
        LIMIT 1
        """,
        (document_id, ontology_id),
    )
    existing = cur.fetchone()
    if existing:
        return str(existing["document_context_id"]), False

    cur.execute(
        """
        INSERT INTO rag_ingestion.rag_document_context (
            document_id,
            ontology_id,
            context_type,
            organization_id,
            processing_status,
            is_active
        )
        VALUES (%s::uuid, %s, 'CORPUS', %s, 'PENDING', TRUE)
        RETURNING document_context_id
        """,
        (document_id, ontology_id, organization_id),
    )
    return str(cur.fetchone()["document_context_id"]), True


def _normalize_positive_int(value: Optional[int], field: str) -> int:
    if value is None:
        raise UploadError(f"{field} obbligatorio")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise UploadError(f"{field} deve essere un intero") from exc
    if normalized <= 0:
        raise UploadError(f"{field} deve essere maggiore di zero")
    return normalized


def upload_evidence(payload: EvidenceUpload, *, max_file_bytes: int) -> dict[str, Any]:
    prepared = prepare_file(payload.file, max_file_bytes=max_file_bytes)
    organization_id = _normalize_positive_int(payload.organization_id, "organization_id")
    user_id = _normalize_positive_int(payload.user_id, "user_id")
    assessment_id = _normalize_positive_int(payload.assessment_id, "assessment_id")
    response_id = _normalize_positive_int(payload.response_id, "response_id")

    _, Binary, _, RealDictCursor = _load_psycopg2()
    conn = connect()
    tenant_context_set = False
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            verify_database_contract(cur)
            set_tenant_context(cur, organization_id, user_id)
            tenant_context_set = True

            cur.execute(
                """
                SELECT *
                FROM rag_ingestion.fn_upload_response_evidence(
                    p_assessment_id          => %s,
                    p_assessment_response_id => %s,
                    p_uploaded_by_user_id    => %s,
                    p_original_filename      => %s,
                    p_mime_type              => %s,
                    p_content_data           => %s,
                    p_encryption_required    => %s
                )
                """,
                (
                    assessment_id,
                    response_id,
                    user_id,
                    prepared.filename,
                    prepared.mime_type,
                    Binary(prepared.data),
                    bool(payload.encryption_required),
                ),
            )
            result = cur.fetchone()
            if not result:
                raise UploadError("fn_upload_response_evidence non ha restituito alcun record")

            cur.execute(
                """
                SELECT job_id, job_type, status, priority, available_at
                FROM rag_ingestion.rag_ingestion_job
                WHERE document_id = %s
                  AND status IN ('PENDING', 'RUNNING')
                ORDER BY created_at DESC
                """,
                (result["document_id"],),
            )
            jobs = [dict(row) for row in cur.fetchall()]
            clear_tenant_context(cur)
            tenant_context_set = False
            conn.commit()

            return {
                "mode": "evidence",
                "filename": prepared.filename,
                "mime_type": prepared.mime_type,
                "sha256": prepared.sha256,
                "file_size_bytes": prepared.size,
                **dict(result),
                "jobs": jobs,
            }
    except UploadError:
        try:
            conn.rollback()
        finally:
            raise
    except Exception as exc:
        try:
            conn.rollback()
        finally:
            raise DatabaseOperationError("Upload evidence non completato") from exc
    finally:
        if tenant_context_set:
            # Il rollback/close annulla comunque il contesto transazionale.
            pass
        conn.close()


def upload_corpus(payload: CorpusUpload, *, max_file_bytes: int) -> dict[str, Any]:
    prepared = prepare_file(payload.file, max_file_bytes=max_file_bytes)

    tier = str(payload.tier or "").strip().upper()
    if tier not in ALLOWED_TIERS:
        raise UploadError("tier deve essere A, B oppure C")

    classification = str(payload.classification or "").strip().lower()
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise UploadError("classification non valida")

    pipeline_version = str(payload.pipeline_version or "").strip()
    corpus_version = str(payload.corpus_version or "").strip()
    if not pipeline_version:
        raise UploadError("pipeline_version non può essere vuota")
    if not corpus_version:
        raise UploadError("corpus_version non può essere vuota")

    if tier == "A":
        scope = "GLOBAL"
        organization_id = None
        user_id = None
        if payload.organization_id is not None or payload.user_id is not None:
            raise UploadError("TIER A è GLOBAL: non passare organization_id o user_id")
    else:
        scope = "ACCOUNT"
        organization_id = _normalize_positive_int(payload.organization_id, "organization_id")
        user_id = _normalize_positive_int(payload.user_id, "user_id")

    _, Binary, Json, RealDictCursor = _load_psycopg2()
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            verify_database_contract(cur)
            if scope == "ACCOUNT":
                set_tenant_context(cur, organization_id, user_id)

            ontology_code, ontology_label = resolve_ontology_code(
                cur,
                ontology_code=payload.ontology_code,
                ontology_label=payload.ontology_label,
                area=payload.area,
                subarea=payload.subarea,
            )
            ontology_id = get_or_create_manual_ontology(
                cur,
                tier=tier,
                scope=scope,
                organization_id=organization_id,
                user_id=user_id,
                ontology_code=ontology_code,
                ontology_label=ontology_label,
                area=payload.area,
                subarea=payload.subarea,
            )
            file_blob_id = get_or_create_blob(
                cur,
                Binary=Binary,
                Json=Json,
                scope=scope,
                organization_id=organization_id,
                user_id=user_id,
                data=prepared.data,
                sha256=prepared.sha256,
                size=prepared.size,
                mime_type=prepared.mime_type,
                original_filename=prepared.filename,
            )
            document_id, document_created, processing_status = get_or_create_document(
                cur,
                file_blob_id=file_blob_id,
                tier=tier,
                scope=scope,
                organization_id=organization_id,
                classification=classification,
                source_format=detect_source_format(
                    prepared.filename,
                    prepared.mime_type,
                ),
                pipeline_version=pipeline_version,
                corpus_version=corpus_version,
                embedding_model=payload.embedding_model,
            )
            document_context_id, context_created = get_or_create_corpus_context(
                cur,
                document_id=document_id,
                ontology_id=ontology_id,
                organization_id=organization_id,
            )

            if not document_created and processing_status == "PENDING":
                cur.execute(
                    """
                    INSERT INTO rag_ingestion.rag_ingestion_job (
                        job_type, document_id, status, priority, available_at
                    )
                    SELECT 'CONTENT_INGESTION', %s::uuid, 'PENDING', 100, NOW()
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM rag_ingestion.rag_ingestion_job
                        WHERE document_id = %s::uuid
                          AND job_type = 'CONTENT_INGESTION'
                          AND status IN ('PENDING', 'RUNNING')
                    )
                    """,
                    (document_id, document_id),
                )

            cur.execute(
                """
                SELECT job_id, job_type, status, priority, available_at
                FROM rag_ingestion.rag_ingestion_job
                WHERE document_id = %s::uuid
                  AND status IN ('PENDING', 'RUNNING')
                ORDER BY created_at DESC
                """,
                (document_id,),
            )
            jobs = [dict(row) for row in cur.fetchall()]

            if scope == "ACCOUNT":
                clear_tenant_context(cur)
            conn.commit()

            return {
                "mode": "corpus",
                "filename": prepared.filename,
                "mime_type": prepared.mime_type,
                "sha256": prepared.sha256,
                "file_size_bytes": prepared.size,
                "tier": tier,
                "scope": scope,
                "organization_id": organization_id,
                "ontology_id": ontology_id,
                "ontology_code": ontology_code,
                "ontology_label": ontology_label,
                "file_blob_id": file_blob_id,
                "document_id": document_id,
                "document_created": document_created,
                "document_context_id": document_context_id,
                "context_created": context_created,
                "jobs": jobs,
            }
    except UploadError:
        try:
            conn.rollback()
        finally:
            raise
    except Exception as exc:
        try:
            conn.rollback()
        finally:
            raise DatabaseOperationError("Upload corpus non completato") from exc
    finally:
        conn.close()


def healthcheck(*, deep: bool = False) -> dict[str, dict[str, Any]]:
    if not deep:
        try:
            _load_psycopg2()
            return {"postgres_source": {"ready": True, "detail": "driver disponibile"}}
        except Exception as exc:
            return {"postgres_source": {"ready": False, "detail": str(exc)}}

    conn = None
    try:
        _, _, _, RealDictCursor = _load_psycopg2()
        conn = connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            contract = verify_database_contract(cur)
        conn.rollback()
        return {
            "postgres_source": {
                "ready": True,
                "detail": (
                    f"db={contract.get('database_name')} | "
                    f"role={contract.get('session_user')} | schema=rag_ingestion"
                ),
            }
        }
    except Exception as exc:
        return {
            "postgres_source": {
                "ready": False,
                "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        }
    finally:
        if conn is not None:
            conn.close()
