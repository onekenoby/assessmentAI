#!/usr/bin/env python3
"""
Caricamento di PDF/Markdown come BYTEA nel database assessment_gestio_tier.

Modalità disponibili:

1) evidence
   Usa l'API ufficiale rag_ingestion.fn_upload_response_evidence().
   Crea atomicamente blob, file_asset, attachment, documento, contesto ontology
   e job PENDING. È il flusso corretto per evidenze collegate a una risposta.

2) corpus
   Modalità amministrativa/test per simulare TIER A/B/C, scope, tenant e
   ontology senza predisporre un assessment completo. Inserisce un contesto
   CORPUS e lascia ai trigger della DDL la creazione del job di ingestion.

Requisiti:
- psycopg2 installato nel venv;
- DDL applicativa + DDL RAG MULTI_TENANT_SAFE già eseguite;
- per evidence: ruolo rag_ingestion_app o rag_ingestion_admin e dati applicativi validi;
- per corpus: ruolo rag_ingestion_admin (o superuser, solo in test).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2 import Binary
from psycopg2.extras import Json, RealDictCursor


DEFAULT_DB_CONFIG = {
    "host": os.getenv("PG_HOST", "127.0.0.1"),
    "port": int(os.getenv("PG_PORT", "5433")),
    "dbname": os.getenv("SOURCE_PG_DB", "assessment_gestio_tier"),
    "user": os.getenv("PG_USER", "admin"),
    "password": os.getenv("PG_PASS", "admin_password"),
}

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown"}
ALLOWED_TIERS = {"A", "B", "C"}
ALLOWED_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted"}


class UploadError(RuntimeError):
    """Errore leggibile relativo al caricamento."""


def detect_mime_type(path: Path, override: Optional[str]) -> str:
    if override:
        return override.strip().lower()
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() in {".md", ".markdown"}:
        return "text/markdown"
    guessed, _ = mimetypes.guess_type(path.name)
    return (guessed or "application/octet-stream").lower()


def read_file(path_value: str) -> tuple[Path, bytes, str, int]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise UploadError(f"File inesistente: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadError("Sono ammessi soltanto file PDF o Markdown (.pdf, .md, .markdown)")
    data = path.read_bytes()
    if not data:
        raise UploadError("Il file è vuoto")
    return path, data, hashlib.sha256(data).hexdigest(), len(data)


def connect():
    return psycopg2.connect(**DEFAULT_DB_CONFIG)


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


def verify_database_contract(cur) -> None:
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


def upload_evidence(args: argparse.Namespace) -> dict[str, Any]:
    path, data, sha256, size = read_file(args.file)
    mime_type = detect_mime_type(path, args.mime_type)

    if args.organization_id is None or args.user_id is None:
        raise UploadError("evidence richiede --organization-id e --user-id")
    if args.assessment_id is None or args.response_id is None:
        raise UploadError("evidence richiede --assessment-id e --response-id")

    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            verify_database_contract(cur)
            set_tenant_context(cur, args.organization_id, args.user_id)

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
                    args.assessment_id,
                    args.response_id,
                    args.user_id,
                    path.name,
                    mime_type,
                    Binary(data),
                    bool(args.encryption_required),
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
            conn.commit()

            return {
                "mode": "evidence",
                "filename": path.name,
                "mime_type": mime_type,
                "sha256": sha256,
                "file_size_bytes": size,
                **dict(result),
                "jobs": jobs,
            }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def resolve_ontology_code(cur, args: argparse.Namespace) -> tuple[str, str]:
    if args.ontology_code:
        code = args.ontology_code.strip().lower()
    elif args.area and args.subarea:
        cur.execute(
            "SELECT rag_ingestion.fn_build_ontology_code(%s, %s) AS ontology_code",
            (args.area, args.subarea),
        )
        code = cur.fetchone()["ontology_code"]
    else:
        raise UploadError("corpus richiede --ontology-code oppure la coppia --area e --subarea")

    label = (args.ontology_label or "").strip()
    if not label:
        if args.area and args.subarea:
            label = f"{args.area.strip()} / {args.subarea.strip()}"
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


def upload_corpus(args: argparse.Namespace) -> dict[str, Any]:
    path, data, sha256, size = read_file(args.file)
    mime_type = detect_mime_type(path, args.mime_type)
    tier = args.tier.upper()
    if tier not in ALLOWED_TIERS:
        raise UploadError("--tier deve essere A, B oppure C")

    classification = args.classification.lower()
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise UploadError("classification non valida")

    if tier == "A":
        scope = "GLOBAL"
        organization_id = None
        user_id = None
        if args.organization_id is not None or args.user_id is not None:
            raise UploadError("TIER A è GLOBAL: non passare --organization-id o --user-id")
    else:
        scope = "ACCOUNT"
        if args.organization_id is None or args.user_id is None:
            raise UploadError("TIER B/C richiedono --organization-id e --user-id")
        organization_id = int(args.organization_id)
        user_id = int(args.user_id)

    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            verify_database_contract(cur)
            if scope == "ACCOUNT":
                set_tenant_context(cur, organization_id, user_id)

            ontology_code, ontology_label = resolve_ontology_code(cur, args)
            ontology_id = get_or_create_manual_ontology(
                cur,
                tier=tier,
                scope=scope,
                organization_id=organization_id,
                user_id=user_id,
                ontology_code=ontology_code,
                ontology_label=ontology_label,
                area=args.area,
                subarea=args.subarea,
            )
            file_blob_id = get_or_create_blob(
                cur,
                scope=scope,
                organization_id=organization_id,
                user_id=user_id,
                data=data,
                sha256=sha256,
                size=size,
                mime_type=mime_type,
                original_filename=path.name,
            )
            document_id, document_created, processing_status = get_or_create_document(
                cur,
                file_blob_id=file_blob_id,
                tier=tier,
                scope=scope,
                organization_id=organization_id,
                classification=classification,
                source_format=path.suffix.lower().lstrip("."),
                pipeline_version=args.pipeline_version,
                corpus_version=args.corpus_version,
                embedding_model=args.embedding_model,
            )
            document_context_id, context_created = get_or_create_corpus_context(
                cur,
                document_id=document_id,
                ontology_id=ontology_id,
                organization_id=organization_id,
            )

            # Un documento nuovo genera automaticamente CONTENT_INGESTION.
            # Per un documento PENDING preesistente senza job aperto, ripristina la coda.
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
                "filename": path.name,
                "mime_type": mime_type,
                "sha256": sha256,
                "file_size_bytes": size,
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
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Carica PDF/Markdown come BYTEA nel database assessment_gestio_tier"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--file", required=True, help="Percorso del PDF o Markdown")
    common.add_argument("--mime-type", help="Override MIME type")
    common.add_argument("--organization-id", type=int)
    common.add_argument("--user-id", type=int)

    evidence = sub.add_parser(
        "evidence",
        parents=[common],
        help="Upload ufficiale di evidenza collegata a una risposta",
    )
    evidence.add_argument("--assessment-id", type=int, required=True)
    evidence.add_argument("--response-id", type=int, required=True)
    evidence.add_argument(
        "--encryption-required",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    corpus = sub.add_parser(
        "corpus",
        parents=[common],
        help="Upload amministrativo/test con metadati simulati",
    )
    corpus.add_argument("--tier", required=True, choices=sorted(ALLOWED_TIERS))
    corpus.add_argument("--ontology-code")
    corpus.add_argument("--ontology-label")
    corpus.add_argument("--area")
    corpus.add_argument("--subarea")
    corpus.add_argument(
        "--classification",
        default="internal",
        choices=sorted(ALLOWED_CLASSIFICATIONS),
    )
    corpus.add_argument("--pipeline-version", default="v1")
    corpus.add_argument("--corpus-version", default="v1")
    corpus.add_argument("--embedding-model")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = upload_evidence(args) if args.mode == "evidence" else upload_corpus(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not result.get("jobs"):
            print("ATTENZIONE: nessun job PENDING/RUNNING risulta associato al documento.", file=sys.stderr)
            return 2
        return 0
    except (UploadError, psycopg2.Error, OSError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
