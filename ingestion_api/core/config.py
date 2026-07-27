"""Configurazione statica della Ingestion API.

Il modulo non apre connessioni e non importa il motore di ingestion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} deve essere booleano")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} deve essere un intero") from exc


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def configure_process_environment() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cpu_threads = _env_str("EMBED_CPU_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
    os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpu_threads)


configure_process_environment()


@dataclass(frozen=True, slots=True)
class IngestionApiSettings:
    service_name: str = _env_str("INGESTION_API_SERVICE_NAME", "ingestion-api")
    api_prefix: str = _env_str("INGESTION_API_PREFIX", "/api/v1")
    api_key: str = _env_str("INGESTION_API_KEY", "")
    api_key_header: str = _env_str("INGESTION_API_KEY_HEADER", "X-Ingestion-Api-Key")
    initialize_on_startup: bool = _env_bool("INGESTION_INITIALIZE_ON_STARTUP", True)
    startup_strict: bool = _env_bool("INGESTION_STARTUP_STRICT", False)
    default_max_jobs: int = _env_int("INGESTION_DEFAULT_MAX_JOBS", 1)
    max_jobs_per_run: int = _env_int("INGESTION_MAX_JOBS_PER_RUN", 100)
    run_history_limit: int = _env_int("INGESTION_RUN_HISTORY_LIMIT", 200)
    expose_error_details: bool = _env_bool("INGESTION_EXPOSE_ERROR_DETAILS", False)

    def __post_init__(self) -> None:
        if not self.api_prefix.startswith("/"):
            raise ValueError("INGESTION_API_PREFIX deve iniziare con /")
        if self.default_max_jobs <= 0:
            raise ValueError("INGESTION_DEFAULT_MAX_JOBS deve essere > 0")
        if self.max_jobs_per_run <= 0:
            raise ValueError("INGESTION_MAX_JOBS_PER_RUN deve essere > 0")
        if self.default_max_jobs > self.max_jobs_per_run:
            raise ValueError("INGESTION_DEFAULT_MAX_JOBS supera il massimo")
        if self.run_history_limit <= 0:
            raise ValueError("INGESTION_RUN_HISTORY_LIMIT deve essere > 0")


settings = IngestionApiSettings()
