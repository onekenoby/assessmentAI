"""Configurazione della Byte API.

Il modulo legge esclusivamente variabili ambiente e non apre connessioni.
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


@dataclass(frozen=True, slots=True)
class ByteApiSettings:
    service_name: str = _env_str("BYTE_API_SERVICE_NAME", "byte-api")
    api_prefix: str = _env_str("BYTE_API_PREFIX", "/api/v1")
    api_key: str = _env_str("BYTE_API_KEY", "")
    api_key_header: str = _env_str("BYTE_API_KEY_HEADER", "X-Byte-Api-Key")
    initialize_on_startup: bool = _env_bool("BYTE_API_INITIALIZE_ON_STARTUP", True)
    startup_strict: bool = _env_bool("BYTE_API_STARTUP_STRICT", False)
    expose_error_details: bool = _env_bool("BYTE_API_EXPOSE_ERROR_DETAILS", False)
    max_file_bytes: int = _env_int("BYTE_API_MAX_FILE_BYTES", 262_144_000)
    max_concurrent_uploads: int = _env_int("BYTE_API_MAX_CONCURRENT_UPLOADS", 4)

    def __post_init__(self) -> None:
        if not self.api_prefix.startswith("/"):
            raise ValueError("BYTE_API_PREFIX deve iniziare con /")
        if self.max_file_bytes <= 0:
            raise ValueError("BYTE_API_MAX_FILE_BYTES deve essere > 0")
        if self.max_concurrent_uploads <= 0:
            raise ValueError("BYTE_API_MAX_CONCURRENT_UPLOADS deve essere > 0")
        if not self.api_key_header:
            raise ValueError("BYTE_API_KEY_HEADER non può essere vuoto")


settings = ByteApiSettings()
