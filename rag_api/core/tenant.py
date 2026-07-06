"""Contesto tenant e regole di segregazione del servizio RAG.

Il modulo è indipendente da FastAPI, Reflex, PostgreSQL, Qdrant e Neo4j.
Definisce esclusivamente:

- identità trusted ricevuta dal futuro layer di autenticazione;
- contesto tenant immutabile associato alla singola richiesta;
- propagazione asincrona/thread-safe tramite ``ContextVar``;
- invarianti TIER/scope;
- controlli di visibilità fail-closed condivisi dai repository.

Il body pubblico dell'API non può costruire direttamente un ``TenantContext``.
In produzione il contesto deve essere derivato da JWT, API Gateway, mTLS o da
un altro meccanismo autenticato. In modalità PoC il resolver usa esclusivamente
i valori trusted definiti in ``core.config``.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final, Literal, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.config import RagSettings, settings


TenantScope = Literal["GLOBAL", "ACCOUNT"]
TenantTier = Literal["A", "B", "C", "GRAPH", "USER"]

_ALLOWED_SCOPES: Final[frozenset[str]] = frozenset({"GLOBAL", "ACCOUNT"})
_GLOBAL_TIERS: Final[frozenset[str]] = frozenset({"A"})
_ACCOUNT_TIERS: Final[frozenset[str]] = frozenset({"B", "C"})
_GRAPH_TIER: Final[str] = "GRAPH"
_USER_TIER: Final[str] = "USER"
_ACTIVE_STATUS: Final[str] = "active"
_ALLOWED_USER_SOURCE_IDS: Final[frozenset[str]] = frozenset({"user_input", "error"})


class TenantContextError(RuntimeError):
    """Errore nella costruzione o nell'utilizzo del contesto tenant."""


class TenantAuthorizationError(PermissionError):
    """Il principal autenticato non è autorizzato al tenant richiesto."""


class TrustedTenantIdentity(BaseModel):
    """Identità interna prodotta dal futuro adapter di autenticazione.

    Questo modello non è uno schema HTTP pubblico. Il layer API dovrà
    costruirlo esclusivamente da credenziali già validate.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    organization_id: int = Field(gt=0)
    user_id: str = Field(min_length=1, max_length=256)
    roles: tuple[str, ...] = Field(default_factory=lambda: ("user",))
    allowed_scopes: tuple[TenantScope, ...] = Field(
        default_factory=lambda: ("GLOBAL", "ACCOUNT")
    )
    is_super_admin: bool = False

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(role.strip().lower() for role in value if role.strip())
        )
        if not normalized:
            raise ValueError("roles non può essere vuoto")
        return normalized

    @field_validator("allowed_scopes")
    @classmethod
    def normalize_allowed_scopes(
        cls,
        value: tuple[TenantScope, ...],
    ) -> tuple[TenantScope, ...]:
        normalized = tuple(
            dict.fromkeys(str(scope).strip().upper() for scope in value if str(scope).strip())
        )
        unknown = sorted(set(normalized) - _ALLOWED_SCOPES)
        if unknown:
            raise ValueError(f"scope non validi: {', '.join(unknown)}")
        if not normalized:
            raise ValueError("allowed_scopes non può essere vuoto")
        return normalized  # type: ignore[return-value]


class TenantContext(BaseModel):
    """Contesto immutabile associato a una singola richiesta RAG."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    organization_id: int = Field(gt=0)
    user_id: str = Field(min_length=1, max_length=256)
    roles: tuple[str, ...] = Field(default_factory=lambda: ("user",))
    request_id: str = Field(min_length=36, max_length=36)
    is_super_admin: bool = False
    allowed_scopes: tuple[TenantScope, ...] = Field(
        default_factory=lambda: ("GLOBAL", "ACCOUNT")
    )

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("request_id deve essere un UUID valido") from exc

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(role.strip().lower() for role in value if role.strip())
        )
        if not normalized:
            raise ValueError("roles non può essere vuoto")
        return normalized

    @field_validator("allowed_scopes")
    @classmethod
    def normalize_allowed_scopes(
        cls,
        value: tuple[TenantScope, ...],
    ) -> tuple[TenantScope, ...]:
        normalized = tuple(
            dict.fromkeys(str(scope).strip().upper() for scope in value if str(scope).strip())
        )
        unknown = sorted(set(normalized) - _ALLOWED_SCOPES)
        if unknown:
            raise ValueError(f"scope non validi: {', '.join(unknown)}")
        if not normalized:
            raise ValueError("allowed_scopes non può essere vuoto")
        return normalized  # type: ignore[return-value]

    @model_validator(mode="after")
    def reject_cross_tenant_super_admin_semantics(self) -> "TenantContext":
        """Documenta e preserva il comportamento tenant-bound.

        ``is_super_admin`` può essere usato dal layer autorizzativo per funzioni
        amministrative, ma non concede automaticamente visibilità cross-tenant.
        Anche un super-admin deve avere un ``organization_id`` esplicito per la
        singola richiesta.
        """

        return self

    @property
    def tenant_key(self) -> str:
        return build_tenant_key("ACCOUNT", self.organization_id)


_CURRENT_TENANT_CONTEXT: contextvars.ContextVar[TenantContext | None] = (
    contextvars.ContextVar("rag_tenant_context", default=None)
)


def _new_request_id(request_id: str | UUID | None = None) -> str:
    if request_id is None:
        return str(uuid4())
    try:
        return str(UUID(str(request_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise TenantContextError("request_id deve essere un UUID valido") from exc


def build_poc_identity(config: RagSettings = settings) -> TrustedTenantIdentity:
    """Costruisce l'identità trusted configurata per il PoC."""

    if not config.poc_mode:
        raise TenantContextError("build_poc_identity è consentito solo con POC_MODE attivo")

    return TrustedTenantIdentity(
        organization_id=config.poc_organization_id,
        user_id=config.default_user_id,
        roles=config.default_user_roles,
        allowed_scopes=config.allowed_scopes,
        is_super_admin=False,
    )


def resolve_tenant_context(
    *,
    identity: TrustedTenantIdentity | None = None,
    request_id: str | UUID | None = None,
    config: RagSettings = settings,
) -> TenantContext:
    """Crea il contesto tenant per una richiesta.

    In produzione ``identity`` è obbligatoria. Il fallback configurato è
    disponibile soltanto in modalità PoC e non legge dati dal body HTTP.
    """

    trusted_identity = identity

    if trusted_identity is None:
        if not config.poc_mode:
            raise TenantContextError(
                "Identità tenant trusted obbligatoria quando POC_MODE è disattivato"
            )
        trusted_identity = build_poc_identity(config)

    return TenantContext(
        organization_id=trusted_identity.organization_id,
        user_id=trusted_identity.user_id,
        roles=trusted_identity.roles,
        request_id=_new_request_id(request_id),
        is_super_admin=trusted_identity.is_super_admin,
        allowed_scopes=trusted_identity.allowed_scopes,
    )


def get_tenant_context() -> TenantContext:
    """Restituisce il contesto corrente oppure fallisce in modo chiuso."""

    context = _CURRENT_TENANT_CONTEXT.get()
    if context is None:
        raise TenantContextError(
            "Nessun TenantContext associato alla richiesta corrente"
        )
    return context


def try_get_tenant_context() -> TenantContext | None:
    """Versione non bloccante, utile esclusivamente per logging e health check."""

    return _CURRENT_TENANT_CONTEXT.get()


def current_organization_id() -> int:
    return get_tenant_context().organization_id


def current_request_id() -> str:
    return get_tenant_context().request_id


def current_user_id() -> str:
    return get_tenant_context().user_id


@contextmanager
def bind_tenant_context(context: TenantContext) -> Iterator[TenantContext]:
    """Associa il contesto alla richiesta e lo ripristina sempre in uscita.

    ``ContextVar`` propaga correttamente il valore tra coroutine asyncio. Per
    esecuzioni spostate manualmente su thread, il chiamante deve usare
    ``contextvars.copy_context()`` oppure un adapter che propaghi il contesto.
    """

    token = _CURRENT_TENANT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TENANT_CONTEXT.reset(token)


@contextmanager
def tenant_request_scope(
    *,
    identity: TrustedTenantIdentity | None = None,
    request_id: str | UUID | None = None,
    config: RagSettings = settings,
) -> Iterator[TenantContext]:
    """Risoluzione e binding del tenant in un'unica operazione."""

    context = resolve_tenant_context(
        identity=identity,
        request_id=request_id,
        config=config,
    )
    with bind_tenant_context(context):
        yield context


def optional_positive_int(value: Any) -> int | None:
    """Converte un valore in intero positivo; valori invalidi diventano ``None``."""

    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_scope(value: Any) -> str:
    return str(value or "").strip().upper()


def normalize_tier(value: Any) -> str:
    """Normalizza i tier senza usare confronti per sottostringa.

    In particolare evita il vecchio errore per cui ``GRAPH`` poteva essere
    scambiato per Tier A perché contiene la lettera ``A``.
    """

    tier = str(value or "").strip().upper()
    if tier.startswith("GRAPH"):
        return _GRAPH_TIER
    if tier.startswith("USER"):
        return _USER_TIER
    if tier == "A" or tier == "TIER_A_METHODOLOGY" or tier.endswith("_A_METHODOLOGY"):
        return "A"
    if tier == "B" or tier == "TIER_B_REFERENCE" or tier.endswith("_B_REFERENCE"):
        return "B"
    if (
        tier == "C"
        or tier == "TIER_C_EVIDENCE"
        or tier.endswith("_C_EVIDENCE")
        or "EVIDENCE" in tier
        or "EVIDENZA" in tier
    ):
        return "C"
    return tier


def build_tenant_key(scope: Any, organization_id: Any) -> str:
    """Costruisce il namespace persistente usato da ingestion e RAG."""

    normalized_scope = normalize_scope(scope)

    if normalized_scope == "GLOBAL":
        if organization_id not in (None, ""):
            raise TenantContextError(
                "organization_id deve essere nullo per scope GLOBAL"
            )
        return "GLOBAL"

    if normalized_scope != "ACCOUNT":
        raise TenantContextError(f"Scope multi-tenant non valido: {scope!r}")

    org_id = optional_positive_int(organization_id)
    if org_id is None:
        raise TenantContextError(
            "organization_id positivo obbligatorio per scope ACCOUNT"
        )

    return f"ORG:{org_id}"


def validate_tier_scope_invariants(
    *,
    scope: Any,
    organization_id: Any,
    tier: Any,
    allow_graph_tier: bool = False,
) -> tuple[str, int | None, str]:
    """Valida gli invarianti di persistenza e restituisce valori normalizzati."""

    scope_norm = normalize_scope(scope)
    tier_norm = normalize_tier(tier)
    org_norm = optional_positive_int(organization_id)

    if scope_norm == "GLOBAL":
        valid_global_tiers = set(_GLOBAL_TIERS)
        if allow_graph_tier:
            valid_global_tiers.add(_GRAPH_TIER)
        if org_norm is not None:
            raise TenantContextError(
                "I record GLOBAL non possono avere organization_id"
            )
        if tier_norm not in valid_global_tiers:
            raise TenantContextError(
                f"Tier {tier_norm!r} non valido per scope GLOBAL"
            )
        return scope_norm, None, tier_norm

    if scope_norm == "ACCOUNT":
        valid_account_tiers = set(_ACCOUNT_TIERS)
        if allow_graph_tier:
            valid_account_tiers.add(_GRAPH_TIER)
        if org_norm is None:
            raise TenantContextError(
                "organization_id positivo obbligatorio per scope ACCOUNT"
            )
        if tier_norm not in valid_account_tiers:
            raise TenantContextError(
                f"Tier {tier_norm!r} non valido per scope ACCOUNT"
            )
        return scope_norm, org_norm, tier_norm

    raise TenantContextError(f"Scope non valido: {scope_norm!r}")


def tenant_record_is_visible(
    *,
    scope: Any,
    organization_id: Any,
    tier: Any,
    status: Any = _ACTIVE_STATUS,
    context: TenantContext | None = None,
    allow_graph_tier: bool = False,
    allow_user_tier: bool = False,
    source_id: Any = None,
) -> bool:
    """Regola fail-closed condivisa da tutti i repository.

    Visibilità ordinaria:
    - Tier A: ``GLOBAL`` e ``organization_id IS NULL``;
    - Tier B/C: ``ACCOUNT`` e stesso ``organization_id`` della richiesta;
    - ``GRAPH``: ammesso solo quando il chiamante lo dichiara esplicitamente;
    - ``USER``: ammesso solo per le fonti sintetiche interne autorizzate.

    ``is_super_admin`` non introduce bypass cross-tenant impliciti.
    """

    try:
        tenant = context or get_tenant_context()
    except TenantContextError:
        return False

    scope_norm = normalize_scope(scope)
    tier_norm = normalize_tier(tier)
    status_norm = str(status or "").strip().lower()
    org_norm = optional_positive_int(organization_id)

    if status_norm != _ACTIVE_STATUS:
        return False

    if scope_norm not in tenant.allowed_scopes:
        return False

    if tier_norm == _USER_TIER:
        return (
            allow_user_tier
            and scope_norm == "ACCOUNT"
            and org_norm == tenant.organization_id
            and str(source_id or "").strip() in _ALLOWED_USER_SOURCE_IDS
        )

    if scope_norm == "GLOBAL":
        return (
            org_norm is None
            and (
                tier_norm in _GLOBAL_TIERS
                or (allow_graph_tier and tier_norm == _GRAPH_TIER)
            )
        )

    if scope_norm == "ACCOUNT":
        return (
            org_norm == tenant.organization_id
            and (
                tier_norm in _ACCOUNT_TIERS
                or (allow_graph_tier and tier_norm == _GRAPH_TIER)
            )
        )

    return False


def _read_record_field(record: Any, field: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def record_is_visible(
    record: Any,
    *,
    context: TenantContext | None = None,
    allow_graph_tier: bool = False,
    allow_user_tier: bool = False,
) -> bool:
    """Applica il tenant guard a mapping o modelli/oggetti con attributi omonimi."""

    return tenant_record_is_visible(
        scope=_read_record_field(record, "scope", ""),
        organization_id=_read_record_field(record, "organization_id", None),
        tier=_read_record_field(record, "tier", ""),
        status=_read_record_field(record, "status", ""),
        context=context,
        allow_graph_tier=allow_graph_tier,
        allow_user_tier=allow_user_tier,
        source_id=(
            _read_record_field(record, "id", None)
            or _read_record_field(record, "source_id", None)
        ),
    )


def qdrant_payload_is_visible(
    payload: Mapping[str, Any],
    *,
    context: TenantContext | None = None,
) -> bool:
    """Guard dedicato ai payload Qdrant, senza dipendere dal client Qdrant."""

    return record_is_visible(payload, context=context, allow_graph_tier=False)


T = TypeVar("T")


def filter_visible_records(
    records: Iterable[T],
    *,
    context: TenantContext | None = None,
    allow_graph_tier: bool = False,
    allow_user_tier: bool = False,
    record_adapter: Callable[[T], Any] | None = None,
) -> list[T]:
    """Filtra record eterogenei mantenendo soltanto quelli tenant-visible."""

    visible: list[T] = []
    for item in records:
        record = record_adapter(item) if record_adapter else item
        if record_is_visible(
            record,
            context=context,
            allow_graph_tier=allow_graph_tier,
            allow_user_tier=allow_user_tier,
        ):
            visible.append(item)
    return visible


def tenant_metadata(
    *,
    scope: Any,
    organization_id: Any,
    tier: Any,
    status: Any = _ACTIVE_STATUS,
    corpus_version: str | None = None,
    allow_graph_tier: bool = False,
) -> dict[str, Any]:
    """Produce metadati tenant normalizzati dopo aver validato gli invarianti."""

    scope_norm, org_norm, tier_norm = validate_tier_scope_invariants(
        scope=scope,
        organization_id=organization_id,
        tier=tier,
        allow_graph_tier=allow_graph_tier,
    )

    status_norm = str(status or "").strip().lower()
    if status_norm != _ACTIVE_STATUS:
        raise TenantContextError("Lo stato iniziale supportato è esclusivamente 'active'")

    return {
        "scope": scope_norm,
        "organization_id": org_norm,
        "tier": tier_norm,
        "tenant_key": build_tenant_key(scope_norm, org_norm),
        "status": status_norm,
        "corpus_version": (corpus_version or settings.corpus_version).strip(),
    }
