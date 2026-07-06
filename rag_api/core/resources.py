"""Gestione centralizzata delle risorse condivise del servizio RAG.

Il modulo sostituisce le variabili globali e ``init_resources()`` presenti nel
PoC Reflex con un resource manager esplicito, thread-safe e indipendente dal
framework HTTP.

Responsabilità:
- caricamento lazy/controllato di embedder e reranker;
- inizializzazione dei client Ollama/OpenAI-compatible, Qdrant e Neo4j;
- creazione del pool PostgreSQL concorrente;
- verifica delle difese PostgreSQL multi-tenant;
- health check superficiali e profondi;
- acquisizione tenant-safe delle connessioni PostgreSQL;
- chiusura ordinata delle risorse allo shutdown dell'applicazione.

Il modulo NON:
- inizializza risorse durante l'import;
- contiene endpoint FastAPI;
- esegue retrieval o generazione;
- costruisce il TenantContext da dati HTTP non autenticati.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final
from urllib.parse import urlsplit

from core.config import RagSettings, settings
from core.tenant import TenantContext, get_tenant_context


logger = logging.getLogger(__name__)

# Riduce il rumore prodotto dal driver Neo4j senza nascondere le eccezioni che
# il ResourceManager registra e propaga.
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class ResourceError(RuntimeError):
    """Errore base del ciclo di vita delle risorse RAG."""


class ResourceInitializationError(ResourceError):
    """Una o più risorse obbligatorie non sono state inizializzate."""


class ResourceNotReadyError(ResourceError):
    """La risorsa richiesta non è disponibile nello stato corrente."""


class PostgresSecurityError(ResourceError):
    """Le difese PostgreSQL/RLS non rispettano gli invarianti richiesti."""


class ResourceState(StrEnum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """Stato interno di una singola dipendenza."""

    name: str
    enabled: bool
    ready: bool
    required: bool
    detail: str = ""
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceHealthSnapshot:
    """Fotografia serializzabile dello stato complessivo delle risorse."""

    state: ResourceState
    ready: bool
    degraded: bool
    initialization_error: str
    dependencies: tuple[DependencyHealth, ...]
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["dependencies"] = [item.to_dict() for item in self.dependencies]
        return payload


class ResourceManager:
    """Owner unico delle risorse pesanti e delle connessioni del backend.

    L'inizializzazione è idempotente e protetta da ``RLock``. Le risorse
    vengono dapprima costruite in variabili locali e pubblicate sull'istanza
    soltanto quando il ciclo di startup ha raggiunto uno stato coerente.

    ``strict=True`` considera Neo4j obbligatorio quando ``neo4j_enabled`` è
    attivo. Con ``strict=False`` un errore Neo4j produce stato ``DEGRADED``;
    embedder, reranker, Ollama, Qdrant e PostgreSQL (se abilitato) restano
    comunque dipendenze essenziali.
    """

    def __init__(self, config: RagSettings = settings) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._state = ResourceState.NEW
        self._initialization_error = ""
        self._dependency_errors: dict[str, str] = {}
        self._neo4j_required = True

        self._embedder: Any | None = None
        self._reranker: Any | None = None
        self._openai_client: Any | None = None
        self._ollama_session: Any | None = None
        self._qdrant_client: Any | None = None
        self._neo4j_driver: Any | None = None
        self._pg_pool: Any | None = None

    # ------------------------------------------------------------------
    # Stato e proprietà
    # ------------------------------------------------------------------
    @property
    def config(self) -> RagSettings:
        return self._config

    @property
    def state(self) -> ResourceState:
        return self._state

    @property
    def initialization_error(self) -> str:
        return self._initialization_error

    @property
    def is_ready(self) -> bool:
        return self._state in {ResourceState.READY, ResourceState.DEGRADED}

    @property
    def is_degraded(self) -> bool:
        return self._state == ResourceState.DEGRADED

    def require_ready(self) -> None:
        if not self.is_ready:
            detail = f": {self._initialization_error}" if self._initialization_error else ""
            raise ResourceNotReadyError(
                f"ResourceManager non pronto (state={self._state.value}){detail}"
            )

    def get_embedder(self) -> Any:
        self.require_ready()
        if self._embedder is None:
            raise ResourceNotReadyError("Embedder non inizializzato")
        return self._embedder

    def get_reranker(self) -> Any:
        self.require_ready()
        if self._reranker is None:
            raise ResourceNotReadyError("Reranker non inizializzato")
        return self._reranker

    def get_openai_client(self) -> Any:
        """Client OpenAI-compatible verso Ollama.

        La generazione primaria userà il futuro adapter native ``/api/chat``;
        il client rimane disponibile per compatibilità ed evaluation.
        """

        self.require_ready()
        if self._openai_client is None:
            raise ResourceNotReadyError("Client OpenAI-compatible non inizializzato")
        return self._openai_client

    def get_ollama_session(self) -> Any:
        self.require_ready()
        if self._ollama_session is None:
            raise ResourceNotReadyError("Sessione HTTP Ollama non inizializzata")
        return self._ollama_session

    def get_qdrant_client(self) -> Any:
        self.require_ready()
        if self._qdrant_client is None:
            raise ResourceNotReadyError("Client Qdrant non inizializzato")
        return self._qdrant_client

    def get_neo4j_driver(self, *, required: bool = True) -> Any | None:
        self.require_ready()
        if self._neo4j_driver is None and required:
            raise ResourceNotReadyError("Driver Neo4j non disponibile")
        return self._neo4j_driver

    def get_postgres_pool(self, *, required: bool = True) -> Any | None:
        self.require_ready()
        if self._pg_pool is None and required:
            raise ResourceNotReadyError("Pool PostgreSQL non disponibile")
        return self._pg_pool

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def initialize(self, *, strict: bool = True, force: bool = False) -> None:
        """Inizializza tutte le risorse configurate.

        Non viene mai invocato durante l'import. Il futuro ``main.py`` lo
        chiamerà nel lifespan FastAPI.
        """

        with self._lock:
            if self.is_ready and not force:
                return
            if self._state == ResourceState.INITIALIZING:
                raise ResourceInitializationError(
                    "Inizializzazione delle risorse già in corso"
                )
            if force:
                self._close_unlocked()

            self._state = ResourceState.INITIALIZING
            self._initialization_error = ""
            self._dependency_errors = {}
            self._neo4j_required = bool(strict and self._config.neo4j_enabled)

            local: dict[str, Any | None] = {
                "embedder": None,
                "reranker": None,
                "openai_client": None,
                "ollama_session": None,
                "qdrant_client": None,
                "neo4j_driver": None,
                "pg_pool": None,
            }

            try:
                logger.info("Inizializzazione risorse RAG avviata")

                # 1) Verifica Ollama prima di caricare i modelli locali pesanti.
                local["ollama_session"] = self._create_ollama_session()
                local["openai_client"] = self._create_openai_client()
                self._verify_ollama(
                    local["ollama_session"],
                    require_model=True,
                )

                # 2) Database e data stores.
                local["qdrant_client"] = self._create_qdrant_client()
                self._verify_qdrant(local["qdrant_client"], require_collection=True)

                if self._config.neo4j_enabled:
                    try:
                        local["neo4j_driver"] = self._create_neo4j_driver()
                        local["neo4j_driver"].verify_connectivity()
                    except Exception as exc:
                        self._dependency_errors["neo4j"] = self._safe_error(exc)
                        self._safe_close(local.get("neo4j_driver"))
                        local["neo4j_driver"] = None
                        if strict:
                            raise
                        logger.warning(
                            "Neo4j non disponibile: avvio in modalità degradata: %s",
                            exc,
                        )

                if self._config.pg_enrich_enabled:
                    local["pg_pool"] = self._create_postgres_pool()
                    self._ensure_postgres_rag_security(local["pg_pool"])
                    self._verify_postgres(local["pg_pool"])

                # 3) Modelli locali. Nessun download implicito per l'embedder.
                local["embedder"] = self._create_embedder()
                local["reranker"] = self._create_reranker()

                # Pubblicazione atomica delle risorse.
                self._embedder = local["embedder"]
                self._reranker = local["reranker"]
                self._openai_client = local["openai_client"]
                self._ollama_session = local["ollama_session"]
                self._qdrant_client = local["qdrant_client"]
                self._neo4j_driver = local["neo4j_driver"]
                self._pg_pool = local["pg_pool"]

                self._state = (
                    ResourceState.DEGRADED
                    if self._dependency_errors
                    else ResourceState.READY
                )
                logger.info(
                    "Risorse RAG inizializzate (state=%s)",
                    self._state.value,
                )

            except Exception as exc:
                self._initialization_error = self._safe_error(exc)
                self._state = ResourceState.FAILED
                self._close_local_resources(local)
                logger.exception("Inizializzazione risorse RAG fallita")
                raise ResourceInitializationError(
                    f"Inizializzazione risorse RAG fallita: {self._initialization_error}"
                ) from exc

    def _create_embedder(self) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Caricamento embedder %s su %s",
            self._config.embedding_model_name,
            self._config.embedding_device,
        )
        return SentenceTransformer(
            self._config.embedding_model_name,
            device=self._config.embedding_device,
            local_files_only=True,
        )

    def _create_reranker(self) -> Any:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Caricamento reranker %s su %s",
            self._config.reranker_model_name,
            self._config.reranker_device,
        )
        return CrossEncoder(
            self._config.reranker_model_name,
            device=self._config.reranker_device,
        )

    def _create_openai_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            base_url=self._config.ollama_openai_url,
            api_key=self._config.ollama_api_key,
            timeout=float(self._config.llm_timeout_seconds),
        )

    def _create_ollama_session(self) -> Any:
        import requests

        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "assessment-rag-api/1.0",
            }
        )
        return session

    def _create_qdrant_client(self) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
        )

    def _create_neo4j_driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            self._config.neo4j_uri,
            auth=self._config.neo4j_auth,
        )

    def _create_postgres_pool(self) -> Any:
        # FastAPI può servire richieste su più thread; SimpleConnectionPool non
        # è sufficiente. ThreadedConnectionPool protegge getconn/putconn.
        from psycopg2.pool import ThreadedConnectionPool

        return ThreadedConnectionPool(
            self._config.pg_min_connections,
            self._config.pg_max_connections,
            host=self._config.pg_host,
            port=self._config.pg_port,
            dbname=self._config.pg_database,
            user=self._config.pg_user,
            password=self._config.pg_password,
            application_name="assessment-rag-api",
        )

    # ------------------------------------------------------------------
    # PostgreSQL tenant-safe connection handling
    # ------------------------------------------------------------------
    @contextmanager
    def postgres_connection(
        self,
        *,
        context: TenantContext | None = None,
    ) -> Iterator[Any]:
        """Fornisce una connessione con GUC tenant e request impostate.

        Il contesto è obbligatorio: se non viene passato, viene letto dalla
        ``ContextVar`` fail-closed di ``core.tenant``.

        Il chiamante mantiene il controllo esplicito di ``commit``/``rollback``.
        Prima di restituire la connessione al pool, il manager esegue rollback
        difensivo e reset delle GUC per impedire contaminazioni tra richieste.
        """

        pool = self.get_postgres_pool(required=True)
        tenant = context or get_tenant_context()
        conn = pool.getconn()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT "
                    "set_config('app.current_customer_account_id', %s, false), "
                    "set_config('app.current_request_id', %s, false)",
                    (str(tenant.organization_id), tenant.request_id),
                )
            conn.commit()

            yield conn

        except Exception:
            self._safe_rollback(conn)
            raise
        finally:
            self._reset_and_return_postgres_connection(pool, conn)

    def _reset_and_return_postgres_connection(self, pool: Any, conn: Any) -> None:
        if conn is None:
            return

        if getattr(conn, "closed", True):
            try:
                pool.putconn(conn, close=True)
            except Exception:
                logger.exception("Errore rimozione connessione PostgreSQL chiusa")
            return

        try:
            # Chiude qualunque transazione di lettura/scrittura rimasta aperta.
            self._safe_rollback(conn)
            with conn.cursor() as cur:
                cur.execute("RESET app.current_customer_account_id")
                cur.execute("RESET app.current_request_id")
            conn.commit()
        except Exception:
            logger.exception("Errore reset del contesto tenant PostgreSQL")
            self._safe_rollback(conn)
            try:
                pool.putconn(conn, close=True)
            except Exception:
                logger.exception("Errore chiusura connessione PostgreSQL contaminata")
            return

        pool.putconn(conn)

    @staticmethod
    def _safe_rollback(conn: Any) -> None:
        try:
            if conn is not None and not getattr(conn, "closed", True):
                conn.rollback()
        except Exception:
            logger.exception("Rollback PostgreSQL fallito")

    @contextmanager
    def _raw_postgres_connection(self, pool: Any) -> Iterator[Any]:
        """Connessione interna senza tenant, solo per startup e health check."""

        conn = pool.getconn()
        try:
            yield conn
        finally:
            self._safe_rollback(conn)
            pool.putconn(conn)

    def _check_postgres_role_security(self, cur: Any) -> None:
        cur.execute(
            "SELECT current_user, rolsuper, rolbypassrls "
            "FROM pg_roles WHERE rolname = current_user"
        )
        row = cur.fetchone() or ("unknown", False, False)
        user_name = str(row[0])
        is_superuser = bool(row[1])
        bypass_rls = bool(row[2])

        if not (is_superuser or bypass_rls):
            return

        if self._config.poc_mode:
            logger.warning(
                "POC_MODE: ruolo PostgreSQL privilegiato consentito "
                "(user=%s, superuser=%s, bypassrls=%s)",
                user_name,
                is_superuser,
                bypass_rls,
            )
            return

        if self._config.pg_enforce_least_privilege:
            raise PostgresSecurityError(
                "PG_USER non può essere SUPERUSER o BYPASSRLS nel backend RAG"
            )

    def _ensure_postgres_rag_security(self, pool: Any) -> None:
        """Verifica che lo schema sia già stato hardenizzato dall'ingestion."""

        if self._config.pg_auto_harden_schema and not self._config.poc_mode:
            raise PostgresSecurityError(
                "Il backend RAG non può modificare lo schema: "
                "PG_AUTO_HARDEN_SCHEMA deve essere 0"
            )

        with self._raw_postgres_connection(pool) as conn:
            previous_autocommit = bool(getattr(conn, "autocommit", False))
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    self._check_postgres_role_security(cur)

                    if self._config.poc_mode:
                        cur.execute("SELECT 1")
                        return

                    required_columns = {
                        "status",
                        "ingestion_run_id",
                        "tenant_key",
                        "corpus_version",
                        "classification",
                        "embedding_model",
                        "organization_id",
                        "tier",
                        "scope",
                    }
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'document_chunks'
                        """
                    )
                    existing = {str(row[0]) for row in cur.fetchall()}
                    missing = sorted(required_columns - existing)
                    if missing:
                        raise PostgresSecurityError(
                            "Schema document_chunks incompleto; colonne mancanti: "
                            + ", ".join(missing)
                        )

                    self._assert_rls_enabled(cur, "public.document_chunks")
                    self._assert_policy_exists(
                        cur,
                        table_name="document_chunks",
                        policy_name="document_chunks_tenant_select",
                    )

                    cur.execute("SELECT to_regclass('public.rag_query_audit')")
                    if cur.fetchone()[0] is None:
                        raise PostgresSecurityError(
                            "Tabella public.rag_query_audit assente"
                        )

                    self._assert_rls_enabled(cur, "public.rag_query_audit")
                    self._assert_policy_exists(
                        cur,
                        table_name="rag_query_audit",
                        policy_name="rag_query_audit_tenant_all",
                    )
            finally:
                try:
                    conn.autocommit = previous_autocommit
                except Exception:
                    logger.exception("Ripristino autocommit PostgreSQL fallito")

    @staticmethod
    def _assert_rls_enabled(cur: Any, regclass_name: str) -> None:
        cur.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE oid = %s::regclass
            """,
            (regclass_name,),
        )
        row = cur.fetchone()
        if not row or not all(bool(value) for value in row):
            raise PostgresSecurityError(
                f"RLS e FORCE RLS non attive su {regclass_name}"
            )

    @staticmethod
    def _assert_policy_exists(
        cur: Any,
        *,
        table_name: str,
        policy_name: str,
    ) -> None:
        cur.execute(
            """
            SELECT count(*)
            FROM pg_policies
            WHERE schemaname = 'public'
              AND tablename = %s
              AND policyname = %s
            """,
            (table_name, policy_name),
        )
        if int(cur.fetchone()[0]) != 1:
            raise PostgresSecurityError(
                f"Policy {policy_name} assente su public.{table_name}"
            )

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    def health_snapshot(self, *, deep: bool = False) -> ResourceHealthSnapshot:
        """Restituisce lo stato delle dipendenze senza esporre credenziali."""

        checked_at = datetime.now(UTC).isoformat()
        dependencies: list[DependencyHealth] = []

        dependencies.append(
            self._health_for_object(
                "embedder",
                self._embedder,
                enabled=True,
                required=True,
                checked_at=checked_at,
            )
        )
        dependencies.append(
            self._health_for_object(
                "reranker",
                self._reranker,
                enabled=True,
                required=True,
                checked_at=checked_at,
            )
        )

        dependencies.append(
            self._probe_dependency(
                name="ollama",
                enabled=True,
                required=True,
                checked_at=checked_at,
                deep=deep,
                present=self._ollama_session is not None,
                probe=lambda: self._verify_ollama(
                    self._ollama_session,
                    require_model=True,
                ),
            )
        )
        dependencies.append(
            self._probe_dependency(
                name="qdrant",
                enabled=True,
                required=True,
                checked_at=checked_at,
                deep=deep,
                present=self._qdrant_client is not None,
                probe=lambda: self._verify_qdrant(
                    self._qdrant_client,
                    require_collection=True,
                ),
            )
        )
        dependencies.append(
            self._probe_dependency(
                name="neo4j",
                enabled=self._config.neo4j_enabled,
                required=self._neo4j_required,
                checked_at=checked_at,
                deep=deep,
                present=self._neo4j_driver is not None,
                probe=lambda: self._neo4j_driver.verify_connectivity(),
            )
        )
        dependencies.append(
            self._probe_dependency(
                name="postgresql",
                enabled=self._config.pg_enrich_enabled,
                required=self._config.pg_enrich_enabled,
                checked_at=checked_at,
                deep=deep,
                present=self._pg_pool is not None,
                probe=lambda: self._verify_postgres(self._pg_pool),
            )
        )

        required_ok = all(
            (not item.enabled) or (not item.required) or item.ready
            for item in dependencies
        )
        ready = self.is_ready and required_ok
        degraded = self.is_degraded or (
            self.is_ready
            and any(item.enabled and not item.ready for item in dependencies)
        )

        return ResourceHealthSnapshot(
            state=self._state,
            ready=ready,
            degraded=degraded,
            initialization_error=self._initialization_error,
            dependencies=tuple(dependencies),
            checked_at=checked_at,
        )

    def _health_for_object(
        self,
        name: str,
        value: Any,
        *,
        enabled: bool,
        required: bool,
        checked_at: str,
    ) -> DependencyHealth:
        detail = self._dependency_errors.get(name, "")
        return DependencyHealth(
            name=name,
            enabled=enabled,
            ready=(value is not None) if enabled else True,
            required=required,
            detail=detail,
            checked_at=checked_at,
        )

    def _probe_dependency(
        self,
        *,
        name: str,
        enabled: bool,
        required: bool,
        checked_at: str,
        deep: bool,
        present: bool,
        probe: Any,
    ) -> DependencyHealth:
        if not enabled:
            return DependencyHealth(
                name=name,
                enabled=False,
                ready=True,
                required=False,
                detail="disabled",
                checked_at=checked_at,
            )

        if not present:
            return DependencyHealth(
                name=name,
                enabled=True,
                ready=False,
                required=required,
                detail=self._dependency_errors.get(name, "not initialized"),
                checked_at=checked_at,
            )

        if not deep:
            return DependencyHealth(
                name=name,
                enabled=True,
                ready=True,
                required=required,
                detail=self._dependency_errors.get(name, ""),
                checked_at=checked_at,
            )

        try:
            probe()
            return DependencyHealth(
                name=name,
                enabled=True,
                ready=True,
                required=required,
                checked_at=checked_at,
            )
        except Exception as exc:
            return DependencyHealth(
                name=name,
                enabled=True,
                ready=False,
                required=required,
                detail=self._safe_error(exc),
                checked_at=checked_at,
            )

    def _verify_ollama(self, session: Any, *, require_model: bool) -> None:
        if session is None:
            raise ResourceNotReadyError("Sessione Ollama assente")

        response = session.get(
            self._ollama_tags_url(),
            timeout=(
                self._config.ollama_connect_timeout_seconds,
                self._config.llm_timeout_seconds,
            ),
        )
        response.raise_for_status()
        payload = response.json() or {}

        if not require_model:
            return

        models = payload.get("models") or []
        available = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in models
            if isinstance(item, Mapping)
        }
        if self._config.llm_model_name not in available:
            raise ResourceInitializationError(
                f"Modello Ollama {self._config.llm_model_name!r} non disponibile"
            )

    def _verify_qdrant(self, client: Any, *, require_collection: bool) -> None:
        if client is None:
            raise ResourceNotReadyError("Client Qdrant assente")

        result = client.get_collections()
        if not require_collection:
            return

        collection_names = {
            str(getattr(item, "name", "") or "")
            for item in (getattr(result, "collections", None) or [])
        }
        if self._config.qdrant_collection not in collection_names:
            raise ResourceInitializationError(
                f"Collection Qdrant {self._config.qdrant_collection!r} assente"
            )

    def _verify_postgres(self, pool: Any) -> None:
        if pool is None:
            raise ResourceNotReadyError("Pool PostgreSQL assente")

        with self._raw_postgres_connection(pool) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                if not row or int(row[0]) != 1:
                    raise ResourceNotReadyError(
                        "Health check PostgreSQL non valido"
                    )

    def _ollama_tags_url(self) -> str:
        parsed = urlsplit(self._config.ollama_native_chat_url)
        if not parsed.scheme or not parsed.netloc:
            raise ResourceInitializationError(
                "OLLAMA_NATIVE_CHAT_URL non è un URL assoluto valido"
            )
        return f"{parsed.scheme}://{parsed.netloc}/api/tags"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Chiude tutte le risorse; l'operazione è idempotente."""

        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        self._safe_close(self._neo4j_driver)
        self._neo4j_driver = None

        if self._pg_pool is not None:
            try:
                self._pg_pool.closeall()
            except Exception:
                logger.exception("Chiusura pool PostgreSQL fallita")
            self._pg_pool = None

        self._safe_close(self._qdrant_client)
        self._qdrant_client = None

        self._safe_close(self._openai_client)
        self._openai_client = None

        self._safe_close(self._ollama_session)
        self._ollama_session = None

        self._embedder = None
        self._reranker = None
        self._dependency_errors = {}

        self._release_accelerator_memory()
        self._state = ResourceState.CLOSED

    def _close_local_resources(self, local: dict[str, Any | None]) -> None:
        self._safe_close(local.get("neo4j_driver"))

        pool = local.get("pg_pool")
        if pool is not None:
            try:
                pool.closeall()
            except Exception:
                logger.exception("Chiusura pool PostgreSQL parziale fallita")

        self._safe_close(local.get("qdrant_client"))
        self._safe_close(local.get("openai_client"))
        self._safe_close(local.get("ollama_session"))

        local["embedder"] = None
        local["reranker"] = None
        self._release_accelerator_memory()

    @staticmethod
    def _safe_close(resource: Any) -> None:
        if resource is None:
            return
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception(
                    "Chiusura risorsa %s fallita",
                    type(resource).__name__,
                )

    @staticmethod
    def _release_accelerator_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Lo shutdown non deve fallire se torch/CUDA non sono disponibili.
            pass

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        text = str(exc).strip()
        return text[:1000] if text else type(exc).__name__


# Singleton di processo. Non apre connessioni e non carica modelli all'import.
resources: Final[ResourceManager] = ResourceManager(settings)


# -----------------------------------------------------------------------------
# Adapter funzionali per ridurre l'accoppiamento dei moduli successivi.
# -----------------------------------------------------------------------------
def initialize_resources(*, strict: bool = True, force: bool = False) -> None:
    resources.initialize(strict=strict, force=force)


def close_resources() -> None:
    resources.close()


def get_embedder() -> Any:
    return resources.get_embedder()


def get_reranker() -> Any:
    return resources.get_reranker()


def get_openai_client() -> Any:
    return resources.get_openai_client()


def get_ollama_session() -> Any:
    return resources.get_ollama_session()


def get_qdrant_client() -> Any:
    return resources.get_qdrant_client()


def get_neo4j_driver(*, required: bool = True) -> Any | None:
    return resources.get_neo4j_driver(required=required)


@contextmanager
def postgres_connection(
    *,
    context: TenantContext | None = None,
) -> Iterator[Any]:
    with resources.postgres_connection(context=context) as conn:
        yield conn
