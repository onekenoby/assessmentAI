"""Generazione delle risposte tramite API nativa Ollama ``/api/chat``.

Il modulo riceve un ``PromptBundle`` già costruito dal prompt layer e restituisce
un risultato strutturato. Non contiene logica di retrieval, routing degli intent,
validazione delle fonti o dipendenze da FastAPI/Reflex.

Scelte architetturali:
- usa esclusivamente l'endpoint nativo ``/api/chat`` per la generazione primaria;
- invia ``stream=False`` e ``think=False`` come parametri top-level;
- non usa mai ``message.thinking`` come risposta finale;
- non effettua fallback automatico all'endpoint OpenAI-compatible;
- applica retry soltanto a errori temporanei o a risposte vuote;
- non inizializza risorse durante l'import.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from core.config import RagSettings, settings
from core.prompting import PromptBundle, PromptMessage
from core.resources import ResourceManager, resources
from core.tenant import try_get_tenant_context


logger = logging.getLogger(__name__)

ThinkLevel: TypeAlias = Literal["low", "medium", "high"]
ThinkMode: TypeAlias = bool | ThinkLevel

_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_EMPTY_RESPONSE_HINT = (
    "Return the final answer now in message.content. Do not return an empty "
    "answer and do not expose reasoning or thinking text."
)


# =============================================================================
# ECCEZIONI
# =============================================================================
class GenerationError(RuntimeError):
    """Errore base del layer di generazione."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        attempt: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.attempt = attempt


class GenerationTransportError(GenerationError):
    """Errore di rete o timeout verso Ollama."""


class GenerationHttpError(GenerationError):
    """Risposta HTTP non valida da Ollama."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retryable: bool,
        retry_after_seconds: float | None = None,
        attempt: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, attempt=attempt)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds


class GenerationProtocolError(GenerationError):
    """Payload Ollama formalmente inatteso o incompleto."""


class EmptyGenerationError(GenerationProtocolError):
    """Ollama ha completato la chiamata senza produrre ``message.content``."""

    def __init__(
        self,
        message: str,
        *,
        thinking_chars: int = 0,
        attempt: int | None = None,
    ) -> None:
        super().__init__(message, retryable=True, attempt=attempt)
        self.thinking_chars = max(0, int(thinking_chars))


# =============================================================================
# MODELLI DEL GENERATION LAYER
# =============================================================================
@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Override per una singola generazione.

    I valori ``None`` ereditano la configurazione globale. ``max_attempts``
    include il primo tentativo; il valore predefinito ``2`` consente quindi un
    solo retry.
    """

    model: str | None = None
    think: ThinkMode = False
    temperature: float | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    repeat_penalty: float | None = None
    keep_alive: str | int | None = None

    max_attempts: int = 2
    retry_backoff_seconds: float = 0.75
    retry_on_empty_content: bool = True
    max_output_chars: int | None = None

    def __post_init__(self) -> None:
        if self.model is not None and not str(self.model).strip():
            raise ValueError("model non può essere vuoto")

        if not isinstance(self.think, bool) and self.think not in {"low", "medium", "high"}:
            raise ValueError("think deve essere bool oppure low/medium/high")

        if self.temperature is not None and not 0.0 <= float(self.temperature) <= 1.0:
            raise ValueError("temperature deve essere compresa tra 0 e 1")

        for name, value in (
            ("num_ctx", self.num_ctx),
            ("num_predict", self.num_predict),
            ("max_attempts", self.max_attempts),
            ("max_output_chars", self.max_output_chars),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} deve essere maggiore di zero")

        if self.repeat_penalty is not None and float(self.repeat_penalty) <= 0:
            raise ValueError("repeat_penalty deve essere maggiore di zero")

        if self.max_attempts > 5:
            raise ValueError("max_attempts non può essere maggiore di 5")

        if not math.isfinite(float(self.retry_backoff_seconds)) or self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds deve essere finito e non negativo")


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Metriche restituite da Ollama, convertite in millisecondi."""

    created_at: str = ""
    done: bool = True
    done_reason: str = ""

    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0

    prompt_eval_count: int = 0
    eval_count: int = 0
    thinking_chars: int = 0

    @property
    def tokens_per_second(self) -> float:
        if self.eval_count <= 0 or self.eval_duration_ms <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ms / 1000.0)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Risultato interno della generazione LLM."""

    content: str
    model: str
    request_id: str
    attempts: int
    elapsed_ms: float
    response_sha256: str
    metrics: GenerationMetrics
    warnings: tuple[str, ...] = field(default_factory=tuple)
    output_truncated: bool = False


@dataclass(frozen=True, slots=True)
class _ResolvedGenerationOptions:
    model: str
    think: ThinkMode
    temperature: float
    num_ctx: int
    num_predict: int
    repeat_penalty: float
    keep_alive: str | int | None
    max_attempts: int
    retry_backoff_seconds: float
    retry_on_empty_content: bool
    max_output_chars: int


# =============================================================================
# GENERATORE OLLAMA NATIVE
# =============================================================================
class OllamaNativeGenerator:
    """Adapter sincrono e asincrono per Ollama ``/api/chat``."""

    def __init__(
        self,
        *,
        resource_manager: ResourceManager = resources,
        config: RagSettings = settings,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self._resources = resource_manager
        self._config = config
        self._sleep = sleep_fn

    def generate(
        self,
        prompt: PromptBundle | Sequence[PromptMessage | Mapping[str, Any]],
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Esegue una generazione non-streaming.

        Il metodo è bloccante. Il layer FastAPI dovrà chiamare
        ``generate_async()`` oppure spostarlo esplicitamente su un worker thread.
        """

        resolved = self._resolve_options(
            options or GenerationOptions(max_attempts=self._config.llm_max_attempts)
        )
        messages = self._normalize_messages(prompt)
        request_id = self._request_id_for_logging()
        session = self._resources.get_ollama_session()

        payload = self._build_payload(messages, resolved)
        warnings: list[str] = []
        started = time.perf_counter()
        last_error: GenerationError | None = None

        for attempt in range(1, resolved.max_attempts + 1):
            attempt_payload = payload
            if attempt > 1 and isinstance(last_error, EmptyGenerationError):
                attempt_payload = self._payload_with_empty_response_hint(payload)
                warnings.append(
                    "Ollama ha restituito message.content vuoto; eseguito un retry "
                    "con richiesta esplicita della sola risposta finale."
                )

            try:
                response_payload = self._post_json(
                    session=session,
                    payload=attempt_payload,
                    attempt=attempt,
                )
                content, metrics, parse_warnings = self._parse_response(
                    response_payload,
                    attempt=attempt,
                )
                warnings.extend(parse_warnings)

                content, truncated = self._enforce_output_limit(
                    content,
                    resolved.max_output_chars,
                )
                if truncated:
                    warnings.append(
                        "Risposta troncata perché eccede MAX_ASSISTANT_CHARS."
                    )

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

                logger.info(
                    "Ollama generation completed | request_id=%s model=%s "
                    "attempt=%s chars=%s elapsed_ms=%.0f eval_count=%s",
                    request_id or "-",
                    resolved.model,
                    attempt,
                    len(content),
                    elapsed_ms,
                    metrics.eval_count,
                )

                return GenerationResult(
                    content=content,
                    model=str(response_payload.get("model") or resolved.model),
                    request_id=request_id,
                    attempts=attempt,
                    elapsed_ms=elapsed_ms,
                    response_sha256=digest,
                    metrics=metrics,
                    warnings=tuple(dict.fromkeys(warnings)),
                    output_truncated=truncated,
                )

            except EmptyGenerationError as exc:
                last_error = exc
                logger.warning(
                    "Ollama empty content | request_id=%s model=%s attempt=%s "
                    "thinking_chars=%s",
                    request_id or "-",
                    resolved.model,
                    attempt,
                    exc.thinking_chars,
                )
                can_retry = (
                    resolved.retry_on_empty_content
                    and attempt < resolved.max_attempts
                )
                if not can_retry:
                    raise

            except GenerationError as exc:
                last_error = exc
                logger.warning(
                    "Ollama generation error | request_id=%s model=%s "
                    "attempt=%s retryable=%s error=%s",
                    request_id or "-",
                    resolved.model,
                    attempt,
                    exc.retryable,
                    self._safe_error(exc),
                )
                if not exc.retryable or attempt >= resolved.max_attempts:
                    raise

            delay = self._retry_delay(
                attempt=attempt,
                base_seconds=resolved.retry_backoff_seconds,
                error=last_error,
            )
            if delay > 0:
                self._sleep(delay)

        # Difesa statica: il ciclo restituisce oppure solleva sempre.
        if last_error is not None:
            raise last_error
        raise GenerationError("Generazione terminata senza risultato")

    async def generate_async(
        self,
        prompt: PromptBundle | Sequence[PromptMessage | Mapping[str, Any]],
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Versione asincrona basata su ``asyncio.to_thread``.

        In Python 3.11+ ``asyncio.to_thread`` propaga il ``ContextVar`` della
        richiesta al worker thread, preservando il request/tenant context usato
        per logging e audit.
        """

        return await asyncio.to_thread(self.generate, prompt, options=options)

    # ------------------------------------------------------------------
    # Risoluzione configurazione e payload
    # ------------------------------------------------------------------
    def _resolve_options(self, options: GenerationOptions) -> _ResolvedGenerationOptions:
        return _ResolvedGenerationOptions(
            model=str(options.model or self._config.llm_model_name).strip(),
            think=options.think,
            temperature=float(
                self._config.llm_temperature
                if options.temperature is None
                else options.temperature
            ),
            num_ctx=int(
                self._config.llm_num_ctx
                if options.num_ctx is None
                else options.num_ctx
            ),
            num_predict=int(
                self._config.llm_num_predict
                if options.num_predict is None
                else options.num_predict
            ),
            repeat_penalty=float(
                self._config.llm_repeat_penalty
                if options.repeat_penalty is None
                else options.repeat_penalty
            ),
            keep_alive=options.keep_alive,
            max_attempts=int(options.max_attempts),
            retry_backoff_seconds=float(options.retry_backoff_seconds),
            retry_on_empty_content=bool(options.retry_on_empty_content),
            max_output_chars=int(
                self._config.max_assistant_chars
                if options.max_output_chars is None
                else options.max_output_chars
            ),
        )

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        options: _ResolvedGenerationOptions,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": options.model,
            "messages": messages,
            "stream": False,
            # Deve rimanere top-level. Non inserirlo dentro ``options``.
            "think": options.think,
            "options": {
                "temperature": options.temperature,
                "num_ctx": options.num_ctx,
                "num_predict": options.num_predict,
                "repeat_penalty": options.repeat_penalty,
            },
        }
        if options.keep_alive is not None:
            payload["keep_alive"] = options.keep_alive
        return payload

    @staticmethod
    def _normalize_messages(
        prompt: PromptBundle | Sequence[PromptMessage | Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        raw_messages: Sequence[PromptMessage | Mapping[str, Any]]
        if isinstance(prompt, PromptBundle):
            raw_messages = prompt.messages
        else:
            raw_messages = prompt

        normalized: list[dict[str, str]] = []
        for index, item in enumerate(raw_messages or ()):  # type: ignore[arg-type]
            if isinstance(item, PromptMessage):
                role = item.role
                content = item.content
            elif isinstance(item, Mapping):
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "").strip()
            else:
                raise TypeError(
                    f"Messaggio in posizione {index} non supportato: "
                    f"{type(item).__name__}"
                )

            if role not in _ALLOWED_ROLES:
                raise ValueError(
                    f"Ruolo messaggio non supportato in posizione {index}: {role!r}"
                )
            if not content:
                raise ValueError(
                    f"Contenuto messaggio vuoto in posizione {index}"
                )
            normalized.append({"role": role, "content": content})

        if not normalized:
            raise ValueError("La richiesta a Ollama deve contenere almeno un messaggio")
        if not any(item["role"] == "user" for item in normalized):
            raise ValueError("La richiesta a Ollama deve contenere almeno un messaggio user")

        return normalized

    # ------------------------------------------------------------------
    # HTTP e parsing
    # ------------------------------------------------------------------
    def _post_json(
        self,
        *,
        session: Any,
        payload: dict[str, Any],
        attempt: int,
    ) -> Mapping[str, Any]:
        try:
            response = session.post(
                self._config.ollama_native_chat_url,
                json=payload,
                timeout=(
                    self._config.ollama_connect_timeout_seconds,
                    self._config.llm_timeout_seconds,
                ),
            )
        except Exception as exc:
            raise GenerationTransportError(
                f"Errore di connessione verso Ollama: {self._safe_error(exc)}",
                retryable=True,
                attempt=attempt,
            ) from exc

        status_code = int(getattr(response, "status_code", 200) or 200)
        if status_code >= 400:
            detail = self._response_error_detail(response)
            raise GenerationHttpError(
                f"Ollama HTTP {status_code}: {detail}",
                status_code=status_code,
                retryable=status_code in _TRANSIENT_HTTP_STATUSES,
                retry_after_seconds=self._parse_retry_after(response),
                attempt=attempt,
            )

        try:
            payload_out = response.json()
        except Exception as exc:
            raise GenerationProtocolError(
                "Ollama ha restituito una risposta non JSON",
                retryable=False,
                attempt=attempt,
            ) from exc

        if not isinstance(payload_out, Mapping):
            raise GenerationProtocolError(
                "Il payload Ollama deve essere un oggetto JSON",
                retryable=False,
                attempt=attempt,
            )

        error_value = payload_out.get("error")
        if error_value:
            raise GenerationProtocolError(
                f"Ollama ha restituito un errore applicativo: "
                f"{str(error_value).strip()[:1000]}",
                retryable=False,
                attempt=attempt,
            )

        return payload_out

    def _parse_response(
        self,
        payload: Mapping[str, Any],
        *,
        attempt: int,
    ) -> tuple[str, GenerationMetrics, tuple[str, ...]]:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise GenerationProtocolError(
                "Campo message assente o non valido nella risposta Ollama",
                retryable=False,
                attempt=attempt,
            )

        raw_content = message.get("content")
        if raw_content is None:
            content = ""
        elif isinstance(raw_content, str):
            content = raw_content.strip()
        else:
            raise GenerationProtocolError(
                "message.content deve essere una stringa",
                retryable=False,
                attempt=attempt,
            )

        raw_thinking = message.get("thinking")
        thinking_chars = len(raw_thinking.strip()) if isinstance(raw_thinking, str) else 0

        metrics = GenerationMetrics(
            created_at=str(payload.get("created_at") or ""),
            done=bool(payload.get("done", True)),
            done_reason=str(payload.get("done_reason") or ""),
            total_duration_ms=self._nanoseconds_to_ms(payload.get("total_duration")),
            load_duration_ms=self._nanoseconds_to_ms(payload.get("load_duration")),
            prompt_eval_duration_ms=self._nanoseconds_to_ms(
                payload.get("prompt_eval_duration")
            ),
            eval_duration_ms=self._nanoseconds_to_ms(payload.get("eval_duration")),
            prompt_eval_count=self._non_negative_int(payload.get("prompt_eval_count")),
            eval_count=self._non_negative_int(payload.get("eval_count")),
            thinking_chars=thinking_chars,
        )

        if not content:
            raise EmptyGenerationError(
                "Ollama ha completato la chiamata senza contenuto finale",
                thinking_chars=thinking_chars,
                attempt=attempt,
            )

        warnings: list[str] = []
        if not metrics.done:
            warnings.append(
                "Ollama ha restituito done=false nonostante stream=false."
            )
        if thinking_chars:
            warnings.append(
                "La risposta conteneva un campo thinking, ignorato dal backend."
            )

        return content, metrics, tuple(warnings)

    # ------------------------------------------------------------------
    # Retry, limiti e diagnostica
    # ------------------------------------------------------------------
    @staticmethod
    def _payload_with_empty_response_hint(payload: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {
            **payload,
            "options": dict(payload.get("options") or {}),
            "messages": [dict(item) for item in (payload.get("messages") or [])],
        }

        messages = cloned["messages"]
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                original = str(messages[index].get("content") or "").rstrip()
                messages[index]["content"] = f"{original}\n\n{_EMPTY_RESPONSE_HINT}"
                break
        else:
            messages.append({"role": "user", "content": _EMPTY_RESPONSE_HINT})

        # Il retry deve mantenere il thinking disabilitato a livello top-level.
        cloned["think"] = False
        cloned["stream"] = False
        return cloned

    @staticmethod
    def _enforce_output_limit(content: str, max_chars: int) -> tuple[str, bool]:
        if len(content) <= max_chars:
            return content, False

        suffix = "\n\n[OUTPUT TRUNCATED BY API LIMIT]"
        if max_chars <= len(suffix):
            return suffix[-max_chars:], True

        available = max_chars - len(suffix)
        candidate = content[:available]

        # Preferisce un confine di paragrafo o riga vicino al limite.
        paragraph_cut = candidate.rfind("\n\n")
        line_cut = candidate.rfind("\n")
        cut = max(paragraph_cut, line_cut)
        if cut >= int(available * 0.70):
            candidate = candidate[:cut]

        return candidate.rstrip() + suffix, True

    @staticmethod
    def _retry_delay(
        *,
        attempt: int,
        base_seconds: float,
        error: GenerationError | None,
    ) -> float:
        if isinstance(error, GenerationHttpError) and error.retry_after_seconds is not None:
            return min(max(0.0, error.retry_after_seconds), 30.0)
        return min(base_seconds * (2 ** max(0, attempt - 1)), 10.0)

    @staticmethod
    def _parse_retry_after(response: Any) -> float | None:
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            return None
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        try:
            parsed = float(str(raw).strip())
            return parsed if math.isfinite(parsed) and parsed >= 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _response_error_detail(response: Any) -> str:
        try:
            data = response.json()
            if isinstance(data, Mapping):
                detail = data.get("error") or data.get("detail") or data.get("message")
                if detail:
                    return str(detail).strip()[:1000]
        except Exception:
            pass

        text = str(getattr(response, "text", "") or "").strip()
        return text[:1000] if text else "nessun dettaglio disponibile"

    @staticmethod
    def _nanoseconds_to_ms(value: Any) -> float:
        try:
            parsed = float(value or 0.0)
            if not math.isfinite(parsed) or parsed < 0:
                return 0.0
            return parsed / 1_000_000.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _non_negative_int(value: Any) -> int:
        try:
            parsed = int(value or 0)
            return max(0, parsed)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _request_id_for_logging() -> str:
        context = try_get_tenant_context()
        return context.request_id if context is not None else ""

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        text = str(exc).strip()
        return text[:1000] if text else type(exc).__name__


# Singleton di processo. Non apre connessioni all'import.
generator = OllamaNativeGenerator()


def generate_response(
    prompt: PromptBundle | Sequence[PromptMessage | Mapping[str, Any]],
    *,
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """Adapter funzionale sincrono."""

    return generator.generate(prompt, options=options)


async def generate_response_async(
    prompt: PromptBundle | Sequence[PromptMessage | Mapping[str, Any]],
    *,
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """Adapter funzionale asincrono per il futuro service FastAPI."""

    return await generator.generate_async(prompt, options=options)


__all__ = [
    "EmptyGenerationError",
    "GenerationError",
    "GenerationHttpError",
    "GenerationMetrics",
    "GenerationOptions",
    "GenerationProtocolError",
    "GenerationResult",
    "GenerationTransportError",
    "OllamaNativeGenerator",
    "generate_response",
    "generate_response_async",
    "generator",
]
