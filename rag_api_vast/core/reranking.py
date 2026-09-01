"""Scoring, reranking e diversificazione dei candidati RAG.

Il modulo isola la fase di ranking presente nell'ultimo ``gui_reflex.py``:

1. Reciprocal Rank Fusion tra score vettoriale, PostgreSQL BM25 e Neo4j;
2. boost controllati per filename, documento target, pagina e tipo di contenuto;
3. politica TIER A/B/C;
4. reranking tramite CrossEncoder;
5. diversificazione per documento e pagina.

Non contiene retrieval, query ai database, stato tenant, FastAPI o Reflex. Il
chiamante deve fornire candidati già filtrati secondo il perimetro tenant.

La combinazione degli score conserva il comportamento del PoC:

    final_score = cross_encoder_score + pre_rerank_score

In assenza del CrossEncoder, o in caso di errore, ``final_score`` coincide con
``pre_rerank_score``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.config import RagSettings, settings
from core.models import RagIntent, RetrievalCandidate


logger = logging.getLogger(__name__)


# =============================================================================
# TIPI ED ERRORI
# =============================================================================
class RerankingError(RuntimeError):
    """Errore irreversibile nella configurazione del ranking."""


@runtime_checkable
class CrossEncoderLike(Protocol):
    """Interfaccia minima richiesta al reranker SentenceTransformers."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Pesi della fase precedente al CrossEncoder.

    I default corrispondono ai valori utilizzati nell'ultimo RAG Reflex.
    Sono raccolti in un oggetto separato per rendere la policy testabile e
    sostituibile senza introdurre costanti sparse nel codice.
    """

    rrf_k: int = 60
    rrf_weight: float = 1.0

    filename_boost_per_token: float = 0.02
    filename_max_token_hits: int = 3

    document_scope_boost: float = 0.20
    requested_page_boost: float = 0.30

    formula_type_boost: float = 0.25
    chart_type_boost: float = 0.20
    table_type_boost: float = 0.20

    cross_encoder_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k deve essere maggiore di zero")
        if self.filename_max_token_hits < 0:
            raise ValueError("filename_max_token_hits non può essere negativo")

        numeric_values = (
            self.rrf_weight,
            self.filename_boost_per_token,
            self.document_scope_boost,
            self.requested_page_boost,
            self.formula_type_boost,
            self.chart_type_boost,
            self.table_type_boost,
            self.cross_encoder_weight,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("tutti i pesi devono essere numeri finiti")


@dataclass(frozen=True, slots=True)
class RankingContext:
    """Contesto già classificato dal router della query.

    ``wants_evidence`` e ``formula_query`` devono essere calcolati dal futuro
    router/retrieval service. In questo modo il reranker non duplica la logica
    di intent detection e resta deterministico.
    """

    query_text: str
    intent: RagIntent | str = RagIntent.TEXT
    wants_evidence: bool = False
    formula_query: bool = False
    requested_pages: tuple[int, ...] = ()
    target_document: str = ""
    query_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        query = str(self.query_text or "").strip()
        if not query:
            raise ValueError("query_text non può essere vuota")
        object.__setattr__(self, "query_text", query)

        intent_value = str(self.intent or RagIntent.TEXT).strip().lower()
        allowed_intents = {item.value for item in RagIntent}
        if intent_value not in allowed_intents:
            raise ValueError(f"intent non valido: {self.intent!r}")
        object.__setattr__(self, "intent", intent_value)

        pages: list[int] = []
        seen_pages: set[int] = set()
        for raw_page in self.requested_pages:
            page = int(raw_page)
            if page <= 0:
                raise ValueError("requested_pages può contenere solo pagine positive")
            if page not in seen_pages:
                seen_pages.add(page)
                pages.append(page)
        object.__setattr__(self, "requested_pages", tuple(pages))

        tokens = tuple(
            dict.fromkeys(
                str(token).strip().lower()
                for token in self.query_tokens
                if str(token).strip()
            )
        )
        object.__setattr__(self, "query_tokens", tokens)
        object.__setattr__(self, "target_document", str(self.target_document or "").strip())


@dataclass(frozen=True, slots=True)
class FilenameBoostStat:
    filename: str
    raw_token_matches: int
    affected_candidates: int


@dataclass(frozen=True, slots=True)
class RerankingResult:
    """Risultato completo della fase ranking."""

    candidates: tuple[RetrievalCandidate, ...]
    input_count: int
    preselected_count: int
    final_count: int
    reranker_used: bool
    exhaustive: bool
    elapsed_ms: float
    filename_boosts: tuple[FilenameBoostStat, ...] = ()
    warnings: tuple[str, ...] = ()


# =============================================================================
# NORMALIZZAZIONE E TOKEN DI FILENAME
# =============================================================================
_FILENAME_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_\-]+")

# Il filename boost non deve essere guidato da parole conversazionali o da
# riferimenti generici a documenti/fonti.
_FILENAME_STOPWORDS = frozenset(
    {
        "della", "delle", "degli", "dello", "dalla", "dalle", "dagli",
        "nella", "nelle", "negli", "nello", "alla", "alle", "agli",
        "sulla", "sulle", "sugli", "sullo", "questo", "questa", "questi",
        "queste", "quello", "quella", "quelli", "quelle", "sono",
        "presente", "presenti", "quale", "quali", "cosa", "come", "dove",
        "quando", "perché", "perche", "spiega", "spiegami", "riporta",
        "mostra", "mostrami", "dimmi", "elenca", "trova", "cerca", "voglio",
        "vorrei", "riguardo", "inerente", "relativo", "secondo", "basandoti",
        "documento", "documenti", "file", "fonte", "fonti", "testo", "pagina",
        "pagine", "sezione", "capitolo", "document", "documents", "source",
        "sources", "text", "page", "pages", "section", "chapter", "what",
        "which", "where", "when", "explain", "show", "tell", "list", "find",
        "search", "report", "about", "this", "that", "these", "those",
        "according", "regarding", "based", "give", "please",
    }
)


def normalize_document_name(value: str) -> str:
    """Normalizza un filename nello stesso modo usato dal PoC."""

    if not value:
        return ""

    normalized = os.path.basename(str(value).lower().strip())
    normalized = re.sub(r"\.(pdf|md|txt|docx|html|csv|xlsx)$", "", normalized)
    normalized = re.sub(r"[_\-\s]+out$", "", normalized)
    normalized = re.sub(r"[_\-\s]+output$", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def extract_filename_tokens(query_text: str) -> tuple[str, ...]:
    """Estrae token utili esclusivamente al filename boost.

    Mantiene acronimi brevi e parole di almeno quattro caratteri, replicando
    l'obiettivo di ``extract_rag_tokens`` senza importare la futura pipeline di
    retrieval.
    """

    output: list[str] = []
    seen: set[str] = set()

    for raw in _FILENAME_TOKEN_RE.findall(query_text or ""):
        clean = raw.strip().strip(".,:;!?()[]{}\"'")
        if not clean:
            continue

        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10
        is_mixed_acronym = bool(re.fullmatch(r"[A-Za-z]{1,5}\d{0,3}", clean)) and 2 <= len(clean) <= 10
        is_useful_word = len(clean) > 3

        token = clean.lower()
        if not (is_acronym or is_mixed_acronym or is_useful_word):
            continue
        if token in _FILENAME_STOPWORDS or token in seen:
            continue

        seen.add(token)
        output.append(token)

    return tuple(output)


# =============================================================================
# RRF E POLICY TIER
# =============================================================================
def _vector_score(candidate: RetrievalCandidate) -> float:
    return float(candidate.score_vec or candidate.score_base or 0.0)


def apply_rrf_scoring(
    candidates: Sequence[RetrievalCandidate],
    *,
    k: int = 60,
) -> list[RetrievalCandidate]:
    """Applica Reciprocal Rank Fusion in-place e restituisce una lista.

    Le tre graduatorie sono:
    - score vettoriale Qdrant, con fallback a ``score_base``;
    - score PostgreSQL BM25;
    - score Neo4j.

    I candidati con score non positivo non partecipano alla relativa
    graduatoria, come nel RAG originale.
    """

    if k <= 0:
        raise ValueError("k deve essere maggiore di zero")

    ranked_candidates = list(candidates)
    for candidate in ranked_candidates:
        candidate.score_rrf = 0.0

    vector_rank = sorted(
        (candidate for candidate in ranked_candidates if _vector_score(candidate) > 0.0),
        key=lambda candidate: _vector_score(candidate),
        reverse=True,
    )
    bm25_rank = sorted(
        (candidate for candidate in ranked_candidates if candidate.score_bm25 > 0.0),
        key=lambda candidate: candidate.score_bm25,
        reverse=True,
    )
    graph_rank = sorted(
        (candidate for candidate in ranked_candidates if candidate.score_graph > 0.0),
        key=lambda candidate: candidate.score_graph,
        reverse=True,
    )

    for rank, candidate in enumerate(vector_rank):
        candidate.score_rrf += 1.0 / (k + rank + 1)
    for rank, candidate in enumerate(bm25_rank):
        candidate.score_rrf += 1.0 / (k + rank + 1)
    for rank, candidate in enumerate(graph_rank):
        candidate.score_rrf += 1.0 / (k + rank + 1)

    return ranked_candidates


def tier_score_delta(
    tier: str,
    *,
    wants_evidence: bool,
    config: RagSettings = settings,
) -> float:
    """Calcola il boost/penalty TIER senza analizzare nuovamente la query."""

    normalized = str(tier or "").strip().upper()

    if normalized == "A":
        return float(config.tier_boost_a)
    if normalized == "B":
        return float(config.tier_boost_b)
    if normalized == "C":
        if config.tier_c_penalty_if_not_evidence and not wants_evidence:
            return -float(config.tier_penalty_c)
        return 0.0

    # GRAPH e USER non ricevono boost metodologici.
    return 0.0


# =============================================================================
# PRE-RERANK SCORING
# =============================================================================
def _candidate_matches_target_document(
    candidate: RetrievalCandidate,
    target_document: str,
) -> bool:
    target_norm = normalize_document_name(target_document)
    if not target_norm:
        return False

    candidate_names = (
        normalize_document_name(candidate.filename),
        normalize_document_name(str(candidate.metadata.get("source_name") or "")),
        normalize_document_name(str(candidate.metadata.get("filename") or "")),
    )
    return target_norm in candidate_names


def _has_explicit_document_scope(candidate: RetrievalCandidate) -> bool:
    try:
        return float(candidate.metadata.get("score_doc_scope") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _append_origin_marker(origin: str, marker: str) -> str:
    current = str(origin or "Unknown").strip() or "Unknown"
    if marker in current:
        return current
    return f"{current} {marker}".strip()


def _intent_type_boost(
    candidate: RetrievalCandidate,
    context: RankingContext,
    weights: RankingWeights,
) -> float:
    source_type = str(candidate.type or "text").lower()

    if context.formula_query and source_type == "formula":
        return weights.formula_type_boost
    if context.intent == RagIntent.CHART.value and source_type in {"image", "chart"}:
        return weights.chart_type_boost
    if context.intent == RagIntent.TABLE.value and source_type == "table":
        return weights.table_type_boost
    return 0.0


def apply_pre_rerank_scoring(
    candidates: Sequence[RetrievalCandidate],
    *,
    context: RankingContext,
    config: RagSettings = settings,
    weights: RankingWeights = RankingWeights(),
) -> tuple[list[RetrievalCandidate], tuple[FilenameBoostStat, ...]]:
    """Applica RRF e tutti i boost precedenti al CrossEncoder."""

    ranked = apply_rrf_scoring(candidates, k=weights.rrf_k)
    tokens = context.query_tokens or extract_filename_tokens(context.query_text)
    requested_pages = set(context.requested_pages)

    filename_counter: Counter[tuple[str, int]] = Counter()

    for candidate in ranked:
        filename_lower = candidate.filename.lower()
        raw_filename_hits = sum(1 for token in tokens if token in filename_lower)
        capped_filename_hits = min(raw_filename_hits, weights.filename_max_token_hits)
        filename_boost = weights.filename_boost_per_token * capped_filename_hits

        if capped_filename_hits > 0:
            candidate.origin = _append_origin_marker(candidate.origin, "[TARGET FILE]")
            filename_counter[(candidate.filename, raw_filename_hits)] += 1

        candidate.score_tier_delta = tier_score_delta(
            str(candidate.tier),
            wants_evidence=context.wants_evidence,
            config=config,
        )

        doc_scope_match = (
            _has_explicit_document_scope(candidate)
            or _candidate_matches_target_document(candidate, context.target_document)
        )
        document_boost = weights.document_scope_boost if doc_scope_match else 0.0

        page_boost = (
            weights.requested_page_boost
            if requested_pages and candidate.page in requested_pages
            else 0.0
        )

        intent_boost = _intent_type_boost(candidate, context, weights)

        pre_rerank_score = (
            weights.rrf_weight * candidate.score_rrf
            + filename_boost
            + candidate.score_tier_delta
            + document_boost
            + page_boost
            + intent_boost
        )

        # Prima del CrossEncoder il punteggio finale coincide con il pre-score.
        candidate.final_score = float(pre_rerank_score)

        ranking_components = {
            "rrf": float(candidate.score_rrf),
            "rrf_weight": float(weights.rrf_weight),
            "filename_token_hits_raw": raw_filename_hits,
            "filename_token_hits_used": capped_filename_hits,
            "filename_boost": float(filename_boost),
            "tier_delta": float(candidate.score_tier_delta),
            "document_scope_boost": float(document_boost),
            "requested_page_boost": float(page_boost),
            "intent_type_boost": float(intent_boost),
            "pre_rerank_score": float(pre_rerank_score),
        }
        candidate.metadata = {
            **candidate.metadata,
            "ranking_components": ranking_components,
        }

    ranked.sort(key=_candidate_sort_key)

    stats = tuple(
        FilenameBoostStat(
            filename=filename,
            raw_token_matches=raw_hits,
            affected_candidates=affected,
        )
        for (filename, raw_hits), affected in sorted(
            filename_counter.items(),
            key=lambda item: (-item[0][1], item[0][0].lower()),
        )
    )

    return ranked, stats


# =============================================================================
# CROSSENCODER
# =============================================================================
def _coerce_scores(raw_scores: Any, expected: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()

    if isinstance(raw_scores, (int, float)):
        values = [float(raw_scores)]
    else:
        try:
            values = list(raw_scores)
        except TypeError as exc:
            raise RerankingError("Il reranker ha restituito uno score non iterabile") from exc

    flattened: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RerankingError(
                    "Il reranker ha restituito score multidimensionali non supportati"
                )
            value = value[0]

        parsed = float(value)
        if not math.isfinite(parsed):
            raise RerankingError("Il reranker ha restituito uno score non finito")
        flattened.append(parsed)

    if len(flattened) != expected:
        raise RerankingError(
            f"Numero score reranker inatteso: attesi={expected}, ricevuti={len(flattened)}"
        )

    return flattened


def _predict_cross_encoder(
    reranker: CrossEncoderLike,
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int,
) -> list[float]:
    """Invoca CrossEncoder mantenendo compatibilità con fake e versioni diverse."""

    try:
        raw_scores = reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except TypeError:
        # Fake test doubles o versioni che non accettano tutti i kwargs.
        raw_scores = reranker.predict(pairs)

    return _coerce_scores(raw_scores, len(pairs))


def apply_cross_encoder_reranking(
    candidates: Sequence[RetrievalCandidate],
    *,
    query_text: str,
    reranker: CrossEncoderLike | None,
    batch_size: int = 16,
    cross_encoder_weight: float = 1.0,
) -> tuple[list[RetrievalCandidate], bool, tuple[str, ...]]:
    """Applica il CrossEncoder con fallback deterministico al pre-score."""

    ranked = list(candidates)
    if not ranked:
        return ranked, False, ()

    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di zero")
    if not math.isfinite(float(cross_encoder_weight)):
        raise ValueError("cross_encoder_weight deve essere finito")

    if reranker is None:
        for candidate in ranked:
            candidate.score_rerank = 0.0
            # final_score è già il pre_rerank_score.
        ranked.sort(key=_candidate_sort_key)
        return ranked, False, ()

    pairs = [(query_text, candidate.content or "") for candidate in ranked]

    try:
        scores = _predict_cross_encoder(reranker, pairs, batch_size=batch_size)

        for candidate, raw_score in zip(ranked, scores, strict=True):
            components = dict(candidate.metadata.get("ranking_components") or {})
            pre_score = float(components.get("pre_rerank_score", candidate.final_score))

            candidate.score_rerank = raw_score
            candidate.final_score = pre_score + cross_encoder_weight * raw_score

            components.update(
                {
                    "cross_encoder_score": raw_score,
                    "cross_encoder_weight": float(cross_encoder_weight),
                    "final_score": float(candidate.final_score),
                }
            )
            candidate.metadata = {
                **candidate.metadata,
                "ranking_components": components,
            }

        ranked.sort(key=_candidate_sort_key)
        return ranked, True, ()

    except Exception as exc:
        warning = f"Reranker non disponibile; usato il pre-score: {exc}"
        logger.warning(warning)

        for candidate in ranked:
            candidate.score_rerank = 0.0
            components = dict(candidate.metadata.get("ranking_components") or {})
            pre_score = float(components.get("pre_rerank_score", candidate.final_score))
            candidate.final_score = pre_score
            components.update(
                {
                    "cross_encoder_score": 0.0,
                    "cross_encoder_weight": float(cross_encoder_weight),
                    "final_score": pre_score,
                    "reranker_fallback": True,
                }
            )
            candidate.metadata = {
                **candidate.metadata,
                "ranking_components": components,
            }

        ranked.sort(key=_candidate_sort_key)
        return ranked, False, (warning,)


# =============================================================================
# DIVERSIFICAZIONE
# =============================================================================
def _candidate_document_key(candidate: RetrievalCandidate) -> str:
    return (
        candidate.doc_id
        or normalize_document_name(candidate.filename)
        or candidate.filename.lower()
        or candidate.id
    )


def _candidate_sort_key(candidate: RetrievalCandidate) -> tuple[Any, ...]:
    """Ordinamento deterministico: score decrescente, poi provenance."""

    return (
        -float(candidate.final_score),
        candidate.filename.lower(),
        int(candidate.page),
        int(candidate.page_chunk_index),
        candidate.id,
    )


def diversify_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    max_per_page: int,
    max_per_document: int,
    final_k: int,
) -> list[RetrievalCandidate]:
    """Limita duplicazioni per documento e pagina mantenendo i migliori score."""

    if max_per_page <= 0:
        raise ValueError("max_per_page deve essere maggiore di zero")
    if max_per_document <= 0:
        raise ValueError("max_per_document deve essere maggiore di zero")
    if final_k <= 0:
        raise ValueError("final_k deve essere maggiore di zero")

    selected: list[RetrievalCandidate] = []
    page_counts: Counter[tuple[str, int]] = Counter()
    document_counts: Counter[str] = Counter()

    for candidate in sorted(candidates, key=_candidate_sort_key):
        document_key = _candidate_document_key(candidate)
        page_key = (document_key, int(candidate.page))

        if document_counts[document_key] >= max_per_document:
            continue
        if page_counts[page_key] >= max_per_page:
            continue

        selected.append(candidate)
        document_counts[document_key] += 1
        page_counts[page_key] += 1

        if len(selected) >= final_k:
            break

    return selected


# Alias compatibile con il nome usato nel PoC.
def diversify(
    candidates: Sequence[RetrievalCandidate],
    max_per_page: int,
    max_per_doc: int,
    final_k: int,
) -> list[RetrievalCandidate]:
    return diversify_candidates(
        candidates,
        max_per_page=max_per_page,
        max_per_document=max_per_doc,
        final_k=final_k,
    )


# =============================================================================
# ORCHESTRAZIONE
# =============================================================================
class RerankingEngine:
    """Orchestratore stateless della fase di ranking.

    L'istanza può essere riutilizzata tra richieste. Il modello CrossEncoder è
    posseduto da ``core.resources.ResourceManager`` e viene soltanto referenziato
    da questo engine.
    """

    def __init__(
        self,
        *,
        config: RagSettings = settings,
        reranker: CrossEncoderLike | None = None,
        weights: RankingWeights = RankingWeights(),
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size deve essere maggiore di zero")

        self._config = config
        self._reranker = reranker
        self._weights = weights
        self._batch_size = batch_size

    @property
    def config(self) -> RagSettings:
        return self._config

    @property
    def weights(self) -> RankingWeights:
        return self._weights

    def rank(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        context: RankingContext,
        exhaustive: bool = False,
        rerank_limit: int | None = None,
        final_k: int | None = None,
        max_per_page: int | None = None,
        max_per_document: int | None = None,
    ) -> RerankingResult:
        """Esegue l'intera fase RRF -> CrossEncoder -> diversification.

        ``exhaustive=True`` replica il lookup documentale esaustivo delle
        formule: nessun taglio prima del reranker e nessuna diversificazione.
        """

        started = time.perf_counter()
        input_count = len(candidates)

        # Copia profonda: il retrieval mantiene i propri candidati originali e
        # questa fase può annotare score e metadati senza side effect esterni.
        working = [candidate.model_copy(deep=True) for candidate in candidates]

        if not working:
            return RerankingResult(
                candidates=(),
                input_count=0,
                preselected_count=0,
                final_count=0,
                reranker_used=False,
                exhaustive=exhaustive,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        effective_rerank_limit = (
            int(rerank_limit)
            if rerank_limit is not None
            else int(self._config.rerank_candidates)
        )
        effective_final_k = (
            int(final_k) if final_k is not None else int(self._config.final_sources)
        )
        effective_max_per_page = (
            int(max_per_page)
            if max_per_page is not None
            else int(self._config.max_per_page)
        )
        effective_max_per_document = (
            int(max_per_document)
            if max_per_document is not None
            else int(self._config.max_per_document)
        )

        for name, value in (
            ("rerank_limit", effective_rerank_limit),
            ("final_k", effective_final_k),
            ("max_per_page", effective_max_per_page),
            ("max_per_document", effective_max_per_document),
        ):
            if value <= 0:
                raise ValueError(f"{name} deve essere maggiore di zero")

        pre_ranked, filename_stats = apply_pre_rerank_scoring(
            working,
            context=context,
            config=self._config,
            weights=self._weights,
        )

        if exhaustive:
            preselected = pre_ranked
        else:
            preselected = pre_ranked[:effective_rerank_limit]

        reranked, reranker_used, warnings = apply_cross_encoder_reranking(
            preselected,
            query_text=context.query_text,
            reranker=self._reranker,
            batch_size=self._batch_size,
            cross_encoder_weight=self._weights.cross_encoder_weight,
        )

        if exhaustive:
            selected = list(reranked)
        else:
            selected = diversify_candidates(
                reranked,
                max_per_page=effective_max_per_page,
                max_per_document=effective_max_per_document,
                final_k=effective_final_k,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return RerankingResult(
            candidates=tuple(selected),
            input_count=input_count,
            preselected_count=len(preselected),
            final_count=len(selected),
            reranker_used=reranker_used,
            exhaustive=exhaustive,
            elapsed_ms=elapsed_ms,
            filename_boosts=filename_stats,
            warnings=warnings,
        )


__all__ = [
    "CrossEncoderLike",
    "FilenameBoostStat",
    "RankingContext",
    "RankingWeights",
    "RerankingEngine",
    "RerankingError",
    "RerankingResult",
    "apply_cross_encoder_reranking",
    "apply_pre_rerank_scoring",
    "apply_rrf_scoring",
    "diversify",
    "diversify_candidates",
    "extract_filename_tokens",
    "normalize_document_name",
    "tier_score_delta",
]
