"""Hybrid retrieval engine for the multi-tenant RAG backend.

The module extracts the retrieval responsibilities that were embedded in the
last ``gui_reflex.py`` and exposes the contract required by
``core.rag_service.RetrievalPort``.

Pipeline:

1. semantic query embedding;
2. Qdrant vector search with a trusted tenant filter;
3. PostgreSQL BM25, exact-phrase and document-scope search;
4. Neo4j entity, formula and relation search;
5. optional graph-neighbour expansion;
6. merge and deduplication of heterogeneous results;
7. PostgreSQL canonical-content enrichment;
8. final tenant/document/page quality gate;
9. return of ``RetrievalCandidate`` objects to ``core.reranking``.

RRF, CrossEncoder scoring and final diversification intentionally do not live
here.  They are implemented in ``core.reranking`` and orchestrated by
``core.rag_service``.

The module has no FastAPI or Reflex dependency and performs no work at import
time.  All model/database resources are obtained from ``ResourceManager`` only
when a query is executed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePath
from typing import Any

from .config import RagSettings, settings
from .models import (
    GraphEntity,
    RagAnswerMode,
    RagIntent,
    RetrievalCandidate,
    RetrievalDebug,
    SourceItem,
)
from .resources import ResourceManager, ResourceNotReadyError, resources
from .tenant import (
    TenantContext,
    TenantContextError,
    bind_tenant_context,
    optional_positive_int,
    qdrant_payload_is_visible,
    record_is_visible,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ERRORS
# =============================================================================
class RetrievalError(RuntimeError):
    """Base error for the retrieval layer."""


class RetrievalConfigurationError(RetrievalError):
    """A required runtime resource or dependency is not configured."""


class RetrievalBackendError(RetrievalError):
    """All available retrieval backends failed for the request."""


class RetrievalProtocolError(RetrievalError):
    """A backend returned malformed or tenant-incoherent data."""


# =============================================================================
# GENERIC NORMALISATION
# =============================================================================
_ALLOWED_CLASSIFICATIONS = frozenset(
    {"public", "internal", "confidential", "restricted"}
)
_UNKNOWN_FILENAMES = frozenset(
    {"", "unknown", "neo4j", "kg", "neo4j knowledge graph"}
)
_FILENAME_EXTENSIONS = r"pdf|md|txt|docx|html|csv|xlsx"

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_\-]+")
_FILENAME_RE = re.compile(
    rf"\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*\.(?:{_FILENAME_EXTENSIONS}))\b",
    flags=re.IGNORECASE,
)

_SEARCH_STOPWORDS = frozenset(
    {
        # Italian grammar / conversational terms.
        "della", "delle", "degli", "dello", "dalla", "dalle", "dagli",
        "nella", "nelle", "negli", "nello", "alla", "alle", "agli",
        "sulla", "sulle", "sugli", "sullo", "questo", "questa", "questi",
        "queste", "quello", "quella", "quelli", "quelle", "sono",
        "presente", "presenti", "quale", "quali", "cosa", "come", "dove",
        "quando", "perché", "perche", "spiega", "spiegami", "riporta",
        "mostra", "mostrami", "dimmi", "elenca", "trova", "cerca", "voglio",
        "vorrei", "riguardo", "inerente", "relativo", "secondo", "basandoti",
        # English grammar / conversational terms.
        "what", "which", "where", "when", "explain", "show", "tell", "list",
        "find", "search", "report", "present", "available", "each", "about",
        "these", "those", "this", "that", "there", "their", "would", "could",
        "should", "please", "according", "regarding", "based", "give",
        # Document meta-language.
        "documento", "documenti", "document", "documents", "file", "fonte",
        "fonti", "source", "sources", "testo", "text", "context", "pagina",
        "pagine", "page", "pages", "paragrafo", "paragraph", "sezione",
        "section", "capitolo", "chapter",
    }
)

_GRAPH_NOISE = frozenset(
    {
        "neo4j", "cypher", "grafo", "grafi", "graph", "graphs", "nodo", "nodi",
        "node", "nodes", "arco", "archi", "edge", "edges", "relazione",
        "relazioni", "relation", "relations", "relationship", "relationships",
        "collegamento", "collegamenti", "link", "links", "connessione",
        "connessioni", "connection", "connections", "percorso", "path",
        "traversamento", "traversal", "tabella", "table", "markdown", "entità",
        "entita", "entity", "entities", "concetto", "concetti", "concept",
        "concepts", "mostra", "trova", "cerca", "elenca", "riporta", "descrivi",
        "spiega", "analizza", "verifica", "interroga", "show", "find", "search",
        "list", "report", "describe", "explain", "analyze", "analyse", "verify",
        "query",
    }
)


def _safe_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _classification(value: Any) -> str:
    normalized = str(value or "internal").strip().lower()
    return normalized if normalized in _ALLOWED_CLASSIFICATIONS else "internal"


def _source_type(value: Any) -> str:
    normalized = str(value or "text").strip().lower()
    aliases = {
        "": "text",
        "testo": "text",
        "math": "formula",
        "equation": "formula",
        "immagine": "image",
        "imagine": "image",
        "visual": "image",
        "screenshot": "image",
        "grafico": "chart",
        "chart_analysis": "chart",
        "diagram": "chart",
        "diagramma": "chart",
        "tabella": "table",
    }
    return aliases.get(normalized, normalized)


def _payload_text(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("text_sem")
        or payload.get("content_semantic")
        or payload.get("content_raw")
        or payload.get("content")
        or payload.get("text")
        or ""
    ).strip()


def _payload_page(payload: Mapping[str, Any]) -> int:
    return max(0, _safe_int(payload.get("page") or payload.get("page_no"), 0))


def _payload_filename(payload: Mapping[str, Any], default: str = "Unknown") -> str:
    return str(
        payload.get("filename")
        or payload.get("source_name")
        or default
    ).strip() or default


def _payload_type(payload: Mapping[str, Any]) -> str:
    return _source_type(payload.get("toon_type") or payload.get("type") or "text")


def _normalize_document_name(value: str) -> str:
    if not value:
        return ""
    name = PurePath(str(value).strip()).name.casefold()
    name = re.sub(rf"\.({_FILENAME_EXTENSIONS})$", "", name)
    name = re.sub(r"[_\-\s]+(?:out|output)$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def _document_matches(source_filename: str, requested_document: str | None) -> bool:
    if not requested_document:
        return True
    source_name = PurePath(str(source_filename or "").strip()).name
    requested_name = PurePath(str(requested_document or "").strip()).name
    if not source_name or source_name.casefold() in _UNKNOWN_FILENAMES:
        return False
    if source_name.casefold() == requested_name.casefold():
        return True
    return bool(
        _normalize_document_name(source_name)
        and _normalize_document_name(source_name)
        == _normalize_document_name(requested_name)
    )


def _search_tokens(query: str) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query or ""):
        clean = raw.strip().strip(".,:;!?()[]{}\"'")
        if not clean:
            continue
        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10
        is_mixed = bool(re.fullmatch(r"[A-Za-z]{1,5}\d{0,3}", clean)) and 2 <= len(clean) <= 10
        is_word = len(clean) > 3
        token = clean.casefold()
        if not (is_acronym or is_mixed or is_word):
            continue
        if token in _SEARCH_STOPWORDS or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return tuple(output)


def _exact_phrases(query: str) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(
        item.strip()
        for item in re.findall(r"[\"“'«]([^\"”'»]+)[\"”'»]", query or "")
        if len(item.strip()) > 2
    )
    values.extend(re.findall(r"\b[A-Z]{2,10}\d{0,3}\b", query or ""))

    q = (query or "").casefold()
    formula_metric = any(
        term in q
        for term in (
            "formula", "formule", "metric", "metriche", "scoring", "score",
            "calcolo", "calculate", "equation", "equazione",
        )
    )
    if formula_metric:
        if any(term in q for term in ("tempo di rilevamento", "detection time", "time to detect", "rilevamento")):
            values.extend(("MTTD", "Mean Time to Detect", "tempo medio di rilevamento"))
        if any(term in q for term in ("tempo di risoluzione", "resolution time", "time to resolution", "tempo di riparazione", "repair time")):
            values.extend(("MTTR", "Mean Time to Resolution", "Mean Time to Repair"))

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return tuple(output)


def _expand_query(query: str, *, formula_mode: bool) -> str:
    if not formula_mode:
        return str(query or "").strip()
    aliases = [p for p in _exact_phrases(query) if p.casefold() not in query.casefold()]
    if not aliases:
        return str(query or "").strip()
    return (str(query or "").strip() + "\n" + " ".join(aliases)).strip()


def _graph_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in _search_tokens(query) if token not in _GRAPH_NOISE)


def _extract_glossary_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []

    for item in re.findall(r"[\"“'«]([^\"”'»]+)[\"”'»]", query or ""):
        clean = item.strip()
        if clean and not _FILENAME_RE.fullmatch(clean):
            terms.append(clean)

    compounds = re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:[-_/][A-Z0-9]{1,})+\b", query or "")
    terms.extend(compounds)

    if not compounds:
        terms.extend(re.findall(r"\b[A-Z]{2,10}\d{0,3}\b", query or ""))

    # Fallback for: "definisci identity governance" / "significato di ...".
    if not terms:
        match = re.search(
            r"(?:definisci|definizione\s+di|significato\s+di|cosa\s+significa|"
            r"cosa\s+vuol\s+dire|define|definition\s+of|meaning\s+of|"
            r"what\s+does)\s+(.+?)(?:[?.!]|$)",
            query or "",
            flags=re.IGNORECASE,
        )
        if match:
            clean = match.group(1).strip(" \t\n\r.,;:!?\"'")
            if clean:
                terms.append(clean)

    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = re.sub(r"\s+", " ", str(term or "")).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return tuple(output[:12])


def _definition_snippet(term: str, text: str, max_chars: int = 900) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Voce trovata, ma il chunk non contiene testo utilizzabile."

    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    term_lower = term.casefold()
    for index, line in enumerate(lines):
        if term_lower in line.casefold():
            snippet = " ".join(lines[index:index + 5]).strip()
            return snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")

    position = raw.casefold().find(term_lower)
    if position >= 0:
        start = max(0, position - 160)
        end = min(len(raw), position + max_chars)
        snippet = re.sub(r"\s+", " ", raw[start:end]).strip()
        return snippet + ("..." if end < len(raw) else "")

    snippet = re.sub(r"\s+", " ", raw[:max_chars]).strip()
    return snippet + ("..." if len(raw) > max_chars else "")


def _usable_content(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 2:
        return False
    replacement_count = text.count("\ufffd") + text.count("□")
    return replacement_count / max(1, len(text)) < 0.15


def _origin_join(*values: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value or "").split(" + "):
            clean = part.strip()
            if clean and clean not in seen:
                seen.add(clean)
                parts.append(clean)
    return " + ".join(parts) or "Unknown"


def _candidate_sort_key(candidate: RetrievalCandidate) -> tuple[float, str, int, int, str]:
    provisional = max(
        float(candidate.score_vec),
        float(candidate.score_bm25),
        float(candidate.score_graph),
        float(candidate.score_base),
    )
    return (
        -provisional,
        candidate.filename.casefold(),
        int(candidate.page),
        int(candidate.page_chunk_index),
        candidate.id,
    )


def _candidate_payload(candidate: RetrievalCandidate) -> dict[str, Any]:
    data = candidate.model_dump(mode="python", exclude={"effective_score"})
    # computed_field may still be present depending on Pydantic minor version.
    data.pop("effective_score", None)
    return data


def _replace_candidate(candidate: RetrievalCandidate, **updates: Any) -> RetrievalCandidate:
    payload = _candidate_payload(candidate)
    payload.update(updates)
    return RetrievalCandidate.model_validate(payload)


def _dedupe_graph_entities(values: Iterable[GraphEntity]) -> list[GraphEntity]:
    output: list[GraphEntity] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in values:
        key = (entity.name.casefold(), entity.type.casefold(), entity.relation.casefold())
        if key in seen:
            continue
        seen.add(key)
        output.append(entity)
    return output


def _merge_candidates(
    existing: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    """Merge two records referring to the same canonical chunk."""

    existing_is_graph = str(existing.tier) == "GRAPH"
    incoming_is_graph = str(incoming.tier) == "GRAPH"

    # Non-GRAPH provenance is canonical when the same chunk also arrives from
    # Neo4j.  This mirrors the final PostgreSQL enrichment of the PoC.
    canonical = incoming if existing_is_graph and not incoming_is_graph else existing
    secondary = existing if canonical is incoming else incoming

    content = canonical.content
    source_type = canonical.type
    if incoming.type == "formula" or existing.type == "formula":
        formula_content = incoming.content if incoming.type == "formula" else existing.content
        if formula_content and formula_content not in content:
            content = (
                content.rstrip()
                + "\n\n--- Formula collegata dal Knowledge Graph ---\n"
                + formula_content.strip()
            ).strip()
        source_type = "formula"
    elif len(secondary.content.strip()) > len(content.strip()) and canonical.filename.casefold() in _UNKNOWN_FILENAMES:
        content = secondary.content

    filename = canonical.filename
    if filename.casefold() in _UNKNOWN_FILENAMES and secondary.filename.casefold() not in _UNKNOWN_FILENAMES:
        filename = secondary.filename

    page = canonical.page or secondary.page
    page_chunk_index = canonical.page_chunk_index or secondary.page_chunk_index
    doc_id = canonical.doc_id or secondary.doc_id
    section_hint = canonical.section_hint or secondary.section_hint
    image_id = canonical.image_id if canonical.image_id is not None else secondary.image_id

    metadata = {**secondary.metadata, **canonical.metadata}
    if secondary.metadata.get("score_doc_scope"):
        metadata["score_doc_scope"] = secondary.metadata["score_doc_scope"]
    if canonical.metadata.get("score_doc_scope"):
        metadata["score_doc_scope"] = canonical.metadata["score_doc_scope"]
    if secondary.metadata.get("exact_phrase") or canonical.metadata.get("exact_phrase"):
        metadata["exact_phrase"] = True

    return _replace_candidate(
        canonical,
        content=content,
        filename=filename,
        page=page,
        page_chunk_index=page_chunk_index,
        doc_id=doc_id,
        type=source_type,
        section_hint=section_hint,
        image_id=image_id,
        origin=_origin_join(existing.origin, incoming.origin),
        metadata=metadata,
        graph_context=_dedupe_graph_entities(
            [*existing.graph_context, *incoming.graph_context]
        ),
        score_base=max(existing.score_base, incoming.score_base),
        score_vec=max(existing.score_vec, incoming.score_vec),
        score_bm25=max(existing.score_bm25, incoming.score_bm25),
        score_graph=max(existing.score_graph, incoming.score_graph),
    )


# =============================================================================
# ENGINE
# =============================================================================
class HybridRetrievalEngine:
    """Tenant-safe Hybrid-RAG retrieval implementation."""

    def __init__(
        self,
        *,
        config: RagSettings = settings,
        resource_manager: ResourceManager = resources,
    ) -> None:
        self._config = config
        self._resources = resource_manager

    async def retrieve_candidates(
        self,
        *,
        query: str,
        intent: RagIntent,
        answer_mode: RagAnswerMode,
        target_document: str | None,
        target_pages: tuple[int, ...],
        wants_evidence: bool,
        graph_relation_mode: bool,
        formula_mode: bool,
        exhaustive_formula_lookup: bool,
        tenant_context: TenantContext,
    ) -> tuple[tuple[RetrievalCandidate, ...], RetrievalDebug]:
        """Execute blocking database/model work in a worker thread.

        ``asyncio.to_thread`` propagates the current ContextVar context.  The
        explicit tenant binding inside ``_retrieve_sync`` remains a defensive
        invariant and also supports direct unit invocation.
        """

        return await asyncio.to_thread(
            self._retrieve_sync,
            query=query,
            intent=intent,
            answer_mode=answer_mode,
            target_document=target_document,
            target_pages=target_pages,
            wants_evidence=wants_evidence,
            graph_relation_mode=graph_relation_mode,
            formula_mode=formula_mode,
            exhaustive_formula_lookup=exhaustive_formula_lookup,
            tenant_context=tenant_context,
        )

    def _retrieve_sync(
        self,
        *,
        query: str,
        intent: RagIntent,
        answer_mode: RagAnswerMode,
        target_document: str | None,
        target_pages: tuple[int, ...],
        wants_evidence: bool,
        graph_relation_mode: bool,
        formula_mode: bool,
        exhaustive_formula_lookup: bool,
        tenant_context: TenantContext,
    ) -> tuple[tuple[RetrievalCandidate, ...], RetrievalDebug]:
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("query non può essere vuota")

        target_pages = tuple(sorted(set(int(page) for page in target_pages or ())))
        if any(page <= 0 for page in target_pages):
            raise ValueError("target_pages accetta soltanto pagine positive")

        debug = RetrievalDebug(
            query=query_text,
            intent=intent,
            answer_mode=answer_mode,
            wants_evidence=wants_evidence,
            default_tiers=tuple(self._config.rag_default_tiers),
            qdrant_candidates=int(self._candidate_limit(query_text, graph_relation_mode)),
            target_document=target_document,
            target_pages=target_pages,
        )

        warnings: list[str] = []
        backend_successes = 0
        backend_attempts = 0
        started = time.perf_counter()
        candidates: dict[str, RetrievalCandidate] = {}
        expanded_query = _expand_query(query_text, formula_mode=formula_mode)

        with bind_tenant_context(tenant_context):
            # -----------------------------------------------------------------
            # 1) Qdrant semantic retrieval
            # -----------------------------------------------------------------
            backend_attempts += 1
            qdrant_started = time.perf_counter()
            try:
                vector_hits = self._search_qdrant(
                    expanded_query,
                    limit=debug.qdrant_candidates,
                    tenant_context=tenant_context,
                )
                backend_successes += 1
                debug.qdrant_hits = len(vector_hits)
                for candidate in vector_hits:
                    self._put_candidate(candidates, candidate)
            except Exception as exc:
                logger.exception("Qdrant retrieval failed")
                warnings.append(f"Qdrant non disponibile: {type(exc).__name__}.")
            debug.record_timing("embed_qdrant", time.perf_counter() - qdrant_started)

            # -----------------------------------------------------------------
            # 2) PostgreSQL hybrid lexical retrieval
            # -----------------------------------------------------------------
            if self._config.pg_enrich_enabled:
                backend_attempts += 1
                pg_started = time.perf_counter()
                try:
                    bm25_hits = self._search_pg_bm25(
                        expanded_query,
                        limit=60,
                        tenant_context=tenant_context,
                    )
                    exact_hits = self._search_pg_exact_phrases(
                        query_text,
                        limit=40,
                        tenant_context=tenant_context,
                    )
                    doc_hits: list[RetrievalCandidate] = []
                    if target_document:
                        doc_hits = self._search_pg_document_scope(
                            target_document,
                            query_text,
                            limit=250 if exhaustive_formula_lookup else 100,
                            tenant_context=tenant_context,
                        )

                    # Generic acronym injection from glossary chunks.
                    glossary_injected: list[RetrievalCandidate] = []
                    for acronym in dict.fromkeys(
                        re.findall(r"\b[A-Z]{2,10}\d{0,3}\b", query_text)
                    ):
                        for hit in self._search_pg_glossary_term(
                            acronym,
                            aliases=(acronym,),
                            limit=2,
                            tenant_context=tenant_context,
                        ):
                            glossary_injected.append(
                                _replace_candidate(
                                    hit,
                                    score_bm25=max(hit.score_bm25, 3.0),
                                    origin=_origin_join(
                                        hit.origin,
                                        "PostgresGlossaryInjectDynamic",
                                    ),
                                    metadata={**hit.metadata, "exact_phrase": True},
                                )
                            )

                    backend_successes += 1
                    debug.postgres_bm25_hits = len(bm25_hits)
                    debug.postgres_exact_phrase_hits = len(exact_hits) + len(glossary_injected)
                    for candidate in (*doc_hits, *exact_hits, *glossary_injected, *bm25_hits):
                        self._put_candidate(candidates, candidate)
                except Exception as exc:
                    logger.exception("PostgreSQL retrieval failed")
                    warnings.append(f"PostgreSQL retrieval degradato: {type(exc).__name__}.")
                debug.record_timing("postgres_search", time.perf_counter() - pg_started)

            # -----------------------------------------------------------------
            # 3) Neo4j direct retrieval and explicit graph relations
            # -----------------------------------------------------------------
            driver = None
            if self._config.neo4j_enabled:
                try:
                    driver = self._resources.get_neo4j_driver(required=False)
                except ResourceNotReadyError:
                    driver = None

            relation_candidates: list[RetrievalCandidate] = []
            if driver is not None:
                backend_attempts += 1
                graph_started = time.perf_counter()
                try:
                    entity_hits = self._search_neo4j_entities(
                        expanded_query,
                        limit=30,
                        tenant_context=tenant_context,
                        driver=driver,
                    )
                    formula_hits = (
                        self._search_neo4j_formulas(
                            query_text,
                            limit=int(self._config.graph_max_formulas) * 4,
                            tenant_context=tenant_context,
                            driver=driver,
                        )
                        if formula_mode
                        else []
                    )
                    if target_document:
                        entity_hits = [
                            item for item in entity_hits
                            if _document_matches(item.filename, target_document)
                        ]
                        formula_hits = [
                            item for item in formula_hits
                            if _document_matches(item.filename, target_document)
                        ]

                    if graph_relation_mode:
                        relation_candidates = self._search_neo4j_relations(
                            query_text,
                            limit=60,
                            target_document=target_document,
                            tenant_context=tenant_context,
                            driver=driver,
                        )

                    backend_successes += 1
                    debug.neo4j_direct_hits = (
                        len(entity_hits) + len(formula_hits) + len(relation_candidates)
                    )
                    for candidate in (*entity_hits, *formula_hits, *relation_candidates):
                        self._put_candidate(candidates, candidate)
                except Exception as exc:
                    logger.exception("Neo4j direct retrieval failed")
                    warnings.append(f"Neo4j direct retrieval degradato: {type(exc).__name__}.")
                debug.record_timing("neo4j_direct", time.perf_counter() - graph_started)

            # -----------------------------------------------------------------
            # 4) Graph expansion: Chunk -> Entity -> neighbouring Chunk
            # -----------------------------------------------------------------
            if (
                self._config.graph_expand_enabled
                and driver is not None
                and candidates
            ):
                expand_started = time.perf_counter()
                try:
                    seed_ids = [
                        candidate.id
                        for candidate in sorted(candidates.values(), key=_candidate_sort_key)
                        if candidate.type != "graph_relations"
                    ][:10]
                    neighbour_ids = self._get_neighbor_chunk_ids(
                        seed_ids,
                        limit=int(self._config.graph_max_neighbor_chunks),
                        tenant_context=tenant_context,
                        driver=driver,
                    )
                    expanded = self._retrieve_qdrant_points_by_ids(
                        neighbour_ids,
                        tenant_context=tenant_context,
                    )
                    for candidate in expanded:
                        candidate = _replace_candidate(
                            candidate,
                            score_graph=max(candidate.score_graph, 1.0),
                            origin=_origin_join(candidate.origin, "Neo4jExpansion"),
                        )
                        self._put_candidate(candidates, candidate)
                    debug.neo4j_expanded_hits = len(expanded)
                    debug.graph_expand_used = bool(expanded)
                except Exception as exc:
                    logger.exception("Neo4j graph expansion failed")
                    warnings.append(f"Graph expansion non completata: {type(exc).__name__}.")
                debug.record_timing("neo4j_expand", time.perf_counter() - expand_started)

            if backend_attempts and backend_successes == 0:
                raise RetrievalBackendError(
                    "Nessun backend di retrieval ha completato correttamente la richiesta"
                )

            # -----------------------------------------------------------------
            # 5) Canonical PostgreSQL enrichment before CrossEncoder reranking
            # -----------------------------------------------------------------
            if candidates and self._config.pg_enrich_enabled:
                enrich_started = time.perf_counter()
                try:
                    canonical_rows = self._fetch_pg_chunks_by_uuid(
                        [
                            candidate.id
                            for candidate in candidates.values()
                            if candidate.type != "graph_relations"
                        ],
                        tenant_context=tenant_context,
                    )
                    for candidate_id, pg_row in canonical_rows.items():
                        current = candidates.get(candidate_id)
                        if current is None:
                            continue
                        candidates[candidate_id] = self._enrich_candidate_from_pg(
                            current,
                            pg_row,
                            formula_mode=formula_mode,
                        )
                except Exception as exc:
                    logger.exception("PostgreSQL canonical enrichment failed")
                    warnings.append(f"Arricchimento PostgreSQL degradato: {type(exc).__name__}.")
                debug.record_timing("postgres_enrich", time.perf_counter() - enrich_started)

            # Attach graph entities/formulas only after canonical provenance is known.
            if candidates and driver is not None:
                graph_context_started = time.perf_counter()
                real_chunk_ids = [
                    candidate.id
                    for candidate in candidates.values()
                    if candidate.type != "graph_relations"
                ]
                try:
                    entity_map = self._get_graph_entities(
                        real_chunk_ids,
                        tenant_context=tenant_context,
                        driver=driver,
                    )
                    for candidate_id, entities in entity_map.items():
                        current = candidates.get(candidate_id)
                        if current is not None:
                            candidates[candidate_id] = _replace_candidate(
                                current,
                                graph_context=_dedupe_graph_entities(
                                    [*current.graph_context, *entities]
                                ),
                            )

                    if formula_mode:
                        formulas = self._get_formulas_for_chunks(
                            real_chunk_ids,
                            limit_per_chunk=int(self._config.graph_max_formulas),
                            tenant_context=tenant_context,
                            driver=driver,
                        )
                        for candidate_id, values in formulas.items():
                            current = candidates.get(candidate_id)
                            if current is None or not values:
                                continue
                            appendix = "\n".join(values)
                            content = current.content
                            if appendix not in content:
                                content = (
                                    content.rstrip()
                                    + "\n\n--- Formule collegate dal Knowledge Graph ---\n"
                                    + appendix
                                ).strip()
                            candidates[candidate_id] = _replace_candidate(
                                current,
                                content=content,
                                type="formula",
                                origin=_origin_join(current.origin, "Neo4jFormulaLink"),
                            )
                except Exception as exc:
                    logger.exception("Neo4j context enrichment failed")
                    warnings.append(f"Contesto Neo4j non completato: {type(exc).__name__}.")
                debug.record_timing("neo4j_context", time.perf_counter() - graph_context_started)

        # ---------------------------------------------------------------------
        # 6) Final deterministic quality / tenant / scope gate
        # ---------------------------------------------------------------------
        final_candidates: list[RetrievalCandidate] = []
        dropped_tenant = 0
        dropped_content = 0
        dropped_document = 0
        dropped_page = 0

        for candidate in candidates.values():
            if not record_is_visible(
                candidate,
                context=tenant_context,
                allow_graph_tier=True,
                allow_user_tier=False,
            ):
                dropped_tenant += 1
                continue
            if not _usable_content(candidate.content):
                dropped_content += 1
                continue
            if target_document and not _document_matches(candidate.filename, target_document):
                dropped_document += 1
                continue
            if target_pages and candidate.page not in set(target_pages):
                dropped_page += 1
                continue
            final_candidates.append(candidate)

        if dropped_tenant:
            # A backend returning out-of-scope records is a security anomaly.
            # The service will apply another guard, but retrieval reports it.
            warnings.append(
                f"Tenant guard retrieval: scartati {dropped_tenant} candidati fuori perimetro."
            )
        if dropped_content:
            warnings.append(f"Scartati {dropped_content} candidati senza contenuto utilizzabile.")
        if target_document and dropped_document:
            warnings.append(
                f"Document scope: scartati {dropped_document} candidati di altri documenti."
            )
        if target_pages and dropped_page:
            warnings.append(
                f"Page scope: scartati {dropped_page} candidati fuori dalle pagine richieste."
            )

        final_candidates.sort(key=_candidate_sort_key)

        # Bound the material sent to the reranker.  Exhaustive formula lookup is
        # allowed a larger document-scoped cap but remains finite.
        cap = 500 if exhaustive_formula_lookup and target_document else max(
            200,
            int(self._config.qdrant_candidates) + 120,
        )
        if len(final_candidates) > cap:
            final_candidates = final_candidates[:cap]
            warnings.append(f"Candidate cap applicato: mantenuti i primi {cap} risultati.")

        debug.kept_after_quality_filters = len(final_candidates)
        debug.tier_counts = dict(Counter(str(item.tier) for item in final_candidates))
        debug.warnings = tuple(dict.fromkeys(warnings))
        debug.record_timing("total", time.perf_counter() - started)

        return tuple(final_candidates), debug

    async def lookup_glossary(
        self,
        *,
        query: str,
        tenant_context: TenantContext,
    ) -> tuple[str, tuple[SourceItem, ...], str] | None:
        """Atomic deterministic glossary lookup used by ``RagService``."""

        return await asyncio.to_thread(
            self._lookup_glossary_sync,
            query=query,
            tenant_context=tenant_context,
        )

    def _lookup_glossary_sync(
        self,
        *,
        query: str,
        tenant_context: TenantContext,
    ) -> tuple[str, tuple[SourceItem, ...], str] | None:
        if not self._config.pg_enrich_enabled:
            return None

        terms = _extract_glossary_terms(query)
        if not terms:
            return None

        answer_lines: list[str] = []
        evidence_lines: list[str] = []
        sources: list[SourceItem] = []
        seen_sources: set[str] = set()

        with bind_tenant_context(tenant_context):
            try:
                for term in terms:
                    hits = self._search_pg_glossary_term(
                        term,
                        aliases=(term,),
                        limit=5,
                        tenant_context=tenant_context,
                    )
                    if not hits:
                        answer_lines.append(
                            f"- **{term}**: voce non trovata nel glossario recuperato."
                        )
                        evidence_lines.append(
                            f"- **{term}**: nessun chunk di glossario recuperato."
                        )
                        continue

                    best = hits[0]
                    answer_lines.append(
                        f"- **{term}**: {_definition_snippet(term, best.content)}"
                    )
                    page_label = best.page if best.page > 0 else "N/D"
                    evidence_lines.append(
                        f"- **{term}**: recuperato da `{best.filename}`, pag. {page_label}."
                    )

                    if best.id not in seen_sources:
                        seen_sources.add(best.id)
                        sources.append(
                            SourceItem(
                                id=best.id,
                                content=best.content[:5000],
                                filename=best.filename,
                                page=best.page,
                                page_chunk_index=best.page_chunk_index,
                                doc_id=best.doc_id,
                                type=best.type,
                                score=best.score_bm25 or best.score_base,
                                graph_context=list(best.graph_context),
                                section_hint=best.section_hint,
                                image_id=best.image_id,
                                tier=best.tier,
                                scope=best.scope,
                                organization_id=best.organization_id,
                                status=best.status,
                                ingestion_run_id=best.ingestion_run_id,
                                corpus_version=best.corpus_version,
                                classification=best.classification,
                                embedding_model=best.embedding_model,
                                request_id=tenant_context.request_id,
                                pg_ingestion_ts=str(best.metadata.get("pg_ingestion_ts") or ""),
                                pg_source_name=str(best.metadata.get("source_name") or ""),
                                pg_source_type=str(best.metadata.get("source_type") or ""),
                                pg_log_id=_safe_int(best.metadata.get("log_id"), 0),
                                pg_chunk_id=_safe_int(best.metadata.get("chunk_index"), 0),
                                pg_page_chunk_index=_safe_int(
                                    best.metadata.get("page_chunk_index"), 0
                                ),
                                pg_toon_type=str(best.metadata.get("toon_type") or ""),
                                db_origin=best.origin,
                            )
                        )
            except (ResourceNotReadyError, TenantContextError):
                return None
            except Exception:
                logger.exception("Glossary direct lookup failed")
                return None

        used_files = list(dict.fromkeys(source.filename for source in sources))
        answer = (
            "**A) Risposta**\n\n"
            + "\n".join(answer_lines)
            + "\n\n**B) Evidenze**\n\n"
            + "\n".join(evidence_lines)
            + "\n\n**C) Limiti / Conflitti**\n\n"
            + "- Risposta generata tramite lookup deterministico di glossario.\n"
            + "- Una voce è dichiarata assente soltanto dopo il lookup atomico nei chunk di glossario tenant-visible.\n\n"
            + "**D) Fonti**\n\n"
            + (
                "\n".join(f"- {filename}" for filename in used_files)
                if used_files
                else "- Nessuna fonte di glossario recuperata."
            )
        )
        audit_markdown = (
            "### Audit glossary deterministic mode\n"
            f"- Termini richiesti: `{', '.join(terms)}`\n"
            f"- Fonti recuperate: **{len(sources)}**\n"
            "- Retrieval generativo bypassato per il solo lookup definitorio."
        )
        return answer, tuple(sources), audit_markdown

    # =========================================================================
    # QDRANT
    # =========================================================================
    def _candidate_limit(self, query: str, graph_relation_mode: bool) -> int:
        complex_terms = (
            "confronta", "confronto", "mappatura", "crosswalk", "documenti",
            "fonti", "assessment", "audit", "step-by-step", "completo",
            "compare", "comparison", "mapping", "documents", "sources",
            "complete", "comprehensive", "end-to-end",
        )
        complex_query = len(query) > 300 or sum(
            1 for term in complex_terms if term in query.casefold()
        ) >= 2
        return max(int(self._config.qdrant_candidates), 140) if (complex_query or graph_relation_mode) else int(self._config.qdrant_candidates)

    def _build_qdrant_filter(self, context: TenantContext) -> Any:
        try:
            from qdrant_client import models
        except ImportError:
            # Import-safe fallback used by isolated unit tests.  A real Qdrant
            # client cannot exist without qdrant-client installed.
            branches: list[dict[str, Any]] = []
            if "GLOBAL" in context.allowed_scopes:
                branches.append(
                    {
                        "must": [
                            {"key": "scope", "match": {"value": "GLOBAL"}},
                            {"key": "tier", "match": {"value": "A"}},
                        ]
                    }
                )
            if "ACCOUNT" in context.allowed_scopes:
                branches.append(
                    {
                        "must": [
                            {"key": "scope", "match": {"value": "ACCOUNT"}},
                            {"key": "organization_id", "match": {"value": context.organization_id}},
                            {"key": "tier", "match": {"any": ["B", "C"]}},
                        ]
                    }
                )
            return {
                "must": [{"key": "status", "match": {"value": "active"}}],
                "should": branches,
            }

        branches: list[Any] = []
        if "GLOBAL" in context.allowed_scopes:
            branches.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="scope", match=models.MatchValue(value="GLOBAL")
                        ),
                        models.FieldCondition(
                            key="tier", match=models.MatchValue(value="A")
                        ),
                    ]
                )
            )
        if "ACCOUNT" in context.allowed_scopes:
            branches.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="scope", match=models.MatchValue(value="ACCOUNT")
                        ),
                        models.FieldCondition(
                            key="organization_id",
                            match=models.MatchValue(value=context.organization_id),
                        ),
                        models.FieldCondition(
                            key="tier", match=models.MatchAny(any=["B", "C"])
                        ),
                    ]
                )
            )
        if not branches:
            raise RetrievalConfigurationError(
                "TenantContext senza scope autorizzati per Qdrant"
            )
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                )
            ],
            should=branches,
        )

    def _search_qdrant(
        self,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        try:
            embedder = self._resources.get_embedder()
            client = self._resources.get_qdrant_client()
        except ResourceNotReadyError as exc:
            raise RetrievalConfigurationError(
                "Risorse embedding/Qdrant non inizializzate"
            ) from exc

        vector = embedder.encode(query, normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vector = list(vector)
        if not vector or any(not math.isfinite(float(value)) for value in vector):
            raise RetrievalProtocolError("Embedding query non valido")

        tenant_filter = self._build_qdrant_filter(tenant_context)
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self._config.qdrant_collection,
                query=vector,
                query_filter=tenant_filter,
                limit=int(limit),
                with_payload=True,
            )
            hits = list(getattr(response, "points", response) or ())
        else:
            hits = list(
                client.search(
                    collection_name=self._config.qdrant_collection,
                    query_vector=vector,
                    query_filter=tenant_filter,
                    limit=int(limit),
                    with_payload=True,
                )
                or ()
            )

        output: list[RetrievalCandidate] = []
        for hit in hits:
            payload = _safe_json_mapping(getattr(hit, "payload", None))
            if not qdrant_payload_is_visible(payload, context=tenant_context):
                logger.warning(
                    "Qdrant tenant guard rejected point=%s request_id=%s",
                    getattr(hit, "id", ""),
                    tenant_context.request_id,
                )
                continue
            candidate = self._candidate_from_payload(
                candidate_id=str(getattr(hit, "id", "")),
                payload=payload,
                origin="Qdrant",
                score_vec=_safe_float(getattr(hit, "score", 0.0)),
            )
            if candidate is not None:
                output.append(candidate)
        return output

    def _retrieve_qdrant_points_by_ids(
        self,
        ids: Sequence[str],
        *,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        unique_ids = list(dict.fromkeys(str(value) for value in ids if str(value).strip()))
        if not unique_ids:
            return []
        client = self._resources.get_qdrant_client()
        points = client.retrieve(
            collection_name=self._config.qdrant_collection,
            ids=unique_ids,
            with_payload=True,
        )
        output: list[RetrievalCandidate] = []
        for point in points or ():
            payload = _safe_json_mapping(getattr(point, "payload", None))
            if not qdrant_payload_is_visible(payload, context=tenant_context):
                continue
            candidate = self._candidate_from_payload(
                candidate_id=str(getattr(point, "id", "")),
                payload=payload,
                origin="QdrantGraphExpansion",
                score_graph=1.0,
            )
            if candidate is not None:
                output.append(candidate)
        return output

    def _candidate_from_payload(
        self,
        *,
        candidate_id: str,
        payload: Mapping[str, Any],
        origin: str,
        score_vec: float = 0.0,
        score_graph: float = 0.0,
    ) -> RetrievalCandidate | None:
        content = _payload_text(payload)
        if not candidate_id or not _usable_content(content):
            return None
        try:
            return RetrievalCandidate(
                id=candidate_id,
                content=content,
                filename=_payload_filename(payload),
                page=_payload_page(payload),
                page_chunk_index=max(0, _safe_int(payload.get("page_chunk_index"), 0)),
                doc_id=str(payload.get("doc_id") or ""),
                type=_payload_type(payload),
                tier=str(payload.get("tier") or ""),
                scope=str(payload.get("scope") or ""),
                organization_id=optional_positive_int(payload.get("organization_id")),
                status=str(payload.get("status") or ""),
                ingestion_run_id=str(payload.get("ingestion_run_id") or ""),
                corpus_version=str(payload.get("corpus_version") or self._config.corpus_version),
                classification=_classification(payload.get("classification")),
                embedding_model=str(payload.get("embedding_model") or ""),
                section_hint=str(payload.get("section_hint") or ""),
                image_id=optional_positive_int(payload.get("image_id")),
                origin=origin,
                metadata=dict(payload),
                score_base=score_vec,
                score_vec=score_vec,
                score_graph=score_graph,
            )
        except (ValueError, TenantContextError) as exc:
            logger.warning("Invalid Qdrant candidate %s: %s", candidate_id, exc)
            return None

    # =========================================================================
    # POSTGRESQL
    # =========================================================================
    def _search_pg_bm25(
        self,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        tokens = _search_tokens(query)
        if not tokens:
            return []
        pg_query = " OR ".join(tokens)
        sql = """
            WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq)
            SELECT
                d.chunk_uuid::text,
                d.content_raw,
                d.content_semantic,
                d.metadata_json,
                d.scope,
                d.organization_id,
                d.tier,
                d.status,
                d.ingestion_run_id,
                d.corpus_version,
                d.classification,
                d.embedding_model,
                d.ingestion_ts,
                ts_rank_cd(
                    to_tsvector(
                        'simple',
                        COALESCE(d.content_semantic, '') || ' ' ||
                        COALESCE(d.content_raw, '') || ' ' ||
                        COALESCE(d.metadata_json::text, '')
                    ),
                    q.tsq
                ) AS rank
            FROM public.document_chunks d, q
            WHERE d.status = 'active'
              AND to_tsvector(
                    'simple',
                    COALESCE(d.content_semantic, '') || ' ' ||
                    COALESCE(d.content_raw, '') || ' ' ||
                    COALESCE(d.metadata_json::text, '')
                  ) @@ q.tsq
              AND (
                    (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
                    OR
                    (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
                  )
            ORDER BY rank DESC
            LIMIT %s
        """
        with self._resources.postgres_connection(context=tenant_context) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (pg_query, tenant_context.organization_id, int(limit)))
                rows = cursor.fetchall()
        return [
            candidate
            for row in rows
            if (candidate := self._candidate_from_pg_row(row, origin="PostgresBM25"))
            is not None
        ]

    def _search_pg_exact_phrases(
        self,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        phrases = _exact_phrases(query)[:12]
        if not phrases:
            return []

        per_phrase = max(3, int(limit) // len(phrases))
        found: dict[str, RetrievalCandidate] = {}
        sql_template = """
            SELECT
                chunk_uuid::text,
                content_raw,
                content_semantic,
                metadata_json,
                scope,
                organization_id,
                tier,
                status,
                ingestion_run_id,
                corpus_version,
                classification,
                embedding_model,
                ingestion_ts,
                2.0 AS rank
            FROM public.document_chunks
            WHERE status = 'active'
              AND {condition}
              AND (
                    (scope = 'GLOBAL' AND organization_id IS NULL AND tier = 'A')
                    OR
                    (scope = 'ACCOUNT' AND organization_id = %s AND tier IN ('B', 'C'))
                  )
            ORDER BY ingestion_ts DESC
            LIMIT %s
        """

        with self._resources.postgres_connection(context=tenant_context) as conn:
            with conn.cursor() as cursor:
                for phrase in phrases:
                    condition, parameters = self._pg_term_condition(phrase)
                    cursor.execute(
                        sql_template.format(condition=condition),
                        (*parameters, tenant_context.organization_id, per_phrase),
                    )
                    for row in cursor.fetchall():
                        candidate = self._candidate_from_pg_row(
                            row,
                            origin="PostgresExactPhrase",
                            metadata_flags={"exact_phrase": True},
                        )
                        if candidate is None:
                            continue
                        current = found.get(candidate.id)
                        if current is None:
                            found[candidate.id] = candidate
                        else:
                            found[candidate.id] = _replace_candidate(
                                _merge_candidates(current, candidate),
                                score_bm25=current.score_bm25 + 1.0,
                            )
        return sorted(found.values(), key=_candidate_sort_key)[:limit]

    def _search_pg_document_scope(
        self,
        requested_document: str,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        wanted = _normalize_document_name(requested_document)
        if not wanted:
            return []
        sql = """
            WITH q AS (SELECT plainto_tsquery('simple', %s) AS tsq),
            visible AS (
                SELECT
                    d.*,
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                lower(coalesce(
                                    d.metadata_json->>'filename',
                                    d.metadata_json->>'source_name',
                                    ''
                                )),
                                '\\.(pdf|md|txt|docx|html|csv|xlsx)$', '', 'g'
                            ),
                            '[_\\-\\s]+(out|output)$', '', 'g'
                        ),
                        '[^a-z0-9]+', '', 'g'
                    ) AS filename_norm,
                    ts_rank_cd(
                        to_tsvector(
                            'simple',
                            coalesce(d.content_semantic, '') || ' ' ||
                            coalesce(d.content_raw, '') || ' ' ||
                            coalesce(d.metadata_json::text, '')
                        ),
                        q.tsq
                    ) AS rank
                FROM public.document_chunks d, q
                WHERE d.status = 'active'
                  AND (
                        (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
                        OR
                        (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
                      )
            ),
            ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY chunk_uuid, scope, organization_id
                    ORDER BY ingestion_ts DESC
                ) AS rn
                FROM visible
            )
            SELECT
                chunk_uuid::text,
                content_raw,
                content_semantic,
                metadata_json,
                scope,
                organization_id,
                tier,
                status,
                ingestion_run_id,
                corpus_version,
                classification,
                embedding_model,
                ingestion_ts,
                rank
            FROM ranked
            WHERE rn = 1
              AND filename_norm = %s
            ORDER BY rank DESC, ingestion_ts DESC
            LIMIT %s
        """
        with self._resources.postgres_connection(context=tenant_context) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (query, tenant_context.organization_id, wanted, int(limit)),
                )
                rows = cursor.fetchall()
        output: list[RetrievalCandidate] = []
        for row in rows:
            candidate = self._candidate_from_pg_row(
                row,
                origin="PostgresDocScope",
                metadata_flags={"score_doc_scope": 1.0},
            )
            if candidate is not None:
                output.append(candidate)
        return output

    def _search_pg_glossary_term(
        self,
        canonical_term: str,
        *,
        aliases: Sequence[str],
        limit: int,
        tenant_context: TenantContext,
    ) -> list[RetrievalCandidate]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for alias in aliases:
            condition, values = self._pg_term_condition(alias)
            clauses.append(condition)
            parameters.extend(values)
        if not clauses:
            return []

        sql = f"""
            SELECT
                chunk_uuid::text,
                content_raw,
                content_semantic,
                metadata_json,
                scope,
                organization_id,
                tier,
                status,
                ingestion_run_id,
                corpus_version,
                classification,
                embedding_model,
                ingestion_ts,
                3.0 AS rank
            FROM public.document_chunks
            WHERE status = 'active'
              AND (
                    lower(coalesce(metadata_json->>'filename', '')) LIKE %s
                    OR lower(coalesce(metadata_json->>'source_name', '')) LIKE %s
                    OR lower(coalesce(metadata_json::text, '')) LIKE %s
                  )
              AND ({' OR '.join(clauses)})
              AND (
                    (scope = 'GLOBAL' AND organization_id IS NULL AND tier = 'A')
                    OR
                    (scope = 'ACCOUNT' AND organization_id = %s AND tier IN ('B', 'C'))
                  )
            ORDER BY ingestion_ts DESC
            LIMIT %s
        """
        params = ["%glossar%", "%glossar%", "%glossar%", *parameters,
                  tenant_context.organization_id, int(limit)]
        with self._resources.postgres_connection(context=tenant_context) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        return [
            candidate
            for row in rows
            if (candidate := self._candidate_from_pg_row(
                row,
                origin="PostgresGlossaryTerm",
                metadata_flags={"glossary_term": canonical_term},
            )) is not None
        ]

    @staticmethod
    def _pg_term_condition(alias: str) -> tuple[str, tuple[Any, ...]]:
        clean = str(alias or "").strip()
        if not clean:
            return "FALSE", ()
        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10
        if is_acronym:
            pattern = r"(^|[^A-Za-z0-9])" + re.escape(clean) + r"([^A-Za-z0-9]|$)"
            return (
                """(
                    coalesce(content_semantic, '') ~* %s OR
                    coalesce(content_raw, '') ~* %s OR
                    coalesce(metadata_json::text, '') ~* %s
                )""",
                (pattern, pattern, pattern),
            )
        like = f"%{clean.casefold()}%"
        return (
            """(
                lower(coalesce(content_semantic, '')) LIKE %s OR
                lower(coalesce(content_raw, '')) LIKE %s OR
                lower(coalesce(metadata_json::text, '')) LIKE %s
            )""",
            (like, like, like),
        )

    def _candidate_from_pg_row(
        self,
        row: Sequence[Any],
        *,
        origin: str,
        metadata_flags: Mapping[str, Any] | None = None,
    ) -> RetrievalCandidate | None:
        if len(row) < 14:
            raise RetrievalProtocolError(
                f"Riga PostgreSQL incompleta per {origin}: colonne={len(row)}"
            )
        (
            chunk_uuid,
            raw,
            semantic,
            metadata_raw,
            scope,
            organization_id,
            tier,
            status,
            ingestion_run_id,
            corpus_version,
            classification,
            embedding_model,
            ingestion_ts,
            rank,
            *_rest,
        ) = row
        metadata = _safe_json_mapping(metadata_raw)
        metadata.update(metadata_flags or {})
        if ingestion_ts is not None:
            metadata["pg_ingestion_ts"] = (
                ingestion_ts.isoformat()
                if hasattr(ingestion_ts, "isoformat")
                else str(ingestion_ts)
            )

        content = (
            str(raw or semantic or "")
            if self._config.pg_prefer_raw
            else str(semantic or raw or "")
        ).strip()
        if not chunk_uuid or not _usable_content(content):
            return None
        try:
            return RetrievalCandidate(
                id=str(chunk_uuid),
                content=content,
                filename=_payload_filename(metadata),
                page=_payload_page(metadata),
                page_chunk_index=max(0, _safe_int(metadata.get("page_chunk_index"), 0)),
                doc_id=str(metadata.get("doc_id") or ""),
                type=_payload_type(metadata),
                tier=str(tier or metadata.get("tier") or ""),
                scope=str(scope or metadata.get("scope") or ""),
                organization_id=optional_positive_int(
                    organization_id
                    if organization_id is not None
                    else metadata.get("organization_id")
                ),
                status=str(status or metadata.get("status") or ""),
                ingestion_run_id=str(
                    ingestion_run_id or metadata.get("ingestion_run_id") or ""
                ),
                corpus_version=str(
                    corpus_version or metadata.get("corpus_version") or self._config.corpus_version
                ),
                classification=_classification(
                    classification or metadata.get("classification")
                ),
                embedding_model=str(
                    embedding_model or metadata.get("embedding_model") or ""
                ),
                section_hint=str(metadata.get("section_hint") or ""),
                image_id=optional_positive_int(metadata.get("image_id")),
                origin=origin,
                metadata=metadata,
                score_base=_safe_float(rank),
                score_bm25=_safe_float(rank),
            )
        except (ValueError, TenantContextError) as exc:
            logger.warning("Invalid PostgreSQL candidate %s: %s", chunk_uuid, exc)
            return None

    def _fetch_pg_chunks_by_uuid(
        self,
        ids: Sequence[str],
        *,
        tenant_context: TenantContext,
    ) -> dict[str, dict[str, Any]]:
        unique_ids = list(dict.fromkeys(str(value) for value in ids if str(value).strip()))
        if not unique_ids:
            return {}
        sql = """
            WITH visible AS (
                SELECT d.*
                FROM public.document_chunks d
                WHERE d.chunk_uuid::text = ANY(%s)
                  AND d.status = 'active'
                  AND (
                        (d.scope = 'GLOBAL' AND d.organization_id IS NULL AND d.tier = 'A')
                        OR
                        (d.scope = 'ACCOUNT' AND d.organization_id = %s AND d.tier IN ('B', 'C'))
                      )
            ),
            ranked AS (
                SELECT
                    d.*,
                    row_number() OVER (
                        PARTITION BY d.chunk_uuid, d.scope, d.organization_id
                        ORDER BY d.ingestion_ts DESC
                    ) AS rn
                FROM visible d
            )
            SELECT
                chunk_uuid::text,
                content_raw,
                content_semantic,
                metadata_json,
                ingestion_ts,
                scope,
                organization_id,
                tier,
                status,
                ingestion_run_id,
                corpus_version,
                classification,
                embedding_model
            FROM ranked
            WHERE rn = 1
        """
        with self._resources.postgres_connection(context=tenant_context) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (unique_ids, tenant_context.organization_id))
                rows = cursor.fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            (
                chunk_uuid,
                raw,
                semantic,
                metadata,
                ingestion_ts,
                scope,
                organization_id,
                tier,
                status,
                ingestion_run_id,
                corpus_version,
                classification,
                embedding_model,
            ) = row
            result[str(chunk_uuid)] = {
                "content_raw": str(raw or ""),
                "content_semantic": str(semantic or ""),
                "metadata": _safe_json_mapping(metadata),
                "ingestion_ts": (
                    ingestion_ts.isoformat()
                    if hasattr(ingestion_ts, "isoformat")
                    else str(ingestion_ts or "")
                ),
                "scope": scope,
                "organization_id": organization_id,
                "tier": tier,
                "status": status,
                "ingestion_run_id": ingestion_run_id,
                "corpus_version": corpus_version,
                "classification": classification,
                "embedding_model": embedding_model,
            }
        return result

    def _enrich_candidate_from_pg(
        self,
        candidate: RetrievalCandidate,
        row: Mapping[str, Any],
        *,
        formula_mode: bool,
    ) -> RetrievalCandidate:
        metadata = _safe_json_mapping(row.get("metadata"))
        raw = str(row.get("content_raw") or "")
        semantic = str(row.get("content_semantic") or "")
        if formula_mode or self._config.pg_prefer_raw:
            content = raw or semantic or candidate.content
        else:
            content = semantic or raw or candidate.content

        filename = candidate.filename
        pg_filename = _payload_filename(metadata, filename)
        if filename.casefold() in _UNKNOWN_FILENAMES:
            filename = pg_filename

        source_type = candidate.type
        pg_type = _payload_type(metadata)
        if source_type in {"", "graph"}:
            source_type = pg_type

        merged_metadata = {
            **candidate.metadata,
            **metadata,
            "pg_ingestion_ts": str(row.get("ingestion_ts") or ""),
        }
        return _replace_candidate(
            candidate,
            content=content,
            filename=filename,
            page=candidate.page or _payload_page(metadata),
            page_chunk_index=(
                candidate.page_chunk_index
                or max(0, _safe_int(metadata.get("page_chunk_index"), 0))
            ),
            doc_id=candidate.doc_id or str(metadata.get("doc_id") or ""),
            type=source_type,
            tier=str(row.get("tier") or metadata.get("tier") or candidate.tier),
            source_tier=str(candidate.tier) if str(candidate.tier) == "GRAPH" else candidate.source_tier,
            scope=str(row.get("scope") or metadata.get("scope") or candidate.scope),
            organization_id=optional_positive_int(
                row.get("organization_id")
                if row.get("organization_id") is not None
                else metadata.get("organization_id", candidate.organization_id)
            ),
            status=str(row.get("status") or metadata.get("status") or "active"),
            ingestion_run_id=str(
                row.get("ingestion_run_id")
                or metadata.get("ingestion_run_id")
                or candidate.ingestion_run_id
            ),
            corpus_version=str(
                row.get("corpus_version")
                or metadata.get("corpus_version")
                or candidate.corpus_version
                or self._config.corpus_version
            ),
            classification=_classification(
                row.get("classification")
                or metadata.get("classification")
                or candidate.classification
            ),
            embedding_model=str(
                row.get("embedding_model")
                or metadata.get("embedding_model")
                or candidate.embedding_model
            ),
            section_hint=candidate.section_hint or str(metadata.get("section_hint") or ""),
            image_id=candidate.image_id or optional_positive_int(metadata.get("image_id")),
            metadata=merged_metadata,
            origin=_origin_join(candidate.origin, "PostgresCanonicalEnrich"),
        )

    # =========================================================================
    # NEO4J
    # =========================================================================
    def _search_neo4j_entities(
        self,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
        driver: Any,
    ) -> list[RetrievalCandidate]:
        tokens = _search_tokens(query)
        if not tokens:
            return []
        cypher = """
            MATCH (c:Chunk)-[m:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)
            WHERE c.status = 'active' AND e.status = 'active' AND m.status = 'active'
              AND (
                    (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
                    OR
                    (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
                  )
              AND any(tok IN $tokens WHERE
                    toLower(coalesce(e.name, e.canonical_id, e.id, '')) CONTAINS tok OR
                    toLower(coalesce(e.description, '')) CONTAINS tok OR
                    toLower(coalesce(e.category, e.type, labels(e)[0], '')) CONTAINS tok OR
                    any(s IN coalesce(e.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok) OR
                    toLower(coalesce(c.filename, '')) CONTAINS tok OR
                    toLower(coalesce(c.text, '')) CONTAINS tok
                  )
            WITH c,
                 collect(DISTINCT coalesce(e.name, e.canonical_id, e.id)) AS entities,
                 count(DISTINCT e) AS rel_count
            RETURN
                coalesce(c.chunk_id, c.id) AS chunk_id,
                coalesce(c.doc_id, '') AS doc_id,
                coalesce(c.filename, 'Neo4j') AS filename,
                coalesce(c.page, 0) AS page,
                coalesce(c.page_chunk_index, 0) AS page_chunk_index,
                c.scope AS scope,
                c.organization_id AS organization_id,
                c.tier AS source_tier,
                c.status AS status,
                c.ingestion_run_id AS ingestion_run_id,
                c.corpus_version AS corpus_version,
                c.classification AS classification,
                entities,
                rel_count,
                toFloat(rel_count) * 2.0 AS graph_score
            ORDER BY graph_score DESC, page ASC, page_chunk_index ASC
            LIMIT $limit
        """
        output: list[RetrievalCandidate] = []
        with driver.session() as session:
            rows = session.run(
                cypher,
                tokens=list(tokens),
                org_id=tenant_context.organization_id,
                limit=int(limit),
            )
            for row in rows:
                chunk_id = str(row.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                entities = [str(value) for value in (row.get("entities") or ()) if value]
                try:
                    output.append(
                        RetrievalCandidate(
                            id=chunk_id,
                            content="Entity match: " + ", ".join(entities[:12]),
                            filename=str(row.get("filename") or "Neo4j"),
                            page=max(0, _safe_int(row.get("page"), 0)),
                            page_chunk_index=max(0, _safe_int(row.get("page_chunk_index"), 0)),
                            doc_id=str(row.get("doc_id") or ""),
                            type="graph",
                            tier="GRAPH",
                            source_tier=str(row.get("source_tier") or ""),
                            scope=str(row.get("scope") or ""),
                            organization_id=optional_positive_int(row.get("organization_id")),
                            status=str(row.get("status") or "active"),
                            ingestion_run_id=str(row.get("ingestion_run_id") or ""),
                            corpus_version=str(row.get("corpus_version") or self._config.corpus_version),
                            classification=_classification(row.get("classification")),
                            section_hint="Entities: " + ", ".join(entities[:5]),
                            origin="Neo4jEntitySearch",
                            score_graph=_safe_float(row.get("graph_score"), 1.0),
                        )
                    )
                except ValueError as exc:
                    logger.warning("Invalid Neo4j entity candidate %s: %s", chunk_id, exc)
        return output

    def _search_neo4j_formulas(
        self,
        query: str,
        *,
        limit: int,
        tenant_context: TenantContext,
        driver: Any,
    ) -> list[RetrievalCandidate]:
        tokens = _search_tokens(query)
        cypher = """
            MATCH (c:Chunk)
            WHERE c.status = 'active'
              AND (
                    (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
                    OR
                    (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
                  )
            CALL (c) {
                MATCH (c)-[rf:HAS_FORMULA|MENTIONS|MENTIONED_IN|PRESENT_IN]-(f)
                WHERE rf.status = 'active' AND f.status = 'active'
                  AND (
                        f:Formula
                        OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                        OR toUpper(coalesce(f.type, '')) = 'FORMULA'
                      )
                RETURN f
                UNION
                MATCH (c)-[re:MENTIONS|MENTIONED_IN|PRESENT_IN]-(e:Entity)
                      -[hf:HAS_FORMULA]-(f)
                WHERE re.status = 'active' AND e.status = 'active'
                  AND hf.status = 'active' AND f.status = 'active'
                  AND (
                        f:Formula
                        OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                        OR toUpper(coalesce(f.type, '')) = 'FORMULA'
                      )
                RETURN f
            }
            WITH c, f,
                 coalesce(f.latex, f.formula, '') AS latex,
                 coalesce(f.plain, f.name, f.canonical_id, f.fid, f.id, '') AS plain,
                 coalesce(f.meaning_it, f.meaning, f.description, '') AS meaning
            WHERE size($tokens) = 0
               OR any(tok IN $tokens WHERE
                    toLower(coalesce(c.filename, '')) CONTAINS tok OR
                    toLower(coalesce(c.text, '')) CONTAINS tok OR
                    toLower(toString(latex)) CONTAINS tok OR
                    toLower(toString(plain)) CONTAINS tok OR
                    toLower(toString(meaning)) CONTAINS tok OR
                    any(k IN coalesce(f.keywords, []) WHERE toLower(toString(k)) CONTAINS tok)
                  )
            RETURN DISTINCT
                coalesce(c.chunk_id, c.id) AS chunk_id,
                coalesce(c.doc_id, '') AS doc_id,
                coalesce(c.filename, 'Neo4j') AS filename,
                coalesce(c.page, 0) AS page,
                coalesce(c.page_chunk_index, 0) AS page_chunk_index,
                c.scope AS scope,
                c.organization_id AS organization_id,
                c.tier AS source_tier,
                c.status AS status,
                c.ingestion_run_id AS ingestion_run_id,
                c.corpus_version AS corpus_version,
                c.classification AS classification,
                latex,
                plain,
                meaning,
                coalesce(f.fid, f.entity_key, f.id, plain, latex) AS formula_key
            ORDER BY page ASC, page_chunk_index ASC
            LIMIT $limit
        """
        output: list[RetrievalCandidate] = []
        seen: set[tuple[str, str]] = set()
        with driver.session() as session:
            rows = session.run(
                cypher,
                tokens=list(tokens),
                org_id=tenant_context.organization_id,
                limit=int(limit),
            )
            for row in rows:
                chunk_id = str(row.get("chunk_id") or "").strip()
                formula_key = str(row.get("formula_key") or "").strip()
                if not chunk_id or (chunk_id, formula_key) in seen:
                    continue
                seen.add((chunk_id, formula_key))
                parts: list[str] = []
                latex = str(row.get("latex") or "").strip()
                plain = str(row.get("plain") or "").strip()
                meaning = str(row.get("meaning") or "").strip()
                if latex:
                    parts.append(f"LaTeX: {latex}")
                if plain and plain != latex:
                    parts.append(f"Plain: {plain}")
                if meaning:
                    parts.append(f"Meaning: {meaning}")
                if not parts:
                    continue
                try:
                    output.append(
                        RetrievalCandidate(
                            id=chunk_id,
                            content="Formula from Knowledge Graph:\n" + "\n".join(parts),
                            filename=str(row.get("filename") or "Neo4j"),
                            page=max(0, _safe_int(row.get("page"), 0)),
                            page_chunk_index=max(0, _safe_int(row.get("page_chunk_index"), 0)),
                            doc_id=str(row.get("doc_id") or ""),
                            type="formula",
                            tier="GRAPH",
                            source_tier=str(row.get("source_tier") or ""),
                            scope=str(row.get("scope") or ""),
                            organization_id=optional_positive_int(row.get("organization_id")),
                            status=str(row.get("status") or "active"),
                            ingestion_run_id=str(row.get("ingestion_run_id") or ""),
                            corpus_version=str(row.get("corpus_version") or self._config.corpus_version),
                            classification=_classification(row.get("classification")),
                            section_hint="Formula node",
                            origin="Neo4jFormulaSearch",
                            metadata={"formula_key": formula_key},
                            score_graph=5.0,
                        )
                    )
                except ValueError as exc:
                    logger.warning("Invalid Neo4j formula candidate %s: %s", chunk_id, exc)
        return output

    def _search_neo4j_relations(
        self,
        query: str,
        *,
        limit: int,
        target_document: str | None,
        tenant_context: TenantContext,
        driver: Any,
    ) -> list[RetrievalCandidate]:
        tokens = _graph_tokens(query) or _search_tokens(query)
        if not tokens:
            return []
        cypher = """
            MATCH (e1:Entity)-[rel]->(e2:Entity)
            WHERE rel.status = 'active'
              AND e1.status = 'active'
              AND e2.status = 'active'
              AND type(rel) IN $allowed_rels
              AND (
                    (
                        rel.scope = 'GLOBAL' AND rel.organization_id IS NULL
                        AND e1.scope = 'GLOBAL' AND e1.organization_id IS NULL AND e1.tier = 'A'
                        AND e2.scope = 'GLOBAL' AND e2.organization_id IS NULL AND e2.tier = 'A'
                    )
                    OR
                    (
                        rel.scope = 'ACCOUNT' AND rel.organization_id = $org_id
                        AND e1.scope = 'ACCOUNT' AND e1.organization_id = $org_id AND e1.tier IN ['B','C']
                        AND (
                            (e2.scope = 'ACCOUNT' AND e2.organization_id = $org_id AND e2.tier IN ['B','C'])
                            OR
                            (e2.scope = 'GLOBAL' AND e2.organization_id IS NULL AND e2.tier = 'A')
                        )
                    )
                  )
              AND any(tok IN $tokens WHERE
                    toLower(coalesce(e1.name, e1.canonical_id, e1.id, '')) CONTAINS tok OR
                    toLower(coalesce(e2.name, e2.canonical_id, e2.id, '')) CONTAINS tok OR
                    toLower(coalesce(e1.description, '')) CONTAINS tok OR
                    toLower(coalesce(e2.description, '')) CONTAINS tok OR
                    any(s IN coalesce(e1.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok) OR
                    any(s IN coalesce(e2.synonyms, []) WHERE toLower(toString(s)) CONTAINS tok)
                  )
            RETURN
                coalesce(e1.name, e1.canonical_id, e1.id) AS source,
                type(rel) AS relation,
                coalesce(e2.name, e2.canonical_id, e2.id) AS target,
                coalesce(rel.source_file, head(coalesce(rel.source_files, [])), '') AS filename,
                coalesce(rel.page_no, head(coalesce(rel.page_nos, [])), 0) AS page,
                rel.scope AS scope,
                rel.organization_id AS organization_id,
                rel.status AS status,
                rel.ingestion_run_id AS ingestion_run_id,
                rel.corpus_version AS corpus_version,
                rel.classification AS classification
            LIMIT $limit
        """
        output: list[RetrievalCandidate] = []
        with driver.session() as session:
            rows = session.run(
                cypher,
                tokens=list(tokens),
                org_id=tenant_context.organization_id,
                allowed_rels=list(self._config.neo4j_allowed_relationships),
                limit=max(int(limit) * 4, int(limit)),
            )
            for row in rows:
                source = str(row.get("source") or "").strip()
                target = str(row.get("target") or "").strip()
                relation = re.sub(
                    r"[^A-Z0-9_]+",
                    "_",
                    str(row.get("relation") or "RELATES_TO").upper(),
                ).strip("_") or "RELATES_TO"
                if not source or not target:
                    continue
                relation_text = f"{source} {relation} {target}".casefold()
                hit_count = sum(1 for token in tokens if token in relation_text)
                if hit_count < 2 and len(tokens) >= 2:
                    continue
                filename = str(row.get("filename") or "Neo4j Knowledge Graph")
                if target_document and not _document_matches(filename, target_document):
                    continue
                scope = str(row.get("scope") or "").upper()
                organization_id = optional_positive_int(row.get("organization_id"))
                tier = "GRAPH"
                digest = hashlib.sha256(
                    f"{source}|{relation}|{target}|{filename}|{row.get('page')}".encode("utf-8")
                ).hexdigest()[:24]
                source_md = source.replace("|", "\\|")
                target_md = target.replace("|", "\\|")
                filename_md = filename.replace("|", "\\|")
                page_no = max(0, _safe_int(row.get("page"), 0))

                content = (
                    "Relazione Neo4j esplicita:\n\n"
                    "| Entità sorgente | Relazione | Entità target | Documento | Pagina |\n"
                    "|---|---|---|---|---:|\n"
                    f"| {source_md} | {relation} | "
                    f"{target_md} | {filename_md} | "
                    f"{page_no} |"
                )
                try:
                    output.append(
                        RetrievalCandidate(
                            id=f"neo4j-relation-{digest}",
                            content=content,
                            filename=filename,
                            page=max(0, _safe_int(row.get("page"), 0)),
                            type="graph_relations",
                            tier=tier,
                            scope=scope,
                            organization_id=organization_id,
                            status=str(row.get("status") or "active"),
                            ingestion_run_id=str(row.get("ingestion_run_id") or ""),
                            corpus_version=str(row.get("corpus_version") or self._config.corpus_version),
                            classification=_classification(row.get("classification")),
                            section_hint="Explicit Neo4j relation",
                            origin="Neo4jRelationSearch",
                            metadata={
                                "source": source,
                                "relation": relation,
                                "target": target,
                            },
                            score_graph=max(1.0, float(hit_count)),
                        )
                    )
                except ValueError as exc:
                    logger.warning("Invalid Neo4j relation candidate: %s", exc)
                if len(output) >= limit:
                    break
        return output

    def _get_neighbor_chunk_ids(
        self,
        seed_ids: Sequence[str],
        *,
        limit: int,
        tenant_context: TenantContext,
        driver: Any,
    ) -> list[str]:
        unique_ids = list(dict.fromkeys(str(value) for value in seed_ids if str(value).strip()))
        if not unique_ids:
            return []
        cypher = """
            MATCH
                (c1:Chunk)-[r1:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)
                -[r2:MENTIONS|PRESENT_IN|MENTIONED_IN]-(c2:Chunk)
            WHERE coalesce(c1.chunk_id, c1.id) IN $ids
              AND c1.status = 'active' AND c2.status = 'active' AND e.status = 'active'
              AND r1.status = 'active' AND r2.status = 'active'
              AND NOT coalesce(c2.chunk_id, c2.id) IN $ids
              AND (
                    (c1.scope = 'GLOBAL' AND c1.organization_id IS NULL AND c1.tier = 'A')
                    OR
                    (c1.scope = 'ACCOUNT' AND c1.organization_id = $org_id AND c1.tier IN ['B', 'C'])
                  )
              AND (
                    (c2.scope = 'GLOBAL' AND c2.organization_id IS NULL AND c2.tier = 'A')
                    OR
                    (c2.scope = 'ACCOUNT' AND c2.organization_id = $org_id AND c2.tier IN ['B', 'C'])
                  )
              AND (
                    (e.scope = 'GLOBAL' AND e.organization_id IS NULL AND e.tier = 'A')
                    OR
                    (e.scope = 'ACCOUNT' AND e.organization_id = $org_id AND e.tier IN ['B', 'C'])
                  )
              AND NOT toUpper(coalesce(e.type, e.category, labels(e)[0], '')) IN ['GENERIC', 'YEAR', 'DATE']
            WITH c2, count(DISTINCT e) AS entity_count
            WHERE entity_count >= 2
            RETURN coalesce(c2.chunk_id, c2.id) AS chunk_id
            ORDER BY entity_count DESC,
                     coalesce(c2.page, 0),
                     coalesce(c2.page_chunk_index, 0)
            LIMIT $limit
        """
        with driver.session() as session:
            rows = session.run(
                cypher,
                ids=unique_ids,
                org_id=tenant_context.organization_id,
                limit=int(limit),
            )
            return [
                str(row.get("chunk_id"))
                for row in rows
                if row.get("chunk_id")
            ]

    def _get_graph_entities(
        self,
        chunk_ids: Sequence[str],
        *,
        tenant_context: TenantContext,
        driver: Any,
    ) -> dict[str, list[GraphEntity]]:
        ids = list(dict.fromkeys(str(value) for value in chunk_ids if str(value).strip()))
        if not ids:
            return {}
        cypher = """
            UNWIND $ids AS target_id
            MATCH (c:Chunk)
            WHERE coalesce(c.chunk_id, c.id) = target_id
              AND c.status = 'active'
              AND (
                    (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
                    OR
                    (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
                  )
            CALL (c) {
                MATCH (c)-[r:MENTIONS|PRESENT_IN|MENTIONED_IN]-(e:Entity)
                WHERE r.status = 'active' AND e.status = 'active'
                RETURN
                    coalesce(e.name, e.label, e.canonical_id, e.id) AS entity_name,
                    coalesce(e.category, labels(e)[0], 'Entity') AS entity_type,
                    type(r) AS relation
                LIMIT 10
            }
            RETURN target_id AS chunk_id, entity_name, entity_type, relation
        """
        result: dict[str, list[GraphEntity]] = {}
        with driver.session() as session:
            rows = session.run(
                cypher,
                ids=ids,
                org_id=tenant_context.organization_id,
            )
            for row in rows:
                chunk_id = str(row.get("chunk_id") or "")
                name = str(row.get("entity_name") or "").strip()
                if not chunk_id or not name:
                    continue
                result.setdefault(chunk_id, []).append(
                    GraphEntity(
                        name=name,
                        type=str(row.get("entity_type") or "Entity"),
                        relation=str(row.get("relation") or "MENTIONED"),
                    )
                )
        return result

    def _get_formulas_for_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        limit_per_chunk: int,
        tenant_context: TenantContext,
        driver: Any,
    ) -> dict[str, list[str]]:
        ids = list(dict.fromkeys(str(value) for value in chunk_ids if str(value).strip()))
        if not ids:
            return {}
        cypher = """
            UNWIND $ids AS target_id
            MATCH (c:Chunk)
            WHERE coalesce(c.chunk_id, c.id) = target_id
              AND c.status = 'active'
              AND (
                    (c.scope = 'GLOBAL' AND c.organization_id IS NULL AND c.tier = 'A')
                    OR
                    (c.scope = 'ACCOUNT' AND c.organization_id = $org_id AND c.tier IN ['B', 'C'])
                  )
            CALL (c) {
                MATCH (c)-[rf:HAS_FORMULA|MENTIONS|MENTIONED_IN|PRESENT_IN]-(f)
                WHERE rf.status = 'active' AND f.status = 'active'
                  AND (
                        f:Formula
                        OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                        OR toUpper(coalesce(f.type, '')) = 'FORMULA'
                      )
                RETURN f
                UNION
                MATCH (c)-[re:MENTIONS|MENTIONED_IN|PRESENT_IN]-(e:Entity)
                      -[hf:HAS_FORMULA]-(f)
                WHERE re.status = 'active' AND e.status = 'active'
                  AND hf.status = 'active' AND f.status = 'active'
                  AND (
                        f:Formula
                        OR toUpper(coalesce(f.category, '')) = 'FORMULA'
                        OR toUpper(coalesce(f.type, '')) = 'FORMULA'
                      )
                RETURN f
            }
            WITH target_id, f,
                 coalesce(f.latex, f.formula, '') AS latex,
                 coalesce(f.plain, f.name, f.canonical_id, f.fid, f.id, '') AS plain,
                 coalesce(f.meaning_it, f.meaning, f.description, '') AS meaning
            WHERE trim(toString(latex)) <> ''
               OR trim(toString(plain)) <> ''
               OR trim(toString(meaning)) <> ''
            WITH target_id, collect(DISTINCT {latex: latex, plain: plain, meaning: meaning})[0..$limit] AS formulas
            UNWIND formulas AS formula
            RETURN target_id AS chunk_id,
                   formula.latex AS latex,
                   formula.plain AS plain,
                   formula.meaning AS meaning
        """
        result: dict[str, list[str]] = {}
        with driver.session() as session:
            rows = session.run(
                cypher,
                ids=ids,
                org_id=tenant_context.organization_id,
                limit=max(1, int(limit_per_chunk)),
            )
            for row in rows:
                parts: list[str] = []
                latex = str(row.get("latex") or "").strip()
                plain = str(row.get("plain") or "").strip()
                meaning = str(row.get("meaning") or "").strip()
                if latex:
                    parts.append(f"LaTeX: {latex}")
                if plain and plain != latex:
                    parts.append(f"Plain: {plain}")
                if meaning:
                    parts.append(f"Meaning: {meaning}")
                if parts:
                    result.setdefault(str(row.get("chunk_id")), []).append(
                        " | ".join(parts)
                    )
        return result

    # =========================================================================
    # MERGE
    # =========================================================================
    @staticmethod
    def _put_candidate(
        target: dict[str, RetrievalCandidate],
        candidate: RetrievalCandidate,
    ) -> None:
        current = target.get(candidate.id)
        target[candidate.id] = (
            candidate if current is None else _merge_candidates(current, candidate)
        )


# Singleton without import-time resource initialisation.
retrieval_engine = HybridRetrievalEngine(
    config=settings,
    resource_manager=resources,
)


__all__ = [
    "HybridRetrievalEngine",
    "RetrievalBackendError",
    "RetrievalConfigurationError",
    "RetrievalError",
    "RetrievalProtocolError",
    "retrieval_engine",
]
