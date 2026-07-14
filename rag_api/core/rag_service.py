"""Application service for the multi-tenant Hybrid-RAG backend.

This module is the orchestration boundary of the RAG engine.  It coordinates:

1. trusted tenant context;
2. deterministic query routing;
3. optional direct solvers / glossary lookup;
4. hybrid retrieval through a typed port;
5. reranking and source materialisation;
6. prompt construction;
7. Ollama generation;
8. deterministic answer validation;
9. optional faithfulness evaluation;
10. immutable audit persistence.

The module intentionally has no dependency on FastAPI or Reflex.  HTTP schemas
are mapped by the future ``api/routes_rag.py`` layer.

``core/retrieval.py`` is not required at import time.  The default lazy adapter
expects that module to expose ``retrieval_engine`` with a compatible
``retrieve_candidates`` method.  This makes the contract explicit while still
allowing ``rag_service.py`` to be developed and tested independently.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence, runtime_checkable
from uuid import UUID

from .audit import (
    AuditIdentityError,
    AuditService,
    append_rag_eval_log_async,
    audit_service,
    create_query_audit,
    format_evaluation_audit_markdown,
    format_retrieval_audit_markdown,
)
from .config import RagSettings, settings

from .generation import (
    GenerationError,
    GenerationResult,
    OllamaNativeGenerator,
    generator,
)

from .generation import (
    GenerationError,
    GenerationResult,
    OllamaNativeGenerator,
    generator,
)
from .graph_relations import answer_graph_relations_strict

from .models import (
    RagAnswerMode,
    RagEvalResult,
    RagExecutionMode,
    RagIntent,
    RagServiceResult,
    RetrievalCandidate,
    RetrievalDebug,
    SourceItem,
)
from .prompting import (
    PromptBuildOptions,
    PromptBuilder,
    PromptBundle,
    document_matches,
    prompt_builder,
)
from .reranking import RankingContext, RerankingEngine, RerankingResult
from .resources import ResourceManager, ResourceNotReadyError, resources
from .solvers import SolverResult, is_calculation_request, solve_deterministic_query
from .tenant import (
    TenantContext,
    TenantContextError,
    bind_tenant_context,
    filter_visible_records,
    get_tenant_context,
)
from .validation import (
    AnswerValidator,
    FaithfulnessEvaluator,
    ValidationPolicy,
    answer_validator,
    evaluation_requires_block,
    faithfulness_evaluator,
    strict_evaluation_fallback,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ERRORI DEL SERVICE LAYER
# =============================================================================
class RagServiceError(RuntimeError):
    """Errore base del motore applicativo RAG."""


class RagServiceConfigurationError(RagServiceError):
    """Alberatura o dipendenza applicativa non configurata."""


class RagServiceRetrievalError(RagServiceError):
    """Errore prodotto dalla pipeline di retrieval/reranking."""


class RagServiceGenerationError(RagServiceError):
    """Errore prodotto dal modello generativo."""


class RagServiceValidationError(RagServiceError):
    """Il quality gate ha bloccato la risposta per una violazione critica."""


# =============================================================================
# CONTRATTI INTERNI
# =============================================================================
@dataclass(frozen=True, slots=True)
class RagQueryCommand:
    """Comando applicativo indipendente dal trasporto HTTP.

    Il futuro router FastAPI mapperà ``api.schemas.RagQueryRequest`` in questa
    struttura.  Nessun dato tenant è accettato nel comando.
    """

    query: str
    conversation_id: str | None = None
    history: tuple[Any, ...] = field(default_factory=tuple)

    target_document: str | None = None
    target_pages: tuple[int, ...] = field(default_factory=tuple)
    max_sources: int | None = None

    include_evaluation: bool = False

    def __post_init__(self) -> None:
        query = str(self.query or "").strip()
        if not query:
            raise ValueError("query non può essere vuota")
        if "\x00" in query:
            raise ValueError("query contiene caratteri null non validi")
        object.__setattr__(self, "query", query)

        conversation_id = (
            str(self.conversation_id).strip()
            if self.conversation_id is not None
            else None
        )
        object.__setattr__(self, "conversation_id", conversation_id or None)

        if conversation_id and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}", conversation_id
        ):
            raise ValueError("conversation_id non valido")

        object.__setattr__(self, "history", tuple(self.history or ()))

        target_document = _safe_document_name(self.target_document)
        object.__setattr__(self, "target_document", target_document or None)

        target_pages = tuple(sorted(set(int(page) for page in self.target_pages or ())))
        if any(page <= 0 for page in target_pages):
            raise ValueError("target_pages accetta soltanto pagine 1-based positive")
        if len(target_pages) > 50:
            raise ValueError("target_pages può contenere al massimo 50 pagine")
        object.__setattr__(self, "target_pages", target_pages)

        if self.max_sources is not None:
            parsed_max = int(self.max_sources)
            if not 1 <= parsed_max <= 50:
                raise ValueError("max_sources deve essere compreso tra 1 e 50")
            object.__setattr__(self, "max_sources", parsed_max)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Decisione deterministica del router applicativo."""

    intent: RagIntent
    answer_mode: RagAnswerMode
    execution_mode: RagExecutionMode

    wants_evidence: bool = False
    calculation_mode: bool = False
    analytics_mode: bool = False
    
   
    strict_checklist_mode: bool = False
    crosswalk_mode: bool = False

    # La query richiede retrieval/supporto dal Knowledge Graph.
    graph_search_mode: bool = False

    # La query richiede una risposta deterministica su archi/path/relazioni.
    graph_relation_mode: bool = False

    formula_strict_mode: bool = False

    glossary_candidate: bool = False
    exhaustive_formula_lookup: bool = False

    requested_document: str | None = None
    retrieval_query: str = ""

    solver_result: SolverResult | None = None
    math_needs_document_context: bool = False

    @property
    def is_direct_math(self) -> bool:
        return bool(
            self.solver_result
            and not self.math_needs_document_context
            and self.execution_mode == RagExecutionMode.MATH_DIRECT
        )

    @property
    def deterministic(self) -> bool:
        return self.solver_result is not None


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """Risposta deterministica opzionale fornita da un adapter specializzato."""

    answer: str
    sources: tuple[SourceItem, ...] = field(default_factory=tuple)
    execution_mode: RagExecutionMode = RagExecutionMode.GLOSSARY_DIRECT
    audit_markdown: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.answer or "").strip():
            raise ValueError("DirectAnswer.answer non può essere vuoto")


@runtime_checkable
class RetrievalPort(Protocol):
    """Contratto che il futuro ``core/retrieval.py`` deve implementare.

    Il retrieval restituisce candidati non ancora materializzati come fonti
    finali.  RRF, CrossEncoder e diversificazione restano responsabilità del
    service tramite ``core/reranking.py``.
    """

    def retrieve_candidates(
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
    ) -> (
        tuple[Sequence[RetrievalCandidate], RetrievalDebug]
        | Awaitable[tuple[Sequence[RetrievalCandidate], RetrievalDebug]]
    ):
        ...


@runtime_checkable
class GlossaryPort(Protocol):
    """Lookup atomico opzionale per le definizioni di glossario."""

    def lookup_glossary(
        self,
        *,
        query: str,
        tenant_context: TenantContext,
    ) -> DirectAnswer | None | Awaitable[DirectAnswer | None]:
        ...


class LazyRetrievalAdapter:
    """Risolve ``core.retrieval.retrieval_engine`` solo al primo utilizzo."""

    def __init__(self) -> None:
        self._resolved: Any | None = None

    def _resolve(self) -> Any:
        if self._resolved is not None:
            return self._resolved

        try:
            from . import retrieval as retrieval_module  # type: ignore
        except ImportError as exc:
            raise RagServiceConfigurationError(
                "core/retrieval.py non è ancora disponibile. Il RagService è "
                "correttamente costruito, ma le query documentali richiedono "
                "l'implementazione del RetrievalPort."
            ) from exc

        engine = getattr(retrieval_module, "retrieval_engine", None)
        if engine is None:
            raise RagServiceConfigurationError(
                "core.retrieval deve esporre il singleton retrieval_engine"
            )

        self._resolved = engine
        return engine

    async def retrieve_candidates(self, **kwargs: Any) -> tuple[Sequence[RetrievalCandidate], RetrievalDebug]:
        engine = self._resolve()
        method = getattr(engine, "retrieve_candidates", None)
        if method is None:
            method = getattr(engine, "retrieve", None)
        if method is None:
            raise RagServiceConfigurationError(
                "retrieval_engine deve esporre retrieve_candidates(...)"
            )

        result = method(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return _validate_retrieval_result(result)

    async def lookup_glossary(
        self,
        *,
        query: str,
        tenant_context: TenantContext,
    ) -> DirectAnswer | None:
        engine = self._resolve()
        method = getattr(engine, "lookup_glossary", None)
        if method is None:
            return None
        result = method(query=query, tenant_context=tenant_context)
        if inspect.isawaitable(result):
            result = await result
        return _coerce_direct_answer(result)


# =============================================================================
# ROUTER DETERMINISTICO
# =============================================================================
class RagQueryRouter:
    """Router generalista estratto dal flusso ``handle_submit`` del PoC."""

    _FORMULA_TERMS = (
        "formula", "formule", "equazione", "equazioni", "disequazione",
        "disequazioni", "algebra", "algebrica", "risolvi", "calcola",
        "calcolo", "deriva", "esprimi", "isola", "formulae", "formulas",
        "equation", "equations", "inequality", "inequalities", "solve",
        "calculate", "compute", "derive", "express", "rosi", "roi",
    )
    _TABLE_TERMS = (
        "tabella", "table", "matrice", "matrix", "crosswalk", "mappatura",
        "mapping", "righe", "rows", "colonne", "columns",
    )
    _GRAPH_TERMS = (
        "neo4j", "cypher", "grafo", "graph", "relazioni", "relationships",
        "nodi", "nodes", "archi", "edges", "percorso", "path",
        "traversamento", "traversal", "multi-hop", "rete semantica",
        "knowledge graph",
    )
    _AUDIT_TERMS = (
        "audit", "compliance", "conformità", "conformita", "assessment",
        "requisito", "requisiti", "requirement", "requirements", "controllo",
        "controlli", "control", "controls", "normativa", "regolamento",
        "regulation", "policy", "evidenza", "evidenze", "evidence",
        "violazione", "obbligo", "obblighi", "gap", "remediation",
    )
    _EVIDENCE_TERMS = (
        "evidenza", "evidenze", "evidence", "proof", "log", "logs",
        "configurazione", "configuration", "implementazione", "implementation",
        "screenshot", "record", "records", "ticket", "dimostra", "demonstrate",
    )
    _GLOSSARY_TERMS = (
        "glossario", "definisci", "definizione", "significato", "acronimo",
        "sta per", "cosa significa", "cosa vuol dire", "cosa si intende",
        "glossary", "define", "definition", "meaning", "acronym",
        "stands for", "what does it mean", "what is meant by",
    )
    _REASONING_TERMS = (
        "spiega", "confronta", "differenza", "valuta", "analizza", "perché",
        "perche", "relazione", "impatto", "conseguenza", "giustifica",
        "explain", "compare", "difference", "evaluate", "analyze", "analyse",
        "why", "relationship", "impact", "justify",
    )

    def route(self, command: RagQueryCommand) -> RoutingDecision:
        query = command.query
        q = query.casefold()

        requested_document = command.target_document or _extract_requested_document(query)
        evidence_relevance = _is_evidence_relevance_query(query)

        intent = RagIntent.AUDIT if evidence_relevance else self._detect_intent(query)
        answer_mode = self._detect_answer_mode(query, evidence_relevance)

        solver_result = None if evidence_relevance else solve_deterministic_query(query)
        math_needs_context = bool(
            solver_result and _needs_math_document_context(query)
        )

        analytics_mode = bool(
            not evidence_relevance
            and solver_result is None
            and _is_user_data_analytics(query)
        )
        
        graph_search_mode = _is_graph_relation_query(query)

        graph_relation_mode = _should_use_graph_relation_strict_mode(query)
                    
        formula_strict_mode = bool(
            not evidence_relevance
            and _is_formula_lookup_query(query)
        )        

        strict_checklist_mode = _is_strict_checklist_query(query)
        crosswalk_mode = _is_crosswalk_query(query)
        glossary_candidate = self._is_pure_glossary(query)
        wants_evidence = evidence_relevance or any(term in q for term in self._EVIDENCE_TERMS)
        exhaustive_formula_lookup = _is_exhaustive_formula_lookup(query)

        if solver_result and not math_needs_context:
            execution_mode = RagExecutionMode.MATH_DIRECT
        elif analytics_mode:
            execution_mode = RagExecutionMode.ANALYTICS
        elif graph_relation_mode:
            execution_mode = RagExecutionMode.GRAPH_RELATION_STRICT
        elif formula_strict_mode:
            execution_mode = RagExecutionMode.FORMULA_STRICT
        else:
            execution_mode = RagExecutionMode.RAG_GENERATION

        retrieval_query = _build_retrieval_query(
            query,
            evidence_relevance=evidence_relevance,
            math_needs_context=math_needs_context,
            graph_relation_mode=graph_search_mode,
            formula_mode=formula_strict_mode,
        )

        return RoutingDecision(
            intent=intent,
            answer_mode=answer_mode,
            execution_mode=execution_mode,
            wants_evidence=wants_evidence,
            calculation_mode=is_calculation_request(query),
            analytics_mode=analytics_mode,
            strict_checklist_mode=strict_checklist_mode,
            
            crosswalk_mode=crosswalk_mode,
            graph_search_mode=graph_search_mode,
            graph_relation_mode=graph_relation_mode,
            formula_strict_mode=formula_strict_mode,

            glossary_candidate=glossary_candidate,
            exhaustive_formula_lookup=exhaustive_formula_lookup,
            requested_document=requested_document,
            retrieval_query=retrieval_query,
            solver_result=solver_result,
            math_needs_document_context=math_needs_context,
        )


    @staticmethod
    def _is_regulatory_classification_query(query_text: str) -> bool:
        """
        Riconosce domande normative e classificatorie che devono essere
        tratt devono essere
        trattate come audit, non come formula, glossario o testo generico.

        La regola è framework-agnostic e non dipende da documenti o test.
        """

        q = str(query_text or "").lower().strip()

        if not q:
            return False

        classification_starters = (
            # Italiano
            "chi sono",
            "quali sono",
            "qual è",
            "quale è",

            # Inglese
            "what are",
            "who are",
            "which are",
            "what is",
        )

        regulatory_terms = (
            # Italiano
            "soggetti",
            "soggetto",
            "categorie",
            "categoria",
            "tipologie",
            "tipologia",
            "regime",
            "vigilanza",
            "obblighi",
            "obbligo",
            "requisiti",
            "requisito",
            "normativa",
            "regolamento",
            "direttiva",
            "legge",
            "classificazione",
            "classifica",
            "autorità",
            "responsabilità",
            "categorie normative",

            # Inglese
            "subjects",
            "entities",
            "categories",
            "category",
            "types",
            "classification",
            "regime",
            "supervision",
            "oversight",
            "obligations",
            "requirements",
            "regulation",
            "directive",
            "law",
            "authority",
            "responsibilities",
        )

        has_classification_starter = any(
            term in q
            for term in classification_starters
        )

        has_regulatory_term = any(
            term in q
            for term in regulatory_terms
        )

        if has_classification_starter and has_regulatory_term:
            return True

        classification_density = sum(
            1
            for term in regulatory_terms
            if term in q
        )

        return classification_density >= 2




    def _detect_intent(self, query: str) -> RagIntent:
        q = str(query or "").casefold()

        # Le domande su soggetti, categorie, obblighi, requisiti,
        # classificazione e vigilanza sono query normative.
        # Questo controllo deve precedere Formula Mode.
        if self._is_regulatory_classification_query(query):
            return RagIntent.AUDIT

        if (
            is_calculation_request(query)
            or any(term in q for term in self._FORMULA_TERMS)
        ):
            return RagIntent.FORMULA

        if any(term in q for term in self._GRAPH_TERMS):
            return RagIntent.CHART

        if any(term in q for term in self._TABLE_TERMS):
            return RagIntent.TABLE

        if any(term in q for term in self._AUDIT_TERMS):
            return RagIntent.AUDIT

        return RagIntent.TEXT


    def _detect_answer_mode(
        self,
        query: str,
        evidence_relevance: bool,
    ) -> RagAnswerMode:
        """
        Determina la modalità di risposta.

        Priorità:
        1. valutazione dell'attinenza di un'evidenza;
        2. audit, conformità, evidenze, gap e verifiche;
        3. risposta informativa generale.
        """

        if evidence_relevance:
            return RagAnswerMode.EVIDENCE_RELEVANCE

        q = str(query or "").strip().casefold()

        if not q:
            return RagAnswerMode.KNOWLEDGE

        audit_eval_terms = (
            # Italiano
            "verifica conformità",
            "valutazione conformità",
            "non conformità",
            "non conforme",
            "audit",
            "evidenza",
            "evidenze",
            "evidenze implementazione",
            "policy contro evidenza",
            "tier b",
            "tier c",
            "gap tecnico",
            "gap analysis",
            "analisi dei gap",
            "scostamento",
            "discrepanza",
            "ispezione",
            "allineamento tecnico",
            "deviazione",

            # Inglese
            "compliance check",
            "compliance assessment",
            "non-compliance",
            "non-compliant",
            "evidence",
            "implementation evidence",
            "policy vs evidence",
            "technical gap",
            "deviation",
            "discrepancy",
            "inspection",
            "technical alignment",
        )

        if any(
            re.search(rf"\b{re.escape(term)}\b", q)
            for term in audit_eval_terms
        ):
            return RagAnswerMode.AUDIT

        return RagAnswerMode.KNOWLEDGE

    def _is_pure_glossary(self, query: str) -> bool:
        q = query.casefold()
        if not any(term in q for term in self._GLOSSARY_TERMS):
            return False
        if any(term in q for term in self._REASONING_TERMS):
            return False
        if _is_mixed_glossary_query(query):
            return False
        if _is_graph_relation_query(query):
            return False
        if is_calculation_request(query):
            return False
        return True


# =============================================================================
# SERVICE ORCHESTRATOR
# =============================================================================
class RagService:
    """Orchestratore stateless e riutilizzabile del motore Hybrid-RAG."""

    def __init__(
        self,
        *,
        config: RagSettings = settings,
        resource_manager: ResourceManager = resources,
        retriever: RetrievalPort | Any | None = None,
        router: RagQueryRouter | None = None,
        prompt_factory: PromptBuilder = prompt_builder,
        llm_generator: OllamaNativeGenerator = generator,
        validator: AnswerValidator = answer_validator,
        evaluator: FaithfulnessEvaluator = faithfulness_evaluator,
        auditor: AuditService = audit_service,
        reranking_engine: RerankingEngine | None = None,
    ) -> None:
        self._config = config
        self._resources = resource_manager
        self._retriever = retriever or LazyRetrievalAdapter()
        self._router = router or RagQueryRouter()
        self._prompt_builder = prompt_factory
        self._generator = llm_generator
        self._validator = validator
        self._evaluator = evaluator
        self._auditor = auditor
        self._reranking_engine = reranking_engine

    async def query(
        self,
        command: RagQueryCommand,
        *,
        tenant_context: TenantContext | None = None,
    ) -> RagServiceResult:
        """Esegue una query RAG nel tenant context corrente.

        Il metodo non crea autonomamente il tenant.  Il futuro middleware/API
        deve passare un ``TenantContext`` trusted oppure averlo già associato con
        ``tenant_request_scope``.
        """

        context = tenant_context or get_tenant_context()
        with bind_tenant_context(context):
            return await self._query_bound(command, context)

    def query_sync(
        self,
        command: RagQueryCommand,
        *,
        tenant_context: TenantContext | None = None,
    ) -> RagServiceResult:
        """Adapter sincrono per test e job non-ASGI.

        Non può essere invocato dentro un event loop già attivo.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.query(command, tenant_context=tenant_context))
        raise RuntimeError("query_sync non può essere usato dentro un event loop attivo")

    async def _query_bound(
        self,
        command: RagQueryCommand,
        context: TenantContext,
    ) -> RagServiceResult:
        started = time.perf_counter()
        route = self._router.route(command)
        warnings: list[str] = []

        debug = RetrievalDebug(
            query=command.query,
            intent=route.intent,
            answer_mode=route.answer_mode,
            wants_evidence=route.wants_evidence,
            default_tiers=tuple(self._config.rag_default_tiers),
            target_document=route.requested_document,
            target_pages=command.target_pages,
        )

        execution_mode = route.execution_mode
        answer = ""
        sources: tuple[SourceItem, ...] = ()
        audit_sources: tuple[SourceItem, ...] = ()
        prompt_bundle: PromptBundle | None = None
        generation: GenerationResult | None = None
        evaluation: RagEvalResult | None = None
        direct_audit_markdown = ""

        if route.is_direct_math:
            answer = route.solver_result.answer if route.solver_result else ""
        else:
            direct = await self._try_glossary_direct(route, command, context)
            if direct is not None:
                execution_mode = direct.execution_mode
                answer = direct.answer
                sources = tuple(direct.sources)
                audit_sources = sources
                direct_audit_markdown = direct.audit_markdown
                warnings.extend(direct.warnings)
            else:
                if route.analytics_mode:
                    sources = (self._make_user_input_source(command.query, context),)
                else:
                    sources, debug, rerank_warnings = await self._retrieve_and_rank(
                        command=command,
                        route=route,
                        context=context,
                        debug=debug,
                    )
                    warnings.extend(rerank_warnings)

                audit_sources = sources
                
                if not sources:
                    answer = _no_sources_fallback(route.requested_document)
                    warnings.append("Nessuna fonte tenant-visible recuperata.")

                elif (
                    route.solver_result is not None
                    and route.math_needs_document_context
                ):
                    execution_mode = RagExecutionMode.MATH_DIRECT

                    answer = _build_math_answer_with_document_context(
                        route.solver_result.answer,
                        sources,
                    )

                    direct_audit_markdown = (
                        "### Math-First Mode\n"
                        "- Calcolo deterministico eseguito prima della generazione LLM.\n"
                        "- Il modello generativo è stato bypassato.\n"
                        "- Le fonti recuperate sono state utilizzate esclusivamente "
                        "per la contestualizzazione documentale."
                    )

                    warnings.append(
                        "Math-First Mode: risultato deterministico preservato; "
                        "generazione LLM non eseguita."
                    )

                elif route.graph_relation_mode:
                    execution_mode = RagExecutionMode.GRAPH_RELATION_STRICT

                    graph_answer = answer_graph_relations_strict(
                        command.query,
                        sources,
                    )

                    if graph_answer:
                        answer = graph_answer
                    else:
                        answer = (
                            "**A) Risposta**\n\n"
                            "Non è stato possibile identificare almeno due concetti "
                            "distinti da mettere in relazione.\n\n"
                            "**B) Evidenze**\n\n"
                            "- Il retrieval è stato eseguito, ma la richiesta non contiene "
                            "un insieme sufficiente di entità confrontabili.\n\n"
                            "**C) Limiti / Conflitti**\n\n"
                            "- Non vengono inventati archi o relazioni non presenti nel "
                            "Knowledge Graph o nelle fonti documentali.\n\n"
                            "**D) Fonti**\n\n"
                            "- Nessuna relazione deterministica costruibile."
                        )

                    direct_audit_markdown = (
                        "### Graph Relation Strict Mode\n"
                        "- Risposta costruita deterministicamente dalle fonti recuperate.\n"
                        "- Gli archi Neo4j sono distinti dal semplice supporto testuale.\n"
                        "- Le co-occorrenze non vengono trasformate in relazioni esplicite.\n"
                        "- Il modello generativo è stato bypassato."
                    )

                    warnings.append(
                        "Graph Relation Strict Mode: risposta deterministica; "
                        "generazione LLM non eseguita."
                    )

                else:
                    prompt_bundle = self._prompt_builder.build(
                        query=command.query,
                        sources=sources,
                        history=command.history,
                        options=PromptBuildOptions(
                            intent=route.intent,
                            answer_mode=route.answer_mode,
                            requested_document=route.requested_document,
                            strict_checklist_mode=route.strict_checklist_mode,
                            crosswalk_mode=route.crosswalk_mode,
                            graph_relation_mode=route.graph_search_mode,
                            calculation_mode=route.calculation_mode,
                            analytics_mode=route.analytics_mode,
                            wants_evidence=route.wants_evidence,
                            deterministic_math_answer=(
                                route.solver_result.answer
                                if route.solver_result is not None
                                else None
                            ),
                            math_needs_document_context=route.math_needs_document_context,
                        ),
                        tenant_context=context,
                    )
                    warnings.extend(prompt_bundle.warnings)

                    try:
                        generation = await self._generator.generate_async(prompt_bundle)
                    except GenerationError as exc:
                        await self._audit_failed_generation(
                            command=command,
                            route=route,
                            execution_mode=execution_mode,
                            debug=debug,
                            sources=audit_sources,
                            prompt_bundle=prompt_bundle,
                            context=context,
                            started=started,
                            error=exc,
                        )
                        raise RagServiceGenerationError(str(exc)) from exc

                    answer = generation.content
                    warnings.extend(generation.warnings)

        validation_policy = ValidationPolicy(
            intent=route.intent,
            answer_mode=route.answer_mode,
            execution_mode=execution_mode,
            requested_document=route.requested_document,
            require_sources=bool(sources) and execution_mode != RagExecutionMode.MATH_DIRECT,
            rebuild_sources_section=(execution_mode != RagExecutionMode.MATH_DIRECT),
            enforce_requested_document=bool(route.requested_document),
            allow_graph_tier=True,
            allow_user_tier=True,
            max_answer_chars=self._config.max_assistant_chars,
        )

        validation = self._validator.validate(
            answer=answer,
            query=command.query,
            sources=sources,
            policy=validation_policy,
            tenant_context=context,
        )
        warnings.extend(validation.warnings)

        if validation.blocked:
            # Il validator ha già sostituito il contenuto con una risposta sicura.
            logger.error(
                "RAG answer blocked | request_id=%s issues=%s",
                context.request_id,
                [str(item.code) for item in validation.issues],
            )

        final_answer = validation.answer
        public_sources = tuple(validation.visible_sources)

        strict_eval_blocked = False
        if command.include_evaluation:
            evaluation = await self._evaluator.evaluate_async(
                query=command.query,
                answer=final_answer,
                sources=public_sources,
                requested_document=route.requested_document or "",
                tenant_context=context,
            )

            if evaluation_requires_block(evaluation, config=self._config):
                strict_eval_blocked = True
                final_answer = strict_evaluation_fallback(evaluation)
                public_sources = ()
                warnings.append(
                    "La risposta è stata bloccata dalla policy EVAL_STRICT_BLOCK."
                )

            try:
                eval_audit = await append_rag_eval_log_async(
                    query=command.query,
                    answer=final_answer,
                    sources=audit_sources,
                    evaluation=evaluation,
                    requested_document=route.requested_document or "",
                    strict_block_applied=strict_eval_blocked,
                    warnings=tuple(warnings),
                    context=context,
                    config=self._config,
                )
                if not eval_audit.success and not eval_audit.skipped:
                    warnings.append("Persistenza audit evaluation non completata.")
            except AuditIdentityError:
                raise
            except Exception as exc:  # audit best effort, identità esclusa
                logger.exception(
                    "Evaluation audit failed | request_id=%s", context.request_id
                )
                warnings.append(f"Audit evaluation degradato: {type(exc).__name__}.")

        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        prompt_hash = prompt_bundle.prompt_sha256 if prompt_bundle else ""
        context_chars = prompt_bundle.context.context_chars if prompt_bundle else 0
        model_name = generation.model if generation else ""

        audit = create_query_audit(
            query=command.query,
            sources=audit_sources,
            intent=route.intent,
            answer_mode=route.answer_mode,
            execution_mode=execution_mode,
            retrieval=debug,
            filters=self._audit_filters(command, route, validation.blocked),
            prompt_sha256=prompt_hash,
            context_chars=context_chars,
            deterministic=route.deterministic,
            llm_model=model_name or self._config.llm_model_name,
            elapsed_ms=elapsed_ms,
            warnings=tuple(warnings),
            context=context,
            config=self._config,
        )

        try:
            audit_result = await self._auditor.persist_query_audit_async(
                audit,
                context=context,
                raise_on_failure=False,
            )
            if not audit_result.success and not audit_result.skipped:
                warnings.append("Persistenza audit query non completata.")
        except AuditIdentityError:
            raise
        except Exception as exc:  # audit infrastrutturale best effort
            logger.exception("Query audit failed | request_id=%s", context.request_id)
            warnings.append(f"Audit query degradato: {type(exc).__name__}.")

        audit_markdown_parts = [format_retrieval_audit_markdown(audit)]
        if direct_audit_markdown.strip():
            audit_markdown_parts.append(direct_audit_markdown.strip())
        if evaluation is not None:
            audit_markdown_parts.append(format_evaluation_audit_markdown(evaluation))

        return RagServiceResult(
            request_id=UUID(context.request_id),
            conversation_id=command.conversation_id,
            answer=final_answer,
            intent=route.intent,
            answer_mode=route.answer_mode,
            execution_mode=execution_mode,
            deterministic=route.deterministic or execution_mode == RagExecutionMode.GLOSSARY_DIRECT,
            sources=public_sources,
            retrieval=debug,
            evaluation=evaluation,
            audit_markdown="\n\n".join(
                part for part in audit_markdown_parts if part.strip()
            ),
            warnings=_unique_strings(warnings),
            model=model_name,
            corpus_version=self._config.corpus_version,
            elapsed_ms=elapsed_ms,
        )

    async def _try_glossary_direct(
        self,
        route: RoutingDecision,
        command: RagQueryCommand,
        context: TenantContext,
    ) -> DirectAnswer | None:
        if not route.glossary_candidate:
            return None

        method = getattr(self._retriever, "lookup_glossary", None)
        if method is None:
            return None

        try:
            result = method(query=command.query, tenant_context=context)
            if inspect.isawaitable(result):
                result = await result
            return _coerce_direct_answer(result)
        except RagServiceConfigurationError:
            # Il modulo retrieval non esiste ancora: la query continuerà nel
            # normale ramo RAG e produrrà l'errore configurativo solo quando il
            # retrieval documentale sarà effettivamente richiesto.
            return None

    async def _retrieve_and_rank(
        self,
        *,
        command: RagQueryCommand,
        route: RoutingDecision,
        context: TenantContext,
        debug: RetrievalDebug,
    ) -> tuple[tuple[SourceItem, ...], RetrievalDebug, tuple[str, ...]]:
        retrieve_started = time.perf_counter()

        method = getattr(self._retriever, "retrieve_candidates", None)
        if method is None:
            method = getattr(self._retriever, "retrieve", None)
        if method is None:
            raise RagServiceConfigurationError(
                "Il retriever configurato non implementa retrieve_candidates(...)"
            )

        try:
            result = method(
                query=route.retrieval_query,
                intent=route.intent,
                answer_mode=route.answer_mode,
                target_document=route.requested_document,
                target_pages=command.target_pages,
                wants_evidence=route.wants_evidence,
                graph_relation_mode=route.graph_search_mode,
                formula_mode=route.formula_strict_mode,
                exhaustive_formula_lookup=route.exhaustive_formula_lookup,
                tenant_context=context,
            )
            if inspect.isawaitable(result):
                result = await result
            candidates, retrieved_debug = _validate_retrieval_result(result)
        except (RagServiceConfigurationError, TenantContextError):
            raise
        except Exception as exc:
            raise RagServiceRetrievalError(
                f"Errore retrieval: {type(exc).__name__}: {exc}"
            ) from exc

        # Il retrieval può valorizzare le metriche backend-specifiche. Il service
        # garantisce comunque coerenza con query/routing della richiesta.
        debug = retrieved_debug.model_copy(deep=True)
        debug.query = command.query
        debug.intent = route.intent
        debug.answer_mode = route.answer_mode
        debug.wants_evidence = route.wants_evidence
        debug.target_document = route.requested_document
        debug.target_pages = command.target_pages
        debug.kept_after_quality_filters = max(
            debug.kept_after_quality_filters,
            len(candidates),
        )
        debug.timings_ms.setdefault(
            "retrieval",
            int(round((time.perf_counter() - retrieve_started) * 1000.0)),
        )

        visible_candidates = filter_visible_records(
            list(candidates),
            context=context,
            allow_graph_tier=True,
            allow_user_tier=False,
        )
        if len(visible_candidates) != len(candidates):
            raise RagServiceValidationError(
                "Il retrieval ha restituito candidati fuori dal perimetro tenant."
            )

        if route.requested_document:
            visible_candidates = [
                candidate
                for candidate in visible_candidates
                if document_matches(candidate.filename, route.requested_document)
            ]

        if command.target_pages:
            page_set = set(command.target_pages)
            visible_candidates = [
                candidate for candidate in visible_candidates if candidate.page in page_set
            ]

        if not visible_candidates:
            debug.final_sources = 0
            return (), debug, ()

        reranking_engine = self._get_reranking_engine()
        ranking_context = RankingContext(
            query_text=command.query,
            intent=route.intent,
            wants_evidence=route.wants_evidence,
            formula_query=route.formula_strict_mode,
            requested_pages=command.target_pages,
            target_document=route.requested_document or "",
        )

        final_limit = min(
            int(command.max_sources or self._config.final_sources),
            int(self._config.final_sources),
        )

        try:
            reranked = await asyncio.to_thread(
                reranking_engine.rank,
                visible_candidates,
                context=ranking_context,
                exhaustive=route.exhaustive_formula_lookup,
                final_k=final_limit,
            )
        except Exception as exc:
            raise RagServiceRetrievalError(
                f"Errore reranking: {type(exc).__name__}: {exc}"
            ) from exc

        sources = tuple(
            candidate.to_source_item(request_id=context.request_id)
            for candidate in reranked.candidates
        )
        sources = _dedupe_sources(sources)

        debug.rerank_candidates = reranked.preselected_count
        debug.final_sources = len(sources)
        debug.reranker_used = reranked.reranker_used
        debug.timings_ms["rerank"] = int(round(reranked.elapsed_ms))
        debug.tier_counts = _tier_counts(sources)
        debug.set_score_values([source.score for source in sources])
        debug.warnings = _unique_strings((*debug.warnings, *reranked.warnings))

        return sources, debug, tuple(reranked.warnings)

    def _get_reranking_engine(self) -> RerankingEngine:
        if self._reranking_engine is not None:
            return self._reranking_engine

        try:
            reranker = self._resources.get_reranker()
        except ResourceNotReadyError as exc:
            raise RagServiceConfigurationError(
                "Le risorse RAG non sono inizializzate. Il lifespan FastAPI deve "
                "chiamare initialize_resources() prima di accettare query."
            ) from exc

        self._reranking_engine = RerankingEngine(
            config=self._config,
            reranker=reranker,
        )
        return self._reranking_engine

    async def _audit_failed_generation(
        self,
        *,
        command: RagQueryCommand,
        route: RoutingDecision,
        execution_mode: RagExecutionMode,
        debug: RetrievalDebug,
        sources: Sequence[SourceItem],
        prompt_bundle: PromptBundle,
        context: TenantContext,
        started: float,
        error: Exception,
    ) -> None:
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        audit = create_query_audit(
            query=command.query,
            sources=sources,
            intent=route.intent,
            answer_mode=route.answer_mode,
            execution_mode=execution_mode,
            retrieval=debug,
            filters={
                **self._audit_filters(command, route, blocked=False),
                "generation_failed": True,
                "generation_error_type": type(error).__name__,
            },
            prompt_sha256=prompt_bundle.prompt_sha256,
            context_chars=prompt_bundle.context.context_chars,
            deterministic=route.deterministic,
            llm_model=self._config.llm_model_name,
            elapsed_ms=elapsed_ms,
            warnings=(f"Generation failed: {type(error).__name__}",),
            context=context,
            config=self._config,
        )
        try:
            await self._auditor.persist_query_audit_async(
                audit,
                context=context,
                raise_on_failure=False,
            )
        except Exception:
            logger.exception(
                "Failure audit could not be persisted | request_id=%s",
                context.request_id,
            )

    def _make_user_input_source(
        self,
        query: str,
        context: TenantContext,
    ) -> SourceItem:
        return SourceItem(
            id="user_input",
            content=query,
            filename="Input utente",
            page=0,
            page_chunk_index=0,
            doc_id="",
            type="text",
            score=1.0,
            tier="USER",
            scope="ACCOUNT",
            organization_id=context.organization_id,
            status="active",
            corpus_version=self._config.corpus_version,
            classification="internal",
            request_id=context.request_id,
            db_origin="USER_INPUT",
        )

    @staticmethod
    def _audit_filters(
        command: RagQueryCommand,
        route: RoutingDecision,
        blocked: bool,
    ) -> dict[str, Any]:
        return {
            "target_document": route.requested_document or "",
            "target_pages": list(command.target_pages),
            "max_sources": command.max_sources,
            "include_evaluation": command.include_evaluation,
            "wants_evidence": route.wants_evidence,
            "calculation_mode": route.calculation_mode,
            "analytics_mode": route.analytics_mode,
            "strict_checklist_mode": route.strict_checklist_mode,
            "crosswalk_mode": route.crosswalk_mode,
            "graph_search_mode": route.graph_search_mode,
            "graph_relation_mode": route.graph_relation_mode,
            "formula_strict_mode": route.formula_strict_mode,
            "exhaustive_formula_lookup": route.exhaustive_formula_lookup,
            "quality_gate_blocked": blocked,
        }


# =============================================================================
# UTILITY DEL SERVICE
# =============================================================================
def _safe_document_name(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    if "\x00" in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError("target_document deve essere un nome file, non un percorso")
    return PurePath(cleaned).name


def _extract_requested_document(query: str) -> str | None:
    """Fallback compatibile col PoC quando il client non usa target_document."""

    match = re.search(
        r"(?:[\"“'«])([^\"”'»]+\.(?:pdf|md|txt|docx|html|csv|xlsx))(?:[\"”'»])",
        query or "",
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b([A-Za-z0-9À-ÿ][A-Za-z0-9À-ÿ._()\- ]{0,250}\.(?:pdf|md|txt|docx|html|csv|xlsx))\b",
            query or "",
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    try:
        return _safe_document_name(match.group(1)) or None
    except ValueError:
        return None


def _is_evidence_relevance_query(query: str) -> bool:
    """
    Rileva richieste di valutazione dell'attinenza o della sufficienza
    di un documento/evidenza rispetto a una domanda, requisito,
    controllo o item di assessment.

    La funzione non dipende da documenti, framework o test specifici.
    """

    q = str(query or "").casefold().strip()

    if not q:
        return False

    evidence_terms = (
        # Italiano
        "evidenza",
        "evidenze",
        "prova",
        "prove",
        "documento",
        "documenti",
        "file",
        "pdf",
        "allegato",
        "allegati",
        "upload",
        "caricato",
        "caricati",
        "documentazione",
        "artefatto",
        "artefatti",
        "record",
        "registrazione",
        "registrazioni",
        "log",
        "screenshot",
        "report",
        "rapporto",
        "procedura",
        "policy",
        "registro",

        # Inglese
        "evidence",
        "evidences",
        "proof",
        "proofs",
        "document",
        "documents",
        "file",
        "pdf",
        "attachment",
        "attachments",
        "upload",
        "uploaded",
        "documentation",
        "artifact",
        "artifacts",
        "record",
        "records",
        "log",
        "logs",
        "screenshot",
        "report",
        "reports",
        "procedure",
        "policy",
        "register",
    )

    assessment_terms = (
        # Italiano
        "domanda",
        "domande",
        "questionario",
        "questionari",
        "assessment",
        "audit",
        "requisito",
        "requisiti",
        "controllo",
        "controlli",
        "checklist",
        "item",
        "punto",
        "punti",
        "criterio",
        "criteri",
        "misura",
        "misure",
        "obbligo",
        "obblighi",
        "clausola",
        "clausole",
        "capitolo",
        "sezione",

        # Inglese
        "question",
        "questions",
        "questionnaire",
        "questionnaires",
        "assessment",
        "audit",
        "requirement",
        "requirements",
        "control",
        "controls",
        "checklist",
        "item",
        "items",
        "criterion",
        "criteria",
        "measure",
        "measures",
        "obligation",
        "obligations",
        "clause",
        "clauses",
        "chapter",
        "section",
    )

    relevance_terms = (
        # Italiano
        "attinente",
        "attinenza",
        "inerente",
        "inerenza",
        "pertinente",
        "pertinenza",
        "rilevante",
        "rilevanza",
        "correlato",
        "correlata",
        "correlazione",
        "collegato",
        "collegata",
        "collegamento",
        "coerente",
        "coerenza",
        "adeguato",
        "adeguata",
        "adeguatezza",
        "sufficiente",
        "sufficienza",
        "idoneo",
        "idonea",
        "idoneità",
        "applicabile",
        "applicabilità",
        "supporta",
        "supportato",
        "supportata",
        "dimostra",
        "dimostrato",
        "dimostrata",
        "comprova",
        "comprovato",
        "comprovata",
        "giustifica",
        "giustificato",
        "giustificata",
        "copre",
        "copertura",
        "risponde",
        "risposta",
        "valuta",
        "valutare",
        "verifica",
        "verificare",

        # Inglese
        "relevant",
        "relevance",
        "pertinent",
        "pertinence",
        "related",
        "relation",
        "relationship",
        "correlated",
        "correlation",
        "linked",
        "link",
        "connection",
        "consistent",
        "consistency",
        "adequate",
        "adequacy",
        "sufficient",
        "sufficiency",
        "suitable",
        "suitability",
        "applicable",
        "applicability",
        "supports",
        "supported",
        "supporting",
        "demonstrates",
        "demonstrated",
        "proves",
        "proven",
        "justifies",
        "justified",
        "covers",
        "coverage",
        "answers",
        "answer",
        "evaluate",
        "assess",
        "verify",
        "check",
    )

    gap_terms = (
        # Italiano
        "gap",
        "lacuna",
        "lacune",
        "mancanza",
        "mancanze",
        "manca",
        "mancano",
        "carente",
        "carenti",
        "carenza",
        "carenze",
        "debole",
        "debolezza",
        "debolezze",
        "incompleto",
        "incompleta",
        "parziale",
        "non sufficiente",
        "non adeguato",
        "non adeguata",
        "non attinente",
        "poco attinente",
        "scostamento",
        "scostamenti",
        "non conformità",
        "non conforme",
        "differenza",
        "differenze",
        "deviazione",
        "deviazioni",

        # Inglese
        "gap",
        "gaps",
        "missing",
        "absence",
        "lack",
        "lacks",
        "weak",
        "weakness",
        "weaknesses",
        "deficiency",
        "deficiencies",
        "incomplete",
        "partial",
        "not sufficient",
        "insufficient",
        "not adequate",
        "inadequate",
        "not relevant",
        "poorly relevant",
        "deviation",
        "deviations",
        "non-compliance",
        "non-compliant",
        "difference",
        "differences",
    )

    remediation_terms = (
        # Italiano
        "remediation",
        "piano di remediation",
        "piano correttivo",
        "azioni correttive",
        "azione correttiva",
        "correzione",
        "correzioni",
        "rimedio",
        "rimedi",
        "miglioramento",
        "miglioramenti",
        "integrazione",
        "integrare",
        "raccomandazione",
        "raccomandazioni",
        "cosa manca",
        "cosa integrare",
        "come migliorare",

        # Inglese
        "remediation",
        "remediation plan",
        "corrective action",
        "corrective actions",
        "correction",
        "corrections",
        "remedy",
        "remedies",
        "improvement",
        "improvements",
        "integration",
        "integrate",
        "recommendation",
        "recommendations",
        "what is missing",
        "what to add",
        "how to improve",
    )

    scoring_terms = (
        # Italiano
        "livello",
        "livelli",
        "percentuale",
        "percentuali",
        "score",
        "punteggio",
        "valutazione",
        "grado",
        "classifica",
        "classificazione",
        "basso",
        "medio",
        "alto",
        "debole",
        "parziale",
        "forte",

        # Inglese
        "level",
        "levels",
        "percentage",
        "percentages",
        "score",
        "scoring",
        "rating",
        "grade",
        "classification",
        "low",
        "medium",
        "high",
        "weak",
        "partial",
        "strong",
    )

    has_evidence = any(
        term in q
        for term in evidence_terms
    )

    has_assessment = any(
        term in q
        for term in assessment_terms
    )

    has_relevance = any(
        term in q
        for term in relevance_terms
    )

    has_gap = any(
        term in q
        for term in gap_terms
    )

    has_remediation = any(
        term in q
        for term in remediation_terms
    )

    has_scoring = any(
        term in q
        for term in scoring_terms
    )

    return (
        has_evidence
        and has_assessment
        and (
            has_relevance
            or has_gap
            or has_remediation
            or has_scoring
        )
    )


def _needs_math_document_context(query: str) -> bool:
    q = (query or "").casefold()
    terms = (
        "collega ai documenti", "collegalo ai documenti", "collegala ai documenti",
        "collega alle fonti", "usa le fonti recuperate", "usando le fonti recuperate",
        "secondo i documenti recuperati", "con evidenze documentali",
        "con supporto documentale", "giustifica con le fonti",
        "cita le fonti nel calcolo", "using retrieved sources",
        "according to retrieved documents", "with documentary evidence",
    )
    return any(term in q for term in terms)


def _is_user_data_analytics(query: str) -> bool:
    q = (query or "").casefold()
    has_array = bool(re.search(r"[\[(]\s*[\d,.\s-]{3,}\s*[\])]", q))
    has_md_table = bool(re.search(r"\|[^\n|]+\|[^\n|]+\|", q))
    has_json = bool(re.search(r'"[^"\n]+"\s*:\s*(?:\d|\[|\{)', q))
    has_csv = len(re.findall(r"(?:^|\n)[^\n,;\t]+[,;\t][^\n]+", q)) >= 3
    number_count = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", q))
    action_terms = (
        "analizza", "analyze", "analyse", "elabora", "calcola", "calculate",
        "raggruppa", "filtra", "ordina", "confronta", "compare", "valuta",
        "assess", "sintetizza", "summarize", "distribuzione", "distribution",
        "media", "average", "totale", "total", "trend",
    )
    return (has_array or has_md_table or has_json or has_csv or number_count >= 6) and any(
        term in q for term in action_terms
    )


def _is_graph_relation_query(query: str) -> bool:
    q = (query or "").casefold()
    explicit = (
        "neo4j", "cypher", "interroga neo4j", "usando neo4j", "query neo4j",
        "grafo", "graph", "relazioni esplicite", "explicit relationships",
        "archi", "edges", "nodi", "nodes", "traversamento", "traversal",
        "multi-hop", "path", "percorso", "catena semantica", "semantic chain",
        "tabella relazioni", "relationship table",
    )
    return any(term in q for term in explicit)




def _should_use_graph_relation_strict_mode(query: str) -> bool:
    """
    Decide se una query sul Knowledge Graph richiede una risposta
    deterministica e tabellare.

    Le normali domande esplicative possono usare il grafo come supporto,
    senza essere trasformate automaticamente in tabelle di archi.
    """

    q = str(query or "").casefold().strip()

    if not q:
        return False

    explicit_graph_terms = (
        # Italiano
        "neo4j",
        "cypher",
        "interroga neo4j",
        "usando neo4j",
        "query neo4j",
        "archi",
        "arco",
        "nodi",
        "nodo",
        "path",
        "percorso",
        "travers4j",
        "archi",
        "arco",
        "nodi",
        "nodo",
        "path",
        "percorso",
        "traversamento",
        "multi-hop",
        "catena semantica",
        "relazioni esplicite",
        "relazioni nel grafo",
        "tabella relazioni",

        # Inglese
        "graph query",
        "query neo4j",
        "using neo4j",
        "cypher query",
        "nodes",
        "node",
        "edges",
        "edge",
        "path",
        "traversal",
        "multi-hop",
        "semantic chain",
        "explicit relationships",
        "relationship table",
    )

    if any(term in q for term in explicit_graph_terms):
        return True

    explanatory_terms = (
        # Italiano
        "qual è",
        "quale è",
        "quali sono",
        "che cosa",
        "cosa significa",
        "ruolo",
        "scopo",
        "funzione",
        "descrivi",
        "spiega",
        "analizza",
        "valuta",
        "giustifica",
        "perché",
        "perche",
        "in che modo",
        "come funziona",
        "elabora",
        "sintetizza",

        # Inglese
        "what is",
        "what are",
        "role",
        "purpose",
        "function",
        "describe",
        "explain",
        "analyze",
        "analyse",
        "evaluate",
        "justify",
        "why",
        "how does",
        "how do",
        "summarize",
        "summarise",
    )

    if any(term in q for term in explanatory_terms):
        return False

    strong_relation_terms = (
        # Italiano
        "relazioni tra",
        "collegamenti tra",
        "mostra le relazioni",
        "traccia la catena",
        "connessioni",
        "mappa",
        "mappatura",
        "rete semantica",
        "triple",

        # Inglese
        "relations between",
        "links between",
        "show relationships",
        "trace the chain",
        "connections",
        "mapping",
        "semantic network",
        "triples",
    )
    return any(term in q for term in strong_relation_terms)



def _is_formula_lookup_query(query: str) -> bool:
    """
    Riconosce richieste documentali di recupero di formule, equazioni,
    metriche, indicatori o regole di scoring.

    Non intercetta i calcoli operativi, che devono essere gestiti dai
    solver deterministici o dal normale Calculation Mode.
    """

    q = str(query or "").casefold().strip()

    if not q:
        return False

    # Un calcolo richiesto dall'utente non è automaticamente una ricerca
    # documentale di formule.
    if is_calculation_request(query):
        return False

    has_formula_object = bool(
        re.search(
            r"\b(?:"
            r"formula|formule|"
            r"equazione|equazioni|"
            r"metrica|metriche|"
            r"indicatore|indicatori|"
            r"modello\s+matematico|modelli\s+matematici|"
            r"regola\s+di\s+scoring|regole\s+di\s+scoring|"
            r"formulas?|"
            r"equations?|"
            r"metrics?|"
            r"indicators?|"
            r"mathematical\s+models?|"
            r"scoring\s+rules?"
            r")\b",
            q,
        )
    )

    has_lookup_action = bool(
        re.search(
            r"\b(?:"
            r"elenca|estrai|mostra|riporta|fornisci|dammi|"
            r"indica|trova|quale|quali|"
            r"list|extract|show|report|provide|give|"
            r"identify|find|which"
            r")\b",
            q,
        )
    )

    has_source_or_collection_cue = bool(
        re.search(
            r"\b(?:"
            r"documento|documenti|"
            r"fonte|fonti|"
            r"testo|corpus|"
            r"presente|presenti|"
            r"menzionata|menzionate|"
            r"citata|citate|"
            r"contenuta|contenute|"
            r"definita|definite|"
            r"tutte|tutti|"
            r"document|documents|"
            r"source|sources|"
            r"text|corpus|"
            r"present|mentioned|cited|contained|defined|all"
            r")\b",
            q,
        )
    )

    return has_formula_object and (
        has_lookup_action
        or has_source_or_collection_cue
    )




def _is_formula_strict_query(query: str) -> bool:
    q = (query or "").casefold()
    explicit = (
        "formula", "formule", "equazione", "equazioni", "disequazione",
        "disequazioni", "algebra", "algebrica", "esprimi", "isola",
        "in funzione di", "risolvi", "deriva", "equation", "inequality",
        "algebraic", "solve", "derive", "express", "as a function of",
    )
    if any(term in q for term in explicit):
        return True
    has_symbols = bool(re.search(r"(?:<=|>=|≤|≥|=|>|<|\\times|×|\*|/|\\frac|%)", query or ""))
    operational = (
        "calcola", "risolvi", "scrivi", "esprimi", "isola", "determina",
        "calculate", "compute", "solve", "express", "determine",
    )
    return has_symbols and any(term in q for term in operational)


def _is_strict_checklist_query(query: str) -> bool:
    q = (query or "").casefold()
    strong = ("checklist", "crosswalk", "matrice", "matrix", "griglia", "grid")
    if any(term in q for term in strong):
        return True
    weak = (
        "assessment", "evidenza", "evidence", "controllo", "control",
        "requisito", "requirement", "audit", "linee guida", "guidelines",
        "elenco", "lista", "list", "kpi", "indicatore", "indicator",
        "questionario", "questionnaire",
    )
    return sum(1 for term in weak if term in q) >= 2


def _is_crosswalk_query(query: str) -> bool:
    q = (query or "").casefold()
    terms = (
        "crosswalk", "mappatura", "mapping", "matrice di conformità",
        "compliance matrix", "allineamento controlli", "control alignment",
        "correlazione requisiti", "requirements mapping",
    )
    return any(term in q for term in terms)


def _is_mixed_glossary_query(query: str) -> bool:
    q = (query or "").casefold()
    glossary = ("glossario", "glossary", "voce di glossario", "glossary entry")
    if not any(term in q for term in glossary):
        return False
    mixed = (
        "documenti", "documents", "fonti", "sources", "normativa", "regulation",
        "evidenze", "evidence", "audit", "assessment", "grafo", "graph",
        "relazioni", "relationships", "controlli", "controls", "requisiti",
        "requirements", "usa sia", "using both",
    )
    return any(term in q for term in mixed)


def _is_exhaustive_formula_lookup(query: str) -> bool:
    q = (query or "").casefold()
    exhaustive = (
        "tutte le formule", "tutte le equazioni", "elenca tutte le formule",
        "estrai tutte le formule", "all formulas", "all equations",
        "list all formulas", "extract all formulas",
    )
    return any(term in q for term in exhaustive)


def _build_retrieval_query(
    query: str,
    *,
    evidence_relevance: bool,
    math_needs_context: bool,
    graph_relation_mode: bool,
    formula_mode: bool,
) -> str:
    additions: list[str] = []
    if evidence_relevance:
        additions.append(
            "evidence relevance assessment question requirement control "
            "attinenza evidenza domanda questionario requisito controllo "
            "gap remediation corrective action sufficiency adequacy"
        )
    if math_needs_context:
        additions.append(
            "risk assessment evidence assessment valutazione del rischio "
            "controlli evidenze assessment integrato"
        )
    if graph_relation_mode:
        additions.append(
            "entity relationship graph edge node semantic relation "
            "entità relazione grafo arco nodo"
        )
    if formula_mode:
        additions.append(
            "formula equation metric scoring rule mathematical model "
            "equazione metrica regola di scoring modello matematico"
        )
    if not additions:
        return query
    return query.rstrip() + "\n" + "\n".join(additions)


def _validate_retrieval_result(
    result: Any,
) -> tuple[Sequence[RetrievalCandidate], RetrievalDebug]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RagServiceRetrievalError(
            "Il RetrievalPort deve restituire (candidates, RetrievalDebug)"
        )
    candidates_raw, debug_raw = result
    if not isinstance(debug_raw, RetrievalDebug):
        raise RagServiceRetrievalError(
            "Il secondo valore del RetrievalPort deve essere RetrievalDebug"
        )

    candidates: list[RetrievalCandidate] = []
    for item in candidates_raw or ():
        if isinstance(item, RetrievalCandidate):
            candidates.append(item)
        elif isinstance(item, Mapping):
            candidates.append(RetrievalCandidate.model_validate(dict(item)))
        else:
            raise RagServiceRetrievalError(
                "Il RetrievalPort ha restituito un candidato non compatibile"
            )
    return tuple(candidates), debug_raw


def _coerce_direct_answer(value: Any) -> DirectAnswer | None:
    if value is None:
        return None
    if isinstance(value, DirectAnswer):
        return value
    if isinstance(value, tuple) and len(value) in {2, 3}:
        answer = str(value[0] or "").strip()
        sources = tuple(value[1] or ())
        audit_md = str(value[2] or "") if len(value) == 3 else ""
        if not answer:
            return None
        return DirectAnswer(answer=answer, sources=sources, audit_markdown=audit_md)
    if isinstance(value, Mapping):
        return DirectAnswer(
            answer=str(value.get("answer") or ""),
            sources=tuple(value.get("sources") or ()),
            execution_mode=RagExecutionMode(
                str(value.get("execution_mode") or RagExecutionMode.GLOSSARY_DIRECT)
            ),
            audit_markdown=str(value.get("audit_markdown") or ""),
            warnings=tuple(value.get("warnings") or ()),
        )
    raise RagServiceConfigurationError(
        "lookup_glossary ha restituito un valore non compatibile con DirectAnswer"
    )


def _build_math_answer_with_document_context(
    math_answer: str,
    sources: Sequence[SourceItem],
    max_items: int = 3,
) -> str:
    """
    Integra un risultato matematico deterministico con fonti documentali.

    Il risultato del solver resta autoritativo.
    Le fonti vengono usate esclusivamente per contestualizzare il risultato
    e non possono modificarne valori, formule o conclusioni numeriche.
    """

    answer = str(math_answer or "").strip()

    if not answer:
        return ""

    if not sources:
        return answer

    clean_sources: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()

    for source in sources:
        tier = str(source.tier or "").strip().upper()
        source_type = str(source.type or "").strip().casefold()

        # Le righe sintetiche di grafo o formula non costituiscono
        # contesto documentale concettuale principale.
        if tier == "GRAPH":
            continue

        if source_type in {
            "graph",
            "graph_relations",
            "formula",
        }:
            continue

        filename = str(source.filename or "").strip() or "N/D"
        page = int(source.page or 0)
        content = re.sub(
            r"\s+",
            " ",
            str(source.content or ""),
        ).strip()

        if not content:
            continue

        source_key = (
            PurePath(filename).name.casefold(),
            page,
        )

        if source_key in seen:
            continue

        seen.add(source_key)
        clean_sources.append(
            (
                filename,
                page,
                content,
            )
        )

        if len(clean_sources) >= max(1, int(max_items)):
            break

    if not clean_sources:
        return answer

    context_lines = [
        "Collegamento documentale",
        "",
        "- Il risultato numerico è calcolato esclusivamente sui dati "
        "forniti dall'utente.",
        "- Le fonti recuperate vengono usate soltanto per contestualizzare "
        "il risultato nel risk/evidence/control assessment; non modificano "
        "il calcolo.",
    ]

    for filename, page, content in clean_sources:
        snippet = content[:360].rstrip()

        if len(content) > 360:
            snippet += "..."

        context_lines.append(
            f"- `{filename}` (p.{page}): {snippet}"
        )

    context_block = "\n".join(context_lines)

    used_files: list[str] = []

    for filename, _, _ in clean_sources:
        if filename not in used_files:
            used_files.append(filename)

    additional_sources = "\n".join(
        f"- {filename}"
        for filename in used_files
    )

    sources_marker = "**D) Fonti**"

    if sources_marker in answer:
        before_sources, existing_sources = answer.split(
            sources_marker,
            1,
        )

        existing_sources = existing_sources.strip()

        if additional_sources:
            existing_sources = (
                existing_sources
                + "\n"
                + additional_sources
            )

        return (
            before_sources.rstrip()
            + "\n\n"
            + context_block
            + "\n\n"
            + sources_marker
            + "\n\n"
            + existing_sources
        )

    return (
        answer.rstrip()
        + "\n\n"
        + context_block
        + "\n\n"
        + sources_marker
        + "\n\n"
        + additional_sources
    )



def _dedupe_sources(sources: Sequence[SourceItem]) -> tuple[SourceItem, ...]:
    out: list[SourceItem] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for source in sources:
        key = (
            source.id,
            source.filename.casefold(),
            int(source.page),
            int(source.page_chunk_index),
            source.type,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return tuple(out)


def _tier_counts(sources: Sequence[SourceItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        tier = str(source.tier).upper()
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value or "").strip()
        )
    )


def _no_sources_fallback(requested_document: str | None) -> str:
    scope = (
        f" nel documento `{requested_document}`"
        if requested_document
        else " nei documenti recuperabili"
    )
    return (
        "**A) Risposta**\n\n"
        f"Non ho trovato evidenze sufficienti{scope} per formulare una risposta documentale affidabile.\n\n"
        "**B) Evidenze**\n\n"
        "- Nessuna fonte tenant-visible ha superato i filtri di retrieval e document scope.\n\n"
        "**C) Limiti / Conflitti**\n\n"
        "- Non vengono introdotte informazioni esterne o non recuperate.\n"
        "- Verificare che il documento sia stato ingerito, attivo e assegnato al TIER/scope corretto.\n\n"
        "**D) Fonti**\n\n"
        "- Nessuna fonte recuperata."
    )


# Singleton senza side effect.  Il retrieval concreto viene risolto solo quando
# una query documentale viene eseguita.
rag_service = RagService()


__all__ = [
    "DirectAnswer",
    "GlossaryPort",
    "LazyRetrievalAdapter",
    "RagQueryCommand",
    "RagQueryRouter",
    "RagService",
    "RagServiceConfigurationError",
    "RagServiceError",
    "RagServiceGenerationError",
    "RagServiceRetrievalError",
    "RagServiceValidationError",
    "RetrievalPort",
    "RoutingDecision",
    "rag_service",
]
