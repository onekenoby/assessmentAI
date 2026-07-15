"""Validazione e quality gate delle risposte prodotte dal motore RAG.

Il modulo è indipendente da FastAPI e Reflex. Opera dopo la generazione e prima
che il risultato venga restituito dal service layer.

Responsabilità:

- rimozione di reasoning, identificativi e metadati tecnici eventualmente
  ripetuti dal modello;
- normalizzazione della struttura Markdown A/B/C/D;
- ricostruzione deterministica della sezione ``D) Fonti`` usando esclusivamente
  i ``SourceItem`` realmente recuperati;
- controllo fail-closed della visibilità tenant delle fonti;
- enforcement dell'eventuale document scope;
- controlli specifici per ``evidence_relevance``;
- rimozione di URL esterni non autorizzati;
- verifica di formule visibili nei rami formula;
- valutazione opzionale di faithfulness tramite un judge LLM separato.

Il quality gate può correggere soltanto difetti formali sicuri. Non riscrive la
risposta per migliorarne il contenuto e non inventa evidenze mancanti.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.config import RagSettings, settings
from core.generation import (
    GenerationError,
    GenerationOptions,
    OllamaNativeGenerator,
    generator,
)
from core.models import (
    EvaluationVerdict,
    RagAnswerMode,
    RagEvalResult,
    RagExecutionMode,
    RagIntent,
    SourceItem,
)
from core.prompting import (
    PromptMessage,
    document_matches,
    truncate_structured_content,
)
from core.tenant import (
    TenantContext,
    TenantContextError,
    filter_visible_records,
    get_tenant_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COSTANTI
# =============================================================================
_REQUIRED_HEADERS: Final[tuple[str, str, str, str]] = (
    "**A) Risposta**",
    "**B) Evidenze**",
    "**C) Limiti / Conflitti**",
    "**D) Fonti**",
)

_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?P<letter>[ABCD])\s*[\)\.\-:]\s*"
    r"(?P<label>"
    r"Risposta|Answer|"
    r"Evidenze|Evidence|Evidences|"
    r"Limiti\s*/\s*Conflitti|Limitations\s*/\s*Conflicts|Limiti|Conflitti|Limitations|Conflicts|"
    r"Fonti|Sources|Riferimenti|References"
    r")\s*(?:\*\*)?\s*$"
)

_EXTERNAL_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s<>()\[\]{}]+",
    flags=re.IGNORECASE,
)

_UUID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)

_FILE_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<quoted>[`\"'“‘«][^`\"'”’»\n]{1,240}?\.(?:pdf|md|txt|docx|html|csv|xlsx|log|json|ya?ml|xml)[`\"'”’»])"
    r"|(?P<bare>\b[A-Za-z0-9À-ÿ_()\-]+(?:\.[A-Za-z0-9À-ÿ_()\-]+)*\.(?:pdf|md|txt|docx|html|csv|xlsx|log|json|ya?ml|xml)\b)",
    flags=re.IGNORECASE,
)

_UNRETRIEVED_SOURCE_MARKER: Final[str] = "[riferimento a fonte non recuperata rimosso]"
_SYNTHETIC_GRAPH_FILENAMES: Final[frozenset[str]] = frozenset({
    "kg",
    "neo4j knowledge graph",
})
_FINAL_TRUNCATION_MARKER: Final[str] = "...[contenuto troncato dal quality gate]"

_EVIDENCE_TABLE_HEADER: Final[str] = (
    "| Livello di attinenza | Percentuale stimata | Esito sintetico |"
)
_EVIDENCE_TABLE_SEPARATOR: Final[str] = "|---:|---:|---|"

_RELEVANCE_LABELS: Final[dict[int, str]] = {
    0: "Non attinente",
    1: "Debolmente attinente",
    2: "Parzialmente attinente",
    3: "Fortemente attinente",
}

_RELEVANCE_BANDS: Final[dict[int, tuple[float, float]]] = {
    0: (0.0, 25.0),
    1: (26.0, 50.0),
    2: (51.0, 75.0),
    3: (76.0, 100.0),
}


# =============================================================================
# MODELLI DEL QUALITY GATE
# =============================================================================
class ValidationSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationCode(StrEnum):
    EMPTY_ANSWER = "EMPTY_ANSWER"
    CONTROL_CHARACTERS_REMOVED = "CONTROL_CHARACTERS_REMOVED"
    INTERNAL_METADATA_REMOVED = "INTERNAL_METADATA_REMOVED"
    EXTERNAL_URL_REMOVED = "EXTERNAL_URL_REMOVED"
    STRUCTURE_NORMALIZED = "STRUCTURE_NORMALIZED"
    STRUCTURE_REPAIRED = "STRUCTURE_REPAIRED"
    DUPLICATE_SECTION = "DUPLICATE_SECTION"
    SOURCES_SECTION_REBUILT = "SOURCES_SECTION_REBUILT"
    MISSING_RETRIEVED_SOURCES = "MISSING_RETRIEVED_SOURCES"
    SOURCE_CITATION_MISSING = "SOURCE_CITATION_MISSING"
    UNRETRIEVED_SOURCE_REFERENCE_REMOVED = "UNRETRIEVED_SOURCE_REFERENCE_REMOVED"
    TENANT_SOURCE_VIOLATION = "TENANT_SOURCE_VIOLATION"
    TENANT_CONTEXT_MISSING = "TENANT_CONTEXT_MISSING"
    REQUESTED_DOCUMENT_SCOPE_VIOLATION = "REQUESTED_DOCUMENT_SCOPE_VIOLATION"
    REQUESTED_DOCUMENT_NOT_FOUND = "REQUESTED_DOCUMENT_NOT_FOUND"
    EVIDENCE_TABLE_MISSING = "EVIDENCE_TABLE_MISSING"
    EVIDENCE_TABLE_REPAIRED = "EVIDENCE_TABLE_REPAIRED"
    EVIDENCE_LEVEL_INVALID = "EVIDENCE_LEVEL_INVALID"
    EVIDENCE_PERCENTAGE_INVALID = "EVIDENCE_PERCENTAGE_INVALID"
    EVIDENCE_SCORE_MISMATCH = "EVIDENCE_SCORE_MISMATCH"
    EVIDENCE_REMEDIATION_MISSING = "EVIDENCE_REMEDIATION_MISSING"
    FORMULA_NOT_VISIBLE = "FORMULA_NOT_VISIBLE"
    GRAPH_WORDING_REPAIRED = "GRAPH_WORDING_REPAIRED"
    ANSWER_TOO_LONG = "ANSWER_TOO_LONG"
    ANSWER_TRUNCATED = "ANSWER_TRUNCATED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    EVALUATION_BLOCKED = "EVALUATION_BLOCKED"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )

    code: ValidationCode
    severity: ValidationSeverity
    message: str = Field(min_length=1, max_length=2_000)
    repaired: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class AnswerSections(BaseModel):
    """Rappresentazione canonica delle quattro sezioni della risposta."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_default=True,
    )

    answer: str = ""
    evidence: str = ""
    limitations: str = ""
    sources: str = ""

    def render(self) -> str:
        return (
            f"{_REQUIRED_HEADERS[0]}\n\n{self.answer.strip()}\n\n"
            f"{_REQUIRED_HEADERS[1]}\n\n{self.evidence.strip()}\n\n"
            f"{_REQUIRED_HEADERS[2]}\n\n{self.limitations.strip()}\n\n"
            f"{_REQUIRED_HEADERS[3]}\n\n{self.sources.strip()}"
        ).strip()


class ValidationResult(BaseModel):
    """Esito del quality gate deterministico."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_default=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    answer: str = Field(min_length=1)
    valid: bool = True
    blocked: bool = False
    repaired: bool = False
    issues: tuple[ValidationIssue, ...] = Field(default_factory=tuple)
    visible_sources: tuple[SourceItem, ...] = Field(default_factory=tuple)
    sections: AnswerSections
    evaluation: RagEvalResult | None = None

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            issue.message
            for issue in self.issues
            if issue.severity in {
                ValidationSeverity.WARNING,
                ValidationSeverity.ERROR,
                ValidationSeverity.CRITICAL,
            }
        )


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Policy applicata a una singola risposta.

    Il service layer deve valorizzare esplicitamente le modalità già rilevate.
    ``validation.py`` non replica il router di intenti.
    """

    intent: RagIntent | str = RagIntent.TEXT
    answer_mode: RagAnswerMode | str = RagAnswerMode.KNOWLEDGE
    execution_mode: RagExecutionMode | str = RagExecutionMode.RAG_GENERATION

    requested_document: str | None = None
    require_sources: bool = False
    rebuild_sources_section: bool = True
    enforce_requested_document: bool = True

    allow_external_urls: bool = False
    allow_graph_tier: bool = True
    allow_user_tier: bool = True

    repair_structure: bool = True
    require_evidence_table: bool = True
    block_on_tenant_violation: bool = True
    block_on_document_scope_violation: bool = True

    max_answer_chars: int | None = None

    def resolved_intent(self) -> str:
        return str(self.intent)

    def resolved_answer_mode(self) -> str:
        return str(self.answer_mode)

    def resolved_execution_mode(self) -> str:
        return str(self.execution_mode)


# =============================================================================
# UTILITY GENERALI
# =============================================================================
def _issue(
    code: ValidationCode,
    severity: ValidationSeverity,
    message: str,
    *,
    repaired: bool = False,
    **details: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        repaired=repaired,
        details=details,
    )


def _normalize_newlines(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n")


def _remove_control_characters(value: str) -> tuple[str, int]:
    text = _normalize_newlines(value)
    cleaned, count = re.subn(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return cleaned, count


def strip_internal_metadata(value: str) -> tuple[str, int]:
    """Rimuove metadati tecnici eventualmente ripetuti dall'LLM.

    Il testo del reasoning viene sempre eliminato e non viene mai usato come
    fallback della risposta finale.
    """

    text = value or ""
    removed = 0

    patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"<(?:reasoning|thinking|think)>.*?</(?:reasoning|thinking|think)>",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "",
        ),
        (
            re.compile(r"</?(?:reasoning|thinking|think)>", flags=re.IGNORECASE),
            "",
        ),
        (
            re.compile(r"\[SourceID:\s*[^\]]+\]", flags=re.IGNORECASE),
            "",
        ),
        (
            re.compile(r"(?im)^\s*>>>\s*SOURCE\s*\[[^\]]+\].*$"),
            "",
        ),
        (
            re.compile(
                r"(?im)^\s*(?:chunk_id|chunk_uuid|doc_id|ingestion_run_id|"
                r"pg_chunk_id|pg_log_id|organization_id|tenant_key|request_id)\s*[:=].*$"
            ),
            "",
        ),
        (
            re.compile(r"(?im)^\s*Tier\s*:\s*[ABC]\s*$"),
            "",
        ),
        (
            _UUID_PATTERN,
            "[identificativo tecnico rimosso]",
        ),
    )

    for pattern, replacement in patterns:
        text, count = pattern.subn(replacement, text)
        removed += count

    # Percorsi locali del container/host non devono essere restituiti al client.
    local_path_pattern = re.compile(
        r"(?<![\w])(?:/workspace|/mnt/data|[A-Za-z]:\\)[^\s`\]\[(){}<>]+",
        flags=re.IGNORECASE,
    )
    text, count = local_path_pattern.subn("[percorso interno rimosso]", text)
    removed += count

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, removed


def _remove_external_urls(value: str) -> tuple[str, int]:
    cleaned, count = _EXTERNAL_URL_PATTERN.subn(
        "[Link esterno non autorizzato rimosso]",
        value or "",
    )
    return cleaned, count


def _header_key(letter: str) -> str:
    return {
        "A": "answer",
        "B": "evidence",
        "C": "limitations",
        "D": "sources",
    }[letter.upper()]


def parse_answer_sections(value: str) -> tuple[AnswerSections, tuple[str, ...]]:
    """Estrae le sezioni riconosciute senza modificarne il contenuto.

    Restituisce anche le lettere duplicate, utili per il report di validazione.
    """

    text = _normalize_newlines(value).strip()
    matches = list(_HEADER_PATTERN.finditer(text))

    if not matches:
        return AnswerSections(answer=text), ()

    contents: dict[str, str] = {
        "answer": "",
        "evidence": "",
        "limitations": "",
        "sources": "",
    }
    duplicates: list[str] = []

    prefix = text[: matches[0].start()].strip()
    if prefix:
        contents["answer"] = prefix

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        letter = match.group("letter").upper()
        key = _header_key(letter)
        body = text[start:end].strip()

        if contents[key]:
            duplicates.append(letter)
            contents[key] = (contents[key].rstrip() + "\n\n" + body).strip()
        else:
            contents[key] = body

    return AnswerSections(**contents), tuple(dict.fromkeys(duplicates))


def _complete_sections(sections: AnswerSections) -> AnswerSections:
    return AnswerSections(
        answer=sections.answer.strip() or "Risposta non disponibile.",
        evidence=sections.evidence.strip() or "- Nessuna evidenza esplicitata nella risposta.",
        limitations=(
            sections.limitations.strip()
            or "- Nessun limite esplicitato nella risposta generata."
        ),
        sources=sections.sources.strip() or "- Nessuna fonte disponibile.",
    )


def _has_complete_structure(value: str) -> bool:
    letters = [match.group("letter").upper() for match in _HEADER_PATTERN.finditer(value or "")]
    return all(letter in letters for letter in ("A", "B", "C", "D"))


def _empty_answer_fallback(*, sources_available: bool) -> AnswerSections:
    return AnswerSections(
        answer="Il modello non ha restituito contenuto utile.",
        evidence=(
            "- Il retrieval ha prodotto fonti, ma la generazione non ha restituito una risposta finale."
            if sources_available
            else "- Non sono disponibili evidenze recuperate per questa richiesta."
        ),
        limitations=(
            "- La risposta non è stata generata correttamente."
            "\n- Consultare l'audit tecnico della richiesta senza esporre dettagli interni al client."
        ),
        sources="- Nessuna fonte disponibile.",
    )


def _tenant_violation_fallback() -> AnswerSections:
    return AnswerSections(
        answer="La richiesta non può essere completata perché il perimetro delle fonti non è valido.",
        evidence="- Nessuna evidenza è stata restituita.",
        limitations=(
            "- Il quality gate ha rilevato fonti non compatibili con il contesto tenant della richiesta."
        ),
        sources="- Nessuna fonte disponibile.",
    )


def _document_not_found_fallback(requested_document: str) -> AnswerSections:
    safe_name = requested_document.strip() or "documento richiesto"
    return AnswerSections(
        answer=(
            f"Non ho trovato evidenze sufficienti nel documento `{safe_name}` tra le fonti recuperate."
        ),
        evidence=f"- Nessun chunk tenant-visible recuperato per `{safe_name}`.",
        limitations=(
            "- Non è possibile valutare il contenuto del documento senza evidenze recuperate dal documento stesso."
            "\n- Non sono state usate fonti alternative fuori scope."
        ),
        sources="- Nessuna fonte disponibile per il documento richiesto.",
    )


def _normalized_source_filename(value: str) -> str:
    """Return a single-line Markdown-safe source label."""

    filename = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    filename = re.sub(r"\s{2,}", " ", filename)
    filename = filename.replace("`", "'")
    for character in ("|", "*", "#", ">"):
        filename = filename.replace(character, f"\\{character}")
    return filename


def _is_synthetic_graph_filename(value: str) -> bool:
    return _normalized_source_filename(value).casefold() in _SYNTHETIC_GRAPH_FILENAMES


def _citable_source_filenames(sources: Sequence[SourceItem]) -> tuple[str, ...]:
    filenames: list[str] = []
    for source in sources:
        filename = _normalized_source_filename(source.filename)
        if not filename or _is_synthetic_graph_filename(filename):
            continue
        if filename.casefold() not in {item.casefold() for item in filenames}:
            filenames.append(filename)
    return tuple(filenames)


def _compact_sources(sources: Sequence[SourceItem], max_sources: int = 20) -> str:
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    graph_evidence_used = False

    for source in sources:
        filename = _normalized_source_filename(source.filename)
        if not filename:
            continue
        if _is_synthetic_graph_filename(filename):
            graph_evidence_used = True
            continue

        page = int(source.page or 0)
        key = (filename.casefold(), page)
        if key in seen:
            continue
        seen.add(key)

        page_text = f" (p.{page})" if page > 0 else ""
        lines.append(f"- {filename}{page_text}")
        if len(lines) >= max_sources:
            break

    if lines:
        if graph_evidence_used and len(lines) < max_sources:
            lines.append("- Knowledge Graph Neo4j (evidenza strutturale)")
        return "\n".join(lines)
    if graph_evidence_used:
        return "- Knowledge Graph Neo4j (evidenza strutturale)"
    return "- Nessuna fonte disponibile."


def _answer_mentions_any_source(answer: str, sources: Sequence[SourceItem]) -> bool:
    answer_lower = re.sub(r"\s+", " ", answer or "").casefold()
    filenames = _citable_source_filenames(sources)
    if not filenames:
        return any(
            _is_synthetic_graph_filename(source.filename)
            for source in sources
        ) and "neo4j" in answer_lower
    return any(filename.casefold() in answer_lower for filename in filenames)


def _remove_unretrieved_file_references(
    sections: AnswerSections,
    sources: Sequence[SourceItem],
) -> tuple[AnswerSections, tuple[str, ...]]:
    """Redact explicit filenames that are not present in the visible source set.

    The deterministic D section is rebuilt separately. This guard only inspects
    A/B/C so an LLM cannot retain a plausible but unretrieved filename as
    evidence or as a factual limitation.
    """

    removed: list[str] = []

    def clean_text(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            raw = match.group("quoted") or match.group("bare") or ""
            reference = raw.strip("`\"'“”‘’«» ")
            if any(document_matches(source.filename, reference) for source in sources):
                return raw
            removed.append(reference)
            return _UNRETRIEVED_SOURCE_MARKER

        return _FILE_REFERENCE_PATTERN.sub(replace, value or "")

    cleaned = sections.model_copy(
        update={
            "answer": clean_text(sections.answer),
            "evidence": clean_text(sections.evidence),
            "limitations": clean_text(sections.limitations),
        }
    )
    return cleaned, tuple(dict.fromkeys(item for item in removed if item))


def _has_visible_formula(answer: str) -> bool:
    text = answer or ""
    return bool(
        re.search(r"[A-Za-z0-9_)]\s*(?:=|>|<|≤|≥|×|\*|/)\s*[A-Za-z0-9_(]", text)
        or re.search(r"\$\$.*?\$\$", text, flags=re.DOTALL)
        or re.search(r"(?<!\$)\$[^\n$]+\$(?!\$)", text)
        or re.search(r"\\frac\{|\\sum|\\prod|\\sqrt", text)
    )


def _repair_graph_wording(value: str) -> tuple[str, int]:
    text = value or ""
    phrases = (
        "Poiché non è stato fornito un grafo Neo4j",
        "poiché non è stato fornito un grafo Neo4j",
        "non è stato fornito un grafo Neo4j",
        "simulando una query Neo4j",
        "simulando Neo4j",
        "non contengono un grafo Neo4j preesistente",
        "assenza di un grafo Neo4j",
    )
    count = 0
    for phrase in phrases:
        if phrase in text:
            count += text.count(phrase)
            text = text.replace(
                phrase,
                "Non sono stati recuperati archi Neo4j espliciti sufficienti",
            )
    return text, count


def _truncate_sections(sections: AnswerSections, max_chars: int) -> AnswerSections:
    """Truncate the final answer while preserving A/B/C/D and Markdown blocks."""

    rendered = sections.render()
    if len(rendered) <= max_chars:
        return sections

    empty_render = AnswerSections().render()
    available = max(0, int(max_chars) - len(empty_render))
    if available <= 0:
        return AnswerSections(
            answer="Risposta troncata.",
            evidence="- Evidenze non disponibili nel budget residuo.",
            limitations="- Limite massimo della risposta raggiunto.",
            sources="- Nessuna fonte visualizzabile nel budget residuo.",
        )

    bodies = (
        sections.answer,
        sections.evidence,
        sections.limitations,
        sections.sources,
    )
    weights = (0.48, 0.24, 0.16, 0.12)
    budgets = [max(1, int(available * weight)) for weight in weights]

    # Redistribute budget unused by short sections to longer sections.
    for _ in range(2):
        unused = 0
        needs: list[int] = []
        for index, body in enumerate(bodies):
            if len(body) < budgets[index]:
                unused += budgets[index] - len(body)
                budgets[index] = len(body)
            elif len(body) > budgets[index]:
                needs.append(index)
        if unused <= 0 or not needs:
            break
        share, remainder = divmod(unused, len(needs))
        for offset, index in enumerate(needs):
            budgets[index] += share + (1 if offset < remainder else 0)

    answer_type = "auto"
    evidence_type = "auto"
    limitations_type = "text"
    sources_type = "lines"

    answer, _ = truncate_structured_content(
        sections.answer,
        budgets[0],
        content_type=answer_type,
    )
    evidence, _ = truncate_structured_content(
        sections.evidence,
        budgets[1],
        content_type=evidence_type,
    )
    limitations, _ = truncate_structured_content(
        sections.limitations,
        budgets[2],
        content_type=limitations_type,
    )
    sources, _ = truncate_structured_content(
        sections.sources,
        budgets[3],
        content_type=sources_type,
    )

    truncated = AnswerSections(
        answer=answer or _FINAL_TRUNCATION_MARKER,
        evidence=evidence or f"- {_FINAL_TRUNCATION_MARKER}",
        limitations=limitations or f"- {_FINAL_TRUNCATION_MARKER}",
        sources=sources or f"- {_FINAL_TRUNCATION_MARKER}",
    )

    # Guard against separator/rounding drift. Reduce the largest mutable body
    # until the canonical rendering is within the configured hard limit.
    for _ in range(8):
        final_render = truncated.render()
        overflow = len(final_render) - max_chars
        if overflow <= 0:
            return truncated

        values = [
            truncated.answer,
            truncated.evidence,
            truncated.limitations,
            truncated.sources,
        ]
        index = max(range(len(values)), key=lambda item: len(values[item]))
        new_limit = max(1, len(values[index]) - overflow - 4)
        mode = (answer_type, evidence_type, limitations_type, sources_type)[index]
        shortened, _ = truncate_structured_content(
            values[index],
            new_limit,
            content_type=mode,
        )
        values[index] = shortened or _FINAL_TRUNCATION_MARKER[:new_limit]
        truncated = AnswerSections(
            answer=values[0],
            evidence=values[1],
            limitations=values[2],
            sources=values[3],
        )

    return truncated


def _extract_relevance_level(answer_section: str) -> int | None:
    patterns = (
        r"Livello\s+di\s+attinenza\s*[:|]\s*\*{0,2}([0-3])\b",
        r"\|\s*([0-3])\s*\|\s*\d+(?:[.,]\d+)?\s*%",
    )
    for pattern in patterns:
        match = re.search(pattern, answer_section or "", flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_relevance_percentage(answer_section: str) -> float | None:
    patterns = (
        r"Percentuale\s+stimata\s*[:|]\s*\*{0,2}(\d{1,3}(?:[.,]\d+)?)\s*%",
        r"\|\s*[0-3]\s*\|\s*(\d{1,3}(?:[.,]\d+)?)\s*%",
    )
    for pattern in patterns:
        match = re.search(pattern, answer_section or "", flags=re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", "."))
                return value if math.isfinite(value) else None
            except ValueError:
                return None
    return None


def _has_relevance_table(answer_section: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer_section or "").strip().casefold()
    header = re.sub(r"\s+", " ", _EVIDENCE_TABLE_HEADER).strip().casefold()
    return header in normalized and "|---" in (answer_section or "")


def _insert_relevance_table(
    sections: AnswerSections,
    *,
    level: int,
    percentage: float,
) -> AnswerSections:
    label = _RELEVANCE_LABELS[level]
    percentage_text = (
        str(int(percentage))
        if abs(percentage - round(percentage)) < 1e-9
        else f"{percentage:.2f}".replace(".", ",")
    )
    table = (
        f"{_EVIDENCE_TABLE_HEADER}\n"
        f"{_EVIDENCE_TABLE_SEPARATOR}\n"
        f"| {level} | {percentage_text}% | {label} |"
    )
    answer_body = sections.answer.strip()
    return sections.model_copy(
        update={"answer": f"{table}\n\n{answer_body}".strip()}
    )


def _has_remediation(limitations: str) -> bool:
    return bool(
        re.search(
            r"\b(?:remediation|piano\s+correttivo|azione\s+correttiva|"
            r"azioni\s+correttive|integrare|reperire|migliorare|raccomandazione)\b",
            limitations or "",
            flags=re.IGNORECASE,
        )
    )


# =============================================================================
# ANSWER VALIDATOR
# =============================================================================
class AnswerValidator:
    """Quality gate deterministico delle risposte RAG."""

    def __init__(self, config: RagSettings = settings) -> None:
        self._config = config

    def validate(
        self,
        *,
        answer: str,
        query: str,
        sources: Sequence[SourceItem],
        policy: ValidationPolicy | None = None,
        tenant_context: TenantContext | None = None,
    ) -> ValidationResult:
        resolved = policy or ValidationPolicy()
        issues: list[ValidationIssue] = []
        repaired = False
        blocked = False

        raw_answer, control_count = _remove_control_characters(answer or "")
        if control_count:
            repaired = True
            issues.append(
                _issue(
                    ValidationCode.CONTROL_CHARACTERS_REMOVED,
                    ValidationSeverity.WARNING,
                    "Sono stati rimossi caratteri di controllo non validi.",
                    repaired=True,
                    count=control_count,
                )
            )

        cleaned, metadata_count = strip_internal_metadata(raw_answer)
        if metadata_count:
            repaired = True
            issues.append(
                _issue(
                    ValidationCode.INTERNAL_METADATA_REMOVED,
                    ValidationSeverity.WARNING,
                    "Sono stati rimossi reasoning o metadati tecnici dalla risposta.",
                    repaired=True,
                    count=metadata_count,
                )
            )

        if not resolved.allow_external_urls:
            cleaned, url_count = _remove_external_urls(cleaned)
            if url_count:
                repaired = True
                issues.append(
                    _issue(
                        ValidationCode.EXTERNAL_URL_REMOVED,
                        ValidationSeverity.WARNING,
                        "Sono stati rimossi URL esterni non autorizzati.",
                        repaired=True,
                        count=url_count,
                    )
                )

        context = tenant_context
        if context is None:
            try:
                context = get_tenant_context()
            except TenantContextError:
                context = None

        if context is None:
            blocked = True
            issues.append(
                _issue(
                    ValidationCode.TENANT_CONTEXT_MISSING,
                    ValidationSeverity.CRITICAL,
                    "Il quality gate non dispone di un TenantContext valido.",
                )
            )
            visible_sources: list[SourceItem] = []
        else:
            visible_sources = filter_visible_records(
                sources,
                context=context,
                allow_graph_tier=resolved.allow_graph_tier,
                allow_user_tier=resolved.allow_user_tier,
            )
            if len(visible_sources) != len(sources):
                hidden_count = len(sources) - len(visible_sources)
                issues.append(
                    _issue(
                        ValidationCode.TENANT_SOURCE_VIOLATION,
                        ValidationSeverity.CRITICAL,
                        "Sono state rilevate fonti non visibili al tenant corrente.",
                        hidden_sources=hidden_count,
                    )
                )
                if resolved.block_on_tenant_violation:
                    blocked = True

        requested_document = (resolved.requested_document or "").strip()
        scoped_sources = list(visible_sources)

        if requested_document and resolved.enforce_requested_document:
            matching = [
                source
                for source in visible_sources
                if document_matches(source.filename, requested_document)
            ]
            outside = [
                source
                for source in visible_sources
                if not document_matches(source.filename, requested_document)
            ]

            if outside:
                issues.append(
                    _issue(
                        ValidationCode.REQUESTED_DOCUMENT_SCOPE_VIOLATION,
                        ValidationSeverity.CRITICAL,
                        "Il set di fonti contiene documenti esterni al document scope richiesto.",
                        requested_document=requested_document,
                        outside_filenames=sorted({source.filename for source in outside}),
                    )
                )
                if resolved.block_on_document_scope_violation:
                    blocked = True

            scoped_sources = matching
            if not matching:
                issues.append(
                    _issue(
                        ValidationCode.REQUESTED_DOCUMENT_NOT_FOUND,
                        ValidationSeverity.ERROR,
                        "Il documento richiesto non è presente tra le fonti tenant-visible.",
                        requested_document=requested_document,
                    )
                )

        if blocked:
            sections = _tenant_violation_fallback()
            safe_answer = sections.render()
            return ValidationResult(
                answer=safe_answer,
                valid=False,
                blocked=True,
                repaired=True,
                issues=tuple(issues),
                visible_sources=(),
                sections=sections,
            )

        if requested_document and resolved.enforce_requested_document and not scoped_sources:
            sections = _document_not_found_fallback(requested_document)
            safe_answer = sections.render()
            return ValidationResult(
                answer=safe_answer,
                valid=False,
                blocked=False,
                repaired=True,
                issues=tuple(issues),
                visible_sources=(),
                sections=sections,
            )

        if not cleaned.strip():
            sections = _empty_answer_fallback(sources_available=bool(scoped_sources))
            issues.append(
                _issue(
                    ValidationCode.EMPTY_ANSWER,
                    ValidationSeverity.ERROR,
                    "La generazione non ha prodotto una risposta utile.",
                    repaired=True,
                )
            )
            repaired = True
        else:
            complete_before = _has_complete_structure(cleaned)
            sections, duplicates = parse_answer_sections(cleaned)

            for letter in duplicates:
                issues.append(
                    _issue(
                        ValidationCode.DUPLICATE_SECTION,
                        ValidationSeverity.WARNING,
                        f"La sezione {letter} era presente più volte ed è stata accorpata.",
                        repaired=True,
                        section=letter,
                    )
                )
                repaired = True

            if not complete_before:
                if resolved.repair_structure:
                    sections = _complete_sections(sections)
                    issues.append(
                        _issue(
                            ValidationCode.STRUCTURE_REPAIRED,
                            ValidationSeverity.WARNING,
                            "La struttura A/B/C/D è stata completata automaticamente.",
                            repaired=True,
                        )
                    )
                    repaired = True
                else:
                    issues.append(
                        _issue(
                            ValidationCode.STRUCTURE_REPAIRED,
                            ValidationSeverity.ERROR,
                            "La risposta non contiene tutte le sezioni A/B/C/D obbligatorie.",
                        )
                    )
            else:
                normalized_render = _complete_sections(sections).render()
                if normalized_render.strip() != cleaned.strip():
                    issues.append(
                        _issue(
                            ValidationCode.STRUCTURE_NORMALIZED,
                            ValidationSeverity.INFO,
                            "Gli header A/B/C/D sono stati normalizzati.",
                            repaired=True,
                        )
                    )
                    repaired = True
                sections = _complete_sections(sections)

        sections, removed_source_references = _remove_unretrieved_file_references(
            sections,
            scoped_sources,
        )
        if removed_source_references:
            repaired = True
            issues.append(
                _issue(
                    ValidationCode.UNRETRIEVED_SOURCE_REFERENCE_REMOVED,
                    ValidationSeverity.WARNING,
                    "Sono stati rimossi riferimenti espliciti a file non presenti tra le fonti recuperate.",
                    repaired=True,
                    references=list(removed_source_references),
                )
            )

        graph_mode = (
            resolved.resolved_execution_mode() == RagExecutionMode.GRAPH_RELATION_STRICT
            or resolved.resolved_intent() == RagIntent.CHART
        )
        if graph_mode:
            rendered = sections.render()
            rendered, graph_count = _repair_graph_wording(rendered)
            if graph_count:
                sections, _ = parse_answer_sections(rendered)
                sections = _complete_sections(sections)
                repaired = True
                issues.append(
                    _issue(
                        ValidationCode.GRAPH_WORDING_REPAIRED,
                        ValidationSeverity.WARNING,
                        "È stata corretta una formulazione che descriveva Neo4j come assente o simulato.",
                        repaired=True,
                        count=graph_count,
                    )
                )

        answer_mode = resolved.resolved_answer_mode()
        if answer_mode == RagAnswerMode.EVIDENCE_RELEVANCE:
            sections, evidence_issues, evidence_repaired = self._validate_evidence_relevance(
                sections,
                requested_document=requested_document,
                sources=scoped_sources,
                require_table=resolved.require_evidence_table,
            )
            issues.extend(evidence_issues)
            repaired = repaired or evidence_repaired

        formula_mode = (
            resolved.resolved_intent() == RagIntent.FORMULA
            or resolved.resolved_execution_mode() == RagExecutionMode.FORMULA_STRICT
        )
        if formula_mode and not _has_visible_formula(sections.answer):
            issues.append(
                _issue(
                    ValidationCode.FORMULA_NOT_VISIBLE,
                    ValidationSeverity.ERROR,
                    "La risposta formula non contiene una formula o relazione matematica visibile.",
                )
            )

        if resolved.require_sources and not scoped_sources:
            issues.append(
                _issue(
                    ValidationCode.MISSING_RETRIEVED_SOURCES,
                    ValidationSeverity.ERROR,
                    "La modalità corrente richiede fonti, ma nessuna fonte è disponibile.",
                )
            )

        if resolved.rebuild_sources_section:
            source_text = _compact_sources(scoped_sources)
            if sections.sources.strip() != source_text.strip():
                sections = sections.model_copy(update={"sources": source_text})
                repaired = True
                issues.append(
                    _issue(
                        ValidationCode.SOURCES_SECTION_REBUILT,
                        ValidationSeverity.INFO,
                        "La sezione D) Fonti è stata ricostruita dalle fonti realmente recuperate.",
                        repaired=True,
                        source_count=len(scoped_sources),
                    )
                )

        if scoped_sources and not _answer_mentions_any_source(
            sections.evidence + "\n" + sections.sources,
            scoped_sources,
        ):
            issues.append(
                _issue(
                    ValidationCode.SOURCE_CITATION_MISSING,
                    ValidationSeverity.WARNING,
                    "La risposta finale non cita esplicitamente alcuna fonte recuperata nella sezione evidenze/fonti.",
                )
            )

        max_chars = int(resolved.max_answer_chars or self._config.max_assistant_chars)
        rendered = sections.render()
        if len(rendered) > max_chars:
            issues.append(
                _issue(
                    ValidationCode.ANSWER_TOO_LONG,
                    ValidationSeverity.WARNING,
                    "La risposta supera il limite massimo configurato.",
                    length=len(rendered),
                    max_length=max_chars,
                )
            )
            sections = _truncate_sections(sections, max_chars)
            rendered = sections.render()
            repaired = True
            issues.append(
                _issue(
                    ValidationCode.ANSWER_TRUNCATED,
                    ValidationSeverity.WARNING,
                    "La risposta è stata troncata preservando le quattro sezioni obbligatorie.",
                    repaired=True,
                    final_length=len(rendered),
                )
            )

        valid = not any(
            issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
            for issue in issues
        )

        return ValidationResult(
            answer=rendered,
            valid=valid,
            blocked=False,
            repaired=repaired,
            issues=tuple(issues),
            visible_sources=tuple(scoped_sources),
            sections=sections,
        )

    @staticmethod
    def _validate_evidence_relevance(
        sections: AnswerSections,
        *,
        requested_document: str,
        sources: Sequence[SourceItem],
        require_table: bool,
    ) -> tuple[AnswerSections, list[ValidationIssue], bool]:
        issues: list[ValidationIssue] = []
        repaired = False

        level = _extract_relevance_level(sections.answer)
        percentage = _extract_relevance_percentage(sections.answer)

        if level is None:
            issues.append(
                _issue(
                    ValidationCode.EVIDENCE_LEVEL_INVALID,
                    ValidationSeverity.ERROR,
                    "La risposta evidence relevance non contiene un livello di attinenza valido 0-3.",
                )
            )

        if percentage is None or not 0.0 <= percentage <= 100.0:
            issues.append(
                _issue(
                    ValidationCode.EVIDENCE_PERCENTAGE_INVALID,
                    ValidationSeverity.ERROR,
                    "La risposta evidence relevance non contiene una percentuale valida 0-100%.",
                )
            )

        has_table = _has_relevance_table(sections.answer)
        if require_table and not has_table:
            if level is not None and percentage is not None and 0.0 <= percentage <= 100.0:
                sections = _insert_relevance_table(
                    sections,
                    level=level,
                    percentage=percentage,
                )
                repaired = True
                issues.append(
                    _issue(
                        ValidationCode.EVIDENCE_TABLE_REPAIRED,
                        ValidationSeverity.WARNING,
                        "La tabella obbligatoria di attinenza è stata ricostruita dai valori presenti nella risposta.",
                        repaired=True,
                    )
                )
            else:
                issues.append(
                    _issue(
                        ValidationCode.EVIDENCE_TABLE_MISSING,
                        ValidationSeverity.ERROR,
                        "La sezione A non contiene la tabella Markdown obbligatoria per evidence relevance.",
                    )
                )

        if level is not None and percentage is not None and 0.0 <= percentage <= 100.0:
            lower, upper = _RELEVANCE_BANDS[level]
            if not lower <= percentage <= upper:
                issues.append(
                    _issue(
                        ValidationCode.EVIDENCE_SCORE_MISMATCH,
                        ValidationSeverity.WARNING,
                        "Livello di attinenza e percentuale non rispettano la banda di scoring configurata.",
                        level=level,
                        percentage=percentage,
                        expected_min=lower,
                        expected_max=upper,
                    )
                )

            if level <= 2 and not _has_remediation(sections.limitations):
                issues.append(
                    _issue(
                        ValidationCode.EVIDENCE_REMEDIATION_MISSING,
                        ValidationSeverity.ERROR,
                        "Per attinenza 0, 1 o 2 la sezione C deve contenere un remediation plan.",
                        level=level,
                    )
                )

        if requested_document and sources:
            evidence_text = sections.evidence.casefold()
            requested_name = requested_document.casefold()
            if requested_name not in evidence_text:
                issues.append(
                    _issue(
                        ValidationCode.SOURCE_CITATION_MISSING,
                        ValidationSeverity.WARNING,
                        "La sezione B non cita esplicitamente il documento richiesto.",
                        requested_document=requested_document,
                    )
                )

        return sections, issues, repaired


# =============================================================================
# FAITHFULNESS EVALUATION OPZIONALE
# =============================================================================
def _extract_json_object(value: str) -> dict[str, Any]:
    text = (value or "").strip()
    if not text:
        return {}

    # Rimuove eventuali code fence senza accettare testo arbitrario come JSON.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def _clamp01(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(1.0, max(0.0, parsed))


def build_evaluation_context(
    sources: Sequence[SourceItem],
    *,
    max_chars: int | None = None,
    config: RagSettings = settings,
    tenant_context: TenantContext | None = None,
) -> str:
    """Costruisce il contesto del judge senza identificativi tecnici."""

    context = tenant_context
    if context is None:
        context = get_tenant_context()

    visible = filter_visible_records(
        sources,
        context=context,
        allow_graph_tier=True,
        allow_user_tier=True,
    )

    limit = int(max_chars or config.evaluation_max_context_chars)
    parts: list[str] = []
    used = 0

    for index, source in enumerate(visible, start=1):
        header = (
            f"--- SOURCE [{index}] ---\n"
            f"filename: {source.filename}\n"
            f"page: {source.page}\n"
            f"type: {source.type}\n"
            f"tier: {source.tier}\n"
        )
        body = (source.content or "").strip()
        block = header + body + "\n\n"

        if used + len(block) > limit:
            remaining = limit - used - len(header) - 2
            if remaining <= 200:
                break
            block = header + body[:remaining].rstrip() + "...\n\n"

        parts.append(block)
        used += len(block)
        if used >= limit:
            break

    return "".join(parts).strip()


def _faithfulness_messages(
    *,
    query: str,
    answer: str,
    sources_context: str,
    requested_document: str,
) -> tuple[PromptMessage, PromptMessage]:
    scope_rule = (
        f"The requested document scope is exactly: {requested_document}. "
        "Set source_scope_violation=true if the answer relies on another document."
        if requested_document
        else "No explicit document scope was requested."
    )

    system_message = PromptMessage(
        role="system",
        content=(
            "You are a strict RAG faithfulness evaluator. "
            "Evaluate the answer only against the supplied sources. "
            "Do not use external knowledge and do not reward plausible but unsupported claims. "
            "Return only one valid JSON object with this exact schema: "
            '{"faithfulness":0.0,"answer_relevance":0.0,"context_support":0.0,'
            '"hallucination_risk":1.0,"source_scope_violation":false,'
            '"verdict":"PASS|WARN|FAIL","unsupported_claims":[],'
            '"supported_claims":[],"reason":""}. '
            "All scores must be numbers from 0 to 1. "
            "A correct statement that evidence is insufficient may receive high faithfulness."
        ),
    )

    user_message = PromptMessage(
        role="user",
        content=(
            f"### USER QUESTION\n{query}\n\n"
            f"### REQUESTED SOURCE SCOPE\n{scope_rule}\n\n"
            f"### SOURCES\n{sources_context or 'No sources supplied.'}\n\n"
            f"### ANSWER TO EVALUATE\n{answer}"
        ),
    )
    return system_message, user_message


class FaithfulnessEvaluator:
    """Judge opzionale basato sullo stesso endpoint Ollama nativo.

    Il testo prodotto dal judge non viene mai restituito direttamente al client;
    viene accettato soltanto il JSON validato e convertito in ``RagEvalResult``.
    """

    def __init__(
        self,
        *,
        config: RagSettings = settings,
        llm_generator: OllamaNativeGenerator = generator,
    ) -> None:
        self._config = config
        self._generator = llm_generator

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        sources: Sequence[SourceItem],
        requested_document: str = "",
        tenant_context: TenantContext | None = None,
    ) -> RagEvalResult:
        if not self._config.evaluation_enabled:
            return RagEvalResult.disabled()

        try:
            context = tenant_context or get_tenant_context()
            visible = filter_visible_records(
                sources,
                context=context,
                allow_graph_tier=True,
                allow_user_tier=True,
            )
            if len(visible) != len(sources):
                return RagEvalResult(
                    source_scope_violation=True,
                    verdict=EvaluationVerdict.FAIL,
                    reason="Tenant source violation detected before LLM evaluation.",
                )

            if requested_document and any(
                not document_matches(source.filename, requested_document)
                for source in visible
            ):
                return RagEvalResult(
                    source_scope_violation=True,
                    verdict=EvaluationVerdict.FAIL,
                    reason="Requested document scope violation detected before LLM evaluation.",
                )

            eval_context = build_evaluation_context(
                visible,
                config=self._config,
                tenant_context=context,
            )
            messages = _faithfulness_messages(
                query=query,
                answer=answer,
                sources_context=eval_context,
                requested_document=requested_document,
            )

            generation = self._generator.generate(
                messages,
                options=GenerationOptions(
                    model=self._config.evaluation_model_name,
                    think=False,
                    temperature=self._config.evaluation_temperature,
                    num_ctx=self._config.llm_num_ctx,
                    num_predict=min(2_048, self._config.llm_num_predict),
                    repeat_penalty=self._config.evaluation_repeat_penalty,
                    max_attempts=1,
                    retry_on_empty_content=False,
                    max_output_chars=8_000,
                ),
            )
            return self._parse_result(generation.content)

        except (GenerationError, TenantContextError, ValueError, TypeError) as exc:
            logger.warning("RAG evaluation failed: %s", str(exc)[:1000])
            return RagEvalResult.error(str(exc)[:20_000])

    async def evaluate_async(
        self,
        *,
        query: str,
        answer: str,
        sources: Sequence[SourceItem],
        requested_document: str = "",
        tenant_context: TenantContext | None = None,
    ) -> RagEvalResult:
        if not self._config.evaluation_enabled:
            return RagEvalResult.disabled()

        try:
            context = tenant_context or get_tenant_context()
            visible = filter_visible_records(
                sources,
                context=context,
                allow_graph_tier=True,
                allow_user_tier=True,
            )
            if len(visible) != len(sources):
                return RagEvalResult(
                    source_scope_violation=True,
                    verdict=EvaluationVerdict.FAIL,
                    reason="Tenant source violation detected before LLM evaluation.",
                )

            if requested_document and any(
                not document_matches(source.filename, requested_document)
                for source in visible
            ):
                return RagEvalResult(
                    source_scope_violation=True,
                    verdict=EvaluationVerdict.FAIL,
                    reason="Requested document scope violation detected before LLM evaluation.",
                )

            eval_context = build_evaluation_context(
                visible,
                config=self._config,
                tenant_context=context,
            )
            messages = _faithfulness_messages(
                query=query,
                answer=answer,
                sources_context=eval_context,
                requested_document=requested_document,
            )

            generation = await self._generator.generate_async(
                messages,
                options=GenerationOptions(
                    model=self._config.evaluation_model_name,
                    think=False,
                    temperature=self._config.evaluation_temperature,
                    num_ctx=self._config.llm_num_ctx,
                    num_predict=min(2_048, self._config.llm_num_predict),
                    repeat_penalty=self._config.evaluation_repeat_penalty,
                    max_attempts=1,
                    retry_on_empty_content=False,
                    max_output_chars=8_000,
                ),
            )
            return self._parse_result(generation.content)

        except (GenerationError, TenantContextError, ValueError, TypeError) as exc:
            logger.warning("RAG evaluation failed: %s", str(exc)[:1000])
            return RagEvalResult.error(str(exc)[:20_000])

    def _parse_result(self, raw: str) -> RagEvalResult:
        data = _extract_json_object(raw)
        if not data:
            return RagEvalResult.error("Il judge non ha restituito JSON valido.")

        verdict_raw = str(data.get("verdict") or "UNKNOWN").strip().upper()
        try:
            verdict = EvaluationVerdict(verdict_raw)
        except ValueError:
            verdict = EvaluationVerdict.UNKNOWN

        result = RagEvalResult(
            faithfulness=_clamp01(data.get("faithfulness"), 0.0),
            answer_relevance=_clamp01(data.get("answer_relevance"), 0.0),
            context_support=_clamp01(data.get("context_support"), 0.0),
            hallucination_risk=_clamp01(data.get("hallucination_risk"), 1.0),
            source_scope_violation=bool(data.get("source_scope_violation", False)),
            verdict=verdict,
            unsupported_claims=tuple(data.get("unsupported_claims") or ()),
            supported_claims=tuple(data.get("supported_claims") or ()),
            reason=str(data.get("reason") or "")[:20_000],
        )
        result.resolve_verdict(
            minimum_faithfulness=self._config.evaluation_min_faithfulness,
            minimum_answer_relevance=self._config.evaluation_min_answer_relevance,
        )
        return result


def evaluation_requires_block(
    result: RagEvalResult,
    *,
    config: RagSettings = settings,
) -> bool:
    """Applica la policy EVAL_STRICT_BLOCK senza riscrivere la risposta."""

    if not config.evaluation_strict_block:
        return False
    return bool(
        result.source_scope_violation
        or str(result.verdict) == EvaluationVerdict.FAIL
        or result.faithfulness < config.evaluation_min_faithfulness
        or result.answer_relevance < config.evaluation_min_answer_relevance
    )


def strict_evaluation_fallback(result: RagEvalResult) -> str:
    """Risposta sicura usata dal service quando il judge blocca l'output."""

    sections = AnswerSections(
        answer=(
            "La risposta generata non ha superato il controllo di affidabilità e non viene restituita."
        ),
        evidence=(
            "- Il quality gate non ha confermato un supporto documentale sufficiente per tutte le affermazioni."
        ),
        limitations=(
            f"- Verdetto evaluation: `{result.verdict}`.\n"
            f"- Faithfulness: `{result.faithfulness:.2f}`.\n"
            f"- Answer relevance: `{result.answer_relevance:.2f}`.\n"
            f"- Source scope violation: `{result.source_scope_violation}`."
        ),
        sources="- Nessuna risposta documentale pubblicata.",
    )
    return sections.render()


# Singleton senza side effect: non chiama Ollama durante l'import.
answer_validator = AnswerValidator()
faithfulness_evaluator = FaithfulnessEvaluator()


__all__ = [
    "AnswerSections",
    "AnswerValidator",
    "FaithfulnessEvaluator",
    "ValidationCode",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationSeverity",
    "answer_validator",
    "build_evaluation_context",
    "evaluation_requires_block",
    "faithfulness_evaluator",
    "parse_answer_sections",
    "strict_evaluation_fallback",
    "strip_internal_metadata",
]
