"""Solver deterministici del motore RAG.

Il modulo contiene esclusivamente calcolo e parsing deterministico su valori
forniti dall'utente. Non accede a database, tenant, modelli LLM, retrieval,
FastAPI o Reflex.

I solver derivano dai rami ``math_direct`` presenti nell'ultimo
``gui_reflex.py`` e mantengono la struttura Markdown A/B/C/D già utilizzata
dal PoC. Il modulo aggiunge:

- un risultato strutturato ``SolverResult``;
- una pipeline esplicita e ordinata;
- uso di ``Decimal`` nei calcoli economici e percentuali;
- validazioni fail-safe sui valori estratti;
- un valutatore AST limitato per espressioni aritmetiche;
- compatibilità con ``try_solve_math_query() -> str | None``.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import StrEnum
from typing import Any, Callable, Mapping


# =============================================================================
# RISULTATO E REGISTRO SOLVER
# =============================================================================
class SolverName(StrEnum):
    CONTROL_COVERAGE = "control_coverage"
    ROSI = "rosi"
    SLA_CUMULATIVE_HOURS = "sla_cumulative_hours"
    RISK_PRODUCT = "risk_product"
    PERCENTAGE_REMAINDER = "percentage_remainder"
    USER_ALGEBRA = "user_algebra"
    DATE_OFFSETS = "date_offsets"


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Risultato interno di un solver deterministico."""

    solver: SolverName
    answer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.answer or "").strip():
            raise ValueError("answer del solver non può essere vuota")


SolverFunction = Callable[[str], str | None]


# =============================================================================
# FORMATTAZIONE COMUNE
# =============================================================================
def _build_markdown_answer(
    *,
    answer: str,
    calculation_lines: list[str] | tuple[str, ...] = (),
    evidence_lines: list[str] | tuple[str, ...] = (),
    limitation_lines: list[str] | tuple[str, ...] = (),
    source_lines: list[str] | tuple[str, ...] = (),
    calculation_title: str = "Calcolo deterministico",
) -> str:
    """Costruisce la risposta Markdown A/B/C/D usata dal servizio RAG."""

    sections: list[str] = ["**A) Risposta**", "", answer.strip()]

    if calculation_lines:
        sections.extend(
            [
                "",
                f"**{calculation_title}:**",
                "",
                *[line.rstrip() for line in calculation_lines if line.strip()],
            ]
        )

    sections.extend(
        [
            "",
            "**B) Evidenze**",
            "",
            *[line.rstrip() for line in evidence_lines if line.strip()],
            "",
            "**C) Limiti / Conflitti**",
            "",
            *[line.rstrip() for line in limitation_lines if line.strip()],
            "",
            "**D) Fonti**",
            "",
            *[line.rstrip() for line in source_lines if line.strip()],
        ]
    )

    return "\n".join(sections).strip()


# =============================================================================
# PARSING E FORMATTAZIONE NUMERICA
# =============================================================================
def _parse_decimal(value: str | int | float | Decimal) -> Decimal:
    """Converte numeri nei principali formati italiani e inglesi.

    Esempi:
    - ``250.000`` -> ``250000``
    - ``250,5`` -> ``250.5``
    - ``250.000,50`` -> ``250000.50``
    - ``250,000.50`` -> ``250000.50``
    """

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numero non finito")
        result = Decimal(str(value))
    else:
        normalized = str(value or "").strip()
        normalized = (
            normalized.replace("€", "")
            .replace("EUR", "")
            .replace("eur", "")
            .replace(" ", "")
        )

        if not normalized:
            raise ValueError("numero vuoto")

        if not re.fullmatch(r"[+-]?[0-9][0-9.,]*", normalized):
            raise ValueError(f"formato numerico non valido: {value!r}")

        # Formato italiano: 1.234,56
        if "," in normalized and "." in normalized and normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        # Formato inglese: 1,234.56
        elif "," in normalized and "." in normalized and normalized.rfind(".") > normalized.rfind(","):
            normalized = normalized.replace(",", "")
        # Solo virgola: comportamento compatibile con il PoC, virgola decimale.
        elif "," in normalized and "." not in normalized:
            normalized = normalized.replace(",", ".")
        # Solo punti: gruppi da tre sono interpretati come separatori migliaia.
        elif "." in normalized and re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", normalized):
            normalized = normalized.replace(".", "")

        try:
            result = Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError(f"formato numerico non valido: {value!r}") from exc

    if not result.is_finite():
        raise ValueError("numero non finito")

    return result


def _parse_it_number(value: str) -> float:
    """Wrapper compatibile con il precedente RAG."""

    return float(_parse_decimal(value))


def _parse_probability_number(value: str) -> float:
    """Parsing sicuro di una probabilità decimale."""

    parsed = _parse_decimal(value)
    if parsed < 0 or parsed > 1:
        raise ValueError("la probabilità deve essere compresa tra 0 e 1")
    return float(parsed)


def _decimal_plain(value: Decimal, *, max_places: int = 8) -> str:
    """Formatta un Decimal senza notazione scientifica e senza zeri superflui."""

    quantizer = Decimal(1).scaleb(-max_places)
    if value.as_tuple().exponent < -max_places:
        value = value.quantize(quantizer, rounding=ROUND_HALF_UP)

    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"-0", ""}:
        rendered = "0"
    return rendered


def _format_decimal_it(value: Decimal, *, max_places: int = 4) -> str:
    return _decimal_plain(value, max_places=max_places).replace(".", ",")


def _format_euro_it(value: Decimal | float | int) -> str:
    parsed = _parse_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{parsed:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =============================================================================
# VALUTATORE ARITMETICO AST LIMITATO
# =============================================================================
class ArithmeticEvaluationError(ValueError):
    """Espressione aritmetica non valida o oltre i limiti di sicurezza."""


_ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_CHARS = 256
_MAX_AST_NODES = 80
_MAX_AST_DEPTH = 16
_MAX_ABS_RESULT = 10**18
_MAX_ABS_EXPONENT = 12


def _validate_arithmetic_result(value: float) -> float:
    if not math.isfinite(value):
        raise ArithmeticEvaluationError("risultato non finito")
    if abs(value) > _MAX_ABS_RESULT:
        raise ArithmeticEvaluationError("risultato oltre il limite consentito")
    return value


def _eval_expr_node(node: ast.AST, *, depth: int = 0) -> float:
    if depth > _MAX_AST_DEPTH:
        raise ArithmeticEvaluationError("espressione troppo annidata")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithmeticEvaluationError("sono ammessi soltanto valori numerici")
        return _validate_arithmetic_result(float(node.value))

    # Compatibilità con AST Python precedenti.
    if isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8
        return _validate_arithmetic_result(float(node.n))

    if isinstance(node, ast.UnaryOp):
        operation = _ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ArithmeticEvaluationError("operatore unario non consentito")
        return _validate_arithmetic_result(
            operation(_eval_expr_node(node.operand, depth=depth + 1))
        )

    if isinstance(node, ast.BinOp):
        operation = _ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ArithmeticEvaluationError("operatore binario non consentito")

        left = _eval_expr_node(node.left, depth=depth + 1)
        right = _eval_expr_node(node.right, depth=depth + 1)

        if isinstance(node.op, ast.Div) and right == 0:
            raise ArithmeticEvaluationError("divisione per zero")
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_ABS_EXPONENT:
            raise ArithmeticEvaluationError("esponente oltre il limite consentito")

        try:
            result = operation(left, right)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ArithmeticEvaluationError("operazione aritmetica non valida") from exc

        return _validate_arithmetic_result(float(result))

    raise ArithmeticEvaluationError("elemento AST non consentito")


def evaluate_arithmetic_expression(expression: str) -> float:
    """Valuta un'espressione composta solo da numeri e operatori ammessi."""

    cleaned = str(expression or "").strip()
    if not cleaned:
        raise ArithmeticEvaluationError("espressione vuota")
    if len(cleaned) > _MAX_EXPRESSION_CHARS:
        raise ArithmeticEvaluationError("espressione troppo lunga")
    if not re.fullmatch(r"[0-9+\-*/().\s]+", cleaned):
        raise ArithmeticEvaluationError("caratteri non consentiti")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticEvaluationError("sintassi aritmetica non valida") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ArithmeticEvaluationError("espressione troppo complessa")

    return _eval_expr_node(tree.body)


def eval_expr(node: ast.AST) -> float:
    """Wrapper compatibile con il valutatore AST del PoC."""

    return _eval_expr_node(node)


def calcolatrice_universale(espressione_matematica: str) -> str:
    """Calcolatrice legacy con output numerico nel formato italiano.

    Come nel PoC, elimina il testo non matematico; a differenza del PoC applica
    limiti su lunghezza, profondità, numero di nodi, potenze e risultato.
    """

    cleaned = re.sub(r"[^0-9+\-*/().]", "", str(espressione_matematica or ""))
    if not cleaned:
        return ""

    try:
        result = evaluate_arithmetic_expression(cleaned)
    except ArithmeticEvaluationError:
        return ""

    return f"{result:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =============================================================================
# SOLVER: COPERTURA CONTROLLI
# =============================================================================
def solve_control_coverage(query_text: str) -> str | None:
    """Calcola la copertura equivalente di controlli completi e parziali."""

    query = str(query_text or "")
    normalized = query.lower().replace("\\%", "%")

    if not any(term in normalized for term in ("controlli", "checklist", "controls", "control")):
        return None
    if not any(term in normalized for term in ("copertura", "coverage", "equivalente", "complessiva", "overall")):
        return None

    total_match = re.search(
        r"(?:checklist\s+(?:di|of)|totale\s+(?:di|of)|su|of)\s+(\d+)\s+(?:controlli|controls)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not total_match:
        total_match = re.search(
            r"(\d+)\s+(?:controlli|controls)\s+(?:totali|total)",
            normalized,
            flags=re.IGNORECASE,
        )

    implemented_match = re.search(
        r"(\d+)\s+(?:risultano\s+)?(?:implementati|implemented|completi|complete)",
        normalized,
        flags=re.IGNORECASE,
    )
    partial_match = re.search(
        r"(\d+)\s+(?:controlli\s+)?(?:parziali|partial|partially\s+implemented)",
        normalized,
        flags=re.IGNORECASE,
    )
    weight_match = re.search(
        r"(?:valgono|valgano|worth|weighted\s+at|peso|pesati|pesate)\s+(?:al\s+|del\s+)?(\d+(?:[.,]\d+)?)\s*%",
        normalized,
        flags=re.IGNORECASE,
    )

    if not (total_match and implemented_match and partial_match and weight_match):
        return None

    total = int(total_match.group(1))
    implemented = int(implemented_match.group(1))
    partial = int(partial_match.group(1))
    partial_weight_pct = _parse_decimal(weight_match.group(1))

    if total <= 0 or implemented < 0 or partial < 0:
        return None
    if implemented + partial > total:
        return None
    if partial_weight_pct < 0 or partial_weight_pct > 100:
        return None

    equivalent_controls = Decimal(implemented) + Decimal(partial) * partial_weight_pct / Decimal(100)
    coverage_pct = equivalent_controls / Decimal(total) * Decimal(100)

    return _build_markdown_answer(
        answer=f"La copertura equivalente complessiva è **{coverage_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}%**.",
        calculation_lines=[
            f"- Controlli totali = `{total}`",
            f"- Controlli implementati = `{implemented}`",
            f"- Controlli parziali = `{partial}`",
            f"- Peso controlli parziali = `{_decimal_plain(partial_weight_pct)}%`",
            (
                f"- Controlli equivalenti = `{implemented} + {partial} × "
                f"{_decimal_plain(partial_weight_pct)}% = {_decimal_plain(equivalent_controls)}`"
            ),
            (
                f"- Copertura = `{_decimal_plain(equivalent_controls)} / {total} × 100 = "
                f"{coverage_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}%`"
            ),
        ],
        evidence_lines=[
            "- I valori numerici usati nel calcolo sono stati estratti dalla domanda dell'utente.",
            "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.",
        ],
        limitation_lines=[
            "- Il calcolo assume che i controlli implementati valgano al 100%.",
            "- I controlli implementati e parziali sono considerati categorie non sovrapposte.",
            "- Il peso dei controlli parziali è quello indicato nella domanda.",
        ],
        source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
    )


# =============================================================================
# SOLVER: PRODOTTO DEL RISCHIO
# =============================================================================
def solve_risk_product(query_text: str) -> str | None:
    """Ordina scenari etichettati tramite probabilità × impatto."""

    query = str(query_text or "")
    normalized = query.replace("×", "x").replace("*", "x")

    pairs = re.findall(
        r"\b([A-Z])\s*(\d+(?:[,.]\d+)?)\s*x\s*(\d+(?:[,.]\d+)?)",
        normalized,
        flags=re.IGNORECASE,
    )

    if len(pairs) < 2 or not re.search(r"rischio|risk|probabil", normalized, flags=re.IGNORECASE):
        return None

    results: list[tuple[str, Decimal, Decimal, Decimal]] = []
    labels_seen: set[str] = set()

    for label, probability_raw, impact_raw in pairs:
        canonical_label = label.upper()
        if canonical_label in labels_seen:
            return None
        labels_seen.add(canonical_label)

        probability = _parse_decimal(probability_raw)
        impact = _parse_decimal(impact_raw)
        if probability < 0 or impact < 0:
            return None
        results.append((canonical_label, probability, impact, probability * impact))

    results.sort(key=lambda row: row[3], reverse=True)
    ranking = ", ".join(f"{label}={_decimal_plain(score)}" for label, _, _, score in results)

    return _build_markdown_answer(
        answer=f"Ordinamento dal rischio più critico al meno critico: **{ranking}**.",
        calculation_lines=[
            f"- Scenario {label}: `{_decimal_plain(probability)} × {_decimal_plain(impact)} = {_decimal_plain(score)}`."
            for label, probability, impact, score in results
        ],
        evidence_lines=[
            "- Le coppie numeriche e le etichette degli scenari sono state estratte dalla domanda dell'utente.",
            "- L'ordinamento è stato calcolato in modo deterministico.",
        ],
        limitation_lines=[
            "- La formula `rischio = probabilità × impatto` deve essere fornita o chiaramente implicata dalla domanda.",
            "- Il risultato numerico non dimostra da solo la conformità: va collegato al risk assessment documentale.",
        ],
        source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
    )


# =============================================================================
# SOLVER: ALLOCAZIONE PERCENTUALE RESIDUA
# =============================================================================
def solve_percentage_remainder_allocation(query_text: str) -> str | None:
    """Calcola percentuale e importo residui di un totale allocato per quote."""

    query = str(query_text or "").replace("\\%", "%")
    normalized = query.lower()

    remainder_terms = (
        "restante", "residuo", "residua", "residui", "residue", "rimanente",
        "rimanenza", "quota residua", "quota restante", "quota rimanente",
        "destinabile", "da destinare", "non allocato", "non allocata",
        "non allocati", "non allocate", "non assegnato", "non assegnata",
        "remaining", "remainder", "residual", "leftover", "remaining share",
        "remaining amount", "unallocated", "unassigned", "not allocated",
        "not assigned", "to allocate", "to assign",
    )
    allocation_terms = (
        "budget", "totale", "importo", "ammontare", "stanziamento",
        "valore complessivo", "importo complessivo", "totale disponibile",
        "costo", "costi", "alloca", "allocato", "allocata", "allocazione",
        "ripartito", "ripartizione", "quota", "quote", "percentuale",
        "percentuali", "destinato", "assegnato", "effort", "total", "amount",
        "overall amount", "available budget", "cost", "costs", "allocate",
        "allocated", "allocation", "distribute", "distribution", "share",
        "shares", "percentage", "percentages", "assigned", "dedicated", "earmarked",
    )

    if not any(term in normalized for term in remainder_terms):
        return None
    if not any(term in normalized for term in allocation_terms):
        return None

    percentages = [
        _parse_decimal(match.group(1))
        for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%", query)
    ]
    if len(percentages) < 2:
        return None
    if any(value < 0 or value > 100 for value in percentages):
        return None

    numeric_candidates: list[Decimal] = []
    for match in re.finditer(
        r"\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\b|\b\d+(?:[,.]\d+)?\b",
        query,
    ):
        try:
            value = _parse_decimal(match.group(0))
        except ValueError:
            continue
        if value > 100:
            numeric_candidates.append(value)

    if not numeric_candidates:
        return None

    total = max(numeric_candidates)
    used_pct = sum(percentages, Decimal(0))
    remaining_pct = Decimal(100) - used_pct

    if total <= 0 or remaining_pct < 0:
        return None

    remaining_amount = total * remaining_pct / Decimal(100)
    pct_details = " + ".join(f"{_decimal_plain(value)}%" for value in percentages)

    return _build_markdown_answer(
        answer=(
            f"La quota residua è **{_decimal_plain(remaining_pct)}%** e corrisponde a "
            f"**{_format_euro_it(remaining_amount)} euro**."
        ),
        calculation_lines=[
            f"- Totale = `{_format_euro_it(total)} euro`",
            f"- Percentuali già allocate = `{pct_details} = {_decimal_plain(used_pct)}%`",
            f"- Percentuale residua = `100% - {_decimal_plain(used_pct)}% = {_decimal_plain(remaining_pct)}%`",
            (
                f"- Importo residuo = `{_format_euro_it(total)} × {_decimal_plain(remaining_pct)}% "
                f"= {_format_euro_it(remaining_amount)} euro`"
            ),
        ],
        evidence_lines=[
            "- I valori numerici usati nel calcolo sono stati estratti dalla domanda dell'utente.",
            "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.",
        ],
        limitation_lines=[
            "- Il calcolo considera tutte le percentuali indicate come quote dello stesso totale.",
            "- Eventuali costi indiretti, arrotondamenti contabili o imposte non sono considerati se non esplicitamente forniti.",
        ],
        source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
    )


# =============================================================================
# SOLVER: ALGEBRA FORNITA DALL'UTENTE
# =============================================================================
def _normalize_algebra_query(query_text: str) -> str:
    query = str(query_text or "")
    query = query.replace("\\%", "%")
    query = query.replace("\\_", "_")
    query = query.replace("\\times", "×")
    query = query.replace("\\leq", "≤").replace("\\le", "≤")
    query = query.replace("\\geq", "≥").replace("\\ge", "≥")
    query = query.replace("$", "")
    query = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", query)
    return query.replace("{", "").replace("}", "")


def try_solve_user_provided_algebra(query_text: str) -> str | None:
    """Deriva semplici equazioni/disequazioni esplicitamente fornite."""

    query = _normalize_algebra_query(query_text)
    if not query.strip():
        return None

    normalized = query.lower()
    algebra_triggers = (
        "equazione", "disequazione", "algebrica", "algebricamente",
        "in funzione di", "isola", "formula", "variabile", "risolvi",
        "esprimi", "deriva", "supera", "superare", "superi", "maggiore",
        "superiore", "rischio residuo", "rischio inerente", "equation",
        "inequality", "algebraic", "solve for", "as a function of", "derive",
        "express", "exceeds", "exceed", "greater", "higher", "more than",
        "residual risk", "inherent risk",
    )
    if not any(term in normalized for term in algebra_triggers):
        return None

    # Caso 1: Ri = K × V, Rr <= Ri / N, richiesta di Vm.
    inherent_match = re.search(
        r"\bR[_\s]?i\b\s*(?:=|è\s+definito\s+come|e\s+definito\s+come|defined\s+as)\s*([A-Z])\s*(?:×|x|\*)\s*V\b",
        query,
        flags=re.IGNORECASE,
    )
    residual_match = re.search(
        r"\bR[_\s]?r\b\s*(?:≤|<=|<)\s*\bR[_\s]?i\b\s*/\s*(\d+)",
        query,
        flags=re.IGNORECASE,
    )
    asks_vm = bool(re.search(r"\bV[_\s]?m\b", query, flags=re.IGNORECASE))

    if inherent_match and residual_match and asks_vm:
        factor = inherent_match.group(1).upper()
        denominator = int(residual_match.group(1))
        if denominator <= 0:
            return None

        return _build_markdown_answer(
            answer=f"La vulnerabilità mitigata deve rispettare **Vm ≤ V / {denominator}**.",
            calculation_lines=[
                f"- Rischio inerente: `Ri = {factor} × V`",
                f"- Rischio residuo coerente con la vulnerabilità mitigata: `Rr = {factor} × Vm`",
                f"- Vincolo richiesto: `Rr ≤ Ri / {denominator}`",
                f"- Sostituzione: `{factor} × Vm ≤ ({factor} × V) / {denominator}`",
                f"- Poiché `{factor}` è costante e positiva: `Vm ≤ V / {denominator}`",
                f"- Formula LaTeX:\n\n$$\nV_m \\leq \\frac{{V}}{{{denominator}}}\n$$",
            ],
            evidence_lines=[
                "- Le relazioni algebriche sono state estratte dalla domanda dell'utente.",
                "- La derivazione è stata eseguita in modo deterministico da Python, non dal modello LLM.",
            ],
            limitation_lines=[
                f"- La semplificazione richiede che `{factor}` sia costante e positiva.",
                "- La relazione residua usa la stessa struttura moltiplicativa del rischio inerente, sostituendo `V` con `Vm`.",
                "- Il risultato è una derivazione matematica dei dati forniti, non una validazione empirica del modello di rischio.",
            ],
            source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
        )

    # Caso 2: P% di una variabile supera una soglia in euro/milioni.
    percentage_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", query)
    threshold_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(milioni|milione|million|millions)\b",
        query,
        flags=re.IGNORECASE,
    )
    threshold_is_millions = bool(threshold_match)

    if not threshold_match:
        threshold_match = re.search(
            r"(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)\s*(?:euro|€)",
            query,
            flags=re.IGNORECASE,
        )

    has_greater_condition = any(
        term in normalized
        for term in (
            "supera", "superare", "superi", "maggiore", "superiore",
            "exceeds", "exceed", "greater", "higher", "more than",
        )
    ) or ">" in query

    if percentage_match and threshold_match and has_greater_condition:
        percentage = _parse_decimal(percentage_match.group(1))
        if percentage <= 0 or percentage > 100:
            return None

        variable_match = re.search(
            r"\b(?:fatturato|revenue|turnover)\s+(?:annuo|annual)?\s*([A-Z])\b",
            query,
            flags=re.IGNORECASE,
        )
        if not variable_match:
            variable_match = re.search(
                r"\b(?:valore|importo|totale|amount|total)\s+(?:annuo|annual)?\s*([A-Z])\b",
                query,
                flags=re.IGNORECASE,
            )
        if not variable_match:
            variable_match = re.search(
                r"\b(?:annuo|annual)\s+([A-Z])\b",
                query,
                flags=re.IGNORECASE,
            )

        variable = variable_match.group(1).upper() if variable_match else "X"
        threshold = _parse_decimal(threshold_match.group(1))
        threshold_euro = threshold * Decimal(1_000_000) if threshold_is_millions else threshold
        threshold_millions = threshold_euro / Decimal(1_000_000)
        percentage_decimal = percentage / Decimal(100)
        result_euro = threshold_euro / percentage_decimal
        result_millions = result_euro / Decimal(1_000_000)

        return _build_markdown_answer(
            answer=(
                f"La disequazione è **{_format_decimal_it(percentage_decimal)} × {variable} > "
                f"{_format_decimal_it(threshold_millions)} milioni**."
            ),
            calculation_lines=[
                f"- Percentuale = `{_format_decimal_it(percentage)}% = {_format_decimal_it(percentage_decimal)}`",
                f"- Soglia = `{_format_decimal_it(threshold_millions)} milioni`",
                (
                    f"- Disequazione = `{_format_decimal_it(percentage_decimal)} × {variable} > "
                    f"{_format_decimal_it(threshold_millions)} milioni`"
                ),
                (
                    f"- Divisione per `{_format_decimal_it(percentage_decimal)}`: `{variable} > "
                    f"{_format_decimal_it(threshold_millions)} / {_format_decimal_it(percentage_decimal)}`"
                ),
                (
                    f"- Risultato = **{variable} > {_format_decimal_it(result_millions)} milioni**, "
                    f"cioè **{_format_euro_it(result_euro)} euro**."
                ),
                (
                    f"- Formula LaTeX:\n\n$$\n{_decimal_plain(percentage_decimal)}{variable} > "
                    f"{_decimal_plain(threshold_millions)} \\Rightarrow {variable} > "
                    f"{_decimal_plain(result_millions)}\n$$"
                ),
            ],
            evidence_lines=[
                "- La percentuale, la variabile e la soglia sono state estratte dalla domanda dell'utente.",
                "- La derivazione è stata eseguita in modo deterministico da Python, non dal modello LLM.",
            ],
            limitation_lines=[
                "- Il calcolo considera la soglia nella stessa unità indicata nella domanda.",
                "- Non interpreta ulteriori criteri normativi non presenti nella domanda.",
            ],
            source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
        )

    return None


# =============================================================================
# SOLVER: SLA CUMULATIVO
# =============================================================================
def solve_sla_cumulative_hours(query_text: str) -> str | None:
    """Somma numero di elementi × ore massime per categoria di severità."""

    query = str(query_text or "")
    normalized = query.lower()

    if not any(
        term in normalized
        for term in ("tempo cumulativo", "cumulativo", "ore massime", "maximum time", "cumulative", "maximum hours")
    ):
        return None

    severity_aliases: dict[str, tuple[str, ...]] = {
        "critici": ("critici", "critico", "critical"),
        "alti": ("alti", "alto", "high"),
        "medi": ("medi", "medio", "medium"),
        "bassi": ("bassi", "basso", "low"),
    }
    all_aliases = "|".join(re.escape(alias) for aliases in severity_aliases.values() for alias in aliases)

    time_by_alias: dict[str, Decimal] = {}
    for match in re.finditer(
        rf"(\d+(?:[.,]\d+)?)\s*(?:ore|hours?)[^\n,;.]{{0,60}}?\b({all_aliases})\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        time_by_alias[match.group(2).lower()] = _parse_decimal(match.group(1))

    count_by_alias: dict[str, int] = {}
    for match in re.finditer(
        rf"(\d+)\s+(?:(?:incidenti|incidents?)\s+)?({all_aliases})\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        count_by_alias[match.group(2).lower()] = int(match.group(1))

    rows: list[tuple[str, int, Decimal, Decimal]] = []
    for canonical, aliases in severity_aliases.items():
        hours = next((time_by_alias[alias] for alias in aliases if alias in time_by_alias), None)
        count = next((count_by_alias[alias] for alias in aliases if alias in count_by_alias), None)
        if hours is not None and count is not None:
            if hours < 0 or count < 0:
                return None
            rows.append((canonical, count, hours, Decimal(count) * hours))

    if len(rows) < 2:
        return None

    total = sum((row[3] for row in rows), Decimal(0))

    return _build_markdown_answer(
        answer=f"Il tempo cumulativo massimo è **{_decimal_plain(total)} ore**.",
        calculation_lines=[
            f"- Incidenti {label}: `{count} × {_decimal_plain(hours)} ore = {_decimal_plain(subtotal)} ore`"
            for label, count, hours, subtotal in rows
        ] + [f"- Totale = **{_decimal_plain(total)} ore**"],
        evidence_lines=[
            "- I tempi massimi e il numero di incidenti sono stati estratti dalla domanda dell'utente.",
            "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.",
        ],
        limitation_lines=[
            "- Il calcolo somma i massimali per categoria.",
            "- Non considera sovrapposizioni operative o parallelizzazione se non esplicitamente indicate.",
        ],
        source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
    )


# =============================================================================
# SOLVER: ROSI
# =============================================================================
def _extract_probabilities(query: str) -> list[Decimal]:
    probabilities: list[Decimal] = []

    for raw in re.findall(r"\b0[.,]\d+\b|\b1[.,]0+\b", query):
        parsed = _parse_decimal(raw)
        if Decimal(0) <= parsed <= Decimal(1):
            probabilities.append(parsed)

    if len(probabilities) >= 2:
        return probabilities

    for raw in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", query):
        parsed = _parse_decimal(raw) / Decimal(100)
        if Decimal(0) <= parsed <= Decimal(1):
            probabilities.append(parsed)

    return probabilities


def solve_rosi_query(query_text: str) -> str | None:
    """Calcola ALE iniziale/finale, benefici e ROSI."""

    query = str(query_text or "")
    normalized = query.lower()

    if not any(
        term in normalized
        for term in ("rosi", "beneficio lordo", "beneficio netto", "return on security investment")
    ):
        return None

    impact_match = re.search(
        r"(?:impatto economico|asset ha impatto|impact)[^\d]{0,60}(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
        normalized,
        flags=re.IGNORECASE,
    )
    cost_match = re.search(
        r"(?:costo annuo|costo|cost)[^\d]{0,60}(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
        normalized,
        flags=re.IGNORECASE,
    )
    probabilities = _extract_probabilities(normalized)

    if not impact_match or not cost_match or len(probabilities) < 2:
        return None

    impact = _parse_decimal(impact_match.group(1))
    initial_probability = probabilities[0]
    final_probability = probabilities[1]
    cost = _parse_decimal(cost_match.group(1))

    if impact < 0 or cost <= 0:
        return None

    initial_ale = impact * initial_probability
    final_ale = impact * final_probability
    gross_benefit = initial_ale - final_ale
    net_benefit = gross_benefit - cost
    rosi = net_benefit / cost * Decimal(100)

    return _build_markdown_answer(
        answer=(
            f"Il beneficio lordo è **{_format_euro_it(gross_benefit)} euro**, "
            f"il beneficio netto è **{_format_euro_it(net_benefit)} euro** "
            f"e il ROSI è **{rosi.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}%**."
        ),
        calculation_lines=[
            (
                f"- ALE iniziale = `{_format_euro_it(impact)} × {_decimal_plain(initial_probability)} "
                f"= {_format_euro_it(initial_ale)} euro`"
            ),
            (
                f"- ALE dopo misura = `{_format_euro_it(impact)} × {_decimal_plain(final_probability)} "
                f"= {_format_euro_it(final_ale)} euro`"
            ),
            (
                f"- Beneficio lordo = `{_format_euro_it(initial_ale)} - {_format_euro_it(final_ale)} "
                f"= {_format_euro_it(gross_benefit)} euro`"
            ),
            (
                f"- Beneficio netto = `{_format_euro_it(gross_benefit)} - {_format_euro_it(cost)} "
                f"= {_format_euro_it(net_benefit)} euro`"
            ),
            (
                f"- ROSI = `{_format_euro_it(net_benefit)} / {_format_euro_it(cost)} × 100 "
                f"= {rosi.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}%`"
            ),
        ],
        evidence_lines=[
            "- Impatto, probabilità iniziale, probabilità post-misura e costo sono stati estratti dalla domanda dell'utente.",
            "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.",
        ],
        limitation_lines=[
            "- Il calcolo usa un modello annuo semplificato.",
            "- Non considera costi indiretti, attualizzazione o variazione temporale del rischio se non indicati nella domanda.",
        ],
        source_lines=["- Input utente: valori e relazioni matematiche presenti nella domanda."],
    )


# =============================================================================
# SOLVER: OFFSET TEMPORALI
# =============================================================================
_WEEKDAY_MAP: dict[str, int] = {
    "lunedì": 0, "lunedi": 0, "monday": 0,
    "martedì": 1, "martedi": 1, "tuesday": 1,
    "mercoledì": 2, "mercoledi": 2, "wednesday": 2,
    "giovedì": 3, "giovedi": 3, "thursday": 3,
    "venerdì": 4, "venerdi": 4, "friday": 4,
    "sabato": 5, "saturday": 5,
    "domenica": 6, "sunday": 6,
}
_WEEKDAY_NAMES_IT = (
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"
)


def _weekday_index_it_en(value: str) -> int | None:
    return _WEEKDAY_MAP.get(str(value or "").lower().strip())


def _weekday_name_it(index: int) -> str:
    return _WEEKDAY_NAMES_IT[index % 7]


def _format_it_date(value: datetime) -> str:
    return value.strftime("%d/%m/%Y")


def _parse_base_datetime_or_weekday(query_text: str) -> tuple[datetime | None, int | None, str, bool]:
    """Estrae data esplicita oppure giorno della settimana con ora.

    Ritorna ``(datetime, weekday, label, has_explicit_date)``.
    """

    query = str(query_text or "")
    normalized = query.lower()

    date_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:.*?\b(?:ore|at|alle)?\s*(\d{1,2})[:.](\d{2}))?",
        query,
        flags=re.IGNORECASE,
    )
    if date_match:
        day, month, year = map(int, date_match.group(1, 2, 3))
        hour = int(date_match.group(4) or 0)
        minute = int(date_match.group(5) or 0)
        try:
            base = datetime(year, month, day, hour, minute)
        except ValueError:
            return None, None, "", False
        return base, base.weekday(), f"{_format_it_date(base)} {hour:02d}:{minute:02d}", True

    weekdays_pattern = "|".join(re.escape(day) for day in _WEEKDAY_MAP)
    weekday_match = re.search(
        rf"\b({weekdays_pattern})\b.{{0,40}}?\b(?:ore|at|alle|le)?\s*(\d{{1,2}})[:.](\d{{2}})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not weekday_match:
        return None, None, "", False

    weekday = _weekday_index_it_en(weekday_match.group(1))
    hour = int(weekday_match.group(2))
    minute = int(weekday_match.group(3))
    if weekday is None or hour > 23 or minute > 59:
        return None, None, "", False

    # Data fittizia con lunedì 03/01/2000 per il solo calcolo ciclico.
    base = datetime(2000, 1, 3, hour, minute) + timedelta(days=weekday)
    return base, weekday, f"{_weekday_name_it(weekday)} {hour:02d}:{minute:02d}", False


def _extract_hour_offsets(query_text: str) -> list[Decimal]:
    normalized = str(query_text or "").lower().replace("\\%", "%")
    offsets: list[Decimal] = []

    for match in re.finditer(
        r"(?:entro|within)\s+(\d+(?:[.,]\d+)?)\s*(?:ore|hours?)",
        normalized,
        flags=re.IGNORECASE,
    ):
        offsets.append(_parse_decimal(match.group(1)))

    if not offsets:
        for match in re.finditer(
            r"(\d+(?:[.,]\d+)?)\s*(?:ore|hours?)",
            normalized,
            flags=re.IGNORECASE,
        ):
            offsets.append(_parse_decimal(match.group(1)))

    unique = sorted({value for value in offsets if value >= 0})
    return unique


def try_solve_date_offsets(query_text: str) -> str | None:
    """Calcola una o più scadenze espresse come offset in ore."""

    query = str(query_text or "")
    normalized = query.lower()
    if not any(
        term in normalized
        for term in ("ore", "ora", "entro", "scadenza", "within", "delta", "hours", "deadline")
    ):
        return None

    base, weekday, base_label, explicit_date = _parse_base_datetime_or_weekday(query)
    offsets = _extract_hour_offsets(query)
    if base is None or weekday is None or not offsets:
        return None

    calculation_lines: list[str] = []
    deadlines: list[datetime] = []

    for offset in offsets:
        minutes_decimal = offset * Decimal(60)
        minutes = int(minutes_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        deadline = base + timedelta(minutes=minutes)
        deadlines.append(deadline)

        if explicit_date:
            rendered_deadline = (
                f"{_weekday_name_it(deadline.weekday())} {_format_it_date(deadline)} "
                f"{deadline.hour:02d}:{deadline.minute:02d}"
            )
        else:
            rendered_deadline = (
                f"{_weekday_name_it(deadline.weekday())} "
                f"{deadline.hour:02d}:{deadline.minute:02d}"
            )

        calculation_lines.append(
            f"- `+{_decimal_plain(offset)} ore` → **{rendered_deadline}**"
        )

    if len(deadlines) >= 2:
        delta_seconds = (max(deadlines) - min(deadlines)).total_seconds()
        delta_hours = Decimal(str(delta_seconds)) / Decimal(3600)
        calculation_lines.append(
            f"- Delta tra prima e ultima scadenza = **{_decimal_plain(delta_hours)} ore**"
        )

    return _build_markdown_answer(
        answer=f"Base temporale: **{base_label}**.",
        calculation_lines=calculation_lines,
        calculation_title="Scadenze calcolate",
        evidence_lines=[
            "- La base temporale e gli offset in ore sono stati estratti dalla domanda dell'utente.",
            "- Il calcolo è stato eseguito in modo deterministico da Python, non dal modello LLM.",
        ],
        limitation_lines=[
            "- Il calcolo considera gli offset come ore solari continue.",
            "- Non considera festività, sospensioni operative, fusi orari o calendari lavorativi se non indicati nella domanda.",
        ],
        source_lines=["- Input utente: valori e relazioni temporali presenti nella domanda."],
    )


# =============================================================================
# ROUTING DEI SOLVER
# =============================================================================
_SOLVER_PIPELINE: tuple[tuple[SolverName, SolverFunction], ...] = (
    (SolverName.CONTROL_COVERAGE, solve_control_coverage),
    (SolverName.ROSI, solve_rosi_query),
    (SolverName.SLA_CUMULATIVE_HOURS, solve_sla_cumulative_hours),
    (SolverName.RISK_PRODUCT, solve_risk_product),
    (SolverName.PERCENTAGE_REMAINDER, solve_percentage_remainder_allocation),
    (SolverName.USER_ALGEBRA, try_solve_user_provided_algebra),
    (SolverName.DATE_OFFSETS, try_solve_date_offsets),
)


def solve_deterministic_query(query_text: str) -> SolverResult | None:
    """Esegue il primo solver compatibile, rispettando l'ordine del PoC."""

    query = str(query_text or "").strip()
    if not query:
        return None

    for solver_name, solver_function in _SOLVER_PIPELINE:
        answer = solver_function(query)
        if answer:
            return SolverResult(
                solver=solver_name,
                answer=answer,
                metadata={"deterministic": True},
            )

    return None


def try_solve_math_query(query_text: str) -> str | None:
    """Wrapper compatibile con ``gui_reflex.py``.

    Il nuovo service layer dovrebbe preferire ``solve_deterministic_query`` per
    conoscere anche quale solver ha prodotto la risposta.
    """

    result = solve_deterministic_query(query_text)
    return result.answer if result else None


def is_calculation_request(query_text: str) -> bool:
    """Riconosce richieste di calcolo operativo o formulazione algebrica."""

    normalized = str(query_text or "").lower().strip()
    if not normalized:
        return False

    calculation_verbs = (
        "calcola", "calcolare", "calcolo", "quantifica", "quantificare",
        "determina", "determinare", "quanto", "entro quale", "cifra esatta",
        "importo esatto", "tempo totale", "totale cumulativo", "delta",
        "risultato", "risolvi", "esprimi", "isola", "calculate", "compute",
        "quantify", "determine", "how much", "deadline", "total", "cumulative",
        "result", "solve", "express", "isolate",
    )
    algebra_terms = (
        "equazione", "equazioni", "disequazione", "disequazioni",
        "algebrica", "algebricamente", "in funzione di", "equation",
        "inequality", "algebraic", "as a function of",
    )

    has_verb = any(term in normalized for term in calculation_verbs)
    has_algebra = any(term in normalized for term in algebra_terms)
    has_values_or_symbols = bool(
        re.search(r"\d", normalized)
        or re.search(r"(?:<=|>=|≤|≥|=|>|<|%|×|\*|/|\\frac|\\times)", query_text or "")
    )

    return (has_verb and has_values_or_symbols) or has_algebra


__all__ = [
    "ArithmeticEvaluationError",
    "SolverName",
    "SolverResult",
    "calcolatrice_universale",
    "eval_expr",
    "evaluate_arithmetic_expression",
    "is_calculation_request",
    "solve_control_coverage",
    "solve_deterministic_query",
    "solve_percentage_remainder_allocation",
    "solve_risk_product",
    "solve_rosi_query",
    "solve_sla_cumulative_hours",
    "try_solve_date_offsets",
    "try_solve_math_query",
    "try_solve_user_provided_algebra",
]
