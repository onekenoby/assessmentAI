"""Deterministic Formula Strict rendering extracted from gui_reflex.py.

This module contains the pure classification/rendering logic over already
retrieved SourceItem objects. Retrieval orchestration and optional generated
examples remain outside this module.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import SourceItem



def candidate_matches_requested_doc(candidate: Any, requested_doc: str) -> bool:
    """Compatibility helper preserving the Reflex document-scope semantics."""
    if not requested_doc:
        return True
    wanted = normalize_doc_name(requested_doc)
    if not wanted:
        return True
    if isinstance(candidate, dict):
        raw_filename = str(candidate.get("filename", "") or "").strip()
    else:
        raw_filename = str(getattr(candidate, "filename", "") or "").strip()
    if raw_filename in ("", "Unknown", "Neo4j", "KG", "Neo4j Knowledge Graph"):
        return False
    filename = normalize_doc_name(raw_filename)
    if not filename:
        return False
    return wanted in filename or filename in wanted


_FORMULA_KG_ARTIFACT_MARKERS_V411 = [
    "Plain:", "Meaning:", "Formula::", "Formule collegate", "Formula from Knowledge Graph",
]


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def is_regulatory_classification_query(query_text: str) -> bool:
    """
    Riconosce domande normative/classificatorie che NON devono entrare
    in Formula Strict Mode solo perché contengono parole come soglia,
    sanzione, regolamento, soggetti o categorie.

    Non è adattativa:
    - non contiene nomi di test;
    - non contiene nomi di documenti;
    - non forza risposte;
    - riconosce una classe generale di domande normative.
    """
    q = (query_text or "").lower().strip()

    if not q:
        return False

    classification_starters = [
        "chi sono", "quali sono", "qual è", "quale è",
        "what are", "who are", "which are", "what is",
    ]

    regulatory_terms = [
        # IT
        "soggetti", "soggetto", "categorie", "categoria",
        "tipologie", "tipologia", "regime", "vigilanza",
        "obblighi", "obbligo", "requisiti", "requisito",
        "normativa", "regolamento", "direttiva", "legge",
        "classificazione", "classifica", "autorità",
        "responsabilità", "categorie normative",

        # EN
        "subjects", "entities", "categories", "category",
        "types", "classification", "regime", "supervision",
        "oversight", "obligations", "requirements",
        "regulation", "directive", "law", "authority",
        "responsibilities",
    ]

    has_classification_starter = any(t in q for t in classification_starters)
    has_regulatory_term = any(t in q for t in regulatory_terms)

    if has_classification_starter and has_regulatory_term:
        return True

    # Anche senza starter esplicito, una domanda su regime/categorie/soggetti
    # è classificatoria se non chiede calcolo o derivazione.
    classification_density = sum(1 for t in regulatory_terms if t in q)

    return classification_density >= 2

def is_formula_strict_query(query_text: str) -> bool:
    """
    Riconosce query matematiche/algebriche in modo non adattativo.

    Regola:
    - Formula Strict Mode parte solo se l'utente chiede davvero formula,
      equazione, disequazione, derivazione, calcolo o algebra.
    - Le domande normative/classificatorie NON devono entrare qui solo perché
      contengono soglie, sanzioni, soggetti, regolamenti o categorie.
    """
    q_raw = query_text or ""
    q = q_raw.lower()

    if not q.strip():
        return False

    # Se è una domanda normativa/classificatoria, NON usare formula mode
    # salvo che ci siano segnali matematici/algebrici espliciti.
    explicit_formula_terms = [
        # IT
        "formula", "formule",
        "equazione", "equazioni",
        "disequazione", "disequazioni",
        "algebra", "algebrica", "algebrico", "algebricamente",
        "esprimi", "isola", "in funzione di",
        "risolvi", "deriva", "derivazione",
        "scrivi la disequazione", "scrivi l'equazione",

        # EN
        "equation", "equations",
        "inequality", "inequalities",
        "algebraic", "algebraically",
        "solve", "solve for",
        "derive", "express", "as a function of",
        "formula", "formulas",
    ]

    if any(t in q for t in explicit_formula_terms):
        return True

    if is_regulatory_classification_query(query_text):
        return False

    # Calcolo numerico esplicito.
    calculation_terms = [
        # IT
        "calcola", "calcolo", "quantifica", "quanto vale",
        "cifra esatta", "importo esatto", "risultato",
        "percentuale", "budget", "roi", "rosi",
        "probabilità", "calcolo del rischio",

        # EN
        "calculate", "compute", "quantify", "how much",
        "exact amount", "result", "percentage",
        "budget", "risk score", "probability",
    ]

    if any(t in q for t in calculation_terms):
        return True

    # Simboli matematici + verbo operativo.
    has_math_symbols = bool(
        re.search(r"(<=|>=|≤|≥|=|>|<|\\times|×|\*|/|\\frac|%)", q_raw)
    )

    has_operational_verb = any(t in q for t in [
        "calcola", "risolvi", "scrivi", "esprimi", "isola",
        "verifica", "determina", "derive", "solve", "express",
        "calculate", "compute", "determine",
    ])

    if has_math_symbols and has_operational_verb:
        return True

    return False

def normalize_source_type(value: str) -> str:
    t = str(value or "").lower().strip()

    if t in {"formula", "math", "equation"}:
        return "formula"

    if t in {"image", "immagine", "imagine", "visual", "screenshot"}:
        return "image"

    if t in {"chart", "grafico", "chart_analysis", "diagram", "diagramma"}:
        return "chart"

    if t in {"table", "tabella"}:
        return "table"

    if t in {"text", "testo", ""}:
        return "text"

    return t

def normalize_doc_name(value: str) -> str:
    """
    Normalizza un nome documento per confronti robusti:
    - lowercase
    - rimuove estensioni
    - rimuove caratteri non alfanumerici
    - rimuove suffissi tecnici comuni tipo _out / output
    """
    if not value:
        return ""

    v = os.path.basename(str(value).lower().strip())

    v = re.sub(r"\.(pdf|md|txt|docx|html)$", "", v)
    v = re.sub(r"[_\-\s]+out$", "", v)
    v = re.sub(r"[_\-\s]+output$", "", v)
    v = re.sub(r"[^a-z0-9]+", "", v)

    return v

def extract_requested_document(query_text: str) -> str:
    """
    Estrae il documento richiesto dalla query in modo robusto.
    Supporta:
    - virgolette dritte: "file.pdf"
    - virgolette curve: “file.pdf”
    - apici: 'file.pdf'
    - filename libero nel testo: file.pdf
    """
    q = query_text or ""

    q = (
        q.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
    )

    # 1. Documento/file/pdf seguito da nome tra virgolette
    patterns = [
        r'\b(?:documento|file|pdf)\s+["\']([^"\']+\.(?:pdf|md|txt|docx|csv|html))["\']',
        r'\b(?:nel|nella|dal|dalla)\s+(?:documento|file|pdf)\s+["\']([^"\']+\.(?:pdf|md|txt|docx|csv|html))["\']',

        # 2. Documento/file/pdf seguito da filename non quotato
        r'\b(?:documento|file|pdf)\s+([A-Za-z0-9_\-\s\.]+\.(?:pdf|md|txt|docx|csv|html))\b',

        # 3. Qualunque filename esplicito nel testo
        r'\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)*\.(?:pdf|md|txt|docx|csv|html))\b',
    ]

    for pattern in patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,:;!?\"'")

    return ""

def _strip_math_wrappers(value: str) -> str:
    """Rimuove soltanto delimitatori matematici esterni completi."""
    v = (value or "").strip()
    v = re.sub(r"^`+|`+$", "", v).strip()

    changed = True
    while changed and v:
        changed = False
        wrapper_pairs = [
            ("$$", "$$"),
            ("$", "$"),
            (r"\[", r"\]"),
            (r"\(", r"\)"),
        ]
        for left, right in wrapper_pairs:
            if v.startswith(left) and v.endswith(right) and len(v) >= len(left) + len(right):
                v = v[len(left):-len(right)].strip()
                changed = True
                break

    return v

def _looks_definitional_metric(latex: str, meaning: str = "") -> bool:
    """
    Riconosce assegnazioni definitorie testuali senza usare termini di dominio.
    """
    value = _strip_math_wrappers(_normalize_latex_value(latex or ""))

    if "=" in value:
        _, right = value.split("=", 1)
        right = right.strip()
        right_plain = _formula_display_text(right, 1000)

        if re.fullmatch(r"\\?text\{[^}]+\}", right):
            return True

        words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", right_plain)
        operators = re.findall(
            r"[+\-*/×÷^]|\\frac|\\sum|\\prod|\\operatorname",
            right,
            flags=re.IGNORECASE,
        )
        if len(words) >= 3 and not operators:
            return True

    meaning_plain = _formula_display_text(meaning or "", 1000)
    meaning_words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", meaning_plain)
    return len(meaning_words) >= 5 and bool(
        re.search(
            r"\b(?:definizione|definition|metrica|metric|indicatore|indicator)\b",
            meaning_plain,
            flags=re.IGNORECASE,
        )
    )

def _looks_threshold_rule(text: str) -> bool:
    """
    v4.5: riconosce regole soglia anche quando i valori sono in LaTeX,
    es. 5\\% oppure 1\\text{ milione}.
    """
    raw = text or ""
    plain = _formula_plain_text(raw).lower()
    threshold_terms = [
        "oltre", "superiore", "almeno", "non inferiore", "maggiore di",
        "greater than", "over", "more than", "at least", "threshold",
        "soglia", "condizione", "condition",
    ]
    has_threshold_word = any(x in plain for x in threshold_terms)
    has_threshold_value = bool(
        re.search(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)", plain)
    )
    return has_threshold_word and has_threshold_value

def filter_sources_for_formula_answer(query_text: str, sources: List[SourceItem]) -> List[SourceItem]:
    """Riduce le fonti UI alle pagine effettivamente usate dalla tabella formule."""
    rows = clean_formula_rows(extract_formula_rows_from_sources(sources), max_rows=10)
    if not rows:
        return sources
    keys = {(normalize_doc_name(str(r.get("filename") or "")), int(r.get("page") or 0)) for r in rows}
    filtered: List[SourceItem] = []
    seen = set()
    for s in sources or []:
        key = (normalize_doc_name(str(getattr(s, "filename", "") or "")), int(getattr(s, "page", 0) or 0))
        if key in keys and key not in seen:
            seen.add(key)
            filtered.append(s)
    return filtered or sources

def _formula_display_text(value: Any, max_len: int = 600) -> str:
    """
    Produce una rappresentazione testuale leggibile senza modificare i
    comandi LaTeX per sostituzioni parziali.

    In particolare ``\\left`` non deve mai essere interpretato come ``\\le``.
    """
    text = _strip_math_wrappers(_normalize_latex_value(str(value or "")))
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("$$$", "$$")
    text = re.sub(r"^`+|`+$", "", text).strip()

    for _ in range(8):
        previous = text
        text = re.sub(
            r"\\(?:mathrm|mathbf|mathit|text|operatorname)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
        if text == previous:
            break

    text = text.replace(r"\left", "").replace(r"\right", "")

    def replace_fraction_plain(expression: str) -> str:
        marker = r"\frac"
        while marker in expression:
            pos = expression.find(marker)
            cursor = pos + len(marker)
            while cursor < len(expression) and expression[cursor].isspace():
                cursor += 1
            if cursor >= len(expression) or expression[cursor] != "{":
                break

            def read_group(open_index: int):
                depth = 0
                for i in range(open_index, len(expression)):
                    if expression[i] == "{":
                        depth += 1
                    elif expression[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return expression[open_index + 1:i], i + 1
                return None, open_index

            numerator, after_num = read_group(cursor)
            if numerator is None:
                break
            while after_num < len(expression) and expression[after_num].isspace():
                after_num += 1
            if after_num >= len(expression) or expression[after_num] != "{":
                break
            denominator, after_den = read_group(after_num)
            if denominator is None:
                break
            replacement = (
                f"({replace_fraction_plain(numerator)})/"
                f"({replace_fraction_plain(denominator)})"
            )
            expression = expression[:pos] + replacement + expression[after_den:]
        return expression

    text = replace_fraction_plain(text)

    text = text.replace(r"\sum", "Σ")
    text = text.replace(r"\prod", "Π")
    text = text.replace(r"\times", " × ")
    text = text.replace(r"\cdot", " · ")

    # Boundary di comando obbligatorio: evita ``\left -> ≤ft``.
    text = re.sub(r"\\leq?(?![A-Za-z])", " ≤ ", text)
    text = re.sub(r"\\geq?(?![A-Za-z])", " ≥ ", text)
    text = re.sub(r"\\neq(?![A-Za-z])", " ≠ ", text)
    text = re.sub(r"\\Rightarrow(?![A-Za-z])", " ⇒ ", text)
    text = re.sub(r"\\rightarrow(?![A-Za-z])", " → ", text)

    text = text.replace(r"\%", "%")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text

def _formula_plain_text(value: str) -> str:
    return _formula_display_text(value, 1000)

def _normalize_latex_value(value: str) -> str:
    """
    Normalizza escape tecnici senza riscrivere la semantica della formula.

    Supporta sia comandi LaTeX normali (``\\frac``) sia comandi rimasti
    doppiamente escapati nel testo persistito (``\\\\frac``).
    """
    v = str(value or "").strip()

    v = re.sub(r"\t(?=imes\b|ext\{)", r"\\t", v)
    v = re.sub(r"\r(?=ight\b)", r"\\r", v)
    v = re.sub(r"\f(?=rac\b)", r"\\f", v)
    v = re.sub("\x08(?=egin\\b)", r"\\b", v)
    v = v.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    latex_commands = (
        "frac", "sum", "prod", "left", "right", "mathrm", "mathbf",
        "mathit", "text", "operatorname", "cdot", "times", "leq", "le",
        "geq", "ge", "neq", "sqrt", "%", "_", "[", "]", "(", ")",
    )
    command_pattern = "|".join(re.escape(cmd) for cmd in latex_commands)
    v = re.sub(
        rf"\\{{2,}}(?=(?:{command_pattern})(?:\b|[^A-Za-z]))",
        lambda _m: "\\",
        v,
    )

    v = v.replace("$$$", "$$")
    v = re.sub(r"(?<![A-Za-z\\])ight(?=\s*[)\]}])", r"\\right", v)
    v = re.sub(r"(?<![A-Za-z\\])imes(?=\s*(?:\d|[A-Za-z\\({]))", r"\\times", v)
    v = re.sub(r"(?<![A-Za-z\\])rac(?=\s*\{)", r"\\frac", v)
    v = re.sub(r"(?<![A-Za-z\\])ext\{", r"\\text{", v)

    v = re.sub(r"\${3,}", "$$", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

def _threshold_rule_name(name: str, formula_or_text: str) -> str:
    plain = _formula_plain_text(formula_or_text)
    n = _formula_display_text(name, 120)
    generic = {"formula recuperata", "formula/metric", "metrica/indicatore citato", "elemento recuperato", "regola soglia"}
    m = re.search(r"\b(condizione\s*\d+)\b", plain, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    if n and n.lower() not in generic:
        return n
    return "Regola soglia"

def _extract_definition_from_latex(latex: str) -> str:
    v = _normalize_latex_value(latex or "")
    m = re.search(r"=\s*\\?text\{([^}]+)\}", v)
    if m:
        return _formula_display_text(m.group(1), 500)
    if "=" in v:
        return _formula_display_text(v.split("=", 1)[1], 500)
    return ""

def _extract_left_name_from_equation(latex: str) -> str:
    v = _strip_math_wrappers(_normalize_latex_value(latex or ""))
    if "=" not in v:
        return ""
    left = _formula_display_text(v.split("=", 1)[0], 120)
    left = re.sub(r"[^A-Za-zÀ-ÿ0-9_\-/ ]+", "", left).strip()
    return left[:80]

def _classify_formula_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(row)
    original_name = _formula_display_text(rr.get("name") or "", 120) or "Elemento recuperato"
    latex_raw = str(rr.get("latex") or "").strip()
    latex = _normalize_latex_value(latex_raw)
    meaning_raw = str(rr.get("meaning") or "")
    meaning = _formula_display_text(meaning_raw, 700)
    combined = " ".join([original_name, latex, meaning])

    if _looks_threshold_rule(combined):
        formula_plain = _formula_display_text(latex or combined, 700)
        rr["name"] = _threshold_rule_name(original_name, formula_plain)
        rr["tipo"] = "Regola soglia"
        rr["latex"] = formula_plain
        rr["meaning"] = "Criterio/soglia normativa recuperata; non è una formula computazionale."
        return rr

    if _looks_computational_formula(latex):
        rr["name"] = original_name
        rr["tipo"] = "Formula computazionale"

        # Il LaTeX deve essere preservato, non convertito in plain text.
        rr["latex"] = latex

        rr["meaning"] = (
            meaning
            or "Formula computazionale esplicita presente nella fonte recuperata."
        )
        return rr

    if _looks_definitional_metric(latex, meaning):
        left_name = _extract_left_name_from_equation(latex)
        definition = _extract_definition_from_latex(latex)
        rr["name"] = left_name or original_name
        rr["tipo"] = "Metrica definitoria"
        rr["latex"] = "formula computazionale non recuperata"
        rr["meaning"] = (
            f"Definizione testuale della metrica: {definition}. Formula computazionale non recuperata nella fonte."
            if definition else
            "Definizione testuale della metrica; formula computazionale non recuperata nella fonte."
        )
        return rr

    rr["name"] = original_name
    rr["tipo"] = "Metrica/elemento citato"
    rr["latex"] = "formula esplicita non recuperata"
    rr["meaning"] = meaning or "Elemento citato nelle fonti recuperate; nessuna formula esplicita è stata individuata nello stesso chunk."
    return rr

def _is_noise_formula_row_v45(row: Dict[str, Any]) -> bool:
    name = _formula_display_text(row.get("name") or "", 160).strip().lower()
    formula = _formula_display_text(row.get("latex") or "", 400).strip().lower()
    tipo = _formula_display_text(row.get("tipo") or "", 120).strip().lower()

    generic_names = {
        "", "formula/metric", "formula recuperata", "contenuto", "variabili",
        "metrica/indicatore citato", "formula", "metric",
        "formule e modelli matematici", "elemento recuperato",
    }

    combined = " ".join([
        name,
        formula,
        _formula_display_text(row.get("meaning") or "", 500).lower(),
    ])

    structural_noise = (
        "tikzpicture",
        "begintikzpicture",
        "\\draw",
        "\\node",
        "ode[",
        "cm,x=",
        "cm,y=",
    )

    if any(marker in combined for marker in structural_noise):
        return True

    if re.fullmatch(
        r"formule e modelli matematici(?:\s*-\s*pagina\s*\d+\s*--?)?",
        name,
    ):
        return True

    if tipo == "regola soglia":
        return not _looks_threshold_rule(" ".join([name, formula, str(row.get("meaning") or "")]))

    if (
        name in generic_names
        and tipo not in {"formula computazionale", "regola soglia"}
    ):
        return True

    # Exclude isolated values if they are not part of a threshold rule.
    plain = _formula_display_text(formula, 120).lower()
    if re.fullmatch(r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)?", plain):
        return True

    return False

def _formula_md_cell(value: Any, max_len: int = 600) -> str:
    return _formula_display_text(value, max_len)

def _extract_threshold_domain_from_rule(text: str) -> str:
    """Estrae un ambito leggibile senza dipendere da un corpus specifico (es. NIS2)."""
    plain = _formula_display_text(text, 160)
    
    # Rimuove l'intestazione tecnica se presente
    plain = re.sub(r"^(Condizione|Regola|Soglia|Threshold)\s*\d*\s*:\s*", "", plain, flags=re.IGNORECASE)
    
    # Prende semplicemente le prime parole significative come "ambito" descrittivo
    words = plain.split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")

def _formula_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        f"**{title}**",
        "",
        "| Nome / metrica | Tipo | Formula / regola | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {_formula_md_cell(r.get('name') or 'N/D', 180)} | "
            f"{_formula_md_cell(r.get('tipo') or 'N/D', 120)} | "
            f"{_formula_md_cell(r.get('latex') or 'formula esplicita non recuperata', 520)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 340)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )
    return "\n".join(lines)

def _extract_threshold_criterion(rule_text: str) -> str:
    """
    Estrae il criterio numerico comune da una regola soglia senza legarsi al corpus.
    Esempio generico: percentuale utenti + numero assoluto utenti.
    """
    plain = _formula_display_text(rule_text, 900)

    # Percent threshold, e.g. "oltre il 5% degli utenti ... nell'Unione"
    percent_part = ""
    m_percent = re.search(
        r"\b(oltre|superiore\s+a|maggiore\s+di|almeno|more\s+than|over|above)?\s*(?:il\s*)?(\d+(?:[,.]\d+)?)\s*%",
        plain,
        flags=re.IGNORECASE,
    )
    if m_percent:
        op = (m_percent.group(1) or "oltre").strip()
        value = m_percent.group(2).replace(",", ".")
        # Preserve a human-friendly Italian wording when the source is Italian.
        if re.search(r"\butenti\b", plain, flags=re.IGNORECASE):
            percent_part = f"oltre il {value}% degli utenti"
        else:
            percent_part = f"oltre il {value}%"
        if re.search(r"nell['’]Unione|Unione\s+europea|\bUE\b|\bEU\b", plain, flags=re.IGNORECASE):
            percent_part += " nell'Unione"

    # Absolute threshold, e.g. "oltre 1 milione di utenti ... nell'Unione"
    number_part = ""
    m_abs = re.search(
        r"\b(oltre|superiore\s+a|maggiore\s+di|almeno|more\s+than|over|above)?\s*(\d+(?:[,.]\d+)?)\s*(milione|milioni|million|millions)\b(?:\s+di\s+utenti)?",
        plain,
        flags=re.IGNORECASE,
    )
    if m_abs:
        value_raw = m_abs.group(2).replace(",", ".")
        unit = m_abs.group(3).lower()
        # Normalize English/Italian units only for display, not for logic.
        try:
            value_num = float(value_raw)
            value_display = f"{value_num:g}"
        except Exception:
            value_num = None
            value_display = value_raw

        is_one = value_display in {"1", "1.0"}
        unit_it = "milione" if is_one else "milioni"
        if unit in {"million", "millions"}:
            unit_it = "milione" if is_one else "milioni"
        number_part = f"oltre {value_display} {unit_it}"
        if re.search(r"\butenti\b", plain, flags=re.IGNORECASE):
            number_part += " di utenti"
        if re.search(r"nell['’]Unione|Unione\s+europea|\bUE\b|\bEU\b", plain, flags=re.IGNORECASE):
            number_part += " nell'Unione"

    parts = [p for p in [percent_part, number_part] if p]
    if parts:
        return " oppure ".join(parts)

    return plain

def _aggregate_threshold_rules(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    v4.8: aggrega soglie ripetute separando criterio e ambito.
    Non usa nomi/codici specifici del corpus: estrae domini e criteri dai testi recuperati.
    """
    threshold_rows = [r for r in rows if str(r.get("tipo") or "").lower() == "regola soglia"]
    if not threshold_rows:
        return []

    groups: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for r in threshold_rows:
        rule_text = _formula_display_text(r.get("latex") or "", 900)
        fname = str(r.get("filename") or "N/D")
        page = int(r.get("page") or 0)
        criterion = _extract_threshold_criterion(rule_text)
        criterion_key = re.sub(r"\s+", " ", criterion.lower()).strip()
        key = (fname, page, criterion_key)

        domain = _extract_threshold_domain_from_rule(rule_text)

        if key not in groups:
            groups[key] = {
                "elemento": "Soglie normative recuperate",
                "tipo": "Soglia normativa non di scoring",
                "criterio": criterion,
                "ambito": [],
                "meaning": "Criterio/condizione normativa recuperata. Non è una formula computazionale e non è una regola di scoring.",
                "filename": fname,
                "page": page,
            }

        if domain and domain not in groups[key]["ambito"]:
            groups[key]["ambito"].append(domain)

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        ambiti = g.get("ambito") or []
        g["ambito"] = "; ".join(ambiti[:8]) if ambiti else "ambito non specificato nella soglia recuperata"
        out.append(g)

    return out

def _formula_metrics_table(
    title: str,
    rows: List[Dict[str, Any]],
) -> str:
    if not rows:
        return ""

    lines = [f"**{title}**", ""]

    for index, row in enumerate(rows, start=1):
        name = _formula_display_text(
            row.get("name") or f"Formula {index}", 180
        )
        formula = _strip_dangling_math_delimiters_v416(
            str(row.get("latex") or "formula esplicita non recuperata")
        )
        row_type = _formula_display_text(row.get("tipo") or "N/D", 120)
        meaning = _formula_display_text(row.get("meaning") or "", 340)
        filename = _formula_display_text(row.get("filename") or "N/D", 180)
        page = int(row.get("page") or 0)

        plain_formula = _formula_plain_text(formula)
        lines.extend([
            f"### {index}. {name}",
            "",
            f"- **Formula testuale:** `{plain_formula}`",
            "",
            "- **Formula LaTeX:**",
            "",
            "$$",
            formula,
            "$$",
            "",
        ])

        if row_type.lower() != "formula computazionale":
            lines.append(f"- **Tipo:** {row_type}")
            if meaning:
                lines.append(f"- **Significato:** {meaning}")

        lines.extend([
            f"- **Fonte:** {filename}, pagina {page}",
            "",
        ])

    return "\n".join(lines).strip()

def _threshold_rules_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        f"**{title}**",
        "",
        "| Elemento | Tipo | Criterio | Ambito | Significato | Fonte | Pagina |",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {_formula_md_cell(r.get('elemento') or 'Soglia normativa recuperata', 180)} | "
            f"{_formula_md_cell(r.get('tipo') or 'Soglia normativa non di scoring', 150)} | "
            f"{_formula_md_cell(r.get('criterio') or 'criterio non recuperato puntualmente', 320)} | "
            f"{_formula_md_cell(r.get('ambito') or 'ambito non specificato', 260)} | "
            f"{_formula_md_cell(r.get('meaning') or '', 300)} | "
            f"{_formula_md_cell(r.get('filename') or 'N/D', 180)} | "
            f"{int(r.get('page') or 0)} |"
        )
    return "\n".join(lines)

def _is_formula_metric_intent_v410(query_text: str) -> bool:
    """True only when the query is about formulas/metrics/scoring/calculation."""
    try:
        return bool(is_formula_strict_query(query_text))
    except Exception:
        q = (query_text or "").lower()
        return any(t in q for t in ["formula", "formule", "metric", "metriche", "scoring", "score", "calcolo"])

def _temporal_metric_aliases_v410(query_text: str) -> List[str]:
    """
    Generic IT/EN synonym expansion for incident-response time metrics.
    It is activated only for formula/metric/scoring queries.
    """
    if not _is_formula_metric_intent_v410(query_text):
        return []

    q = (query_text or "").lower()

    detection_cues = [
        "tempo di rilevamento", "tempi di rilevamento", "tempo medio di rilevamento",
        "rilevamento", "detection time", "time to detect", "mean time to detect",
        "detect time",
    ]

    resolution_cues = [
        "tempo di risoluzione", "tempi di risoluzione", "tempo medio di risoluzione",
        "tempo di riparazione", "tempi di riparazione", "risoluzione", "riparazione",
        "resolution time", "time to resolution", "mean time to resolution",
        "repair time", "time to repair", "mean time to repair",
    ]

    aliases: List[str] = []

    if any(cue in q for cue in detection_cues):
        aliases.extend(["MTTD", "Mean Time to Detect", "tempo medio impiegato per rilevare", "tempo medio di rilevamento"])

    if any(cue in q for cue in resolution_cues):
        aliases.extend(["MTTR", "Mean Time to Resolution", "Mean Time to Repair", "tempo medio necessario per risolvere", "tempo medio di risoluzione", "tempo medio di riparazione"])

    # If the user says "tempi di rilevamento/risoluzione" or similar compact wording,
    # both branches should be retrieved.
    if re.search(r"rilevament[oa]\s*/\s*risoluzion[ea]|detect(?:ion)?\s*/\s*resolution", q):
        aliases.extend([
            "MTTD", "Mean Time to Detect", "tempo medio impiegato per rilevare",
            "MTTR", "Mean Time to Resolution", "Mean Time to Repair", "tempo medio necessario per risolvere",
        ])

    out: List[str] = []
    seen = set()
    for a in aliases:
        key = a.lower().strip()
        if a and key not in seen:
            seen.add(key)
            out.append(a)
    return out

def _requested_formula_terms_missing(query_text: str, rows: List[Dict[str, Any]]) -> List[str]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra:
    - logica base per termini generici richiesti ma non recuperati;
    - estensione v4.10 per severity/severità e MTTD/MTTR.
    """
    ql = (query_text or "").lower()
    found_text = " ".join([
        str(r.get("name", "")) + " " +
        str(r.get("latex", "")) + " " +
        str(r.get("meaning", ""))
        for r in rows or []
    ]).lower()

    requested_generic = [
        "cvss", "rischio", "risk", "maturità", "maturity", "copertura", "coverage"
    ]

    missing: List[str] = []

    for term in requested_generic:
        if term in ql and term not in found_text:
            missing.append(term)

    for term in ["severity", "severità"]:
        if term in ql and term not in found_text:
            missing.append(term)

    try:
        temporal_aliases = _temporal_metric_aliases_v410(query_text)
    except Exception:
        temporal_aliases = []

    if temporal_aliases:
        wants_detection = any(
            a.lower() in {"mttd", "mean time to detect"} or "rileva" in a.lower()
            for a in temporal_aliases
        )
        wants_resolution = any(
            a.lower() in {"mttr", "mean time to resolution", "mean time to repair"}
            or "risolvere" in a.lower()
            or "riparazione" in a.lower()
            for a in temporal_aliases
        )
        if wants_detection and "mttd" not in found_text:
            missing.append("MTTD")
        if wants_resolution and "mttr" not in found_text:
            missing.append("MTTR")

    return sorted(set(missing))

def _formula_has_kg_artifacts_v411(value: Any) -> bool:
    text = str(value or "")
    return any(marker.lower() in text.lower() for marker in _FORMULA_KG_ARTIFACT_MARKERS_V411)

def _formula_is_kg_aggregate_source_v411(row: Dict[str, Any]) -> bool:
    fname = str(row.get("filename") or "").strip().lower()
    page = int(row.get("page") or 0)
    name = str(row.get("name") or "").strip().lower()
    return (
        fname in {"kg", "neo4j", "neo4j knowledge graph"}
        or (page == 0 and name in {"formule collegate", "latex"})
        or _formula_has_kg_artifacts_v411(row.get("latex"))
        or _formula_has_kg_artifacts_v411(row.get("meaning"))
    )

def _looks_computational_formula(latex: str) -> bool:
    """
    Riconosce formule computazionali tramite struttura matematica, non tramite
    nomi di metriche o parole legate a uno specifico dominio.
    """
    value = _strip_math_wrappers(
        _normalize_latex_value(str(latex or ""))
    )
    value_lower = value.lower()

    if not value or "formula esplicita non recuperata" in value_lower:
        return False

    if _formula_has_kg_artifacts_v411(value):
        return False

    if "=" not in value:
        return bool(
            re.search(
                r"\\frac|\\sum|\\prod|[+\-*/×÷^]",
                value,
                flags=re.IGNORECASE,
            )
        )

    left, right = value.split("=", 1)
    left = left.strip()
    right = right.strip()

    if not left or not right:
        return False

    if re.fullmatch(r"\\?text\{[^}]+\}", right):
        return False

    right_plain = _formula_display_text(right, 1000)
    word_pairs = re.findall(
        r"\b[A-Za-zÀ-ÿ]{2,}\s+[A-Za-zÀ-ÿ]{2,}\b",
        right_plain,
    )
    operator_tokens = re.findall(
        r"[+\-*/×÷^]|\\frac|\\sum|\\prod|\\operatorname",
        right,
        flags=re.IGNORECASE,
    )
    has_strong_math_construct = bool(
        re.search(
            r"\\frac|\\sum|\\prod|\\operatorname|[()^]",
            right,
            flags=re.IGNORECASE,
        )
    )

    # Una sequenza prevalentemente discorsiva con un simbolo incidentale
    # non è una formula computazionale.
    if len(word_pairs) >= 2 and len(operator_tokens) <= 1:
        return False

    if operator_tokens and (
        has_strong_math_construct
        or len(operator_tokens) >= 1
    ):
        return True

    # Un'assegnazione numerica isolata è un valore, non una formula.
    if re.fullmatch(
        r"\d+(?:[,.]\d+)?\s*(?:%|per cento|percent)?",
        right_plain,
    ):
        return False

    return False

def _formula_row_quality_v411(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Attribuisce priorità alla formula documentale nominata e ben formata."""
    fname = str(row.get("filename") or "")
    page = int(row.get("page") or 0)
    name = str(row.get("name") or "")
    latex = str(row.get("latex") or "")
    text = " ".join([name, latex, str(row.get("meaning") or "")])
    origin = str(row.get("formula_origin") or "")
    is_kg = _formula_is_kg_aggregate_source_v411(row)
    has_artifacts = _formula_has_kg_artifacts_v411(text)
    has_real_doc = bool(
        fname
        and fname.lower() not in {"kg", "neo4j", "neo4j knowledge graph", "n/d"}
        and page > 0
    )
    named = 0 if _is_generic_formula_name(name) else 1
    clean_latex = 0 if re.search(r"(?<![A-Za-z])(?:ight|imes|rac)(?![A-Za-z])", latex) else 1
    origin_score = {
        "document_equation": 3,
        "latex": 2,
        "knowledge_graph": 1,
    }.get(origin, 0)
    return (
        2 if has_real_doc else (0 if is_kg else 1),
        named,
        clean_latex,
        origin_score,
        0 if has_artifacts else 1,
    )

def _formula_has_invalid_latex_syntax_v414(value: Any) -> bool:
    """
    Rifiuta formule sintatticamente corrotte o ricostruite da nomi riservati
    LaTeX scambiati per variabili/funzioni.
    """
    v = _normalize_latex_value(str(value or "").strip())
    if not v:
        return False

    if re.search(r"\\(?:operatorname|mathrm|mathbf|mathit|text)(?!\s*\{)", v):
        return True
    if "\\№" in v or "№(" in v:
        return True
    if re.search(r"(?:≤|≥)\s*ft\b", v, flags=re.IGNORECASE):
        return True
    if v.count("{") != v.count("}"):
        return True

    reserved = r"left|right|frac|mathrm|mathbf|mathit|text|cdot|times|leq|geq|neq"
    if re.search(rf"\\operatorname\{{(?:{reserved})\}}", v, flags=re.IGNORECASE):
        return True
    if re.search(rf"\\mathrm\{{(?:{reserved})\}}", v, flags=re.IGNORECASE):
        return True
    if "```" in v or re.search(r"(?:Formula\s+LaTeX|Formula\s+testuale)\s*:", v, re.I):
        return True

    return False

def _scope_formula_sources_to_requested_document_v414(
    query_text: str,
    sources: List[SourceItem],
) -> List[SourceItem]:
    """
    Applica il document scope e, quando disponibile, usa la copia canonica
    arricchita da PostgreSQL al posto dei duplicati vettoriali/semantici.
    """
    requested_doc = extract_requested_document(query_text)
    if not requested_doc:
        return list(sources or [])

    scoped: List[SourceItem] = []
    for source in sources or []:
        candidate = {"filename": str(getattr(source, "filename", "") or "")}
        if candidate_matches_requested_doc(candidate, requested_doc):
            scoped.append(source)

    canonical = [
        source
        for source in scoped
        if (
            bool(str(getattr(source, "pg_ingestion_ts", "") or "").strip())
            or bool(str(getattr(source, "pg_source_name", "") or "").strip())
            or "PG_Enrich" in str(getattr(source, "db_origin", "") or "")
            or str(getattr(source, "db_origin", "") or "").startswith("PostgresDocScope")
        )
    ]

    return canonical or scoped

def clean_formula_rows(rows: List[Dict[str, Any]], max_rows: int = 10) -> List[Dict[str, Any]]:
    """Classifica e deduplica formule equivalenti provenienti da più database."""
    classified: List[Dict[str, Any]] = []

    for order, row in enumerate(rows or []):
        # v4.14 - Non tentare di riparare semanticamente formule corrotte.
        # Vengono scartate; una sorgente raw/documentale valida ha priorità.
        if _formula_has_invalid_latex_syntax_v414(row.get("latex")):
            continue

        classified_row = _classify_formula_row(row)
        classified_row["_formula_order"] = order

        name_lower = _formula_display_text(
            classified_row.get("name") or "", 160
        ).lower()
        if name_lower in {"formule collegate", "latex", "formula from knowledge graph"}:
            continue
        if _is_noise_formula_row_v45(classified_row):
            continue

        classified.append(classified_row)

    def identity_key(row: Dict[str, Any]) -> Tuple[str, str]:
        row_type = str(row.get("tipo") or "").lower()
        if row_type == "formula computazionale":
            identity = _canonical_formula_identity(str(row.get("latex") or ""))
            return "formula", identity
        if row_type == "regola soglia":
            identity = re.sub(
                r"\s+", " ", _formula_display_text(row.get("latex") or "", 900).lower()
            ).strip()
            return "threshold", identity[:400]
        name = re.sub(r"[^a-z0-9]+", "", str(row.get("name") or "").lower())
        return row_type, name

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    first_order: Dict[Tuple[str, str], int] = {}

    for row in classified:
        key = identity_key(row)
        if not key[1]:
            continue
        first_order.setdefault(key, int(row.get("_formula_order") or 0))
        existing = by_key.get(key)

        if existing is None:
            by_key[key] = row
            continue

        current_quality = _formula_row_quality_v411(row)
        existing_quality = _formula_row_quality_v411(existing)
        if current_quality > existing_quality:
            # Se la riga scelta ha un nome generico, conserva il nome migliore
            # già trovato per la stessa identità matematica.
            if _is_generic_formula_name(row.get("name")) and not _is_generic_formula_name(existing.get("name")):
                row["name"] = existing.get("name")
            by_key[key] = row
        elif _is_generic_formula_name(existing.get("name")) and not _is_generic_formula_name(row.get("name")):
            existing["name"] = row.get("name")

    deduped = list(by_key.items())
    priority = {
        "formula computazionale": 0,
        "regola soglia": 1,
        "metrica definitoria": 2,
        "metrica/elemento citato": 3,
    }
    deduped.sort(
        key=lambda pair: (
            priority.get(str(pair[1].get("tipo") or "").lower(), 9),
            first_order.get(pair[0], 10**9),
        )
    )

    result = []
    for _, row in deduped[:max_rows]:
        row.pop("_formula_order", None)
        result.append(row)
    return result

def _formula_examples_requested(query_text: str) -> bool:
    q = (query_text or "").lower()

    return bool(
        re.search(
            r"\b(esempio|esempi|example|examples)\b",
            q,
        )
    )

def _strip_dangling_math_delimiters_v416(value: str) -> str:
    """Rimuove delimitatori matematici esterni, anche se rimasti spaiati."""
    v = _strip_math_wrappers(_normalize_latex_value(value or "")).strip()
    v = re.sub(r"^\s*\${1,2}\s*", "", v)
    v = re.sub(r"\s*\${1,2}\s*$", "", v)
    return v.strip()

def _answer_formula_strict_core(query_text: str, sources: List[SourceItem]) -> Optional[str]:
    """
    Costruisce una risposta deterministica per formule, metriche e soglie.

    Le categorie richieste vengono determinate semanticamente; le formule
    estratte non vengono inventate o riscritte dall'LLM.
    """
    rows = clean_formula_rows(
        extract_formula_rows_from_sources(sources),
        max_rows=30,
    )

    if not rows:
        return (
            "**A) Risposta**\n\n"
            "Non ho trovato formule computazionali, metriche definitorie "
            "o regole di scoring esplicite nelle fonti recuperate.\n\n"
            "**B) Evidenze**\n\n"
            "- Il sistema ha cercato formule, metriche e regole di scoring "
            "nei chunk recuperati e nel Knowledge Graph.\n\n"
            "\n\n**C) Limiti / Conflitti**\n\n"
            "- La risposta non inventa formule mancanti.\n"
            "- Percentuali isolate, intestazioni o righe generiche non sono "
            "state considerate formule.\n\n"
            "**D) Fonti**\n\n"
            "- Vedi pannello Fonti/Audit per i chunk recuperati."
        )

    computational = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "formula computazionale"
    ]
    definitional = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "metrica definitoria"
    ]
    thresholds = _aggregate_threshold_rules(rows)
    cited = [
        row
        for row in rows
        if str(row.get("tipo") or "").lower()
        == "metrica/elemento citato"
    ]

    query_lower = (query_text or "").lower()
    has_formula_terms = bool(
        re.search(
            r"\b(?:formula|formule|equazione|equazioni|formulas?|equations?)\b",
            query_lower,
        )
    )
    has_metric_terms = bool(
        re.search(
            r"\b(?:metrica|metriche|indicatore|indicatori|"
            r"scoring|metrics?|indicators?)\b",
            query_lower,
        )
    )
    asks_only_formulas = has_formula_terms and not has_metric_terms

    primary_rows = (
        computational
        if asks_only_formulas
        else computational + definitional
    )

    blocks: List[str] = []

    if primary_rows:
        title = (
            "Formule computazionali recuperate"
            if asks_only_formulas
            else "Formule computazionali e metriche recuperate"
        )
        blocks.append(
            _formula_metrics_table(
                title,
                primary_rows,
            )
        )

        if _formula_examples_requested(query_text) and computational:
            examples_text = _generate_formula_examples(
                query_text,
                computational,
            )
            if examples_text:
                blocks.append(
                    "### Esempi applicativi illustrativi\n\n"
                    + examples_text
                )
            else:
                blocks.append(
                    "### Esempi applicativi illustrativi\n\n"
                    "Le formule sono state recuperate, ma non è stato possibile "
                    "produrre esempi strutturati validi."
                )

    if thresholds and not asks_only_formulas:
        blocks.append(
            _threshold_rules_table(
                "Soglie normative recuperate ma non classificabili come scoring",
                thresholds,
            )
        )

    if (
        cited
        and not asks_only_formulas
        and not primary_rows
        and not thresholds
    ):
        blocks.append(
            _formula_metrics_table(
                "Elementi citati senza formula esplicita",
                cited,
            )
        )

    if not blocks:
        blocks.append(
            "Non ho trovato formule computazionali sufficientemente "
            "esplicite nelle fonti recuperate."
        )

    rows_for_sources: List[Dict[str, Any]] = (
        primary_rows
        if asks_only_formulas
        else primary_rows + thresholds + cited
    )

    used_files: List[str] = []
    seen_files = set()
    for row in rows_for_sources:
        filename = str(row.get("filename") or "").strip()
        page = int(row.get("page") or 0)
        if not filename:
            continue
        label = f"{filename} (p.{page})" if page else filename
        if label not in seen_files:
            seen_files.add(label)
            used_files.append(label)

    missing_terms = _requested_formula_terms_missing(
        query_text,
        rows_for_sources,
    )

    evidence_lines = [
        "- Gli elementi sono stati classificati in modo deterministico."
    ]
    if asks_only_formulas:
        evidence_lines.append(
            "- Sono state incluse esclusivamente le formule computazionali "
            "esplicite recuperate dalle fonti."
        )
    else:
        evidence_lines.append(
            "- Le metriche definitorie sono distinte dalle formule "
            "computazionali."
        )

    if thresholds and not asks_only_formulas:
        evidence_lines.append(
            "- Le soglie normative sono riportate separatamente perché non "
            "sono automaticamente formule o regole di scoring."
        )
    if missing_terms:
        evidence_lines.append(
            "- Non sono state recuperate formule computazionali esplicite per: "
            + ", ".join(missing_terms)
            + "."
        )

    return (
        "**A) Risposta**\n\n"
        + "\n\n".join(blocks)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_lines)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "- La risposta non inventa formule mancanti.\n"
        + "- Una metrica definitoria non viene trattata come formula "
        + "computazionale se la fonte non contiene un calcolo esplicito.\n"
        + "- Una soglia normativa indica una condizione o un criterio; non "
        + "misura automaticamente un punteggio o una maturità.\n\n"
        + "**D) Fonti**\n\n"
        + (
            "\n".join(f"- {item}" for item in used_files)
            if used_files
            else "- Fonti non disponibili."
        )
    )

def _threshold_rule_segments_v413(text: str, max_segments: int = 8) -> List[str]:
    """Extract readable threshold-rule segments from arbitrary text."""
    raw = str(text or "")
    if not raw.strip():
        return []

    # Convert common bullet/list separators into split points, but keep sentences readable.
    candidates = re.split(r"(?<=[\.\;\!\?])\s+|\n+|\r+", raw)

    out: List[str] = []
    seen = set()

    for c in candidates:
        seg = _formula_display_text(c, 900)
        if not seg:
            continue

        if not _looks_threshold_rule(seg):
            continue

        # Avoid isolated numeric fragments such as only "5%" or "1 milione".
        words = re.findall(r"[A-Za-zÀ-ÿ]+", seg)
        if len(words) < 5:
            continue

        key = re.sub(r"\s+", " ", seg.lower())[:260]
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)

        if len(out) >= max_segments:
            break

    # Fallback: if the whole chunk contains a threshold but splitting missed it,
    # take a window around the first threshold-looking expression.
    if not out and _looks_threshold_rule(raw):
        plain = _formula_display_text(raw, 3000)
        m = re.search(
            r"(?:oltre|superiore|almeno|non inferiore|maggiore di|greater than|over|more than|at least|threshold|soglia|condizione|condition).{0,420}?(?:\d+(?:[,.]\d+)?\s*(?:%|per cento|percent|milione|milioni|million|millions)).{0,420}",
            plain,
            flags=re.IGNORECASE,
        )
        if m:
            out.append(_formula_display_text(m.group(0), 900))

    return out

def _formula_text_to_latex(lhs: str, rhs: str) -> str:
    """
    Converte un'espressione aritmetica testuale in LaTeX valido.

    Il parser è indipendente dal dominio e supporta identificatori, numeri,
    chiamate di funzione, parentesi e operatori aritmetici comuni.
    """
    token_pattern = re.compile(
        r"\s*(?:(?P<number>\d+(?:[.,]\d+)?)|"
        r"(?P<name>[A-Za-z][A-Za-z0-9_]*)|"
        r"(?P<op>[+\-*/^(),]))"
    )

    def tokenize(expression: str):
        tokens = []
        position = 0
        while position < len(expression):
            match = token_pattern.match(expression, position)
            if not match:
                position += 1
                continue
            kind = "number" if match.group("number") else "name" if match.group("name") else "op"
            value = match.group(kind)
            tokens.append((kind, value))
            position = match.end()
        return tokens

    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0

        def peek(self, value=None):
            if self.pos >= len(self.tokens):
                return False
            return value is None or self.tokens[self.pos][1] == value

        def take(self):
            token = self.tokens[self.pos]
            self.pos += 1
            return token

        def parse(self):
            return self.expression()

        def expression(self):
            node = self.term()
            while self.peek("+") or self.peek("-"):
                op = self.take()[1]
                node = ("bin", op, node, self.term())
            return node

        def term(self):
            node = self.power()
            while self.peek("*") or self.peek("/"):
                op = self.take()[1]
                node = ("bin", op, node, self.power())
            return node

        def power(self):
            node = self.unary()
            if self.peek("^"):
                self.take()
                node = ("bin", "^", node, self.power())
            return node

        def unary(self):
            if self.peek("+") or self.peek("-"):
                return ("unary", self.take()[1], self.unary())
            return self.primary()

        def primary(self):
            if self.pos >= len(self.tokens):
                return ("raw", "")

            kind, value = self.take()
            if kind == "number":
                return ("number", value.replace(",", "."))

            if kind == "name":
                if self.peek("("):
                    self.take()
                    args = []
                    if not self.peek(")"):
                        while True:
                            args.append(self.expression())
                            if self.peek(","):
                                self.take()
                                continue
                            break
                    if self.peek(")"):
                        self.take()
                    return ("call", value, args)
                return ("name", value)

            if value == "(":
                node = self.expression()
                if self.peek(")"):
                    self.take()
                return ("group", node)

            return ("raw", value)

    def render_name(name: str) -> str:
        if "_" in name:
            base, suffix = name.split("_", 1)
            base_latex = base if len(base) == 1 else rf"\mathrm{{{base}}}"
            suffix_latex = suffix if len(suffix) == 1 else rf"\mathrm{{{suffix}}}"
            return rf"{base_latex}_{{{suffix_latex}}}"
        return name if len(name) == 1 else rf"\mathrm{{{name}}}"

    def render(node, parent_precedence=0):
        kind = node[0]
        if kind == "number":
            return node[1]
        if kind == "name":
            return render_name(node[1])
        if kind == "raw":
            return node[1]
        if kind == "group":
            return rf"\left({render(node[1])}\right)"
        if kind == "unary":
            return node[1] + render(node[2], 4)
        if kind == "call":
            function_name = node[1]
            args = node[2]
            rendered_args = ", ".join(render(arg) for arg in args)
            if function_name.lower() == "sum":
                return rf"\sum\left({rendered_args}\right)"
            return rf"\operatorname{{{function_name}}}\left({rendered_args}\right)"
        if kind == "bin":
            op, left, right = node[1], node[2], node[3]
            if op == "/":
                return rf"\frac{{{render(left)}}}{{{render(right)}}}"
            if op == "^":
                return rf"{{{render(left, 3)}}}^{{{render(right)}}}"
            precedence = 1 if op in {"+", "-"} else 2
            symbol = r" \cdot " if op == "*" else f" {op} "
            rendered = render(left, precedence) + symbol + render(right, precedence + 1)
            if precedence < parent_precedence:
                return rf"\left({rendered}\right)"
            return rendered
        return ""

    lhs_clean = re.sub(r"\s+", " ", str(lhs or "").strip())
    rhs_clean = re.sub(r"\s+", " ", str(rhs or "").strip())

    lhs_latex = r"\%" if lhs_clean == "%" else render_name(lhs_clean)
    tokens = tokenize(rhs_clean)
    rhs_latex = render(Parser(tokens).parse()) if tokens else rhs_clean
    return f"{lhs_latex} = {rhs_latex}"

def _canonical_formula_identity(value: str) -> str:
    """Restituisce una chiave canonica per deduplicare LaTeX e testo equivalenti."""
    v = _normalize_latex_value(value)
    v = _strip_math_wrappers(v)
    v = v.replace("\\(", "").replace("\\)", "")
    v = v.replace("\\[", "").replace("\\]", "")
    v = v.replace(r"\left", "").replace(r"\right", "")

    # Mantiene il contenuto dei wrapper tipografici.
    for _ in range(8):
        previous = v
        v = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", v)
        if v == previous:
            break

    # Normalizza underscore escapati e pedici dopo aver rimosso i wrapper tipografici.
    v = re.sub(r"\\+_", "_", v)
    v = re.sub(r"([A-Za-z][A-Za-z0-9]*)_\{([^{}]+)\}", r"\1\2", v)
    v = re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)", r"\1\2", v)

    def replace_fractions(expression: str) -> str:
        """Sostituisce \frac{A}{B} rispettando parentesi graffe annidate."""
        marker = r"\frac"
        start = 0
        while True:
            pos = expression.find(marker, start)
            if pos < 0:
                return expression

            index = pos + len(marker)
            while index < len(expression) and expression[index].isspace():
                index += 1
            if index >= len(expression) or expression[index] != "{":
                start = index
                continue

            def read_group(open_index: int):
                depth = 0
                for cursor in range(open_index, len(expression)):
                    char = expression[cursor]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return expression[open_index + 1:cursor], cursor + 1
                return None, open_index

            numerator, after_num = read_group(index)
            if numerator is None:
                start = index + 1
                continue
            while after_num < len(expression) and expression[after_num].isspace():
                after_num += 1
            if after_num >= len(expression) or expression[after_num] != "{":
                start = after_num
                continue
            denominator, after_den = read_group(after_num)
            if denominator is None:
                start = after_num + 1
                continue

            replacement = f"({numerator})/({denominator})"
            expression = expression[:pos] + replacement + expression[after_den:]
            start = max(0, pos - 1)

    v = replace_fractions(v)
    v = v.replace(r"\sum", "sum")
    v = v.replace(r"\cdot", "*").replace(r"\times", "*")
    v = v.replace("×", "*").replace(r"\%", "%")
    v = re.sub(r"[${}`\\]", "", v)
    v = re.sub(r"[{}()]", "", v)
    v = re.sub(r"\s+", "", v).lower()
    return v

def _is_generic_formula_name(value: Any) -> bool:
    name = _formula_display_text(value or "", 180).strip().lower()
    return name in {
        "", "formula", "formula recuperata", "formula/metric",
        "formula from knowledge graph", "elemento recuperato", "score", "%", "level",
    }

def _formula_context_name(content: str, match_start: int, previous_end: int) -> str:
    """
    Ricava il nome associato all'equazione dal blocco testuale precedente.

    Usa solo caratteristiche tipografiche generali: prossimità, brevità,
    parentesi descrittive e presenza di un identificatore iniziale.
    """
    start = max(0, previous_end)
    segment = (content or "")[start:match_start]
    lines = [re.sub(r"\s+", " ", line).strip() for line in segment.splitlines()]
    lines = [line for line in lines if line and "=" not in line]
    lines = lines[-10:]

    # Combina solo continuazioni brevi che chiudono una parentesi aperta.
    combined = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.count("(") > line.count(")")
            and index + 1 < len(lines)
            and lines[index + 1].endswith(")")
            and len(lines[index + 1].split()) <= 4
        ):
            line = line + " " + lines[index + 1]
            index += 1
        combined.append(line)
        index += 1

    header_words = {
        "formula", "formulas", "formule", "equation", "equations",
        "modello", "modelli", "model", "models", "descrizione", "description",
    }

    best_name = ""
    best_score = -999
    for position, line in enumerate(combined):
        clean = line.strip(" -–—•|\t")
        if not clean or len(clean) > 100:
            continue
        words = re.findall(r"[A-Za-zÀ-ÿ0-9_.-]+", clean)
        if not words or len(words) > 12:
            continue
        if clean.lower() in header_words:
            continue

        score = 0
        first = words[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,30}", first):
            score += 2
        if "(" in clean:
            score += 3
        if re.fullmatch(r"[A-Z0-9_.-]{2,30}", first):
            score += 2
        if len(words) <= 6:
            score += 1
        if clean.endswith(('.', ':', ';')):
            score -= 2
        # A parità di qualità, preferisce il candidato più vicino.
        score += position / 100.0

        if score > best_score:
            best_score = score
            best_name = clean

    if best_score < 3:
        return ""
    if best_name.count("(") == best_name.count(")") + 1:
        best_name += ")"
    return best_name

def _formula_name_from_equation_line_v415(content: str, match_start: int) -> str:
    """Ricava il nome del modello dalla stessa riga/cella dell'equazione."""
    raw = content or ""
    line_start = raw.rfind("\n", 0, match_start) + 1
    prefix = raw[line_start:match_start].strip(" \t|;:-")
    if not prefix:
        return ""

    header_terms = {"modello", "model", "descrizione", "description", "formula"}
    if "|" in prefix:
        cells = [c.strip() for c in prefix.split("|") if c.strip()]
    elif "\t" in prefix:
        cells = [c.strip() for c in prefix.split("\t") if c.strip()]
    else:
        cells = [c.strip() for c in re.split(r"\s{2,}", prefix) if c.strip()]

    for cell in cells:
        clean = re.sub(r"^[-*•\d.()\s]+", "", cell).strip()
        if not clean or clean.lower() in header_terms or "=" in clean:
            continue
        if len(clean) <= 120:
            code_match = re.match(
                r"^([A-Za-z][A-Za-z0-9_.-]{1,30}(?:\s*\([^)]{1,80}\))?)",
                clean,
            )
            if code_match:
                candidate = code_match.group(1).strip()
                if candidate.count("(") > candidate.count(")"):
                    line_end = raw.find("\n", match_start)
                    if line_end >= 0:
                        next_end = raw.find("\n", line_end + 1)
                        next_line = raw[line_end + 1: next_end if next_end >= 0 else len(raw)]
                        continuation = re.split(r"\s{2,}|\t|\|", next_line.strip(), maxsplit=1)[0].strip()
                        if ")" in continuation and len(continuation) <= 60:
                            candidate += " " + continuation.split(")", 1)[0].strip() + ")"
                return candidate

    match = re.match(
        r"^([A-Za-z][A-Za-z0-9_.-]{1,30}(?:\s*\([^)]{1,80}\))?)",
        prefix,
    )
    return match.group(1).strip() if match else ""

def _extract_formula_rows_from_sources_core(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    Estrae formule e metriche dai SourceItem recuperati.

    Le equazioni testuali del documento sono considerate la rappresentazione
    primaria perché conservano il contesto e il nome associato. Le versioni
    LaTeX o Knowledge Graph restano disponibili come fallback e vengono poi
    deduplicate semanticamente.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()

    latex_pat = re.compile(
        r"(?<!\\)(\$\$.*?\$\$|\$[^$\n]{2,500}\$)",
        re.DOTALL,
    )
    explicit_equation_pat = re.compile(
        r"(?im)(?<![A-Za-z0-9_])"
        r"(?P<lhs>%|[A-Za-z][A-Za-z0-9_]{0,60})"
        r"\s*=\s*(?P<rhs>[^\n;|]{2,320})"
    )
    metric_line_pat = re.compile(
        r"(?i)\b(formula|formulas|formulae|equation|equations|formule|"
        r"equazione|equazioni|metric|metrics|metrica|metriche|indicator|"
        r"indicators|indicatore|indicatori|score|scoring|punteggio|"
        r"calculation|calcolo|mean time|tempo medio|index|indice|ratio|"
        r"coverage|copertura|maturity|maturità|severity|severità)\b"
    )

    for source in sources or []:
        content = source.content or ""
        filename = source.filename or "N/D"
        page = int(source.page or 0)
        source_type = normalize_source_type(getattr(source, "type", "") or "")

        # 1. Equazioni testuali: preservano nome e ordine del documento.
        previous_equation_end = 0
        for equation_match in explicit_equation_pat.finditer(content):
            lhs = re.sub(r"\s+", " ", equation_match.group("lhs")).strip()
            rhs = re.sub(r"\s+", " ", equation_match.group("rhs")).strip()
            if not lhs or len(rhs) < 2:
                continue

            raw_equation = f"{lhs} = {rhs}"
            identity = _canonical_formula_identity(raw_equation)
            if not identity:
                continue

            line_name = _formula_name_from_equation_line_v415(
                content,
                equation_match.start(),
            )
            context_name = line_name or _formula_context_name(
                content,
                equation_match.start(),
                previous_equation_end,
            )
            previous_equation_end = equation_match.end()

            key = ("equation", identity, filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rhs_words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", rhs)
            rhs_operators = re.findall(r"[+\-*/×÷^]|\b(?:sum|round)\s*\(", rhs, flags=re.IGNORECASE)
            textual_assignment = len(rhs_words) >= 3 and len(rhs_operators) <= 1

            rows.append({
                "name": context_name or lhs,
                "latex": (
                    raw_equation
                    if textual_assignment
                    else _formula_text_to_latex(lhs, rhs)
                ),
                "meaning": (
                    "Definizione testuale esplicita presente nella fonte recuperata."
                    if textual_assignment
                    else "Equazione esplicita presente nella fonte recuperata."
                ),
                "filename": filename,
                "page": page,
                "formula_origin": "document_equation",
            })

        # 2. Formule LaTeX esplicite: fallback per contenuti senza testo lineare.
        for latex_match in latex_pat.findall(content):
            latex = _normalize_latex_value(latex_match.strip())
            identity = _canonical_formula_identity(latex)
            if not identity:
                continue
            key = ("latex", identity, filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": _extract_left_name_from_equation(latex) or "Formula recuperata",
                "latex": _strip_math_wrappers(latex),
                "meaning": "Formula LaTeX esplicita presente nella fonte recuperata.",
                "filename": filename,
                "page": page,
                "formula_origin": "latex",
            })

        # 3. Formula nodes del Knowledge Graph: fallback strutturato.
        if "Formula from Knowledge Graph" in content or source_type == "formula":
            latex = ""
            plain = ""
            meaning = ""
            for line in content.splitlines():
                clean = line.strip()
                low = clean.lower()
                if low.startswith("latex:"):
                    latex = clean.split(":", 1)[1].strip()
                elif low.startswith("plain:"):
                    plain = clean.split(":", 1)[1].strip()
                elif low.startswith("meaning:"):
                    meaning = clean.split(":", 1)[1].strip()

            formula_value = latex or plain
            identity = _canonical_formula_identity(formula_value)
            if identity:
                key = ("kg", identity, filename.lower(), page)
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        "name": plain or _extract_left_name_from_equation(latex) or "Formula recuperata",
                        "latex": _strip_math_wrappers(_normalize_latex_value(formula_value)),
                        "meaning": meaning,
                        "filename": filename,
                        "page": page,
                        "formula_origin": "knowledge_graph",
                    })

        # 4. Metriche citate senza equazione esplicita.
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line or "").strip()
            if not line or not metric_line_pat.search(line) or "=" in line:
                continue

            name = "Metrica/indicatore citato"
            name_match = re.match(
                r"^[-*•\s]*([A-Za-zÀ-ÿ0-9_\-/ ]{2,80})\s*[:=–-]",
                line,
            )
            if name_match:
                name = name_match.group(1).strip()

            key = ("metric", name.lower(), filename.lower(), page, line[:120].lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": name,
                "latex": "formula esplicita non recuperata",
                "meaning": (
                    "Metrica o indicatore citato nella fonte; nessuna equazione "
                    "esplicita è stata individuata nella stessa riga."
                ),
                "filename": filename,
                "page": page,
                "formula_origin": "metric_text",
            })

            if len(rows) >= 60:
                return rows

    return rows

_formula_name_from_equation_line_v416 = _formula_name_from_equation_line_v415

def _formula_name_from_equation_line_v415(content: str, match_start: int) -> str:
    """
    Estrae il nome del modello da una riga/tabella in modo generalista.

    Mantiene il parser precedente e, solo se il risultato è generico, cerca
    l'identificatore strutturale più vicino composto da maiuscole/cifre, con
    eventuale descrizione tra parentesi. Non contiene nomi di modelli specifici.
    """
    previous = _formula_name_from_equation_line_v416(content, match_start)
    if previous and not _is_generic_formula_name(previous):
        return previous

    raw = content or ""
    line_start = raw.rfind("\n", 0, match_start) + 1
    previous_line_start = raw.rfind("\n", 0, max(0, line_start - 1)) + 1
    window_start = max(previous_line_start, match_start - 500)
    prefix = raw[window_start:match_start]

    header_tokens = {
        "FORMULA", "FORMULE", "MODEL", "MODELLI", "MODELLO",
        "DESCRIPTION", "DESCRIZIONE", "SCORE", "LEVEL",
    }
    candidates: List[Tuple[int, int, str]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?P<code>[A-Z][A-Z0-9_.-]{1,24})"
        r"(?:\s*\((?P<label>[^)\n]{1,80})\))?"
    )
    for match in pattern.finditer(prefix):
        code = match.group("code").strip()
        if code in header_tokens:
            continue
        # Esclude parole capitalizzate: il codice deve essere realmente
        # maiuscolo oppure contenere almeno una cifra strutturale.
        if code.upper() != code and not re.search(r"\d", code):
            continue
        label = re.sub(r"\s+", " ", match.group("label") or "").strip()
        candidate = f"{code} ({label})" if label else code
        same_line = int("\n" not in prefix[match.end():])
        distance = match_start - (window_start + match.end())
        candidates.append((same_line, -distance, candidate))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return previous or ""

def _formula_equation_pattern_v418() -> re.Pattern:
    """Pattern condiviso per equazioni testuali esplicite."""
    return re.compile(
        r"(?im)(?<![A-Za-z0-9_])"
        r"(?P<lhs>%|[A-Za-z][A-Za-z0-9_]{0,60})"
        r"\s*=\s*(?P<rhs>[^\n;|]{2,320})"
    )

def _structured_model_name_before_equation_v418(
    content: str,
    match_start: int,
    previous_equation_end: int = 0,
) -> str:
    """
    Trova l'identificatore strutturale più vicino prima di un'equazione.

    È indipendente dal dominio: riconosce codici maiuscoli/alfa-numerici con
    eventuale descrizione tra parentesi e funziona anche quando la riga della
    tabella è stata spezzata tra chunk o righe PDF.
    """
    raw = content or ""
    start = max(0, previous_equation_end, match_start - 1400)
    prefix = raw[start:match_start]

    # Se il frammento contiene un'altra equazione, considera soltanto la parte
    # successiva: impedisce di associare il nome della riga precedente.
    last_eq = max(prefix.rfind("="), prefix.rfind("$$"))
    if last_eq >= 0:
        prefix = prefix[last_eq + 1:]

    header_tokens = {
        "FORMULA", "FORMULE", "FORMULAS", "FORMULAE",
        "MODEL", "MODELS", "MODELLI", "MODELLO",
        "DESCRIPTION", "DESCRIZIONE", "SCORE", "LEVEL",
        "PERCENT", "PERCENTAGE", "TABLE", "TABELLA",
    }

    candidates: List[Tuple[int, int, int, str]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?P<code>[A-Z][A-Z0-9_.-]{1,24})"
        r"(?:\s*\((?P<label>[^)\n|]{1,100})\))?"
    )

    for match in pattern.finditer(prefix):
        code = (match.group("code") or "").strip()
        if not code or code in header_tokens:
            continue
        if code.upper() != code:
            continue
        if not re.search(r"[A-Z]", code):
            continue

        label = re.sub(r"\s+", " ", match.group("label") or "").strip()
        candidate = f"{code} ({label})" if label else code

        # I candidati con descrizione parentetica sono più affidabili; poi
        # prevalgono prossimità e presenza sulla stessa riga logica.
        tail = prefix[match.end():]
        same_line = int("\n" not in tail)
        has_label = int(bool(label))
        distance = len(prefix) - match.end()
        candidates.append((has_label, same_line, -distance, candidate))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    return candidates[0][3]

def _ordered_document_texts_v418(
    sources: List[SourceItem],
) -> Dict[Tuple[str, int], str]:
    """Unisce i chunk dello stesso documento/pagina nell'ordine di origine."""
    grouped: Dict[Tuple[str, int], List[SourceItem]] = {}
    for source in sources or []:
        filename = str(getattr(source, "filename", "") or "N/D")
        page = int(getattr(source, "page", 0) or 0)
        key = (filename.lower(), page)
        grouped.setdefault(key, []).append(source)

    result: Dict[Tuple[str, int], str] = {}
    for key, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda s: (
                int(getattr(s, "page_chunk_index", 0) or 0),
                0 if "PG_Enrich" in str(getattr(s, "db_origin", "")) else 1,
                str(getattr(s, "id", "")),
            ),
        )

        unique_parts: List[str] = []
        seen_parts = set()
        for item in ordered:
            part = str(getattr(item, "content", "") or "").strip()
            if not part:
                continue
            normalized = re.sub(r"\s+", " ", part).strip().lower()
            if normalized in seen_parts:
                continue
            seen_parts.add(normalized)
            unique_parts.append(part)

        result[key] = "\n".join(unique_parts)

    return result

def _formula_name_index_v418(
    sources: List[SourceItem],
) -> Dict[Tuple[str, int, str], str]:
    """Indicizza identity matematica -> nome modello per documento e pagina."""
    index: Dict[Tuple[str, int, str], str] = {}
    equation_pattern = _formula_equation_pattern_v418()

    for (filename_lower, page), content in _ordered_document_texts_v418(sources).items():
        previous_end = 0
        for match in equation_pattern.finditer(content):
            lhs = re.sub(r"\s+", " ", match.group("lhs") or "").strip()
            rhs = re.sub(r"\s+", " ", match.group("rhs") or "").strip()
            identity = _canonical_formula_identity(f"{lhs} = {rhs}")
            if not identity:
                previous_end = match.end()
                continue

            name = _structured_model_name_before_equation_v418(
                content,
                match.start(),
                previous_equation_end=previous_end,
            )
            previous_end = match.end()

            if not name or _is_generic_formula_name(name):
                continue

            key = (filename_lower, int(page), identity)
            current = index.get(key, "")
            # Preferisce il candidato con descrizione esplicita.
            if not current or ("(" in name and "(" not in current):
                index[key] = name

    return index

def _repair_formula_row_names_v418(
    rows: List[Dict[str, Any]],
    sources: List[SourceItem],
) -> List[Dict[str, Any]]:
    """Completa soltanto i nomi generici senza alterare formula o fonte."""
    name_index = _formula_name_index_v418(sources)
    repaired: List[Dict[str, Any]] = []

    for original in rows or []:
        row = dict(original)
        if _is_generic_formula_name(row.get("name")):
            filename = str(row.get("filename") or "").lower()
            page = int(row.get("page") or 0)
            identity = _canonical_formula_identity(row.get("latex") or "")
            candidate = name_index.get((filename, page, identity), "")
            if candidate:
                row["name"] = candidate
        repaired.append(row)

    return repaired

def _extract_formula_rows_from_sources_v417(sources: List[SourceItem]) -> List[Dict[str, Any]]:
    """
    Versione finale senza alias _ORIGINAL_*.

    Integra:
    - estrazione base di formule/metriche da SourceItem;
    - estensione v4.13 per aggiungere regole soglia da testo semplice.
    """
    rows = list(_extract_formula_rows_from_sources_core(sources) or [])
    seen = {
        (
            str(r.get("name") or "").lower(),
            _formula_display_text(r.get("latex") or "", 500).lower(),
            str(r.get("filename") or "").lower(),
            int(r.get("page") or 0),
        )
        for r in rows
    }

    for s in sources or []:
        content = getattr(s, "content", "") or ""
        filename = getattr(s, "filename", "N/D") or "N/D"
        page = int(getattr(s, "page", 0) or 0)

        for seg in _threshold_rule_segments_v413(content, max_segments=8):
            key = ("regola soglia", seg.lower()[:500], filename.lower(), page)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": "Regola soglia",
                "latex": seg,
                "meaning": "Criterio/soglia normativa recuperata; non è una formula computazionale.",
                "filename": filename,
                "page": page,
            })

    return rows

def extract_formula_rows_from_sources(
    sources: List[SourceItem],
) -> List[Dict[str, Any]]:
    """v4.18: estrazione precedente + completamento nomi a livello pagina."""
    rows = list(_extract_formula_rows_from_sources_v417(sources) or [])
    return _repair_formula_row_names_v418(rows, sources)


def _generate_formula_examples(
    query_text: str,
    formulas: List[Dict[str, str]],
) -> str:
    """Example generation is intentionally outside the deterministic core."""
    return ""



def answer_formula_strict(
    query_text: str,
    sources: Sequence[SourceItem],
) -> Optional[str]:
    """Build the deterministic Formula Strict answer from retrieved sources."""
    scoped_sources = _scope_formula_sources_to_requested_document_v414(
        query_text,
        list(sources or ()),
    )
    return _answer_formula_strict_core(query_text, scoped_sources)

__all__ = [
    "answer_formula_strict",
    "clean_formula_rows",
    "extract_formula_rows_from_sources",
]
