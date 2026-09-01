from __future__ import annotations

import re
from typing import Any, Sequence

from core.models import SourceItem

GLOSSARY_TERM_ALIASES: dict[str, list[str]] = {}

RAG_STOPWORDS = {
    "della", "delle", "degli", "dello", "dalla", "dalle", "dagli",
    "nella", "nelle", "negli", "nello", "alla", "alle", "agli",
    "sulla", "sulle", "sugli", "sullo",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
    "sono", "presenti", "presente", "ciascuna", "ciascuno", "tutti", "tutte",
    "quale", "quali", "cosa", "come", "dove", "quando", "perché", "perche",
    "spiega", "spiegami", "riporta", "riportale", "mostra", "mostrami",
    "dimmi", "elenca", "trova", "cerca", "voglio", "vorrei", "fammi",
    "riguardo", "inerente", "relativo", "secondo", "base", "basandoti",
    "what", "which", "where", "when", "explain", "show", "tell", "list",
    "find", "search", "report", "present", "available", "each", "about",
    "these", "those", "this", "that", "there", "their", "would", "could",
    "should", "please", "according", "regarding", "based", "give",
    "documento", "documenti", "file", "fonte", "fonti", "testo", "riferisce",
    "document", "documents", "source", "sources", "text", "context",
    "pagina", "pag", "page", "pages", "paragrafo", "sezione", "capitolo",
    "chapter", "section", "paragraph",
    "formula", "formule", "matematica", "matematiche", "latex", "concetto",
}

GRAPH_QUERY_NOISE_TERMS = {
    "mostra", "mostrami", "trova", "cerca", "elenca", "riporta",
    "descrivi", "spiega", "analizza", "verifica", "interroga",
    "show", "find", "search", "list", "report",
    "describe", "explain", "analyze", "analyse", "verify", "query",
    "neo4j", "cypher", "grafo", "grafi", "graph", "graphs",
    "nodo", "nodi", "node", "nodes",
    "arco", "archi", "edge", "edges",
    "relazione", "relazioni", "relation", "relations",
    "relationship", "relationships",
    "collegamento", "collegamenti", "link", "links",
    "connessione", "connessioni", "connection", "connections",
    "percorso", "path", "traversamento", "traversal",
    "multihop", "multi-hop",
    "tabella", "table", "markdown",
    "colonna", "colonne", "column", "columns",
    "riga", "righe", "row", "rows",
    "entità", "entita", "entity", "entities",
    "concetto", "concetti", "concept", "concepts",
    "documento", "documenti", "document", "documents",
    "fonte", "fonti", "source", "sources",
}


def _extract_search_tokens(query_text: str) -> list[str]:
    raw = re.findall(r"[A-Za-zÀ-ÿ0-9_\-]+", query_text or "")
    out: list[str] = []
    for token in raw:
        clean = token.strip().strip(".,:;!?()[]{}\"'")
        if not clean:
            continue
        is_acronym = clean.upper() == clean and 2 <= len(clean) <= 10
        is_mixed_acronym = bool(re.fullmatch(r"[A-Za-z]{1,5}\d{0,3}", clean)) and 2 <= len(clean) <= 10
        is_useful_word = len(clean) > 3
        if is_acronym or is_mixed_acronym or is_useful_word:
            out.append(clean.lower())
    return list(dict.fromkeys(out))


def _graph_relevant_tokens(query_text: str) -> list[str]:
    out: list[str] = []
    for token in _extract_search_tokens(query_text):
        clean = token.lower().strip()
        if not clean or clean in GRAPH_QUERY_NOISE_TERMS or clean in RAG_STOPWORDS or len(clean) < 3:
            continue
        out.append(clean)
    return list(dict.fromkeys(out))


def _extract_exact_phrases(query_text: str) -> list[str]:
    query = query_text or ""
    phrases = [
        value.strip().lower()
        for value in re.findall(r"[\"“'«]([^\"”'»]+)[\"”'»]", query)
        if len(value.strip()) > 2
    ]
    phrases.extend(value.lower() for value in re.findall(r"\b[A-Z]{2,8}\b", query))
    return list(dict.fromkeys(phrase for phrase in phrases if phrase))


def _md_cell(value: Any, max_len: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|")
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text


def _clean_graph_concept(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = text.strip(" \t\n\r.,;:!?()[]{}\"'“”‘’«»`")
    text = re.sub(
        r"^(?:funzione|function|concetto|concept|termine|term|voce|entity|entità)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    leading_noise = (
        r"^(?:(?:e|ed|and|or|oppure|o|il|lo|la|i|gli|le|un|una|uno|the|a|an)\s+"
        r"|(?:l|un)['’])+"
    )
    text = re.sub(leading_noise, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:e|ed|and|or|oppure|o)$", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\n\r.,;:!?()[]{}\"'“”‘’«»`")


def _split_relation_segment(segment: str) -> list[str]:
    segment = re.sub(r"[\n\r]+", " ", segment or "")
    segment = re.sub(
        r"\b(?:usando|using|tramite|through|rispetto a|against|return|do not|non usare|non rispondere)\b.*$",
        "",
        segment,
        flags=re.IGNORECASE,
    )
    raw_parts = re.split(
        r"\s*(?:,|;|\be\b|\bed\b|\band\b|\bo\b|\bor\b|\boppure\b|\bwith\b|\bcon\b|\bversus\b|\bvs\.?\b)\s*",
        segment,
        flags=re.IGNORECASE,
    )
    return [clean for part in raw_parts if (clean := _clean_graph_concept(part))]


def _canonical_graph_concept(concept: str) -> str:
    clean = _clean_graph_concept(concept).lower().strip()
    for canonical, aliases in GLOSSARY_TERM_ALIASES.items():
        for alias in [canonical, *aliases]:
            if clean == str(alias or "").lower().strip():
                return canonical.lower()
    return clean


def _graph_concept_aliases(concept: str) -> list[str]:
    raw = _clean_graph_concept(concept)
    aliases = [raw] if raw else []
    raw_l = raw.lower()

    for canonical, values in GLOSSARY_TERM_ALIASES.items():
        all_aliases = [canonical, *values]
        if any(raw_l == str(alias or "").lower().strip() for alias in all_aliases):
            aliases.extend(all_aliases)

    if raw_l in {
        "access control", "controllo accessi", "controllo degli accessi",
        "controlli di accesso", "controlli degli accessi",
    }:
        aliases.extend([
            "access control", "controllo accessi", "controllo degli accessi",
            "controlli di accesso", "controlli degli accessi",
        ])

    if raw_l in {
        "account privilegiati", "account privilegiato",
        "privileged account", "privileged accounts",
    }:
        aliases.extend([
            "account privilegiati", "account privilegiato", "privileged account",
            "privileged accounts", "utenze privilegiate", "utenze con privilegi",
            "privilegi amministrativi", "administrative privileges",
        ])

    if "accesso non autorizzato" in raw_l or "unauthorized access" in raw_l:
        aliases.extend([
            raw, "accesso non autorizzato", "rischio di accesso non autorizzato",
            "unauthorized access", "unauthorized access risk", "rischio di accesso",
        ])

    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        clean = _clean_graph_concept(str(alias or ""))
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def extract_graph_concepts_from_query(query_text: str, max_concepts: int = 8) -> list[str]:
    query = query_text or ""
    concepts: list[str] = []

    for item in re.findall(r"[\"“'‘«]([^\"”'’»]+)[\"”'’»]", query):
        clean = _clean_graph_concept(item)
        if len(clean) >= 2:
            concepts.append(clean)

    for pattern in (
        r"\b(?:tra|fra)\s+(.+?)(?:[\.?]|$)",
        r"\bbetween\s+(.+?)(?:[\.?]|$)",
        r"\bamong\s+(.+?)(?:[\.?]|$)",
    ):
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            concepts.extend(_split_relation_segment(match.group(1)))

    from_to = re.search(
        r"\b(?:da|from)\s+(.+?)\s+(?:a|to)\s+(.+?)(?:,|\s+passando\s+per|\s+through|\s+via|\.|\?|$)",
        query,
        flags=re.IGNORECASE,
    )
    if from_to:
        concepts.append(_clean_graph_concept(from_to.group(1)))
        concepts.append(_clean_graph_concept(from_to.group(2)))

    via = re.search(
        r"\b(?:passando\s+per|through|via)\s+(.+?)(?:[\.?]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if via:
        concepts.extend(_split_relation_segment(via.group(1)))

    concepts.extend(re.findall(r"\b[A-Z]{2,10}(?:[-_/][A-Z0-9]{1,10})?\b", query))
    concepts.extend(_extract_exact_phrases(query))

    if not concepts:
        concepts.extend(token for token in _graph_relevant_tokens(query) if len(token) >= 4)

    weak_single_terms = {
        "tutti", "tutto", "all", "each", "ogni",
        "fattore", "fattori", "factor", "factors",
        "access", "accesso", "control", "controllo", "controlli",
        "autenticazione", "authentication",
        "rischio", "risk", "utente", "user", "identity", "identità",
        "documenti", "documents", "normativi", "normative",
        "funzione", "function", "processo", "process",
        "catena", "chain", "percorso", "path", "passaggio", "step",
        "traversamento", "traversal", "grafo", "graph", "neo4j",
        "multi-hop", "multihop",
    }

    cleaned: list[str] = []
    seen_canonical: set[str] = set()
    for concept in concepts:
        clean = _clean_graph_concept(concept)
        if not clean:
            continue
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", clean))
        is_acronym = bool(re.fullmatch(r"[A-Z]{2,10}(?:[-_/][A-Z0-9]{1,10})?", clean))
        if not is_acronym and word_count == 1 and clean.lower() in weak_single_terms:
            continue
        canonical = _canonical_graph_concept(clean)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        cleaned.append(clean)
        if len(cleaned) >= max_concepts:
            break
    return cleaned


def _concept_in_text(concept: str, text_l: str) -> bool:
    if not concept or not text_l:
        return False
    for alias in _graph_concept_aliases(concept):
        clean = alias.lower().strip()
        if not clean:
            continue
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", alias))
        is_acronym = alias.upper() == alias and 2 <= len(alias) <= 10
        if is_acronym or word_count == 1:
            if re.search(rf"(^|[^a-z0-9]){re.escape(clean)}([^a-z0-9]|$)", text_l):
                return True
        elif clean in text_l:
            return True
    return False


def _best_alias_for_text(concept: str, text_l: str) -> str:
    for alias in _graph_concept_aliases(concept):
        if alias.lower().strip() in text_l:
            return alias
    return concept


def _evidence_snippet_for_pair(content: str, first: str, second: str, max_chars: int = 260) -> tuple[str, str]:
    if not content:
        return "", "non_supportata"
    text = re.sub(r"\s+", " ", content).strip()
    text_l = text.lower()
    first_l = first.lower()
    second_l = second.lower()

    for sentence in re.split(r"(?<=[\.!?])\s+", text):
        sentence_l = sentence.lower()
        if first_l in sentence_l and second_l in sentence_l:
            return _md_cell(sentence, max_chars), "supporto_testuale_forte"

    if first_l in text_l and second_l in text_l:
        first_pos = text_l.find(first_l)
        second_pos = text_l.find(second_l)
        start = max(0, min(first_pos, second_pos) - 120)
        end = min(len(text), max(first_pos, second_pos) + 180)
        return _md_cell(text[start:end].strip(), max_chars), "co_occorrenza_debole"

    return "", "non_supportata"


def _clean_relation_label(value: Any) -> str:
    text = str(value or "RELATES_TO").strip()
    if "{" in text:
        text = text.split("{", 1)[0].strip()
    text = re.sub(r"[^A-Z0-9_]+", "_", text.upper())
    return re.sub(r"_+", "_", text).strip("_")[:80] or "RELATES_TO"


def _parse_graph_relation_table_from_source(source: SourceItem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(source.content or "").splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 5:
            continue
        if "entità" in columns[0].lower() or "source" in columns[0].lower():
            continue
        rows.append({
            "source": columns[0],
            "relation": _clean_relation_label(columns[1]),
            "target": columns[2],
            "filename": columns[3],
            "page": columns[4],
            "evidence": "Relazione presente nel Knowledge Graph.",
            "status": "esplicita nel grafo",
            "source_id": str(source.id),
        })
    return rows


def answer_graph_relations_strict(
    query_text: str,
    sources: Sequence[SourceItem],
    max_rows: int = 10,
    used_source_ids: set[str] | None = None,
) -> str | None:
    concepts = extract_graph_concepts_from_query(query_text)
    if len(concepts) < 2:
        concepts = [token for token in _graph_relevant_tokens(query_text) if len(token) >= 4][:6]
    concepts = [concept for concept in concepts if len(str(concept).strip()) >= 3]
    if len(concepts) < 2:
        return None


    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    seen_pairs: set[tuple[str, ...]] = set()

    def is_edge_relevant(source: str, target: str) -> bool:
        source_hits = {
            _canonical_graph_concept(concept)
            for concept in concepts
            if _concept_in_text(concept, str(source or "").lower())
        }
        target_hits = {
            _canonical_graph_concept(concept)
            for concept in concepts
            if _concept_in_text(concept, str(target or "").lower())
        }
        return bool(source_hits and target_hits and len(source_hits | target_hits) >= 2)

    def add_row(row: dict[str, Any]) -> None:
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        relation = str(row.get("relation", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        source_canonical = _canonical_graph_concept(source)
        target_canonical = _canonical_graph_concept(target)
        pair_key = tuple(sorted([source_canonical, target_canonical])) + (status,)
        if "testual" in status or "co-occorrenza" in status:
            if pair_key in seen_pairs:
                return
            seen_pairs.add(pair_key)
        key = (
            source_canonical,
            relation.lower(),
            target_canonical,
            str(row.get("filename", "")).lower(),
            str(row.get("page", "")),
            status,
        )
        
        if key in seen:
            return

        seen.add(key)
        rows.append(row)

        source_id = str(
            row.get("source_id") or ""
        ).strip()

        if (
            source_id
            and used_source_ids is not None
        ):
            used_source_ids.add(source_id)

        source_id = str(
            row.get("source_id") or ""
        ).strip()

        if source_id and used_source_ids is not None:
            used_source_ids.add(source_id)

    for source in sources:
        if source.type == "graph_relations" or "Relazioni Neo4j trovate" in str(source.content or ""):
            for row in _parse_graph_relation_table_from_source(source):
                if not is_edge_relevant(str(row.get("source", "")), str(row.get("target", ""))):
                    continue
                add_row(row)
                if len(rows) >= max_rows:
                    break
        if len(rows) >= max_rows:
            break

    if len(rows) < max_rows:
        for source in sources:
            if not source.content or str(source.tier).upper() == "GRAPH" or source.type == "graph_relations":
                continue
            text_l = source.content.lower()
            matched = [concept for concept in concepts if _concept_in_text(concept, text_l)]
            if len(matched) < 2:
                continue
            for index, first in enumerate(matched):
                for second in matched[index + 1:]:
                    first_alias = _best_alias_for_text(first, text_l)
                    second_alias = _best_alias_for_text(second, text_l)
                    snippet, level = _evidence_snippet_for_pair(source.content, first_alias, second_alias)
                    if not snippet:
                        continue
                    status = (
                        "supporto testuale forte, non esplicita come arco"
                        if level == "supporto_testuale_forte"
                        else "co-occorrenza debole, non esplicita come arco"
                    )
                    add_row({
                        "source": first,
                        "relation": "collegamento testuale",
                        "target": second,
                        "filename": source.filename or "N/D",
                        "page": source.page or "",
                        "evidence": snippet,
                        "status": status,
                        "source_id": str(source.id),
                    })
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

    if not rows:
        for index, first in enumerate(concepts):
            for second in concepts[index + 1:]:
                add_row({
                    "source": first,
                    "relation": "collegamento richiesto",
                    "target": second,
                    "filename": "N/D",
                    "page": "",
                    "evidence": "Nessun arco Neo4j esplicito pertinente recuperato.",
                    "status": "non trovato",
                })
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

    table_lines = [
        "| Entità sorgente | Relazione | Entità target | Documento | Pagina | Evidenza | Stato |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows[:max_rows]:
        evidence = str(row.get("evidence", "")).replace("\n", " ").strip()
        if len(evidence) > 220:
            evidence = evidence[:220] + "..."
        table_lines.append(
            "| {source} | {relation} | {target} | {filename} | {page} | {evidence} | {status} |".format(
                source=str(row.get("source", "")).replace("\n", " ").strip(),
                relation=str(row.get("relation", "")).replace("\n", " ").strip(),
                target=str(row.get("target", "")).replace("\n", " ").strip(),
                filename=str(row.get("filename", "N/D")).replace("\n", " ").strip(),
                page=str(row.get("page", "")).replace("\n", " ").strip(),
                evidence=evidence,
                status=str(row.get("status", "")).replace("\n", " ").strip(),
            )
        )

    used_files: list[str] = []
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        if filename and filename != "N/D" and filename not in used_files:
            used_files.append(filename)
    sources_text = (
        "\n".join(f"- {filename}" for filename in used_files[:8])
        if used_files
        else "- Nessuna fonte documentale diretta utilizzabile."
    )

    statuses = [str(row.get("status", "")).lower() for row in rows]
    evidence_notes = [
        "- La tabella è stata costruita in modalità deterministica.",
        "- Sono state escluse relazioni Neo4j esplicite ma fuori target rispetto alle entità richieste.",
        "- Ogni riga distingue tra arco esplicito Neo4j, supporto testuale o relazione non trovata.",
    ]
    if any(status.strip() == "esplicita nel grafo" for status in statuses):
        evidence_notes.append("- Sono presenti relazioni esplicite recuperate dal Knowledge Graph.")
    else:
        evidence_notes.append("- Non sono stati recuperati archi Neo4j espliciti pertinenti; le relazioni riportate sono testuali o non trovate.")
    if any("testual" in status for status in statuses):
        evidence_notes.append("- Alcune relazioni sono supportate testualmente ma non risultano esplicite come archi Neo4j.")
    if any("non trovato" in status or "non supportata" in status for status in statuses):
        evidence_notes.append("- Alcuni collegamenti richiesti non risultano supportati dalle fonti recuperate.")

    limits = [
        "- Una relazione plausibile non viene trasformata in arco esplicito se non è presente nel grafo.",
        "- Relazioni vere ma non pertinenti alla domanda non sono usate come evidenza principale.",
        "- Se il grafo non contiene archi pertinenti, la risposta distingue supporto testuale, inferenza e relazione non trovata.",
    ]
    is_multihop = any(term in query_text.lower() for term in (
        "multi-hop", "multihop", "catena", "percorso", "path",
        "traversamento", "chain", "traversal",
    ))
    explicit_rows = [status for status in statuses if "esplicita" in status or "grafo" in status]
    if is_multihop and len(explicit_rows) < 2:
        limits.append(
            "- La richiesta è multi-hop, ma non sono stati recuperati abbastanza archi Neo4j espliciti "
            "per ricostruire una catena completa. La risposta riporta solo collegamenti testuali o assenze."
        )

    return (
        "**A) Risposta**\n\n"
        + "\n".join(table_lines)
        + "\n\n**B) Evidenze**\n\n"
        + "\n".join(evidence_notes)
        + "\n\n**C) Limiti / Conflitti**\n\n"
        + "\n".join(limits)
        + "\n\n**D) Fonti**\n\n"
        + sources_text
    )


__all__ = ["answer_graph_relations_strict", "extract_graph_concepts_from_query"]
