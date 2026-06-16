#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_pdf_to_md.py

Conversione conservativa PDF -> Markdown per ingestion RAG.

Principio:
- niente dizionari adattativi di parole;
- niente correzioni specifiche tipo "Regu lation" -> "Regulation";
- solo normalizzazione Unicode, legature, soft-hyphen, spazi, header/footer;
- eventuali fix aggressivi dei glifi Word/PDF sono opzionali e disattivati di default.

Uso:
    python preprocess_pdf_to_md.py input.pdf

Uso con output:
    python preprocess_pdf_to_md.py input.pdf --out output.md

Uso mantenendo le note:
    python preprocess_pdf_to_md.py input.pdf --out output.md --keep-footnotes

Uso con fix aggressivi per PDF generati male da Word:
    python preprocess_pdf_to_md.py input.pdf --out output.md --aggressive-glyph-fix
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    print("ERRORE: PyMuPDF non installato. Installa con: pip install pymupdf", file=sys.stderr)
    raise

import unicodedata
from collections import Counter


def is_math_char(ch: str) -> bool:
    code = ord(ch)

    # Mathematical Operators
    if 0x2200 <= code <= 0x22FF:
        return True

    # Supplemental Mathematical Operators
    if 0x2A00 <= code <= 0x2AFF:
        return True

    # Misc Mathematical Symbols-A
    if 0x27C0 <= code <= 0x27EF:
        return True

    # Misc Mathematical Symbols-B
    if 0x2980 <= code <= 0x29FF:
        return True

    # Greek letters often used in formulas
    if 0x0370 <= code <= 0x03FF:
        return True

    return False


def is_allowed_non_ascii(ch: str) -> bool:
    """
    Caratteri non ASCII ma generalmente leciti in documenti normativi/tecnici.
    """
    if is_math_char(ch):
        return True

    category = unicodedata.category(ch)

    # Lettere accentate / lettere unicode
    if category.startswith("L"):
        return True

    # Simboli valuta, es. €, £
    if category == "Sc":
        return True

    # Punteggiatura unicode, es. –, —, “, ”
    if category.startswith("P"):
        return True

    # Simboli tecnici generici
    if category.startswith("S"):
        return True

    return False


def audit_and_clean_chars(text: str, remove_suspicious: bool = False) -> tuple[str, dict]:
    """
    Logga tutti i caratteri non ASCII.
    Rimuove solo caratteri di controllo/corrotti.
    Non elimina lettere accentate o simboli matematici validi.
    """

    non_ascii_counter = Counter()
    suspicious_counter = Counter()
    cleaned_chars = []

    for ch in text:
        code = ord(ch)

        if code > 127:
            non_ascii_counter[ch] += 1

            if not is_allowed_non_ascii(ch):
                suspicious_counter[ch] += 1

                if remove_suspicious:
                    continue

        # Elimina caratteri di controllo non utili
        category = unicodedata.category(ch)
        if category.startswith("C") and ch not in ("\n", "\t"):
            suspicious_counter[ch] += 1
            if remove_suspicious:
                continue

        cleaned_chars.append(ch)

    report = {
        "non_ascii": dict(non_ascii_counter),
        "suspicious": dict(suspicious_counter),
    }

    return "".join(cleaned_chars), report


# Fix sicuri/generici: legature tipografiche standard e caratteri di controllo.
SAFE_GLYPH_FIXES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "\u00ad": "",      # soft hyphen
    "\ufffd": "",      # replacement char
    "￾": "",           # carattere spurio visto in alcuni PDF
}

# Fix NON generici: utili solo per alcuni PDF generati/stampati da Word.
# Disattivati di default.
AGGRESSIVE_GLYPH_FIXES = {
    "Ɵ": "ti",
    "Ʃ": "tt",
    "Ō": "ft",
    "ō": "ft",
}


HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}\s*$"),
    re.compile(r"^L\s+\d+/\d+\s*$", re.IGNORECASE),
    re.compile(r"^Official Journal of the European Union\s*$", re.IGNORECASE),
    re.compile(r"^EN\s*$", re.IGNORECASE),
]

ARTICLE_RE = re.compile(r"^Article\s+\d+[A-Za-z]?$", re.IGNORECASE)
TITLE_RE = re.compile(r"^TITLE\s+[IVXLCDM]+$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+$", re.IGNORECASE)


def apply_glyph_fixes(text: str, aggressive: bool = False) -> str:
    """
    Applica solo fix Unicode/glyph generici.
    I fix aggressivi sono opzionali per evitare correzioni adattative.
    """
    for src, dst in SAFE_GLYPH_FIXES.items():
        text = text.replace(src, dst)

    if aggressive:
        for src, dst in AGGRESSIVE_GLYPH_FIXES.items():
            text = text.replace(src, dst)

    return text


def normalize_unicode(text: str, aggressive_glyph_fix: bool = False) -> str:
    text = apply_glyph_fixes(text, aggressive=aggressive_glyph_fix)
    text = unicodedata.normalize("NFKC", text)

    text, char_report = audit_and_clean_chars(
        text,
        remove_suspicious=False
    )
        
    if char_report["suspicious"]:
        print("Caratteri sospetti trovati:")
        for ch, count in char_report["suspicious"].items():
            print(f"  U+{ord(ch):04X} {repr(ch)} count={count} name={unicodedata.name(ch, 'UNKNOWN')}")

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "\xa0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Mantiene newline e tab, elimina altri caratteri di controllo.
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C")
    )

    return text


def clean_line(line: str, aggressive_glyph_fix: bool = False) -> str:
    line = normalize_unicode(line, aggressive_glyph_fix=aggressive_glyph_fix)
    line = re.sub(r"[ \t]+", " ", line)
    line = line.strip()

    # Fix generici di punteggiatura/lista.
    line = re.sub(r"^(\d+)\.([A-Z])", r"\1. \2", line)
    line = re.sub(r"^(\([a-z]\))([A-Z])", r"\1 \2", line)
    line = re.sub(r"\s+%", " %", line)

    return line


def is_header_footer_line(line: str) -> bool:
    return any(p.match(line) for p in HEADER_FOOTER_PATTERNS)


def split_body_and_footnotes(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Rimuove footer/header tipici della Gazzetta Ufficiale UE.
    È conservativo: taglia solo se il pattern appare nella parte bassa pagina.
    """
    for i, line in enumerate(lines):
        if is_header_footer_line(line) and i > max(8, int(len(lines) * 0.45)):
            return lines[:i], lines[i:]
    return lines, []


def is_markdown_heading(line: str) -> bool:
    return bool(
        TITLE_RE.match(line)
        or CHAPTER_RE.match(line)
        or ARTICLE_RE.match(line)
        or line in {
            "HAVE ADOPTED THIS REGULATION:",
            "SUBJECT MATTER, SCOPE AND DEFINITIONS",
            "CONDITIONS FOR CARRYING OUT STATUTORY AUDIT OF PUBLIC INTEREST ENTITIES",
            "THE APPOINTMENT OF STATUTORY AUDITORS OR AUDIT FIRMS BY PUBLIC-INTEREST ENTITIES",
            "SURVEILLANCE OF THE ACTIVITIES OF STATUTORY AUDITORS AND AUDIT FIRMS CARRYING OUT STATUTORY AUDIT OF PUBLIC-INTEREST ENTITIES",
        }
    )


def md_heading(line: str) -> str:
    clean = line.rstrip(":")
    if TITLE_RE.match(clean):
        return f"\n## {clean}\n"
    if CHAPTER_RE.match(clean):
        return f"\n### {clean}\n"
    if ARTICLE_RE.match(clean):
        return f"\n### {clean}\n"
    if clean.isupper() and len(clean) > 10:
        return f"\n## {clean.title()}\n"
    return f"\n## {clean}\n"


def starts_new_paragraph(line: str) -> bool:
    return bool(
        re.match(r"^\(\d+\)\s*$", line)
        or re.match(r"^\(\d+\)\s+", line)
        or re.match(r"^\d+\.\s*$", line)
        or re.match(r"^\d+\.\s+", line)
        or re.match(r"^\([a-z]\)\s+", line)
        or re.match(r"^\([ivxlcdm]+\)\s+", line, re.IGNORECASE)
    )


def merge_lines_to_markdown(lines: List[str], aggressive_glyph_fix: bool = False) -> str:
    out: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            para = " ".join(current)
            para = re.sub(r"\s+", " ", para).strip()
            if para:
                out.append(para)
            current = []

    for raw in lines:
        line = clean_line(raw, aggressive_glyph_fix=aggressive_glyph_fix)

        if not line:
            flush()
            continue

        if is_header_footer_line(line):
            continue

        if is_markdown_heading(line):
            flush()
            out.append(md_heading(line))
            continue

        if starts_new_paragraph(line):
            flush()
            current.append(line)
            continue

        current.append(line)

    flush()
    md = "\n\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def extract_pdf_to_md(
    input_pdf: Path,
    keep_footnotes: bool = False,
    aggressive_glyph_fix: bool = False,
    page_markers: bool = True,
) -> str:
    doc = fitz.open(str(input_pdf))

    parts: List[str] = [
        f"# {input_pdf.stem}",
        "",
        "<!--",
        f"source_file: {input_pdf.name}",
        "preprocess: pdf_text_to_markdown",
        "mode: conservative",
        "note: generated from PDF text layer, no OCR/Vision used",
        "-->",
        "",
    ]

    all_footnotes: List[str] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        raw_text = page.get_text("text") or ""
        raw_text = normalize_unicode(raw_text, aggressive_glyph_fix=aggressive_glyph_fix)

        lines = [
            clean_line(x, aggressive_glyph_fix=aggressive_glyph_fix)
            for x in raw_text.splitlines()
        ]
        lines = [x for x in lines if x]

        body_lines, footnote_lines = split_body_and_footnotes(lines)

        body_md = merge_lines_to_markdown(
            body_lines,
            aggressive_glyph_fix=aggressive_glyph_fix,
        )

        if body_md:
            if page_markers:
                parts.append(f"\n<!-- page: {page_index + 1} -->\n")
            parts.append(body_md)

        if keep_footnotes and footnote_lines:
            clean_notes = [x for x in footnote_lines if not is_header_footer_line(x)]
            if clean_notes:
                all_footnotes.append(f"\n<!-- footnotes page: {page_index + 1} -->\n")
                all_footnotes.append(
                    merge_lines_to_markdown(
                        clean_notes,
                        aggressive_glyph_fix=aggressive_glyph_fix,
                    )
                )

    if keep_footnotes and all_footnotes:
        parts.append("\n## Footnotes\n")
        parts.extend(all_footnotes)

    md = "\n\n".join(p.strip("\n") for p in parts if p is not None)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    return md.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PDF text layer to clean Markdown for RAG ingestion."
    )
    parser.add_argument("input_pdf", help="Path del PDF di input")
    parser.add_argument("--out", "-o", help="Path del file Markdown di output")
    parser.add_argument("--keep-footnotes", action="store_true", help="Mantiene le note a fondo pagina")
    parser.add_argument(
        "--aggressive-glyph-fix",
        action="store_true",
        help="Applica fix opzionali per PDF generati male da Word. Disattivato di default.",
    )
    parser.add_argument(
        "--no-page-markers",
        action="store_true",
        help="Non inserisce i marker <!-- page: N --> nel Markdown.",
    )

    args = parser.parse_args()

    input_pdf = Path(args.input_pdf).expanduser().resolve()
    if not input_pdf.exists():
        print(f"ERRORE: file non trovato: {input_pdf}", file=sys.stderr)
        return 1

    output_md = Path(args.out).expanduser().resolve() if args.out else input_pdf.with_suffix(".clean.md")
    output_md.parent.mkdir(parents=True, exist_ok=True)

    md = extract_pdf_to_md(
        input_pdf,
        keep_footnotes=args.keep_footnotes,
        aggressive_glyph_fix=args.aggressive_glyph_fix,
        page_markers=not args.no_page_markers,
    )

    output_md.write_text(md, encoding="utf-8")

    print("OK - Markdown generato")
    print(f"Input : {input_pdf}")
    print(f"Output: {output_md}")
    print(f"Chars : {len(md)}")
    print(f"Mode  : conservative")
    print(f"Aggressive glyph fix: {args.aggressive_glyph_fix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
