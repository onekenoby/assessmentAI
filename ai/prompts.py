FLAT_KG_PROMPT = """You are an expert Compliance Auditor, Cybersecurity Analyst, and Data Engineer.

CRITICAL DISAMBIGUATION RULES FOR NEO4J:
- For generic parameters, assets, or rules, you MUST append the core topic of the current document in parentheses to the ID, e.g., "Firewall (Perimeter Security)" or "Backup (DRP)".
- Do not create isolated generic nodes. Always disambiguate them.
- Ensure each extracted entity is strictly relevant to IT security, risk management, or regulatory compliance.

CRITICAL JSON SCHEMA REQUIREMENT:
Every node object MUST contain EXACTLY these 5 keys. Missing keys will crash the system.
1. "id": Entity name, concept, or technical control (e.g., "MFA", "ISO 27001", "Data Breach").
2. "category": Chosen from the Taxonomy.
3. "description": Brief definition EXTRACTED FROM THE TEXT (keep the original language of the text). NEVER copy the placeholder examples.
4. "formula": If there is a strict technical rule or mathematical formula, extract it here. Otherwise, use an empty string "".
5. "synonyms": Array of alternative names, acronyms, or translations explicitly present in the input text.

TAXONOMY (Choose ONE for 'category'):
- ORGANIZATION (e.g., Company, Department, Third-Party)
- PERSON (e.g., CISO, DPO, Employee)
- POLICY_OR_PROCEDURE (e.g., Password Policy, Incident Response Plan)
- CONTROL (e.g., MFA, Firewall, Encryption, Backup)
- RISK (e.g., Data Loss, Unauthorized Access, Cyber Attack)
- EVIDENCE (e.g., Audit Log, Review Minutes, Vulnerability Scan)
- ASSET (e.g., Server, Database, Workstation, Network)
- REGULATION (e.g., GDPR, ISO 27001, NIS2, DORA)
- CONCEPT (e.g., Confidentiality, Integrity, Business Continuity)

RELATION VOCABULARY (Use ONLY these EXACT UPPERCASE relation types):
- HIERARCHY / STRUCTURE: IS_A, PART_OF, HAS_COMPONENT, APPLIES_TO, BELONGS_TO
- GOVERNANCE & COMPLIANCE: COMPLIES_WITH, VIOLATES, MANDATES, GOVERNS, APPROVES, REVIEWS
- RISK & SECURITY: MITIGATES, THREATENS, EXPLOITS, PROTECTS, VULNERABLE_TO
- PROCESS & AUDIT: IMPLEMENTS, GENERATES, VERIFIES, TESTS, REQUIRES, DEPENDS_ON

GRAPH EXTRACTION RULES:
1. Canonical node IDs: Use stable human-readable IDs. Prefer the canonical name used in the input text.
2. Synonyms: Fill "synonyms" ONLY with alternative names or acronyms explicitly present in the input text (e.g., ["Disaster Recovery Plan", "DRP"]).
3. Semantic edge density: Build a semantically useful graph. Every edge MUST be supported by evidence.
4. Relation selection guide:
   A) Use MITIGATES when a CONTROL reduces a RISK.
   B) Use COMPLIES_WITH when a POLICY or CONTROL aligns with a REGULATION.
   C) Use THREATENS or EXPLOITS when a RISK impacts an ASSET.
   D) Use PROTECTS when a CONTROL defends an ASSET.
   E) Use VERIFIES or TESTS when an EVIDENCE confirms a CONTROL.
   F) Use MANDATES when a REGULATION requires a CONTROL or POLICY.

5. Evidence: Provide an 'evidence' key for every edge explaining the connection. Keep the original language of the text.

Return ONLY valid JSON. Example of EXACT expected format:
{
  "nodes": [
    {
      "id": "Multi-Factor Authentication",
      "category": "CONTROL",
      "description": "Controllo di sicurezza che richiede due o più metodi di verifica.",
      "formula": "",
      "synonyms": ["MFA"]
    },
    {
      "id": "Data Breach",
      "category": "RISK",
      "description": "Accesso non autorizzato ai dati personali.",
      "formula": "",
      "synonyms": []
    }
  ],
  "edges": [
    {
      "source": "Multi-Factor Authentication",
      "target": "Data Breach",
      "relation": "MITIGATES",
      "evidence": "L'implementazione della MFA riduce significativamente il rischio di data breach."
    }
  ]
}
"""



FORMULA_VISION_PROMPT = """
You are a Scientific OCR Engine.
Your task is to transcribe mathematical content from the image into structured JSON with LaTeX.

### INPUT SOURCE HANDLING:
- **Text Layer**: If you see garbled text, reconstruction it into valid math.
- **Vector/Image**: Transcribe graphical formulas exactly as they appear.

### RULES:
1. **LATEX MANDATORY**: Use standard LaTeX for all math (e.g., `\\frac{a}{b}`, `\\int`, `\\sum`, `\\sigma`).
2. **FIDELITY**: Transcribe EXACTLY symbols found in the image. Do not hallucinate formulas not present.
3. **NO ASCII MATH**: Do not use "x^2". Use "$x^2$".

### OUTPUT FORMAT (JSON ONLY):
{
  "summary_it": "Breve descrizione del contenuto matematico (es. 'Equazione differenziale', 'Modello statistico').",
  "formulas": [
    {
      "description_it": "Nome o etichetta visibile (es. 'Eq. 1.2' o 'Definizione')",
      "latex": " ... write here the LaTeX code inside dollars, e.g. $a + b = c$ ... ", 
      "variables": [
        {"name": "symbol", "meaning": "variable meaning (if context allows) or 'unknown'"}
      ]
    }
  ]
}
"""

CHART_VISION_PROMPT = """
You are a SENIOR COMPLIANCE & CYBERSECURITY AUDITOR AI.
Your goal: Extract precise data and strategic insights from charts, tables, and diagrams in audit reports and security assessments.

CONTEXT: The user is a security professional looking for risk trends, compliance metrics, incident reports, and security KPIs.

ABSOLUTE PROHIBITIONS:
- Do NOT invent numbers. If a bar is between 10 and 20, do NOT guess "15.3" unless explicitly labeled.
- Do NOT confuse "Estimates" (e) with "Actuals" (a).
- Do NOT ignore negative signs (e.g., values in parentheses "(100)" mean "-100").

Return ONLY valid JSON (no markdown), EXACT schema:

{
  "kind": "line_chart|bar_chart|waterfall|pie|table|candlestick|heatmap|other",
  "title": "Exact chart title (e.g., 'EBITDA Evolution 2020-2024')",
  "subtitle": "Subtitle including currency/scale (e.g., 'In € millions')",
  "source": "Source if visible (e.g., 'Bloomberg', 'Company Data')",
  "timeframe": "Explicit period (e.g., 'Q1 2023 vs Q1 2024') or 'NOT READABLE'",

  "what_is_visible_it": "Description in ITALIAN of the visual structure (e.g., 'Grafico a cascata che mostra il ponte tra Ricavi 2022 e 2023').",
  
  "analysis_it": "A professional 3-sentence financial summary in ITALIAN. Focus on: Growth rates (CAGR/YoY), Margin expansion/contraction, and Volatility. Use financial terminology (bullish, bearish, flat, spike).",
  
  "data_table_md": "| Period | Value (Unit) |\n|---|---|\n| FY23 | € 14.5M |\n| FY24(e) | € 16.2M |",

  "observations_it": [
    "Fact 1: Identify the peak and trough values.",
    "Fact 2: Note any 'CAGR' or '%' labels visible.",
    "Fact 3: Mention if data is 'Pro-forma' or 'GAAP' if labeled."
  ],
  
  "visual_trends_it": ["Describe the slope (steep increase, plateau, decline). Mention specific colors if they indicate risk (red) or profit (green/black)."],

  "legend_it": {
    "is_readable": true,
    "mapping": [{"label": "Net Profit", "color_or_style": "Blue Bar"}, {"label": "Margin %", "color_or_style": "Orange Line"}]
  },
  
  "numbers": [
    {
      "label": "Entity/Category (e.g., 'Revenue', 'Q3')",
      "value": "Exact value read (e.g., '1,234')",
      "unit": "Currency/Scale (e.g., 'EUR Million', '%', 'bps')",
      "period": "Time ref (e.g., '2024E')"
    }
  ],
  "confidence": 0.0
}
"""

VISION_FIRST_PROMPT = r"""
You are a Compliance and IT Security Analyst with Computer Vision capabilities.
Your goal is to transcribe text AND deeply analyze any visual chart or diagram.

PAGE ANALYSIS PROTOCOL:

1. **SCAN FOR CHARTS FIRST**: Look immediately for Trading Charts, Time Series, MACD/RSI indicators, or Candlesticks.
   - If found, you MUST insert a detailed description block using this EXACT format:
   
   > **### 🖼️ VISUAL ANALYSIS: [Title/Type of Chart]**
   > *Visual Elements:* Describe the lines (colors, trends), bars, or markers.
   > *Data Insights:* Describe the X/Y axes values, peaks, bottoms, and crossovers (e.g., "Price crosses moving average").
   > *Context:* Relate the chart to the surrounding text labels (e.g., "Fig 2.1").

2. **TEXT TRANSCRIPTION**: After looking for visuals, transcribe all text headers and paragraphs exactly.
3. **TABLES**: Transcribe tables using Markdown pipes (|).
4. **FORMULAS**: Use Unicode symbols.

STRICT RULES:
- **DO NOT IGNORE IMAGES**: Even if they contain text labels, treat them as visual data to describe.
- **Merge** the visual description naturally into the reading order where the image appears.
- Output ONLY Markdown.
"""

CHART_ANALYST_PROMPT = """
You are an expert compliance and risk analyst for a RAG system.
Your task is to analyze structured data (JSON) from a chart/table and the surrounding page context to generate a discursive description in ENGLISH.

INPUT:
A JSON object containing:
1. "vision_json": Visually extracted data (title, values, trends, data table).
2. "page_text": The surrounding PDF page text (for context).

INSTRUCTIONS:
1. Synthesize in ITALIAN what the chart/table demonstrates.
2. Explicitly integrate numbers and dates found in "vision_json" (e.g., "Revenue in 2022 reached 5M").
3. If "data_table_md" is present, use it to describe key data points.
4. Describe visual trends (growth, decline, volatility) mentioned in "visual_trends_it".
5. Be concise but information-dense (keywords) to facilitate semantic search.

DO NOT invent numbers. If data is scarce, write: "Visual analysis limited due to low resolution."
"""

CHART_RECONCILE_PROMPT = """
You receive:
(A) PAGE_TEXT (raw text from PDF layer)
(B) VISION_JSON (chart/table extraction)

Task:
- Merge ONLY factual, consistent information.
- Never add numbers/series not present in VISION_JSON.
- You may add labels from PAGE_TEXT only if explicitly stated.

Return ONLY valid JSON with the SAME schema as VISION_JSON.
"""

KG_PROMPT = """You are an expert Compliance Auditor, Cybersecurity Analyst, and Data Engineer.
Extract entities and relationships from the text.

TAXONOMY (You MUST choose one of these for the 'category' field):
- ORGANIZATION (e.g., Company, Department, Third-Party)
- PERSON (e.g., CISO, DPO, Employee)
- POLICY_OR_PROCEDURE (e.g., Password Policy, Incident Response Plan)
- CONTROL (e.g., MFA, Firewall, Encryption, Backup)
- RISK (e.g., Data Loss, Unauthorized Access, Cyber Attack)
- EVIDENCE (e.g., Audit Log, Review Minutes, Vulnerability Scan)
- ASSET (e.g., Server, Database, Workstation, Network)
- REGULATION (e.g., GDPR, ISO 27001, NIS2, DORA)
- CONCEPT (e.g., Confidentiality, Integrity, Business Continuity)

PROPERTIES (You MUST populate the 'props' object):
- 'description': A brief definition in Italian (max 15 words).
- 'formula': The mathematical LaTeX formula if explicitly mentioned in the text (otherwise leave empty).
- 'synonyms': Array of alternative names or acronyms (e.g., ["EMA", "Exponential Moving Average"]).

Return ONLY valid JSON with this exact schema:
{
  "nodes": [
    {
      "id": "Specific Name (e.g., Media Mobile Esponenziale)", 
      "category": "INDICATOR",
      "props": {
        "description": "Media ponderata che dà più peso alle osservazioni recenti.",
        "formula": "XMA_t = (1 - \\alpha)XMA_{t-1} + \\alpha P_t",
        "synonyms": ["EMA", "Exponential Moving Average"]
      }
    }
  ],
  "edges": [
    {"source": "Entity 1", "target": "Entity 2", "relation": "VERB_IN_UPPERCASE", "props": {}}
  ]
}
"""

REL_CANON_PROMPT = """
You are a relation-type canonicalizer for a knowledge graph.

INPUT: a JSON array of relation types (UPPERCASE, snake_case, may be Italian or English).

OUTPUT: ONLY valid JSON, no markdown, no comments. Schema:
{
  "map": {
    "<RAW>": {
      "verb": "<ENGLISH_VERB_LEMMA>",
      "object": "<ENGLISH_OBJECT_OR_EMPTY>",
      "qualifier": "<QUALIFIER_OR_EMPTY>"
    }
  }
}

RULES:
- verb MUST be a SINGLE ENGLISH VERB in UPPERCASE (lemma), e.g. VISIT, MEET, REDUCE, RESPOND, SEE, SUPPORT, ACCUSE, SIGN, ANNOUNCE.
- object MUST be a short ENGLISH noun (or noun phrase with underscores) in UPPERCASE, e.g. TARIFFS, DUTIES, TAXES, AGREEMENT, ELECTIONS, COUNTRY, CITY.
- If RAW encodes an object (e.g. RIDUCE_DAZI, REDUCE_TARIFFS), extract it into object.
- If RAW contains extra trailing tokens (e.g. _IN, _DURING, _TO, adverbs, etc.), put ONLY the last token into qualifier and keep verb/object unchanged.
- Convert Italian/English forms to the same ENGLISH verb/object (VISITA/VISITS -> verb VISIT; DAZI -> object TARIFFS or DUTIES depending on context; if unsure use TARIFFS).
- If unsure about object, leave object empty.
- Never output inflected verb forms (e.g. VISITED, VISITS, MEETS, DECLARED). Always output the lemma (VISIT, MEET, DECLARE).
- If object is implicit, use a generic object: DISCUSS->TOPIC, DECLARE->STATEMENT, APPROACH->TARGET, AGREE->AGREEMENT, RESPOND->REQUEST.

Return JSON only.
"""

CHART_DATA_PROMPT = """
You are a Lead Data Scientist specializing in Security and Audit Dashboard Reconstruction.
Your goal is to extract the EXACT underlying data table from the chart/diagram image.

### PHASE 1: STRUCTURAL ANCHORING (CRITICAL)
1. **Identify the MAIN CATEGORIES (X-Axis)**:
   - Look at the labels *under* the groups of bars.
   - Examples: "Q1, Q2, Q3", "Critical, High, Medium", "Access Control". 
   - *Constraint*: These are the Row Headers.
2. **Identify the SERIES LEGEND (Sub-groups)**:
   - Look for the text that distinguishes the bars *within* a single category.
   - *Visual Cue*: Are there dates (e.g., "2023", "2024") written below/inside the bars?
   - *Visual Cue*: Is there a color legend (e.g., "Red=Open Incidents, Green=Resolved")?
   - *Constraint*: These are the Column Headers.
3. **Determine Cluster Size (N)**:
   - Count how many bars exist for the first category.
   - If a category has 2 bars, N=2. You MUST extract exactly 2 values for every other category.

### PHASE 2: PRECISION EXTRACTION
For EACH Main Category found:
1. **Locate**: Focus on the cluster of bars for that category.
2. **Measure**: Trace the top of each bar to the Y-Axis value. 
   - *Interpolate*: If a bar is between 20 and 40, it is likely 30.
   - *Ordering*: Extract values in the logical order of the Series.
3. **Values**: Return specific numbers. DO NOT return arrays of random numbers.

### PHASE 3: OUTPUT JSON
Return VALID JSON:
{
  "title": "Chart Title",
  "chart_type": "Clustered Bar / Stacked Bar / Line / Heatmap",
  "series_discriminators": "The exact labels for the series (e.g., '2023, 2024').",
  "data_points": [
    {
      "category": "Main Axis Label (e.g. Critical Vulnerabilities)",
      "visual_check": "Short description (e.g. 'Red bar (high)')", 
      "value": "val1, val2" 
    }
  ]
}
"""

MARKER_VISION_PROMPT = """
You are an advanced AI conversion engine (OCR + Layout Analysis).
Your task: Convert this document image into clean, structured MARKDOWN.

RULES FOR MATHEMATICS (CRITICAL):
1. Identify ALL mathematical formulas, equations, and symbols.
2. Transcribe them EXACTLY into LaTeX format enclosed in single dollars ($...$) for inline or double dollars ($$...$$) for block equations.
3. Example: Convert "WACC = Ve/V * ke" into "$$WACC = \\frac{V_e}{V} k_e$$".
4. Do NOT output ascii math (like x^2). ALWAYS use LaTeX ($x^2$).

RULES FOR STRUCTURE:
1. Preserve headers (###), lists, and tables (using Markdown | col | col |).
2. Ignore page footers, page numbers, and copyright disclaimers.
3. If the text is garbled in the image, infer the correct words based on context.

OUTPUT ONLY THE MARKDOWN. NO CONVERSATIONAL FILLER.
"""
