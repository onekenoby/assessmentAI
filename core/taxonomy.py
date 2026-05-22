import os
import re
import re
import json
import time
from config import INBOX_DIR

TIER_FOLDERS = {
    "norm_frameworks_intl":    {"tier": "A", "content_type": "framework",   "source_kind": "global_standard"},
    "norm_regulations_local":  {"tier": "A", "content_type": "regulation",  "source_kind": "legal_requirement"},
    "norm_audit_guidelines":   {"tier": "A", "content_type": "guideline",   "source_kind": "best_practice"},
    "gov_infosec_policies":    {"tier": "B", "content_type": "policy",      "source_kind": "internal_governance"},
    "gov_it_procedures":       {"tier": "B", "content_type": "procedure",   "source_kind": "internal_governance"},
    "gov_bcdr_plans":          {"tier": "B", "content_type": "bcdr_plan",   "source_kind": "internal_governance"},
    "gov_hr_policies":         {"tier": "B", "content_type": "hr_policy",   "source_kind": "internal_governance"},
    "gov_org_charts_roles":    {"tier": "B", "content_type": "org_chart",   "source_kind": "internal_governance"},
    "tech_evidence_sys_config":     {"tier": "C", "content_type": "sys_config",     "source_kind": "technical_evidence"},
    "tech_evidence_audit_logs":     {"tier": "C", "content_type": "audit_log",      "source_kind": "technical_evidence"},
    "tech_evidence_vuln_patching":  {"tier": "C", "content_type": "vuln_report",    "source_kind": "technical_evidence"},
    "tech_evidence_backup_restore": {"tier": "C", "content_type": "backup_log",     "source_kind": "technical_evidence"},
    "tech_evidence_net_access":     {"tier": "C", "content_type": "network_config", "source_kind": "technical_evidence"},
    "org_evidence_mgmt_reviews":    {"tier": "C", "content_type": "mgmt_review",    "source_kind": "organizational_evidence"},
    "org_evidence_training_aware":  {"tier": "C", "content_type": "training_log",   "source_kind": "organizational_evidence"},
    "org_evidence_incident_rep":    {"tier": "C", "content_type": "incident_ticket","source_kind": "organizational_evidence"},
    "org_evidence_risk_mgmt":       {"tier": "C", "content_type": "risk_register",  "source_kind": "organizational_evidence"},
    "legal_evidence_vendor_contracts":{"tier": "C","content_type":"vendor_contract","source_kind": "legal_evidence"},
    "legal_evidence_data_privacy":    {"tier": "C","content_type":"privacy_record", "source_kind": "legal_evidence"},
    "legal_evidence_nda_clauses":     {"tier": "C","content_type":"nda_agreement",  "source_kind": "legal_evidence"},
    "phys_evidence_env_controls":   {"tier": "C", "content_type": "env_control",    "source_kind": "physical_evidence"},
    "phys_evidence_access_logs":    {"tier": "C", "content_type": "access_log",     "source_kind": "physical_evidence"},
}

DEFAULT_TIER_META = {"tier": "B", "content_type": "reference", "source_kind": "internal"}
DEFAULT_ONTOLOGY = "generic"

TOPIC_PATTERNS = {
    "Governance_Policies": ["policy", "procedura", "standard", "guideline", "linea guida", "regolamento", "direttiva", "manuale", "organigramma", "ruoli e responsabilità", "segregation of duties", "sod"],
    "Risk_Management": ["rischio", "risk", "minaccia", "threat", "vulnerabilità", "vulnerability", "mitigazione", "risk assessment", "trattamento del rischio", "risk register", "matrice dei rischi"],
    "Compliance_Audit": ["iso 27001", "iso 27002", "gdpr", "nis2", "dora", "nist", "audit", "assessment", "conformità", "compliance", "non-conformità", "certificazione", "ispezione", "evidenza"],
    "Business_Continuity_DR": ["business continuity", "disaster recovery", "bcp", "drp", "backup", "ripristino", "restore", "rto", "rpo", "resilienza", "copia di sicurezza", "continuità operativa"],
    "Incident_Management": ["incidente", "incident", "data breach", "violazione", "anomalia", "segnalazione", "incident response", "triage", "compromissione"],
    "Access_Identity_Control": ["accesso", "access control", "autenticazione", "mfa", "password", "identità", "iam", "credenziali", "privilegi", "active directory", "logon", "sso"],
    "Technical_Network_Security": ["firewall", "crittografia", "encryption", "siem", "log", "monitoraggio", "patch", "endpoint", "antivirus", "malware", "rete", "vlan", "vpn", "vulnerability scan", "penetration test", "pt"],
    "HR_Awareness_Security": ["formazione", "training", "awareness", "consapevolezza", "assunzione", "onboarding", "offboarding", "nda", "codice etico", "dipendente", "phishing", "risorse umane"],
    "Vendor_SupplyChain_Security": ["fornitore", "vendor", "supply chain", "terze parti", "sla", "contratto", "outsourcing", "subfornitore", "cloud provider", "dpa"],
    "Physical_Environmental_Security": ["sicurezza fisica", "controlli ambientali", "badge", "videosorveglianza", "cctv", "sala server", "datacenter", "estintori", "ups", "condizionamento"]
}

KG_KEYWORDS = [
    "policy", "procedura", "procedure", "guideline", "standard", "normativa", "framework",
    "rischio", "risk", "vulnerabilità", "minaccia", "threat", "mitigazione",
    "incidente", "incident", "breach", "data breach", "violazione",
    "accesso", "access", "autenticazione", "mfa", "password", "crittografia", "encryption",
    "backup", "restore", "ripristino", "disaster recovery", "business continuity",
    "audit", "assessment", "conformità", "compliance", "non-conformità",
    "asset", "server", "network", "firewall", "log", "monitoraggio",
    "formazione", "training", "awareness", "dipendente", "fornitore", "vendor"
]

GATEKEEPER_CONCEPTS = [
    "Information security, ISO 27001, cybersecurity, data protection, confidentiality, integrity, availability, risk assessment, risk treatment, asset management, access control",
    "Compliance, GDPR, NIS2, DORA, regulatory requirements, legal obligations, data privacy, personal data, data breaches, incident response, reporting obligations",
    "IT Governance, policies, procedures, standards, guidelines, organizational structure, roles and responsibilities, segregation of duties, management review",
    "Business Continuity, Disaster Recovery, BCP, DRP, backup strategies, RTO, RPO, resilience, crisis management, redundancy",
    "Technical controls, firewalls, encryption, cryptography, MFA, multi-factor authentication, vulnerability management, patch management, SIEM, logging, monitoring",
    "Physical security, environmental controls, access badges, CCTV, secure areas, clear desk policy, clean screen policy",
    "Human resources security, awareness training, onboarding, offboarding, phishing simulations, NDA, non-disclosure agreements, background checks",
    "Vendor management, third-party risk, SLA, supply chain security, cloud security, SOC 2, audits, continuous monitoring"
]

def infer_topics_regex(text: str, max_topics: int = 6) -> list[str]:
    if not text: return []
    t = text
    scores = {}
    for topic, patterns in TOPIC_PATTERNS.items():
        topic_score = 0
        for pat in patterns:
            try:
                topic_score += len(re.findall(pat, t, flags=re.IGNORECASE))
            except re.error:
                continue
        if topic_score > 0:
            scores[topic] = topic_score
    if not scores: return []
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:max_topics]]

def _safe_read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def read_sidecar_meta(file_path: str) -> dict:
    sidecar = file_path + ".meta.json"
    if os.path.exists(sidecar):
        return _safe_read_json(sidecar)
    return {}

def dispatch_document(file_path: str, root_dir: str) -> dict:
    rel = os.path.relpath(root_dir, INBOX_DIR).replace("\\", "/")
    parts = [p for p in rel.split("/") if p and p != "."]
    tier_key = parts[0].upper() if len(parts) >= 1 else ""
    if len(parts) >= 2:
        ontology = parts[1].lower()
    else:
        ontology = TIER_FOLDERS.get(tier_key, {}).get("content_type", DEFAULT_ONTOLOGY)
    base = dict(TIER_FOLDERS.get(tier_key, DEFAULT_TIER_META))
    base["ontology"] = ontology
    fname = os.path.basename(file_path)
    base["topics"] = infer_topics_regex(fname)[:6]
    if base.get("tier") == "C" and not base.get("effective_date"):
        base["effective_date"] = time.strftime("%Y-%m-%d")
    side = read_sidecar_meta(file_path)
    if isinstance(side, dict) and side:
        base.update(side)
    return base

def ensure_inbox_structure(inbox_dir: str):
    structure = {
        "TIER_A_NORMATIVE": ["norm_frameworks_intl", "norm_regulations_local", "norm_audit_guidelines"],
        "TIER_B_GOVERNANCE": ["gov_infosec_policies", "gov_it_procedures", "gov_bcdr_plans", "gov_hr_policies", "gov_org_charts_roles"],
        "TIER_C_EVIDENCES": [
            "1_technical_evidences/tech_evidence_sys_config", "1_technical_evidences/tech_evidence_audit_logs",
            "1_technical_evidences/tech_evidence_vuln_patching", "1_technical_evidences/tech_evidence_backup_restore",
            "1_technical_evidences/tech_evidence_net_access", "2_organizational_evidences/org_evidence_mgmt_reviews",
            "2_organizational_evidences/org_evidence_training_aware", "2_organizational_evidences/org_evidence_incident_rep",
            "2_organizational_evidences/org_evidence_risk_mgmt", "3_legal_vendor_evidences/legal_evidence_vendor_contracts",
            "3_legal_vendor_evidences/legal_evidence_data_privacy", "3_legal_vendor_evidences/legal_evidence_nda_clauses",
            "4_physical_security_evidences/phys_evidence_env_controls", "4_physical_security_evidences/phys_evidence_access_logs"
        ]
    }
    for tier_folder, subfolders in structure.items():
        for sub in subfolders:
            tier_path = os.path.join(inbox_dir, tier_folder, sub)
            os.makedirs(tier_path, exist_ok=True)
    
    # --- DA AGGIUNGERE in core/taxonomy.py sotto a KG_KEYWORDS ---


_KG_PAT = re.compile(r'\b(' + '|'.join(KG_KEYWORDS) + r')\b', re.IGNORECASE)