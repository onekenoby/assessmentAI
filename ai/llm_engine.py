import os
import re
import time
import json
import base64
import requests
import subprocess
import shutil
import torch
from typing import Optional
from threading import Lock
from sentence_transformers import SentenceTransformer, util
from ollama import chat, ChatResponse
from config import *
from core.taxonomy import GATEKEEPER_CONCEPTS

_GK_ANCHOR_EMBEDDINGS = None
embedder = None
OLLAMA_CALL_LOCK = Lock()

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return embedder

def ai_gatekeeper_decision(text: str) -> bool:
    if not text or len(text) < KG_MIN_LEN: return False
    global _GK_ANCHOR_EMBEDDINGS
    embed = get_embedder()
    if _GK_ANCHOR_EMBEDDINGS is None:
        _GK_ANCHOR_EMBEDDINGS = embed.encode(GATEKEEPER_CONCEPTS, convert_to_tensor=True)
    chunk_embedding = embed.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(chunk_embedding, _GK_ANCHOR_EMBEDDINGS)
    max_score = float(torch.max(scores))
    if max_score > 0.3: print(f"   [GK] Score: {max_score:.3f} | Text: {text[:50]}...")
    return max_score >= 0.30

def ai_gatekeeper_decision_from_vec(vec, threshold: float = 0.38) -> bool:
    global _GK_ANCHOR_EMBEDDINGS
    if vec is None: return False
    embed = get_embedder()
    if _GK_ANCHOR_EMBEDDINGS is None:
        _GK_ANCHOR_EMBEDDINGS = embed.encode(GATEKEEPER_CONCEPTS, convert_to_tensor=True)
    if not isinstance(vec, torch.Tensor):
        vec_t = torch.tensor(vec, dtype=torch.float32)
    else:
        vec_t = vec.float()
    if vec_t.dim() == 1: vec_t = vec_t.unsqueeze(0)
    scores = util.cos_sim(vec_t, _GK_ANCHOR_EMBEDDINGS)
    max_score = float(torch.max(scores))
    if max_score > 0.3: print(f"   [GK] Score(vec): {max_score:.3f}")
    return max_score >= threshold

def safe_json_extract(raw: str):
    if raw is None: return None
    s = str(raw)
    fence_pattern = "`" * 3 + r"(?:json)?\s*([\s\S]*?)\s*" + "`" * 3
    fence = re.search(fence_pattern, s, flags=re.IGNORECASE)
    if fence: s = fence.group(1)
    s = "".join(ch for ch in s if ch in "\t\n\r" or ord(ch) >= 32)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    def _first_balanced_json(text: str):
        starts = [i for i, ch in enumerate(text) if ch in "{["]
        for start in starts:
            open_ch = text[start]
            close_ch = "}" if open_ch == "{" else "]"
            depth, in_str, esc = 0, False, False
            for j in range(start, len(text)):
                c = text[j]
                if in_str:
                    if esc: esc = False
                    elif c == "\\": esc = True
                    elif c == '"': in_str = False
                    continue
                else:
                    if c == '"': in_str = True; continue
                    if c == open_ch: depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0: return text[start:j+1]
        return None
    cand = _first_balanced_json(s)
    if not cand: return None
    try: return json.loads(cand)
    except Exception: pass
    repaired = re.sub(r",\s*([}\]])", r"\1", cand).strip()
    repaired = re.sub(r'\\(?=[^"\\/bfnrtu])', r'\\\\', repaired)
    try: return json.loads(repaired)
    except Exception as e:
        print(f"      [JSON-REPAIR-FAILED] Impossibile recuperare il JSON: {e}")
        return None

def force_unload_ollama(model_name: str):
    if not model_name: return
    try:
        url = "http://localhost:11434/api/generate"
        requests.post(url, json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=2)
        time.sleep(2.0)
    except Exception: pass

def force_restart_ollama(num_parallel: str = "1") -> bool:
    print(f"🔄 Resetting Ollama Server (Target Parallelism={num_parallel})...")
    try:
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/f", "/im", "ollama_app.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    except Exception: pass
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_MAX_LOADED_MODELS"] = "2"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    ollama_path = shutil.which("ollama") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    try:
        subprocess.Popen([ollama_path, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=False)
    except Exception as e:
        print(f"   ❌ Errore critico avvio Ollama: {e}")
        return False
    for i in range(20):
        try:
            if requests.get("http://127.0.0.1:11434/api/tags", timeout=2).status_code == 200:
                print(f"   ✨ Ollama is READY (Parallel={num_parallel})")
                return True
        except: time.sleep(1)
    print("   ⚠️ Ollama non ha risposto entro il timeout, ma potrebbe essere attivo.")
    return True

def ensure_ollama_parallel(num_parallel="4"):
    os.environ["OLLAMA_NUM_PARALLEL"] = num_parallel
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"
    try:
        if requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200: return
    except requests.exceptions.ConnectionError: pass
    subprocess.Popen(["ollama", "serve"], env=os.environ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(10):
        try:
            if requests.get("http://localhost:11434/api/tags").status_code == 200: return
        except: time.sleep(2)

def llm_chat(prompt: str, text: str, model: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
    last_err = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            with OLLAMA_CALL_LOCK:
                resp: ChatResponse = chat(model=model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}], options={"temperature": LLM_TEMPERATURE, "num_predict": int(max_tokens) if max_tokens else LLM_MAX_TOKENS})
            return resp["message"]["content"] or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2.0, 0.35 * (attempt + 1)))
    print(f"   ⚠️ llm_chat failed dopo retry | model={model} | err={last_err}")
    return ""

def llm_chat_multimodal(prompt: str, image_bytes: bytes, model: str, max_tokens: int = 4000, num_ctx: int = 4096, response_format_json: bool = False, force_json: Optional[bool] = None) -> str:
    if not image_bytes: return ""
    if force_json is not None: response_format_json = bool(force_json)
    try: img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e: return ""
    payload = {"model": model, "prompt": prompt, "images": [img_b64], "options": {"temperature": 0.0, "num_ctx": int(num_ctx), "num_predict": int(max_tokens) if max_tokens else 4000}, "stream": False}
    if response_format_json: payload["format"] = "json"
    with OLLAMA_CALL_LOCK:
        last_err = None
        for attempt in range(OLLAMA_RETRIES + 1):
            try:
                r = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=OLLAMA_TIMEOUT_S)
                r.raise_for_status()
                return r.json().get("response", "") or ""
            except Exception as e:
                last_err = e
                time.sleep(min(3.0, 0.75 * (attempt + 1)))
        print(f"   ⚠️ Vision generate failed: {last_err}")
        return ""

def llm_extract_kg(filename: str, page_no, text: str, model_name: str):
    from ai.prompts import FLAT_KG_PROMPT
    if not text or len(text) < 50: return [], []
    try:
        resp = chat(model=model_name, messages=[{"role": "system", "content": FLAT_KG_PROMPT}, {"role": "user", "content": f"Extract entities and relations in JSON format from this text:\n\n{text[:3500]}"}], format="json", options={"temperature": 0.0, "num_predict": 2500, "num_ctx": 4096})
        js = safe_json_extract(resp.get("message", {}).get("content", "").strip())
        if not js or not isinstance(js, dict) or not js.get("nodes"): raise ValueError("JSON incompleto o senza nodi")
    except Exception as e:
        print(f"   ⚠️ [KG-RETRY] Pag {page_no}: Fallito JSON mode. Riprovo in RAW mode...")
        try:
            resp = chat(model=model_name, messages=[{"role": "system", "content": FLAT_KG_PROMPT + "\nIMPORTANT: Return ONLY raw JSON. No markdown blocks."}, {"role": "user", "content": f"Extract entities and relations from this text:\n\n{text[:3000]}"}], options={"temperature": 0.1, "num_predict": 2048, "num_ctx": 4096})
            js = safe_json_extract(resp.get("message", {}).get("content", ""))
            if not js or not isinstance(js, dict): return [], []
        except Exception as e2: return [], []
    raw_nodes = js.get("nodes", js.get("entities", []))
    raw_edges = js.get("edges", js.get("relationships", []))
    nodes = [n for n in raw_nodes if isinstance(n, dict)]
    edges = [e for e in raw_edges if isinstance(e, dict)]
    return nodes, edges

# --- DA AGGIUNGERE IN ai/llm_engine.py ---

def _load_rel_canon_cache() -> dict:
    try:
        with open(REL_CANON_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _save_rel_canon_cache(cache: dict) -> None:
    try:
        with open(REL_CANON_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _get_rel_canon_map(rel_types: set[str]) -> dict:
    from ai.prompts import REL_CANON_PROMPT # Import necessario
    cache = _load_rel_canon_cache()

    missing = [rt for rt in sorted(rel_types) if rt and rt not in cache]
    if not missing:
        return cache

    raw = llm_chat(
        REL_CANON_PROMPT,
        json.dumps(missing, ensure_ascii=False),
        LLM_MODEL_NAME,
        max_tokens=REL_CANON_MAX_TOKENS
    )

    js = safe_json_extract(raw) or {}
    mapping = (js.get("map") or {}) if isinstance(js, dict) else {}

    for k, v in mapping.items():
        if not isinstance(v, dict):
            continue

        kk = (k or "").strip().upper()
        verb = (v.get("verb") or "").strip().upper()
        obj = (v.get("object") or "").strip().upper()
        qual = (v.get("qualifier") or "").strip().upper()

        if kk and verb:
            cache[kk] = {"verb": verb, "object": obj, "qualifier": qual}

    _save_rel_canon_cache(cache)
    return cache

RELTYPE_OK = re.compile(r"^[A-Z][A-Z_]{2,60}$")

def _safe_reltype(t: str) -> str:
    t = (t or "").strip().upper()
    if RELTYPE_OK.match(t):
        return t
    return "RELATES_TO"

def _cheap_lemma_en(verb: str) -> str:
    v = (verb or "").strip().upper()

    if len(v) <= 4:
        return v

    if v.endswith("ING") and len(v) > 6:
        v = v[:-3]
    elif v.endswith("ED") and len(v) > 5:
        v = v[:-2]
    elif v.endswith("S") and len(v) > 5:
        v = v[:-1]

    if v.endswith("ISHE") and len(v) >= 8:
        v = v[:-1]  
    if v.endswith("CUS") and len(v) >= 6:
        v = v + "S"

    return v

def canonicalize_edges_to_verb_object(edges: list[dict]) -> list[dict]:
    if not edges:
        return edges

    rel_types = set(
        (e.get("relation") or "").strip().upper()
        for e in edges
        if e.get("relation")
    )
    canon_map = _get_rel_canon_map(rel_types)

    out: list[dict] = []

    for e in edges:
        raw_rel = (e.get("relation") or "").strip().upper()
        raw_rel = raw_rel.replace("__", "_").strip("_")
        if not raw_rel:
            out.append(e)
            continue

        m = canon_map.get(raw_rel) or {}
        verb = (m.get("verb") or raw_rel).strip().upper()
        obj = (m.get("object") or "").strip().upper()
        qual = (m.get("qualifier") or "").strip().upper()

        verb = _cheap_lemma_en(verb)

        ee = dict(e)
        props = dict(ee.get("props") or {})

        canon_type = _safe_reltype(verb)

        raw_audit = raw_rel.replace("__", "_").strip("_")
        if not RELTYPE_OK.match(raw_audit):
            raw_audit = raw_rel

        props.setdefault("raw_relation", raw_audit)
        props.setdefault("canon_verb", verb)

        if obj:
            props.setdefault("canon_object", obj)

        if not obj and canon_type == "VISIT":
            props.setdefault("canon_object", "PLACE")

        if qual:
            props.setdefault("qualifier", qual)

        ee["props"] = props
        ee["relation"] = canon_type
        out.append(ee)

    return out

def canonicalize_edges_by_base_presence(edges: list[dict]) -> list[dict]:
    if not edges:
        return edges

    rel_types = set((e.get("relation") or "").strip().upper() for e in edges if e.get("relation"))
    out = []

    for e in edges:
        rel = (e.get("relation") or "").strip().upper()
        rel = rel.replace("__", "_").strip("_")
        if not rel or "_" not in rel:
            out.append(e)
            continue

        parts = rel.split("_")
        if parts:
            parts[0] = _cheap_lemma_en(parts[0])
            rel = "_".join(parts)

        base, suffix = rel.rsplit("_", 1)

        if base in rel_types and len(base) >= 6 and 1 <= len(suffix) <= 12:
            ee = dict(e)
            ee["relation"] = base
            props = dict(ee.get("props") or {})
            props.setdefault("raw_relation", rel)
            props.setdefault("qualifier", suffix)
            ee["props"] = props
            out.append(ee)
        else:
            out.append(e)

    return out

def generate_chart_analysis_it(chart_json: dict, page_text: str = "") -> str:
    from ai.prompts import CHART_ANALYST_PROMPT
    if not isinstance(chart_json, dict):
        return ""
    try:
        payload = {
            "vision_json": chart_json,
            "page_text": (page_text or "")[:2500]
        }
        resp = chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": CHART_ANALYST_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            format='json', 
            options={"temperature": 0.1, "num_predict": 600, "num_ctx": 4096},
        )
        return (resp.get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        print(f"   ⚠️ Chart Analysis Error: {e}")
        return ""
    
    