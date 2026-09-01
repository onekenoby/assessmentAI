# Batch 13.5 — Final Validation

## Scope della correzione

Il Batch 13.5 corregge i due failure reali rimasti dopo il Batch 13.4:

1. le risposte deterministiche richieste con `include_evaluation=true` restituiscono un `PASS` locale anche quando `EVAL_ENABLED=0`; il judge LLM resta completamente bypassato;
2. le fonti interne con provenance più lunga dei limiti del contratto HTTP vengono proiettate in modo bounded e stabile, evitando falsi `422 validation_error` durante la serializzazione della risposta.

Il mapping pubblico limita deterministicamente:

- `source_id` e `document_id`, con suffisso hash anti-collisione;
- `filename`;
- `excerpt`;
- `section_hint`;
- nome, tipo e relazione del `graph_context`;
- `classification` e `database_origin`.

Un `ValueError` successivo alla costruzione del comando viene ora classificato come errore interno sicuro, non come errore attribuibile al client. Gli errori reali del comando restano `HTTP 422`.

## Preflight

```powershell
..\venv\Scripts\python.exe -m py_compile core\rag_service.py api\routes_rag.py verify_batch13.py tests\test_rag_reflex_alignment_batch13_5.py
```

```powershell
..\venv\Scripts\python.exe -m pytest tests\test_rag_reflex_alignment_batch13_5.py -q
```

Risultato previsto:

```text
5 passed
```

## Collaudo reale 1234 ↔ 8888

```powershell
$env:EMBEDDING_MODEL_NAME="E:/Modelli/bge-m3"; $env:RERANKER_MODEL_NAME="E:/Modelli/ms-marco-reranker"; .\scripts\run_batch13_e2e.ps1 -BaseUrl http://127.0.0.1:8013 -ReportDir reports\batch13_5_org -RequireSecondOrganization
```

## Capacity stress

Eseguire soltanto dopo il PASS del collaudo standard:

```powershell
$env:EMBEDDING_MODEL_NAME="E:/Modelli/bge-m3"; $env:RERANKER_MODEL_NAME="E:/Modelli/ms-marco-reranker"; .\scripts\run_batch13_e2e.ps1 -BaseUrl http://127.0.0.1:8013 -ReportDir reports\batch13_5_capacity -RequireSecondOrganization -CapacityStress
```

## Risultati locali prima della consegna

```text
Test Batch 13.5 e regressioni mirate: 20 passed
Regressione cumulativa del runner: 167 passed
Suite autonoma completa: 196 passed, 22 skipped
Warning noti: 5 ast.Num deprecation
```

I 22 skip locali corrispondono ai test real-service, da eseguire sulla macchina con PostgreSQL, Qdrant, Neo4j e Ollama disponibili.
