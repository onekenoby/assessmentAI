# Batch 13.4 — Final E2E closure

Batch 13.4 corregge i due failure osservati sul Batch 13.3:

1. i rami deterministici espongono `model: "not-used"` nel contratto HTTP;
2. l'API avviata dal runner usa un profilo Ollama bounded, così un timeout di generazione termina prima del timeout HTTP del test e il fallback Batch 10 può essere restituito.

## Profilo E2E predefinito

- timeout HTTP del test: `300s`;
- timeout Ollama dell'API di collaudo: `180s`;
- `num_predict`: `512`;
- tentativi Ollama: `1`;
- durante `-CapacityStress`, timeout Ollama massimo: `60s`.

Questi override valgono soltanto per l'istanza Uvicorn avviata dal runner. La configurazione normale dell'applicazione resta governata dalle variabili ambiente.

## Preflight

```powershell
..\venv\Scripts\python.exe -m py_compile core\config.py core\generation.py core\rag_service.py verify_batch13.py e2e_report.py tests\test_rag_reflex_alignment_batch13_4.py
..\venv\Scripts\python.exe -m pytest tests\test_rag_reflex_alignment_batch13_4.py -q
```

Risultato previsto: `5 passed`.

## Collaudo reale organizzazioni 1234 e 8888

```powershell
$env:EMBEDDING_MODEL_NAME="E:/Modelli/bge-m3"; $env:RERANKER_MODEL_NAME="E:/Modelli/ms-marco-reranker"; .\scripts\run_batch13_e2e.ps1 -BaseUrl http://127.0.0.1:8013 -ReportDir reports\batch13_4_org -RequireSecondOrganization
```

## Capacity stress

```powershell
$env:EMBEDDING_MODEL_NAME="E:/Modelli/bge-m3"; $env:RERANKER_MODEL_NAME="E:/Modelli/ms-marco-reranker"; .\scripts\run_batch13_e2e.ps1 -BaseUrl http://127.0.0.1:8013 -ReportDir reports\batch13_4_capacity -RequireSecondOrganization -CapacityStress
```

## Override espliciti

```powershell
.\scripts\run_batch13_e2e.ps1 -BaseUrl http://127.0.0.1:8013 -ApiLlmTimeoutSeconds 120 -ApiLlmNumPredict 384 -ApiLlmMaxAttempts 1
```

Il timeout Ollama viene comunque limitato sotto il timeout HTTP del collaudo.
