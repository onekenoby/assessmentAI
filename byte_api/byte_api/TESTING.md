# Test della Byte API

La suite è divisa in quattro livelli.

## 1. Test simulati e unitari

Non richiedono PostgreSQL e non modificano alcun database.

```powershell
cd E:\Dev\assessmentAI\byte_api
python -m pip install -r requirements-dev.txt
.\run_tests.ps1
```

Comando equivalente:

```powershell
python -m pytest -m "not integration" `
  --cov=api `
  --cov=core `
  --cov=main `
  --cov=byte_engine `
  --cov-report=term-missing
```

La release è stata validata con:

```text
136 passed, 4 deselected
Coverage totale: 98%
```

La suite copre:

- endpoint multipart `corpus` ed `evidence`;
- mapping dei parametri CLI nei campi HTTP;
- TIER A/GLOBAL e TIER B-C/ACCOUNT;
- validazione di file, dimensione, MIME type e identificativi;
- API key e security headers;
- startup, shutdown, liveness e readiness;
- commit, rollback e chiusura connessioni;
- creazione o riuso di ontologia, blob, documento e contesto;
- ripristino del job `CONTENT_INGESTION` per documenti `PENDING` senza job aperto;
- funzione ufficiale `fn_upload_response_evidence`;
- gestione controllata degli errori PostgreSQL.

## 2. Preflight reale del Database A

È non distruttivo: apre una connessione e verifica la presenza delle tabelle richieste nello schema `rag_ingestion`.

Configurare prima:

```powershell
$env:PG_HOST = "127.0.0.1"
$env:PG_PORT = "5433"
$env:SOURCE_PG_DB = "assessment_gestio_tier"
$env:PG_USER = "admin"
$env:PG_PASS = "admin_password"
```

Poi:

```powershell
cd E:\Dev\assessmentAI\byte_api
.\run_preflight_real.ps1
```

Il test verifica:

```text
rag_ingestion.rag_file_blob
rag_ingestion.rag_document
rag_ingestion.rag_document_context
rag_ingestion.rag_ingestion_job
```

## 3. Test HTTP reale dell'API avviata

Avviare prima il servizio:

```powershell
cd E:\Dev\assessmentAI\byte_api
python -m uvicorn main:app --host 0.0.0.0 --port 8020 --workers 1
```

Da un secondo terminale:

```powershell
cd E:\Dev\assessmentAI\byte_api
$env:BYTE_API_BASE_URL = "http://127.0.0.1:8020"
.\run_api_integration_tests.ps1
```

Questo test chiama soltanto:

```text
GET /health/live
GET /health/ready?deep=true
```

Non carica file e non crea job.

Se è configurata una API key:

```powershell
$env:BYTE_API_KEY = "la-tua-chiave"
```

## 4. Test distruttivo corpus

Questo test carica realmente un file e crea o riusa record nel Database A. Va eseguito solo con un documento di prova.

Con l'API già avviata:

```powershell
cd E:\Dev\assessmentAI\byte_api

$env:BYTE_API_BASE_URL = "http://127.0.0.1:8020"
$env:BYTE_API_RUN_DESTRUCTIVE = "1"
$env:BYTE_API_TEST_CORPUS_FILE = "E:\Dev\assessmentAI\data\documento_test.pdf"
$env:BYTE_API_TEST_TIER = "C"
$env:BYTE_API_TEST_ORGANIZATION_ID = "9999"
$env:BYTE_API_TEST_USER_ID = "123"
$env:BYTE_API_TEST_AREA = "IDENTIFY"
$env:BYTE_API_TEST_SUBAREA = "Risk Assessment"

.\run_destructive_corpus_test.ps1
```

Il test richiede una risposta `201`, un `document_id` e almeno un job in stato `PENDING` o `RUNNING`.

## 5. Test distruttivo evidence

Richiede assessment e response realmente esistenti e coerenti con il tenant.

```powershell
cd E:\Dev\assessmentAI\byte_api

$env:BYTE_API_BASE_URL = "http://127.0.0.1:8020"
$env:BYTE_API_RUN_DESTRUCTIVE = "1"
$env:BYTE_API_TEST_EVIDENCE_FILE = "E:\Dev\assessmentAI\data\evidence_test.pdf"
$env:BYTE_API_TEST_ORGANIZATION_ID = "9999"
$env:BYTE_API_TEST_USER_ID = "123"
$env:BYTE_API_TEST_ASSESSMENT_ID = "10"
$env:BYTE_API_TEST_RESPONSE_ID = "20"
$env:BYTE_API_TEST_ENCRYPTION_REQUIRED = "true"

.\run_destructive_evidence_test.ps1
```

## Ordine raccomandato

```text
1. run_tests.ps1
2. run_preflight_real.ps1
3. avvio Byte API
4. run_api_integration_tests.ps1
5. run_destructive_corpus_test.ps1 su documento di prova
6. opzionale: run_destructive_evidence_test.ps1
7. avvio Ingestion API per consumare il job PENDING
```

I test `destructive` sono esclusi sia dal normale `pytest` sia dal preflight. Non possono partire senza `BYTE_API_RUN_DESTRUCTIVE=1`.
