# Multi-Tenant Ingestion API

Questa API espone come servizio HTTP il motore contenuto in
`ingestion_engine.py`, mantenendo invariata la sorgente autorevole dei dati:
**i documenti, il BYTEA, TIER, scope, organization_id, ontology e tenant_key
sono letti esclusivamente dalle worker API PostgreSQL dello schema
`rag_ingestion` nel Database A**.

Non è previsto un endpoint di upload e il body HTTP non accetta metadati tenant.
In questo modo il client non può costruire o alterare direttamente il perimetro
multi-tenant.

## Architettura scelta

L'API è volutamente separata dalla RAG API:

- la RAG API gestisce richieste interattive e retrieval;
- la Ingestion API gestisce job lunghi, GPU, OCR, embedding e persistenza;
- entrambe possono usare gli stessi PostgreSQL, Qdrant, Neo4j e Ollama;
- il claim del job continua a essere atomico nel Database A;
- un advisory lock PostgreSQL globale impedisce due ingestion contemporanee
  anche con più processi o istanze API.

La chiamata HTTP restituisce `202 Accepted`. L'elaborazione prosegue nel worker
seriale del processo e lo stato viene letto con un endpoint dedicato. Lo stato
del job documentale resta comunque registrato nel Database A tramite le funzioni
`fn_complete_ingestion_job` e `fn_fail_ingestion_job` già usate dal motore.

## Endpoint

### Avvia una esecuzione

```http
POST /api/v1/ingestion/runs
X-Ingestion-Api-Key: change-me
Content-Type: application/json

{
  "max_jobs": 1
}
```

Risposta iniziale:

```json
{
  "run_id": "9f5a6c6a-f190-4aec-9381-8d05c48eec4e",
  "state": "queued",
  "requested_max_jobs": 1,
  "created_at": "2026-07-23T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "jobs_claimed": 0,
  "jobs_completed": 0,
  "jobs_failed": 0,
  "queue_empty": null,
  "processing_time_ms": 0,
  "jobs": [],
  "error_code": null,
  "error_message": null
}
```

### Leggi lo stato

```http
GET /api/v1/ingestion/runs/{run_id}
X-Ingestion-Api-Key: change-me
```

Stati possibili: `queued`, `running`, `succeeded`, `partial_failed`, `failed`.

### Elenco esecuzioni conservate in memoria

```http
GET /api/v1/ingestion/runs
X-Ingestion-Api-Key: change-me
```

La cronologia è locale al processo API. Lo stato autorevole dei singoli job è
quello del Database A.

### Health

```http
GET /health/live
GET /health/ready
GET /health/ready?deep=true
```

Il controllo `deep=true` verifica anche Ollama, Qdrant e Neo4j.

## Test prima dell'avvio

Installare le dipendenze di sviluppo ed eseguire la suite simulata:

```powershell
python -m pip install -r requirements-dev.txt
.\run_tests.ps1
```

La versione consegnata esegue **76 test simulati** con **98% di copertura**
sui moduli API, configurazione e orchestrazione. Gli 8 test reali sono
disabilitati di default.

Prima di avviare Uvicorn è possibile controllare i servizi reali, senza
reclamare job:

```powershell
.\run_preflight_real.ps1
```

Il piano completo, inclusi i test HTTP reali e l'E2E distruttivo esplicitamente
abilitato, è descritto in [`TESTING.md`](TESTING.md).

## Avvio

1. Creare e attivare il virtual environment.
2. Installare le dipendenze:

```bash
pip install -r requirements.txt
```

3. Copiare `.env.example` nei parametri ambiente reali. Il file non viene letto
automaticamente: usare il sistema di configurazione già adottato dal progetto,
Docker Compose, PowerShell oppure un gestore di secret.
4. Avviare **un solo worker Uvicorn**:

```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --workers 1
```

Su Windows è disponibile `start.ps1`; su Linux `start.sh`.

## Chiamata PowerShell

```powershell
$headers = @{ "X-Ingestion-Api-Key" = "change-me" }
$body = @{ max_jobs = 1 } | ConvertTo-Json

$run = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/api/v1/ingestion/runs" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8010/api/v1/ingestion/runs/$($run.run_id)" `
  -Headers $headers
```

## Decisioni di sicurezza

- Nessun upload HTTP: il BYTEA proviene solo dal job reclamato nel Database A.
- Nessun `organization_id`, TIER, scope o ruolo nel body API.
- API key opzionale ma fortemente consigliata; impostare `INGESTION_API_KEY`.
- `X-Request-ID` propagato su tutte le risposte.
- Header `no-store`, `nosniff`, `DENY` e `no-referrer`.
- Una sola esecuzione per processo e advisory lock globale cross-process.
- Errori interni nascosti per default; i dettagli restano nei log.
- Nessuna cancellazione forzata: interrompere OCR/embedding/LLM a metà potrebbe
  lasciare artefatti parziali. La compensazione già presente nel motore continua
  a marcare tali artefatti come `failed`/`PARTIAL_FAILED`.

## Differenze minime applicate al motore originale

`ingestion_engine.py` conserva la pipeline esistente e aggiunge soltanto:

- `initialize_ingestion_runtime()`;
- `runtime_healthcheck()`;
- `run_pending_jobs(max_jobs)`;
- `process_next_db_job()`;
- `shutdown_ingestion_runtime()`;
- advisory lock globale PostgreSQL;
- conversione sicura di `DB_MAX_JOBS_PER_RUN` a intero;
- override ambiente per `EMBEDDING_MODEL_NAME`, mantenendo il path originale
  come default.

## Note operative

- È consigliato eseguire Ollama come servizio separato e configurare:
  `USE_REMOTE_OLLAMA=1`, `OLLAMA_AUTOSTART=0`.
- `pytesseract` richiede anche l'eseguibile Tesseract installato nel sistema.
- L'installazione di PyTorch può richiedere il pacchetto specifico per la CUDA
  disponibile sulla macchina.
- Non usare `--reload` durante ingestion reali: il riavvio del processo web non
  è compatibile con job lunghi in esecuzione.
