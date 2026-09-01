# Piano di test della Ingestion API

La suite è divisa in livelli, perché i test dell'orchestrazione HTTP non devono
reclamare job reali né richiedere GPU o database durante il normale sviluppo.

## 1. Test simulati prima dell'avvio

Questi test non aprono connessioni a PostgreSQL, Qdrant, Neo4j o Ollama e non
importano il motore pesante. Verificano API, configurazione, contratti Pydantic,
worker seriale, lifecycle, sicurezza, mapping degli errori e contratto statico
del motore.

PowerShell:

```powershell
python -m pip install -r requirements-dev.txt
.\run_tests.ps1
```

Comando equivalente:

```powershell
python -m pytest -m "not integration" `
  --cov=api --cov=core --cov=main --cov-report=term-missing
```

Esito atteso della versione consegnata:

```text
76 passed, 8 deselected
TOTAL coverage: 98%
```

La copertura riguarda lo strato API e di orchestrazione. Il file monolitico
`ingestion_engine.py` è verificato staticamente in questa fase e viene provato
contro i servizi reali nel livello successivo.

## 2. Preflight reale prima dell'avvio dell'API

Questo livello controlla direttamente:

- Database A e firme delle cinque worker function `rag_ingestion`;
- Database B e tabelle di output;
- endpoint `/api/tags` di Ollama;
- connettività Qdrant;
- connettività Neo4j, quando abilitato.

Non avvia l'API e non reclama job.

```powershell
.\run_preflight_real.ps1
```

Le variabili di connessione devono essere già impostate nell'ambiente. Il flag
`RUN_REAL_SERVICE_TESTS=1` viene impostato dallo script.

## 3. Test HTTP reali dopo l'avvio

Dopo aver avviato l'API su `http://127.0.0.1:8010`:

```powershell
.\run_api_integration_tests.ps1
```

Questi test verificano liveness, readiness profonda, request ID e header di
sicurezza. Non reclamano job.

Per usare un URL diverso:

```powershell
$env:INGESTION_API_BASE_URL = "http://server:8010"
```

## 4. E2E con claim reale, solo su ambiente di test

Il test seguente chiama `POST /api/v1/ingestion/runs`, reclama al massimo un job
`PENDING` e attende la conclusione. Non deve essere eseguito su un Database A di
produzione o su una coda contenente documenti non predisposti per il test.

```powershell
$env:RUN_REAL_SERVICE_TESTS = "1"
$env:RUN_INGESTION_CLAIM_TEST = "1"
$env:INGESTION_E2E_TIMEOUT_S = "1800"
python -m pytest tests/integration/test_api_real.py `
  -m "integration and destructive" -v
```

Se è configurata l'autenticazione, impostare anche `INGESTION_API_KEY`.

## Aree coperte

La suite simulata include:

- validazione di `max_jobs` e rifiuto di campi tenant/file nel body;
- API key, request ID e header di sicurezza;
- risposte 202, 401, 404, 409, 422 e 503;
- health live/ready e guasti delle dipendenze;
- serializzazione delle esecuzioni e rifiuto del secondo run concorrente;
- stati `succeeded`, `partial_failed` e `failed`;
- occultamento degli errori sensibili;
- inizializzazione strict/non-strict e shutdown idempotente;
- limite e ordinamento della cronologia;
- immutabilità delle snapshot restituite;
- presenza dell'advisory lock globale e delle worker API protette;
- assenza di endpoint di upload e di metadati tenant nel contratto HTTP.
