# Validazione eseguita

Data: 23 luglio 2026

## Suite simulata

Controlli completati:

- compilazione Python di `main.py`, moduli `api`, `core` e test;
- generazione e validazione dello schema OpenAPI;
- contratto HTTP limitato al solo campo `max_jobs`;
- endpoint, autenticazione, request ID e header di sicurezza;
- lifecycle strict/non-strict e shutdown anche su startup fallito;
- worker seriale e rifiuto di una seconda esecuzione concorrente;
- mapping degli stati e degli errori del motore;
- sanitizzazione dei risultati e protezione dei dettagli sensibili;
- cronologia, snapshot e health check;
- contratto statico del motore, worker API `rag_ingestion` e advisory lock.

Esito:

```text
76 passed, 8 deselected
```

Copertura dello strato API/orchestrazione:

```text
TOTAL: 445 statements, 8 missed, 98% coverage
```

Esecuzione completa con i test reali disabilitati:

```text
76 passed, 8 skipped
```

## Test reali predisposti

Sono presenti otto test di integrazione, disabilitati per default:

- Database A e firme delle worker function;
- Database B e tabelle richieste;
- Ollama;
- Qdrant;
- Neo4j;
- liveness API;
- readiness profonda;
- claim ed ingestion di un job, marcato `destructive` e protetto dal flag
  aggiuntivo `RUN_INGESTION_CLAIM_TEST=1`.

I test reali non sono stati eseguiti nell'ambiente di generazione perché i
servizi, le credenziali e una coda di test controllata non erano disponibili.
