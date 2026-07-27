# Rapporto di validazione

## Verifiche eseguite

- Compilazione di tutti i moduli Python con `py_compile`.
- Generazione e caricamento dell'app FastAPI.
- Test degli endpoint multipart corpus ed evidence.
- Test della logica transazionale PostgreSQL tramite connessioni e cursori simulati.
- Test degli invarianti multi-tenant.
- Test del lifecycle FastAPI e degli error handler.
- Test dei marker di integrazione e delle protezioni distruttive.

## Esito simulato

```text
136 passed, 4 deselected
```

Copertura:

```text
TOTAL 621 statements, 11 missed, 98%
```

## Test reali non eseguiti nell'ambiente di costruzione

Non è stato eseguito un upload reale perché l'ambiente di costruzione non dispone delle istanze PostgreSQL e dei dati applicativi dell'utente.

Sono inclusi:

- preflight non distruttivo del Database A;
- test HTTP reale di liveness/readiness;
- test distruttivo corpus protetto da flag;
- test distruttivo evidence protetto da flag.

## Confine funzionale

La Byte API termina il proprio compito quando il documento e il relativo job sono presenti nello schema `rag_ingestion`.

Il trasferimento successivo verso PostgreSQL B, Qdrant e Neo4j resta responsabilità di `ingestion_api_tested.zip`.
