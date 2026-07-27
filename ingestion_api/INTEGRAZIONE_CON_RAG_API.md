# Integrazione con la RAG API esistente

La soluzione consigliata è mantenere due processi ASGI distinti:

- RAG API sulla porta già in uso, ad esempio `8000`;
- Ingestion API sulla porta `8010`.

La separazione evita che OCR, SentenceTransformer, Vision e Knowledge Graph
riducano la capacità delle richieste RAG interattive. I due servizi condividono
solo le infrastrutture dati e i modelli esterni configurati via ambiente.

Non è necessario modificare il router RAG. L'applicazione gestionale che crea i
record e i job nello schema `rag_ingestion` può successivamente invocare:

```http
POST http://ingestion-api:8010/api/v1/ingestion/runs
```

L'invocazione non trasferisce il file: segnala soltanto al worker di reclamare
il prossimo job `PENDING`. Il Database A mantiene quindi segregazione,
tracciabilità e retry.

Per un deployment con reverse proxy si possono pubblicare, ad esempio:

- `/rag/...` verso la RAG API;
- `/ingestion/...` verso la Ingestion API;
- `/health/rag/...` e `/health/ingestion/...` come health separati.
