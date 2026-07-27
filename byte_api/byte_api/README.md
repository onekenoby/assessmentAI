# Byte API

`byte_api` è la trasformazione API di `INGESTION_carica_documento_bytea_rag.py`.

Il servizio svolge esclusivamente il primo tratto della pipeline:

```text
File PDF/Markdown
      ↓
Byte API
      ↓
PostgreSQL Database A: assessment_gestio_tier
      ↓
schema rag_ingestion
  - rag_file_blob
  - rag_document
  - rag_document_context
  - rag_ingestion_job PENDING
      ↓
Ingestion API
      ↓
PostgreSQL B + Qdrant + Neo4j
```

La Byte API **non esegue chunking**, non calcola embedding e non scrive in Qdrant o Neo4j.

## Endpoint

### Upload corpus

```http
POST /api/v1/byte/corpus
Content-Type: multipart/form-data
```

Campi:

- `file`: PDF, `.md` o `.markdown`;
- `tier`: `A`, `B` o `C`;
- `organization_id`: obbligatorio per TIER B/C, vietato per TIER A;
- `user_id`: obbligatorio per TIER B/C, vietato per TIER A;
- `ontology_code`, oppure la coppia `area` + `subarea`;
- `ontology_label`: opzionale;
- `classification`: `public`, `internal`, `confidential`, `restricted`; default `internal`;
- `pipeline_version`: default `v1`;
- `corpus_version`: default `v1`;
- `embedding_model`: opzionale;
- `mime_type`: override opzionale.

### Upload evidence

```http
POST /api/v1/byte/evidence
Content-Type: multipart/form-data
```

Campi:

- `file`;
- `organization_id`;
- `user_id`;
- `assessment_id`;
- `response_id`;
- `encryption_required`: default `true`;
- `mime_type`: override opzionale.

La modalità evidence continua a usare la funzione ufficiale:

```sql
rag_ingestion.fn_upload_response_evidence(...)
```

## Installazione

Da PowerShell:

```powershell
cd E:\Dev\assessmentAI\byte_api
python -m pip install -r requirements-dev.txt
```

Copiare e configurare le variabili di `.env.example` nell'ambiente operativo.

Le variabili PostgreSQL sono le stesse del programma originario:

```text
PG_HOST
PG_PORT
SOURCE_PG_DB
PG_USER
PG_PASS
```

## Avvio

```powershell
cd E:\Dev\assessmentAI\byte_api
python -m uvicorn main:app --host 0.0.0.0 --port 8020 --workers 1
```

Oppure:

```powershell
.\start.ps1
```

Documentazione Swagger:

```text
http://127.0.0.1:8020/docs
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8020/health/live |
    ConvertTo-Json -Depth 10

Invoke-RestMethod "http://127.0.0.1:8020/health/ready?deep=true" |
    ConvertTo-Json -Depth 10
```

## Esempio corpus equivalente al comando CLI

Il precedente comando:

```powershell
python .\INGESTION_carica_documento_bytea_rag.py corpus `
  --file "E:\Dev\assessmentAI\data\documento.pdf" `
  --tier C `
  --organization-id 9999 `
  --user-id 123 `
  --area "IDENTIFY" `
  --subarea "Risk Assessment"
```

diventa:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8020/api/v1/byte/corpus" `
  -F "file=@E:\Dev\assessmentAI\data\documento.pdf" `
  -F "tier=C" `
  -F "organization_id=9999" `
  -F "user_id=123" `
  -F "area=IDENTIFY" `
  -F "subarea=Risk Assessment"
```

Con API key configurata:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8020/api/v1/byte/corpus" `
  -H "X-Byte-Api-Key: change-me" `
  -F "file=@E:\Dev\assessmentAI\data\documento.pdf" `
  -F "tier=C" `
  -F "organization_id=9999" `
  -F "user_id=123" `
  -F "area=IDENTIFY" `
  -F "subarea=Risk Assessment"
```

## Esempio evidence

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8020/api/v1/byte/evidence" `
  -F "file=@E:\Dev\assessmentAI\data\evidence.pdf" `
  -F "organization_id=9999" `
  -F "user_id=123" `
  -F "assessment_id=10" `
  -F "response_id=20" `
  -F "encryption_required=true"
```

## Sicurezza e comportamento

- Il nome file viene ridotto al solo basename; eventuali path del client non vengono usati dal server.
- Sono accettati soltanto PDF e Markdown.
- Il limite predefinito è 250 MiB ed è configurabile con `BYTE_API_MAX_FILE_BYTES`.
- Il TIER A è sempre `GLOBAL` con `organization_id` nullo.
- I TIER B/C sono sempre `ACCOUNT` e richiedono `organization_id` e `user_id` positivi.
- L'API key è opzionale ma raccomandata fuori dallo sviluppo locale.
- Gli errori PostgreSQL dettagliati non vengono esposti per default.
- La scrittura PostgreSQL mantiene commit, rollback, deduplica e ricreazione del job già presenti nella CLI.

## Test

Consultare [TESTING.md](TESTING.md).
