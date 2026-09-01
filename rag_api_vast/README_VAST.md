# RAG API locale con Ollama su Vast.ai

Il backend RAG viene eseguito su Windows locale. PostgreSQL, Qdrant, Neo4j,
embedding BGE-M3 e reranker restano locali; soltanto la generazione LLM usa
`gemma4:12b` esposto dalla VM Vast.ai tramite Ollama.

## 1. Prerequisiti

- Il tunnel usato dall'ingestion deve essere attivo:

```powershell
ssh -N -L 11435:127.0.0.1:11434 -p <PORTA_SSH_VAST> root@<IP_VAST>
```

Usare esattamente host, porta e utente indicati dal comando SSH della propria
istanza Vast.ai. Se il tunnel dell'ingestion è già aperto su `11435`, non aprirne
un secondo.

- Devono essere disponibili localmente:
  - `E:/Modelli/bge-m3`
  - `E:/Modelli/ms-marco-reranker`
- PostgreSQL, Qdrant e Neo4j devono essere raggiungibili sugli stessi endpoint
  usati dall'ingestion.
- Sulla VM devono risultare attivi Ollama e il modello `gemma4:12b`.

## 2. Installazione dipendenze

Dalla cartella `rag_api_vast`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se il virtual environment condiviso si trova nella cartella superiore,
`start.ps1` lo rileva automaticamente.

## 3. Verifica e inizializzazione manuale di Ollama

Questo passaggio è facoltativo perché `start.ps1` esegue il warm-up durante lo
startup. È utile per diagnosticare il tunnel prima di avviare l'API:

```powershell
.\check_vast_ollama.ps1
```

Lo script verifica `/api/tags`, carica `gemma4:12b` in VRAM tramite `/api/chat`
e imposta `keep_alive=30m`.

## 4. Avvio del RAG

### Terminale Windows locale 1

```powershell
cd <PERCORSO>\rag_api_vast
.\start.ps1
```

L'API ascolta su `http://127.0.0.1:8013`. Durante lo startup:

1. verifica il tunnel Ollama;
2. verifica la presenza di `gemma4:12b`;
3. inizializza il modello sulla GPU Vast.ai;
4. carica localmente embedder e reranker;
5. verifica PostgreSQL, Qdrant e Neo4j.

Attendere che il log riporti il completamento del warm-up e l'inizializzazione
delle risorse RAG.

### Terminale Windows locale 2

Verifica readiness completa:

```powershell
Invoke-RestMethod "http://127.0.0.1:8013/health/ready?deep=true" |
    ConvertTo-Json -Depth 10
```

Invio di una query di prova:

```powershell
.\invoke_rag.ps1 -Query "Quali controlli sono richiesti per la gestione degli accessi?"
```

In alternativa, chiamata PowerShell diretta:

```powershell
$body = @{
    query = "Quali controlli sono richiesti per la gestione degli accessi?"
    conversation_id = "test-vast-001"
    history = @()
    options = @{
        include_sources = $true
        include_debug = $true
        include_evaluation = $false
        max_sources = 8
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8013/api/v1/rag/query" `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 900 | ConvertTo-Json -Depth 30
```

## 5. Parametri modificabili

`start.ps1` accetta parametri senza richiedere modifiche al codice:

```powershell
.\start.ps1 `
    -OllamaBaseUrl "http://127.0.0.1:11435" `
    -ApiPort 8013 `
    -EmbeddingModel "E:/Modelli/bge-m3" `
    -RerankerModel "E:/Modelli/ms-marco-reranker"
```

Gli URL `OLLAMA_URL` e `OLLAMA_NATIVE_CHAT_URL` restano supportati come override,
ma normalmente vengono derivati automaticamente da `OLLAMA_BASE_URL`.
