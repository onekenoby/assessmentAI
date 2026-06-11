# Requirements Docker ingestion - versione suddivisa

Questi file separano i pacchetti Python in gruppi logici per evitare build Docker troppo pesanti e ridurre errori di I/O su Docker Desktop/WSL.

## File inclusi

- `requirements.ingestion.txt`: dipendenze minime per `ingestion.py`.
- `requirements.extra.main.txt`: extra principali per runtime, documenti, dati, DB e utility.
- `requirements.extra.ai_docs.txt`: extra pesanti per AI/document parsing/OCR avanzato.
- `requirements.extra.gui.txt`: extra per UI, Reflex, Streamlit, socket/web.
- `requirements.extra.dev.txt`: extra per notebook, debug e sviluppo.
- `Dockerfile.ingestion.split`: Dockerfile con installazione a blocchi opzionali tramite build args.

## Uso consigliato

Per una build minima e stabile:

```powershell
docker compose --verbose --progress=plain --profile assessment --profile ingestion build ingestion_assessment 2>&1 | Tee-Object -FilePath docker_build_ingestion.log
```

Per installare anche gli extra principali, modifica temporaneamente il Dockerfile usato da compose oppure rinomina `Dockerfile.ingestion.split` in `Dockerfile.ingestion` e usa:

```powershell
docker compose --verbose --progress=plain --profile assessment --profile ingestion build --build-arg INSTALL_EXTRA_MAIN=1 ingestion_assessment 2>&1 | Tee-Object -FilePath docker_build_ingestion.log
```

Per installare anche il gruppo AI/documentale pesante:

```powershell
docker compose --verbose --progress=plain --profile assessment --profile ingestion build --build-arg INSTALL_EXTRA_MAIN=1 --build-arg INSTALL_EXTRA_AI_DOCS=1 ingestion_assessment 2>&1 | Tee-Object -FilePath docker_build_ingestion.log
```

## Raccomandazione

Per il container `ingestion_assessment`, partire dalla build minima. Aggiungere gruppi extra solo quando un import reale fallisce con `ModuleNotFoundError` oppure quando una funzionalità specifica li richiede.

Per `gui_reflex.py`, Reflex o Streamlit è preferibile creare un Dockerfile separato, perché i pacchetti GUI non sono necessari al runtime ingestion.
