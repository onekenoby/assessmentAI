BATCH FILES - ASSESSMENT AI

Dove metterli
------------
Copia tutti i file .bat nella root del progetto, cioe' nella stessa cartella dove si trova docker-compose.yml.

Ordine consigliato
------------------
1) 00_verifica_compose.bat
   Verifica che il docker-compose.yml sia valido.

2) 01_avvia_stack_assessment.bat
   Avvia lo stack base: Postgres/Timescale, Qdrant, Neo4j, LibreOffice, Ollama.

3) Copia i documenti da ingerire dentro:
   .\data\assessment\INBOX

4) 02_lancia_ingestion.bat
   Esegue l'ingestion. I file processati finiscono in processed o failed.

5) 03_avvia_rag_gui.bat
   Avvia la GUI RAG Reflex.

6) Browser:
   http://127.0.0.1:3000

7) 04_log_rag_gui.bat
   Visualizza i log della GUI RAG.

8) 05_stop_stack_assessment.bat
   Ferma lo stack assessment/RAG. Non cancella i dati persistenti.

Note
----
- Non serve rebuild se modifichi solo docker-compose.yml, variabili d'ambiente o aggiungi documenti in INBOX.
- Serve rebuild se modifichi Dockerfile, requirements o codice copiato dentro l'immagine.
- Il comando ingestion usa:
  docker compose --profile assessment --profile ingestion run --rm ingestion_assessment
- Il comando RAG GUI usa:
  docker compose --profile assessment --profile rag_gui up -d rag_gui_assessment


AGGIUNTA - BATCH DI CANCELLAZIONE DOCUMENTI DAI DB
-------------------------------------------------

06_cancella_singolo_documento_dai_db.bat
  Esegue uno script Python di pulizia per un singolo documento/file.
  Il batch chiede:
  - nome del documento da cancellare;
  - modalita' di passaggio parametro:
    1) --filename "nomefile"
    2) argomento posizionale
    3) --source-name "nomefile"
    4) nessun argomento

07_cancella_tutti_documenti_dai_db.bat
  Esegue uno script Python di pulizia totale dei documenti dai DB.
  Richiede conferma esplicita:
  CANCELLA_TUTTO

Nota importante
---------------
I batch cercano automaticamente alcuni nomi comuni dentro .\utils.
Se il tuo script ha un nome diverso, il batch ti chiede di inserirlo manualmente.

Esempi:
  utils\cleanup_single_file.py
  utils\cleanup_all_dbs.py

Se hai modificato o aggiunto gli script in utils dopo la build dell'immagine Docker,
potrebbe essere necessario ricostruire il container ingestion:

  docker compose --profile assessment --profile ingestion build ingestion_assessment

Poi rilancia il batch.
