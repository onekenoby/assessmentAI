@echo off

REM ==============================
REM Modelli usati da ingestion.py
REM ==============================
set LLM_MODEL_NAME=llama3.1:8b
set VISION_MODEL_NAME=ministral-3:8b

REM ==============================
REM Ollama containerizzato
REM Valide SOLO se hai aggiunto queste variabili in ingestion.py
REM ==============================
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set USE_REMOTE_OLLAMA=1
set OLLAMA_AUTOSTART=0

REM ==============================
REM Endpoint Ollama già supportati dal tuo ingestion.py
REM ==============================
set OLLAMA_API_CHAT=http://127.0.0.1:11434/api/chat
set OLLAMA_API_GENERATE=http://127.0.0.1:11434/api/generate

REM ==============================
REM Database assessment da docker-compose.yml
REM ==============================
set PG_HOST=127.0.0.1
set PG_PORT=5433
set PG_DB=assessment_ingestion
set PG_USER=admin
set PG_PASS=admin_password

set QDRANT_HOST=127.0.0.1
set QDRANT_PORT=6334
set QDRANT_COLLECTION=assessment_docs

set NEO4J_URI=bolt://127.0.0.1:7688
set NEO4J_USER=neo4j
set NEO4J_PASS=admin_password

REM ==============================
REM Esecuzione ingestion
REM ==============================
python ingestion.py

pause
