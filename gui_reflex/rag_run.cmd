@echo off

REM =====================================================
REM GUI REFLEX - CONFIGURAZIONE MODELLI
REM =====================================================
set LLM_MODEL_NAME=gemma4:12b
set VISION_MODEL_NAME=gemma4:12b

REM =====================================================
REM OLLAMA CONTAINERIZZATO
REM =====================================================
set OLLAMA_URL=http://127.0.0.1:11434/v1
set OLLAMA_NATIVE_CHAT_URL=http://127.0.0.1:11434/api/chat
set OLLAMA_API_KEY=ollama

REM =====================================================
REM PARAMETRI LLM
REM =====================================================
set LLM_NUM_CTX=16384
set LLM_NUM_PREDICT=4096
set LLM_TIMEOUT_S=300

REM =====================================================
REM DATABASE / VECTOR DB / GRAPH DB
REM Allineati al docker-compose assessment
REM =====================================================
set PG_ENRICH_ENABLED=1
set PG_HOST=127.0.0.1
set PG_PORT=5433
set PG_DB=assessment_ingestion
set PG_USER=admin
set PG_PASS=admin_password

set QDRANT_HOST=127.0.0.1
set QDRANT_PORT=6334
set QDRANT_COLLECTION=assessment_docs

set NEO4J_ENABLED=1
set NEO4J_URI=bolt://127.0.0.1:7688
set NEO4J_USER=neo4j
set NEO4J_PASS=admin_password

REM =====================================================
REM DEVICE MODELLI LOCALI
REM Conviene CPU per non competere con Ollama sulla GPU
REM =====================================================
set EMBED_DEVICE=cpu
set RERANK_DEVICE=cpu

REM =====================================================
REM LOG
REM =====================================================
set AUDIT_ENABLED=1
set RAG_LOG_DIR=%USERPROFILE%\ai_rag_logs

REM =====================================================
REM AVVIO GUI REFLEX
REM =====================================================
reflex run

pause