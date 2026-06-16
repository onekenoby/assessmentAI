rem se
rem volumes:
rem   - ./utils:/app/utils:ro   
rem allora
rem docker compose --profile assessment --profile ingestion run --rm -v "%cd%\utils:/app/utils:ro" ingestion_assessment python /app/utils/reset_dbs.py %*



@echo off
title Reset totale documenti dai DB - Assessment AI

cd /d "%~dp0"

docker compose --profile assessment --profile ingestion run --rm -v "%cd%\utils:/app/utils:ro" ingestion_assessment python /app/utils/reset_dbs.py %*

pause
