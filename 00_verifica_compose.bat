@echo off
title Verifica Docker Compose - Assessment AI

echo ======================================================
echo VERIFICA CONFIGURAZIONE DOCKER COMPOSE
echo ======================================================
echo.
echo Questo comando verifica il docker-compose.yml includendo
echo i profili assessment, ingestion e rag_gui.
echo.

docker compose --profile assessment --profile ingestion --profile rag_gui config

echo.
echo ======================================================
echo Verifica completata.
echo Se non sono comparsi errori, la configurazione YAML e' valida.
echo ======================================================
pause
