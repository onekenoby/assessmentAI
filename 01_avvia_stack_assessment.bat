@echo off
title Avvio Stack Assessment AI

echo ======================================================
echo AVVIO STACK BASE ASSESSMENT
echo ======================================================
echo.
echo Avvio servizi base:
echo - TimescaleDB/Postgres
echo - Qdrant
echo - Neo4j
echo - LibreOffice
echo - Ollama
echo - Pull modelli Ollama
echo.

docker compose --profile assessment up -d

echo.
echo ======================================================
echo Stack assessment avviato.
echo Puoi controllare i container con:
echo docker ps
echo ======================================================
pause
