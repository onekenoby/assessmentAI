@echo off
title Stop Stack Assessment AI

echo ======================================================
echo STOP STACK ASSESSMENT
echo ======================================================
echo.
echo Questo comando ferma i servizi assessment e rag_gui.
echo I dati persistenti restano nelle cartelle ./data.
echo.

docker compose --profile assessment --profile rag_gui down

echo.
echo ======================================================
echo Stack fermato.
echo ======================================================
pause
