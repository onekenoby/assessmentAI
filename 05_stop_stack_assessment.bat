@echo off
title Stop Stack Assessment AI

echo ======================================================
echo STOP STACK ASSESSMENT
echo ======================================================
echo.
echo Questo comando ferma i servizi assessment e rag_gui.
echo I dati persistenti restano nelle cartelle ./data.
echo.

rem docker compose --profile assessment --profile rag_gui down

docker compose --profile assessment --profile rag_gui up -d --force-recreate rag_gui_assessment

echo.
echo ======================================================
echo Stack fermato.
echo ======================================================
pause
