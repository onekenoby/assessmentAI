@echo off
title Avvio RAG GUI Assessment AI

echo ======================================================
echo AVVIO RAG GUI
echo ======================================================
echo.
echo Questo comando avvia la GUI Reflex del RAG
echo usando i profili assessment + rag_gui.
echo.

docker compose --profile assessment --profile rag_gui up -d rag_gui_assessment

echo.
echo ======================================================
echo RAG GUI avviata.
echo Apri nel browser:
echo http://127.0.0.1:3000
echo.
echo Backend Reflex:
echo http://127.0.0.1:8000
echo ======================================================
pause
