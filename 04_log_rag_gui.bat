@echo off
title Log RAG GUI Assessment AI

echo ======================================================
echo LOG RAG GUI
echo ======================================================
echo.
echo Visualizzo i log del container rag_gui_assessment.
echo Premi CTRL+C per uscire.
echo.

docker compose --profile assessment --profile rag_gui logs -f rag_gui_assessment

pause
