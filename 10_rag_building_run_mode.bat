@echo off
setlocal EnableExtensions

title Build RAG GUI Assessment AI

cd /d "%~dp0"

echo ======================================================
echo BUILD RAG GUI CON CACHE
echo ======================================================
echo.

docker info >nul 2>&1

if errorlevel 1 (
    echo ERRORE: Docker Desktop non e' disponibile.
    pause
    exit /b 1
)

docker compose --profile assessment --profile rag_gui config --quiet

if errorlevel 1 (
    echo ERRORE: docker-compose.yml non valido.
    pause
    exit /b 1
)

docker compose --progress=plain --profile assessment --profile rag_gui build rag_gui_assessment

if errorlevel 1 (
    echo.
    echo BUILD RAG GUI FALLITO
    pause
    exit /b 1
)

echo.
echo BUILD RAG GUI COMPLETATO
echo.
pause
exit /b 0