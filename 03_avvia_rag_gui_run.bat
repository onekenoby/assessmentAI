@echo off
setlocal EnableExtensions

title Avvio RAG GUI Assessment AI

cd /d "%~dp0"

echo ======================================================
echo AVVIO RAG GUI
echo ======================================================
echo.
echo Directory progetto:
echo %CD%
echo.
echo La GUI viene eseguita in primo piano.
echo La finestra deve rimanere aperta.
echo Per arrestare la GUI premi CTRL+C.
echo.

docker info >nul 2>&1

if errorlevel 1 (
    echo ERRORE: Docker Desktop non e' disponibile.
    echo Avvia Docker Desktop e riprova.
    echo.
    pause
    exit /b 1
)

echo Verifica docker-compose.yml...
docker compose --profile assessment --profile rag_gui config --quiet

if errorlevel 1 (
    echo.
    echo ERRORE: docker-compose.yml non valido.
    echo.
    pause
    exit /b 1
)

echo.
echo Arresto dell'eventuale precedente container GUI...
docker compose --profile assessment --profile rag_gui run --rm --service-ports rag_gui_assessment reflex run --env dev --backend-host 0.0.0.0 --backend-port 8000 --frontend-port 3000

echo.
echo ======================================================
echo ESECUZIONE RAG GUI
echo ======================================================
echo.
echo Comando:
echo docker compose --profile assessment --profile rag_gui run --rm --service-ports rag_gui_assessment
echo.
echo Frontend:
echo http://127.0.0.1:3000
echo.
echo Backend:
echo http://127.0.0.1:8000
echo.
echo Attendere il completamento della compilazione Reflex.
echo.

docker compose --profile assessment --profile rag_gui run --rm --service-ports rag_gui_assessment

set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ======================================================

if not "%EXIT_CODE%"=="0" (
    echo RAG GUI TERMINATA CON ERRORE
    echo Codice di uscita: %EXIT_CODE%
    echo.
    echo Stato stack:
    docker compose --profile assessment --profile rag_gui ps -a
    echo.
    echo Log disponibili:
    docker compose --profile assessment --profile rag_gui logs --tail=250
) else (
    echo RAG GUI ARRESTATA
)

echo ======================================================
echo.
pause
exit /b %EXIT_CODE%
