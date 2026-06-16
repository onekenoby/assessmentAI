@echo off
setlocal EnableExtensions
title Preprocess PDF in Markdown - Assessment AI

cd /d "%~dp0"

echo ======================================================
echo PREPROCESS PDF UFFICIALE IN MARKDOWN
echo ======================================================
echo.
echo Questo batch converte un PDF testuale in un .md pulito,
echo evitando Vision/OCR durante l'ingestion.
echo.
echo Richiede lo script:
echo utils\preprocess_pdf_to_md.py
echo.
echo Se lo script non e' in utils, copialo prima li'.
echo.

if not exist "utils\preprocess_pdf_to_md.py" (
  echo ERRORE: non trovo utils\preprocess_pdf_to_md.py
  echo Copia preprocess_pdf_to_md.py nella cartella utils del progetto.
  pause
  exit /b 1
)

set /p PDFPATH=Percorso PDF input, es. data\assessment\INBOX\TIER_A\file.pdf: 

if not defined PDFPATH (
  echo ERRORE: nessun PDF indicato.
  pause
  exit /b 1
)

set /p OUTPATH=Percorso MD output, invio per default accanto al PDF: 

if defined OUTPATH (
  python utils\preprocess_pdf_to_md.py "%PDFPATH%" --out "%OUTPATH%"
) else (
  python utils\preprocess_pdf_to_md.py "%PDFPATH%"
)

echo.
echo ======================================================
echo Preprocess terminato.
echo Metti il file .md nella INBOX corretta e lancia ingestion.
echo ======================================================
pause
