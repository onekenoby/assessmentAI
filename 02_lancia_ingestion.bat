@echo off
title Lancio Ingestion Assessment AI

echo ======================================================
echo LANCIO INGESTION
echo ======================================================
echo.
echo Questo comando esegue il container ingestion_assessment
echo usando i profili assessment + ingestion.
echo.
echo I documenti devono essere in:
echo .\data\assessment\INBOX
echo.

docker compose --profile assessment --profile ingestion run --rm ingestion_assessment

echo.
echo ======================================================
echo Ingestion terminata.
echo Controlla:
echo .\data\assessment\processed
echo .\data\assessment\failed
echo .\data\assessment\logs
echo ======================================================
pause
