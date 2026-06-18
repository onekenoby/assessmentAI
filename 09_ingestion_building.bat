@echo off
title Building Ingestion and RAG GUI Services
echo Building Ingestion and RAG GUI Services using Docker Compose...

docker compose --profile assessment --profile ingestion build ingestion_assessment

