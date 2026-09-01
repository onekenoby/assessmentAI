param(
    [string]$Query = "Quali sono i principali requisiti di sicurezza descritti nei documenti disponibili?",
    [string]$BaseUrl = "http://127.0.0.1:8013"
)

$ErrorActionPreference = "Stop"
$base = $BaseUrl.TrimEnd("/")

Write-Host "Readiness RAG..." -ForegroundColor Cyan
$ready = Invoke-RestMethod -Method Get -Uri "$base/health/ready?deep=true" -TimeoutSec 120
$ready | ConvertTo-Json -Depth 10

$body = @{
    query = $Query
    conversation_id = "vast-rag-test-001"
    history = @()
    options = @{
        include_sources = $true
        include_debug = $true
        include_evaluation = $false
        max_sources = 8
    }
} | ConvertTo-Json -Depth 10

Write-Host "Invio query RAG..." -ForegroundColor Cyan
$response = Invoke-RestMethod `
    -Method Post `
    -Uri "$base/api/v1/rag/query" `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 900

$response | ConvertTo-Json -Depth 30
