param(
    [string]$OllamaBaseUrl = "http://127.0.0.1:11435",
    [string]$Model = "gemma4:12b"
)

$ErrorActionPreference = "Stop"
$base = $OllamaBaseUrl.TrimEnd("/")

Write-Host "Verifica Ollama su $base ..." -ForegroundColor Cyan
$tags = Invoke-RestMethod -Method Get -Uri "$base/api/tags" -TimeoutSec 20
$modelNames = @(
    $tags.models | ForEach-Object {
        if ($_.name) { [string]$_.name }
        elseif ($_.model) { [string]$_.model }
    }
)

if ($modelNames -notcontains $Model) {
    throw "Modello $Model non disponibile. Modelli presenti: $($modelNames -join ', ')"
}

$payload = @{
    model = $Model
    messages = @(
        @{ role = "user"; content = "Rispondi soltanto con OK." }
    )
    stream = $false
    think = $false
    keep_alive = "30m"
    options = @{
        temperature = 0.0
        num_ctx = 2048
        num_predict = 8
    }
} | ConvertTo-Json -Depth 8

$result = Invoke-RestMethod `
    -Method Post `
    -Uri "$base/api/chat" `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 600

Write-Host "Ollama raggiungibile; modello inizializzato e mantenuto caricato per 30 minuti." -ForegroundColor Green
$result | ConvertTo-Json -Depth 10
