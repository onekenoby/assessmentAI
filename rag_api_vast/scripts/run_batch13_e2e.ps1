[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSeconds = 300,
    [int]$ApiStartTimeoutSeconds = 180,
    [int]$ApiLlmTimeoutSeconds = 180,
    [int]$ApiLlmNumPredict = 512,
    [int]$ApiLlmMaxAttempts = 1,
    [string]$ReportDir = "reports\batch13",
    [switch]$NoStartApi,
    [switch]$AllowEmptyData,
    [switch]$RequireSecondOrganization,
    [switch]$CapacityStress,
    [switch]$SkipUnit,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    $LocalPython = Join-Path (Split-Path -Parent $ProjectRoot) "venv\Scripts\python.exe"
    if (Test-Path $LocalPython) {
        $Python = $LocalPython
    }
    elseif ($env:VIRTUAL_ENV -and (Test-Path (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
        $Python = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    }
    else {
        $Python = "python"
    }

    $RunnerArgs = @(
        "verify_batch13.py",
        "--base-url", $BaseUrl,
        "--timeout", $TimeoutSeconds,
        "--api-start-timeout", $ApiStartTimeoutSeconds,
        "--api-llm-timeout", $ApiLlmTimeoutSeconds,
        "--api-llm-num-predict", $ApiLlmNumPredict,
        "--api-llm-max-attempts", $ApiLlmMaxAttempts,
        "--report-dir", $ReportDir
    )

    if (-not $NoStartApi) { $RunnerArgs += "--start-api" }
    if ($AllowEmptyData) { $RunnerArgs += "--allow-empty-data" }
    if ($RequireSecondOrganization) { $RunnerArgs += "--require-second-organization" }
    if ($CapacityStress) { $RunnerArgs += "--capacity-stress" }
    if ($SkipUnit) { $RunnerArgs += "--skip-unit" }
    if ($Quiet) { $RunnerArgs += "--quiet" }

    Write-Host "Batch 13 final end-to-end verification"
    Write-Host "Python: $Python"
    Write-Host "API: $BaseUrl"
    Write-Host "Reports: $ReportDir"
    Write-Host ""

    & $Python @RunnerArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
