param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

while (Get-NetTCPConnection -LocalAddress $HostName -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Puerto $Port ocupado, probando $($Port + 1)..."
    $Port++
}

$env:MODEL_PATH = "models/trained/multitaxon/best_efficientnet.pt"
$env:MODEL_TYPE = "efficientnet"
$env:MODEL_DEVICE = "cpu"

$Url = "http://${HostName}:$Port/"
$ArgsList = @("-m", "uvicorn", "src.api.main:app", "--host", $HostName, "--port", "$Port")

if ($Background) {
    New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
    $Process = Start-Process `
        -FilePath "python" `
        -ArgumentList $ArgsList `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput ".\logs\web_api_$Port.out.log" `
        -RedirectStandardError ".\logs\web_api_$Port.err.log" `
        -PassThru

    Write-Host "Servidor iniciado en segundo plano."
    Write-Host "PID: $($Process.Id)"
    Write-Host "URL: $Url"
} else {
    Write-Host "URL: $Url"
    python @ArgsList
}
