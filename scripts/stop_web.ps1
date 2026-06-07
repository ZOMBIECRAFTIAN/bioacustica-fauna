$ErrorActionPreference = "Stop"

$Processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*uvicorn*src.api.main*" }

if (-not $Processes) {
    Write-Host "No hay servidores uvicorn del proyecto corriendo."
    exit 0
}

foreach ($Process in $Processes) {
    Write-Host "Deteniendo servidor PID $($Process.ProcessId)..."
    Stop-Process -Id $Process.ProcessId
}

Write-Host "Servidor detenido."
