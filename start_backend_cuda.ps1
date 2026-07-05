param(
    [switch]$Install,
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8000,
    [string]$Decoder = "greedy"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

if ($Install) {
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements-backend-gpu.txt
}

$env:QURAN_ASR_PROVIDER = "cuda"
$env:QURAN_ASR_DECODER = $Decoder

.\.venv\Scripts\python.exe tool\asr_runtime_probe.py
.\.venv\Scripts\python.exe -m uvicorn backend:app --host $HostName --port $Port
