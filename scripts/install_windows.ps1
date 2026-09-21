# lookat - Windows setup.
# Run from the project root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Find-Python {
    foreach ($candidate in @("py -3.12", "py -3", "python")) {
        $exe, $arg = $candidate.Split(" ", 2)
        try {
            $version = & $exe $arg --version 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match "Python 3\.(\d+)") {
                if ([int]$Matches[1] -ge 9) { return ,@($exe, $arg) }
            }
        } catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "No suitable Python found. Installing Python 3.12 via winget..." -ForegroundColor Yellow
    winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    Write-Host "Python installed. Close and reopen PowerShell, then run this script again." -ForegroundColor Green
    exit 0
}

Write-Host "Using Python: $($py -join ' ')" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    & $py[0] $py[1] -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -e .

# Face landmark model (~3.8 MB)
if (-not (Test-Path "models\face_landmarker.task")) {
    New-Item -ItemType Directory -Force -Path models | Out-Null
    $url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    Write-Host "Downloading face landmark model..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile "models\face_landmarker.task"
}

Write-Host ""
Write-Host "Done. Try it with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe run.py --list-cameras"
Write-Host "  .\.venv\Scripts\python.exe run.py --windowed --debug"
