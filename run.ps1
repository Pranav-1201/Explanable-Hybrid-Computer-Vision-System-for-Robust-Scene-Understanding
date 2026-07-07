# run.ps1 — always execute with the project's CUDA-enabled venv, never system Python.
# Usage:
#   .\run.ps1 python training\train_phase2.py
#   .\run.ps1 python app.py
# Any command is run with venv\Scripts on PATH and the venv python as `python`.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Error "venv not found at $venvPy. Create it first: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

# Preflight: report interpreter + device so a wrong/CPU run is never silent.
& $venvPy -c "import sys,torch; print('[run.ps1] python=',sys.executable); print('[run.ps1] torch=',torch.__version__,'cuda=',torch.cuda.is_available())"

if ($args.Count -eq 0) {
    Write-Host "Nothing to run. Example: .\run.ps1 python app.py"
    exit 0
}

# Put venv Scripts first on PATH, then exec the requested command.
$env:PATH = (Join-Path $root "venv\Scripts") + ";" + $env:PATH
if ($args[0] -eq "python") {
    & $venvPy @($args[1..($args.Count-1)])
} else {
    & $args[0] @($args[1..($args.Count-1)])
}
exit $LASTEXITCODE
