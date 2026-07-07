@echo off
REM run.bat - always execute with the project's CUDA-enabled venv, never system Python.
REM Usage:
REM   run.bat python training\train_phase2.py
REM   run.bat python app.py
setlocal
set "ROOT=%~dp0"
set "VENVPY=%ROOT%venv\Scripts\python.exe"

if not exist "%VENVPY%" (
    echo venv not found at %VENVPY%.
    echo Create it: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

REM Preflight: report interpreter + device so a wrong/CPU run is never silent.
"%VENVPY%" -c "import sys,torch; print('[run.bat] python=',sys.executable); print('[run.bat] torch=',torch.__version__,'cuda=',torch.cuda.is_available())"

if "%~1"=="" (
    echo Nothing to run. Example: run.bat python app.py
    exit /b 0
)

set "PATH=%ROOT%venv\Scripts;%PATH%"
if /I "%~1"=="python" (
    shift
    "%VENVPY%" %*
) else (
    %*
)
exit /b %ERRORLEVEL%
