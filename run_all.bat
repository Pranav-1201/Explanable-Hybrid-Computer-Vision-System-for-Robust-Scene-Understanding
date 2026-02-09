@echo off
cd /d %~dp0

echo ============================================================
echo  EXPLAINABLE HYBRID CV SYSTEM - FULL RUN
echo ============================================================
echo.

call venv\Scripts\activate

echo Detecting compute device...
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')"
echo.

python pipeline_timer.py

pause