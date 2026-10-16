@echo off
cd /d "%~dp0"
set PY=..\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" demo_live.py --config C3 --mode rule
echo.
pause
