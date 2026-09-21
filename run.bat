@echo off
REM Convenience launcher: uses the venv if scripts\install_windows.ps1 made one.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py %*
) else (
    python run.py %*
)
