@echo off
setlocal
cd /d "%~dp0"

if exist ".venv_studio\Scripts\python.exe" (
    set "PY_EXE=.venv_studio\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    echo Graphite Implicit Studio virtual environment not found.
    echo Run once from repository root:
    echo     python scripts\setup_implicit_studio_env.py --create-venv .venv_studio
    pause
    exit /b 1
)

"%PY_EXE%" scripts\launch_implicit_gui.py %*
endlocal
