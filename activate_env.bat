@echo off
REM Activate environment on Windows

if exist "environment.yml" (
    where conda >nul 2>&1
    if %errorlevel% == 0 (
        call conda activate msrk-v3
        echo ✅ Conda environment 'msrk-v3' activated
        goto end
    )
)

if exist ".venv" (
    call .venv\Scripts\activate.bat
    echo ✅ Python venv activated
    goto end
)

echo ❌ No environment found. Please run setup_env.bat first

:end
