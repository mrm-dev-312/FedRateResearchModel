@echo off
REM Environment setup script for Macro Signal Research Kit v3 (Windows)
REM Supports both conda and venv isolation

echo 🚀 Setting up Macro Signal Research Kit v3 Environment...

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Error: Please run this script from the MSRK v3 root directory
    exit /b 1
)
if not exist "prisma\schema.prisma" (
    echo ❌ Error: Please run this script from the MSRK v3 root directory
    exit /b 1
)

REM Check for conda
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo 📦 Conda detected
    set ENV_TYPE=conda
    goto setup
)

REM Check for python
where python >nul 2>&1
if %errorlevel% == 0 (
    echo 🐍 Python detected
    set ENV_TYPE=venv
    goto setup
)

echo ❌ Error: Neither conda nor python found in PATH
echo Please install either Anaconda/Miniconda or Python 3.8+
exit /b 1

:setup
echo 📋 Using environment type: %ENV_TYPE%

if "%ENV_TYPE%" == "conda" goto setup_conda
if "%ENV_TYPE%" == "venv" goto setup_venv

:setup_conda
echo 🔧 Setting up conda environment...

REM Check if environment exists
conda info --envs | findstr "msrk-v3" >nul
if %errorlevel% == 0 (
    echo ⚠️  Environment 'msrk-v3' already exists. Updating...
    conda env update -f environment.yml
) else (
    echo 📦 Creating new conda environment 'msrk-v3'...
    conda env create -f environment.yml
)

echo ✅ Conda environment created/updated
echo 🔄 To activate: conda activate msrk-v3
set ACTIVATION_CMD=conda activate msrk-v3
goto install_node

:setup_venv
echo 🔧 Setting up Python venv environment...

set VENV_DIR=.venv

if exist "%VENV_DIR%" (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q "%VENV_DIR%"
)

echo 📦 Creating new virtual environment...
python -m venv "%VENV_DIR%"

echo 🔄 Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

echo 📦 Installing Python packages...
pip install -r requirements.txt

set ACTIVATION_CMD=%VENV_DIR%\Scripts\activate.bat

:install_node
echo 📦 Installing Node.js dependencies...

REM Check for Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Node.js not found
    if "%ENV_TYPE%" == "conda" (
        echo Installing Node.js via conda...
        conda install -c conda-forge nodejs npm -y
    ) else (
        echo ❌ Please install Node.js manually from https://nodejs.org/
        echo    Node.js is required for Prisma database operations
        exit /b 1
    )
)

echo 🗄️  Setting up Prisma...
call npm install prisma @prisma/client
call npx prisma generate

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy ".env.template" ".env"
    echo ⚠️  Please edit .env file with your actual API keys and database URL
)

echo 🔍 Verifying installation...
python -c "import sys; print(f'✅ Python: {sys.version}'); import pandas, numpy; print('✅ Core packages imported')"

echo.
echo 🎉 Environment setup complete!
echo.
echo 📋 Next steps:
echo 1. Activate environment: %ACTIVATION_CMD%
echo 2. Edit .env file with your API keys and database URL
echo 3. Initialize database: npx prisma db push
echo 4. Run workflow: jupyter notebook notebooks/daily_workflow.ipynb
echo.
echo 📖 For more information, see README.md

pause
