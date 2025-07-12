#!/bin/bash
# Environment setup script for Macro Signal Research Kit v3
# Supports both conda and venv isolation

set -e  # Exit on any error

echo "🚀 Setting up Macro Signal Research Kit v3 Environment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -f "prisma/schema.prisma" ]; then
    echo "❌ Error: Please run this script from the MSRK v3 root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect environment preference
ENV_TYPE=""
if command_exists conda; then
    echo "📦 Conda detected"
    ENV_TYPE="conda"
elif command_exists python3 && python3 -m venv --help >/dev/null 2>&1; then
    echo "🐍 Python venv available"
    ENV_TYPE="venv"
else
    echo "❌ Error: Neither conda nor python venv available"
    echo "Please install either Anaconda/Miniconda or Python 3.8+"
    exit 1
fi

# Ask user for preference if both available
if [ "$ENV_TYPE" = "conda" ] && command_exists python3; then
    echo "Both conda and venv are available. Which would you prefer?"
    echo "1) conda (recommended for ML/data science)"
    echo "2) venv (lightweight, Python standard)"
    read -p "Enter choice (1 or 2): " choice
    
    case $choice in
        1) ENV_TYPE="conda" ;;
        2) ENV_TYPE="venv" ;;
        *) echo "Invalid choice, defaulting to conda"; ENV_TYPE="conda" ;;
    esac
fi

echo "📋 Using environment type: $ENV_TYPE"

# Setup conda environment
if [ "$ENV_TYPE" = "conda" ]; then
    echo "🔧 Setting up conda environment..."
    
    # Create environment from yml file
    if conda env list | grep -q "msrk-v3"; then
        echo "⚠️  Environment 'msrk-v3' already exists. Updating..."
        conda env update -f environment.yml
    else
        echo "📦 Creating new conda environment 'msrk-v3'..."
        conda env create -f environment.yml
    fi
    
    echo "✅ Conda environment created/updated"
    echo "🔄 Activating environment..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate msrk-v3
    
    ACTIVATION_CMD="conda activate msrk-v3"

# Setup venv environment
elif [ "$ENV_TYPE" = "venv" ]; then
    echo "🔧 Setting up Python venv environment..."
    
    VENV_DIR=".venv"
    
    if [ -d "$VENV_DIR" ]; then
        echo "⚠️  Virtual environment already exists. Removing old one..."
        rm -rf "$VENV_DIR"
    fi
    
    echo "📦 Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    echo "🔄 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    echo "📦 Installing Python packages..."
    pip install -r requirements.txt
    
    ACTIVATION_CMD="source .venv/bin/activate"
fi

# Install Node.js dependencies (Prisma)
echo "📦 Installing Node.js dependencies..."
if ! command_exists node; then
    echo "⚠️  Node.js not found. Installing via conda/system package manager..."
    if [ "$ENV_TYPE" = "conda" ]; then
        conda install -c conda-forge nodejs npm -y
    else
        echo "❌ Please install Node.js manually from https://nodejs.org/"
        echo "   Node.js is required for Prisma database operations"
        exit 1
    fi
fi

# Install Prisma
echo "🗄️  Setting up Prisma..."
npm install prisma @prisma/client
npx prisma generate

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your actual API keys and database URL"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'✅ Python: {sys.version}')

try:
    import pandas, numpy, torch, transformers
    print('✅ Core ML packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

try:
    import fredapi, yfinance
    print('✅ Data APIs imported successfully')
except ImportError as e:
    print(f'⚠️  Data API warning: {e}')

try:
    from prisma import Prisma
    print('✅ Prisma client imported successfully')
except ImportError as e:
    print(f'⚠️  Prisma warning: {e}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: $ACTIVATION_CMD"
echo "2. Edit .env file with your API keys and database URL"
echo "3. Initialize database: npx prisma db push"
echo "4. Run workflow: jupyter notebook notebooks/daily_workflow.ipynb"
echo ""
echo "📖 For more information, see README.md"
