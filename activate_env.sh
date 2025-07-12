# Activate conda environment on Linux/Mac
if [ -f "environment.yml" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate msrk-v3
    echo "✅ Conda environment 'msrk-v3' activated"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Python venv activated"
else
    echo "❌ No environment found. Please run setup_env.sh first"
fi
