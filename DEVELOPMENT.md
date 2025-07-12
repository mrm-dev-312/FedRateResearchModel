# Development Guide - MSRK v3

## 🔧 Environment Setup Options

### Quick Start (Recommended)
```bash
# Windows
setup_env.bat

# Linux/Mac
chmod +x setup_env.sh
./setup_env.sh
```

### Manual Setup Options

#### Option 1: Conda Environment (Recommended for ML)
```bash
# Create environment
conda env create -f environment.yml
conda activate msrk-v3

# Install Node.js dependencies
npm install prisma @prisma/client
npx prisma generate
```

#### Option 2: Python venv (Lightweight)
```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install prisma @prisma/client
npx prisma generate
```

## 🚀 VS Code Tasks

Access via `Ctrl+Shift+P` → "Tasks: Run Task":

1. **Setup MSRK v3 Environment** - Complete environment setup
2. **Activate Environment** - Activate conda/venv environment
3. **Initialize Database** - Run Prisma database setup
4. **Start Jupyter Notebook** - Launch daily workflow notebook

## 📝 Configuration

### Database Setup
1. Get PostgreSQL database (local or cloud):
   - **Local**: Install PostgreSQL locally
   - **Cloud**: Use Supabase, Neon, or AWS RDS
   - **Test**: Use SQLite for development

2. Set connection string in `.env`:
   ```bash
   DATABASE_URL="postgresql://username:password@localhost:5432/msrk_v3"
   ```

3. Initialize database:
   ```bash
   npx prisma db push
   ```

### API Keys Setup
Edit `.env` file:
```bash
# Required for data ingestion
FRED_API_KEY=your_fred_api_key_here

# Optional for advanced forecasting
TIMEGPT_API_KEY=your_nixtla_api_key_here

# Optional for text sentiment
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## 🔄 Development Workflow

### 1. Daily Development
```bash
# Activate environment
conda activate msrk-v3
# or source .venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/daily_workflow.ipynb
```

### 2. Database Operations
```bash
# Update schema
npx prisma db push

# Generate client
npx prisma generate

# View data
npx prisma studio
```

### 3. Testing
```bash
# Run unit tests
pytest src/ -v

# Test specific module
python -m src.data_ingest.fred
python -m src.models.patchtst
```

## 🏗️ Architecture Overview

### Data Flow
```
FRED API → PostgreSQL → Time Join → Features → PatchTST → Backtest → Results
Yahoo $              As-of Join    Technical    TimeGPT    Engine    Export
```

### Module Dependencies
```
src/
├── db/client.py          # ← Database foundation
├── data_ingest/          # ← Raw data collection
│   ├── fred.py          # ← Macro data (requires FRED_API_KEY)
│   └── yahoo.py         # ← Market data
├── timejoin/spine.py     # ← Data alignment (uses db + ingest)
├── features/tech.py      # ← Feature engineering (uses timejoin)
├── models/patchtst.py    # ← ML models (uses features)
└── backtest/engine.py    # ← Strategy testing (uses models)
```

## 🧪 Testing & Validation

### Environment Verification
```bash
python -c "
import pandas, numpy, torch, transformers
print('✅ Core ML packages')

import fredapi, yfinance
print('✅ Data APIs')

from prisma import Prisma
print('✅ Database client')
"
```

### Database Health Check
```bash
python -c "
import asyncio
from src.db.client import health_check
result = asyncio.run(health_check())
print(f'Database: {result}')
"
```

### Model Training Test
```bash
python -c "
from src.models.patchtst import PatchTSTWrapper
import numpy as np
model = PatchTSTWrapper(random_state=42)
print('✅ Model initialized')
"
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'prisma'"
```bash
pip install prisma
npx prisma generate
```

#### "Command 'npx' not found"
```bash
# Install Node.js
conda install -c conda-forge nodejs npm
# or download from https://nodejs.org/
```

#### "Database connection failed"
1. Check `.env` file has correct `DATABASE_URL`
2. Ensure PostgreSQL is running
3. Test connection: `psql $DATABASE_URL`

#### "Import errors for ML packages"
```bash
# Reinstall core packages
pip install torch transformers --upgrade
```

#### "Jupyter kernel not found"
```bash
# Install kernel in environment
python -m ipykernel install --user --name=msrk-v3
```

### Environment Issues

#### Conda environment not found
```bash
# List environments
conda env list

# Recreate if missing
conda env create -f environment.yml
```

#### Virtual environment not working
```bash
# Remove and recreate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📊 Performance Optimization

### Memory Management
- Use chunked data processing for large datasets
- Clear unused DataFrames: `del df; gc.collect()`
- Monitor memory: `psutil.virtual_memory()`

### GPU Utilization
- Check availability: `torch.cuda.is_available()`
- Monitor usage: `nvidia-smi`
- Use mixed precision: `torch.cuda.amp`

### Database Optimization
- Use indexes on timestamp columns
- Batch inserts with `create_many()`
- Connection pooling for high-frequency queries

## 🔧 Advanced Configuration

### Custom Model Configuration
Edit `config/example_strategy.yaml`:
```yaml
model:
  type: "patchtst"
  context_length: 512  # Increase for more history
  prediction_length: 60  # Longer forecasts
  hidden_size: 256     # Larger model
```

### Kaggle GPU Setup
```python
# In Kaggle notebook
!pip install -r /kaggle/working/requirements.txt
%env DATABASE_URL=your_connection_string
%env FRED_API_KEY=your_api_key

# Use GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

### Production Deployment
1. **Environment Variables**: Use proper secrets management
2. **Database**: Use connection pooling and read replicas
3. **Monitoring**: Add logging and metrics collection
4. **Scaling**: Containerize with Docker for multi-instance deployment

## 📚 Learning Resources

### Key Papers
- **PatchTST**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
- **TimeGPT**: "TimeGPT-1" by Nixtla
- **TimesNet**: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"

### Documentation
- **Prisma**: https://prisma.io/docs
- **Transformers**: https://huggingface.co/docs/transformers
- **PyTorch**: https://pytorch.org/docs
- **FRED API**: https://fred.stlouisfed.org/docs/api/

### Community
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Examples**: See `notebooks/` for working examples
