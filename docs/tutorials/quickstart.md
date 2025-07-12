# Quick Start Guide

Get up and running with MSRK v3 in minutes. This guide will walk you through setting up your environment, configuring the system, and running your first analysis.

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 13+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/msrk-v3.git
cd msrk-v3
```

### 2. Set Up Environment

Choose your preferred environment manager:

#### Option A: Conda (Recommended)
```bash
# Run the setup script
./setup_env.sh  # Linux/Mac
# or
setup_env.bat   # Windows

# Activate environment
./activate_env.sh  # Linux/Mac
# or  
activate_env.bat   # Windows
```

#### Option B: Virtual Environment
```bash
python -m venv msrk_env
source msrk_env/bin/activate  # Linux/Mac
# or
msrk_env\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Install Node Dependencies

```bash
npm install -g prisma
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp config/env.sample .env

# Edit .env with your API keys and database URL
nano .env  # or use your preferred editor
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `FRED_API_KEY`: FRED API key (get from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))
- `YAHOO_USER_AGENT`: User agent for Yahoo Finance requests

### 5. Set Up Database

```bash
# Initialize database schema
npx prisma db push

# Seed with sample data (optional)
python scripts/seed_db.py
```

### 6. Verify Installation

```bash
python scripts/verify_env.py
```

## Your First Analysis

### 1. Start Jupyter Notebook

```bash
jupyter notebook notebooks/daily_workflow.ipynb
```

### 2. Fetch Some Data

```python
from src.data_ingest.fred import FredClient
from src.data_ingest.yahoo import YahooClient
from src.db.client import get_db_client

# Initialize clients
fred = FredClient()
yahoo = YahooClient()
db = await get_db_client()

# Fetch unemployment rate data
unemployment_data = await fred.get_series("UNRATE", start_date="2020-01-01")

# Fetch EUR/USD price data  
eurusd_data = await yahoo.get_price_data("EURUSD=X", start_date="2020-01-01")

# Store in database
await db.macrorelease.create_many(data=unemployment_data)
await db.marketprice.create_many(data=eurusd_data)
```

### 3. Generate Features

```python
from src.features.tech import TechnicalIndicators

# Calculate technical indicators
indicators = TechnicalIndicators()
features = indicators.calculate_all(eurusd_data)

# Store features
await db.feature.create_many(data=features)
```

### 4. Create a Simple Strategy

```python
from src.backtest.engine import BacktestEngine

# Define strategy
strategy_config = {
    "name": "SMA Crossover",
    "description": "Simple moving average crossover strategy",
    "config": {
        "fast_period": 10,
        "slow_period": 20,
        "symbol": "EURUSD"
    }
}

# Store strategy
strategy = await db.strategy.create(data=strategy_config)

# Run backtest
engine = BacktestEngine()
results = await engine.run_backtest(
    strategy_id=strategy.id,
    start_date="2020-01-01",
    end_date="2023-12-31"
)

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## Next Steps

Now that you have MSRK v3 running, explore these areas:

### Learn the Core Concepts

- [Working with Macro Data](macro_data.md) - Economic indicators and FRED API
- [Building Custom Features](custom_features.md) - Feature engineering techniques
- [Model Training](model_training.md) - PatchTST and other ML models
- [Backtesting Strategies](backtesting.md) - Strategy development and testing

### Explore the Data

```python
# Check what data you have
macro_releases = await db.macrorelease.find_many()
market_prices = await db.marketprice.find_many()
features = await db.feature.find_many()

print(f"Macro releases: {len(macro_releases)}")
print(f"Market prices: {len(market_prices)}")
print(f"Features: {len(features)}")
```

### Build Your First Model

```python
from src.models.patchtst import PatchTSTModel

# Prepare training data
training_data = await db.feature.find_many(
    where={
        "feature_type": "TECHNICAL",
        "symbol": "EURUSD"
    }
)

# Train model
model = PatchTSTModel(config={
    "patch_len": 16,
    "stride": 8,
    "d_model": 128
})

await model.train(training_data)
```

### Monitor Your Environment

```bash
# Check system status
python scripts/verify_env.py

# View database statistics
python scripts/db_migrate.py studio

# Run tests
pytest tests/ -v
```

## Common Issues

### Database Connection Error
- Verify PostgreSQL is running
- Check DATABASE_URL in .env file
- Run `npx prisma db push` to sync schema

### API Key Errors
- Verify API keys in .env file
- Check FRED API key is valid
- Ensure no trailing spaces in .env values

### Import Errors
- Activate your environment: `./activate_env.sh`
- Verify installation: `python scripts/verify_env.py`
- Check Python path includes src/

### Memory Issues
- Reduce batch sizes in data ingestion
- Use streaming for large datasets
- Monitor memory usage in Jupyter

## Getting Help

- **Documentation**: Browse the [docs/](../README.md) directory
- **Issues**: Report problems on GitHub Issues
- **Community**: Join discussions on GitHub Discussions
- **Development**: See [DEVELOPMENT.md](../../DEVELOPMENT.md) for contributing

## What's Next?

You're now ready to:

1. **Explore Data Sources**: Set up additional data feeds
2. **Build Strategies**: Create sophisticated trading algorithms  
3. **Train Models**: Develop custom ML models for forecasting
4. **Scale Up**: Deploy to production environments
5. **Contribute**: Help improve MSRK v3 for everyone

Happy researching! 🚀
