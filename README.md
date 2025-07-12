# Macro Signal Research Kit v3

**Advanced time-series forecasting for financial markets using transformer models and macro-economic data.**

## 🎯 Overview

MSRK v3 is a sophisticated research platform for forecasting financial time series using:
- **Mixed-frequency data**: Macro releases + market prices with as-of-point joins
- **Modern ML**: PatchTST, TimesNet, LSTM baselines + TimeGPT API
- **Production-ready**: Prisma + PostgreSQL storage, Kaggle GPU execution
- **Reproducible**: Fixed seeds, comprehensive logging, version control

## 🚀 Quick Start

### Prerequisites
- VS Code with Dev Containers extension
- Docker Desktop
- PostgreSQL database (local or cloud)
- API keys: FRED, TimeGPT (optional), Gemini (optional)

### Setup

1. **Clone and open in VS Code**:
```bash
git clone <repository-url>
cd FinForecastForex
code .
```

2. **Setup isolated environment** (Choose one):

   **Option A: Automated Setup (Recommended)**
   ```bash
   # Linux/Mac
   chmod +x setup_env.sh
   ./setup_env.sh
   
   # Windows
   setup_env.bat
   ```

   **Option B: Manual Conda Setup**
   ```bash
   conda env create -f environment.yml
   conda activate msrk-v3
   npm install prisma @prisma/client
   npx prisma generate
   ```

   **Option C: Manual venv Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate.bat  # Windows
   pip install -r requirements.txt
   npm install prisma @prisma/client
   npx prisma generate
   ```

3. **Configure environment**:
```bash
cp .env.template .env
# Edit .env with your API keys and database URL
```

4. **Initialize database**:
```bash
npx prisma db push
```

5. **Run daily workflow**:
```bash
jupyter notebook notebooks/daily_workflow.ipynb
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| End-to-end time (fresh clone) | ≤ 60 min | - |
| Full CV + backtest (Kaggle T4) | ≤ 6 hours | - |
| Sharpe uplift vs baseline | ≥ 15% | - |
| Reproducibility (std dev) | ≤ 1e-4 | - |

## 🏗️ Architecture

```
Data Sources → Ingestion → Time Join → Features → Models → Backtest → Export
     ↓             ↓          ↓          ↓        ↓        ↓        ↓
   FRED API    PostgreSQL   As-of     Technical  PatchTST  Engine  Kaggle
   Yahoo $      (Prisma)     Join      Macro     TimeGPT   Rules   Datasets
   BLS API                  Spine    Sentiment   LSTM             Reports
```

## 📁 Project Structure

```
msrk/
├── prisma/                 # Database schema and migrations
│   └── schema.prisma
├── src/
│   ├── data_ingest/        # Data collection modules
│   │   ├── fred.py         # FRED API macro data
│   │   ├── yahoo.py        # Yahoo Finance market data
│   │   └── bls.py          # Bureau of Labor Statistics
│   ├── timejoin/           # Time series alignment
│   │   ├── spine.py        # Time spine creation
│   │   └── asof_join.py    # As-of-point joins
│   ├── features/           # Feature engineering
│   │   ├── tech.py         # Technical indicators
│   │   ├── macro_event.py  # Macro surprise features
│   │   └── text_sentiment.py # LLM sentiment analysis
│   ├── models/             # ML models
│   │   ├── patchtst.py     # PatchTST transformer
│   │   ├── timesnet.py     # TimesNet implementation
│   │   └── lstm.py         # LSTM baseline
│   ├── tuning/             # Hyperparameter optimization
│   │   └── optuna_utils.py
│   ├── backtest/           # Strategy backtesting
│   │   └── engine.py
│   └── db/                 # Database client
│       └── client.py
├── notebooks/              # Jupyter workflows
│   ├── daily_workflow.ipynb
│   └── intraday_transformer.ipynb
├── config/                 # Configuration files
│   ├── example_strategy.yaml
│   └── model_cfg.yaml
├── requirements.txt
├── .env.template
└── README.md
```

## 🔧 Key Components

### Data Ingestion
- **FRED API**: Macro economic indicators with release times
- **Yahoo Finance**: OHLCV market data for major assets
- **BLS**: Employment and inflation data
- **Custom scrapers**: Fed statements, earnings calls

### Time Join Engine
- **Spine creation**: Regular time grids (daily/intraday)
- **As-of joins**: Latest macro values at each market timestamp
- **Mixed frequency**: Handle daily macro + minute market data

### Feature Engineering
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR
- **Macro surprises**: Actual vs consensus deviations
- **Text sentiment**: Gemini API analysis of Fed statements
- **Lagged features**: Multi-horizon historical values

### Models
- **PatchTST**: State-of-the-art transformer for time series
- **TimeGPT**: Zero-shot forecasting via Nixtla API
- **TimesNet**: Multi-scale temporal analysis
- **LSTM**: Baseline recurrent model

### Backtesting
- **Hybrid signals**: Rule-based + ML predictions
- **Transaction costs**: Realistic trading friction
- **Risk management**: Position sizing, stop losses
- **Performance metrics**: Sharpe, Calmar, max drawdown

## 📈 Usage Examples

### Basic Daily Forecast
```python
from src.timejoin.spine import create_feature_spine
from src.models.patchtst import PatchTSTWrapper

# Create feature matrix
df = await create_feature_spine('SPY', '2020-01-01', '2024-01-01')

# Train model
model = PatchTSTWrapper(context_length=256, prediction_length=30)
X, y = model.prepare_data(df)
metrics = model.fit(X_train, y_train, X_val, y_val, epochs=25)

# Generate forecasts
predictions = model.predict(X_test)
```

### TimeGPT Integration
```python
from nixtla import TimeGPT

tg = TimeGPT(api_key=API_KEY)
forecast = tg.forecast(df[['ds', 'y']], h=30)

# Store in database
await db.feature.create_many(
    forecast.reset_index().to_dict('records')
)
```

### Macro Surprise Analysis
```python
from src.features.tech import macro_surprise_features

# Generate surprise-based features
surprise_df = macro_surprise_features(df)
surprise_index = surprise_df['macro_surprise_index']

# Correlate with future returns
correlation = surprise_index.rolling(20).corr(
    df['close'].pct_change(5).shift(-5)
)
```

## 🔄 Kaggle Workflow

1. **Develop locally** in VS Code with full debugging
2. **Push to Kaggle**:
   ```bash
   kaggle kernels push -p .
   ```
3. **Run on GPU**: Heavy training jobs on Kaggle T4
4. **Pull results**: Download artifacts and datasets
5. **Iterate**: Refine models based on results

## ⚙️ Configuration

### Strategy Config (`config/example_strategy.yaml`)
```yaml
strategy:
  name: "SPY_PatchTST_Daily"
  ticker: "SPY"
  
model:
  type: "patchtst"
  context_length: 256
  prediction_length: 30
  
training:
  epochs: 25
  learning_rate: 0.0001
  random_seed: 42
```

### Environment Variables (`.env`)
```bash
DATABASE_URL=postgresql://user:pass@host:port/db
FRED_API_KEY=your_fred_key
TIMEGPT_API_KEY=your_timegpt_key
GEMINI_API_KEY=your_gemini_key
```

## 📊 Database Schema

### Core Tables
- **MacroRelease**: Economic indicators with consensus/actual/surprise
- **MarketPrice**: OHLCV data with timestamps
- **Feature**: Engineered features (technical, macro, sentiment)
- **ModelArtifact**: Serialized models with metadata
- **BacktestResult**: Performance metrics and trade logs

### Optimizations
- Indexed on (ticker, timestamp) for fast time-series queries
- Efficient storage with PostgreSQL TIMESTAMP and BIGINT types
- Async operations with Prisma Python client

## 🧪 Testing & Validation

### Unit Tests
```bash
pytest src/ -v
```

### Reproducibility Check
```bash
python -c "
from src.models.patchtst import PatchTSTWrapper
import numpy as np

# Run same model twice with fixed seed
results = []
for i in range(2):
    model = PatchTSTWrapper(random_state=42)
    # ... train model ...
    results.append(metrics['final_val_loss'])

assert abs(results[0] - results[1]) < 1e-4, 'Not reproducible!'
"
```

### Performance Benchmark
```bash
time jupyter nbconvert --execute notebooks/daily_workflow.ipynb
```

## 🚧 Current Limitations

- **ETL Scheduling**: Manual notebook execution (no Airflow)
- **Transaction Costs**: Simplified model in backtester
- **Real-time Deployment**: No production inference pipeline
- **Data Quality**: Limited validation and anomaly detection

## 🗺️ Roadmap

### Q1 2025
- [ ] Finalize Prisma schema and migrations
- [ ] Complete PatchTST vs TimeGPT benchmark
- [ ] Add text sentiment features with Gemini
- [ ] Implement comprehensive backtesting engine

### Q2 2025
- [ ] Add TimesNet and advanced transformer models
- [ ] Integrate transaction cost modeling
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Performance optimization for large datasets

### Q3 2025
- [ ] Real-time inference API
- [ ] Advanced risk management features
- [ ] Multi-asset portfolio optimization
- [ ] Production deployment on cloud platforms

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See `docs/` folder for detailed guides

## 🙏 Acknowledgments

- **PatchTST**: "A Time Series is Worth 64 Words" paper
- **TimeGPT**: Nixtla team for zero-shot forecasting API
- **FRED**: Federal Reserve Economic Data
- **Transformers**: Hugging Face ecosystem
