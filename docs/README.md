# MSRK v3 Documentation

Welcome to the Macro Signal Research Kit v3 documentation. This comprehensive guide will help you understand, use, and contribute to the platform.

## Table of Contents

### Getting Started
- [Quick Start Guide](tutorials/quickstart.md)
- [Installation](../README.md#installation)
- [Configuration](tutorials/configuration.md)

### API Reference
- [Database Client](api/database.md)
- [Data Ingestion](api/data_ingest.md)
- [Feature Engineering](api/features.md)
- [Models](api/models.md)
- [Backtesting](api/backtest.md)

### Tutorials
- [Setting up Your First Strategy](tutorials/first_strategy.md)
- [Working with Macro Data](tutorials/macro_data.md)
- [Building Custom Features](tutorials/custom_features.md)
- [Running Backtests](tutorials/backtesting.md)
- [Model Training and Evaluation](tutorials/model_training.md)

### Development
- [Development Setup](../DEVELOPMENT.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Code Style Guide](development/style_guide.md)
- [Testing Guide](development/testing.md)

### Deployment
- [Production Deployment](deployment/production.md)
- [Monitoring and Logging](deployment/monitoring.md)
- [Scaling Considerations](deployment/scaling.md)

## Architecture Overview

MSRK v3 is built as a modular, scalable platform for financial time-series forecasting:

```
┌─────────────────────────────────────────────────────────────┐
│                    MSRK v3 Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Jupyter Notebooks  │  Web Interface  │  CLI Tools         │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
│  • Strategy Engine  • Backtest Engine  • Model Training    │
├─────────────────────────────────────────────────────────────┤
│                     Core Services                           │
│  • Data Ingestion  • Feature Engineering  • Time Join      │
├─────────────────────────────────────────────────────────────┤
│                    Data Sources                             │
│  • FRED API       • Yahoo Finance     • Custom APIs        │
├─────────────────────────────────────────────────────────────┤
│                     Storage Layer                           │
│  • PostgreSQL Database  • File Storage  • Model Artifacts  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-Source Data Ingestion**: Automated collection from FRED, Yahoo Finance, and custom APIs
- **Advanced Feature Engineering**: Technical indicators, macro signals, and custom transformations
- **State-of-the-Art Models**: PatchTST, TimeGPT, and ensemble methods
- **Robust Backtesting**: Event-driven simulation with realistic execution modeling
- **Production Ready**: Comprehensive testing, monitoring, and deployment tools

## Quick Examples

### Data Ingestion
```python
from src.data_ingest.fred import FredClient
from src.db.client import get_db_client

# Fetch unemployment rate data
fred = FredClient(api_key="your_api_key")
data = await fred.get_series("UNRATE", start_date="2020-01-01")

# Store in database
db = await get_db_client()
await db.macrorelease.create_many(data=data)
```

### Feature Engineering
```python
from src.features.tech import TechnicalIndicators

# Calculate technical indicators
indicators = TechnicalIndicators()
features = indicators.calculate_all(price_data)
```

### Model Training
```python
from src.models.patchtst import PatchTSTModel

# Train PatchTST model
model = PatchTSTModel(config={
    "patch_len": 16,
    "stride": 8,
    "d_model": 128
})
await model.train(features, targets)
```

### Backtesting
```python
from src.backtest.engine import BacktestEngine

# Run strategy backtest
engine = BacktestEngine()
results = await engine.run_backtest(
    strategy="macro_momentum",
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/your-org/msrk-v3/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/your-org/msrk-v3/discussions)
- **Documentation**: This documentation is continuously updated and improved

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
