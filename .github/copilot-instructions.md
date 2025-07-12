# Copilot Instructions for Macro Signal Research Kit v3

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a sophisticated time-series forecasting project focused on macro economic data and financial market predictions using transformer models.

## Key Technologies
- **Database**: Prisma ORM with PostgreSQL (no SQLAlchemy/Alembic)
- **ML Models**: PatchTST, TimesNet, LSTM, TimeGPT API
- **Data Sources**: FRED API, Yahoo Finance, BLS, custom scrapers
- **Execution**: Local development in VS Code, heavy compute on Kaggle GPU
- **Languages**: Python primarily, with some SQL

## Coding Standards
- Use async/await patterns for database operations with Prisma
- Follow the folder structure: src/data_ingest/, src/models/, src/features/, etc.
- Prefer pandas for data manipulation, torch for deep learning
- All database models should use Prisma schema definitions
- Keep functions focused and modular for Kaggle notebook execution
- Use type hints and docstrings for all functions
- Seed random operations for reproducibility (target std ≤ 1e-4)

## Architecture Principles
- Mixed-frequency data handling (daily/intraday)
- As-of-point time joins for macro releases
- Feature engineering with technical indicators and macro surprises
- Hybrid rule + ML signal generation for backtesting
- Export-friendly for Kaggle datasets and reporting

## Performance Targets
- End-to-end pipeline: ≤ 60 minutes from fresh clone
- Full CV + backtest on Kaggle T4: ≤ 6 hours
- Sharpe uplift vs baseline: ≥ 15%
