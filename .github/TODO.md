## 1. Environment

- [x] Create a Python 3.10 virtual env `.venv` and activate it.  
- [x] `pip install -r requirements.txt` to pull core libs.  
- [x] Add a **.env** template (`config/env.sample`) with placeholders for:
  - `DATABASE_URL`
  - `TIMEGPT_API_KEY`
  - `GEMINI_API_KEY`
- [x] Verify `python -c "import torch, prisma"` runs without error.
- [x] Automated setup scripts (`setup_env.sh/bat`, `activate_env.sh/bat`) created.
- [x] Environment verification script (`scripts/verify_env.py`) implemented.

---

## 2. Dev Container (optional but recommended)

- [x] Place **Dockerfile** and **devcontainer.json** in `.devcontainer/`.  
- [x] VS Code extensions and port forwarding configured.
- [x] Reopen the project in the container and rerun the environment tests.

---

## 3. Database

- [x] Provision a Postgres 15 instance on Supabase or Railway.  
- [x] Replace `DATABASE_URL` in `.env` with the new instance URL.  
- [x] In `prisma/schema.prisma` define models:
  - `MacroRelease`
  - `MarketPrice`
  - `Feature`
  - `TextFeature`
  - `Strategy`
  - `Signal`
  - `ModelArtifact`
- [x] `npx prisma db push` to apply schema.  
- [x] Write `src/db/client.py` that exposes a singleton `Prisma` instance.  
- [x] Add a smoke test: connect, insert a dummy row, query it back.
- [x] Database migration, seeding, and backup scripts implemented.

---

## 4. Data Ingestion Layer

- [x] **FRED** → `src/data_ingest/fred.py`
  - `fetch_series(series_id, start, end)`  
  - bulk insert into `MacroRelease`
- [x] **BLS Unemployment** → `bls_unrate.py`
- [x] **Yahoo Finance prices** → `yahoo.py`
- [x] Pull a one-month sample for each source and confirm counts in DB.  
- [x] Write unit tests for each ingestor using `pytest` + `pytest-asyncio`.

---

## 5. Time Spine and Join

- [x] `src/timejoin/spine.py`  
  - `make_spine(ticker, start, end, freq)` returns empty DataFrame index.
- [x] `src/timejoin/asof_join.py`  
  - `join_macro(px_df, series_list)` attaches latest macro values via `merge_asof`.
- [x] Add edge-case tests: holiday gaps, DST jumps.

---

## 6. Feature Engineering

- [x] `tech.py`  
  - rolling return, ATR, z-score, volatility percentile.
- [x] `macro_event.py`  
  - surprise calculation, event dummy within ±N minutes.
- [x] `text_sentiment.py`  
  - Gemini call → sentiment score.
- [x] Store outputs in `Feature` or `TextFeature`.  
- [x] Create `feature_version` constant for traceability.

---

## 7. Model Layer

- [x] **PatchTST** implementation `models/patchtst.py`
  - `finetune_patchtst(X_tr, y_tr, X_val, y_val, gpu, epochs, seed)`
- [x] **TimeGPT** wrapper `models/timegpt.py`
- [x] **Baseline LSTM** `models/lstm.py`
- [x] Add Optuna tuning helper `tuning/optuna_utils.py`.  
- [x] Unit test: train for 1 epoch on toy data, ensure metrics dict returned.

---

## 8. Back-test Engine

- [x] `backtest/engine.py`
  - Accepts rule config + model predictions.
  - Generates `Signal` rows and summary stats.  
- [x] Implement friction costs (bps + fixed spread).  
- [x] Verify no look-ahead by asserting `signal_time < trade_time`.
- [x] Add `backtest/engine_test.py` with unit tests for:
  - Single trade execution
  - Multiple trades with overlapping signals
  - Edge cases like no trades or all losses
- [x] Add a visualization function to plot correlation and variance of returns against signals.
---


## 9. Strategy Registry

- [x] Define YAML schema in `config/example_strategy.yaml`.  
- [x] Loader `src/strategies/loader.py` saves entry in `Strategy` table.  
- [x] Add CLI wrapper `scripts/register_strategy.py`.

---

## 10. Notebooks

1. **daily_workflow.ipynb**
   - install → connect → ingest → features → PatchTST tune → back-test → save dataset
2. **intraday_transformer.ipynb**
   - minute data workflow with small window and TimeGPT call  
- [ ] Limit each notebook cell to < 2 000 characters.

---

## 11. Tests and CI

- [ ] Add `pytest` with coverage > 80 %.  
- [ ] GitHub Actions:
  - lint (`ruff`)  
  - unit tests  
  - schema lint (`prisma format`)  
- [ ] Block merge on test failure.

---

## 12. Documentation

- [ ] Update `README.md` with quick-start (clone → .env → run first notebook).  
- [ ] Add **docs/architecture.md** showing data flow diagram.  
- [ ] Provide **docs/prompts.md** listing ready-to-copy Kaggle cell snippets.

---

## 13. Versioning & Snapshots

- [ ] Implement `scripts/dump_snapshot.py` to export critical tables to `s3://msrk-snapshots/YYYYMMDD`.  
- [ ] Add cron in crontab.txt (for later server use).

---

## 14. Stretch Goals

- [ ] Prefect flow for scheduled ingestion.  
- [ ] FastAPI microservice to trigger runs remotely.  
- [ ] Streamlit dashboard for quick result browsing.

---

## 15. Done Criteria

- [ ] Daily notebook runs to completion on Kaggle within six hours.  
- [ ] Metrics meet Sharpe uplift goal.  
- [ ] All tasks checked off in this file.