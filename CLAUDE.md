# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.12 cryptocurrency price prediction project using QLib framework and LightGBM for BTCUSDT trading. Script-driven workflow (not a standard Python package).

## Environment

- Python: 3.12 (see `.python-version`)
- Package manager: **uv** (required)
- Install dependencies: `uv sync`
- Run scripts: `uv run python <script>.py [args...]`

## Common Commands

```bash
# Download Binance 1h data
uv run python -m scripts.data.request_1h

# Clean CSV → QLib source CSV
uv run python -m scripts.data.clean_data --input data/raw/BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv

# QLib source CSV → QLib binary
uv run python -m scripts.data.dump_bin dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min

# Train LightGBM model (single or rolling mode)
uv run python -m scripts.training.lgbm_workflow

# Export feature importance
uv run python -m scripts.analysis.dump_lgbm_feature_importance --importance-type gain --out reports/feature_importance/lgbm_feature_importance.csv --top 20

# Run spot backtest
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "/absolute/path/to/pred_2024*.pkl" --provider-uri ./qlib_data/my_crypto_data

# Run phase 2 baseline / scan
uv run python -m scripts.analysis.run_phase2_baseline --mlruns-root ./mlruns --provider-uri ./qlib_data/my_crypto_data --year 2024 --scan
```

## Architecture

```
scripts/data/request_1h.py                → Download raw Binance data
scripts/data/clean_data.py                → Format CSV for QLib
scripts/data/dump_bin.py                  → Convert source CSV to QLib binary
scripts/training/lgbm_workflow.py         → Training entrypoint
nusri_project/training/lgbm_workflow.py   → Training workflow logic
scripts/analysis/backtest_spot_strategy.py → Spot backtest entrypoint
nusri_project/strategy/*                  → Strategy and scan helpers
```

**Key modules:**
- `nusri_project/config/alpha261_config.py` — Alpha261 and Top23 feature configurations
- `nusri_project/training/lgbm_workflow.py` — Model training with single/rolling modes, configurable feature sets
- `nusri_project/strategy/backtest_spot_strategy.py` — QLib-first spot backtest wrapper
- `nusri_project/strategy/phase2_strategy_research.py` — phase 2 scan helpers
- `qlib_data/my_crypto_data/` — QLib binary data directory

**Configurable options in `nusri_project/training/lgbm_workflow.py`:**
- `FEATURE_SET`: "alpha261" or "top23"
- `RUN_MODE`: "single" or "rolling"

## Data Paths

- Raw downloads: `data/raw/`
- QLib source CSV: `qlib_source_data/`
- QLib binary: `qlib_data/my_crypto_data/`
- MLflow logs: `mlruns/`
- Archived artifacts: `archive/artifacts/`

## Important Notes

- QLib binary data must exist before training (run the pipeline in order)
- High-frequency time format: `%Y-%m-%d %H:%M:%S`
- Alpha261 factor names must be unique (raises `ValueError` on duplicates)
- Do not commit: `qlib_data/`, `mlruns/`, large CSV files (see `.gitignore`)
- Before writing custom backtest or portfolio-analysis code, check whether QLib already provides the needed capability through `qlib.backtest.backtest`, `qlib.contrib.evaluate.backtest_daily`, or `qlib.workflow.record_temp.PortAnaRecord`
- If QLib has a suitable built-in path, prefer configuring and integrating it over maintaining a parallel handwritten backtest stack
