# NUSRI Project

Python 3.12 cryptocurrency research repository built around QLib and LightGBM for BTCUSDT hourly prediction, signal evaluation, and spot strategy backtesting.

## Structure

### Active entrypoints

- `scripts/data/` — data download and preparation commands
- `scripts/training/` — training commands
- `scripts/analysis/` — feature importance, backtest, and strategy scan commands

### Reusable logic

- `nusri_project/config/` — factor and workflow config
- `nusri_project/training/` — training workflow logic
- `nusri_project/strategy/` — spot strategy, backtest wrapper, and phase 2 scan helpers

### Data and outputs

- `data/raw/` — active raw market downloads
- `qlib_source_data/` — cleaned CSVs for QLib ingestion
- `qlib_data/my_crypto_data/` — QLib binary data
- `mlruns/` — MLflow experiment outputs

### Archived legacy files

- `archive/artifacts/` — archived generated CSV artifacts from earlier iterations
- `archive/legacy_test/` — archived old `test/` directory and Alpha158 notes/scripts

## Common Commands

### Download Binance 1h data

```bash
uv run python -m scripts.data.request_1h
```

### Clean raw CSV into QLib source format

```bash
uv run python -m scripts.data.clean_data --input data/raw/BTCUSDT_1h_binance_data.csv --output qlib_source_data/BTCUSDT.csv
```

### Dump QLib binary data

```bash
uv run python -m scripts.data.dump_bin dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min
```

### Train LightGBM workflow

```bash
uv run python -m scripts.training.lgbm_workflow
```

Optional:

```bash
uv run python -m scripts.training.lgbm_workflow --feature-set top23 --run-mode rolling
```

### Export feature importance

```bash
uv run python -m scripts.analysis.dump_lgbm_feature_importance --importance-type gain --out reports/feature_importance/lgbm_feature_importance.csv --top 20
```

### Run spot backtest on saved prediction files

```bash
uv run python -m scripts.analysis.backtest_spot_strategy --pred-glob "/absolute/path/to/pred_2024*.pkl" --provider-uri ./qlib_data/my_crypto_data
```

### Run phase 2 baseline and parameter scan

```bash
uv run python -m scripts.analysis.run_phase2_baseline --mlruns-root ./mlruns --provider-uri ./qlib_data/my_crypto_data --year 2024 --scan
```

## Notes

- QLib binary data must exist before training or backtesting.
- Spot backtests now use a QLib-first implementation with a custom single-asset order generator for fractional crypto sizing.
- Generated reports and temporary research outputs should not be committed unless explicitly needed.
