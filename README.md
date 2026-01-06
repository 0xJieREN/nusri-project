# NUSRI Project

This project implements machine learning workflows for cryptocurrency price prediction using QLib and LightGBM models.

## Project Structure

### Data Acquisition

- **`request_1d.py`** - Downloads daily (1d) BTCUSDT data from Binance API with retry logic
- **`request_1h.py`** - Downloads hourly (1h) BTCUSDT data from Binance API with retry logic

### Data Processing

- **`clean_data.py`** - Prepares raw CSV data for QLib format (adds symbol, date formatting, factor column, and `vwap`)
- **`dump_bin.py`** - Converts cleaned CSV data into QLib binary format for efficient data loading

### Model Training

- **`LGBM_workflow.py`** - Main training pipeline using QLib framework with LightGBM model
  - Configures Alpha158 features for feature engineering
  - Trains on historical data (2018-2022)
  - Validates on out-of-sample period (2023-2024)
  - Logs metrics via MLflow

### Testing

- **`test_qlib.py`** - Tests QLib data loading and feature retrieval

### Data

- **`qlib_source_data/`** - Cleaned raw data in CSV format
- **`qlib_data/my_crypto_data/`** - Processed binary data for QLib
- **`mlruns/`** - MLflow experiment tracking logs

## Dependencies

- pyqlib >= 0.9.7
- requests >= 2.32.5

## Workflow

1. Run `request_1d.py` or `request_1h.py` to download Binance data
2. Run `clean_data.py` to prepare data for QLib (for 1h, use Qlib freq `60min`)
3. Run `dump_bin.py` to convert cleaned data into binary format (for 1h, use `--freq 60min`)
4. Run `LGBM_workflow.py` to train the model
5. Check `mlruns/` for training metrics and results

Example (1h data):
- `python clean_data.py --input BTCUSDT_1h_binance_data.csv --freq 60min --output qlib_source_data/BTCUSDT.csv`
- `python dump_bin.py dump_all --data_path qlib_source_data --qlib_dir qlib_data/my_crypto_data --freq 60min`
