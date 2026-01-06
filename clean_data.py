import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_1H_INPUT = "BTCUSDT_1h_binance_data.csv"
DEFAULT_1D_INPUT = "BTCUSDT_1d_binance_data.csv"


def _infer_default_input() -> str:
    if Path(DEFAULT_1H_INPUT).exists():
        return DEFAULT_1H_INPUT
    return DEFAULT_1D_INPUT


def _format_date(series: pd.Series, freq: str) -> pd.Series:
    ts = pd.to_datetime(series, utc=False)
    if freq.lower() in {"day", "d", "1d"}:
        return ts.dt.strftime("%Y-%m-%d")
    # Qlib high-freq expects second-level timestamps in text
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S")


def _process_chunk(chunk: pd.DataFrame, symbol: str, freq: str) -> pd.DataFrame:
    if "date" not in chunk.columns:
        raise ValueError("missing required column: date")

    chunk = chunk.copy()
    chunk["symbol"] = symbol
    chunk["date"] = _format_date(chunk["date"], freq=freq)
    chunk["factor"] = 1.0

    if "amount" in chunk.columns and "volume" in chunk.columns and "vwap" not in chunk.columns:
        vol = pd.to_numeric(chunk["volume"], errors="coerce")
        amt = pd.to_numeric(chunk["amount"], errors="coerce")
        close_num = pd.to_numeric(chunk.get("close"), errors="coerce")
        vwap = (amt / vol.replace(0, pd.NA)).astype("float64")
        chunk["vwap"] = vwap.where(~vwap.isna(), close_num)

    preferred = ["date", "open", "high", "low", "close", "volume", "amount", "vwap", "symbol", "factor"]
    cols = [c for c in preferred if c in chunk.columns] + [c for c in chunk.columns if c not in preferred]
    return chunk.loc[:, cols]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare raw Binance CSV data into Qlib source CSV.")
    parser.add_argument("--input", default=_infer_default_input(), help="Input raw CSV path (default: auto-detect).")
    parser.add_argument("--sep", default=";", help="Input CSV delimiter (default: ';').")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to write into Qlib source data.")
    parser.add_argument(
        "--freq",
        default="60min",
        help="Target data frequency for date formatting (e.g. 'day' or '60min' for 1h).",
    )
    parser.add_argument(
        "--output",
        default="qlib_source_data/BTCUSDT.csv",
        help="Output Qlib source CSV path (default: qlib_source_data/BTCUSDT.csv).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk when reading large CSVs (default: 200000).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = True
    for chunk in pd.read_csv(input_path, sep=args.sep, chunksize=args.chunksize):
        out_chunk = _process_chunk(chunk, symbol=args.symbol, freq=args.freq)
        out_chunk.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False

    print(f"Qlib 源数据准备完成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
