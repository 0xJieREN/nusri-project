from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from strategy_config import SpotStrategyConfig


REQUIRED_COLUMNS = ["pred_return", "real_return"]


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing required columns: {missing}")

    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    if isinstance(normalized.index, pd.MultiIndex):
        normalized.index = pd.to_datetime(normalized.index.get_level_values(0))
    else:
        normalized.index = pd.to_datetime(normalized.index)

    normalized = normalized.sort_index()
    if normalized.index.has_duplicates:
        raise ValueError("prediction frame contains duplicate timestamps after normalization")
    return normalized


def load_prediction_frames(paths: Iterable[Path]) -> pd.DataFrame:
    normalized_frames = [normalize_prediction_frame(pd.read_pickle(path)) for path in paths]
    if not normalized_frames:
        raise ValueError("no prediction files were provided")

    combined = pd.concat(normalized_frames).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("combined prediction frames contain duplicate timestamps")
    return combined


def _compute_hours_since(timestamp: pd.Timestamp, previous: pd.Timestamp | None) -> float:
    if previous is None:
        return float("inf")
    delta = timestamp - previous
    return delta.total_seconds() / 3600.0


def _resolve_target_position(
    pred_return: float,
    current_position: float,
    drawdown: float,
    holding_hours: float,
    hours_since_trade: float,
    config: SpotStrategyConfig,
) -> float:
    if pred_return >= config.full_position_threshold:
        target_position = config.max_position
    elif pred_return >= config.entry_threshold:
        target_position = min(config.max_position, 0.5)
    elif pred_return <= config.exit_threshold:
        target_position = 0.0
    else:
        target_position = current_position

    if drawdown >= config.drawdown_de_risk_threshold:
        target_position = min(target_position, config.de_risk_position)

    reducing_position = target_position < current_position
    increasing_position = target_position > current_position

    if reducing_position and holding_hours < config.min_holding_hours:
        return current_position

    if increasing_position and hours_since_trade < config.cooldown_hours:
        return current_position

    return target_position


def run_spot_backtest(frame: pd.DataFrame, config: SpotStrategyConfig) -> pd.DataFrame:
    config.validate()
    normalized = normalize_prediction_frame(frame)

    equity = config.initial_cash
    high_watermark = equity
    current_position = 0.0
    holding_started_at: pd.Timestamp | None = None
    last_trade_at: pd.Timestamp | None = None
    records: list[dict] = []

    for timestamp, row in normalized.iterrows():
        equity_before = equity
        drawdown_before = 0.0 if high_watermark <= 0 else 1.0 - (equity_before / high_watermark)
        holding_hours = (
            _compute_hours_since(timestamp, holding_started_at) if current_position > 0 else 0.0
        )
        hours_since_trade = _compute_hours_since(timestamp, last_trade_at)

        target_position = _resolve_target_position(
            pred_return=float(row["pred_return"]),
            current_position=current_position,
            drawdown=drawdown_before,
            holding_hours=holding_hours,
            hours_since_trade=hours_since_trade,
            config=config,
        )

        turnover = abs(target_position - current_position)
        fee_paid = equity_before * turnover * config.fee_rate
        equity_after_fee = equity_before - fee_paid
        gross_return = target_position * float(row["real_return"])
        equity = equity_after_fee * (1.0 + gross_return)
        strategy_return = (equity / equity_before) - 1.0

        if turnover > 0:
            last_trade_at = timestamp
            if current_position == 0 and target_position > 0:
                holding_started_at = timestamp
            elif target_position == 0:
                holding_started_at = None

        current_position = target_position
        high_watermark = max(high_watermark, equity)
        drawdown_after = 0.0 if high_watermark <= 0 else 1.0 - (equity / high_watermark)

        records.append(
            {
                "pred_return": float(row["pred_return"]),
                "real_return": float(row["real_return"]),
                "position": current_position,
                "turnover": turnover,
                "fee_paid": fee_paid,
                "gross_return": gross_return,
                "strategy_return": strategy_return,
                "equity_before": equity_before,
                "equity": equity,
                "drawdown": drawdown_after,
            }
        )

    return pd.DataFrame.from_records(records, index=normalized.index)


def summarize_backtest(result: pd.DataFrame, annualization_periods: int = 24 * 365) -> dict[str, float]:
    if result.empty:
        raise ValueError("cannot summarize an empty backtest result")

    equity_before = float(result.iloc[0].get("equity_before", result.iloc[0]["equity"]))
    ending_equity = float(result.iloc[-1]["equity"])
    total_return = (ending_equity / equity_before) - 1.0
    periods = len(result)

    if total_return <= -1.0:
        annualized_return = -1.0
    else:
        annualized_return = (1.0 + total_return) ** (annualization_periods / periods) - 1.0

    strategy_returns = result["strategy_return"].astype(float)
    annualized_volatility = strategy_returns.std(ddof=0) * (annualization_periods**0.5)

    equity_curve = result["equity"].astype(float)
    rolling_peak = equity_curve.cummax()
    drawdowns = 1.0 - (equity_curve / rolling_peak)
    max_drawdown = float(drawdowns.max())

    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    bar_hours = 1.0
    if len(result.index) > 1:
        deltas = result.index.to_series().diff().dropna()
        if not deltas.empty:
            bar_hours = deltas.dt.total_seconds().median() / 3600.0

    holding_hours: list[float] = []
    current_holding = 0.0
    for position in result["position"]:
        if position > 0:
            current_holding += bar_hours
        elif current_holding > 0:
            holding_hours.append(current_holding)
            current_holding = 0.0
    if current_holding > 0:
        holding_hours.append(current_holding)

    avg_holding_hours = sum(holding_hours) / len(holding_hours) if holding_hours else 0.0

    return {
        "starting_equity": equity_before,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "turnover": float(result["turnover"].sum()),
        "fee_paid": float(result["fee_paid"].sum()),
        "avg_holding_hours": avg_holding_hours,
        "exposure_ratio": float(result["position"].mean()),
    }


def compute_monthly_returns(result: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        result["strategy_return"]
        .groupby(result.index.to_period("M"))
        .apply(lambda series: (1.0 + series).prod() - 1.0)
        .to_frame("monthly_return")
    )
    monthly.index = monthly.index.astype(str)
    return monthly


def _expand_prediction_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        paths.extend(matches)

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("no prediction files matched the provided glob patterns")
    return unique_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal fee-aware BTCUSDT spot long/flat backtest on prediction files."
    )
    parser.add_argument(
        "--pred-glob",
        action="append",
        required=True,
        help="Glob pattern for prediction pickle files. Repeat the flag for multiple patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/spot_backtest",
        help="Directory where equity curve, monthly returns, and summary JSON are written.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--entry-threshold", type=float, default=0.0025)
    parser.add_argument("--exit-threshold", type=float, default=0.0005)
    parser.add_argument("--full-position-threshold", type=float, default=0.005)
    parser.add_argument("--min-holding-hours", type=int, default=24)
    parser.add_argument("--cooldown-hours", type=int, default=12)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--drawdown-de-risk-threshold", type=float, default=0.08)
    parser.add_argument("--de-risk-position", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SpotStrategyConfig(
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        full_position_threshold=args.full_position_threshold,
        min_holding_hours=args.min_holding_hours,
        cooldown_hours=args.cooldown_hours,
        max_position=args.max_position,
        drawdown_de_risk_threshold=args.drawdown_de_risk_threshold,
        de_risk_position=args.de_risk_position,
    )

    prediction_paths = _expand_prediction_globs(args.pred_glob)
    combined = load_prediction_frames(prediction_paths)
    result = run_spot_backtest(combined, config)
    summary = summarize_backtest(result, annualization_periods=config.annualization_periods)
    monthly = compute_monthly_returns(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_dir / "equity_curve.csv")
    monthly.to_csv(output_dir / "monthly_returns.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
