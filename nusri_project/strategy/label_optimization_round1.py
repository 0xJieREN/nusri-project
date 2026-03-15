from __future__ import annotations

from pathlib import Path

import pandas as pd

from nusri_project.strategy.phase2_strategy_research import run_strategy_config
from nusri_project.strategy.strategy_config import SpotStrategyConfig


def build_round1_horizons() -> list[int]:
    return [24, 48, 72]


def build_round1_trading_shells() -> dict[str, dict]:
    return {
        "balanced": {
            "entry_threshold": 0.0015,
            "exit_threshold": 0.0,
            "full_position_threshold": 0.003,
            "max_position": 0.25,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.25,
        },
        "conservative": {
            "entry_threshold": 0.0015,
            "exit_threshold": 0.0,
            "full_position_threshold": 0.003,
            "max_position": 0.15,
            "min_holding_hours": 48,
            "cooldown_hours": 12,
            "drawdown_de_risk_threshold": 0.02,
            "de_risk_position": 0.25,
        },
    }


def build_round1_matrix() -> list[dict[str, object]]:
    matrix: list[dict[str, object]] = []
    for label_horizon_hours in build_round1_horizons():
        for shell_name in build_round1_trading_shells():
            matrix.append(
                {
                    "label_horizon_hours": label_horizon_hours,
                    "shell_name": shell_name,
                }
            )
    return matrix


def build_round1_run_plan(output_root: Path) -> list[dict[str, object]]:
    prediction_root = output_root / "predictions"
    plan: list[dict[str, object]] = []
    for item in build_round1_matrix():
        horizon = int(item["label_horizon_hours"])
        shell_name = str(item["shell_name"])
        plan.append(
            {
                **item,
                "prediction_dir": prediction_root / f"{horizon}h",
                "result_dir": output_root / "results" / f"{horizon}h" / shell_name,
            }
        )
    return plan


def find_horizon_prediction_files(
    prediction_dir: Path,
    *,
    label_horizon_hours: int,
    year: int,
) -> list[Path]:
    pattern = f"pred_{label_horizon_hours}h_{year}[0-1][0-9].pkl"
    return sorted(prediction_dir.glob(pattern), key=lambda path: path.name)


def evaluate_round1_predictions(
    *,
    predictions_root: Path,
    output_root: Path,
    provider_uri: str,
    year: int,
    instrument: str = "BTCUSDT",
    freq: str = "60min",
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.001,
    min_cost: float = 0.0,
    deal_price: str = "close",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shells = build_round1_trading_shells()

    for item in build_round1_run_plan(output_root):
        horizon = int(item["label_horizon_hours"])
        shell_name = str(item["shell_name"])
        prediction_dir = Path(item["prediction_dir"])
        prediction_files = find_horizon_prediction_files(
            predictions_root / f"{horizon}h",
            label_horizon_hours=horizon,
            year=year,
        )
        if not prediction_files:
            continue

        shell = shells[shell_name]
        config = SpotStrategyConfig(
            provider_uri=provider_uri,
            instrument=instrument,
            start_time=f"{year}-01-01 00:00:00",
            end_time=f"{year}-12-31 23:00:00",
            freq=freq,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            min_cost=min_cost,
            deal_price=deal_price,
            entry_threshold=shell["entry_threshold"],
            exit_threshold=shell["exit_threshold"],
            full_position_threshold=shell["full_position_threshold"],
            max_position=shell["max_position"],
            min_holding_hours=shell["min_holding_hours"],
            cooldown_hours=shell["cooldown_hours"],
            drawdown_de_risk_threshold=shell["drawdown_de_risk_threshold"],
            de_risk_position=shell["de_risk_position"],
        )
        summary = run_strategy_config(
            prediction_files,
            config,
            output_dir=Path(item["result_dir"]),
        )
        rows.append(
            {
                "label_horizon_hours": horizon,
                "shell_name": shell_name,
                **shell,
                **summary,
            }
        )

    return pd.DataFrame(rows)
