from __future__ import annotations


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
