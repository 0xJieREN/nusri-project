from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpotStrategyConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.001
    entry_threshold: float = 0.0025
    exit_threshold: float = 0.0005
    full_position_threshold: float = 0.005
    min_holding_hours: int = 24
    cooldown_hours: int = 12
    max_position: float = 1.0
    drawdown_de_risk_threshold: float = 0.08
    de_risk_position: float = 0.5
    annualization_periods: int = 24 * 365

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if not 0 <= self.max_position <= 1:
            raise ValueError("max_position must be in [0, 1]")
        if self.full_position_threshold < self.entry_threshold:
            raise ValueError("full_position_threshold must be >= entry_threshold")
        if self.entry_threshold < self.exit_threshold:
            raise ValueError("entry_threshold must be >= exit_threshold")
        if self.min_holding_hours < 0:
            raise ValueError("min_holding_hours must be non-negative")
        if self.cooldown_hours < 0:
            raise ValueError("cooldown_hours must be non-negative")
        if not 0 <= self.drawdown_de_risk_threshold <= 1:
            raise ValueError("drawdown_de_risk_threshold must be in [0, 1]")
        if not 0 <= self.de_risk_position <= self.max_position:
            raise ValueError("de_risk_position must be in [0, max_position]")
