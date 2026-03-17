from __future__ import annotations

from typing import Any

from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

from nusri_project.strategy.qlib_spot_strategy import QlibSingleAssetOrderGen, hours_to_bars
from nusri_project.strategy.return_signal_strategy import _apply_trade_guards


def compute_target_weight_from_probability_signal(
    *,
    pred_prob: float,
    current_weight: float,
    max_position: float,
    enter_prob_threshold: float,
    exit_prob_threshold: float,
    full_prob_threshold: float,
    min_holding_bars: int,
    holding_bars: int,
    cooldown_bars: int,
    bars_since_trade: float,
    drawdown: float,
    drawdown_de_risk_threshold: float,
    de_risk_position: float,
) -> float:
    if pred_prob >= full_prob_threshold:
        target_weight = max_position
    elif pred_prob >= enter_prob_threshold:
        target_weight = max_position * 0.5
    elif pred_prob <= exit_prob_threshold:
        target_weight = 0.0
    else:
        target_weight = current_weight

    if drawdown >= drawdown_de_risk_threshold:
        target_weight = min(target_weight, de_risk_position)

    return _apply_trade_guards(
        target_weight=target_weight,
        current_weight=current_weight,
        min_holding_bars=min_holding_bars,
        holding_bars=holding_bars,
        cooldown_bars=cooldown_bars,
        bars_since_trade=bars_since_trade,
    )


class QlibProbabilityLongFlatStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        instrument: str,
        time_per_step: str = "60min",
        enter_prob_threshold: float = 0.65,
        exit_prob_threshold: float = 0.50,
        full_prob_threshold: float = 0.80,
        min_holding_hours: int = 24,
        cooldown_hours: int = 12,
        max_position: float = 1.0,
        drawdown_de_risk_threshold: float = 0.08,
        de_risk_position: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(order_generator_cls_or_obj=QlibSingleAssetOrderGen(), **kwargs)
        self.instrument = instrument
        self.time_per_step = time_per_step
        self.enter_prob_threshold = enter_prob_threshold
        self.exit_prob_threshold = exit_prob_threshold
        self.full_prob_threshold = full_prob_threshold
        self.min_holding_bars = hours_to_bars(min_holding_hours, time_per_step)
        self.cooldown_bars = hours_to_bars(cooldown_hours, time_per_step)
        self.max_position = max_position
        self.drawdown_de_risk_threshold = drawdown_de_risk_threshold
        self.de_risk_position = de_risk_position
        self.high_watermark: float | None = None
        self.last_trade_step: int | None = None

    def _extract_pred_prob(self, score) -> float | None:
        if score is None:
            return None
        if hasattr(score, "columns"):
            if self.instrument not in score.index:
                return None
            row = score.loc[self.instrument]
            if hasattr(row, "iloc"):
                return float(row.iloc[0])
            return float(row)
        if self.instrument not in score.index:
            return None
        return float(score.loc[self.instrument])

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        pred_prob = self._extract_pred_prob(score)
        if pred_prob is None:
            return current.get_stock_weight_dict()

        equity = float(current.calculate_value())
        if self.high_watermark is None:
            self.high_watermark = equity
        else:
            self.high_watermark = max(self.high_watermark, equity)

        drawdown = 0.0 if self.high_watermark <= 0 else 1.0 - (equity / self.high_watermark)
        current_weight = current.get_stock_weight_dict().get(self.instrument, 0.0)
        holding_bars = 0
        if self.instrument in current.get_stock_list():
            holding_bars = int(current.get_stock_count(self.instrument, bar=self.trade_calendar.get_freq()))

        trade_step = self.trade_calendar.get_trade_step()
        bars_since_trade = float("inf") if self.last_trade_step is None else trade_step - self.last_trade_step

        target_weight = compute_target_weight_from_probability_signal(
            pred_prob=pred_prob,
            current_weight=current_weight,
            max_position=self.max_position,
            enter_prob_threshold=self.enter_prob_threshold,
            exit_prob_threshold=self.exit_prob_threshold,
            full_prob_threshold=self.full_prob_threshold,
            min_holding_bars=self.min_holding_bars,
            holding_bars=holding_bars,
            cooldown_bars=self.cooldown_bars,
            bars_since_trade=bars_since_trade,
            drawdown=drawdown,
            drawdown_de_risk_threshold=self.drawdown_de_risk_threshold,
            de_risk_position=self.de_risk_position,
        )

        if abs(target_weight - current_weight) > 1e-12:
            self.last_trade_step = trade_step

        if target_weight <= 0:
            return {}
        return {self.instrument: target_weight}
