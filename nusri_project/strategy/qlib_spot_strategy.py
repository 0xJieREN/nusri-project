from __future__ import annotations

import math

from qlib.contrib.strategy.order_generator import OrderGenerator
from qlib.utils.time import Freq


def hours_to_bars(hours: int, time_per_step: str) -> int:
    if hours <= 0:
        return 0

    count, base = Freq.parse(time_per_step)
    hours_per_bar = {
        Freq.NORM_FREQ_MINUTE: count / 60.0,
        Freq.NORM_FREQ_DAY: count * 24.0,
        Freq.NORM_FREQ_WEEK: count * 24.0 * 7.0,
        Freq.NORM_FREQ_MONTH: count * 24.0 * 30.0,
    }[base]
    return int(math.ceil(hours / hours_per_bar))

class QlibSingleAssetOrderGen(OrderGenerator):
    def generate_order_list_from_target_weight_position(
        self,
        current,
        trade_exchange,
        target_weight_position: dict,
        risk_degree: float,
        pred_start_time,
        pred_end_time,
        trade_start_time,
        trade_end_time,
    ):
        if target_weight_position is None:
            return []

        current_amount_dict = current.get_stock_amount_dict()
        current_total_value = current.calculate_value()
        investable_cash = risk_degree * current_total_value
        investable_cash /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)

        target_amount_dict = {}
        for stock_id, target_weight in target_weight_position.items():
            if target_weight <= 0:
                continue
            if not trade_exchange.is_stock_tradable(stock_id, start_time=trade_start_time, end_time=trade_end_time):
                continue
            deal_price = trade_exchange.get_deal_price(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=1,
            )
            target_amount_dict[stock_id] = investable_cash * target_weight / deal_price

        return trade_exchange.generate_order_for_target_amount_position(
            target_position=target_amount_dict,
            current_position=current_amount_dict,
            start_time=trade_start_time,
            end_time=trade_end_time,
        )
