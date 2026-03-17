from __future__ import annotations

import unittest

from nusri_project.strategy.probability_signal_strategy import compute_target_weight_from_probability_signal


class ProbabilitySignalStrategyTests(unittest.TestCase):
    def test_probability_strategy_respects_holding_and_cooldown_guards(self) -> None:
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.40,
                current_weight=0.25,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=2,
                holding_bars=1,
                cooldown_bars=0,
                bars_since_trade=99,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.0,
            ),
            0.25,
        )
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.90,
                current_weight=0.0,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=3,
                bars_since_trade=1,
                drawdown=0.0,
                drawdown_de_risk_threshold=0.2,
                de_risk_position=0.0,
            ),
            0.0,
        )

    def test_probability_strategy_caps_target_in_drawdown(self) -> None:
        self.assertEqual(
            compute_target_weight_from_probability_signal(
                pred_prob=0.90,
                current_weight=0.0,
                max_position=0.25,
                enter_prob_threshold=0.65,
                exit_prob_threshold=0.50,
                full_prob_threshold=0.80,
                min_holding_bars=0,
                holding_bars=0,
                cooldown_bars=0,
                bars_since_trade=99,
                drawdown=0.05,
                drawdown_de_risk_threshold=0.02,
                de_risk_position=0.10,
            ),
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
