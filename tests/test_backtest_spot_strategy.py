from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from backtest_spot_strategy import (
    _expand_prediction_globs,
    normalize_prediction_frame,
    run_spot_backtest,
    summarize_backtest,
)
from strategy_config import SpotStrategyConfig


class SpotBacktestTests(unittest.TestCase):
    def test_normalize_prediction_frame_flattens_multiindex_and_sorts(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01 02:00:00"), "BTCUSDT"),
                (pd.Timestamp("2024-01-01 01:00:00"), "BTCUSDT"),
            ],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame(
            {
                "pred_return": [0.02, 0.01],
                "real_return": [0.03, -0.01],
            },
            index=index,
        )

        normalized = normalize_prediction_frame(frame)

        self.assertEqual(list(normalized.index), [pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-01-01 02:00:00")])
        self.assertEqual(list(normalized.columns), ["pred_return", "real_return"])

    def test_backtest_charges_fees_on_entry_and_exit(self) -> None:
        frame = pd.DataFrame(
            {
                "pred_return": [0.03, -0.01],
                "real_return": [0.01, 0.00],
            },
            index=pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
        )
        config = SpotStrategyConfig(
            initial_cash=100_000.0,
            fee_rate=0.001,
            entry_threshold=0.01,
            exit_threshold=0.0,
            full_position_threshold=0.02,
            min_holding_hours=0,
            cooldown_hours=0,
            max_position=1.0,
            drawdown_de_risk_threshold=1.0,
            de_risk_position=0.0,
        )

        result = run_spot_backtest(frame, config)

        self.assertAlmostEqual(result.iloc[0]["position"], 1.0)
        self.assertAlmostEqual(result.iloc[0]["fee_paid"], 100.0, places=6)
        self.assertAlmostEqual(result.iloc[0]["equity"], 100_899.0, places=6)
        self.assertAlmostEqual(result.iloc[1]["position"], 0.0)
        self.assertAlmostEqual(result.iloc[1]["fee_paid"], 100.899, places=6)
        self.assertAlmostEqual(result.iloc[1]["equity"], 100_798.101, places=6)

    def test_backtest_respects_min_holding_and_cooldown(self) -> None:
        frame = pd.DataFrame(
            {
                "pred_return": [0.03, -0.02, -0.02, 0.03],
                "real_return": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                    "2024-01-01 03:00:00",
                ]
            ),
        )
        config = SpotStrategyConfig(
            initial_cash=100_000.0,
            fee_rate=0.0,
            entry_threshold=0.01,
            exit_threshold=0.0,
            full_position_threshold=0.02,
            min_holding_hours=2,
            cooldown_hours=2,
            max_position=1.0,
            drawdown_de_risk_threshold=1.0,
            de_risk_position=0.0,
        )

        result = run_spot_backtest(frame, config)

        self.assertEqual(list(result["position"]), [1.0, 1.0, 0.0, 0.0])

    def test_summarize_backtest_reports_core_metrics(self) -> None:
        result = pd.DataFrame(
            {
                "equity": [100_000.0, 101_000.0, 100_500.0],
                "position": [0.0, 1.0, 0.5],
                "turnover": [0.0, 1.0, 0.5],
                "fee_paid": [0.0, 100.0, 50.0],
                "strategy_return": [0.0, 0.01, -0.0049504950495],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]
            ),
        )

        summary = summarize_backtest(result)

        self.assertIn("annualized_return", summary)
        self.assertIn("annualized_volatility", summary)
        self.assertIn("sharpe", summary)
        self.assertIn("calmar", summary)
        self.assertIn("max_drawdown", summary)
        self.assertIn("turnover", summary)
        self.assertIn("avg_holding_hours", summary)
        self.assertIn("exposure_ratio", summary)
        self.assertAlmostEqual(summary["turnover"], 1.5, places=6)

    def test_expand_prediction_globs_accepts_absolute_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "pred_202401.pkl"
            pd.DataFrame(
                {
                    "pred_return": [0.01],
                    "real_return": [0.02],
                },
                index=pd.to_datetime(["2024-01-01 00:00:00"]),
            ).to_pickle(pred_path)

            resolved = _expand_prediction_globs([str(pred_path)])

        self.assertEqual(resolved, [pred_path.resolve()])


if __name__ == "__main__":
    unittest.main()
