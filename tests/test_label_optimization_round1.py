from __future__ import annotations

import unittest

from nusri_project.training.lgbm_workflow import build_label_config, get_label_expr
from nusri_project.strategy.label_optimization_round1 import (
    build_round1_horizons,
    build_round1_trading_shells,
    build_round1_matrix,
)


class LabelOptimizationRound1Tests(unittest.TestCase):
    def test_get_label_expr_builds_expected_forward_return_expression(self) -> None:
        self.assertEqual(get_label_expr(24), "Ref($close, -24) / $close - 1")
        self.assertEqual(get_label_expr(48), "Ref($close, -48) / $close - 1")
        self.assertEqual(get_label_expr(72), "Ref($close, -72) / $close - 1")

    def test_build_label_config_uses_named_label_column(self) -> None:
        label_exprs, label_names = build_label_config(48)

        self.assertEqual(label_exprs, ["Ref($close, -48) / $close - 1"])
        self.assertEqual(label_names, ["label_48h"])

    def test_round1_horizons_are_fixed_to_three_regression_candidates(self) -> None:
        self.assertEqual(build_round1_horizons(), [24, 48, 72])

    def test_round1_trading_shells_define_balanced_and_conservative_profiles(self) -> None:
        shells = build_round1_trading_shells()

        self.assertEqual(set(shells.keys()), {"balanced", "conservative"})
        self.assertEqual(shells["balanced"]["max_position"], 0.25)
        self.assertEqual(shells["conservative"]["max_position"], 0.15)

    def test_round1_matrix_crosses_horizons_and_shells(self) -> None:
        matrix = build_round1_matrix()

        self.assertEqual(len(matrix), 6)
        self.assertEqual(
            matrix[0],
            {"label_horizon_hours": 24, "shell_name": "balanced"},
        )
        self.assertEqual(
            matrix[-1],
            {"label_horizon_hours": 72, "shell_name": "conservative"},
        )


if __name__ == "__main__":
    unittest.main()
