import argparse
from pathlib import Path

import pandas as pd

from nusri_project.strategy.label_optimization_round1 import (
    evaluate_round1_predictions,
    build_round1_horizons,
    build_round1_matrix,
    build_round1_run_plan,
    build_round1_trading_shells,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or inspect label optimization round 1.")
    parser.add_argument("--output-root", default="reports/label_optimization_round1")
    parser.add_argument("--predictions-root", default=None)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    print("label_horizons:", build_round1_horizons())
    print("shells:", sorted(build_round1_trading_shells().keys()))
    print("matrix_size:", len(build_round1_matrix()))
    for item in build_round1_run_plan(output_root):
        print(item)

    if args.predictions_root is not None and args.provider_uri is not None:
        summary = evaluate_round1_predictions(
            predictions_root=Path(args.predictions_root),
            output_root=output_root,
            provider_uri=args.provider_uri,
            year=args.year,
        )
        if not summary.empty:
            output_root.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_root / "summary.csv", index=False)
            print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
