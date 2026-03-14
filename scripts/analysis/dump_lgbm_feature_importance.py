import argparse
from pathlib import Path

import pandas as pd

# Reuse the exact config and qlib init from the training script.
from nusri_project.training.lgbm_workflow import build_conf, init_qlib  # noqa: E402
from qlib.utils import init_instance_by_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LGBM and dump feature importance.")
    parser.add_argument(
        "--importance-type",
        choices=("gain", "split"),
        default="gain",
        help="LightGBM feature importance type.",
    )
    parser.add_argument(
        "--out",
        default="reports/feature_importance/lgbm_feature_importance.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("alpha261", "top23"),
        default="top23",
        help="Feature set used by the training workflow.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Print top-N features to stdout.",
    )
    args = parser.parse_args()

    init_qlib()
    conf = build_conf(feature_set=args.feature_set)
    model = init_instance_by_config(conf["task"]["model"])
    dataset = init_instance_by_config(conf["task"]["dataset"])
    model.fit(dataset)

    if model.model is None:
        raise RuntimeError("LightGBM model is not initialized after training.")

    handler = dataset.handler
    feature_names = handler.get_cols("feature")
    importance = model.model.feature_importance(importance_type=args.importance_type)

    if len(feature_names) != len(importance):
        raise ValueError(
            f"Feature count mismatch: names={len(feature_names)} importance={len(importance)}"
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            f"importance_{args.importance_type}": importance,
        }
    ).sort_values(by=f"importance_{args.importance_type}", ascending=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
