from copy import deepcopy
from datetime import datetime
from typing import cast

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from alpha261_config import get_alpha261_config, get_top23_config

provider_uri = "./qlib_data/my_crypto_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

market = "all"
benchmark = "BTCUSDT"

DATA_START_TIME = "2019-09-10 08:00:00"
DATA_END_TIME = "2025-12-31 23:00:00"

# Rolling training config (monthly retrain)
ROLLING_START = "2024-01-01 00:00:00"
ROLLING_END = "2025-12-31 23:00:00"
ROLLING_TRAIN_YEARS = 2

# Feature set selection: "alpha261" or "top23"
FEATURE_SET = "top23"

# Run mode: "single" or "rolling"
RUN_MODE = "rolling"

if FEATURE_SET == "alpha261":
    feature_config = get_alpha261_config()
elif FEATURE_SET == "top23":
    feature_config = get_top23_config()
else:
    raise ValueError(f"Unknown FEATURE_SET: {FEATURE_SET}")

conf = {
    "task": {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.6,
                "subsample": 0.7,
                "learning_rate": 0.005,
                "lambda_l1": 1.5,
                "lambda_l2": 5.0,
                "max_depth": 5,
                "num_leaves": 31,
                "min_data_in_leaf": 100,
                "bagging_freq": 5,
                "num_threads": 8,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "start_time": DATA_START_TIME,
                        "end_time": DATA_END_TIME,
                        "instruments": market,
                        "data_loader": {
                            "class": "QlibDataLoader",
                            "module_path": "qlib.data.dataset.loader",
                            "kwargs": {
                                "config": {
                                    "feature": feature_config,
                                    "label": (
                                        ["Ref($close, -8) / $close - 1"],
                                        ["label"],
                                    ),
                                },
                                "freq": "60min",
                            },
                        },
                        "infer_processors": [
                            {
                                "class": "RobustZScoreNorm",
                                "kwargs": {
                                    "fields_group": "feature",
                                    "clip_outlier": True,
                                    "fit_start_time": DATA_START_TIME,
                                    "fit_end_time": "2023-12-31 19:00:00",
                                },
                            },
                            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                        ],
                    },
                },
                "segments": {
                    "train": ("2019-09-10 08:00:00", "2023-12-31 19:00:00"),
                    "valid": ("2024-01-01 00:00:00", "2024-12-31 19:00:00"),
                    "test": ("2025-01-01 00:00:00", "2025-12-31 23:00:00"),
                },
            },
        },
    },
}


def _build_dataset_kwargs(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    dataset_kwargs = deepcopy(conf["task"]["dataset"]["kwargs"])
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]

    handler_kwargs["start_time"] = train_start
    handler_kwargs["end_time"] = test_end

    for proc in handler_kwargs.get("infer_processors", []):
        if proc.get("class") == "RobustZScoreNorm":
            proc["kwargs"]["fit_start_time"] = train_start
            proc["kwargs"]["fit_end_time"] = train_end

    dataset_kwargs["segments"] = {
        "train": (train_start, train_end),
        "test": (test_start, test_end),
    }
    return dataset_kwargs


def _make_predictions(dataset, model, segment: str) -> pd.DataFrame:
    pred = model.predict(dataset, segment=segment).to_frame("pred")

    label = dataset.prepare(segment, col_set="label")
    label.columns = ["label"]

    combined = pred.join(label, how="inner").dropna()
    combined.columns = ["pred_return", "real_return"]
    return combined


def _print_summary(combined: pd.DataFrame, label_name: str, head_n: int = 10) -> None:
    print(f"\n=== {label_name} 预测结果 ===")

    if combined.empty:
        print(
            f"\n{label_name} 数据为空：请检查 segments 时间范围是否超出数据范围，或 label horizon 导致尾部被丢弃。"
        )
        return

    ic = combined.corr().iloc[0, 1]
    rank_ic = combined.corr(method="spearman").iloc[0, 1]
    same_sign = combined["pred_return"] * combined["real_return"] > 0
    accuracy = same_sign.sum() / len(same_sign)

    dt_index = combined.index.get_level_values(0)
    month = pd.Series(dt_index).dt.to_period("M")

    def _month_ic(df):
        if len(df) < 5:
            return pd.NA
        return df.corr().iloc[0, 1]

    monthly_ic = combined.groupby(month).apply(_month_ic).dropna()

    print("\n" + "=" * 30)
    print(f"{label_name} IC       : {ic:.4f}")
    print(f"{label_name} RankIC   : {rank_ic:.4f}")
    print(f"{label_name} 方向准确率 (Acc): {accuracy:.2%}")
    if len(monthly_ic) > 0:
        ic_mean = monthly_ic.mean()
        ic_std = monthly_ic.std()
        n_m = len(monthly_ic)
        icir = ic_mean / (ic_std + 1e-12)
        t_stat = ic_mean / ((ic_std + 1e-12) / (n_m**0.5))
        print(f"{label_name} 月度IC均值/标准差: {ic_mean:.4f} / {ic_std:.4f}")
        print(f"{label_name} 月度ICIR/t-stat(n={n_m}): {icir:.2f} / {t_stat:.2f}")
    print("=" * 30)

    print(f"\n前{head_n}条预测:")
    print(combined.head(head_n))


def run_single():
    with R.start(experiment_name="btc_raw_return_lgb"):
        print("正在训练模型...")

        model = init_instance_by_config(conf["task"]["model"])
        dataset = init_instance_by_config(conf["task"]["dataset"])
        model.fit(dataset)

        recorder = R.get_recorder()
        valid_pred = _make_predictions(dataset, model, "valid")
        _print_summary(valid_pred, "验证集", head_n=10)
        recorder.save_objects(**{"pred_valid.pkl": valid_pred})

        test_pred = _make_predictions(dataset, model, "test")
        _print_summary(test_pred, "测试集", head_n=10)
        recorder.save_objects(**{"pred_test.pkl": test_pred})


def run_rolling_monthly():
    start_ts = pd.Timestamp(ROLLING_START)
    end_ts = pd.Timestamp(ROLLING_END)
    data_start_ts = pd.Timestamp(DATA_START_TIME)

    yearly_results: dict[str, list[pd.DataFrame]] = {}

    for month_start in pd.date_range(start=start_ts, end=end_ts, freq="MS"):
        next_month_start = month_start + pd.DateOffset(months=1)
        month_end = next_month_start - pd.Timedelta(hours=1)
        if month_end > end_ts:
            month_end = end_ts

        train_end = month_start - pd.Timedelta(hours=1)
        train_start = month_start - pd.DateOffset(years=ROLLING_TRAIN_YEARS)
        if train_start < data_start_ts:
            train_start = data_start_ts

        month_start_dt = cast(datetime, month_start.to_pydatetime())
        month_end_dt = cast(datetime, month_end.to_pydatetime())
        train_start_dt = cast(datetime, train_start.to_pydatetime())
        train_end_dt = cast(datetime, train_end.to_pydatetime())

        dataset_kwargs = _build_dataset_kwargs(
            train_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            train_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            month_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            month_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

        dataset_conf = deepcopy(conf["task"]["dataset"])
        dataset_conf["kwargs"] = dataset_kwargs

        label_name = month_start_dt.strftime("%Y-%m")
        year_key = month_start_dt.strftime("%Y")
        with R.start(experiment_name="btc_raw_return_lgb_rolling"):
            model = init_instance_by_config(conf["task"]["model"])
            dataset = init_instance_by_config(dataset_conf)
            model.fit(dataset)

            test_pred = _make_predictions(dataset, model, "test")
            yearly_results.setdefault(year_key, []).append(test_pred)

    for year_key in sorted(yearly_results.keys()):
        yearly_pred = pd.concat(yearly_results[year_key]).sort_index()
        _print_summary(yearly_pred, f"{year_key}年", head_n=10)


if __name__ == "__main__":
    if RUN_MODE == "single":
        run_single()
    elif RUN_MODE == "rolling":
        run_rolling_monthly()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
