import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from alpha261_config import get_alpha261_config

provider_uri = './qlib_data/my_crypto_data'
qlib.init(provider_uri=provider_uri, region=REG_CN)

market = "all"
benchmark = "BTCUSDT"

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
                        "start_time": "2019-09-10 08:00:00",
                        "end_time": "2025-12-31 23:00:00",
                        "instruments": market,
                        "data_loader": {
                            "class": "QlibDataLoader",
                            "module_path": "qlib.data.dataset.loader",
                            "kwargs": {
                                "config": {
                                    "feature": get_alpha261_config(),
                                    "label": (["Ref($close, -4) / $close - 1"], ["label"]),
                                },
                                "freq": "60min",
                            },
                        },
                        "infer_processors": [
                            {
                                'class': 'RobustZScoreNorm',
                                'kwargs': {
                                    'fields_group': 'feature',
                                    'clip_outlier': True,
                                    'fit_start_time': '2019-09-10 08:00:00',
                                    'fit_end_time': '2023-12-31 19:00:00',
                                }
                            },
                            {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}
                        ],
                        "learn_processors": [
                            {'class': 'DropnaLabel'},
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

if __name__ == "__main__":
    with R.start(experiment_name="btc_raw_return_lgb"):
        print("正在训练模型...")
        
        model = init_instance_by_config(conf["task"]["model"])
        dataset = init_instance_by_config(conf["task"]["dataset"])
        model.fit(dataset)

        def evaluate_segment(segment: str, label_name: str):
            print(f"\n正在生成预测 ({label_name})...")
            pred = model.predict(dataset, segment=segment).to_frame("pred")

            label = dataset.prepare(segment, col_set="label")
            label.columns = ["label"]

            combined = pred.join(label, how="inner").dropna()
            combined.columns = ["pred_return", "real_return"]

            print("\n=== 预测值 vs 真实值 (前30行) ===")
            print(combined.head(30))

            if combined.empty:
                print(f"\n{label_name} 数据为空：请检查 segments 时间范围是否超出数据范围，或 label horizon 导致尾部被丢弃。")
                return combined

            ic = combined.corr().iloc[0, 1]
            rank_ic = combined.corr(method="spearman").iloc[0, 1]
            same_sign = (combined["pred_return"] * combined["real_return"] > 0)
            accuracy = same_sign.sum() / len(same_sign)

            dt_index = combined.index.get_level_values(0)
            month = pd.Series(dt_index).dt.to_period("M")

            def _month_ic(df):
                if len(df) < 5:
                    return pd.NA
                return df.corr().iloc[0, 1]

            monthly_ic = (
                combined.groupby(month).apply(_month_ic).dropna()
            )

            print("\n" + "=" * 30)
            print(f"{label_name} IC       : {ic:.4f}")
            print(f"{label_name} RankIC   : {rank_ic:.4f}")
            print(f"{label_name} 方向准确率 (Acc): {accuracy:.2%}")
            if len(monthly_ic) > 0:
                ic_mean = monthly_ic.mean()
                ic_std = monthly_ic.std()
                n_m = len(monthly_ic)
                icir = ic_mean / (ic_std + 1e-12)
                t_stat = ic_mean / ((ic_std + 1e-12) / (n_m ** 0.5))
                print(f"{label_name} 月度IC均值/标准差: {ic_mean:.4f} / {ic_std:.4f}")
                print(f"{label_name} 月度ICIR/t-stat(n={n_m}): {icir:.2f} / {t_stat:.2f}")
            print("=" * 30)

            print("\n预测值统计分布:")
            print(combined["pred_return"].describe())

            print("\n真实收益(label)统计分布:")
            print(combined["real_return"].describe())

            return combined

        recorder = R.get_recorder()
        valid_pred = evaluate_segment("valid", "验证集")
        recorder.save_objects(**{"pred_valid.pkl": valid_pred})

        test_pred = evaluate_segment("test", "测试集")
        recorder.save_objects(**{"pred_test.pkl": test_pred})
