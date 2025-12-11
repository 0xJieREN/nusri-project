import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R

# 1. 初始化
provider_uri = './qlib_data/my_crypto_data'
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 2. 配置
market = "all"
benchmark = "BTCUSDT"

conf = {
    "task": {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8, 
                "learning_rate": 0.005,
                "subsample": 0.7,
                "lambda_l1": 0.05,    
                "lambda_l2": 0.1,    
                "max_depth": 6,      
                "num_leaves": 31,
                "min_data_in_leaf": 5,  
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {
                        "start_time": "2018-01-01",
                        "end_time": "2024-12-01",
                        "fit_start_time": "2018-01-01",
                        "fit_end_time": "2022-12-31",
                        "instruments": market,
                        "infer_processors": [
                            {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True, 'fit_start_time': '2018-01-01', 'fit_end_time': '2022-12-31'}},
                            {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}
                        ],
                        "learn_processors": [
                            {'class': 'DropnaLabel'},
                        ],
                        "label": ["Ref($close, -1) / $close - 1"] 
                    },
                },
                "segments": {
                    "train": ("2018-01-01", "2022-12-31"),
                    "valid": ("2023-01-01", "2023-12-31"),
                    "test":  ("2024-01-01", "2024-12-01"),
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

        print("正在生成预测...")
        pred = model.predict(dataset)
        
        # 将 Series 转换为 DataFrame ---
        # 给这一列起个名字叫 'score'
        pred = pred.to_frame("score") 

        
        recorder = R.get_recorder()
        recorder.save_objects(**{"pred.pkl": pred})
        
        # --- 计算 IC ---
        label = dataset.prepare("test", col_set="label")
        label.columns = ['label'] # 确保 label 也是 DataFrame 且有列名
        
        combined = pred.join(label, how='inner') # 使用 inner join 确保索引对齐
        combined.columns = ['pred_return', 'real_return']
        
        print("\n=== 预测值 vs 真实值 (前30行) ===")
        print(combined.head(30))

        # 4. 计算指标
        ic = combined.corr().iloc[0, 1]
        same_sign = (combined['pred_return'] * combined['real_return'] > 0)
        accuracy = same_sign.sum() / len(same_sign)

        print("\n" + "="*30)
        print(f"测试集 IC       : {ic:.4f}")
        print(f"方向准确率 (Acc): {accuracy:.2%}")
        print("="*30)
        
        # 5. 看看模型是不是还在偷懒（检查预测值的分布）
        print("\n预测值统计分布:")
        print(combined['pred_return'].describe())